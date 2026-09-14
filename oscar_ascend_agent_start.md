# Agent 启动指令：将 OSCAR KV Cache INT2 量化适配到 vLLM Ascend 0.23.0

> 目标模型：`Qwen3.5-27B-w8a8-mtp`。目标环境：Ascend NPU、4 卡、TP=4。实现方式：外部适配插件 + 独立 AscendC 算子包。
>
> 使用方式：将本文交给执行 Agent，按“核实参考实现 → 完善 Checklist → 设计 → 实现 → 真机验证 → 一键交付”的顺序持续推进。不要停留在方案、伪代码或能启动的演示版本。
>
> 信息说明：本文保留用户提供的目标、配置、布局线索、性能日志和历史错误。重构本文时未验证参考代码或真机；标为“待核实”的内容必须在实施前取得代码或运行证据。附录中的日志是历史材料，不代表本次执行结果。

## 1. 任务目标与完成定义

将指定 **OSCAR vLLM PR** 中的 KV Cache 量化算法适配到 vLLM Ascend 0.23.0 对应的软件栈，重构相关底层算子，使用 **AscendC** 实现并优化，使其能够运行在 **GDN 线性注意力与 FULL 全注意力混合模型**上。

任务必须同时满足以下条件：

1. **算法范围正确**：OSCAR 仅作用于 FULL 层的 K/V；GDN 层的 conv、SSM 状态及其原生计算、更新语义保持不变。
2. **真实节省 KV 显存**：History 使用 INT2 持久化存储，Sink/Recent 保持 BF16。节省必须反映到物理分配或可用 KV 容量，不能只改变数据格式而保留原来的冗余分配。
3. **功能完整**：支持首次 prefill、chunked prefill、普通 decode、原生 MTP/speculative decoding、混合长度 batch、长上下文和目标启动配置。
4. **性能不慢于原生**：在相同硬件、模型、软件、输入和启动条件下，适配后的端到端性能不得慢于原生 vLLM Ascend。必须实测，不能从 INT2 压缩比推导性能结论。
5. **外部集成**：不修改真机上的 vLLM Ascend 原始源码；通过外部插件、注册、Hook、子类化或受控运行时 monkey-patch 集成。
6. **可直接部署**：交付真实可运行代码，以及一个完成“安装插件 → 编译 AscendC → probe → 清理本任务 NPU 资源 → 正式启动服务”的 Bash 入口。
7. **可审计**：每个完成的 Checkpoint 都有代码位置、相关代码片段、解释和适用的测试或性能证据。没有执行的验证必须明确标记为未运行。

**不得将以下状态报告为完成**：仅完成设计；仅 Python reference 正确；仅编译成功；仅 HTTP 返回 200；仅短序列可运行；未进入 OSCAR 路径；MTP 被关闭；长序列回退；显存收益未落地；性能或精度未验收。

## 2. 执行边界与要求优先级

### 2.1 硬约束

| 编号 | 必须遵守的要求 |
| --- | --- |
| H01 | 不修改、覆盖或替换真机 vLLM Ascend 原始源码文件；适配代码放在独立工程。不要通过修改原生 vLLM 文件绕过此限制。 |
| H02 | 保留用户原限制：**“不准任何 cp”**。交付脚本不得使用 `cp` 命令，也不得用其他复制、写入或替换手段覆盖原生代码以实现适配。插件和算子通过独立工程的安装、构建和加载机制集成。 |
| H03 | 不参考本地任何失败的相关代码仓。历史堆栈中的失败项目路径仅用于理解错误，不构成访问、复用其代码的许可。 |
| H04 | 新增的量化、反量化、旋转、逆旋转、裁剪、移位、打包、解包及相关 KV 数据搬运均在 NPU 上执行，核心数据路径用 AscendC 实现。 |
| H05 | 不通过 `.cpu()`、`.to("cpu")`、NumPy 或逐元素 Python 运算处理在线 KV 数据；热路径不得增加逐请求 Python 循环、`.item()` 等同步开销。 |
| H06 | 仅替换 FULL 层所需的缓存布局与读写/Attention 路径；保留 GDN 状态和原生页号、层号、组共享页的逻辑对应关系。 |
| H07 | 必须重写 Decode Stage1 为真正的 **fused INT2 CV kernel**；CV 的含义与计算分工以指定 PR 和目标硬件实现为准，不能只改函数名。 |
| H08 | 不保留冗余的完整 BF16 History、重复 Sink/Recent staging 或重复历史 MTP shadow cache；不得用 KV 双写抵消压缩收益。 |
| H09 | Decode/verify 不得每步执行全历史 `full_dequant + inverse rotation`，不得把完整历史恢复到 HBM 后再调用普通 Attention。 |
| H10 | chunked prefill 不重复搬运或压缩已完成处理的历史；首次 prefill 尽量减少不必要的压缩及压缩后立即恢复。 |
| H11 | MTP 必须遵循原生 vLLM Ascend 的计算、提交和回滚语义；不能误把 verify 当成普通 prefill，不能增加重复计算。 |
| H12 | MTP 对同一历史 INT2 数据应跨 query 复用，历史 HBM 读取量不得随 `q_len` 线性倍增；必须以 kernel 设计和测量证明。 |
| H13 | 16K–50K 输入长度必须实际进入并使用 OSCAR，不得静默回退；必须兼顾混合长度 batch。 |
| H14 | 必须保持附录 A 的目标启动参数，包括 TP=4、MTP、异步调度、图模式、W8A8 权重量化和 GDN cache dtype；不能通过关闭功能获得验收通过。 |
| H15 | 每次 probe、校准或临时服务结束后，先确认本任务创建的 worker 和 NPU 资源已释放，再启动正式服务。 |

“纯 NPU”约束针对计算和 KV 数据路径。必要的配置读取、形状推导、op_host tiling、批量调度元数据、安装和日志工作可以在 Host 执行，但必须避免由此引入在线 KV 数据回传和频繁设备同步。调试时的小规模结果核对与在线实现分开，不能作为生产 fallback。

### 2.2 遇到冲突时如何处理

遵循以下顺序：**硬约束和任务目标 → 已核实的指定 PR/目标环境行为 → 本文中的初步设计建议**。

- 用户给出的布局数字和历史记录必须保留，实施时核实其适用条件，不能未经验证就硬编码。
- “保持三大条框架”“保持每页大小不变”等属于初步设计方向。如果只是把压缩后空出的空间变成 padding，不能满足真实节省显存的目标，该方案不得作为最终实现。
- 若硬约束之间在目标环境中确有冲突，给出代码级证据、已尝试路径和受影响范围；继续不依赖该冲突的工作，不能偷偷修改源码、改变算法或降低验收要求。
- 已明确授权按设计实施，不要在设计完成后重复询问是否开始。只有缺少无法推导的关键输入、必须改变硬约束或操作超出授权范围时才提出具体问题。

## 3. 目标环境、输入与参考资料

### 3.1 已提供信息

| 项目 | 用户提供的值或线索 |
| --- | --- |
| 模型 | `Qwen3.5-27B-w8a8-mtp`，GDN + FULL 混合架构 |
| 模型路径 | `/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp` |
| 目标适配版本 | vLLM Ascend 0.23.0 |
| 部署规模 | 4 张 Ascend NPU，TP=4，DP=1 |
| vLLM 历史版本字符串 | `0.23.0+empty` |
| vLLM Ascend 历史版本字符串 | `0.23.1.dev0+g5cb98caaa.d20260822` |
| torch / torch_npu | `2.10.0+cpu` / `2.10.0.post4` |
| Triton | `3.2.0`；仅为环境信息，不改变 AscendC 实现要求 |
| 历史源码位置 | `/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend`；先只读核实 |
| 默认 OSCAR 窗口 | Sink 先取 `S=64`，Recent 先取 `R=256`，均须可配置并与 PR 对齐 |
| 正式服务配置 | 完整命令见附录 A，重构时原样保留 |
| 原生性能与错误证据 | 完整历史日志见附录 B、C |

这些版本和路径来自用户提供的历史环境。执行时记录实际版本、commit、加载路径和接口能力；不能只因开发版字符串不等于 `0.23.0` 就拒绝，也不能完全跳过兼容性检查。`torch` 含 `+cpu` 后缀本身不能替代 torch_npu/NPU 能力探测。

### 3.2 启动阶段必须补齐的输入

先在当前任务明确提供的资料、指定 `reference` 目录和正式代码仓范围内查找，不广泛扫描旧项目或失败仓库。建立 `docs/reference_manifest.md`，至少记录：

- OSCAR vLLM PR 的准确链接/编号、目标 commit、reference 路径、关联实现和测试文件。
- 目标 vLLM、vLLM Ascend 的实际 commit、安装路径、Python/torch/torch_npu/CANN/驱动/固件/编译器版本。
- NPU 型号、单卡可用 HBM、可用卡号、TP 拓扑和目标环境访问方式。
- 模型配置中的层类型、层数、KV head 数、head dimension、TP 后各 rank 的本地维度、GDN 状态形状和 dtype。
- 当前生效的 block size、KV cache spec、分组规则、Attention 后端、MTP 路由和图捕获机制。
- PR 的旋转矩阵或检查点来源、量化参数、校准流程、已有精度测试和配置方式。
- 公平对照需要的输入数据、输出长度、并发、采样参数、重复次数和质量验收指标。

**本指令未给出 OSCAR PR 的确切链接、reference 绝对路径或真机连接方式。** 若提供的执行环境中也没有，明确请求缺失项，并继续不依赖它们的整理工作；不得凭相似名称选取另一套 OSCAR 实现，也不得伪称已读过 reference。

### 3.3 必须读懂的调用链

阅读与任务有关的完整调用链，包括接口、调用者、元数据结构和测试，形成带文件/函数位置的说明：

1. OSCAR：配置 → Sink/History/Recent 状态管理 → 旋转/裁剪/量化 → cache store → prefill/decode → 输出合并。
2. vLLM KV 管理：`kv_cache_utils.py` → cache spec/层分组 → block 分配与生命周期 → block table/slot mapping。
3. vLLM Ascend：`model_runner_v1.py` → 物理分配与视图 → FULL/GDN 后端 → worker 初始化和算子调用。
4. Qwen3.5：混合层结构 → FULL KV 路径 → GDN conv/SSM 状态 → TP 分片。
5. MTP：draft → target verify → acceptance/rejection → commit/rollback → 下一轮调度及状态可见性。
6. 执行模式：首次 prefill、chunked prefill、decode、混合 batch、异步调度、图捕获/回放、前缀缓存和页回收。

读完 reference 后，先完善第 9 节的 Checklist，再完成设计并直接实施。

## 4. 已有布局线索与必须核实的歧义

### 4.1 用户提供的“三大条”布局

原稿描述了统一页式缓存中的三条连续区域：

| 区域 | 原稿容量 | 用途 |
| --- | --- | --- |
| A | `nb × 15,360` 字节 | GDN 的 conv |
| B | `nb × 393,216` 字节 | FULL 的 K 或 GDN 的 SSM |
| C | `nb × 393,216` 字节 | FULL 的 V |
| 合计 | `nb × P`，`P = 801,792` 字节 | `15,360 + 393,216 + 393,216` |

原稿同时提供：

- 层分组表述为“4 组 × 16 层，每组包含 `[GDN, GDN, GDN, FULL]` 交替，共 64 层”。
- “每组共享一个 `KVCacheTensor`，共 16 个 Tensor”，每个大小为 `P × nb`。
- 在 `model_runner_v1.py` 中实际分配 16 个 `torch.zeros(P × nb, dtype=int8)`，每个形状为 `(P×nb,)`。
- FULL 使用 B/C 存 K/V，GDN 使用 A/B 存 conv/SSM，没有独立的 FULL 式 K/V。

原始偏移线索，统一以 `b` 表示页号、`t` 表示原稿中的索引：

```text
conv  = A_start + b × 15,360 + t × conv_element_bytes
K/SSM = nb × 15,360 + b × 393,216 + t × 512
V     = nb × 408,576 + b × 393,216 + t × 512
```

这些是核查入口，不是已经确认的地址公式。尤其不能直接把 FULL 的 token 寻址公式套到 GDN 的状态维度上。

### 4.2 在写布局代码前逐项消除歧义

| 疑点 | 实施前必须查清的内容 |
| --- | --- |
| 4 组、16 层、16 个 Tensor 的关系 | 列出真实的层号 → 层类型 → cache group → tensor → 物理区间映射，解释分组与共享关系。 |
| “共享 B 条”的含义 | 区分共享布局模板、共享大分配、不同偏移视图与同时存活的真实地址别名；没有证据不得断言 FULL/GDN 同时写相同字节。 |
| `P` 与实际页布局 | 核实它是每个 Tensor 的页字节预算、组合状态大小还是其他单位，以及是否受 GDN 最大状态尺寸约束。 |
| 128、256、768、512 字节 | 原稿含 head_dim 可能为 128 或 256 的推测，以及 `768 × 2 = 1536` 的疑问。核实 hidden size、head_dim、KV heads、本地 KV 维度和 TP 分片，不能与 512 字节/token 混用。 |
| FP16 与 BF16 | 两者均按 2 字节/元素计容量，但运算和数据解释不同。原稿两种写法都出现；最终以配置和 reference 明确 dtype。 |
| INT2 位宽 | 原稿“INT2（4 bits per element）”与其 `0.25 字节/元素` 计算冲突。按 INT2 应为 **2 bits/元素、每字节 4 个元素**；4 bits/元素属于 INT4。编码、符号、位序和量化映射仍须核实 PR。 |
| History 的 160 字节/token | 原稿 K/V 各 256 元素时，裸 INT2 K+V 为 128 字节；160 字节是含 pad 的参考值。必须列出多出的 32 字节及实际元数据/对齐来源，不可硬编码。 |
| Sink/Recent 的作用域 | 原稿按页描述前 S/后 R；需对照 PR 确认请求全局窗口语义与物理跨页映射，不能在每个物理页重复分配整套 Sink/Recent。 |
| 固定页大小后的“节省” | 若空出的空间仍被原分配占用，必须证明它增加了可用缓存容量或确实减少分配；仅保留为 padding 不算节省。 |

## 5. 算法与缓存设计要求

### 5.1 保留 OSCAR 语义，分别设计存储与计算

按用户提供的算法说明：

```text
Sink    : 固定的初始 tokens，BF16
History : 历史 tokens，旋转 + 裁剪后量化为 INT2
Recent  : 最近 tokens，BF16

逻辑写路径：KV → Rotate & Clip → Quantize/Pack INT2 → Paged Cache
逻辑读路径：Paged Cache → Dequantize → Inverse Rotate → Attention
```

以上读路径描述数学语义，不意味着每个 decode/verify step 都要在 HBM 中生成完整 BF16 历史。必须研究 PR 中旋转、Query 变换、Attention 累加和输出逆变换的具体关系，证明融合实现与参考算法一致，再确定哪些变换可以合并或移动。不能未经证明就把 K/V 的逆旋转随意换到 Query 或输出侧。

`--quantization ascend` 对应的现有模型权重量化配置必须保留；OSCAR KV 量化配置按 PR 的方式通过外部适配接入，避免覆盖或混淆 W8A8 权重量化。

### 5.2 先定义逻辑窗口，再映射物理页

以参考实现确认的请求级窗口语义为基础，定义序列长度 `L`、配置 `S=64`、`R=256`。对于不重叠窗口，可用以下边界核对短序列：

```text
s = min(S, L)
r = min(R, L - s)
h = L - s - r

Sink    : [0, s)
History : [s, L - r)
Recent  : [L - r, L)
```

最终必须对照 PR 确认边界语义，覆盖 `L < S`、`L < S+R`、空 History 和跨页情况；不能直接使用可能为负的 `H = T-S-R` 分配内存。

设计必须回答：

- token 如何由逻辑位置映射到 block table、物理页和各段偏移？页号与层号的原生关系如何保留？
- Recent 滑出后何时压缩为 History？如何做到正常追加时仅处理新增/迁移 token？
- Sink/Recent 如何只保留必要的 BF16 数据，并与 History 分配一起参与容量核算？
- MTP 的已提交与暂存 token 如何区分？接受、部分接受和拒绝后，长度、页表、窗口、量化参数及状态如何一致更新？
- GDN 与 FULL 的共享分配或视图边界如何保证不重叠写坏？若原分组阻止回收压缩空间，如何在外部适配范围内解决？
- 前缀共享、页复制/写时复制、请求完成/取消、页复用、抢占与恢复时，窗口身份和元数据如何保持正确？

### 5.3 必须给出的字节公式

不要把“256 维”当成全局常量。按 **每层、每 rank、每 token** 定义：

```text
E_K = n_kv_heads_local × d_k
E_V = n_kv_heads_local × d_v

BF16_KV_payload = 2 × (E_K + E_V)                           # 字节/token
INT2_K_payload  = ceil(2 × E_K / 8)                         # 字节/token
INT2_V_payload  = ceil(2 × E_V / 8)                         # 字节/token
INT2_KV_payload = INT2_K_payload + INT2_V_payload

align_up(x, a) = ceil(x / a) × a
```

上式仅是 payload 下界；若逐 head/group 单独打包，应按真实打包单元分别取整后求和。继续展开 scale、zero point（若 PR 使用）、旋转参数、分组描述、长度、页表及 padding 的布局和对齐。明确元数据按 token、group、page、layer 还是 rank 计费，防止漏算或重复计费。

原稿示例在 `E_K = E_V = 256` 时：

```text
BF16/FP16 K = 512 字节；V = 512 字节；K+V = 1024 字节/token
裸 INT2 K  = 64 字节；V = 64 字节；K+V = 128 字节/token
参考 History K+V = 160 字节/token（含 pad，组成待核实）
```

为实际布局给出以下完整预算，并区分共享分配，避免重复相加：

```text
M_FULL = M_Sink_BF16 + M_Recent_BF16 + M_History_INT2
       + M_quant_metadata + M_mapping + M_alignment

M_cache_total = 所有真实物理缓存分配的总和
M_runtime_extra_peak = 旋转常量 + 算子 workspace + MTP 必要暂存
                     + 图捕获相关缓冲及其他新增分配的实际峰值
```

必须同时报告固定 token 数下的实际字节和固定 HBM 预算下的可用 token/block 容量。原生 KV 池可能按预算预分配，不能只比较进程总显存；也不能只给 FULL 裸 INT2 的理论压缩比，而忽略 GDN 和新增开销。

允许算法必需的有界工作区和 NPU 片上 tile 暂存，但必须说明生命周期、峰值与复用方式；不得演变为完整 BF16 History 副本。MTP 必要暂存只能服务原生推测生命周期，不得复制整段历史缓存。

### 5.4 三大条框架的采用条件

优先复用原有 KV 管理、block table 和逻辑分组，仅替换 FULL K/V 的存储和读取。原稿建议维持 GDN 的 A/B 区域，改造 FULL 使用的 B/C 区域；这是一条需要验证的候选路径。

设计文档必须证明：

1. GDN 物理视图、状态容量与更新语义保持正确。
2. FULL INT2 的节省能够被 allocator 或容量规划实际利用。
3. 无原生 FULL BF16 KV 与新 INT2 KV 并存的冗余池。
4. 新物理布局、原生页表逻辑和算子寻址完全一致。

若上述条件不成立，调整外部缓存规格/视图/分配适配方案并说明依据；不能以“最小改动”为由保留无效的显存优化，也不能修改原生源码绕过问题。

## 6. AscendC 算子与执行路径

### 6.1 先做完整算子清单

建立 `docs/operator_inventory.md`。逐项记录输入输出 shape/dtype/layout、调用阶段、数据量、同步点、现有实现、AscendC 入口、融合归属和验证方法。至少覆盖：

| 算子/能力 | 必须处理的问题 |
| --- | --- |
| `rotate_clip_quant_int2` | 旋转、裁剪、scale/量化参数、量化、位打包和写入的融合；K/V 参数分别核实。 |
| INT2 pack/unpack | 2-bit 编码、位序、尾部元素、head/group 边界、对齐、并发写入。 |
| KV Store / Recent→History | Sink/Recent 正确写入、窗口迁移、页地址映射，避免 staging 和双写。 |
| `dequant_inverse_rotate` | 用于参考对齐和确有需要的有界恢复；不得成为 decode/verify 全历史恢复路径。 |
| Decode Stage1 | 真正 fused INT2 CV kernel：压缩历史读取、解包/反量化、所需变换与 Attention 计算融合。 |
| Sink/Recent/window attention | 与 History 部分统一注意力语义，避免由大量 Torch 小算子拼接。 |
| Stage2/输出合并（若需要） | 正确合并各段/分块的 softmax 统计量和输出，验证稳定性。 |
| MTP 多 query verify | 同一历史 tile 跨 query 复用，正确 causal mask、位置和接受/拒绝后的缓存可见性。 |
| Prefill/chunked prefill | 复用原生必要计算，增量存储，避免旧 chunk 重搬运和重复量化。 |
| 批量元数据/索引操作 | 长度、slot/block 映射和提交回滚所需处理，避免逐请求 Python 热循环和 AiCPU 数据路径。 |

以上名称用于表达职责，最终函数名、文件名和融合边界由 reference 与目标硬件确定。不能机械地把每一行都实现成独立 kernel。

### 6.2 Decode 与 MTP 的性能不变量

- 不能每个 verify step 对全部历史做显式 `full_dequant + inverse rotation`。原稿指出稠密逆旋转可能额外引入 `O(LD²)` 计算；必须给出实际实现的计算量和带宽分析。
- 不将历史长度为 `L` 的完整 BF16 K/V 写回 HBM，再用另一个 kernel 读取做 Attention。
- MTP 不为每个 query 单独扫描同一份历史 INT2 HBM。设计历史 tile 的读取与跨 query 复用，量化 `q_len` 增长时的历史读取量。
- 多 query 必要的 Attention 算术会随 `q_len` 增加；禁止的是重复历史搬运、重复恢复及重复执行原生已经完成的工作，不能用省略必要计算满足指标。
- KV Store、window attention、恢复与合并不能堆积大量 Torch 小算子；用 profiler 检查 kernel launch 数、设备空隙和 Host 调度开销。
- 对等价的原生算子或 Attention 阶段做分阶段性能对照，融合后的算子性能应接近原生；最终仍须满足端到端不慢于原生的要求。
- 融合必须落到 AscendC 内核的数据复用和执行实现；仅包装多个既有调用为一个 Python API 不算完成。
- 对混合长度 batch 使用批量元数据和合适的 tiling，避免按最长请求无条件搬运所有请求的填充历史。

### 6.3 每个核心内核的设计交付内容

给出 AscendC 伪代码及最终实现，解释全局内存/片上缓冲布局、tile 选择、硬件计算单元分工、异步搬运与同步、累加精度、mask、边界处理、跨核归约和 workspace 生命周期。

图捕获所需的固定地址、workspace、算子注册与动态长度元数据必须与目标 `FULL_DECODE_ONLY` 路径兼容；MTP 中 `enforce_eager=true` 的实际作用域以原生行为为准，不能擅自把整个服务改成 eager。

## 7. 外部集成架构与初始化

以以下结构为起点，补全实际文件、函数和注册位置：

```text
原生 vLLM scheduler / KV manager
  └─ 原生 vLLM Ascend worker
       └─ 外部注册 / Hook / 子类 / 受控 monkey-patch
            └─ OSCAR plugin package
                 ├─ 配置、兼容性探测、FULL 层路由
                 ├─ KV spec / 视图 / 分配适配与批量元数据
                 ├─ prefill / decode / MTP 集成
                 └─ independent operator package
                      ├─ AscendC op_host / op_kernel
                      ├─ custom OPP 构建产物
                      └─ torch extension / 自定义算子注册
```

必须说明每个外部接入点调用或包装的原生符号，以及不改原生文件的实现方式。`vllm.plugins` 等接口是否适用，以当前版本核实为准。

初始化要求：

1. 版本判断采用版本/commit 记录与接口能力 probe，不使用单一字符串相等判断，也不使用无条件兼容放行。
2. 插件导入阶段避免提前触发平台初始化和 `DeviceOperator` 循环导入；明确加载时机，验证 CLI、API server 和各 worker 的导入顺序。
3. monkey-patch 必须幂等，保留函数签名与实例/类/静态方法的绑定语义，支持撤销到原生入口。
4. 校验 worker 中实际加载的插件、OPP/扩展路径和算子版本，不以主进程导入成功代替全 rank 验证。
5. 明确 NPU stream、设备、dtype、workspace 与图模式契约；Host 包装层不做 KV 数值计算。
6. OSCAR 被启用时，注册或路径选择失败必须明确失败，不能打印“跳过注入”后以原生路径冒充成功。

## 8. 一键部署、probe 与历史故障处理

### 8.1 唯一正式入口

最终交付一个 Bash 脚本，例如 `scripts/install_probe_serve.sh`，并在 README 给出一条实际可运行的命令。内部可以调用模块化子脚本；用户无需手动补做安装、编译或清理步骤。

执行顺序必须为：

```text
读取配置并记录环境/源码状态
  → 检查 NPU、依赖、编译器和接口能力
  → 安装外部插件并编译/加载 AscendC 算子
  → 若 PR 要求，生成或验证旋转/校准检查点
  → 执行算子、集成、TP=4 和目标路径 probe
  → 关闭临时引擎/worker，清理本任务资源
  → 验证 NPU 进程与内存已释放
  → 按附录 A 及必要的 OSCAR 插件配置启动正式服务
  → 验证就绪、实际推理及 OSCAR 路由
```

脚本必须：

- 在关键步骤失败时以非零退出码退出，打印阶段、错误、完整日志位置；probe 失败或资源未清理完毕时不得继续拉起正式服务。
- 可重复执行，记录插件/算子/配置版本和构建结果，避免加载旧产物。
- 用清晰的进程生命周期和退出处理管理本任务的 API server、EngineCore、worker、校准进程及共享内存。
- “清理 NPU”指释放本任务创建的进程和资源，并验证释放结果；不能只调用 `empty_cache()` 后假定 worker 已退出，不能无差别终止他人任务或重置整机设备。
- 对编译、worker 初始化、probe、退出等待和健康检查设置可诊断的超时，输出每个 rank 的进展。
- 正式启动前核对原生源码未被修改；保留目标命令各项参数，新增 OSCAR 配置的来源与作用必须明确。

开发期可以使用较小 probe 缩短反馈时间，但最终验收必须恢复目标配置、TP=4、MTP 和规定的长序列场景。

### 8.2 将历史错误转化为回归检查

下表给出需要防止重现的问题，不把单条日志直接当成已确定根因。完整原始材料见附录 C。

| 历史现象 | 必须落实的检查与处理 |
| --- | --- |
| `Unsupported vllm-ascend version: 0.23.1.dev0+g5cb98caaa.d20260822` | 区分目标版本与开发构建版本，核实 commit/接口兼容性，不误拒绝，也不盲目放行。 |
| `DeviceOperator` partially initialized / circular import | 调整外部插件导入依赖和注册时机，测试全新进程启动；不能捕获后跳过 OSCAR 注入。 |
| `Device string must not be empty` | 在创建 LLM/校准引擎前确认平台识别成功；定位先前导入失败，不用硬填 device 字符串掩盖问题。 |
| `LD_PRELOAD detected`、`Invalid thread pool!`、`ParallelOpenMP.cpp:64`、递归 terminate | 记录动态库和线程池环境，检查初始化、进程启动和线程设置顺序；通过复现诊断，不盲目增加预加载库或反复修改线程数。 |
| `No module named 'vllm._C'` | 按 Ascend 实际路径判断是否缺少必需能力，不能照搬 CUDA 扩展修复方式，也不能把所有 warning 都当作可忽略。 |
| Breakable cudagraph 被 Ascend 禁用的提示 | 区分原生平台能力提示与 OSCAR 新增故障，验证目标 `FULL_DECODE_ONLY` 实际执行路径。 |
| 旋转检查点生成时 NPU OOM | 历史记录为 NPU 0 总量 29.49 GiB、已分配 28.97 GiB、尝试再分配 172 MiB。核实 TP、残留进程、重复加载和校准峰值，不得用单卡加载整个模型或在服务占卡时重复加载模型。 |
| `'LLM' object has no attribute 'model'` | 使用当前版本实际支持的 worker/model 访问机制，不能假设前端 `LLM` 暴露内部模型。 |
| `function is not serializable`，提示 `VLLM_ALLOW_INSECURE_SERIALIZATION=1` | 核实 RPC 支持的消息/方法机制，优先使用已注册方法与可序列化数据；不能只因报错提示就默认打开不安全序列化绕过。 |
| `_guarded_pre_register() takes 1 positional argument but 2 were given` | 检查原函数签名与 Python 方法绑定，验证 CLI 参数注册和 worker 两条入口。 |
| probe 结束后 worker 未退、强杀、泄漏 shared_memory | 记录本任务进程与资源所有权，按顺序退出并等待，确认 NPU 释放，再启动下一阶段。 |
| `No available shared memory broadcast block found in 60 seconds` | 定位具体 rank 是否卡在编译、权重/KV 量化或通信；提供各阶段进度、超时和清理，不能只提高超时掩盖挂起。 |
| 原生 `rejection_sample` fallback warning | 记录其在基线中的状态；保持 MTP 语义，并区分原生警告与 OSCAR 新增的错误路由。 |
| 原生 int32/int64 `ArgSort` 转 AiCPU warning | profiler 区分原生已有路径与插件新增操作；插件热路径不得引入同类数据处理回退，也不能未经数值范围验证就把索引强转 float32。 |

旋转检查点若需生成，必须复用正确的权重量化与 TP 配置，明确其与模型/层/rank/维度/PR 版本的对应关系，避免重复校准和错误复用。校准中的旋转、量化及大规模数据计算同样遵守 NPU 约束，不能因处于离线阶段就转移到 CPU 处理。

## 9. 端到端 Checklist 与证据规则

先读 reference，再将以下初始清单细化为项目中的 **`docs/checklist.md`**。这是持续更新的唯一完成状态表，不能只在最终汇报时补勾。

规则：

- `[ ]` 表示尚未通过；`[x]` 仅表示该项验收已通过且证据存在。
- 每项保留稳定 ID。状态可补充 `pending / in_progress / passed / failed / blocked / not_run`；失败或未运行不得勾选。
- 完成后立即更新代码位置、必要片段、解释、执行命令、环境、预期与实测结果、日志/报告路径。
- 代码实现完成与 NPU 验证完成分别记录；静态审查、模拟或编译不能代替真机验收。
- 证据引用固定 commit 和文件/函数/行号；后续修改使证据失效时重新验证并更新。
- 若核实后某个补充项不适用，写明范围与证据，不能用“不适用”撤掉硬约束。

### 阶段 A：输入、环境与 reference 核实

- [ ] **A01** 建立 reference 清单，锁定 OSCAR PR 与相关 vLLM/Ascend commit，确认未参考失败仓库。
- [ ] **A02** 记录实际软件、NPU、CANN/编译环境、模型配置和原生源码状态。
- [ ] **A03** 跑通原生目标配置的基线，保存命令、输入、版本、MTP/图模式与资源记录。
- [ ] **A04** 读懂第 3.3 节调用链，记录 FULL/GDN、TP、KV 分组、MTP 和图模式的真实入口。
- [ ] **A05** 核实第 4 节全部歧义，给出层/组/Tensor/地址映射与字节来源。
- [ ] **A06** 读取 PR 的配置、旋转、量化和精度验证方法，完善本 Checklist，列出仍缺失的输入。

### 阶段 B：设计与预算

- [ ] **B01** 产出 `docs/design.md`，列出外部新增文件/函数/类及其包装的原生入口。
- [ ] **B02** 完成请求窗口、物理页、GDN 隔离、TP 分片、页生命周期和 MTP 提交/回滚设计。
- [ ] **B03** 给出所有 payload、元数据、padding、workspace 和总 HBM 公式，证明压缩收益可落地。
- [ ] **B04** 完成算子清单、AscendC 融合划分、Decode Stage1 伪代码和逐阶段读写流程。
- [ ] **B05** 给出 decode 计算量、HBM 流量、MTP 跨 query 复用和 Host 调度预算。
- [ ] **B06** 固定功能、精度、性能测试方法和验收门槛，设计一键脚本及错误清理流程。

阶段 B 通过后直接进入实现；不必重复等待用户批准已授权的实现工作。

### 阶段 C：独立算子与插件实现

- [ ] **C01** 独立插件/算子工程可安装、可编译、可加载，原生源码未修改，安装脚本不使用 `cp`。
- [ ] **C02** 版本能力 probe、延迟注册和幂等 Hook 通过 CLI、API server、各 worker 初始化验证。
- [ ] **C03** 实现并验证 INT2 编码、旋转/裁剪/量化、KV store 与 Recent→History 增量迁移。
- [ ] **C04** 实现真实 fused INT2 CV Decode Stage1，并完成必要的分块/分段输出合并。
- [ ] **C05** Sink/Recent Attention、History Attention 和参考数学语义一致，数值稳定。
- [ ] **C06** 实现原生页表兼容的物理分配与视图，GDN 不受影响，HBM 中无冗余完整历史副本。
- [ ] **C07** 在确有需要的路径提供有界恢复能力，证明 decode/verify 不走全历史恢复。

### 阶段 D：端到端执行路径

- [ ] **D01** 首次 prefill 正确，未引入不必要的重复压缩和恢复。
- [ ] **D02** chunked prefill 正确，已处理历史不随每个新 chunk 重搬运或重压缩。
- [ ] **D03** 普通 decode 持续进入压缩历史路径，无全历史 BF16 HBM 恢复。
- [ ] **D04** MTP draft/verify 与原生流程一致，完整接受、部分接受、完全拒绝后的缓存和 GDN 状态正确。
- [ ] **D05** MTP 多 query 复用历史 tile，读取量测量支持“不随 `q_len` 线性倍增”。
- [ ] **D06** 混合长度和 prefill/decode/verify 混合调度正确，不依赖逐请求 Python 热循环。
- [ ] **D07** 16K、32K、50K 长输入实际使用 OSCAR，跨页与跨窗口边界均正确。
- [ ] **D08** TP=4、异步调度、目标图模式和 W8A8 权重量化共存，所有 rank 路由正确。
- [ ] **D09** 前缀缓存、页共享/复用、请求取消/结束、抢占恢复等目标环境可达生命周期通过验证。

### 阶段 E：精度、性能与显存验收

- [ ] **E01** 核心算子对齐参考；覆盖 pack/unpack、量化边界、尾块和 attention 数值误差。
- [ ] **E02** GDN 隔离检查通过；FULL 量化后的层输出、模型输出与约定质量指标达标。
- [ ] **E03** MTP 接受率、接受长度、输出正确性和有效生成吞吐完成对照。
- [ ] **E04** 固定 token 数和固定 HBM 预算两种口径下证明真实缓存收益，计入全部新增开销。
- [ ] **E05** profiler 证明无禁止的 CPU/AiCPU 数据处理、同步、冗余 KV 双写及全历史恢复。
- [ ] **E06** 按第 10 节逐工况比较原生性能；无未解释退化，不以平均值掩盖慢项。
- [ ] **E07** 对照第 8.2 节验证历史故障防护，保存实际覆盖结果而非仅声明“已避免”。

### 阶段 F：一键交付

- [ ] **F01** 一条 Bash 命令完成安装、编译、所需校准、probe、资源清理和正式启动。
- [ ] **F02** 验证正常退出、失败退出和中断后的清理，重复执行不会遗留 worker 或加载旧产物。
- [ ] **F03** 正式服务按目标配置启动并完成真实请求，记录 OSCAR compressed decode/MTP 路由证据。
- [ ] **F04** 交付代码、设计、完整 Checklist、测试/性能/显存报告、日志及 README 中的一键命令。
- [ ] **F05** 检查所有硬约束和未完成项，最终结论准确列出已验证范围、失败项及阻塞项。

### 每个 Checkpoint 的证据模板

```markdown
### C04 — fused INT2 CV Decode Stage1

- 状态：pending / in_progress / passed / failed / blocked / not_run
- 对应要求：H07、H09，以及相关验收项
- 代码版本：<commit>
- 实现位置：<文件:行号、函数/类名>
- 关键代码片段：<仅列能证明该要求的部分>
- 实现解释：<如何满足该要求；为什么没有走禁止路径>
- 验证环境与配置：<NPU、版本、TP、输入形状、模式>
- 执行命令：<可复现命令>
- 预期结果：<事先定义的判定标准>
- 实测结果：<数值、状态、路径命中、资源或 profiler 结果>
- 证据文件：<日志/测试输出/报告/trace 路径>
- 剩余问题与下一步：<如无则写无>
```

## 10. 验收矩阵与性能比较

### 10.1 功能与精度

至少覆盖下表。边界值应根据真实 block size、head/group 形状扩展，不能只测整齐形状。

| 维度 | 最低覆盖要求 |
| --- | --- |
| 窗口边界 | `S-1/S/S+1`、`S+R-1/S+R/S+R+1`，短序列、空 History、Recent 滑出和跨页。 |
| 数值与位布局 | INT2 全部编码值、裁剪边界、scale 边界、尾部打包、K/V 量化参数与层/rank 对应。 |
| 地址与 head 映射 | 非连续 block table、非零偏移、尾块、TP 本地分片，以及实际模型使用的 GQA/head 对应关系。 |
| 输入长度 | 短序列及 16K、32K、50K；测试中明确 K 的计数口径，至少包含 16,384、32,768、50,000 token。 |
| Prefill | 单次 prefill、多个 chunk、不同 chunk 划分产生一致的算法结果，超出单批 token 预算的实际路径。 |
| Decode | 多步追加、窗口迁移、页边界、请求结束后页复用。 |
| MTP | 正式配置 `num_speculative_tokens=3`；覆盖原生实际产生的 `q_len`、全接受/部分接受/全拒绝及跨窗口/页边界。 |
| Batch | 单请求、多个并发等级、长短混合、prefill/decode/verify 共存、请求动态加入/退出。 |
| 运行方式 | TP=4、异步调度、目标 `FULL_DECODE_ONLY` 及 MTP 原生 eager 作用域。 |
| KV 生命周期 | 前缀命中/未命中、共享页修改隔离、取消、回收复用，以及目标配置实际可达的抢占恢复。 |
| 配置上限 | 保留 `max_model_len=262144`、`max_num_seqs=128` 等原设置；核实容量与边界，报告极限长度的实测范围和资源限制，不能静默缩短配置。 |

质量评估需区分 **量化算法固有误差** 与 **移植实现错误**：先对齐指定 OSCAR reference，再与原生模型比较层输出、logits/任务指标和 MTP 接受行为。原稿未给出统一数值容差或质量下降门槛，必须在调优前从 PR/现有标准提取并写入验收配置；缺少依据的门槛标记为待确认，不能事后按结果放宽。

### 10.2 公平的原生性能对照

“不比原生慢”是验收目标，不是未测量就可以做出的保证。附录 B 是历史线索，不能代替同环境重跑的配对基线。

固定并记录：NPU 与 TP、软件 commit、模型与权重量化、启动参数、输入内容及 token 数、输出 token 数、采样参数、并发/到达方式、prefix cache 状态、MTP 参数、预热和计时范围。

至少报告：

- TTFT、TPOT/ITL、端到端延迟及相应 P50/P95。
- Prompt throughput、有效生成吞吐、完成请求数、失败率与超时。
- MTP acceptance length、draft acceptance rate、各位置接受率及 draft/verify 耗时。
- KV 实际字节、可用容量、进程峰值 HBM、workspace 和 MTP 暂存。
- 核心 kernel 时延、launch 数、Host 时间、同步次数和历史读取流量。

在测试前固定重复次数、统计方法和测量噪声处理方式。每个必测工况单独列原生/OSCAR/比值/差异，不把不同 Running、Waiting 或缓存命中状态的日志直接相比。若观察到退化，应定位并优化；证据不足、只在部分场景达标或仍有退化时，报告尚未完成性能验收，不能用总体平均加速抵消单项失败。

历史原生日志包含：单请求生成吞吐约 9.9–12.2 tokens/s 的若干窗口，以及 Running=3/6/8/11/14/15 时的 8.2/19.4/38.0/45.8/58.4/78.7 tokens/s。对应 prompt throughput、waiting、cache 使用率与 MTP 指标均不同，完整证据见附录 B。这些数字仅用于复现场景和发现异常。

### 10.3 必须证明“压缩路径真的生效”

使用可关闭的诊断计数、trace 或 profiler，至少证明：

- 哪些 FULL 层使用 OSCAR，GDN 层仍走正确原生路径。
- prefill、decode、MTP verify 的实际后端/内核选择，以及 16K–50K 长输入的路径命中。
- HBM 中实际分配了哪些缓存，是否有完整 BF16 History、重复 staging 或 shadow cache。
- 改变历史长度 `L` 和实际 `q_len` 时，历史读取量、恢复量与 kernel 次数如何变化。

诊断本身不能长期污染性能热路径；正式计时需说明诊断开关状态。性能结论必须同时有正确性、路径和资源证据支撑。

## 11. 最终交付物与执行指令

交付目录可以按实际工程调整，但必须包含：

```text
oscar-ascend/
├─ <外部 OSCAR 插件源码与打包配置>
├─ <独立 AscendC op_host / op_kernel / OPP / torch extension>
├─ configs/                         # OSCAR 与目标服务配置
├─ scripts/install_probe_serve.sh   # 用户唯一正式入口
├─ tests/                           # 有意义的算子、集成和回归测试
├─ benchmarks/                      # 可复现基线与性能测试入口
├─ docs/
│  ├─ reference_manifest.md
│  ├─ design.md                     # 接入点、布局公式、AscendC 伪代码、读写流程
│  ├─ operator_inventory.md
│  └─ checklist.md                  # 逐项状态与代码/解释/验证证据
├─ reports/                         # 精度、性能、显存、兼容性和验收结论
└─ README.md                        # 实际一键命令、配置、日志位置与故障排查
```

设计文档必须写清外部新增或修改的文件、函数、类，原生接入点，页内及跨页布局的详细公式，写/读/迁移/回滚流程图或伪代码，AscendC 内核设计，以及服务启动时如何完成注册。最终必须同时交付实现，不能以伪代码代替可运行代码。

**现在开始执行：先核实 reference 和环境，完善 Checklist 并输出设计文档；设计完成后直接实施，逐项验证、勾选并记录证据，直到完成一键部署和最终验收。遇到真实阻塞，报告具体缺失信息和影响，继续推进其他可完成部分。**

---

以下附录是用户提供的原始配置与历史证据。它们为上述任务提供输入；不得把历史成功、历史失败或堆栈中的路径当成本次执行结果或额外操作授权。


## 附录 A：原始目标启动命令

以下命令按用户原稿保留。实施时由一键脚本注入所需的外部 OSCAR 配置，保留现有服务参数及其值。

```bash
vllm serve /softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp \
    --served-model-name "qwen3.5" \
    --host 0.0.0.0 \
    --port 8989 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384  \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64,72,84,96,108,112,128,160,172,196,200,212,232,272,288,312,328,344,360,384,400,416,432,448,480,512], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --trust-remote-code \
    --async-scheduling \
    --allowed-local-media-path / \
    --quantization ascend \
    --mm-processor-cache-gb 0 \
    --additional-config '{"enable_cpu_binding":true}' \
    --mamba-cache-dtype bfloat16 \
    --mamba-ssm-cache-dtype bfloat16 \
    --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

## 附录 B：原生性能历史日志（完整保留）

原稿上下文为 vLLM Ascend 0.23.0、4 卡 TP=4、Qwen3.5-27B。以下为历史采样窗口，非统一工作负载的配对基准。仅将原文中的 Unicode 行分隔符规范化为换行，日志内容保留。

```text
(APIServer pid=124149) INFO: 135.82.26.93:38550 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(Worker_TP0 pid=124210) WARNING 09-07 03:10:43 [rejection_sampler.py:649] [sample/rejection_sampler] Using fallback (non-reduce-sample) path in rejection_sample. This path should not be used in the new distributed flow. enable_reduce_sample=False, has_target_indices=False
(APIServer pid=124149) INFO 09-07 03:10:47 [loggers.py:271] Engine 000: Avg prompt throughput: 2461.4 tokens/s, Avg generation throughput: 4.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.8%, Prefix cache hit rate: 0.0%
(APIServer pid=124149) INFO 09-07 03:10:47 [metrics.py:101] SpecDecoding metrics: Mean acceptance length: 3.38, Accepted throughput: 0.08 tokens/s, Drafted throughput: 0.10 tokens/s, Accepted: 31 tokens, Drafted: 39 tokens, Per-position acceptance rate: 0.846, 0.846, 0.692, Avg Draft acceptance rate: 79.5%
(APIServer pid=124149) INFO 09-07 03:10:57 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 11.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.8%, Prefix cache hit rate: 0.0%
(APIServer pid=124149) INFO 09-07 03:10:57 [metrics.py:101] SpecDecoding metrics: Mean acceptance length: 3.05, Accepted throughput: 7.60 tokens/s, Drafted throughput: 11.10 tokens/s, Accepted: 76 tokens, Drafted: 111 tokens, Per-position acceptance rate: 0.811, 0.703, 0.541, Avg Draft acceptance rate: 68.5%
(APIServer pid=124149) INFO 09-07 03:11:07 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 12.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.8%, Prefix cache hit rate: 0.0%
(APIServer pid=124149) INFO 09-07 03:11:07 [metrics.py:101] SpecDecoding metrics: Mean acceptance length: 3.30, Accepted throughput: 8.50 tokens/s, Drafted throughput: 11.10 tokens/s, Accepted: 85 tokens, Drafted: 111 tokens, Per-position acceptance rate: 0.946, 0.730, 0.622, Avg Draft acceptance rate: 76.6%
(APIServer pid=124149) INFO 09-07 03:11:17 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 10.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.8%, Prefix cache hit rate: 0.0%
(APIServer pid=124149) INFO 09-07 03:11:17 [metrics.py:101] SpecDecoding metrics: Mean acceptance length: 2.79, Accepted throughput: 6.80 tokens/s, Drafted throughput: 11.40 tokens/s, Accepted: 68 tokens, Drafted: 114 tokens, Per-position acceptance rate: 0.789, 0.605, 0.395, Avg Draft acceptance rate: 59.6%
(APIServer pid=124149) INFO 09-07 03:11:27 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 9.9 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.8%, Prefix cache hit rate: 0.0%
(APIServer pid=124149) INFO 09-07 03:11:27 [metrics.py:101] SpecDecoding metrics: Mean acceptance length: 2.83, Accepted throughput: 6.40 tokens/s, Drafted throughput: 10.50 tokens/s, Accepted: 64 tokens, Drafted: 105 tokens, Per-position acceptance rate: 0.829, 0.600, 0.400, Avg Draft acceptance rate: 61.0%
(APIServer pid=124149) INFO: 135.82.26.93:42704 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42706 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42714 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42722 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42724 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42732 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42742 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42744 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42748 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42758 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42774 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42784 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42788 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42790 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO 09-07 03:11:37 [loggers.py:271] Engine 000: Avg prompt throughput: 80.7 tokens/s, Avg generation throughput: 9.1 tokens/s, Running: 1 reqs, Waiting: 6 reqs, GPU KV cache usage: 2.2%, Prefix cache hit rate: 26.1%
(APIServer pid=124149) INFO 09-07 03:11:37 [metrics.py:101] SpecDecoding metrics: Mean acceptance length: 3.13, Accepted throughput: 6.40 tokens/s, Drafted throughput: 8.99 tokens/s, Accepted: 64 tokens, Drafted: 90 tokens, Per-position acceptance rate: 0.933, 0.700, 0.500, Avg Draft acceptance rate: 71.1%
(APIServer pid=124149) INFO: 135.82.26.93:42798 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42814 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42826 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42840 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42844 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42856 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42870 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42878 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42894 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42910 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42926 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42930 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42940 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42954 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42970 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42980 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42986 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42988 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42704 - "POST /v1/chat/completions HTTP/1.1" 200 OK
[rank1]:[W907 03:11:39.571834370 ArgSortKernelNpuOpApi.cpp:41] Warning: Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel is running on AiCpu.If you are more concerned about high-performance execution,please cast dtype to float32. (function operator())
[rank2]:[W907 03:11:39.572183330 ArgSortKernelNpuOpApi.cpp:41] Warning: Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel is running on AiCpu.If you are more concerned about high-performance execution,please cast dtype to float32. (function operator())
[rank3]:[W907 03:11:39.573440150 ArgSortKernelNpuOpApi.cpp:41] Warning: Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel is running on AiCpu.If you are more concerned about high-performance execution,please cast dtype to float32. (function operator())
[rank0]:[W907 03:11:39.573642340 ArgSortKernelNpuOpApi.cpp:41] Warning: Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel is running on AiCpu.If you are more concerned about high-performance execution,please cast dtype to float32. (function operator())
(APIServer pid=124149) INFO: 135.82.26.93:42722 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO 09-07 03:11:47 [loggers.py:271] Engine 000: Avg prompt throughput: 6229.9 tokens/s, Avg generation throughput: 8.2 tokens/s, Running: 3 reqs, Waiting: 28 reqs, GPU KV cache usage: 10.1%, Prefix cache hit rate: 3.8%
(APIServer pid=124149) INFO 09-07 03:11:47 [metrics.py:101] SpecDecoding metrics: Mean acceptance length: 3.20, Accepted throughput: 5.50 tokens/s, Drafted throughput: 7.50 tokens/s, Accepted: 55 tokens, Drafted: 75 tokens, Per-position acceptance rate: 0.840, 0.720, 0.640, Avg Draft acceptance rate: 73.3%
(APIServer pid=124149) INFO 09-07 03:11:57 [loggers.py:271] Engine 000: Avg prompt throughput: 6964.8 tokens/s, Avg generation throughput: 19.4 tokens/s, Running: 6 reqs, Waiting: 26 reqs, GPU KV cache usage: 20.8%, Prefix cache hit rate: 2.2%
(APIServer pid=124149) INFO 09-07 03:11:57 [metrics.py:101] SpecDecoding metrics: Mean acceptance length: 3.24, Accepted throughput: 13.20 tokens/s, Drafted throughput: 17.70 tokens/s, Accepted: 132 tokens, Drafted: 177 tokens, Per-position acceptance rate: 0.881, 0.712, 0.644, Avg Draft acceptance rate: 74.6%
(APIServer pid=124149) INFO: 135.82.26.93:42758 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO 09-07 03:12:07 [loggers.py:271] Engine 000: Avg prompt throughput: 6731.4 tokens/s, Avg generation throughput: 38.0 tokens/s, Running: 8 reqs, Waiting: 23 reqs, GPU KV cache usage: 28.6%, Prefix cache hit rate: 1.6%
(APIServer pid=124149) INFO 09-07 03:12:07 [metrics.py:101] SpecDecoding metrics: Mean acceptance length: 3.35, Accepted throughput: 26.50 tokens/s, Drafted throughput: 33.90 tokens/s, Accepted: 265 tokens, Drafted: 339 tokens, Per-position acceptance rate: 0.920, 0.779, 0.646, Avg Draft acceptance rate: 78.2%
(APIServer pid=124149) INFO: 135.82.26.93:42774 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO 09-07 03:12:17 [loggers.py:271] Engine 000: Avg prompt throughput: 7926.7 tokens/s, Avg generation throughput: 45.8 tokens/s, Running: 11 reqs, Waiting: 21 reqs, GPU KV cache usage: 37.4%, Prefix cache hit rate: 1.2%
(APIServer pid=124149) INFO 09-07 03:12:17 [metrics.py:101] SpecDecoding metrics: Mean acceptance length: 3.30, Accepted throughput: 31.69 tokens/s, Drafted throughput: 41.39 tokens/s, Accepted: 317 tokens, Drafted: 414 tokens, Per-position acceptance rate: 0.899, 0.739, 0.659, Avg Draft acceptance rate: 76.6%
(APIServer pid=124149) INFO 09-07 03:12:27 [loggers.py:271] Engine 000: Avg prompt throughput: 5274.4 tokens/s, Avg generation throughput: 58.4 tokens/s, Running: 14 reqs, Waiting: 18 reqs, GPU KV cache usage: 47.9%, Prefix cache hit rate: 1.0%
(APIServer pid=124149) INFO 09-07 03:12:27 [metrics.py:101] SpecDecoding metrics: Mean acceptance length: 3.23, Accepted throughput: 40.19 tokens/s, Drafted throughput: 53.99 tokens/s, Accepted: 402 tokens, Drafted: 540 tokens, Per-position acceptance rate: 0.872, 0.744, 0.617, Avg Draft acceptance rate: 74.4%
(APIServer pid=124149) INFO: 135.82.26.93:42844 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO: 135.82.26.93:42856 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=124149) INFO 09-07 03:12:37 [loggers.py:271] Engine 000: Avg prompt throughput: 6750.8 tokens/s, Avg generation throughput: 78.7 tokens/s, Running: 15 reqs, Waiting: 16 reqs, GPU KV cache usage: 53.1%, Prefix cache hit rate: 0.8%
```

## 附录 C：历史环境与失败日志（完整保留）

以下为故障证据。JSON 外观的片段存在原始格式缺项，堆栈含截断或转写内容，均按原样保留，不将其视为可直接执行的配置。日志中的旧项目路径不在允许参考的代码范围内。

```text
[W908 06:04:38.048124580 FunctionLoader.cpp:48] Warning: LD_PRELOAD detected, FunctionLoader prefers RTLD_DEFAULT for symbol resolution. (function operator())
{
"status": "failed",
"npu_acceptance": "not_run",
"versions": {
"vllm": "0.23.0+empty",
"vllm-ascend": "0.23.1.dev0+g5cb98caaa.d20260822",
"torch": "2.10.0+cpu",
"torch_npu": "2.10.0.post4",
"triton": "3.2.0"
"error": "Unsupported vllm-ascend version: 0.23.1.dev0+g5cb98caaa.d20260822"
}


"error": "cannot import name 'DeviceOperator' from partially initialized module 'vllm_ascend.device.device_op' (most likely due to a circular import) (/vllm-workspace/vllm-ascend/vllm_ascend/device/device_op.py)"

Exception raised from set_num_threads at /pytorch/aten/src/ATen/ParallelOpenMP.cpp:64 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits, std::allocator >) + 0xc8 (0xffff841c5978 in /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__cxx11::basic_string<char, std::char_traits, std::allocator > const&) + 0xc4 (0xffff84169adc in /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #2: c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, char const*) + 0x48 (0xffff841c2028 in /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #3: + 0x14b2868 (0xffff85342868 in /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
frame #4: torch::autograd::Engine::thread_init(int, std::shared_ptrtorch::autograd::ReadyQueue const&, bool) + 0x234 (0xffff8992a124 in /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
frame #5: + 0xad58c4 (0xffff8f0658c4 in /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libtorch_python.so)
frame #6: + 0xd29cc (0xffff929529cc in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #7: + 0x803c8 (0xffff92b303c8 in /lib/aarch64-linux-gnu/libc.so.6)
frame #8: + 0xe9edc (0xffff92b99edc in /lib/aarch64-linux-gnu/libc.so.6)

terminate called recursively
terminate called after throwing an instance of 'c10::Error'
what(): pool INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/ParallelOpenMP.cpp":64, please report a bug to PyTorch. Invalid thread pool!
Exception raised from set_num_threads at /pytorch/aten/src/ATen/ParallelOpenMP.cpp:64 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits, std::allocator >) + 0xc8 (0xffff841c5978 in /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__cxx11::basic_string<char, std::char_traits, std::allocator > const&) + 0xc4 (0xffff84169adc in /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #2: c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, char const*) + 0x48 (0xffff841c2028 in /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #3: + 0x14b2868 (0xffff85342868 in /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
frame #4: torch::autograd::Engine::thread_init(int, std::shared_ptrtorch::autograd::ReadyQueue const&, bool) + 0x234 (0xffff8992a124 in /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
frame #5: + 0xad58c4 (0xffff8f0658c4 in /usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libtorch_python.so)
frame #6: + 0xd29cc (0xffff929529cc in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #7: + 0x803c8 (0xffff92b303c8 in /lib/aarch64-linux-gnu/libc.so.6)
frame #8: + 0xe9edc (0xffff92b99edc in /lib/aarch64-linux-gnu/libc.so.6)


WARNING 09-03 12:43:08 [interface.py:255] Failed to import from vllm._C: ModuleNotFoundError("No module named 'vllm._C'")
WARNING 09-03 12:43:08 [interface.py:255] Failed to import from vllm._C: ModuleNotFoundError("No module named 'vllm._C'")
WARNING 09-03 12:43:08 [interface.py:255] Failed to import from vllm._C: ModuleNotFoundError("No module named 'vllm._C'")
WARNING 09-03 12:43:08 [interface.py:255] Failed to import from vllm._C: ModuleNotFoundError("No module named 'vllm._C'")
INFO 09-03 12:43:08 [platform.py:62] Breakable cudagraph is force disabled on Ascend because DeepSeek V4 PIECEWISE cudagraph is not supported yet.
[oscar-ascend] 平台不可用，跳过注入: cannot import name 'DeviceOperator' from partially initialized module 'vllm_ascend.device.device_op' (most likely due to a circular import) (/vllm-workspace/vllm-ascend/vllm_ascend/device/device_op.py)
INFO 09-03 12:43:08 [api_utils.py:273] non-default args: {'dtype': 'bfloat16', 'max_model_len': 128, 'disable_log_stats': True, 'enforce_eager': True, 'model': '/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp'}
Traceback (most recent call last):
File "/workspace/new_oscar_triton/tools/gen_rotations.py", line 193, in
sys.exit(main())
^^^^^^
File "/workspace/new_oscar_triton/tools/gen_rotations.py", line 55, in main
llm = LLM(
^^^^
File "/vllm-workspace/vllm/vllm/entrypoints/llm.py", line 349, in init
self.llm_engine = LLMEngine.from_engine_args(
^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/vllm-workspace/vllm/vllm/v1/engine/llm_engine.py", line 162, in from_engine_args
vllm_config = engine_args.create_engine_config(usage_context)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/vllm-workspace/vllm/vllm/engine/arg_utils.py", line 1714, in create_engine_config
device_config = DeviceConfig(device=cast(Device, current_platform.device_type))
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/python3.12.13/lib/python3.12/site-packages/pydantic/_internal/_dataclasses.py", line 121, in init
s.pydantic_validator.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
File "/vllm-workspace/vllm/vllm/config/device.py", line 78, in post_init
self.device = torch.device(self.device_type)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Device string must not be empty
[ERROR] 2026-09-03-12:43:08 (PID:372, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
❌ [oscar-ascend] 旋转检查点生成失败
日志: /tmp/oscar_ascend_logs/20260903_124227

File "/vllm-workspace/vllm/vllm/engine/arg_utils.py", line 1714, in create_engine_config
device_config = DeviceConfig(device=cast(Device, current_platform.device_type))
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/python3.12.13/lib/python3.12/site-packages/pydantic/_internal/_dataclasses.py", line 121, in init
s.pydantic_validator.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
File "/vllm-workspace/vllm/vllm/config/device.py", line 78, in post_init
self.device = torch.device(self.device_type)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Device string must not be empty
[ERROR] 2026-09-03-12:49:01 (PID:453, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
❌ [oscar-ascend] 旋转检查点生成失败
日志: /tmp/oscar_ascend_logs/20260903_124820

(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] return Qwen3_5DecoderLayer(
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] ^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] File "/vllm-workspace/vllm/vllm/model_executor/models/qwen3_5.py", line 163, in init
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] self.mlp = Qwen3NextMLP(
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] ^^^^^^^^^^^^^
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] File "/vllm-workspace/vllm/vllm/model_executor/models/qwen2_moe.py", line 90, in init
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] self.gate_up_proj = MergedColumnParallelLinear(
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] File "/vllm-workspace/vllm-ascend/vllm_ascend/ops/linear.py", line 248, in init
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] AscendColumnParallelLinear.init(
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] File "/vllm-workspace/vllm-ascend/vllm_ascend/ops/linear.py", line 425, in init
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] self.quant_method.create_weights(
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] File "/vllm-workspace/vllm-ascend/vllm_ascend/quantization/method_adapters.py", line 64, in create_weights
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] weight_dict = self.quant_method.get_weight(input_size_per_partition, output_size_per_partition, params_dtype)
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] File "/vllm-workspace/vllm-ascend/vllm_ascend/quantization/methods/w8a8_dynamic.py", line 62, in get_weight
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/utils/_device.py", line 109, in torch_function
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] return func(*args, **kwargs)
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=610) ERROR 09-03 12:54:51 [core.py:1195] torch.OutOfMemoryError: NPU out of memory. Tried to allocate 172.00 MiB (NPU 0; 29.49 GiB total capacity; 28.97 GiB already allocated; 28.97 GiB current active; 122.25 MiB free; 28.98 GiB reserved in total by PyTorch).If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
(EngineCore pid=610) Process EngineCore:

File "/workspace/new_oscar_triton/tools/gen_rotations.py", line 138, in main
model_mod = llm.model.model if hasattr(llm.model, "model") else llm.model
^^^^^^^^^
AttributeError: 'LLM' object has no attribute 'model'
(EngineCore pid=1703) INFO 09-04 00:07:36 [core.py:1178] [shutdown] EngineCore: trigger received signal=SIGTERM
(EngineCore pid=1703) INFO 09-04 00:07:36 [core.py:1297] [shutdown] EngineCore: start mode=abort timeout=0s
(EngineCore pid=1703) INFO 09-04 00:07:36 [core.py:1328] [shutdown] EngineCore: request processing complete; starting resource teardown
(EngineCore pid=1703) INFO 09-04 00:07:36 [core.py:1191] [shutdown] EngineCore: exiting busy loop
(EngineCore pid=1703) INFO 09-04 00:07:36 [multiproc_executor.py:428] [shutdown] Executor: waiting for worker exit count=4
(Worker_TP0 pid=1731) INFO 09-04 00:07:36 [multiproc_executor.py:790] Parent process exited, terminating worker queues
(EngineCore pid=1703) WARNING 09-04 00:07:40 [multiproc_executor.py:438] [shutdown] Executor: workers still running after grace period; sending SIGTERM count=4
[ERROR] 2026-09-04-00:07:36 (PID:1672, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
WARNING 09-04 00:07:41 [utils.py:607] [shutdown] Process manager: force killing remaining processes count=1
/usr/local/python3.12.13/lib/python3.12/multiprocessing/resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown
warnings.warn('resource_tracker: There appear to be %d '
❌ [oscar-ascend] 旋转检查点生成失败

File "/workspace/new_oscar_triton/tools/gen_rotations.py", line 134, in main
per_rank = llm.llm_engine.apply_model(calib.capture_cov)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/vllm-workspace/vllm/vllm/v1/engine/llm_engine.py", line 420, in apply_model
return self.collective_rpc("apply_model", args=(func,))
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/vllm-workspace/vllm/vllm/v1/engine/llm_engine.py", line 417, in collective_rpc
return self.engine_core.collective_rpc(method, timeout, args, kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/vllm-workspace/vllm/vllm/v1/engine/core_client.py", line 927, in collective_rpc
return self.call_utility("collective_rpc", method, timeout, args, kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/vllm-workspace/vllm/vllm/v1/engine/core_client.py", line 864, in call_utility
self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))
File "/vllm-workspace/vllm/vllm/v1/engine/core_client.py", line 850, in _send_input
msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/vllm-workspace/vllm/vllm/v1/serial_utils.py", line 171, in encode
bufs[0] = self.encoder.encode(obj)
^^^^^^^^^^^^^^^^^^^^^^^^
File "/vllm-workspace/vllm/vllm/v1/serial_utils.py", line 222, in enc_hook
raise TypeError(
TypeError: Object of type <class 'function'> is not serializableSet VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow fallback to pickle-based serialization.

File "/vllm-workspace/vllm/vllm/entrypoints/cli/main.py", line 88, in main
cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/vllm-workspace/vllm/vllm/entrypoints/cli/serve.py", line 164, in subparser_init
serve_parser = make_arg_parser(serve_parser)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/vllm-workspace/vllm/vllm/entrypoints/openai/cli_args.py", line 382, in make_arg_parser
parser = AsyncEngineArgs.add_cli_args(parser)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/vllm-workspace/vllm/vllm/engine/arg_utils.py", line 2597, in add_cli_args
current_platform.pre_register_and_update(parser)
TypeError: _install_pre_register_guard.._guarded_pre_register() takes 1 positional argument but 2 were given
[ERROR] 2026-09-04-01:19:23 (PID:358, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception

(APIServer pid=53556) INFO 09-08 11:48:06 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=53556) INFO 09-08 11:48:06 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=53556) INFO 09-08 11:48:06 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=53556) INFO: Started server process [53556]
(APIServer pid=53556) INFO: Waiting for application startup.
(APIServer pid=53556) INFO: Application startup complete.
(APIServer pid=53556) INFO: 135.82.26.93:46528 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore pid=53623) INFO 09-08 11:49:29 [patch_shm_broadcast.py:74] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
```
