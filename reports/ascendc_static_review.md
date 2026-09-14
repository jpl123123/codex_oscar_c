# AscendC 静态复核及容量 admission 修正

状态：**静态审查已执行；CANN 编译、NPU 数值、跨核同步与性能验证均未运行**。本机没有 CANN/torch_npu/NPU，也没有可用于本任务的 CMake 命令；没有把普通 C++ 语法分析或 Host 测试标为 AscendC 编译通过。

审查范围为当前独立工程 `csrc/op_kernel`、host tiling、launcher ABI、Torch binding/CMake，以及外部缓存 admission。AscendC API 使用只对照允许的 `references/vllm-ascend`，HEAD 为 `19e436985102f4ed3aad36c137a6481653688a6c`。kernel 文件由算子实现任务持续更新，本任务未修改它们；已把可行动问题直接交给实现任务。

## 1. 问题、处理与剩余风险

| 编号 | 源码位置 | 判断与动作 | 状态 |
| --- | --- | --- | --- |
| S01 | `csrc/op_kernel/oscar_vector.cpp`，`oscar_zero_kernel` | 初版只有负页号检查，没有上界，会允许越过分配末端。已要求传 `numblocks` 并在 device 侧拒绝 `block>=numblocks`；复读见 `115–116` 已修，同时清零完整 stride，包括页末 padding。 | 修正已静态确认；NPU 未测 |
| S02 | 同文件，`oscar_store_kernel`；`csrc/torch/bindings.cpp:55–62` | 初版只过滤负 slot，无法由 shape-only Host 验证 device slot 值。已增加 native block 数参数及 device slot 上界；History `LoadKvHalf` 亦增加页表宽度/slot 上界。 | 修正已静态确认；NPU 未测 |
| S03 | `csrc/op_kernel/oscar_common.h:50–65`，`ClipRow`；`bindings.cpp:22` | Bitonic 网络要求 D 为 2 的幂，否则 `other=i^j` 可超过 D。binding 当前只接收 D=64/128/256，覆盖了 kernel 算法前提。不要以后仅扩大 `kMaxDim` 就开放 D192。 | 当前约束有效 |
| S04 | `csrc/op_kernel/oscar_history_cv.cpp`，`LoadKvHalf` / `SoftmaxHalf` | 初版把 native kernel 的 128-token virtual block ID 当作 manager block ID。已加入 `table_block_size`，先还原线性 slot，再按 History manager block 大小寻址；softmax mask 同步过滤无效 slot。 | 修正已静态确认；跨页 NPU 用例未测 |
| S05 | `csrc/op_kernel/oscar_history_cv.cpp`，`WindowPosition` | 初版独立 Window kernel 只过滤负 row map，没有上界。当前 Window 改为共享 CV 路径，`WindowPosition` 已检查 `row<0 || row>=p.window_rows`；binding 传入真实窗口行数。 | 修正已静态确认；NPU 未测 |
| S06 | `csrc/torch/bindings.cpp:64–81,110–130`；各 device task loop | qsl 内容没有从 Host 回读，这是正确的热路径方向；但 device kernel 仍需要保证 `0<=qb<=qe<=N`、query tile 上界、row map 上界和错误状态的可靠传播。当前形状校验不证明这些运行时值成立。建议 device guard + error/status 输出，不增加每步 `.item()`。 | 调用方/设备契约与真机边界用例待验证 |
| S07 | `csrc/op_kernel/oscar_history_cv.cpp`，`mode==1` BF16 Window/current chunk 路径 | 初版 AIV 逐 query/KV pair dot-product 会在 16K prefill 产生大量 scalar/barrier 开销。提出后已改为共享 CV QK/PV tile 流水，新增 BF16 loader。并进一步将 raw KV tile 末尾裁到当前 query tile 的 causal 上界，避免计算整个未来 raw chunk。仍需实测 Cube 利用率、barrier 与端到端时延。 | 结构修正已静态确认；性能未验收 |
| S08 | `csrc/CMakeLists.txt:17–44`、`op_host/tiling.cpp`、`include/oscar_tiling.h` | 目标头文件、TCubeTiling host/device ABI、链接符号、混合核编译选项与 runtime SoC 都依赖真实 CANN。当前没有可凭 reference 直接确认的致命 API 拼写错误，但这不构成编译成功。 | 真机 build/probe 必须执行 |

S07 的建议已落实为 HistoryCV 的 16-query×32-KV QK/PV 流水和 BF16 Window/current-chunk loader；与 INT2 路径共享在线 softmax、partial 合并和输出路径。此修正没有把全部 History 恢复为完整 BF16 HBM 副本，也没有调用另一种 native Attention 来掩盖性能。代码结构的改进仍不等于完成“不慢于 native”的实测验收。

量化取整曾被列为待核实项。算子实现任务回报已核实指定 PR 使用 `(x + 0.5).to(int32)` 后 clamp，因此当前 half-up 路径不应误报为必须替换成 `torch.round` ties-even。本报告未用“常见量化习惯”覆盖指定 PR 语义。

## 2. API 对照中已取得的实际证据

以下只能证明已有同类调用方式，不能证明所有 shape/SoC/工具链组合合法：

| 本实现用法 | 固定 reference 中的例证 |
| --- | --- |
| `MatmulImpl.Init(&tiling, pipe)` | `references/vllm-ascend/csrc/moe/moe_grouped_matmul/op_kernel/moe_grouped_matmul.h:86`；`.../hamming_dist_top_k/op_kernel/hamming_dist_top_k_parallel.h:35–50` |
| 1C2V 中 AIV 用 `GetBlockIdx()/2` 配对 Cube | `.../hamming_dist_top_k/op_kernel/hamming_dist_top_k_parallel.h:54–59` |
| mode2 的 AIV→AIC / AIC→AIV set/wait | 同文件 `673–688`，调用位置 `140–151`；低编号 flag 也有实际使用 (`699–702`) |
| GM BF16 A/B、FP32 C 的 Matmul tiling | `.../attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_tiling.cpp:200–217` |
| `MultiCoreMatmulTiling` 默认构造与 `SetBufferSpace(l1,l0c,ub)` | 同文件 `159–187` |
| `PlatformAscendCManager::GetInstance()` 和 Core/Mem 查询 | `.../batch_matmul_transpose/op_host/common_tiling.h:60–73` |
| `ascendc_library(... SHARED ...)` 与 CANN 路径发现 | `references/vllm-ascend/CMakeLists.txt:41–52,74–79` |
| Torch/CANN 链接库 `torch_npu, ascendcl, tiling_api, register, platform` | 同文件 `163–181` |
| `GlobalTensor<bool>` | `.../moe/hamming_dist_top_k/op_kernel/hamming_dist_top_k_parallel.h:720` |

没有因为“代码看起来陌生”就判定这些用法必然编译失败。两个 Matmul 对象同时使用同一 TPipe 的资源消耗、各 tiling 的实际 L1/L0 需求和 typed scalar/DMA API 支持仍要由目标编译器与运行结果确认。

## 3. CV 同步、任务分配和读取复用

`HistoryCV::Process` 当前按 `(request, query_head, query_tile_16, history_split)` 分配工作，Cube 与其两个 AIV 使用相同 `core` 和 task stride。

每个 KV tile 的顺序为：

```text
AIV0/1 分别准备 16 个 KV rows，合成 32-row tile
  -> flag 1 -> Cube QK -> flag 2
AIV0/1 各处理 8 个 query rows 的在线 softmax
  -> flag 3 -> Cube PV -> flag 4
AIV0/1 累加各自 8-query 输出
完成 task 后 flag 5，Cube 才进入下一 task
```

在静态控制流上，两 AIV 对同一任务的空区间/跳过条件相同，非空 tile 上均有对应 set/wait；partial 地址按 `(token,query_head,split)` 分开。没有发现直接的“一个分支 return 而另一核仍等待同一 flag”的明显分歧。仍需目标执行确认 mode2 配对、MTE3/FIX 可见性、空 History、单 tile、多 tile、多个 task 重用，以及 q_len=1/4/16 的完成性。

每个 active KV token 在一个 `(query_head, query_tile, split)` 内只在 `LoadKvHalf` 读取一份压缩 slot，随后供 16 query rows 的 QK 和 PV 复用。故对 `q_len<=16` 的单个 query tile，**静态设计**中 INT2 History 读取不随 q_len 逐 query 倍增。该结论不是 HBM 带宽测量；GQA 的不同 query heads 仍重复加载相同 KV head，超过 16 queries 后按 query tile 增加读取。

必须测量区分：压缩 History HBM 读取、global-memory tile bridge 的读写（可能命中 L2）、Q/partial 流量、Cube 有效利用率和端到端时延。不得把“源码只有一个 LoadKvHalf”当作 HBM 已下降或服务已加速的证据。

## 4. UB 与 GM workspace 账目

按当前 `kMaxDim=256`，每个 AIV 的显式 TBuf 峰值：

```text
packed 512 + half 32
+ KV 2×16×256×2 = 16384
+ Q 8×256×2 = 4096
+ scores 8×32×4 = 1024
+ probability 8×32×2 = 512
+ accumulator 8×256×4 = 8192
+ PV 8×256×4 = 8192
+ scratch 512
= 39456 bytes
```

不含编译器栈/临时寄存器 spill、Matmul 内部资源、TPipe 实际对齐和运行时系统预留；不能只凭 39456 小于某个宣传 UB 容量就宣称不会 UB 耗尽。

GM tile bridge 对 D256 为 `60416 bytes / physical Cube core`；partial workspace 为：

```text
align512(N_query_tokens × H_query_local × splits × (D+1) × 4)
```

这不是完整 BF16 History 副本，tile bridge 随处理复用。但长 prefill 的 query/partial/scratch 开销仍可很大，必须进物理预算。当前 runtime 的 scratch reserve 已采用 `max(full-prefill splits=1, bounded MTP splits=8)`，避免把 16K prefill token 数与 8 splits 错误组合。

## 5. 外部 admission 修正已经实施

实际提前拒绝发生于 `references/vllm/vllm/v1/core/kv_cache_utils.py:2048–2057`，早于最终 `get_kv_cache_config_from_groups`。原 planner 本身只生成描述，允许 N=0，不能通过只修改其分配逻辑修复前置拒绝。

`oscar_ascend/cache_integration.py` 已新增 `minimum_cache_bytes` 并以同一可撤销 hook 集合包装 `_max_memory_usage_bytes_from_groups`：

```text
native_template = original_planner(config, native_groups, available_memory=0)
required_global_blocks = 1  # null block
  + sum(ceil(native_group.max_memory_usage_bytes(config)
             / native_group.page_size_bytes))
physical_required = CachePlan.physical_bytes(required_global_blocks)
```

原生 FULL/GDN/speculative 状态生命周期仍决定页数；FULL/GDN 实际分配、BF16 有界窗口、runtime scratch、rotation/metadata reserve 和 zeroer 的 `8×num_blocks` NPU 页号缓冲决定真实字节。`runtime_reserve_fn(vllm_config, groups)` 在 admission 与最终 planner 中共享同一接口。

原生 `_estimate_max_model_len_from_groups` / auto-fit 的二分搜索在每次模型长度试探中动态调用该 helper，所以使用同一物理标准。没有扩大 `available_memory`、跳过检查、修改原生文件或把 GDN spec 换成 FULL。`num_gpu_blocks_override` 的原生预算重写路径目前显式拒绝，避免混合两种计费。

已执行 16 项 Host cache unittest，包括“native 账面不够但真实拆池字节足够”的 admission 边界、下一块超预算、GDN/MTP 三组状态、null block、动态 runtime reserve、签名保留、幂等与全组 undo。测试没有模拟 KV 数值，也不替代 CANN/NPU 编译或数值测试。
