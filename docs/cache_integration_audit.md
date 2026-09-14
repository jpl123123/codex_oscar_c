# FULL/GDN 缓存接入审计

本报告依据当前允许读取的两个 reference checkout；未访问其他本地项目，未修改 reference。本文的运行时布局由源码推导，未把历史日志当作本次真机结果。新增 `oscar_ascend/cache_budget.py` 和 `cache_integration.py` 已实现 Host 物理容量规划及 NPU 分配/视图适配；服务的完整 KV 生命周期、算子接通和真机验收仍未完成。

## 1. 固定来源与适用边界

已通过各 checkout 的 `git rev-parse --show-toplevel` 确认它们有独立 Git 根目录，未误用工作区父目录 Git。`git diff --stat HEAD` 对两个 reference 均无输出。

| 来源 | 当前固定 commit | 说明 |
| --- | --- | --- |
| `references/vllm` | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` | reference provenance 对应 v0.23.0；本报告引用的是该 checkout 实际内容 |
| `references/vllm-ascend` | `19e436985102f4ed3aad36c137a6481653688a6c` | provenance 描述为 v0.23.0 + PR12607 快照，不能冒称真机安装版本就是这个 commit |

下文 `V/` 表示 `references/vllm/`，`A/` 表示 `references/vllm-ascend/`。来源包括实际文件/函数和一基行号，可按以上固定 commit 重查。两份 reference 的 AGENTS.md 已读；只在独立工程实施适配，启动指令 H01–H15 为任务约束。

## 2. 已核实调用链

| 阶段 | 文件 / 函数 / 行号 | 已核实行为 |
| --- | --- | --- |
| FULL spec | `V/vllm/model_executor/layers/attention/attention.py:566`，`Attention.get_kv_cache_spec`；`V/vllm/v1/kv_cache_interface.py:159–200,294–314` | 字节由 block size、本地 KV heads、K/V head size、dtype 和 padding 推导；并非固定 256 维 |
| GDN spec | `V/vllm/model_executor/layers/mamba/abstract.py:44–59`，`MambaBase.get_kv_cache_spec`；`V/vllm/v1/kv_cache_interface.py:608–636` | 一个状态页包含 conv/SSM 形状和 dtype；`num_speculative_blocks` 是独立生命周期预算，不能误解成逐 token KV |
| Ascend 混合页配置 | `A/vllm_ascend/patch/platform/patch_mamba_config.py:31–117`，`verify_and_update_config` | 用 kernel block size 128 对齐单份 K 页和 SSM 页，然后 padded page = FULL K+V 页 + conv 页 |
| Runner 规格采集 | `A/vllm_ascend/worker/model_runner_v1.py:4907–5036`，`NPUModelRunner.get_kv_cache_spec` | 先处理 Attention，后处理 Mamba；FULL 页不足 GDN padded page 时，扩大 FULL 的 `page_size_padded` |
| 分组 | `V/vllm/v1/core/kv_cache_utils.py:1074–1193,1629–1684`，`_get_kv_cache_groups_uniform_page_size` / `get_kv_cache_groups` | 按相同 spec 分类，按 `layers[i::num_groups]` 分组；general path 要求统一页字节 |
| 物理池描述 | `V/vllm/v1/core/kv_cache_utils.py:1247–1332`，`get_kv_cache_config_from_groups` | 每组中相同 ordinal 的层共享一个 `KVCacheTensor`；共有 `max(group.layer_count)` 个池 |
| 原始内存分配 | `A/vllm_ascend/worker/model_runner_v1.py:4080–4132`，`_allocate_kv_cache_tensors` | 混合 FULL/Mamba 为每个 tensor 描述实际分配一个 uint8 等价的 int8 raw buffer，各 `shared_by` 层获得同一对象 |
| FULL 视图 | `A/vllm_ascend/worker/model_runner_v1.py:4493–4502,4559–4577,4637–4642`，`_reshape_kv_cache_tensors` | 在 raw buffer 后部取连续 K/V 条；前部为 conv 尺寸的 padding；KV kernel block 可能拆分 native manager block |
| GDN 视图 | `A/vllm_ascend/worker/model_runner_v1.py:4681–4714`，同上 | 依次取 `N × prod(shape) × dtype_bytes` 的连续 conv/SSM 条，保留原生 tensor shape |
| 全局页所有权 | `V/vllm/v1/core/kv_cache_coordinator.py:90–116`；`V/vllm/v1/core/block_pool.py:149–177,333–363` | 各 group manager 共享同一个 BlockPool；存活的新页从同一 free queue 获取，id 0 留作 null block |
| 视图绑定 | `A/vllm_ascend/worker/model_runner_v1.py:3915–3951`；`V/vllm/v1/worker/utils.py:462–518` | 分配→reshape→cross-layer sharing→bind 到每个 layer 的 `kv_cache` 与 runner 列表 |
| TP 容量一致化 | `V/vllm/v1/core/kv_cache_utils.py:2072–2085` | 取各 rank 最小 num_blocks，并线性缩小每个 `KVCacheTensor.size` |

## 3. 四组、十六池与真实共享地址

`KVCacheGroupSpec` 的定义（`V/vllm/v1/kv_cache_interface.py:839–849`）是“共享 block table 的层集合”，不是连续 16 层小模型；`KVCacheTensor.shared_by`（同文件 `830–836`）则表示共享物理分配。二者不能互换。

若且仅若实际主体有 64 层、按 `[GDN,GDN,GDN,FULL]` 重复且没有额外 cache 层，则有以下映射（0 基层号）：

| group | 层号 | 共享 block table | 第 i 个物理池的层 |
| --- | --- | --- | --- |
| FULL | `4i+3, i=0..15` | `BT_FULL` | `FULL(4i+3)` |
| GDN 0 | `4i, i=0..15` | `BT_G0` | `GDN(4i)` |
| GDN 1 | `4i+1, i=0..15` | `BT_G1` | `GDN(4i+1)` |
| GDN 2 | `4i+2, i=0..15` | `BT_G2` | `GDN(4i+2)` |

这表示四个 group、每组十六层、十六个物理池；每个物理池被来自四组的各一层共享。group 的实际整数编号取决于 runtime spec 插入顺序，不能硬编码。当前 runner 会先插入 Attention 再插入 Mamba。

**共享 B 是实际 storage alias，不是仅共享模板**。若把同一个全局页号 b 强行交给 FULL 与某个 GDN，它们会写同一 B 页。正常生命周期靠共同 BlockPool 的活页所有权隔离，不是靠 group ID 另加一个内存偏移。多个 group 的 b 值不能当作彼此独立的命名空间。前缀缓存 hash 带 group ID，并不能允许任意跨组同时复用同一个活页。

若 MTP 增加一个 FULL 缓存层，FULL 层数可能变为 17，GDN 三组仍各 16 层，physical pool 数将为 17；最后一池可能只有 MTP FULL。其他 drafter/cache-only 结构另有分组路径。必须记录运行时完整的 layer→group→pool，而不是写死 16。

## 4. 801792 的条件推导与 MTP 歧义

按原稿线索，`E_K=E_V=256` 且 BF16，每 token 单份 K 或 V 为 `256×2=512` 字节。若 SSM 为 393216 字节，Ascend 的配置公式得到：

```text
T = 128 × ceil(393216 / (128 × 512)) = 768 tokens / manager block
K_page = V_page = 768 × 512 = 393216 bytes
P = conv_page + K_page + V_page
  = 15360 + 393216 + 393216 = 801792 bytes
```

其中 **768 是 block 内 token 数，512 是每 token 的单份 K/V 字节**；不是 head_dim，也不是 hidden_size。

设 raw 起始地址为 `base`、pool 页数 N、原生页号 b、页内 token t，在上述已满足的条件下：

```text
A_start = base
B_start = base + N × conv_page
C_start = base + N × (conv_page + K_page)
K(b,t)  = B_start + b × K_page + t × 512
V(b,t)  = C_start + b × V_page + t × 512
conv(b) = A_start + b × prod(conv_shape) × sizeof(conv_dtype)
ssm(b)  = B_start + b × prod(ssm_shape) × sizeof(ssm_dtype)
```

GDN 的后续地址必须按其多维 state shape/stride 计算；不能套用 `t×512` 的 FULL token 公式。上述 FULL 寻址是 manager block 形式，native kernel block table 会进一步拆分。`A/vllm_ascend/worker/block_table.py:79–83,108–117` 将页号变为 `b*(B/128)+subblock`，而 `model_runner_v1.py:4750–4760` 从 backend 选择 kernel block size。外部 kernel 必须显式区分二者：`slot=BT[position/128]*128+position%128`，然后 `physical_page=slot/B`、`inpage=slot%B`；不能直接用 virtual block ID 索引 `[N,B,...]` 的物理 History。若 backend 禁用拆分，须同时核实 metadata builder 和 drafter 使用一致 block size。

Qwen3.5 状态形状来源为 `V/vllm/model_executor/models/qwen3_5.py:688–707` → `V/vllm/model_executor/layers/mamba/mamba_utils.py:213–234`，其计算为：

```text
conv_dim_local = (2 × linear_num_key_heads × linear_key_head_dim
                  + linear_num_value_heads × linear_value_head_dim) / TP
conv_width = linear_conv_kernel_dim - 1 + num_speculative_tokens
conv_shape = SD: (conv_width, conv_dim_local)
             DS: (conv_dim_local, conv_width)
ssm_shape = (linear_num_value_heads / TP,
             linear_value_head_dim, linear_key_head_dim)
```

SD/DS 的选择来源同文件 `26–43`；dtype 来源 `89–96,108–116`，正式命令 conv/SSM 都指定 BF16。**MTP 不仅增加 speculative state block，conv_width 也增加 num_speculative_tokens。** 若原 A=15360 对应 `(3,2560)` BF16、SSM 为 `(12,128,128)` BF16，则正式 `num_speculative_tokens=3` 下 A 为 30720，P 变为 **817152**。这是条件算术，并非已经取得目标模型 config 的证明。

FULL TP 分片实际来源为 `V/vllm/model_executor/models/qwen3_next.py:218–235`：`num_kv_heads_local=max(1,total_num_kv_heads/TP)`（有整除/复制校验），head_dim 从配置取得。Qwen3.5 在 `qwen3_5.py:137–151` 明确按层类型选择 GDN 或 Qwen3NextAttention。当前 workspace 没有用户实际模型 config；hidden size、层数和全局 head 数仍待真机 probe 提供。

## 5. 外部实际分配方案与字节预算

实现保留原生 spec、分组、页号、block table 以及 GDN state shape/dtype；在相同 `shared_by` 图上拆开 backing allocations：

1. 同一原池内的 GDN 层共用一个连续 state 池，仅存真实 conv+SSM 条，移除原 V padding。
2. 同一原池内的 FULL 层共用一个 packed History 池，以 native 全局 block ID 索引；不分配 BF16 History。
3. 每个 FULL 层拥有独立、有界 BF16 Sink/Recent/暂存窗口，容量 `S+R+num_spec+1`；不同 FULL 层不错误共享请求窗口内容。BF16 的 request slot 生命周期须由 NPU metadata/commit kernel 管理。
4. `KVCacheTensor.size` 描述真实 physical pool 字节；`num_blocks` 按拆分后的总斜率和窗口/工作区固定预算计算。固定开销不随 TP 最小块数线性缩小。

设每 head 一组、K/V 均 D=256、FP16 scale 与 min；slot 按 `Kbits,Kscale,Kmin,Vbits,Vscale,Vmin` 排列：

```text
Kbits + Vbits = 64 + 64 = 128 bytes / token / head
K(scale,min) + V(scale,min) = 2×(2+2) = 8 bytes
slot = 136 bytes
```

160 字节并非 INT2 的必然结果。若每 head 分多组则需要另一份已核实的算法与 ABI；当前固定为每 head 一组，`group_size < head_dim` 在配置/规划阶段立即拒绝。物理页尾按显式 alignment 对齐；不能无来源补 32 字节。

通用计费（每 rank）：

```text
p_full = align_up(T × H_local × slot_bytes, alignment)
p_gdn  = sum(prod(state_shape_j) × dtype_bytes_j)
P_total = sum(各真正独立 GDN/FULL pool 的 p) + 8
          # 每个全局block预留一个zeroer持久NPU int64页号
W = sum(每个 FULL layer 的 max_num_seqs × (S+R+num_spec+1)
        × 2 × H_local × (D_K+D_V))
F = W + runtime_reserve_bytes
M(N) = N × P_total + F
N_max = floor((HBM_cache_budget - F) / P_total)
usable global blocks = N_max - 1   # 原生 null block 占一个
```

`runtime_reserve_bytes` 必须包括尚未由 vLLM 其他预算计入的旋转常量、固定 metadata、算子 workspace、图缓冲等峰值，不能直接把默认 0 当作真机完整预算。PyTorch 分配器粒度、碎片和 allocator reserved-memory 差额也需在 probe 对照记录。

以原稿每池状态数值为纯算术例子，拆分后每池每全局页 KV 实存为 `408576 + 768×136 = 513024`（每 rank 另计 zeroer 页号 8 字节/global page），原来为 801792；有真实页字节差额，但仍须扣除 W/F 才能比较固定 N 下的实际收益。正式 MTP 形状应重新计算，不能直接套用该百分比。

`cache_integration.install_cache_hooks` 在独立工程内替换 planner、前置物理 admission、runner allocate/reshape、initialize 与 zero-meta，保留原签名、幂等检测和可撤销 handle。`bind_kv_cache` 只保存对象，不强制 Tensor；native connector/Hamming 分支不能处理新容器，因此在分配前拒绝。默认 `PackedBlockZeroer` 去重 FULL raw pools，将 native scheduler 的页号批量传入 AscendC `zero_blocks_out` 清除完整 packed 页；Host tensor 只含调度页号。GDN state 不由新增 zeroer 改写，保留原生初始化/更新路径。若算子未加载则明确报错，避免将非 Tensor 容器传入原生 BF16 zeroer。分配仅允许 `npu` device，torch/vLLM 采用延迟 import。实现没有在线 CPU KV 数据路径。

## 6. 不可直接采用的方案及未接通项

- **只改 FULL spec dtype/INT2 payload 或保留 P padding**：runner 把 FULL pad 回 GDN 大小，native general allocator 仍按 P 分配；没有物理释放。
- **只改 raw view，保留原始大池**：视图缩小不释放底层 storage；不满足容量目标。
- **给每页重复 Sink/Recent**：破坏请求级窗口预算；额外 BF16 token 多算且窗口语义不对。
- **分离 FULL 池但保留原 GDN 完整 P 池**：继续保留原 C 条，必须在真实预算中承认；本实现移除它。
- **仍调用 native FULL cache store、attention、swap/copy**：这些接口认定 BF16 K/V tuple；不能把 packed UINT8 或 FullCacheView 硬塞进去。新 FULL backend 必须接管全部相关路径。
- **静默禁用前缀缓存来假装完整**：短前缀请求的 Recent 若已在另一个长请求中压为 INT2，不能无损重建 BF16 原值。须有跨请求窗口/共享语义设计，未覆盖时显式拒绝相应能力。
- **忽略 zero/reuse**：`A/vllm_ascend/worker/utils.py:76–142` 的 `AscendKVBlockZeroer.init_meta` 只接收 FULL 两份 K/V，并跳过 Mamba。拆池后原共享 B 的清零副作用发生变化，不能用“筛选 GDN”代替（那会直接没有 segment）。`A/vllm_ascend/ops/gdn.py:404–405` 有 prefill 初态清理，但不足以证明所有页复用路径。当前新增 PackedBlockZeroer 对 FULL packed 页执行 AscendC 清零；GDN 保留其原生路径。SSM 在拆池后的页复用回归仍须真机验证，request window epoch 是独立的未完成生命周期条件。
- **只替换最后一层 planner**：native `kv_cache_utils.py:2048–2057` 的前置 admission 原本依据未压缩 spec/page 大小，会提前拒绝压缩可容纳的 max_model_len。当前已外部包装 `_max_memory_usage_bytes_from_groups`，沿用 native group 的 `max_memory_usage_bytes` 推导生命周期所需页数，加入 null block 后以同一 CachePlan 计算真实物理需求；原生 auto-fit 二分搜索动态调用该 helper，也得到相同预算。`runtime_reserve_fn(vllm_config, groups)` 与 planner 共用，固定窗口/实际runtime scratch/8字节页号均计入。没有跳过安全检查或人为膨胀available_memory。`num_gpu_blocks_override` 会先改写预算，当前明确拒绝该未核实路径。
- **把 host 测试当作 NPU 验收**：Host 只测 shape/预算/页地址区间，无编译、数值、graph 或性能结论。

## 7. 证据状态与下一步

已执行：`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m unittest discover -s tests -p 'test_cache_budget.py' -v`，16 项通过。覆盖 INT2 metadata 字节、对齐开销、共享池去重、MTP conv 扩展、真实预算最大容量/null block、TP 最小容量缩减、每 FULL 层独立窗口、未知 spec/重复层拒绝、CPU 分配拒绝、全局页区间、只含shape的地址oracle核对绝对storage offset/共享视图及安全撤销hook、packed zeroer去重/范围/纯页号元数据协议。

未运行：真实 NPU allocation/view、native 分组/spec runtime dump、实际模型 config、分配器真实字节与容量增量、GDN 回归、prefix/reuse/preemption 生命周期、MTP commit/rollback、完整 prefill/decode/chunked/16K–50K、graph capture/replay、数值精度和原生端到端性能对照。当前代码可供本地继续完善和真机 probe，不能报告为完整适配完成。
