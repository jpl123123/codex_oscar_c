# OSCAR AscendC 算子清单

2026-09-14 初版及实现更新。来源/数学契约见 `oscar_reference_audit.md`。表内名称描述算子职责，已写的实际绑定以 `operator_abi.md` 和 `csrc/torch/bindings.cpp` 为准。History及BF16窗口/当前chunk均已写 mixed 1C2V Cube QK/PV；旋转/打包/两phase窗口迁移/页清零已有AscendC代码。编译、NPU运行、图捕获/回放、精度、性能均 **not_run**。当前代码仅接受D=64/128/256，实际模型和TP仍须核实。

符号：N=当前新token数，B=请求数，Hk=本rank KV heads，Hq=本rank Q heads，D=每head维度，T=历史token数，S/R=请求级窗口，J=history splits，Qb=复用一个KV tile的query行数，Tk=片上KV tile长度。INT2每head K/V slot `C=2*ceil(D/4)+8`（单量化组、偶数slot）；映射/窗口池采用最终allocator方案。

| 能力 / 拟定入口 | 输入 → 输出 shape/dtype/layout | 阶段 / 数据量 | 当前参考 / 融合归属 | 同步与验收 |
|---|---|---|---|---|
| 旋转+clip+INT2写 `oscar_rotate_clip_store` | raw K/V `[N,Hk,D]` BF16；Rk/Rv `[D,D]` fp32；迁移/页映射 → 持久INT2 uint8；FP16 scale/min内嵌 | 首次prefill待保留History、新完成chunk、Recent滑出；每token仅首次进入History时写 | PR `_rotate_clip` + `_store_int2_vec`；目标AscendC融合rotate/quantile/pack/scatter | 同一stream依赖；不得Host读KV。位序精确、clip线性插值、FP16舍入、负slot不写、sentinel安全 |
| INT2 pack/unpack | 单head D元素/ceil(D/4) bytes；q0在最低2bit | store与history tile加载 | 吸收在store和CV stage1内；独立debug入口可用于golden | 每byte单writer；尾部mask、metadata低高字节、constant/nearzero行 |
| BF16窗口写/迁移 `oscar_window_store` | raw K/V `[N,Hk,D]` BF16；每请求长度/页映射 → 精确Sink/Recent或History | append、chunk结束、verify临时写入；只处理新token及新增迁移token | PR prototype只有hash staging双写，不能照搬；目标融合写入+迁移调度 | 所有KV搬运NPU；状态可见性服从commit/reject；窗口跨页和页回收测试 |
| Q旋转 `oscar_query_rotate` | Q `[N,Hq,D]` BF16；Rk `[D,D]` fp32 → Qrot fp32或待误差确认dtype | decode/verify/历史prefill，O(NHqD²)计算 | PR `_decode_attention` matmul；可与CV stage1融合或保留有界Q工作区 | 不inverse整个K历史；与显式inverse-K数学参考对齐 |
| History Decode/Verify Stage1 `oscar_int2_attention_stage1_cv` | Qrot querytile `[Qb,D]`；packedHistory；原生block table、query可见长度 → mid `[N,Hq,J,D+1]` fp32 | decode、多query MTP、continuation历史，INT2流量至少THkC/请求/split覆盖；目标同tile跨query复用 | PR stage1数值参考只vector reduce，无CV；AscendC需Vector unpack+scale、Cube QK/PV、online softmax | 必须明确片上UB/L1/L0、Cube/Vector事件和tile生命周期；不写完整History BF16 HBM。真机计量history读字节 vs q_len |
| BF16 Sink/Recent及新chunk attention | Q/raw-KV + 精确窗口映射 → normalized O + LSE | 所有阶段；窗口O((S+R)HkD)，当前chunk按原生QKV | PR `_decode_attention_windowed:717-743`、首次prefill原生FlashAttention数学 | 新chunk因果mask、短序列去重、窗口精确S/R、不hash降级；有界临时tile |
| 跨split/跨段merge `oscar_attention_merge` | mid O/LSE fp32；RvT；BF16窗口O/LSE →最终O `[N,Hq,D]` BF16 | Stage2和输出逆旋转；中间量O(NHqJ(D+1)) | PR导入stage2、windowed logaddexp merge | 输出空间统一后LSE加权；空history/split=-inf、全部masked安全，GQA/混batch |
| 有界debug dequant+inverse `dequant_history_out` | packed rows（显式slot列表）→raw-space K/V `[T,Hk,D]`；slot≤0跳过 | 数值核对或明确需要的小tile，禁止逐步完整历史恢复；亦作prefix恢复的无INT2-staging回退核内同款数学 | PR full_dequant数学供参考，不能继承HBM完整分配 | debug路径不注册为decode/verify fallback；严格限workspace |
| MTP commit/rollback窗口状态 | 原生accept长度、slot/page映射和有界暂存 →窗口边界/可见状态 | accept/partial/reject后；只操作必要新token元数据/暂存 | PR无实现，依目标原生生命周期 | 不独立改变GDN；必要暂存不含历史副本；状态/页复用证据。drafter eager多步复用同一pending/commit机器（q_len=1行），已实现host放行+capture fail-closed，真机D04待验 |
| 请求页迁移、COW、free/reuse | 原生block生命周期 + OSCAR映射 →一致的INT2/BF16 ownership | prefix共享、抢占、恢复、结束 | v1整块只读共享、追加只进新块，无COW字节复制路径；staging owner tag对齐上游 | `stage_window_out`（owner-tag先行的BF16镜像）与 `prefix_restore_out`（命中行窗口重建，INT2回退计数lossy）已实现；预算入 `CachePlan.staging_bytes`；真机D09待验 |
| 校准统计/矩阵生成 | NPU Q/K/V →协方差统计→Rk/Rv `[layers,D,D]` | 独立校准阶段；避免CPU处理在线KV | paper compute_qqt/compute_sst/R·H·Pbr；脚本原CPU实现不可直接作生产fallback | 全FULL层coverage、D/模型/TP身份、orthogonality；只允许离线明确的统计/常量I/O |

## 持久内存与workspace计账

当前实际实现入口（ABI v2）：`rotate_out`、`store_int2`、`history_attention_out`、`window_attention_out`、`merge_out`、`window_state_out`、`zero_blocks_out`、`dequant_history_out`、`stage_window_out`、`prefix_restore_out`（另有校准专用6算子，见 `csrc/torch/calibration_bindings.cpp`）。见 `csrc/torch/bindings.cpp`。QK/PV固定tiling已由 `op_host/tiling.cpp` 按device/SoC/D及片上容量缓存，所有访问受mutex保护；新增请求/层只填写动态descriptor。首轮warmup产生有效缓存，GetTiling失败不缓存。staging深度按上游PR公式 `max(ceil(staging_tokens/block), sink_pages+tail_pages+2)`，默认8192 token可配置；staging是工作集上界而非无损保证，被逐出行回退INT2并计入每层lossy计数器，绝不冒充无损。

性能剩余风险：History/窗口CV目前使用每Cube固定GM tile桥接，并在query head之间重复读取GQA的同一KV；向量旋转及quantile排序也尚未优化到目标吞吐。raw chunk已使用Cube tile并裁掉未来KV tile，不再使用逐token-pair的AIV attention。上述代码改动不是性能通过证据。

```text
M_history = history_tokens * Hk * C
M_window = (sink_tokens + recent_tokens) * Hk * (Dk+Dv) * 2
M_rotation = 每层每rank实际常量张量字节（若存R与RT都计入）
M_mid = N * Hq * J * (D+1) * 4
M_query = N * Hq * D * sizeof(query_work_dtype)
M_total = 原生GDN实际分配 + FULL物理History/窗口池 + 映射/对齐
M_extra_peak = rotation + query + mid + CV workspace + MTP必要暂存 + graph缓冲
```

上述 token 公式为逻辑下界；最终物理预算须计量分配粒度、共享tensor、padding、页表，不能以压缩后padding证明节省。stage1 tile 不应产生与历史长度成比例的 BF16 HBM恢复buffer；CV实现若需要HBM中转须给有界固定大小与实际峰值，且不允许每query重载整历史。

## 统一验证要求

1. Host契约测试只能证明配置/尺寸/边界数学；不替代算子测试。
2. 与PR对齐：store/dequant atol/rtol=2e-3，decode vs 已反量化cache 5e-3，window/continuation relativeL2<2e-2；这些参考容差不是目标模型质量门槛。
3. 额外目标用例：D=256/实际Hq:Hk、S=64/R=256、各block size、空History、各短序列边界、非连续页和混长度、MTP多query接受/回滚、16K–50K真实OSCAR路径。
4. 本次无目标NPU证据；需分别记录 compile、launch+device synchronize、numerical、capture_return、replay_request、model_quality、capacity、end_to_end_perf，不得合并为单个“通过”。
