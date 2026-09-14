# 执行、MTP、图模式与外部加载审计

审计日期：2026-09-14。证据仅来自本工作区 `references/vllm`、`references/vllm-ascend` 的静态阅读；commit、快照状态以 `reference_manifest.md` 为准。未启动 NPU、未执行图捕获或回放、未证明精度或性能。本文不把历史日志当成当前验证结果。

## 1. 必须保留的实际配置

目标是 target 模型 `FULL_DECODE_ONLY` 与 draft 模型 `speculative_config.enforce_eager=true` 同时存在。这两个设置不冲突，不能把它们转换成全局 eager：

| 证据 | 实际作用 |
| --- | --- |
| `vllm/config/compilation.py:53–69` | `FULL_DECODE_ONLY=(FULL,NONE)`；decode 使用 FULL，混合执行模式使用 NONE。它不是所有请求均使用图的保证。 |
| `vllm_ascend/worker/model_runner_v1.py:666–671` | `_use_aclgraph()` 检查 target `model_config.enforce_eager`，以及 compilation mode 和 graph mode。 |
| `vllm_ascend/spec_decode/llm_base_proposer.py:215` | `self.use_cuda_graph = self.runner._use_aclgraph() and not self.speculative_config.enforce_eager`；speculative eager 只关闭 drafter 图。 |
| `vllm/v1/worker/gpu_model_runner.py:814` | `uniform_decode_query_len = 1 + num_spec_tokens`。MTP3 的 target verify 完整请求是 4 个 query。 |
| `vllm_ascend/worker/model_runner_v1.py:2933–2986` | `_determine_batch_execution_and_padding` 根据实际最大 scheduled 长度、总 token 数、request 数和 uniform 条件选择图。 |
| `vllm/v1/cudagraph_dispatcher.py:193–235,274–328` | 只为支持的 descriptor 建 key；超出捕获范围或没有匹配 key 返回 NONE。 |

注意：NPU runner `2950–2952` 注释说 prompt fully computed，但代码仅检查 `num_computed_tokens_cpu > 0`。这不能证明请求已完成 prompt。OSCAR 阶段识别必须读 `is_prefilling` / prompt 边界，不把 graph descriptor 或 q_len=1 单独当作 decode。

`qwen3_5_mtp` 配置在 `vllm/config/speculative.py:554` 归一为 `mtp`；`vllm_ascend/spec_decode/__init__.py:34–54` 将 `mtp` 交给 `AscendEagleProposer`（Step3.5 特例另行处理）。不能仅在运行时匹配字符串 `qwen3_5_mtp`。

## 2. 每请求 q_len、阶段与批元数据

实际 `q_len[i] = query_start_loc[i+1] - query_start_loc[i]`；`actual_seq_lengths_q` 是累计结束位置，不是逐请求长度。混合 batch 不得使用 `tokens // batch` 或全局 `decode_token_per_req` 代替差分。

关键调用链：

1. `Scheduler.schedule`（`vllm/v1/core/sched/scheduler.py:511–535`）按 token budget 裁剪 scheduled draft tokens，可能少于配置的 3 个。
2. `NPUModelRunner._prepare_inputs`（`model_runner_v1.py:1067–1096,1296–1345`）构造 query 起止、spec metadata、每请求 draft 数；chunked prefill 在 `num_decode_draft_tokens` 中用 `-1` 掩码。
3. `_calc_spec_decode_metadata`（同文件 `1525–1608`）按 `num_draft_tokens+1` 选择 target/bonus logits；`num_sampled_tokens` 不等于任意 prefill 的 q_len。
4. `_build_attention_metadata`（同文件 `3118–3189`）逐 KV group 读取原生 block table 与 slot mapping；补齐 token 的 slot 是 `-1`。`is_prefilling = computed < prompt`。
5. `AscendAttentionMetadataBuilder.build`（`attention/attention_v1.py:276–318`）保存累计 q 边界、KV 长度、block table 和 slot；无需新建逐请求 KV 数值处理 Python 循环。

`_build_attn_state`（`model_runner_v1.py:1477–1506`）是批级粗分类，`SpecDecoding`、`ChunkedPrefill` 均可能包含多 query。OSCAR 应给 kernel 同时传每请求 q 边界、绝对已计算位置、是否 prefill、真实 slot 与有效行掩码；kernel 按每个 query 的绝对位置做因果遮罩。

异步模式下 `seq_lens` NPU tensor 才是权威值（同文件 `3155–3175`）；`_seq_lens_cpu` 是 optimistic 上界。用它立即推进 committed window 会把被拒绝 token 当作已经提交。原生在 `_prepare_inputs`（`1162–1263`）结合 `prev_num_draft_tokens`、先前输出及 request 行重排修正 GPU 长度；插件必须在这之后消费实际长度，并保持 GPU 更新与当前 stream 顺序。

## 3. MTP draft → verify → acceptance → 可见状态

| 阶段 | 调用者与符号 | OSCAR 必须保持的语义 |
| --- | --- | --- |
| target prefill/verify | `NPUModelRunner.execute_model` → `_prepare_inputs` / model forward；`model_runner_v1.py:2080–2321` | 先提交已有确认 token 的迁移，再写本轮候选；verify 是最多 4 query 的因果注意力，不是独立无历史 prefill。 |
| sample/reject | `sample_tokens` → `_sample` → `AscendRejectionSampler`；同文件 `2457–2458,2616–2642` | 保留原生 rejection sampler，不能用 argmax 或 fixed acceptance 代替。 |
| draft 输入修正 | `propose_draft_token_ids` → drafter `prepare_inputs_padded`；`llm_base_proposer.py:1893–1984` | 被拒绝 token 仍作为 padding 保留，`num_rejected_tokens_gpu = num_draft+1-valid_sampled_count`；`token_indices_to_sample` 选择最后有效输出。 |
| draft 首轮 | `set_inputs_first_pass`；同文件 `1337–1399` | target tokens 移位，最后一个有效输入替换成新 sample，使用 target hidden states；不能把所有 padded rows 写成有效缓存。 |
| 后续 draft | 同文件 `1201–1335` | 原生按 `num_speculative_tokens-1` 迭代并递增位置。draft 与 target 的层、cache 身份必须分开。 |
| GDN postprocess | runner `_update_states_after_model_execute`；`729–811` | `(output!=-1).sum(dim=1)` 包含 bonus token；保持原生 align/all 后处理与 accepted event。 |
| scheduler rollback | `Scheduler.update_from_output`；`scheduler.py:1414–1438` | accepted draft=`len(generated)-1`；rejected=draft-accepted；扣减 computed 和 async placeholders。 |
| async prefix commit | `AsyncScheduler._update_request_with_output`；`async_scheduler.py:43–66` | 原生完成 output 更新后缓存 `computed - placeholders`，排除未确认 token。 |

FULL 原生回滚主要是可见长度回退，并不要求清空所有尾部 KV 字节。OSCAR 在不可逆 INT2 窗口迁移前必须同样限制到已确认边界。若按 verify 的 optimistic L 将 Recent 最老 token 提前量化并覆盖，拒绝后该 token 可能再次处在 Recent BF16 区，已经丢失原值；这会破坏原生语义。

可实施方案是独立的有界 pending 尾部：容量按最大本轮 query 数/原生 lookahead 计算，只存本轮及尚未提交的 K/V；accepted count 到达后 NPU kernel 合并有效前缀、迁移真正滚出的 Recent、丢弃 rejected 后缀。不得为每轮复制完整 History，不得把整段 History 解压到 HBM。需要检验 bonus token 的 KV 可用性：sampled bonus 是下一次输入，当前 target verify 中尚未计算它的 KV，不能按 sampled count 直接搬移不存在的 KV。

## 4. GDN 保留边界

`_build_attention_metadata` 在 `model_runner_v1.py:3212–3218` 仅对 `GDNAttentionMetadataBuilder` 额外提供 accepted counts 与 decode draft masks。`ops/gdn_attn_builder.py:425–454,539` 构造专门的 speculative conv1d/SSM metadata；`ops/gdn.py:167–208,320–354,385–444` 分别运行 speculative、decode、prefill 状态更新与合并。

外部插件不能全局替换 `DeviceOperator.reshape_and_cache`、GDN builder、GDN `forward` 或 mamba postprocess。只对经模型层名/Attention 类型验证的 FULL `Attention` 选择独立 backend/spec。KV group 的共享关系与物理 allocator 必须另行证明，不以“FULL 后端没调用 GDN”证明地址不会重叠。

## 5. 请求、页与 prefix 生命周期

| 入口 | 当前原生行为 | 外部窗口的约束 |
| --- | --- | --- |
| `GPUModelRunner._update_states:1135–1149` | finished 清理 request/batch；同一输出内 finished 与新请求允许同 ID | 窗口身份需带 generation；先释放旧身份，再接纳新身份。 |
| 同文件 `1160–1180` | unscheduled 只移出 batch，保留 cached request state | 不得把暂未调度当成 finished 释放；不得用 batch 行号当永久身份。 |
| 同文件 `1368–1379` | 普通追加新 block；preemption resume 替换全部 block IDs | resumed 不能把新页当旧页延续，需重建/恢复窗口有效区。 |
| `KVCacheManager.allocate_slots:425–434` | prefix cache 只提交 finalized tokens | pending spec 页不可提前共享。 |
| `KVCacheManager.free:438–446` | coordinator 释放 request 引用 | request 结束不等于共享 prefix 页可覆盖；页 refcount/复用仍属原生管理器。 |
| runner `3118–3126` | 每 group 取真实 block table，padding slot=-1 | kernel 必须跳过 invalid slot；元数据应绑定同 group。 |

Prefix 边界是重要未完成设计项：同一 token 在长请求中可能已属于 INT2 History，在新命中较短前缀的请求中却属于 BF16 Recent。单独的 request window 池若在 request 结束后释放，就不能从有损 History 恢复原始 BF16 Recent。必须明确 prefix 复用格式、窗口保存/重建、可共享区及内存预算；不能静默反量化冒充 BF16 原值，也不能直接关闭用户要求的 prefix 支持并宣称功能完整。

## 6. 图捕获与回放接入

`NPUModelRunner._check_and_update_cudagraph_mode`（`5038–5083`）取所有 attention builder 支持能力的最小值并初始化 descriptor。外部 backend 不得声明 `ALWAYS` 作为绕过验证的方式。

`ACLGraphWrapper.__call__`（`compilation/acl_graph.py:138–258`）：

- `138–150` 区分 NONE / 当前 wrapper runtime mode。
- `158–170` 创建 `NPUGraph` 并记录 tensor 输入地址。
- `190–211` 在 `torch.npu.graph` 上下文中执行真正模型及算子。
- `222–232` 捕获返回之后才保存 entry、增加 capture 计数。
- `234–240` 调试模式校验地址。
- `243–258` 真正 `.replay()`；当前某些 FULL 路径在 replay 前同步当前 stream。

OSCAR 固定分配必须覆盖 cache/pending、query 元数据、accepted/epoch、task table 与 workspace。捕获使用的 scalar 长度不能在 Python closure 中永久冻结；变长长度放在固定地址 device buffers，由 NPU kernel读取。图前更新与 graph stream 依赖必须显式保持。动态 `torch.empty`、重绑 metadata tensor、新建逐请求对象进入 captured kernel 参数都需要审计。不能把“capture count 增加”当作当前真实请求已回放，更不能把 HTTP 200 当成 INT2 路径成功。

跨 query 复用应由 fused kernel 在一次 History tile 装载内同时计算同 request 的多个 query，保持各 query 独立 causal mask 与 softmax 统计。仅 launch 4 个单 query kernel 不满足 H12；实际 HBM bytes、kernel 次数与 q_len=1/2/3/4 的变化须 profiler 验证。

## 7. 外部插件加载与循环依赖

可用入口已经核实：

- `setup.py:543–552` 注册 `vllm.platform_plugins` 与 `vllm.general_plugins`。
- `vllm/plugins/__init__.py:54–64` 对 entrypoint `plugin.load()` 导入异常只记录日志并继续；因此插件模块本身必须足够轻，不可依赖此处异常实现 fail-closed。
- 同文件 `69–82` 执行已加载 register callable；真正必需 capability 检查放在 callable / worker 初始化处，异常向上传递。
- `vllm/v1/engine/core.py:107–109` 与 `vllm/v1/worker/worker_base.py:245–253` 各进程加载 general plugins；worker 先 load plugins 再 resolve worker class。
- `vllm/platforms/__init__.py:262–278` 是 lazy current_platform 初始化，提前访问可能重入 platform resolve。
- `NPUPlatform.pre_register_and_update` 是 `@classmethod`，签名 `(cls, parser=None)`（`platform.py:183–202`），包含权重量化注册。不能用普通函数替换破坏绑定。
- `NPUWorker.__init__:107–120` 原生应用 worker patch、注册原生 ops；`init_device:505–520` 建立 device 后创建 runner。

`DeviceOperator` 在 `device/device_op.py:2120` 完成整模块导入后才赋值；该模块顶层先导入 Triton/GDN、`quant_type`、utils（`24–36`）。当前 reference 的 `quantization/__init__.py:17–47` 已使用 lazy imports 明确防循环。因此旧日志不能证明当前 reference 仍有同一个环，也不能捕获 ImportError 后跳过插件。新插件的顶层只应导入标准库与本项目纯配置；在原生 platform 完成加载后的 register callable/worker阶段延迟导入真实 backend。不要在包 `__init__` 导入 torch、vllm、torch_npu、DeviceOperator。

候选受控 hook：包装 `NPUPlatform.get_attn_backend_cls`（`platform.py:814–841`），调用原函数后仅将标准 FULL 返回路径映射到外部 backend；配合 `Attention.get_kv_cache_spec`（`attention.py:566–610`）只替换外部 FULL 实例的 spec，并由独立 runner allocator 处理真实物理容量。所有 wrappers 保留 descriptor、signature、原始方法、幂等标记及撤销；worker probe 检查实际 backend、物理 dtype/shape、op ABI 和全 rank 路径。只有路由而未实现 allocator/算子不能作为启动成功。

## 8. 验证矩阵与当前状态

| 测试组 | 必须覆盖 | 当前状态 |
| --- | --- | --- |
| host 契约 | import 无 torch/vllm；classmethod绑定；幂等/撤销；启用但缺 kernel 不得继续；混合 q_len；finished/new 同ID；batch重排；pending accepted 0/1/2/3；无提前window迁移 | 静态审计完成；测试随实现补齐 |
| 首次/chunked prefill | 1、63/64/65、319/320/321，跨页，16K/32K/50K；chunk大小1/4/128/16384；已处理History只迁移一次 | 未运行 |
| MTP | draft3；全拒、逐位置部分接受、全接受；bonus未计算KV；short/long混合；MTP q_len1/2/3/4；请求结束在draft中；异步重排 | 未运行 |
| prefix/生命周期 | prefix命中停在Sink/History/Recent边界；两个长度不同请求共享页；结束/取消/抢占/恢复/同ID重用；invalid slot；页复用 | 未运行；prefix窗口格式仍需实现证明 |
| 图 | capture进入、capture返回、首次真实replay、连续replay、同descriptor不同长度/页表/请求身份；图与eager切换；target FULL且draft eager | 未运行 |
| GDN | 相同输入/接受结果下conv/SSM与原生逐阶段一致；物理地址不重叠；TP4全rank | 未运行 |
| 精度/性能/HBM | 配对baseline；输出质量、MTP接受率；固定token物理字节、固定预算容量；端到端速度；q_len扩展的History读取量 | 未运行 |

NPU 验证由用户在目标服务器执行。上述未运行项不会阻止继续编写外部实现、构建脚本及可验证测试，但交付报告必须保留未验证状态。

## 9. 本轮已落地的实现与明确缺口

`oscar_ascend/plugin.py` 提供轻量入口、原生接口检查、classmethod 绑定保留、幂等与完整撤销；`runtime.py` 在原生 runner 初始化后自动挂接 target runtime，按实际 cache groups 预算和分配固定 metadata/共享 workspace/逐层状态，保留原生 GDN 与 rejection sampler。`metadata.py` 的身份表不把 batch 行号当 request ID，也不把未调度当结束；`ops.py` 按 `window_state_contract.md` 执行完整 target prepare/attention/finish 两阶段 AscendC 调用。

实际 block table 的粒度由 `runner.input_batch.block_table[group_id].block_size` 读取，常见值128；物理 History 的 manager 页可能是768，两者不能混用。算子先由 virtual table 得到线性 token slot，再按 History 页大小索引物理池。

FULL 的原生绑定已复核：`vllm/v1/worker/utils.py:516–518` 直接把 dict 的 value 赋给 `attn_layer.kv_cache`，`layers/attention/attention.py:682` 原样读出。因此当前 reference 会把 `FullCacheView` 直接交给外部 impl，不会自动包成 `[FullCacheView]`。兼容其他版本需要重新验证，不能盲目多解一层。

当前 host 契约测试15项通过，仅说明导入/绑定/元数据规则和调用顺序。三项 `test_npu_window_pipeline.py` 已编写但本机全部跳过：短窗口 pending/reject、History解码 mixed golden、接受/Recent迁移/长chunk flush。数值门槛为指定 PR window 测试的 relative-L2<0.02，测试会输出实测值；未运行不计通过。

当前 NPU 实现、完整 MTP drafter、prefix BF16恢复、生产设备错误传播、真实 graph capture/replay、自动旋转校准、模型精度与性能仍未验收；服务 gate 必须保留。`RuntimeCapabilities` 的 true 描述代码存在的路径，不是 NPU 通过标志。图参数更新通过固定地址 metadata，不创建 FIA task-group handles；这仍需目标机验证。
