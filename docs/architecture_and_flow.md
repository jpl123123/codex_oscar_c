# OSCAR AscendC：算法理念、接入架构与完整数据流程

本文说明这个独立工程要完成什么、当前源码怎样接到 vLLM Ascend，以及哪些环节已经有实现、哪些仍缺完整集成或真机证据。**存在源码、生成了 .pt、单个算子通过，分别不等于服务可用、校准方法正确、graph replay 正确或性能不慢于原生。** 实际验收状态还须查阅 docs/checklist.md、oscar_ascend/readiness.py 和本次 logs/reports。

## 1. 核心理念与边界

OSCAR 将 FULL Attention 的历史 K/V 放在更适合 INT2 的旋转坐标系中，以少量 scale/min 加每元素 2 bit 持久存储。请求最初的 Sink 和最近的 Recent 保持原始 BF16；GDN 的 conv/SSM 状态、计算和原生推测更新保持原有语义。

这是两个相互独立的问题：

- 数值：真实模型 Q/K/V 统计决定旋转，尽量降低注意力敏感方向上的量化误差。Q 的分布决定 K 误差怎样影响分数，K 与 Q 的关系又决定 V 的加权协方差。
- 系统：真实减少底层分配；读历史时在 NPU 上解包并与 QK/PV 融合，避免每步恢复全部 BF16 History。INT2 裸 payload 减少不自动带来整个混合模型的 8 倍容量或端到端加速。

适配全部位于独立工程；不修改 references 或安装环境原生 vLLM/vLLM Ascend 文件，不复制覆盖源码。Host 可处理配置、shape、批量页号和请求身份；在线 Q/K/V 数值不回传 CPU，没有 CPU fallback。

## 2. 来源与“准确算法”

| 来源 | 身份 | 用途 |
| --- | --- | --- |
| references/vllm | HEAD 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665 | KV specs、groups、全局 BlockPool、speculative 生命周期 |
| references/vllm-ascend | HEAD 19e436985102f4ed3aad36c137a6481653688a6c | NPU runner、物理布局、Attention/GDN、图与原生算子 |
| references/oscar-vllm-pr46774 | 指定 PR 快照；核验见 reports/oscar_upstream_verification.json | INT2 store/decode、旋转域注意力 |
| references/oscar-paper/rotation/compute_kv_rotation.py | SHA256 f7f9f738be5dea75cc1d7e8d6928e4ee91df1cbf1e1c7b7d9c7d308b55d4da8b | QQT/SST、eigendecomposition、Hadamard、bit reversal 准确顺序 |
| references/oscar-paper/README.md | SHA256 4d49f99762efb2acb03ecb0298a417bd2029656c941c2566369c7826d2f7df8f | 推荐 qqt_sst + r_h_pbr 配方及校准流程 |

paper 快照没有独立 Git，不能把父目录 Git 当作其 commit。PROVENANCE 声明与本次文件内容 hash 分开记录。

准确方法要求保持 QQT/SST 的统计对象、全 token/全 head 合并、权重归一化及 U @ H @ P 顺序；不能用随机矩阵、固定 Hadamard、K-only PCA、普通 V covariance、各 rank 独立旋转替代。语料和浮点 eigensolver 也是实验条件：相同方法不意味着相同原始样本或逐 bit 相同矩阵。

## 3. 总体流程

~~~mermaid
flowchart TD
    A[外部安装与 AscendC 构建] --> B[目标环境和接口检查]
    B --> C[匹配 rotation artifact 身份]
    C -->|有效匹配| F[复用已验证 pt]
    C -->|缺失或不匹配| D[原生模型校准: 紧凑 NPU 统计]
    D --> E[全 token / 全 head QQT 与 SST]
    E --> P[NPU eigensolver: U H P 和验证]
    P --> F
    F --> G[确认校准与 probe worker/NPU 资源释放]
    G --> H[原生 KV spec / groups / global block IDs]
    H --> I[真实物理预算与分配]
    I --> J[GDN 原生状态池与计算]
    I --> K[FULL INT2 History 与 BF16 窗口]
    K --> L[设备提交和窗口迁移]
    L --> M[History CV 与 Window/Chunk CV]
    M --> N[LSE 合并和输出逆旋转]
~~~

图描述完整交付链。节点有代码不代表真机验收完成，状态区分见第 13 节。

## 4. vLLM Ascend 具体接入符号

| 原生位置 | 外部实现 | 职责 |
| --- | --- | --- |
| vllm.general_plugins | plugin.register | 每个启用进程注册；顶层不初始化 NPU/DeviceOperator |
| NPUPlatform.get_attn_backend_cls | plugin.install_backend_route | 标准 FULL 路由 OscarAttentionBackend；保留 GDN backend |
| NPUModelRunner.__init__ | runtime.install_runtime_hooks | 附加每 runner 的 TargetRuntime |
| NPUModelRunner.get_kv_cache_spec | 保留原生采集结果 | 使用实际层/head/GDN state，而非硬编码层数 |
| kv_cache_utils._max_memory_usage_bytes_from_groups | cache_integration.minimum_cache_bytes / admission wrapper | 原生前置 admission 和 auto-fit 使用真实物理字节 |
| kv_cache_utils.get_kv_cache_config_from_groups | replan_kv_cache_config | 保留 groups/spec/global 页号，重算真实物理描述和容量 |
| NPUModelRunner._allocate_kv_cache_tensors / _reshape_kv_cache_tensors | allocate_cache_views | GDN 原生状态视图与 FULL FullCacheView |
| NPUModelRunner.initialize_kv_cache | runtime/cache wrappers | 准备常量、固定 buffer，再经 native bind 绑定各层 |
| NPUModelRunner._init_kv_zero_meta | PackedBlockZeroer | native 新页号批量触发 AscendC packed FULL 页清零 |
| NPUModelRunner._update_states / _prepare_inputs | TargetRuntime.update_lifecycle / refresh_rows | 请求生命周期及稳定窗口行映射 |
| Attention metadata build / forward | OscarMetadataBuilder / OscarAttentionImpl | native device 长度/页表/slot mapping 进入固定地址 FULL 管线 |

bind_kv_cache 只把新对象绑定给层；连接器、swap/copy、其他 Attention 类型并不会自动兼容，须逐条实现或拒绝。wrapper 保存原签名、幂等状态及可撤销 handle；不能吞掉加载/兼容错误后偷偷继续原生 FULL。

## 5. .pt 的准确 QQT/SST 定义

对应 paper compute_kv_rotation.py:93–136。每层规范为 Q[T,Hq,D]、K/V[T,Hkv,D]，G=Hq/Hkv。所有保留校准 chunk 先在 token 维串联，总数为 T。第 h 个 KV head 对应的 G 个 Query heads 合成 Q_h[T*G,D]。

### K：QQT

~~~
M_h = Q_h.T @ Q_h / (T*G)
Sigma_K = sum_h M_h / Hkv
~~~

这是不减均值的二阶矩，不能改成 centered covariance，分母也不能错写为 T。chunk 长度不等时不能“每 chunk 求平均后再平均”；必须累计 outer-product 总和与真实行数。

### V：SST

~~~
a_h[t] = K[t,h] @ M_h @ K[t,h].T
z_h = max(sum_t a_h[t], 1e-12)
w_h[t] = a_h[t] / z_h * T
Sigma_V = (1/Hkv) * sum_h V_h.T @ diag(w_h) @ V_h / T
        = (1/Hkv) * sum_h [sum_t a_h[t] * outer(V[t,h]) / z_h]
~~~

先形成每 head 的全 token M_h，再累计该 head 所有 K/V 的加权分子、分母，分别归一化各 head，最后对 head 平均。**不能逐 chunk 归一化后平均，也不能先合并不同 head 的权重分母。** 这里没有 softmax，也没有完整 attention probability 或额外 causal 权重；替换为 softmax 权重是另一算法。

### 两遍流式计算与 TP=4

1. 第一遍：各 rank 在 NPU 累计本地 KV head 所属 Query 组的 outer products/计数，得到全 token M_h。
2. 第二遍：重放相同真实 token 和模型配置，累计每 head 的 sum(a*outer(V))、sum(a)。
3. V 先逐 head 归一化，再 TP 全局归约；K 对全部唯一 KV heads 的 M_h 平均。每层最终只有一对全局 Sigma_K/Sigma_V。
4. 同一层各 rank 使用同一对旋转。artifact 的 rank 字段仅描述运行时覆盖与名字映射，不能悄悄变成 rank-local 拟合。
5. Hkv<TP 时需按 global head 身份去重复制 KV，并先合并被分散的 Q 组；未实现该几何时明确拒绝。两遍 token 集不一致也必须失败。

这些统计可以用紧凑累加器计算，不需要把完整离线 Q/K/V 保存在 Host。数值精度和计算位置须遵守本项目 NPU 边界。

## 6. Covariance → eigenvectors → Hadamard → permutation

paper compute_kv_rotation.py:23–46,75–80,234–265 的顺序为：

~~~
Sigma = (Sigma + Sigma.T) / 2
lambda, U = eigh(Sigma)          # U 的列是 eigenvectors
H_1 = [1]
H_2n = [[H_n,H_n],[H_n,-H_n]] / sqrt(2)
sorted_idx = argsort(lambda, descending=True)
perm[bit_reverse(i)] = sorted_idx[i]
P = I[:, perm]
R_loaded = U @ H_D @ P
~~~

P 按列选取；排序用 lambda 数值而非 abs(lambda)。不能先排序 U 再重复用同一 P，也不能改为 U @ P @ H。compute_rotation.sh 显式选择 qqt_sst/r_h_pbr；Python CLI composition 默认却是 plain，自动生成不能依赖该默认。

paper 读入激活后采用 FP64 统计和 torch.linalg.eigh，最终保存 FP32 contiguous [D,D]。本项目 NPU FP32/Jacobi 仍必须保持相同算法顺序，并记录 solver、精度、收敛规则及误差：不能声称逐 bit 复现 FP64。eigenvector 符号/重根子空间本来不唯一，但 U diag(sign) @ H @ P 可能改变具体量化误差，固定符号规则也须记录。

验证至少包括：covariance golden、对称性/有限值、U.T@U、eigenpair residual、R.T@R、AscendC rotation roundtrip。最大 sweep 到期却没收敛须失败；矩阵验证不替代模型精度和 MTP acceptance。

## 7. 语料、warmup 和 Sink/Recent

paper load_tensor('all') 在 compute_kv_rotation.py:49–72 按数字文件名拼接，仅排除 chunk 0；注释明确它是旧 dump schedule 的六 token warmup。脚本没有读取 Sink/Recent，也不按窗口裁剪 Q/K/V。当前快照缺少旧 sglang-dump-qkv 的实际 dump hook，不能假定未见的过滤行为。

本项目应通过 calibration-active/phase 标志排除 engine profile、capture dummy 和 warmup，不能机械跳过第一次真实 native prefill。没有来源时不新增 history-only、去 Sink、去 Recent 策略后声称样本选择完全相同。

参考 producer rotation/_eval_runner/dump_gpqa_prompts.py:48–91 使用 GPQA diamond、198 条、seed 0、随机选项排列；保存脚本默认 30000 token 预算，只生成一个 token 触发 prefill。32 线程到达/预算截断顺序不一定稳定。自动生成固定本地 corpus、请求顺序、chat template/tokenizer、最终 token ID/mask，并记录 hash。同一拟合方法与不同校准样本必须明确区分。

MTP FULL 层可能不会被 max_tokens=1 的主体 prefill 激活。必须通过真实 native drafter 路径采其有效 Q/K/V，记录每层实际计数与过滤策略；不得用主体末层 R、identity 或其他模型 R 填补未采样的 MTP 层。

## 8. 自动 .pt 路径与生命周期

控制入口为 oscar_ascend.prepare_rotations.ensure_rotations；身份计算、产物验证分别由 make_identity、validate_artifact 负责。默认校准配置为 configs/calibration.json，calibration_data.py 固定参考 GPQA diamond prompt recipe、seed 0、198 条、30000 token 预算与 CSV SHA。

最终产物路径：

~~~
artifacts/rotations/<digest(identity)>/rotations.pt
~~~

需要生成时调用独立校准子进程：

~~~
python -m oscar_ascend.calibrate --request <临时目录>/request.json --output <临时目录>/rotations.pt
~~~

用户不需要提供手工 .pt 路径。完整顺序为：

1. 模型配置、权重/量化身份、HF/RoPE overrides、tokenizer/corpus、TP/head 几何、算法与来源形成 identity。
2. 查找 identity 对应产物并验证 manifest、FULL/MTP 覆盖和矩阵；不能“取最新文件”当匹配。
3. 缺失或不匹配则运行原生 TP4/MTP3/W8A8 模型的两遍 NPU 校准，保持真实 native Q/K/V 来源。
4. 仅在收敛、数值和身份验证通过后原子发布 pt/provenance；失败不能留下可被正常命中的产物。
5. 确认校准及 worker/NPU 资源释放，再把确定路径写入正式 worker 配置。
6. 正式 worker 用 torch.load(weights_only=True) 读取常量，仅上传本 rank 所需矩阵并做 NPU roundtrip。

具体入口存在不等于已在本机执行自动校准；当前没有本次真机生成的模型匹配 pt。schema 详见 docs/rotation_artifact_schema.md，K objective 是 qqt_r_h_pbr，V 是 sst_r_h_pbr。clip ratio 独立记录：eigensolver 没有自动选 clip，不能把其他模型的 0.96/0.92 说成本模型已校准参数。

## 9. 真实物理页和容量

native group 表示共享 block table 的层集合，physical tensor 则被不同 group 的相同 ordinal 层共享。各 group 从同一 BlockPool 获取互斥活页号，不能把相同 b 当成每组独立命名空间。

外部保留所有权与页号，拆为：

- GDN 真正 conv+SSM 状态池，保留原始 shape/dtype/连续视图，移除 FULL V padding。
- FULL UINT8 History[N,B,Hkv,C]，使用 native manager 页号。
- 每 FULL 层独立 BF16[max_requests,S+R+M+1,Hkv,D]，含有界 pending；没有 BF16 History 副本。

每 head 一组，slot 为 Kbits,Kscale,Kmin,Vbits,Vscale,Vmin；D256 为 64+2+2+64+2+2=136 字节，不是 160。

~~~
M(N) = N * (sum(distinct physical pool page bytes) + 8)
       + all FULL BF16 windows + runtime_reserve
~~~

8 bytes/global page 是 zeroer 持久 NPU 页号缓冲。reserve 包括旋转、query/partial/scratch、位置/状态、固定 metadata、图缓冲和分配对齐。每个独立分配只计一次，N 仍包括 native null block。

native backend 可把 manager B 拆成 128-token 虚拟页。kernel 先算 slot=BT[pos//128]*128+pos%128，再算 physical_page=slot//B、offset=slot%B。直接用虚拟页号索引 [N,B,...] 会错。

801792 只适用于原稿特定状态形状，MTP 会扩展 conv width；实际数值必须来自运行时。详见 docs/cache_integration_audit.md。

## 10. 每次 FULL forward 的三阶段设备流程

窗口状态是 {epoch,committed_end,pending_count,error}，与稳定请求身份绑定；可重排 batch row 通过 row_to_window 找窗口。

**Prepare。** 用 native 异步修正后的 device seq_lens/query_start_loc 得到已计算前缀 C，要求 old_C<=C<=old_C+old_pending。先 gather 即将滑出 Recent 的原始 BF16 值并 INT2 store，再移动接受的 pending、丢弃拒绝后缀。bonus token 未执行 forward 时尚无 KV，不能提前提交。

**Attention。** History 读 Prepare 确定的压缩区间；Window 读 committed Sink/Recent 加当前 raw K/V，对每个 query 的绝对位置做 causal mask。不能在 Attention 前按最终 chunk 长度覆盖旧 Recent，否则本 chunk 前面 query 所见精度发生变化。

**Finish。** Prefill/chunked prefill 的 prompt token 确定有效，只量化首次进入 History 的当前 token 与滑出的旧 Recent，保留最终 Sink/Recent。Target decode/verify 先写有界 pending，下轮 Prepare 再按 native 可见长度提交/回滚。q_len>1 不代表 prefill；drafter 多步 metadata 不能直接套用 target 公式。

不重新压缩已有 History、不建立整段 MTP shadow cache。prefix/抢占恢复如果缺 BF16 窗口，不能从 INT2 假装无损恢复。详细状态见 docs/window_state_contract.md。

## 11. Query 旋转、CV 和合并

行向量定义 Krot=K Rk、Vrot=V Rv。Query 只需 Qrot=Q Rk；History 输出累计完成后一次 Oh_raw=Oh_rot Rv.T。该正交/线性等价关系避免每步逆旋转整个历史，对齐的是量化参考，不意味着 INT2/clip 无损。

History Stage1 每任务含 request、query head、最多 16 query rows 与一个 history split。两 AIV 各解包 16 个 KV rows，供同一 Cube 做 QK；AIV 做 mask/online softmax，再由 Cube 做 PV。16 query rows 复用同一 tile，MTP 的 4-query verify 不展开成四遍全 History。不同 GQA query heads 仍可能重复读取同一 KV head，HBM 流量须实测。

BF16 Window/current chunk 使用相同 CV 框架的 BF16 loader，并跳过当前 query tile 之后的未来 raw KV tile。每 core GM bridge 是与 History 总长度无关的固定 tile；其 L2/HBM/同步开销必须进入对照测量。

每区域输出局部归一化 O 和 logsumexp L，按 alpha_i=exp(L_i-max L)/sum exp(L_j-max L) 合并。空区域输出 0、L=-inf；全部为空单独处理。History 先映回原空间，再与 BF16 Window 输出合并。

## 12. 图、错误和部署验收

capture 前分配 cache/workspace/位置/长度/row map/epoch，每次只改内容、不换地址。dummy capture 使用独立或无效请求身份，不能推进真实窗口。native FULL_DECODE_ONLY 与 drafter eager 作用域不同，不能全局切 eager 换取启动成功。

分别记录编译、device completion、capture 返回、真实请求 replay、精度、容量、MTP acceptance、性能。HTTP 200 不证明 INT2 生效；长上下文和不支持功能不得静默 fallback。

一键入口 scripts/install_probe_serve.sh → oscar_ascend.deploy.main 保存配置/环境/安装/编译/算子与窗口 probe 日志；build-only/probe-only 有独立状态。正式服务前还需完整集成 probe 和本任务 worker/NPU 释放确认。

## 13. 已有代码与缺口

| 环节 | 源码覆盖 | 尚不能据此宣称 |
| --- | --- | --- |
| spec/group/预算/admission/分配 | cache_budget.py、cache_integration.py，真实物理描述与 global IDs | 未完成目标容量/GDN 拆池生命周期实测 |
| FULL store/CV/Window/merge | csrc/op_kernel（vector/history_cv/window/prefix）、Torch binding ABI v2、ops.FullAttentionPipeline | 未证明 CANN 编译、数值、同步、图和性能通过；staging/prefix 恢复数学见 window_state_contract 第 5 节 |
| runtime/固定 metadata | backend.py、runtime.py、metadata.py，真实物理描述与 global IDs | drafter eager 多步窗口接入、prefix staging 恢复与设备错误一步延迟传播已实现；capability 不等于真机 drafter/prefix/图 replay 验收（D04/D08/D09 未运行） |
| pt 身份与加载 | prepare_rotations、runtime.validate_rotation_metadata、roundtrip 检查 | 仍须本次自动校准 provenance、eigen residual、FULL/MTP 覆盖和模型精度 |
| 自动 pt 生成 | 本轮落实原生模型两遍统计、TP global head mean、NPU U H P、原子发布 | 当前没有目标 NPU 上生成并验收的 pt，不能把 Host 测试算作校准完成 |
| 数学契约 | tests/test_calibration_math_contract.py，8 个小规模有理数/矩阵 oracle 通过 | 只验证归一化/合并/P/H 顺序，不验证生产 eigensolver |
| 一键部署骨架 | deploy.py、processes.py、service_config.py | readiness 中仍未闭环项不得被单算子 probe 掩盖 |

本文件是算法与工程流程说明。最终运行验收必须来自目标机器的新报告；源码数量、Host pass 数或旧日志都不是完成定义。
