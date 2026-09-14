# 执行 Checklist

本表是唯一完成状态表。2026-09-14 开始执行；用户明确真机验证由用户在目标机执行。`[x]` 只用于有对应证据的完整验收项，代码实现与 NPU 验证分开。阶段 A/B 审计与阶段 C 代码已全部落盘（含 drafter/prefix/错误传播/自动校准）；除 A01 的上游逐字节核验外，其余各项的 NPU 验收均未运行。

| ID | 验收项 | 代码/审计状态 | NPU 验证 | 证据/未完成内容 |
|---|---|---|---|---|
| [x] A01 | 建立 reference 清单，锁定 OSCAR PR 与相关 vLLM/Ascend commit，确认未参考失败仓库。 | passed | n/a（纯参考核验） | 本地 HEAD 与文件 SHA256 已锁定；PR 快照 11 个 .py 与上游 commit 57286d5 逐字节核验全部 matched，见 `../reports/oscar_upstream_verification.json`（18:31 重试成功；此前 TLS 失败为暂态，未关闭证书验证） |
| [ ] A02 | 记录实际软件、NPU、CANN/编译环境、模型配置和原生源码状态。 | in_progress | not_run | 已采集本机环境；目标软件/NPU/模型config仍需用户采集 |
| [ ] A03 | 跑通原生目标配置的基线，保存命令、输入、版本、MTP/图模式与资源记录。 | in_progress | not_run | 见 reference_manifest.md 和三份 reference 审计；目标环境待用户采集 |
| [ ] A04 | 读懂第 3.3 节调用链，记录 FULL/GDN、TP、KV 分组、MTP 和图模式的真实入口。 | in_progress | not_run | 三份调用链审计 + drafter eager 多步循环（llm_base_proposer attn_update_stack_num_spec_norm）已写入 seam-map/审计文档；实际模型形状、安装版本仍待目标确认 |
| [ ] A05 | 核实第 4 节全部歧义，给出层/组/Tensor/地址映射与字节来源。 | in_progress | not_run | 已推导条件布局、MTP对GDN状态的影响；实际模型映射尚未实测 |
| [ ] A06 | 读取 PR 的配置、旋转、量化和精度验证方法，完善本 Checklist，列出仍缺失的输入。 | in_progress | not_run | PR/paper格式与clip差异、量化容差、staging(8192)/sink(64)/recent(256) 默认已核查；模型质量门槛未指定 |
| [ ] B01 | 产出 `docs/design.md`，列出外部新增文件/函数/类及其包装的原生入口。 | in_progress | not_run | design.md、operator_abi.md(ABI v2)、window_state_contract.md 及外部 hook 源码已写 |
| [ ] B02 | 完成请求窗口、物理页、GDN 隔离、TP 分片、页生命周期和 MTP 提交/回滚设计。 | in_progress | not_run | 窗口契约含 drafter 多步、prefix 恢复、抢占 epoch 语义；代码与测试已落 |
| [ ] B03 | 给出所有 payload、元数据、padding、workspace 和总 HBM 公式，证明压缩收益可落地。 | in_progress | not_run | cache_budget.py/cache_integration.py 含真实池拆分、staging 池预算与 admission；86 项 Host 缓存契约通过，实物 HBM 未测 |
| [ ] B04 | 完成算子清单、AscendC 融合划分、Decode Stage1 伪代码和逐阶段读写流程。 | in_progress | not_run | 算子清单（含 dequant/stage/restore）及 CV 源码已写；静态 review 见 reports/ascendc_static_review.md |
| [ ] B05 | 给出 decode 计算量、HBM 流量、MTP 跨 query 复用和 Host 调度预算。 | in_progress | not_run | 16-query共享tile/GM桥接/tiling缓存已写；GQA重复读和实际开销仍待测 |
| [ ] B06 | 固定功能、精度、性能测试方法和验收门槛，设计一键脚本及错误清理流程。 | in_progress | not_run | acceptance_matrix.json 与配对比较工具已写；全服务负载和质量门槛仍缺 |
| [ ] C01 | 独立插件/算子工程可安装、可编译、可加载，原生源码未修改，安装脚本不使用 `cp`。 | in_progress | not_run | 独立 pyproject/CMake(含 oscar_prefix.cpp)/torch 绑定 ABI v2 已写；本机无 CANN/cmake，未安装编译验证 |
| [ ] C02 | 版本能力 probe、延迟注册和幂等 Hook 通过 CLI、API server、各 worker 初始化验证。 | in_progress | not_run | 延迟 plugin、版本/接口检查、可撤销 hooks 及 runtime 自动挂接已写；CLI/worker 真机未测 |
| [ ] C03 | 实现并验证 INT2 编码、旋转/裁剪/量化、KV store 与 Recent→History 增量迁移。 | in_progress | not_run | AscendC 旋转/clip/INT2 store/两 phase 窗口迁移/staging 镜像已写；NPU 测试未运行 |
| [ ] C04 | 实现真实 fused INT2 CV Decode Stage1，并完成必要的分块/分段输出合并。 | in_progress | not_run | 真实 Cube QK/PV+Vector 解包 softmax 源码已写；编译/device/数值/性能均未验收 |
| [ ] C05 | Sink/Recent Attention、History Attention 和参考数学语义一致，数值稳定。 | in_progress | not_run | LSE 合并与正交变换依据已记录，CV window/raw 路径已写；NPU 数值未验收 |
| [ ] C06 | 实现原生页表兼容的物理分配与视图，GDN 不受影响，HBM 中无冗余完整历史副本。 | in_progress | not_run | 外部分配/视图（含 staging 池）/页清零/物理 admission 已写；GDN 地址与生命周期真机未验收 |
| [ ] C07 | 在确有需要的路径提供有界恢复能力，证明 decode/verify 不走全历史恢复。 | in_progress | not_run | `dequant_history_out` 有界行级恢复算子 + probe 黄金往返 + prefix 恢复的同款核内回退已实现；decode/verify 路径不调用它（pipeline 调用序列测试固定）；NPU 数值未验收 |
| [ ] D01 | 首次 prefill 正确，未引入不必要的重复压缩和恢复。 | pending | not_run | target FullAttentionPipeline 首次 prefill 源码与 NPU 测试入口已写，未运行 |
| [ ] D02 | chunked prefill 正确，已处理历史不随每个新 chunk 重搬运或重压缩。 | pending | not_run | 两 phase 增量 chunk 源码和迁移测试入口已写，未运行 |
| [ ] D03 | 普通 decode 持续进入压缩历史路径，无全历史 BF16 HBM 恢复。 | pending | not_run | 融合历史路径源码已写；模型真实 decode 未运行 |
| [ ] D04 | MTP draft/verify 与原生流程一致，完整接受、部分接受、完全拒绝后的缓存和 GDN 状态正确。 | in_progress | not_run | drafter eager 多步复用 pending/commit 窗口机（q_len=1 行、A<=old_P、拒绝位原位覆盖），draft capability 放行、drafter 图 capture fail-closed，契约测试已过；真机 drafter metadata 与窗口交互未验证 |
| [ ] D05 | MTP 多 query 复用历史 tile，读取量测量支持“不随 `q_len` 线性倍增”。 | pending | not_run | 单 16-query tile 共享一次压缩读取；q_len 流量测量未运行 |
| [ ] D06 | 混合长度和 prefill/decode/verify 混合调度正确，不依赖逐请求 Python 热循环。 | in_progress | not_run | refresh_rows 已改为 numpy 批量 join（契约测试固定），阶段元数据批处理已写；真实混批调度未运行 |
| [ ] D07 | 16K、32K、50K 长输入实际使用 OSCAR，跨页与跨窗口边界均正确。 | pending | not_run | 窗口/恢复边界 oracle 测试覆盖边界值；16K/32K/50K 实际路由未测 |
| [ ] D08 | TP=4、异步调度、目标图模式和 W8A8 权重量化共存，所有 rank 路由正确。 | pending | not_run | 保留目标配置且有固定 buffer 源代码；真实 capture/replay 与全模型图未完成 |
| [ ] D09 | 前缀缓存、页共享/复用、请求取消/结束、抢占恢复等目标环境可达生命周期通过验证。 | in_progress | not_run | v1 整块只读共享无 COW 字节复制（swap/copy fail-closed）；prefix 命中经 staging 精确恢复、INT2 回退计 lossy（NPU 测试已写）；抢占 recompute+epoch 递增已实现；真机生命周期未验证 |
| [ ] E01 | 核心算子对齐参考；覆盖 pack/unpack、量化边界、尾块和 attention 数值误差。 | pending | not_run | 真机 probe（含新 dequant/stage/restore 黄金用例）与 4 项 NPU 测试已写且本地跳过；不可计数值通过 |
| [ ] E02 | GDN 隔离检查通过；FULL 量化后的层输出、模型输出与约定质量指标达标。 | pending | not_run | 待补真机运行；质量门槛仍待确认 |
| [ ] E03 | MTP 接受率、接受长度、输出正确性和有效生成吞吐完成对照。 | pending | not_run | 待补真机运行 |
| [ ] E04 | 固定 token 数和固定 HBM 预算两种口径下证明真实缓存收益，计入全部新增开销。 | pending | not_run | staging/窗口/reserve 全部入 CachePlan 计账；真实显存收益未测 |
| [ ] E05 | profiler 证明无禁止的 CPU/AiCPU 数据处理、同步、冗余 KV 双写及全历史恢复。 | in_progress | not_run | 设备错误一步延迟 pinned 镜像观察已实现（无逐步同步）；profiler 证据未采集 |
| [ ] E06 | 按第 10 节逐工况比较原生性能；无未解释退化，不以平均值掩盖慢项。 | pending | not_run | 待补真机运行 |
| [ ] E07 | 对照第 8.2 节验证历史故障防护，保存实际覆盖结果而非仅声明“已避免”。 | pending | not_run | 待补真机运行 |
| [ ] F01 | 一条 Bash 命令完成安装、编译、所需校准、probe、资源清理和正式启动。 | pending | not_run | 入口可执行 host-check/build/probe/calibrate 及目标流程；readiness 仅剩图集成与全服务 TP4 probe 两项真机闸门 |
| [ ] F02 | 验证正常退出、失败退出和中断后的清理，重复执行不会遗留 worker 或加载旧产物。 | pending | not_run | 日志和 owned-process 管理已写；3 项实进程测试因权限未获批准跳过，NPU 释放未验证 |
| [ ] F03 | 正式服务按目标配置启动并完成真实请求，记录 OSCAR compressed decode/MTP 路由证据。 | pending | not_run | 待真机执行 |
| [ ] F04 | 交付代码、设计、完整 Checklist、测试/性能/显存报告、日志及 README 中的一键命令。 | pending | not_run | 已有代码/文档/静态报告；仍缺完整运行、精度、性能、显存报告 |
| [ ] F05 | 检查所有硬约束和未完成项，最终结论准确列出已验证范围、失败项及阻塞项。 | pending | not_run | 本轮准确列出未完成项；全任务未达到完成定义 |

## 本地子检查（不替代上表验收）

- [x] L01：完整读取启动指令；保留原文件。
- [x] L02：确认 vLLM 与 vLLM Ascend 的本地 git HEAD，tracked changes 为空；见 `../reports/reference_inventory.before.json`。
- [x] L03：记录本机环境；macOS arm64，没有 torch/torch_npu/CANN，见 `../reports/local_environment.json`。
- [x] L04：逐参数保存附录 A 到 `../configs/target_service.json`，`tests/test_service_config.py` 验证。
- [x] L05：当前可运行 Host 检查 86 项通过 79、跳过 7（4 NPU 窗口/prefix、3 实进程），不计通过；Python 编译/脚本语法通过。见 `../reports/host_checks/host-contracts.log` 与 `../reports/host_checks/status.json`。
- [ ] L06：AscendC 目标编译、算子 device 完成、TP4、capture、replay、精度、性能：全部未运行。
- [ ] L07：真机验收 D04/D09 的 drafter 多步与 prefix/抢占生命周期、图 workspace 地址与完整窗口 commit 链。

## 执行范围

本次只读工作区指定 references。已知旧记忆不作为本次算法/实现依据；没有读取失败项目代码。曾有 git 自动上溯祖先目录列出状态，已建立独立工作区仓库并在审计工具加入 git 顶层检查，祖先状态不作来源证据。

2026-09-14 用户指示先本地实现，Git 地址稍后提供；未添加 remote、未推送。

- [x] L08：实现后重新哈希对比全部参考文件，内容不变，见 `../reports/reference_unchanged.json`。
- [x] L09：断网中断后恢复执行——上游 PR 核验重试成功（A01 勾选）；补齐 drafter 窗口接入、prefix staging 恢复、设备错误异步传播、有界 dequant（C07）四个实现缺口并新增 14 项 Host 契约测试；readiness 闸门相应缩减为 2 项真机项。
