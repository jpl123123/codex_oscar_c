# 2026-09-14 本地执行状态（断网恢复后更新）

**整体任务未完成，当前不能交付正式可用的一键服务。** 用户指定先本地实现，之后自行在 Ascend 目标机测试；本次没有连接真机、没有推送远端。所有“实现完成”均指源码+Host 契约测试完成；NPU 编译与数值验收全部未运行。

## 断网前后状态

断网前最后阶段（18:21–18:33）在上游 PR 核验与自动校准实现中中断。恢复后核对发现：

1. **上游 PR 逐字节核验实际已成功**（18:31 重试，11 个 .py 全部 matched，commit 57286d5），此前状态文档中的 TLS 失败记录已过时；A01 据证据勾选。
2. 断网前 18:32–33 的最后修改（oscar_calibration.cpp、calibration_worker.py、prepare_rotations.py）经重跑测试自洽。
3. `TargetRuntime.refresh_rows` 早已是 numpy 批量 join，状态文档中的“每 step Python 请求遍历”缺口已不存在。

## 本次补齐的实现缺口

| 缺口（断网前 readiness 列出） | 本次实现 | 测试 |
|---|---|---|
| 原生 MTP drafter 多步窗口接入 | eager drafter 复用 pending/commit 窗口机；`draft` capability 放行；drafter 图 capture 显式 fail-closed | test_prefix_draft_error_contracts（4 项） |
| prefix 命中无损 Sink/Recent 重建 | `stage_window_out`（owner-tag 先行 BF16 镜像，上游 PR 同款协议）+ `prefix_restore_out`（命中行重建；tag 逐出行回退有界 INT2 并计每层 lossy，不冒充无损）；staging 池入 CachePlan 预算，默认 8192 token 可配置 | Host 契约 8 项 + NPU 测试 test_prefix_hit_restores_window_exactly_then_counts_lossy（待真机） |
| 生产设备错误传播 | `observe_window_errors`：错误列异步 copy 到 pinned 镜像、一步延迟判定（同 stream 先行，无热路径同步），报告 layer/window_row/request/code；`check_window_errors` 供 probe/关停同步检查 | Host 契约 3 项 |
| C07 有界恢复 | `dequant_history_out` 算子（显式行数上界，与 history attention 同款解包+逆旋转数学） | probe 黄金往返 + NPU 测试（待真机） |
| readiness 其余两项（图集成、全服务 TP4 probe） | 保留为真机闸门，不因本地实现移除 | — |

AscendC ABI 升至 v2（新增 3 算子，不动既有 kernel 签名）；CMake 增加 `op_kernel/oscar_prefix.cpp`。

## 实际验证结果

| 检查 | 本次结果 | 证据 |
|---|---|---|
| Host unittest | 86 collected；79 passed；7 skipped（4 NPU、3 实进程权限） | `host_checks/host-contracts.log`、`host_checks/status.json` |
| Python 源码编译 | 通过 | `python3 -m compileall -q oscar_ascend tools tests benchmarks` |
| Bash 入口语法 | 通过 | `bash -n scripts/install_probe_serve.sh` |
| 参考树完整性 | 前后文件 hash 一致 | `reference_unchanged.json` |
| OSCAR PR 快照上游核验 | **matched**（11 文件逐字节） | `oscar_upstream_verification.json` |
| 当前默认入口 | 按预期在 readiness 闸门非零退出（余图集成+全服务 probe 两项）；没有启动服务 | 入口输出 |
| AscendC 接口结构/代码审查 | 已执行，不能代表 CANN 编译 | `ascendc_static_review.md`、`csrc/README.md` |
| CANN 编译 | **not_run**；本机没有 cmake/CANN | `local_environment.json` |
| NPU 算子/device 完成 | **not_run**；NPU 测试本地跳过（窗口管线 3 项 + prefix 恢复 1 项 + probe 3 算子用例），probe 未运行 | `host_checks/host-contracts.log` |
| 实进程清理测试 | 3 项跳过；执行所需 ps 权限未获批准 | `host_checks/host-contracts.log` |
| 全模型 graph capture/replay | **not_run** | 无本次真机证据 |
| 模型正确性/MTP/精度/性能/HBM | **not_run** | 无本次真机证据 |

## 仍需真机验证（用户在目标机执行）

1. `--probe-only`：ABI v2 编译 + 含 dequant/stage/restore 的算子 probe + 窗口管线/prefix 恢复 NPU 测试。
2. `--calibrate-only`：自动两遍 NPU 校准生成模型匹配 rotations.pt（断网前已实现，未跑）。
3. drafter eager 多步与窗口 commit 界的实际 metadata 一致性（D04）；若违反 `C<=old_C+old_P`，设备错误码 1 会显式暴露，不静默放宽。
4. prefix 命中 staging 精确恢复与 lossy 回退计数（D09）、抢占 recompute 恢复。
5. 图 capture/replay、全服务 TP4 probe、NPU 释放后正式启动（readiness 最后两项）。
6. 16K/32K/50K 路由、精度/质量门槛、配对性能、真实显存收益。

`readiness.py` 仍会阻止将图集成与全服务 probe 两项作为已解决而启动正式服务。`--probe-only` 用于独立算子验证，不能作为完整任务验收。
