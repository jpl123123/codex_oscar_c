# OSCAR AscendC 外部适配

按 `oscar_ascend_agent_start.md` 开始实现的独立工程。只读参考位于开发工作区 `references/`；不修改或覆盖原生 vLLM / vLLM Ascend 文件，脚本不使用 `cp`。

**当前仍是开发阶段，不能作为已完成的一键服务交付。** 已有物理缓存规划、外部接入代码、独立 AscendC 算子（含 INT2 store/CV attention、有界 dequant、staging/prefix 恢复、校准统计）、本地/真机测试入口；原生 MTP drafter 窗口接入、prefix 命中窗口恢复、设备错误异步传播、Host 热循环消除与自动 NPU 校准已在本仓库实现并有 Host 契约测试，但**均未在真机验证**。图捕获集成、全服务 TP4 probe 与 NPU 资源释放验证仍未完成，正式入口在缺失能力时明确失败，不以原生BF16回退或HTTP200冒充OSCAR成功。完整状态见 [Checklist](docs/checklist.md)。

## 本地验证

```bash
bash scripts/install_probe_serve.sh --host-check
```

只验证纯Host配置/元数据/物理预算/接口契约。NPU计算没有CPU fallback。本机没有CANN、torch_npu，尚未编译或运行这些AscendC源码。需要真实进程组权限的清理测试默认跳过，可在目标机用 `OSCAR_RUN_PROCESS_TESTS=1` 显式运行；本次该权限请求未获批准，不能声称清理已经实测。

## 目标机独立算子构建与probe

在用户原生环境已经安装匹配 torch/torch_npu/vLLM/Ascend/CANN 的前提下，使用实际CANN路径和SoC型号：

```bash
export ASCEND_HOME_PATH=/实际/CANN/toolkit/路径
export OSCAR_SOC_VERSION=实际编译目标型号
bash scripts/install_probe_serve.sh --probe-only
```

入口安装本独立包、CMake构建，并在物理4–7卡分别运行小规模算子probe。SoC必须来自目标机；当前内核针对910B/910_93混合1C2V架构，其他架构不冒充已支持。四卡独立probe不等于模型TP4验收。构建产物为 `build/ascendc/liboscar_ascend_ops.so`；无需复制进原生工程。

也可以仅编译：

```bash
bash scripts/install_probe_serve.sh --build-only
```

构建成功后，单卡的独立算子图测试入口为：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 python3 -m oscar_ascend.probe \
  --library build/ascendc/liboscar_ascend_ops.so \
  --output reports/npu_operator_probe --capture
```

这会在逻辑NPU0（物理4）验证打包、非连续页、D=64/128/256、多query Attention与primitive图capture/replay。数值比对中的CPU读取仅存在于明确的测试oracle，生产KV路径使用AscendC。窗口事务的目标测试见 `tests/test_npu_window_pipeline.py`，按该文件的显式环境开关运行。

## 正式目标与日志

附录A完整参数保存在 `configs/target_service.json`，测试逐参数比较原文：端口8989、TP4、MTP3、异步调度、FULL_DECODE_ONLY、W8A8权重量化、BF16 GDN和262144长度上限均保留。设备固定4–7，防止继承外部环境误用其他卡。

默认 `bash scripts/install_probe_serve.sh` 是完整交付流程的入口；**当前在完整集成检查处失败退出，不会启动正式服务**。`oscar_ascend/readiness.py` 列出实现缺口，不能通过改环境变量绕过。代码中存在某个op或Python桥接类不表示该缺口已经解决。

每次入口运行输出 `logs/<UTC时间>/status.json` 及逐阶段完整日志。失败、超时、未释放子进程均停止后续阶段；本任务只处理自己创建的进程组，不重置设备或终止其他任务。进程退出与NPU内存释放是不同检查，后者仍需完整集成实现和实测。

## 代码与证据

- [Reference清单](docs/reference_manifest.md)、[算法审计](docs/oscar_reference_audit.md)、[缓存审计](docs/cache_integration_audit.md)、[MTP与图审计](docs/execution_integration_audit.md)。
- [设计](docs/design.md)、[算子清单](docs/operator_inventory.md)、[AscendC ABI](docs/operator_abi.md)。
- `oscar_ascend/cache_budget.py` / `cache_integration.py`：真实物理预算、GDN/FULL池拆分、原生页号/虚拟页映射与清零。
- `oscar_ascend/plugin.py` / `backend.py` / `metadata.py` / `ops.py`：延迟加载、FULL路由、请求生命周期契约、两阶段NPU窗口/Attention计算。
- `csrc/`：独立CMake、Host绑定、Vector旋转/clip/INT2/窗口，以及Cube+Vector历史Attention。当前旋转和BF16窗口实现仍需性能优化，不声称不慢于原生。
- `configs/acceptance_matrix.json`：功能与质量/性能验收矩阵；模型质量门槛仍为待确定，不能事后放宽。
- `benchmarks/compare_runs.py`：配对测量比较；输入/环境不匹配拒绝比较，任何单独工况退化都会报告。它不是完整HTTP负载生成器或最终性能验收。

当前无远端、未推送。用户稍后提供Git地址后再交付分支。参考树不随本项目安装；目标运行时使用用户的原生软件栈，并必须采集其真实版本与源码状态。
