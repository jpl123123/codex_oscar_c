# Reference 与执行环境清单

本次审计日期：2026-09-14。仅使用当前工作区指定 `references/`；其内容保持只读。本次用户明确要求先本地实现，之后由用户 pull 到真机测试，暂不配置远端。

## 已核实的源代码

| 来源 | 本地路径 | 身份与当前证据 |
|---|---|---|
| vLLM 0.23.0 | `references/vllm` | 本地独立 Git HEAD `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`；tracked changes 为空。|
| vLLM Ascend 参考 | `references/vllm-ascend` | 本地独立 Git HEAD `19e436985102f4ed3aad36c137a6481653688a6c`；tracked changes 为空。PROVENANCE 声明它是发布线加 PR12607，因此不能当作用户机器的原生安装 commit。|
| 指定 OSCAR PR | `references/oscar-vllm-pr46774` | [vLLM PR46774](https://github.com/vllm-project/vllm/pull/46774) 页面已在线核实。PROVENANCE 声明 commit `57286d5d2cb08c3dcd8c17bb59e132d6985e6796`；本地无 Git，实际只有11个源码/测试文件，与说明的21文件有差异。逐文件上游核验结果见 `reports/oscar_upstream_verification.json`，不能把声明当作内容校验。|
| 论文实现 | `references/oscar-paper` | PROVENANCE 声明 `FutureMLS-Lab/OSCAR@41ebcdba3db5f0ce1339c3727caea80df575d437`，本地无独立 Git，按本次文件清单SHA256固定阅读内容；未在线核实整个树。|

完整文件SHA256、数量及规范化清单hash：`reports/reference_inventory.before.json`。复核工具 `tools/audit_references.py` 不导入 reference Python、不执行其中脚本、不跟随树外软链接，也不对无 `.git` 的快照执行向父目录上溯的 Git 命令。

旧 PROVENANCE / seam-map 的规则、外部路径和历史结论仅作待核实材料；不执行其中指向旧工程的命令，也不接受其全局 eager 建议覆盖本次启动指令。

## 调用链与数值证据

- `docs/oscar_reference_audit.md`：指定PR配置、旋转、clip、编码、窗口、store、prefill/decode、测试与paper差异。
- `docs/cache_integration_audit.md`：FULL/GDN规格、分组、物理视图、页号与容量预算。
- `docs/execution_integration_audit.md`：MTP接受/拒绝、原生异步元数据、图capture/replay、注册时机和生命周期。
- `docs/operator_inventory.md`：AscendC 算子ABI职责与验收方法。

## 当前本机与目标未知项

本机报告 `reports/local_environment.json`：Darwin arm64，Python3.12；未安装 torch、torch_npu、vLLM、vLLM Ascend、CANN/AscendC编译器、cmake；目标模型及 `/vllm-workspace/` 不存在。没有本次 NPU 编译或运行结果。

正式参数完整保存到 `configs/target_service.json`，并逐参数与启动文档附录A测试；端口8989，TP4，DP1，MTP3，FULL_DECODE_ONLY，异步调度，Ascend权重量化，BF16 GDN，最大长度262144。物理卡4–7是先前用户限制，当前启动说明没有指定新卡号；所有本项目启动器固定为4–7，用户真机测试前需确认这些卡仍属于本任务。

目标机器必须实际记录：NPU型号/HBM/拓扑、Python/torch/torch_npu/vLLM/Ascend/CANN/驱动/固件/编译器、安装源码commit与导入位置、模型config.json及其hash、实际FULL/GDN层与MTP层、本rank维度、block size、cache groups/shared_by。`python -m oscar_ascend.environment --require-target --output <report>` 能采集基础环境；它只做只读发现，**不代表runtime接口兼容、NPU device完成或正确性通过**。

## 校准与验收输入

PR 的目标数据格式为每head一组INT2，FP16 scale+min内嵌；D=256需group_size>=256。RotationZoo的公开位置不能证明已经有Qwen3.5-27B匹配产物。必须记录模型/层/维度/TP/PR身份，验证K/V覆盖与正交性；缺矩阵不能静默identity。

参考数值容差已在算法审计中逐项记录；它们不等于目标模型质量门槛。目标质量任务/数据、允许下降量仍未指定，不能根据移植结果倒推门槛。性能必须同一输入、同一启动参数、原生与OSCAR配对逐工况比较；历史日志不计本次证据。真机验证由用户执行，Checklist中相应项保持not_run。
