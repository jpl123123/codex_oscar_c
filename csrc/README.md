# 独立 OSCAR AscendC 包

本目录为新写的外部算子工程，不构建或修改 `references/`，不覆盖原生 vLLM/vLLM Ascend 文件。实际 ABI 见 `../docs/operator_abi.md`。

```bash
cmake -S csrc -B build/ascendc \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DSOC_VERSION=Ascend910B1 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build/ascendc --parallel
```

SoC必须改为目标机器实际型号；910B/910_93以外架构当前主动拒绝。输出 `build/ascendc/liboscar_ascend_ops.so` 和相邻 kernel `.so` 应保存在一起，Python通过 `torch.ops.load_library` 加载。构建使用 CANN 的 `ascendc.cmake`；Torch C++ ABI flags由当前torch读取。没有执行复制命令，不安装进原生源码路径。

2026-09-14 本地为macOS，未安装cmake/CANN，尝试CMake命令返回`cmake: command not found`。**当前代码未经CANN编译，不宣称可加载、NPU数值通过或性能达标。** 目标机先用独立probe，记录具体首个编译/设备错误。

## 已写实现

- `op_kernel/oscar_history_cv.cpp`：单个 mixed 1C2V stage1包含Vector解包和online softmax、Cube `MatmulImpl` QK/PV、CrossCore事件、按split的FP32累加；随后独立vector stage2 LSE合并。mode=1以相同Cube tile机制处理原空间BF16窗口和当前raw chunk，已删除最初的AIV逐pair attention实现。
- `op_kernel/oscar_vector.cpp`：NPU行旋转、PR线性分位clip、fp16 metadata量化、连续2bit打包、输出V inverse旋转+LSE合并、批量物理页清零。负或越界store/zero slot在设备跳过，不能写出分配区域。
- `op_kernel/oscar_window.cpp`：请求级精确Sink/Recent及pending两phase状态；已有pending按本step设备computed length接受/拒绝；只有真正滑出token进入有界migration buffers；prefill在attention结束后才推进窗口，当前raw History产生masked slots，decode/verify先仅写pending。
- `op_kernel/oscar_prefix.cpp`（ABI v2 新增）：`dequant_history_out` 显式行数上界的INT2解包+逆旋转恢复（probe/参考对齐专用，decode/verify热路径不调用）；`stage_window_out` 按上游PR owner-tag先行协议把Sink/Recent行镜像进BF16 staging池；`prefix_restore_out` 对prefix命中行重建窗口——staging命中的行精确恢复，被逐出行回退同一套解包数学并atomic计入每层lossy计数器（不冒充无损），失败记错误码6。
- `op_host/tiling.cpp`：目标Core/UB/L1/L0C能力查询与QK/PV tile配置。成功的固定16×32×D / 16×D×32 tiling按 `(device,SoC,D,UB,L1,L0C)` 缓存，mutex保护访问，错误不入cache；不按每层/每step重复调用GetTiling，不缓存请求数据。没有读取在线KV数据。性能收益仍未实测。
- `torch/bindings.cpp`：NPU、同设备、shape/dtype/stride、workspace预算和GQA整除检查，调用当前NPU stream的custom handler；不调用device-to-CPU同步。

## 设计与证据边界

每个物理Cube有固定GM桥接tile：Q16行、K/V各32行、softmax概率和QK/PV结果。两个Vector子核分别读取不同的16条INT2行，同一32条历史被该tile所有query消费后才能覆写。固定GM桥接不是全历史恢复；其字节与历史长度无关且计入workspace_size。但它产生额外的BF16 tile写入/读取流量，尚无不慢于原生的证据。

MTP的1..4条query在一个16行tile内，INT2历史加载次数不随这4条query线性增加；query长度超过16时需要多个query tile。当前grid以query head划分，GQA相同KV head之间还有重复读取，必须在性能测量后优化，不能宣称全GQA复用。

Cube输入为BF16 Qrot和BF16反量化tile、FP32累加，PR使用FP32旋转/反量化算分。因此需要测量相对PR的误差；代码不是逐浮点bit等价实现。Store使用PR fp16-rounded scale/min编码。唯一明确的退化行修正：若PR `max(scale,1e-8)`转half变成0，将持久scale设为最小half subnormal `2^-24`，避免常量行除0；这个修正须有独立测试并记录，不能伪称PR本身已有相同处理。

Vector旋转和quantile bitonic排序优先实现清晰的设备数值路径，目前含设备scalar控制循环，未做吞吐优化。BF16窗口和当前chunk现用Cube QK/PV，但16×32小tile和GM桥接仍需测量优化。没有CPU数据fallback，也没有使用Triton替代AscendC。端到端性能验收仍未完成。

窗口state.error：1=commit长度不在prior pending范围，2=所需窗口token缺失，3=超限pending，4=页表范围/无效页，6=prefix恢复失败。prefix恢复（staging精确+INT2有界回退计lossy）与NPU异步错误一步延迟上报、drafter eager多步接入已由service适配实现；完整图捕获隔离与全服务probe仍未运行。不能仅因primitive运行而放开这些服务就绪条件。

## 当前API依据

- `references/vllm-ascend/CMakeLists.txt:40-79`：CANN cmake查找、`ascendc_library`直接kernel库。
- `references/vllm-ascend/csrc/torch_binding.cpp:24-27` 与 `batch_matmul_transpose/batch_matmul_transpose_torch_adpt.h:39-50`：NPUStream、OpCommand customhandler。
- `attention/chunk_gated_delta_rule/op_kernel/arch22/chunk_gated_delta_rule_stage1.h:295-398,809-817`：1C2V paired crosscore flags、`MatmulImpl`调用。
- `attention/chunk_gated_delta_rule/op_kernel/chunk_gated_delta_rule.cpp:29`：`KERNEL_TYPE_MIX_AIC_1_2`。
- `attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_tiling.cpp:165-217`：BF16输入/FP32输出Cube tiling与存储空间查询。

以上是当前workspace参考文件的位置，不是目标CANN版本已经编译验证的声明。
