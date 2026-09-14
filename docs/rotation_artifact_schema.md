# Rotation artifact v1

自动 NPU 校准生成尚未实现。服务 runtime 不会用 identity 或缺层 fallback 替代真实产物；独立算子 probe 中的 identity 仅用于验证数值和状态机。

通过 `OSCAR_ASCEND_CONFIG` 指定 JSON 配置，其中 `rotations_path` 是已验证 `.pt` 文件路径。文件由 `torch.load(weights_only=True)` 读取，仅反序列化旋转常量；只将本 rank 的矩阵上传 NPU，不把其他 rank 的矩阵放入本卡 HBM。

```python
{
    "format": "oscar-ascend-rotations-v1",
    "model": "/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp",
    "model_config_sha256": "SHA256 of exact model/config.json bytes",
    "reference_commit": "57286d5d2cb08c3dcd8c17bb59e132d6985e6796",
    "tensor_parallel_size": 4,
    "group_size": 256,
    "objectives": {"k": "qqt_r_h_pbr", "v": "sst_r_h_pbr"},
    "clip_ratios": {"k": 1.0, "v": 1.0},
    "ranks": {
        "0": {
            "exact.runtime.FULL.layer_name": {
                "k": torch.Tensor,  # contiguous float32 [D,D]
                "v": torch.Tensor,  # contiguous float32 [D,D]
            },
        },
        # ranks 1, 2, 3 use their actual FULL and MTP layer names
    },
}
```

`runtime.validate_rotation_metadata` 验证 model config 字节 hash、PR commit、TP、group size、K/V objective 和独立 clip ratio。每 rank 必须恰好覆盖当前 runtime 的所有 FULL 层，矩阵 shape/dtype/连续性须符合真实 local spec。objectives 对应参考校准的 `Q^TQ` / score-weighted `V^TV` 与 `R H Pbr` composition，不能把 K 产物误当 V 产物。

初始化时在 NPU 上以实际 AscendC `rotate_out` 将 BF16 单位基乘 R、再乘 R^T；FP32 NPU 聚合 max absolute error，只有一个诊断标量返回 host。要求误差 <=0.02，并拒绝 NaN/Inf。这个阈值含两次 BF16 舍入，只是算子 roundtrip 验证；不宣称矩阵在数学上严格正交，也不替代模型精度测试。临时 basis/误差缓冲峰值计入物理 reserve。

目前还不能验证原始模型权重内容 hash、校准样本身份和所有 HF override 对旋转统计的影响。自动产物生成、这些身份字段扩展、模型级精度及 MTP acceptance 率均保留为最终验收缺口；不得仅凭此文件成功加载宣称校准完成。
