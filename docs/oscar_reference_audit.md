# OSCAR 指定参考实现审计

状态：2026-09-14 静态代码审计；CUDA/NPU 编译、运行、精度、性能、图捕获/回放均 **not_run**。本文描述当前允许读取的文件内容，不将 PROVENANCE 声明视为已验证的上游身份。

## 1. 来源与范围

- `references/PROVENANCE.md:17-18` 声明 vLLM PR [#46774](https://github.com/vllm-project/vllm/pull/46774)，head `57286d5d2cb08c3dcd8c17bb59e132d6985e6796`，以及论文仓 [FutureMLS-Lab/OSCAR](https://github.com/FutureMLS-Lab/OSCAR)，commit `41ebcdba3db5f0ce1339c3727caea80df575d437`。
- 当前 PR 目录实际只有 **11 个普通源码/测试文件**，不是 PROVENANCE 所称 21 文件；缺少 PR 修改的完整宿主集成文件。`PROVENANCE.json` 没有 PR 的独立 source 条目。paper 也没有本树独立 `.git`，不能用向父目录上溯得到的 HEAD 声称已验证其 commit。
- 此次只读指定 PR、paper 和 PROVENANCE；没有打开或复用旧失败项目代码。身份核对时一次 Git 命令意外上溯祖先仓库并列出状态，已停止；该输出不用于任何实现/来源判断。后续 Git 检查必须先确认 `.git` 属于目标目录。
- 对应核心文件为：`vllm/model_executor/layers/quantization/oscar/{config,rotation}.py`、`vllm/v1/attention/backends/oscar_attn.py`、`vllm/v1/attention/ops/triton_oscar_{store,decode}.py`、`tests/quantization/test_oscar.py`。下文 `PR/` 代表 `references/oscar-vllm-pr46774/`，`paper/` 代表 `references/oscar-paper/`。
- MLA 文件存在，但 Qwen3.5 FULL 是普通 GQA/MHA 路径；MLA 不是本任务直接接口，不据此替换模型结构。

## 2. 已证实的数值与格式契约

### 配置与几何

`PR/vllm/model_executor/layers/quantization/oscar/config.py:25-28,68-83,146-176`：唯一 preset 为 `oscar_int2`，K/V 各 2 bit，默认 dataclass `group_size=128`、`S=64`、`R=256`、staging=8192。dataclass clip 默认 0；实际 `from_cache_dtype` 从 `VLLM_OSCAR_*` 读取，当前快照没有 envs.py，不能声称已核对这些环境变量的注册默认值。

该 PR **只实现每 token、每 head、K/V 各一个量化组**。`from_cache_dtype:154-163` 对 `group_size < head_dim` 报错，不能只改元数据尺寸假装支持分组：

```python
if group_size < head_dim:
    raise ValueError("... one quantization group per head vector.")
```

`config.py:86-130` 与 `oscar_attn.py:88-103` 给出每个 head-position 的布局：

```text
data_K = ceil(Dk * 2 / 8)
data_V = ceil(Dv * 2 / 8)
meta_K = meta_V = 4                 # 此PR合法配置的一组
slot = align_up(data_K + 4 + data_V + 4, 2)
cache[num_blocks, block_size, Hkv_local, slot] : uint8
slot = [K bits | K scale:f16 | K min:f16 | V bits | V scale:f16 | V min:f16]
```

PR 生产函数实际上使用同一个 D，Dk≠Dv 尚未实现。D=128 时 slot=72 bytes/head/token；D=256 且显式 `group_size>=256` 时推导为 136 bytes/head/token。E_K=E_V=256 可能来自多个 head，不能与 D=256 混为一谈：例如 Hkv_local=2、D=128 时是 144 bytes/token。**160 bytes/token 没有从此 PR 得到依据**。这些是 INT2 存储预算，不含 BF16 窗口、映射、对齐、旋转、工作区和 GDN。

### 旋转与裁剪

`oscar_attn.py:_ensure_rotations:219-233` 按层加载 fp32 `[D,D]` Rk/Rv，保存转置；`_rotate_clip:235-243`：

```python
x_rot = torch.matmul(x.float(), R)
thr = torch.quantile(x_rot.abs(), clip_ratio, dim=-1, keepdim=True)
x_rot = torch.clamp(x_rot, -thr, thr)
```

clip_ratio>0 才裁剪，K/V ratio 独立。这里是每行 abs 百分位阈值，不是 min/max 范围乘 0.96。`torch.quantile` 未指定 interpolation，默认 linear；AscendC 应实现排序后 `u=(D-1)*ratio`、相邻 order statistics 线性插值，且将此规则单独做对齐测试。不能替换成 paper 的 floor(ratio*D) 阈值。

数学关系由 `config.py:40-50` 与 `_decode_attention:599-615` 同时确认。定义存储反量化结果 `Khat_rot=Dequant(Quant(clip(K Rk)))`、`Vhat_rot`，则目标混合精度历史注意力是：

```text
Qrot = Q Rk
P = softmax(scale * Qrot Khat_rot^T + causal_mask)
O_history = (P Vhat_rot) Rv^T
```

对于正交 Rk，`Qrot Khat_rot^T = Q (Khat_rot Rk^T)^T`。对于线性逆旋转，`(P Vhat_rot)Rv^T = P(Vhat_rot Rv^T)`。因此可在 Q 和输出侧各变换一次，无需完整历史 inverse rotation；这是对**反量化混合精度参考**的等价性，不是声称 INT2/clip 无精度损失。BF16 Sink/Recent 用原始空间 Q/K/V，其输出须与已映回原空间的历史输出按 LSE 权重合并。所有路径注意力 scale 保持调用方 `self.scale`。

### 量化、位序与元数据

`PR/vllm/v1/attention/ops/triton_oscar_store.py:_store_int2_vec:40-77`：

```text
s = max((max(x)-min(x))/3, 1e-8)
s16 = fp16(s); z16 = fp16(min(x))
s32 = fp32(s16); z32 = fp32(z16)
q = clamp(trunc_to_int32((x-z32)/s32 + 0.5), 0, 3)
byte[b] = q[4b] | q[4b+1]<<2 | q[4b+2]<<4 | q[4b+3]<<6
reconstructed = fp32(q) * s32 + z32
```

`zero` 在该 PR 是**最小值 min 的浮点偏置**，不是量化整数 zero-point。FP16 scale 和 min 先舍入再参与编码；不能按 fp32 编码、fp16 解码。元数据各拆成低字节、高字节写入，即 little-endian IEEE half。Scale/min 分别位于 `DATA_BYTES+0/2`，V 区起点 `KEY_PACKED`。编码无符号 0..3，四个连续元素的第一个在最低 2 bit。

尾部：源 `BLOCK_D=next_power_of_2(D)`，有效元素 mask；几何支持 ceil(D/4)，但生产断言声称所有正 D、已提供测试只含 D=64/128（见下文），不能据此认可任意 D 的打包安全性。单独覆盖 D=256 及尾部 sentinel 不越界测试后再放开支持。

**PR 边界风险**：`max(scale,1e-8)` 在转 fp16 后仍可能成为 0，常量/近常量行会产生除零；此快照没有这些测试。需要明确退化行处理并对参考裁决，不得把修改后的结果冒充逐 bit 完全复刻 PR。

## 3. 调用链与请求级窗口

`OscarAttentionBackend.forward_includes_kv_cache_update=False`（oscar_attn.py:58）。`do_kv_cache_update:338-363` 先对所有输入 K/V rotate+clip，再 `oscar_store` 按 slot_mapping 写 cache；`triton_oscar_store.py:99-112` 忽略 slot<0，`block=slot//block_size`、`offset=slot%block_size`，按 cache stride + head stride 寻址。当前快照未包含完整调用方，外部插件必须重新核对目标宿主如何调用此方法。

`_staging_write:282-315` 的逻辑位置计算：

```python
pos = seq[req] - qsl[req + 1] + torch.arange(N, device=slot.device)
keep = (slot >= 0) & ((pos >= seq[req] - recent_tokens) | (pos < sink_eff))
```

所以 Sink/Recent 作用于**请求全局位置**，不是每物理页各保留一份。但 PR 物理实现仅为 prototype：

- `_ensure_staging:253-280` 将 sink 向下取整为完整页 `sink_eff=(S//bs)*bs`；bs=128、S=64 会变成 0，并不满足用户要求的精确 S=64。
- staging 哈希行 `(physical_block % rows)` + owner tag。两个 BF16 `[rows,bs,Hkv,D]` 张量独立于完整 INT2 pool。
- `do_kv_cache_update` 所有 token 写 INT2，窗口 token 又 BF16 双写（注释明确见 245-251、285-287）。它不是 Recent→History 迁移算法。
- `_decode_attention_windowed:644-683` 只有完整 sink owner 命中且 seq>S_eff 才激活 sink；尾部从当前 Recent 范围寻找连续仍在 staging 的后缀。碰撞/丢失则退回 INT2。
- History 通过 `bt_eff` 跳过实际生效 sink 整页、`seq_eff=cut-si*bs` 处理（685-706）。BF16 sink/tail 从 staging 收集后 Torch einsum+softmax（717-743）。
- 最后 `new_lse=logaddexp(lse1,lse2)`、`out=o1*exp(lse1-new_lse)+o2*exp(lse2-new_lse)`（745-750）。不能直接相加独立段 normalized output。

本任务 H08/H13 不允许照搬双写和碰撞退化；应保留请求级数学窗口而重新实现无冗余的生命周期/分配。短序列数学契约可用 `s=min(S,L); r=min(R,L-s); h=L-s-r`；这是任务规定的目标契约，**不是 PR prototype 每一种短序列/页大小行为已经与其等价**。

## 4. Prefill、decode、stage2、MTP

### Prefill

- `OscarMetadataBuilder.build:150-174` 记录 `has_context`，仅用 max_query_len==max_seq_len 不能判断全 batch 都是首次 prefill。
- `_prefill_attention:494-513` 在所有请求无旧上下文且可用 flash-attn 时直接使用原始当前 K/V。
- continuation `515-573` 将 qsl/seq_lens 转 list 并逐请求 Python 循环；每请求 full_dequant 旧上下文、Rk/Rv inverse、splice BF16 staging、concat 新 K/V，再 SDPA。`_sdpa:586-590` 因果 mask 为 `k_pos <= cached_len + q_pos`。
- `oscar_full_dequant_kv:357-410` 按缓存前缀长度分配两个 FP16 HBM 张量，并向页尾取整。这不是有界历史 tile。

首次 raw-KV attention 数学规则可复用；continuation 生产实现违反本任务热路径和全历史恢复限制，必须重写成 tile 级历史融合读取 + 新 chunk，不能当 MTP verify 用。

### Decode Stage1/2

`triton_oscar_decode.py:_oscar_decode_stage1:25-150` grid=(batch, query_head, split)；每个 program 一条 Q，按 `kv_head=query_head//GQA` 选 KV head。每 split `ceil(seq_len/num_splits)`；循环每次 `BLOCK_KV=4` token（launch:292-322），从页表读取 INT2 K/V、解包、fp32 反量化、算分与 online softmax，并在寄存器中累计 V。中间 `Mid_o[B,Hq,splits,D+1]:fp32` 保存 normalized output + LSE：

```python
acc = acc * exp(m_prev - new_max) + sum(p[:, None] * values, axis=0)
l = l_prev * exp(m_prev - new_max) + sum(p)
mid_out = acc / l
mid_lse = new_max + log(l)
```

它没有写全历史 BF16 HBM，但只用了 `tl.sum` 向量归约，没有 `tl.dot`，没有 Ascend Cube/Vector 实现。**不能把该 Triton Stage1 改名便称为 AscendC fused CV**。

`oscar_decode_attention:335-353` 导入宿主 `triton_decode_attention._fwd_kernel_stage2` 做跨 split LSE reduction，当前 11 文件快照没有其函数体。PR 的调用、mid-buffer格式与 windowed LSE 合并可确认；完整 stage2 边界行为需要宿主参考/独立实现核对，尤其空 split、零长度历史及 NaN 处理。

### MTP/图模式

`oscar_attn.py:134-141` 明确：

```python
_cudagraph_support = AttentionCGSupport.NEVER
self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
```

`is_prefill=(cam.max_query_len>1)`（168）使多个 verify query 走 prefill 分支；该 snapshot 没有 acceptance/rejection/commit/rollback、MTP shadow 更新或 verify 专用历史 tile 复用契约。将每 query 展开为普通 decode batch 也会每条 query 重读 INT2 历史；不满足 H12。

可实施的目标设计（待代码和真机证明）：以 `(request, kv_head, query_tile, history_split)` 为任务，query_tile 包含该 KV head 下的 GQA query heads 与同请求多条 verify query；历史 packed tile **仅一次 HBM加载**，Vector 解包/scale 后供 Cube QK 与 PV，保留多行 online softmax状态，每行施加原生可见长度/因果mask；片上 tile 在这些 query 完成前不能被覆盖。最后跨split合并，并对 History output做Rv^T；BF16窗口结果在原空间LSE合并。若 query_tile 超片上容量而被分多组，必须报告组数与重复流量，不能声称完全无重复。

该设计的正确性来自上述线性变换和LSE恒等式；当前参考没有证明其 Ascend target tile形状、UB/L1容量、Cube/Vector同步、dtype误差或性能。KV归属与提交长度必须来自原生MTP调度，不由此算子私自提前提交。

## 5. 校准来源与 paper 的差异

`PR/.../rotation.py:56-84` 支持 `{layers:{layer_id:{rotation:Tensor}}}`、平字典、三维堆叠tensor；矩阵为fp32 `[D,D]`。`get_layer_rotation:94-119` 遇空路径/缺层会 identity fallback，且只验shape，没有检查正交性、模型身份、TP身份、K/V objective匹配。交付验收必须显式检查完整FULL层coverage和artifact身份；identity只能是明确调试模式。

paper校准 `rotation/compute_kv_rotation.py`：

- `compute_qqt:93-108` 对匹配KV head的GQA Q行计算 `Q^T Q / n_rows` 后跨KV head平均；K旋转用其eigenvectors。
- `compute_sst:111-136` 先算 `qtq`，每token `w=(K qtq * K).sum(-1)`，按sum归一化至均值1，再 `V^T diag(w) V / n_tokens`，跨KV head平均；V旋转用其eigenvectors。
- `build_hadamard:23-29` 是递归归一化Hadamard，D须二次幂；`make_br_perm_matrix:39-46` 按eigenvalues降序，bit-reversal配置perm。`compose_rotation:234-265` 明确 `r_h_pbr = eigenvectors @ H @ Pbr`。
- `empty_result/add_layer:273-287` 写format_version=1/source_grouping=layer/objective/layer_id/rotation/eigenvalues；脚本`--composition`默认却是 `plain`（341-355），部署生成必须显式指定 `r_h_pbr`，不能依赖README文字默认。
- `load_tensor:49-72` 的离线脚本在CPU加载QKV，all模式跳过chunk 0。该流程不可原样变为本任务在线回传KV校准；应在NPU收集/归约足够统计量并区分离线结果比对。
- `paper/README.md:296-305` 本树没有Qwen3.5-27B校准配方/产物，README指向另一branch。给出的RotationZoo链接不是已存在的目标artifact证明。

**paper 与 PR 并不逐字节同构**（对照 `paper/sglang-research/python/sglang/QuantKernel/oscar_rotation_clip_int2_kv.py`）：

| 项目 | 指定PR | paper所读实现 |
|---|---|---|
| clip threshold | torch.quantile，linear插值 | 100-104、333-345：sort取 `min(int(ratio*D),D-1)` |
| q位序 | 连续 `q[4b+i] << (2i)` | 168-179：四分段 reshape/permute，`q[b+i*(D/4)] << (2i)` |
| zero语义 | FP16 min，`q*scale+min` | 165-177：`zero=-min/scale`，`(q-zero)*scale` |
| metadata | K/V向量各4 byte，内嵌slot | 独立scales_zeros tensor；README:338服务用FP32 |
| 分组 | group_size>=D，仅一组 | 203-319另有grouped kernel |
| Lloyd-Max | 指定PR无此模式 | 110-160可选LM近似分桶，不等同uniform INT2 |

实现要以指定PR的数值/格式契约为准；paper只作为旋转目标、窗口方向和实现交叉核对，不能不加说明地混用。若未来要采用paper非均匀LM，这是算法范围变更。

## 6. 测试证据与未覆盖项

所有测试为**阅读到的测试要求，未在本次运行**。`test_oscar.py:23-25` 整文件要求CUDA/Triton；不能报告NPU通过。

| 测试 | 参考行号 | 已编码的检查 |
|---|---|---|
| 配置几何 | 65-70 | D=128，K/V=36 bytes，slot=72 |
| store/dequant | 84-132 | D=64/128，40tokens、4KV heads、bs=16，atol=rtol=2e-3 |
| decode | 135-216 | (D,Hq,Hkv)=(128,8,2)/(64,4,4)，B=2、L=48，vs已反量化cache，atol=rtol=5e-3 |
| fresh+continuation混batch | 248-361 | Q_A/S_A=32/32、Q_B/S_B=16/32，有/无window，relativeL2<2e-2 |
| window decode | 364-467 | D=128,B=2,L=100,S=16,R=32,clip=.96/.92，relativeL2<2e-2 |
| staging全驱逐 | 470-525 | 与全INT2输出相同，atol=rtol=1e-4；这种允许退化行为不满足本任务 |

`oscar_gpqa_eval.py:3-10,20-24,105-107` 是Qwen3-32B GPQA-Diamond脚本，temperature=0，不是本模型质量验收结果。当前没有目标Qwen3.5-27B基准数据、可接受质量差值、16K–50K性能结果。

关键缺口：目标D=256与配置、全零/常量/near-zero行、FP16 scale underflow、非有限输入、所有短窗口边界、bs=128/S=64、非连续页、跨页Recent迁移、prefix共享/写时复制/回收、MTP部分接受与全拒绝、混长度多query、NPU图捕获返回和真实请求回放、长期数值漂移、真实显存收益及端到端公平性能。

## 7. 可直接推进与必须保留的验收边界

可直接实现纯Host部分：严格配置解析、每head-byte几何和总预算公式、请求级窗口边界、slot/page地址整数契约、layer-name→layer-id解析、artifact schema/coverage检查、原始格式的pack/golden离线测试、算子shape/dtype/tiling元数据验证。不得让这些Host工具接管在线KV运算。

可据此实现的AscendC数学：连续INT2 pack/unpack及内嵌metadata、rotate/linear-quantile-clip/quant写、Q旋转、分块INT2历史attention+LSE、原空间BF16窗口attention、输出逆旋转和LSE合并。真实CV执行还需正确目标Cube/Vector API和同步；编译可运行与真机数值/性能分别验收。

不能沿用的PR生产设计：staging双写/碰撞退化、任意S按页下取整、multiquery走full_dequant、逐请求Python循环、图模式NEVER、identity静默降级。这些是参考已证明的限制，外部实现必须解决，不能引用PR存在便标完成。
