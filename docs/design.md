# 外部 OSCAR AscendC 设计与实现边界

本设计由本地参考代码审计得出。接口代码和独立算子正在落地；CANN编译、NPU数值、图回放、端到端性能都尚未验证。完整完成状态只看 `checklist.md`，不能从存在源文件推断服务可运行。

## 数值算法

每个 FULL head 以原始 BF16 K/V 形成请求全局 Sink/Recent 窗口。History 使用指定 PR 的单组非对称 INT2：旋转到FP32，按abs的linear quantile裁剪，scale/min先round到FP16，再编码连续四元素至一byte最低位优先。`D/4+4`字节分别存K和V；D=256的裸slot是136字节，不是160字节。退化行scale转换为FP16后可能下溢，此参考缺陷必须在实现中明确定义并专门验证。

令反量化后的旋转域历史为Kh、Vh。采用 `Qr=Q Rk`、`Oh=softmax(Qr Kh^T)Vh`、`Oraw=Oh Rv^T`。此变换对指定量化参考等价；Sink/Recent使用原空间Q/K/V，先将History输出逆旋转再按LSE合并。它不要求把整个History恢复成BF16 HBM。

## 物理布局及分配

`cache_budget.py:FullLayout/Pool/CachePlan` 只计算Host形状与字节，不接触KV数值。`cache_integration.py` 包装原生 `get_kv_cache_config_from_groups`、NPU runner的allocate/reshape。输入必须来自实际runtime spec和`KVCacheTensor.shared_by`，不硬编码body层数，也不遗漏MTP层。

原生全局block ID、cache groups和block table保留。每个原共享tensor拆成GDN真实状态池和FULL压缩池，GDN状态shape/dtype保留。第g个GDN状态条在该池内的偏移为 `sum(previous_state_page_bytes)*num_blocks + block_id*state_page_bytes`；状态内stride由真实shape推导。FULL为 `history[block_id, token_offset, local_kv_head, slot_byte]`，每页末尾只做必要对齐。

FULL地址为 `base + block_id*page_stride + token_offset*Hkv*C + head*C`。`block_id=block_table[request, position//block_size]`，`token_offset=position%block_size`，负slot不得写。Sink/Recent另为每层每请求一个有界BF16窗口，含最大候选尾部；没有完整BF16 History池，也不能让同一组的GDN和FULL同时写同一底层地址。

预算：

```text
C = ceil(Dk/4)+4 + ceil(Dv/4)+4               # 每head，仅单组
P_full = align(block_size * Hkv * C, alignment)
P_gdn = sum(align(prod(state_shape)*dtype_bytes, alignment))
M_window = sum_over_FULL_layers(max_requests*(S+R+M+1)*2*Hkv*(Dk+Dv))
M_physical = num_blocks*sum(distinct_pool_page_bytes) + M_window + M_runtime_reserve
num_blocks = floor((budget-M_window-M_runtime_reserve)/sum(distinct_pool_page_bytes))
```

`num_blocks`包含原生null block。运行时新增rotation/query/partial/workspace/metadata/图buffer必须全部进入reserve，不能传0然后声称总预算完整。Host公式证明设计预算，真机物理分配和固定HBM容量收益另验收。

## 窗口与提交

请求级区间采用 `s=min(S,L); r=min(R,L-s)`，分别 `[0,s)`、`[s,L-r)`、`[L-r,L)`。Sink不按物理页下取整。请求身份绑定generation，不能绑定可重排行号；finished/new同ID先释放旧generation。

当前 `TargetRuntime.refresh_rows` 仍每step遍历请求身份映射。它没有CPU处理KV数值，但仍不满足H05禁止新增逐请求Python热循环的要求，因此保留正式服务阻断项；后续应将映射维护移入原生InputBatch的add/remove/swap生命周期事件，step仅批量传输已准备的映射。不能把“只是元数据”作为该要求已经通过的理由。

MTP候选只占有界尾部。下一次target forward读取原生异步修正后的device `computed=seq_lens-q_len`，提交上一轮真正可见KV。bonus是下一次输入，其KV本轮尚不存在。提交时只对Recent真正滑出的token量化；拒绝后缀不进入History。历史不复制到shadow池。Host只填批量调度元数据，KV搬运、索引、commit/rollback在NPU kernel执行。

首次prefill当前Q/K/V使用原始精度Attention；新chunk历史读使用融合INT2 tile，当前chunk使用原始KV。确定提交后仅写首次进入History的token以及必要窗口；已处理旧chunk不得重复量化。实现必须区分`is_prefilling`和verify，不能把q_len>1等同prefill。

Prefix命中有额外语义难点：旧History的有损KV不能恢复新请求Recent原始BF16。实现需要无损窗口保存/可复用前缀边界及有限重算方案，涉及scheduler的命中长度和页生命周期。未完成前明确报缺失，不静默降级量化Recent，不以关闭prefix宣称验收通过。抢占/恢复、缓存连接器、block zeroing也是同一生命周期验收范围。

## AscendC fused CV

Stage1任务包含一个request、KV对应的query head、一个历史split和最多4条verify query。一次加载packed history tile，Vector解包和反量化后供Cube QK与PV；不同query保留独立causal mask、max/sum/output累计。不能把MTP展开成4次全history扫描。

```text
for history_tile in this_split:
    packed = load_tile_once(block_table, history_start, history_end)
    Krot,Vrot = vector_unpack_affine(packed)
    logits[queries,tokens] = cube_matmul(Qrot,Krot.T) * scale
    P,m,l = vector_mask_online_softmax(logits, q_positions)
    acc = rescale(acc) + cube_matmul(P,Vrot)
write split_output=acc/l, split_lse=m+log(l)
```

若Cube/Vector交接使用GM工作区，只允许每个物理core固定大小的双缓冲tile，其峰值与history长度无关；这个中转开销必须计入profiler，不能称作零HBM恢复。若按query head分任务，同一GQA KV会跨head读取，需分别报告；H12关注同组query从1到4是否线性增长。长chunk query数超过tile时的重复流量也须如实报告。

split/segment合并用稳定LSE权重。空split贡献0且LSE=-inf，不能让 `-inf-(-inf)`产生NaN。kernel错误/不支持shape必须显式失败，不能退到完整历史恢复。

## 外部加载和图

general plugin顶层只用标准库；原生platform完成初始化后再导入backend和安装受控hooks。保存原descriptor/signature，幂等安装且可撤销。仅选标准FULL Attention；GDN原生builder、conv/SSM和rejection sampler不改。

所有capture入参的cache、workspace、request映射、长度、query位置保持固定device地址。变长值由device tensor承载；不能在捕获Python闭包固化长度。target FULL_DECODE_ONLY和draft eager按各自原生作用域保留。builder的graph capability必须与真实固定地址实现一致；声明能力、capture返回、真实replay、数值正确分别验收。

## 交付顺序

环境/源码状态采集 → 外部安装 → 独立CMake构建 → 算子device/numerical/TP4 probe → 完整集成gate → probe进程及NPU资源释放 → 目标参数正式服务 → 就绪与真实压缩路由核验。任何阶段失败停止，记录完整日志，不能继续服务并冒充OSCAR生效。不会用 `cp` 或任何覆盖方式修改原生源码。
