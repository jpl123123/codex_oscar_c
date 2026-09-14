# 独立算子 ABI v2

全部在 `torch.ops.oscar_ascend`。本地尚无 CANN 编译/NPU执行证据。输出由调用者预分配，固定地址可用于图捕获；是否capture/replay成功仍需probe。每个tensor除history第一页stride可有padding外须contiguous，同设备NPU。D=64/128/256，group_size>=D。v2 相对 v1 新增 prefix/staging/dequant 三算子与每层 lossy 计数器，既有算子签名不变。

```python
abi_version() -> int  # 2
workspace_size(int n, int hq, int d, int splits) -> int
rotate_out(x, rotation, out, bool transpose=False) -> ()
store_int2(key, value, rk, rv, slots, history, float kclip, float vclip) -> ()
history_attention_out(qrot, history, block_table, qsl, history_start, history_end,
                      query_positions, output_rot, lse, workspace,
                      float scale, int splits, int max_query_len,
                      int block_table_block_size=0) -> ()
window_attention_out(query, key, value, window_k, window_v, window_positions,
                     row_to_window, qsl, query_positions, output, lse, float scale,
                     workspace, int max_query_len) -> ()
merge_out(history_output, history_lse, window_output, window_lse, rv, output) -> ()
zero_blocks_out(history, block_ids) -> ()
dequant_history_out(history, slots, rk, rv, key, value) -> ()
stage_window_out(key, value, seq_lens, qsl, slots, staging_k, staging_v, owner,
                 int sink, int recent) -> ()
prefix_restore_out(seq_lens, qsl, row_to_window, epochs, block_table,
                   window_positions, window_k, window_v, state,
                   staging_k, staging_v, owner, history, rk, rv, lossy_counter,
                   int sink, int recent, int max_pending, int block_table_block_size) -> ()
window_state_out(key, value, seq_lens, qsl, row_to_window, epochs, block_table,
                 native_slots, is_prefill, window_k, window_v, window_positions,
                 state, history_start, history_end, query_positions,
                 migration_k, migration_v, migration_slots, current_slots,
                 int phase, int sink, int recent, int max_pending, int block_size) -> ()
```

- raw Q/K/V, Qrot, window_k/v, migration_k/v, final output: BF16。History/window attention partial outputs及LSE: FP32。Rk/Rv: FP32[D,D]，merge_out接收Rv并在设备使用转置。
- raw `[N,Hq/Hk,D]`；INT2 history `[nb,bs,Hk,C]` uint8；`C=2*(D/4+4)`，每head单组，packed+half scale/min。
- qsl INT32[B+1]；history_start/end INT32[B]；query_positions INT32[N]；block_table INT32[B,maxpages]；slots/current_slots INT64[N]。
- `block_table_block_size` 默认0使用history物理page size；native若将768-token物理页拆成128-token虚拟页必须显式传128。先从virtual page计算flattened slot，再用history.shape[1]与真实stride映射物理地址。window_state的block_size同样指block table的虚拟页粒度。
- window_k/v `[W,S+R+P,Hk,D]`，window_positions INT32[W,S+R+P]；state INT64[W,4] `(epoch,committed_end,pending_count,error)`；row_to_window INT32[B]（padding=-1）；epochs INT64[B]；seq_lens INT32[B]；is_prefill BOOL[B]；P=max_pending=M+1。
- migration_k/v `[B*(R+P),Hk,D]`，migration_slots INT64[B*(R+P)]。每phase先清该请求的migration slots为-1，不读取Host KV。
- window phase=0：按`C=seq_len-query_len`接受prior pending前缀，迁移真正离开Recent的旧token，丢弃拒绝pending，输出历史范围和绝对query positions。
- window phase=1：prompt is_prefill确认当前chunk，gather滑出旧Recent、emit当前新History slots，写新Sink/Recent；普通decode/verify只写至pending区等下一step权威长度决定接受。
- phase0→stage_window→prefix_restore→migration store→Q rotate→History CV→window/current raw attention→merge→phase1→migration store→current maskedslots store。stage/restore 在 phase0 之前：staging 镜像当前 Sink/Recent 行（owner-tag 先行），restore 重建 prefix 命中行窗口。当前prefill attention前不得更新窗口覆盖其旧Recent。
- staging_k/v `[rows,bs,Hk,D]` BF16、owner INT64 `[rows,bs]`（-1空）；行深 `max(ceil(staging_tokens/bs), sink_pages+tail_pages+2)`，键为与 slot_mapping 同名的虚拟 slot。restore 对 epoch 变化且 C>0 的行执行：Sink `[0,min(S,C))`+Recent `[max(S,C-R),C)` 从 staging 精确恢复，tag 逐出行回退有界 INT2 反量化并 atomic 计入 lossy_counter INT32[1]（不冒充无损）；错误码6=恢复失败。C<=0 的行留给 phase0 正常初始化。
- dequant_history_out 为显式行数上界的恢复/核对算子，decode/verify 热路径不调用（pipeline 调用序列由契约测试固定）。
- 新epoch with prefix C>0（经restore）、缺窗口token、超限pending、非法commit区间记录非零state.error；错误码1/2/3/4/6。生产传播由 runtime 一步延迟 pinned 镜像完成，probe 用 check_window_errors 显式同步检查。

workspace_size 精确字节：`align512(((16D+2*32D+16*32)*2 +(16*32+16D)*4))*aic_cores + align512(N*Hq*splits*(D+1)*4)`。第一项为每物理Cube固定32token桥接，第二项为normalized split partials。建议prefill splits=1、decode/verify splits=8，跨层共享workspace，实际容量必须计入allocator。INT2历史读取次数随query tiles=ceil(q_len/16)，q_len=1..4共一个tile；当前grid按query head，GQA之间尚未共享INT2加载。
