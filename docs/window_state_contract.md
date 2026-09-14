# NPU 窗口状态与两阶段提交契约

此文定义外部 target FULL 层的设备状态机。算子与 Python 封装已实现（drafter eager 多步与 prefix 恢复见第 5 节），但尚未在 NPU 编译或执行；真机验收以 docs/checklist.md 为准。

## 1. 物理状态

每个 FULL 层独立拥有以下固定地址状态：

| buffer | dtype / shape | 含义 |
| --- | --- | --- |
| `window_k/window_v` | BF16 `[W,S+R+M+1,Hkv,D]` | Sink、Recent ring、上轮 pending 尾部；没有完整 BF16 History |
| `window_positions` | INT32 `[W,S+R+M+1]` | 每个窗口条目的绝对 token 位置；无效值 `-1` |
| `state` | INT64 `[W,4]` | `{epoch, committed_end, pending_count, error}` |
| `row_to_window` | INT32 `[B]` | 本批行映射到稳定 request window；padding=`-1` |
| `epochs` | INT64 `[B]` | request ID 重用/抢占恢复时递增的身份代次 |

逻辑位置 `p<S` 的 Sink 索引为 `p`；Recent 中 `p>=S` 的 ring 索引为 `S+(p-S)%R`；pending 索引为 `S+R+j`，位置为旧 committed_end+j。R=0 不执行 modulo，所有已确认非 Sink token 直接迁移 History。

共享 scratch 在各 FULL 层顺序执行时复用，不按层重复分配。迁移缓冲 `[B*(R+M+1),Hkv,D]`，对应 INT64 physical slots。长度是固定上界：verify 最多迁移 M+1 行；长 chunk 一次可能迁移全部 R 行，不能只分配 B*4。

## 2. prepare phase

输入是原生异步修正后的 NPU `seq_lens`、`query_start_loc`、group block_table/slot_mapping。NPU 上计算 `q_len=qsl[i+1]-qsl[i]`，`C=seq_len-q_len`。

1. 将迁移 slots 和当前 raw History slots 填为 `-1`。
2. 无效 row 退出。若 epoch 改变，重置所有窗口 positions 为 `-1`，state 设为本 epoch/C=0/P=0。若新身份 `C>0`，记录 prefix/window-restore 错误，不能从 INT2 伪造 BF16 Recent。
3. 要求 `old_C<=C<=old_C+old_P`，令 `A=C-old_C`。输入采样结果中的 bonus 尚未生成 KV，不能让 C 超过 old_C+old_P。
4. 即将成为 History 的区间为 `[max(S,old_C-R),max(S,C-R))`。把这些 position 对应的旧 Recent 或 pending 原值 gather 到迁移缓冲；原生地址为 `block_table[row,p//block_size]*block_size+p%block_size`。
5. 必须先 gather 再写 ring，避免覆盖尚需迁移的值。将接受的 pending `[old_C,C)` 放入 Sink 或新的 Recent `[max(S,C-R),C)`，拒绝的 pending positions 清为无效。
6. state 设为 C/P=0；输出 History start=`min(S,C)`，end=`max(start,C-R)`，query position=`C+j`。
7. 独立 `store_int2` 只压缩 valid migration slots；此刻 History 和窗口共同描述已确认前缀，当前 raw K/V 仍独立存在。

## 3. attention phase

History attention 读取 prepare 输出的 INT2 区间；window attention 读取有效 committed Sink/Recent 加当前 raw K/V。两部分按各 query 绝对位置做因果 mask，以 LSE 合并，不重复包含当前 raw tokens。历史 V 输出由经验证的逆旋转恢复后再与原域 BF16 窗口输出合并。

chunked prefill 必须先完成这一步再覆盖旧 Recent。指定 PR 的 `_prefill_attention` 读取此前 prefix 与当前 raw chunk；若在 attention 前按最终 chunk 长度替换旧窗口，会改变本 chunk query 所见数据。

## 4. finish phase

对 prefill 行，当前 prompt K/V 已确定有效，设 `Cnew=C+q_len`：

1. gather 旧 Recent 中 `[max(S,C-R),min(C,max(S,Cnew-R)))` 到迁移缓冲，最多 R 行。
2. current raw K/V 中 `p>=S && p<max(S,Cnew-R)` 的 original slot 写入 current_history_slots，其余为 `-1`。无需复制整个 raw chunk 来量化。
3. 只保留当前 raw Sink 和最终 Recent；原有仍在最终窗口内的条目保留，其余 positions 清理。
4. state 更新为 Cnew/P=0。按 valid slots 分别压缩旧 Recent 迁移与当前 raw History。下一 chunk 不再次压缩已有 History。

对 target decode/verify 行，要求 `q_len<=M+1`，将当前 raw K/V 写入 pending，state 保持 C，P=q_len。此次不能让 Recent 滚出；由下一 prepare phase 根据真实 C 决定接受范围。

## 5. 错误、图与已实现/待验证边界

- `state.error` 是设备诊断输出。生产路径由 `TargetRuntime.observe_window_errors` 以一步延迟的pinned镜像异步观察（copy与forward同stream先于本步kernel入队，读上一步快照不需同步），发现非零即携带 layer/window_row/request/code fail-closed；probe/关停路径用 `check_window_errors` 显式同步检查。错误码：1 超出已物化KV的commit、2 窗口位置不变量、3 超出pending预算的multi-query、4 迁移块表无效、6 prefix恢复失败。
- 图捕获不能把真实用户请求的窗口推进；dummy 身份/slot无效或独立 dummy 状态必须从真实运行状态隔离。capture/dummy运行使用独立的 capture_row_map/capture_epochs；drafter eager多步（目标配置 `enforce_eager=true`）复用本契约的pending/commit机器，drafter图capture显式fail-closed。
- 捕获前分配 workspace；每次仅原位更新元数据，metadata 与 scratch 地址稳定。各层共享 scratch 要由同一有序执行流管理。
- draft 语义：原生 qwen3_5_mtp drafter 每步为 q_len=1 行（SpecDecoding），raw K/V 走本机 pending 路径；下一轮 prepare 按目标真实接受长度 commit `C=old_C+A`，`A<=old_P`，被拒 draft 位置由后续 CopyRaw 原位覆盖。步骤0 的 C 不含本步正在处理的 bonus token（它是本步query），与 `C<=old_C+old_P` 一致；若真机 metadata 实际违反该界，设备错误码1会被错误传播路径显式暴露，不静默放宽。host侧放行已实现，真机 D04 验证未运行。
- prefix 命中：`prefix_restore_out` 在 phase 0 前运行，对 epoch 变化且 C>0 的行重建窗口——Sink `[0,min(S,C))` 与 Recent `[max(S,C-R),C)`（R=0 时为空）从 staging 池恢复（owner tag 匹配即精确BF16），tag 被逐出的行回退有界 INT2 反量化并计入每层 lossy 计数器，不冒充无损；history 中间区间保持共享页 INT2，不重建。host 镜像 oracle `metadata.restore_window_ranges`。v1 前缀命中按整块只读共享、追加只进新块，无 COW 字节复制路径；backend `swap_blocks/copy_blocks` 保持显式拒绝。
- 抢占恢复：v1 recompute 抢占后请求从头 re-prefill，`resumed=True` 的 epoch 递增已由 `RequestSlots`/`update_lifecycle` 处理，窗口按普通 prefill 重建。
- GDN 拆池后 fresh-page zeroing、真实 capture/replay、全 rank 集成与 NPU 释放验证仍未运行，不能从最终验收清单移除。
