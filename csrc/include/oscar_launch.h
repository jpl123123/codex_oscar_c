#pragma once
#include "oscar_tiling.h"
extern "C" {
void oscar_rotate_launch(void*, const void*, const void*, void*, int, int, bool);
void oscar_store_launch(void*, const void*, const void*, const void*, const void*, const void*, void*,
                        int, int, int, int, int, int64_t, float, float);
void oscar_history_launch(void*, const void*, const void*, const void*, const void*, const void*,
                          const void*, const void*, void*, void*, void*, oscar::HistoryPlan);
void oscar_merge_launch(void*, const void*, const void*, const void*, const void*, const void*, void*, int, int);
void oscar_window_cv_launch(void*, const void*, const void*, const void*, const void*, const void*,
                            const void*, const void*, const void*, const void*, void*, void*, void*, oscar::HistoryPlan);
void oscar_window_state_launch(void*, const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, const void*, const void*, void*, void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, int, int, int, int, int, int, int, int, int, int);
void oscar_zero_launch(void*, void*, const void*, int, int64_t, int64_t);
void oscar_dequant_launch(void* stream, const void* cache, const void* slots,
                          const void* rk, const void* rv, void* kout, void* vout,
                          int n, int hk, int d, int bs, int numblocks, int64_t block_stride);
void oscar_stage_launch(void* stream, const void* k, const void* v, const void* seq,
                        const void* qsl, const void* slots, void* sk, void* sv, void* owner,
                        int batch, int n, int hk, int d, int sink, int recent,
                        int stage_rows, int block_size);
void oscar_restore_launch(void* stream, const void* seq, const void* qsl,
                          const void* rowmap, const void* epochs, const void* bt,
                          void* pos, void* wk, void* wv, void* state,
                          const void* sk, const void* sv, const void* owner,
                          const void* cache, const void* rk, const void* rv, void* lossy,
                          int batch, int hk, int d, int w, int pages, int sink, int recent,
                          int mp, int stage_rows, int block_size, int table_block_size,
                          int num_blocks, int64_t cache_stride);
}
