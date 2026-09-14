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
void oscar_dequant_launch(void*, const void*, const void*, const void*, const void*,
                          void*, void*, int, int, int, int, int, int64_t);
void oscar_stage_launch(void*, const void*, const void*, const void*, const void*,
                        const void*, void*, void*, void*, int, int, int, int, int, int, int, int);
void oscar_restore_launch(void*, const void*, const void*, const void*, const void*,
                          const void*, void*, void*, void*, void*, const void*, const void*,
                          const void*, const void*, const void*, const void*, void*,
                          int, int, int, int, int, int, int, int, int, int, int, int64_t);
}
