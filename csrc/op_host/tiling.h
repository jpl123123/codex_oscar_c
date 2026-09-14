#pragma once
#include "../include/oscar_tiling.h"
namespace oscar {
int CoreCount();
HistoryPlan MakePlan(int n, int hq, int hk, int d, int b, int pages, int block_size,
                     int splits, int max_query_len, int64_t cache_stride, float scale, int table_block_size);
}
