#pragma once
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// The AscendC <<<...>>> launch framework re-declares custom kernel parameter
// types in the global namespace (aclrtlaunch_triple_chevrons_func.h), so this
// struct must not live inside a namespace. oscar::HistoryPlan below keeps the
// existing host-side references unchanged.
struct HistoryPlan {
    int32_t n, hq, hk, d, b, pages, num_blocks, block_size, table_block_size, splits, query_tiles, cores;
    int32_t mode, window_capacity, window_rows;
    int64_t cache_block_stride;
    float scale;
    uint64_t tile_bytes, partial_offset;
    AscendC::tiling::TCubeTiling qk;
    AscendC::tiling::TCubeTiling pv;
};

namespace oscar {
using ::HistoryPlan;
constexpr int kQueryTile = 16;
constexpr int kKvTile = 32;
constexpr int kMaxDim = 256;
constexpr uint64_t kAlign = 512;
inline constexpr uint64_t Align(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }
// One bounded GM tile bridge per physical Cube core. The packed history is
// read once by Vector and shared by all 16 query rows before this is reused.
inline constexpr uint64_t TileBytes(int d) {
    return Align((kQueryTile * d + 2 * kKvTile * d + kQueryTile * kKvTile) * 2
                 + (kQueryTile * kKvTile + kQueryTile * d) * 4);
}
inline constexpr uint64_t WorkspaceBytes(int n, int hq, int d, int splits, int cores) {
    return TileBytes(d) * cores + Align(uint64_t(n) * hq * splits * (d + 1) * sizeof(float));
}
}
