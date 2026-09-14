#pragma once
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// The AscendC <<<...>>> launch framework re-declares custom kernel parameter
// structs in the global namespace AND copies only the struct block verbatim
// into a host stub compiled by the plain system compiler
// (auto_gen/.../host_stub.cpp). Every token inside the struct must therefore
// be self-contained: the tiling byte-image bound is a nested enum so it
// travels with the copied text, and no CANN types may appear as members.
struct HistoryPlan {
    enum : int { kTilingBytes = 512 };
    int32_t n, hq, hk, d, b, pages, num_blocks, block_size, table_block_size, splits, query_tiles, cores;
    int32_t mode, window_capacity, window_rows;
    int64_t cache_block_stride;
    float scale;
    uint64_t tile_bytes, partial_offset;
    uint8_t qk_bytes[kTilingBytes];
    uint8_t pv_bytes[kTilingBytes];
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
