#include "tiling.h"
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include "acl/acl_rt.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace oscar {
int CoreCount() {
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform || platform->GetCoreNumAic() < 1) throw std::runtime_error("Cannot discover Ascend Cube cores");
    return platform->GetCoreNumAic();
}
static AscendC::tiling::TCubeTiling CubePlan(int m, int n, int k, bool trans_b) {
    matmul_tiling::MultiCoreMatmulTiling mm;
    auto* p = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ub = 0, l1 = 0, l0c = 0;
    p->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    p->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    p->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    mm.SetBufferSpace(l1, l0c, ub);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16, trans_b);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm.SetBias(false);
    mm.SetDim(1);
    mm.SetShape(m, n, k);
    mm.SetOrgShape(m, n, k);
    mm.SetFixSplit(m, n, k);
    AscendC::tiling::TCubeTiling out{};
    if (mm.GetTiling(out) == -1) throw std::runtime_error("OSCAR Cube tiling API rejected tile");
    return out;
}
struct CubePlans {
    AscendC::tiling::TCubeTiling qk;
    AscendC::tiling::TCubeTiling pv;
};
static CubePlans CachedCubePlans(int d) {
    // These two fixed tiles depend on hardware and D only. Never cache request
    // lengths, slots, pointers, scalar values read from NPU, or mutable KV state.
    // All map access is serialized; successful entries are immutable copies.
    // Construction errors are not inserted, so a failed tiling is never reused.
    using Key=std::tuple<int32_t,int32_t,int,uint64_t,uint64_t,uint64_t>;
    static std::mutex mutex;
    static std::map<Key,CubePlans> cache;
    int32_t device=-1;
    if(aclrtGetDevice(&device)!=ACL_SUCCESS) throw std::runtime_error("Cannot identify current NPU for tiling cache");
    auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();
    if(!platform) throw std::runtime_error("Cannot query SoC for tiling cache");
    uint64_t ub=0,l1=0,l0c=0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,l0c);
    const Key key{device,static_cast<int32_t>(platform->GetSocVersion()),d,ub,l1,l0c};
    std::lock_guard<std::mutex> lock(mutex);
    auto found=cache.find(key);
    if(found!=cache.end()) return found->second;
    CubePlans plans{CubePlan(kQueryTile,kKvTile,d,true),CubePlan(kQueryTile,d,kKvTile,false)};
    cache.emplace(key,plans);
    return plans;
}
HistoryPlan MakePlan(int n, int hq, int hk, int d, int b, int pages, int block_size,
                     int splits, int max_query_len, int64_t stride, float scale, int table_block_size) {
    HistoryPlan p{};
    p.n=n; p.hq=hq; p.hk=hk; p.d=d; p.b=b; p.pages=pages; p.block_size=block_size;
    p.table_block_size=table_block_size;
    p.splits=splits; p.query_tiles=(max_query_len+kQueryTile-1)/kQueryTile; p.cores=CoreCount();
    p.cache_block_stride=stride; p.scale=scale; p.tile_bytes=TileBytes(d);
    p.partial_offset=p.tile_bytes*p.cores;
    const auto cubes=CachedCubePlans(d);
    p.qk=cubes.qk;
    p.pv=cubes.pv;
    return p;
}
}
