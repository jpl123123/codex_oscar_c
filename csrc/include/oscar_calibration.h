#pragma once
#include <cstdint>

// The AscendC <<<...>>> launch framework re-declares custom kernel parameter
// types in the global namespace (aclrtlaunch_triple_chevrons_func.h), so this
// struct must not live inside a namespace; the alias below keeps host-side
// oscar::CalibrationShape references unchanged.
struct CalibrationShape {
    int32_t n, hq, hk, d;
    int64_t qs0, qs1, qs2, ks0, ks1, ks2, vs0, vs1, vs2;
};

namespace oscar {
using ::CalibrationShape;
}
extern "C" {
void oscar_calib_q_moments_launch(void*, const void*, void*, void*, oscar::CalibrationShape);
void oscar_calib_q_cov_launch(void*, const void*, const void*, void*, void*, int, int, int);
void oscar_calib_sst_moments_launch(void*, const void*, const void*, const void*, void*, void*, void*, oscar::CalibrationShape);
void oscar_calib_sst_cov_launch(void*, const void*, const void*, void*, int, int, int);
void oscar_calib_eigh_rhp_launch(void*, const void*, void*, void*, void*, void*, void*, int, int, float);
void oscar_calib_fingerprint_launch(void*, const void*, const void*, const void*, void*, void*, oscar::CalibrationShape, int64_t);
}
