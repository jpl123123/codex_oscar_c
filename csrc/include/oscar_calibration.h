#pragma once
#include <cstdint>

namespace oscar {
// Host metadata only. Activations and every numerical statistic remain on NPU.
struct CalibrationShape {
    int32_t n, hq, hk, d;
    int64_t qs0, qs1, qs2, ks0, ks1, ks2, vs0, vs1, vs2;
};
}
extern "C" {
void oscar_calib_q_moments_launch(void*, const void*, void*, void*, oscar::CalibrationShape);
void oscar_calib_q_cov_launch(void*, const void*, const void*, void*, void*, int, int, int);
void oscar_calib_sst_moments_launch(void*, const void*, const void*, const void*, void*, void*, void*, oscar::CalibrationShape);
void oscar_calib_sst_cov_launch(void*, const void*, const void*, void*, int, int, int);
void oscar_calib_eigh_rhp_launch(void*, const void*, void*, void*, void*, void*, void*, int, int, float);
void oscar_calib_fingerprint_launch(void*, const void*, const void*, const void*, void*, void*, oscar::CalibrationShape, int64_t);
}
