#pragma once
#include "kernel_operator.h"
#include "../include/oscar_tiling.h"

namespace oscar {
using namespace AscendC;
constexpr float kNegInf = -__builtin_inff();
template<class T> __aicore__ inline T Min(T a, T b) { return a < b ? a : b; }
template<class T> __aicore__ inline T Max(T a, T b) { return a > b ? a : b; }
template<class T> __aicore__ inline void Load(LocalTensor<T> dst, GlobalTensor<T> src, int n) {
    DataCopyExtParams p{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(dst, src, p, pad);
    PipeBarrier<PIPE_ALL>();
}
template<class T> __aicore__ inline void Save(GlobalTensor<T> dst, LocalTensor<T> src, int n) {
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams p{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
    DataCopyPad(dst, src, p);
    PipeBarrier<PIPE_ALL>();
}
// Vector elementary functions, including scalar state, execute on AIV.
__aicore__ inline float VExp(LocalTensor<float> scratch, float x) {
    scratch.SetValue(0, x); PipeBarrier<PIPE_ALL>();
    Exp(scratch, scratch, 1); PipeBarrier<PIPE_ALL>();
    return scratch.GetValue(0);
}
__aicore__ inline float VLog(LocalTensor<float> scratch, float x) {
    scratch.SetValue(0, x); PipeBarrier<PIPE_ALL>();
    Ln(scratch, scratch, 1); PipeBarrier<PIPE_ALL>();
    return scratch.GetValue(0);
}
__aicore__ inline void RotateRow(LocalTensor<float> out, LocalTensor<float> row,
                                 LocalTensor<float> tmp, GlobalTensor<float> rot,
                                 LocalTensor<float> input, int d, bool transpose) {
    Duplicate(out, 0.0f, d); PipeBarrier<PIPE_ALL>();
    for (int j=0; j<d; ++j) {
        if (transpose) {
            for (int k=0; k<d; ++k) row.SetValue(k, rot.GetValue(k*d+j));
        } else Load(row, rot[j*d], d);
        PipeBarrier<PIPE_ALL>();
        Muls(tmp, row, input.GetValue(j), d); PipeBarrier<PIPE_V>();
        Add(out, out, tmp, d); PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_ALL>();
}
// NPU-side bitonic sort of one bounded D<=256 row. This deliberately preserves
// PR linear interpolation. It is a correctness implementation to be profiled;
// it is not claimed to be the final high-performance quantile implementation.
__aicore__ inline void ClipRow(LocalTensor<float> x, LocalTensor<float> scratch, int d, float ratio) {
    if (ratio <= 0.0f) return;
    Abs(scratch, x, d); PipeBarrier<PIPE_ALL>();
    for (int k=2; k<=d; k<<=1) for (int j=k>>1; j>0; j>>=1) {
        for (int i=0; i<d; ++i) {
            int other=i^j;
            if (other>i) {
                float a=scratch.GetValue(i), b=scratch.GetValue(other);
                bool ascending=(i&k)==0;
                scratch.SetValue(i, ascending ? Min(a,b) : Max(a,b));
                scratch.SetValue(other, ascending ? Max(a,b) : Min(a,b));
            }
        }
    }
    float u=(d-1)*ratio; int lo=static_cast<int>(u), hi=Min(lo+1,d-1);
    float threshold=scratch.GetValue(lo)+(u-lo)*(scratch.GetValue(hi)-scratch.GetValue(lo));
    Mins(x,x,threshold,d); PipeBarrier<PIPE_V>();
    Maxs(x,x,-threshold,d); PipeBarrier<PIPE_ALL>();
}
}
