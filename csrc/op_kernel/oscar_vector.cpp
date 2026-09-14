#include "oscar_common.h"
#include "../include/oscar_launch.h"

using namespace AscendC;
using namespace oscar;

extern "C" __global__ __aicore__ void oscar_rotate_kernel(
    GM_ADDR x, GM_ADDR r, GM_ADDR y, int rows, int d, bool transpose) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> fbuf, bbuf;
    pipe.InitBuffer(fbuf, 4*kMaxDim*sizeof(float));
    pipe.InitBuffer(bbuf, kMaxDim*sizeof(bfloat16_t));
    auto f=fbuf.Get<float>(); auto bf=bbuf.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> in, out; GlobalTensor<float> rot;
    in.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
    rot.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(r));
    for (int row=GetBlockIdx(); row<rows; row+=GetBlockNum()) {
        Load(bf,in[row*d],d); Cast(f,bf,RoundMode::CAST_NONE,d); PipeBarrier<PIPE_ALL>();
        RotateRow(f[kMaxDim],f[2*kMaxDim],f[3*kMaxDim],rot,f,d,transpose);
        Cast(bf,f[kMaxDim],RoundMode::CAST_RINT,d); Save(out[row*d],bf,d);
    }
}

extern "C" __global__ __aicore__ void oscar_store_kernel(
    GM_ADDR k, GM_ADDR v, GM_ADDR rk, GM_ADDR rv, GM_ADDR slots, GM_ADDR cache,
    int n, int hk, int d, int bs, int numblocks, int64_t block_stride, float kclip, float vclip) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; TBuf<TPosition::VECCALC> fb, bb, packb, halfb;
    pipe.InitBuffer(fb,4*kMaxDim*sizeof(float));
    pipe.InitBuffer(bb,kMaxDim*sizeof(bfloat16_t));
    pipe.InitBuffer(packb,AlignBuffer(2*(kMaxDim/4+4)));
    pipe.InitBuffer(halfb,32);
    auto f=fb.Get<float>(); auto bf=bb.Get<bfloat16_t>();
    auto packed=packb.Get<uint8_t>(); auto h=halfb.Get<half>(); auto hu=h.ReinterpretCast<uint16_t>();
    GlobalTensor<bfloat16_t> kg,vg; GlobalTensor<float> kr,vr; GlobalTensor<int64_t> sg;
    GlobalTensor<uint8_t> dst;
    kg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
    vg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(v));
    kr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rk));
    vr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rv));
    sg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cache));
    int bytes=d/4, region=bytes+4, slot_bytes=2*region;
    for (int row=GetBlockIdx(); row<n*hk; row+=GetBlockNum()) {
        int token=row/hk, head=row%hk; int64_t slot=sg.GetValue(token);
        if (slot<0 || slot>=int64_t(numblocks)*bs) continue;
        for (int side=0; side<2; ++side) {
            Load(bf,(side==0?kg:vg)[row*d],d);
            Cast(f,bf,RoundMode::CAST_NONE,d); PipeBarrier<PIPE_ALL>();
            RotateRow(f[kMaxDim],f[2*kMaxDim],f[3*kMaxDim],side==0?kr:vr,f,d,false);
            auto x=f[kMaxDim]; ClipRow(x,f[2*kMaxDim],d,side==0?kclip:vclip);
            float low=x.GetValue(0), high=low;
            for(int j=1;j<d;++j) { low=Min(low,x.GetValue(j)); high=Max(high,x.GetValue(j)); }
            float sc=Max((high-low)/3.0f,1e-8f);
            h.SetValue(0,static_cast<half>(sc)); h.SetValue(1,static_cast<half>(low));
            sc=static_cast<float>(h.GetValue(0)); float z=static_cast<float>(h.GetValue(1));
            // PR's 1e-8 floor can underflow when serialized as FP16. Explicit
            // extension: use the smallest FP16 subnormal, rather than divide by 0.
            if(sc==0.0f) { hu.SetValue(0,1); sc=static_cast<float>(h.GetValue(0)); }
            for(int b=0;b<bytes;++b) {
                uint8_t byte=0;
                for(int j=0;j<4;++j) {
                    int q=static_cast<int>((x.GetValue(4*b+j)-z)/sc+0.5f);
                    q=Max(0,Min(3,q)); byte|=static_cast<uint8_t>(q<<(2*j));
                }
                packed.SetValue(side*region+b,byte);
            }
            uint16_t sbits=hu.GetValue(0), zbits=hu.GetValue(1);
            packed.SetValue(side*region+bytes,sbits&255);
            packed.SetValue(side*region+bytes+1,sbits>>8);
            packed.SetValue(side*region+bytes+2,zbits&255);
            packed.SetValue(side*region+bytes+3,zbits>>8);
        }
        int64_t offset=(slot/bs)*block_stride+(slot%bs)*hk*slot_bytes+head*slot_bytes;
        Save(dst[offset],packed,slot_bytes);
    }
}

// Merge History (rotated V coordinates) and BF16 window/current-chunk outputs.
// Input/output buffers are caller-owned; no allocation or CPU synchronization.
extern "C" __global__ __aicore__ void oscar_merge_kernel(
    GM_ADDR hist,GM_ADDR hlse,GM_ADDR win,GM_ADDR wlse,GM_ADDR rv,GM_ADDR output,int rows,int d) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe p; TBuf<TPosition::VECCALC> fb,bb;
    p.InitBuffer(fb,6*kMaxDim*sizeof(float)); p.InitBuffer(bb,kMaxDim*sizeof(bfloat16_t));
    auto f=fb.Get<float>(); auto bf=bb.Get<bfloat16_t>();
    GlobalTensor<float> hg,lg,wg,wl,rot; GlobalTensor<bfloat16_t> out;
    hg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hist)); lg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hlse));
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(win)); wl.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wlse));
    rot.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rv)); out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
    for(int row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
        float a=lg.GetValue(row), b=wl.GetValue(row), m=Max(a,b);
        if(m==kNegInf) { Duplicate(bf,static_cast<bfloat16_t>(0),d); Save(out[row*d],bf,d); continue; }
        float ea=(a==kNegInf?0:VExp(f[5*kMaxDim],a-m));
        float eb=(b==kNegInf?0:VExp(f[5*kMaxDim],b-m)); float norm=ea+eb;
        Load(f,hg[row*d],d);
        RotateRow(f[kMaxDim],f[2*kMaxDim],f[3*kMaxDim],rot,f,d,true);
        Load(f[4*kMaxDim],wg[row*d],d);
        Muls(f[kMaxDim],f[kMaxDim],ea/norm,d); PipeBarrier<PIPE_V>();
        Muls(f[4*kMaxDim],f[4*kMaxDim],eb/norm,d); PipeBarrier<PIPE_V>();
        Add(f,f[kMaxDim],f[4*kMaxDim],d); PipeBarrier<PIPE_V>();
        Cast(bf,f,RoundMode::CAST_RINT,d); Save(out[row*d],bf,d);
    }
}

extern "C" __global__ __aicore__ void oscar_zero_kernel(GM_ADDR data,GM_ADDR ids,int count,int64_t stride,int64_t numblocks) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe p; TBuf<TPosition::VECCALC> zbuf; p.InitBuffer(zbuf,4096);
    auto z=zbuf.Get<uint8_t>(); Duplicate(z.ReinterpretCast<uint32_t>(),uint32_t(0),1024);
    GlobalTensor<uint8_t> dst; GlobalTensor<int64_t> blocks;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(data)); blocks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ids));
    for(int i=GetBlockIdx();i<count;i+=GetBlockNum()) {
        int64_t block=blocks.GetValue(i); if(block<0 || block>=numblocks) continue;
        for(int64_t offset=0;offset<stride;offset+=4096) Save(dst[block*stride+offset],z,static_cast<int>(Min<int64_t>(4096,stride-offset)));
    }
}

extern "C" void oscar_rotate_launch(void* stream,const void* x,const void* r,void* y,int rows,int d,bool tr) {
    oscar_rotate_kernel<<<32,nullptr,stream>>>((GM_ADDR)x,(GM_ADDR)r,(GM_ADDR)y,rows,d,tr);
}
extern "C" void oscar_store_launch(void* stream,const void* k,const void* v,const void* rk,const void* rv,const void* slots,void* cache,
                                    int n,int hk,int d,int bs,int nb,int64_t stride,float kc,float vc) {
    oscar_store_kernel<<<32,nullptr,stream>>>((GM_ADDR)k,(GM_ADDR)v,(GM_ADDR)rk,(GM_ADDR)rv,(GM_ADDR)slots,(GM_ADDR)cache,n,hk,d,bs,nb,stride,kc,vc);
}
extern "C" void oscar_merge_launch(void* stream,const void* h,const void* hl,const void* w,const void* wl,const void* rv,void* o,int rows,int d) {
    oscar_merge_kernel<<<32,nullptr,stream>>>((GM_ADDR)h,(GM_ADDR)hl,(GM_ADDR)w,(GM_ADDR)wl,(GM_ADDR)rv,(GM_ADDR)o,rows,d);
}
extern "C" void oscar_zero_launch(void* stream,void* data,const void* ids,int count,int64_t stride,int64_t numblocks) {
    oscar_zero_kernel<<<32,nullptr,stream>>>((GM_ADDR)data,(GM_ADDR)ids,count,stride,numblocks);
}
