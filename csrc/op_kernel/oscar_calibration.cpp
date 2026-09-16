#include "oscar_common.h"
#include "../include/oscar_calibration.h"

using namespace AscendC;
using namespace oscar;

namespace {
__aicore__ inline float AbsScalar(float x) { return x<0?-x:x; }
__aicore__ inline bool Finite(float x) { return x==x && AbsScalar(x)!=__builtin_inff(); }
__aicore__ inline float Root(LocalTensor<float> tmp,float value) {
    // fp32 vector Sqrt also processes 256-bit lanes; evaluating one scalar
    // with count=1 is undefined on this backend, so widen to a full lane.
    Duplicate(tmp,value,8);PipeBarrier<PIPE_ALL>();Sqrt(tmp,tmp,8);PipeBarrier<PIPE_ALL>();return tmp.GetValue(0);
}
__aicore__ inline float Sum(LocalTensor<float> src,LocalTensor<float> scratch,int d) {
    ReduceSum(scratch,src,scratch[32],d);PipeBarrier<PIPE_ALL>();return scratch.GetValue(0);
}
__aicore__ inline void RawRow(LocalTensor<bfloat16_t> bf,LocalTensor<float> f,
 GlobalTensor<bfloat16_t> source,int token,int head,int d,int64_t s0,int64_t s1,int64_t s2) {
    int64_t offset=int64_t(token)*s0+int64_t(head)*s1;
    if(s2==1) Load(bf,source[offset],d);
    else {
        for(int j=0;j<d;++j) bf.SetValue(j,source.GetValue(offset+int64_t(j)*s2));
        PipeBarrier<PIPE_ALL>();
    }
    Cast(f,bf,RoundMode::CAST_NONE,d);PipeBarrier<PIPE_ALL>();
}
// Kahan vector accumulation reduces the extra FP32 summation error within a
// captured chunk; the stored moments themselves are FP32, not paper FP64.
__aicore__ inline void Kahan(LocalTensor<float> sum,LocalTensor<float> correction,
 LocalTensor<float> value,LocalTensor<float> temp,int d) {
    Sub(value,value,correction,d);PipeBarrier<PIPE_V>();
    Add(temp,sum,value,d);PipeBarrier<PIPE_V>();
    Sub(correction,temp,sum,d);PipeBarrier<PIPE_V>();
    Sub(correction,correction,value,d);PipeBarrier<PIPE_V>();
    Adds(sum,temp,0.0f,d);PipeBarrier<PIPE_ALL>();
}
}

// Each covariance row has one writer. Chunks must be enqueued on the same
// stream for one layer's moments; there are no atomics or host Q/K/V reads.
extern "C" __global__ __aicore__ void oscar_calib_q_moments_kernel(
 GM_ADDR q,GM_ADDR moments,GM_ADDR counts,CalibrationShape s) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;TBuf<TPosition::VECCALC> fb,bb;
    pipe.InitBuffer(fb,5*kMaxDim*4);pipe.InitBuffer(bb,kMaxDim*2);
    auto f=fb.Get<float>();auto bf=bb.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> query;GlobalTensor<float> out;GlobalTensor<int64_t> count;
    query.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(q));out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(moments));
    count.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(counts));int group=s.hq/s.hk;
    for(int task=GetBlockIdx();task<s.hk*s.d;task+=GetBlockNum()) {
        int head=task/s.d,row=task%s.d;auto acc=f[kMaxDim],comp=f[2*kMaxDim],term=f[3*kMaxDim],tmp=f[4*kMaxDim];
        Load(acc,out[int64_t(task)*s.d],s.d);Duplicate(comp,0.0f,s.d);PipeBarrier<PIPE_ALL>();
        for(int token=0;token<s.n;++token) for(int g=0;g<group;++g) {
            RawRow(bf,f,query,token,head*group+g,s.d,s.qs0,s.qs1,s.qs2);
            Muls(term,f,f.GetValue(row),s.d);PipeBarrier<PIPE_V>();Kahan(acc,comp,term,tmp,s.d);
        }
        Save(out[int64_t(task)*s.d],acc,s.d);
        if(row==0) count.SetValue(head,count.GetValue(head)+int64_t(s.n)*group);
    }
}

extern "C" __global__ __aicore__ void oscar_calib_q_cov_kernel(
 GM_ADDR moments,GM_ADDR counts,GM_ADDR perhead,GM_ADDR global,int heads,int d,int divisor) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> src,per,out;GlobalTensor<int64_t> count;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(moments));count.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(counts));
    per.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(perhead));out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(global));
    for(int i=GetBlockIdx();i<d*d;i+=GetBlockNum()) {
        int row=i/d,col=i%d;float total=0;
        for(int h=0;h<heads;++h) {
            int64_t n=count.GetValue(h);float value=__builtin_nanf("");
            if(n>0) value=0.5f*(src.GetValue((h*d+row)*d+col)+src.GetValue((h*d+col)*d+row))/float(n);
            per.SetValue((h*d+row)*d+col,value);total+=value;
        }
        out.SetValue(i,total/float(divisor));
    }
}

extern "C" __global__ __aicore__ void oscar_calib_weights_kernel(
 GM_ADDR key,GM_ADDR cq,GM_ADDR weights,CalibrationShape s) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;TBuf<TPosition::VECCALC> fb,bb;
    pipe.InitBuffer(fb,4*kMaxDim*4+512);pipe.InitBuffer(bb,kMaxDim*2);
    auto f=fb.Get<float>();auto bf=bb.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> k;GlobalTensor<float> cov,out;
    k.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(key));cov.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cq));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    for(int task=GetBlockIdx();task<s.n*s.hk;task+=GetBlockNum()) {
        int token=task/s.hk,head=task%s.hk;RawRow(bf,f,k,token,head,s.d,s.ks0,s.ks1,s.ks2);
        float weight=0,correction=0;
        for(int row=0;row<s.d;++row) {
            Load(f[kMaxDim],cov[(head*s.d+row)*s.d],s.d);
            Mul(f[2*kMaxDim],f,f[kMaxDim],s.d);PipeBarrier<PIPE_V>();
            float term=f.GetValue(row)*Sum(f[2*kMaxDim],f[3*kMaxDim],s.d);
            float y=term-correction,t=weight+y;correction=(t-weight)-y;weight=t;
        }
        // Preserve the actual signed/finite result for the numerical probe.
        // The next kernel rejects every negative weight rather than projecting
        // a bad per-head covariance onto a different calibration algorithm.
        out.SetValue(task,weight);
    }
}

extern "C" __global__ __aicore__ void oscar_calib_sst_moments_kernel(
 GM_ADDR value,GM_ADDR weights,GM_ADDR weighted,GM_ADDR weight_sum,CalibrationShape s) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;TBuf<TPosition::VECCALC> fb,bb;pipe.InitBuffer(fb,5*kMaxDim*4);pipe.InitBuffer(bb,kMaxDim*2);
    auto f=fb.Get<float>();auto bf=bb.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> v;GlobalTensor<float> w,out,den;
    v.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(value));w.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weighted));den.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight_sum));
    for(int task=GetBlockIdx();task<s.hk*s.d;task+=GetBlockNum()) {
        int head=task/s.d,row=task%s.d;auto acc=f[kMaxDim],comp=f[2*kMaxDim],term=f[3*kMaxDim],tmp=f[4*kMaxDim];
        Load(acc,out[int64_t(task)*s.d],s.d);Duplicate(comp,0.0f,s.d);PipeBarrier<PIPE_ALL>();
        float sum=0,correction=0;if(row==0)sum=den.GetValue(head);
        for(int token=0;token<s.n;++token) {
            float weight=w.GetValue(token*s.hk+head);
            // paper sqrt(weight) is undefined for a negative weight. Preserve
            // that failure through covariance/eigensolver validation, including
            // tiny FP32 cancellation cases; no silent clamp is accepted.
            if(!Finite(weight)||weight<0)weight=__builtin_nanf("");
            RawRow(bf,f,v,token,head,s.d,s.vs0,s.vs1,s.vs2);
            Muls(term,f,weight*f.GetValue(row),s.d);PipeBarrier<PIPE_V>();Kahan(acc,comp,term,tmp,s.d);
            if(row==0) {float y=weight-correction,t=sum+y;correction=(t-sum)-y;sum=t;}
        }
        Save(out[int64_t(task)*s.d],acc,s.d);if(row==0)den.SetValue(head,sum);
    }
}

extern "C" __global__ __aicore__ void oscar_calib_sst_cov_kernel(
 GM_ADDR weighted,GM_ADDR weights,GM_ADDR global,int heads,int d,int divisor) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> src,den,out;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weighted));den.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(global));
    for(int i=GetBlockIdx();i<d*d;i+=GetBlockNum()) {
        int row=i/d,col=i%d;float total=0;
        for(int h=0;h<heads;++h) {
            float divisor_h=Max(den.GetValue(h),1e-12f);
            total+=0.5f*(src.GetValue((h*d+row)*d+col)+src.GetValue((h*d+col)*d+row))/divisor_h;
        }
        out.SetValue(i,total/float(divisor));
    }
}

extern "C" __global__ __aicore__ void oscar_calib_fingerprint_partial_kernel(
 GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR partial,CalibrationShape s,int64_t token_offset) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<uint16_t> source[3];GlobalTensor<int64_t> out;
    source[0].SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(q));source[1].SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(k));
    source[2].SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(v));out.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(partial));
    for(int task=GetBlockIdx();task<3*32;task+=GetBlockNum()) {
        int side=task/32,lane=task%32,heads=side==0?s.hq:s.hk;
        int64_t s0=side==0?s.qs0:(side==1?s.ks0:s.vs0),s1=side==0?s.qs1:(side==1?s.ks1:s.vs1),s2=side==0?s.qs2:(side==1?s.ks2:s.vs2);
        uint64_t sum=0,weighted=0;
        for(int64_t i=lane;i<int64_t(s.n)*heads*s.d;i+=32) {
            int token=i/(heads*s.d),head=(i/s.d)%heads,dim=i%s.d;
            uint64_t bits=source[side].GetValue(int64_t(token)*s0+int64_t(head)*s1+int64_t(dim)*s2);
            uint64_t position=uint64_t(token_offset)*heads*s.d+uint64_t(i)+1;
            uint64_t mix=(position*0x9e3779b97f4a7c15ULL)^(position>>13);
            sum+=bits;weighted+=bits*mix;
        }
        out.SetValue(side*32+lane,static_cast<int64_t>(sum));out.SetValue((side+3)*32+lane,static_cast<int64_t>(weighted));
    }
}
extern "C" __global__ __aicore__ void oscar_calib_fingerprint_reduce_kernel(GM_ADDR partial,GM_ADDR fingerprint) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<int64_t> in,out;in.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(partial));out.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(fingerprint));
    for(int side=GetBlockIdx();side<6;side+=GetBlockNum()) {
        uint64_t sum=static_cast<uint64_t>(out.GetValue(side));
        for(int lane=0;lane<32;++lane)sum+=static_cast<uint64_t>(in.GetValue(side*32+lane));
        out.SetValue(side,static_cast<int64_t>(sum));
    }
}

extern "C" void oscar_calib_q_moments_launch(void* stream,const void* q,void* m,void* c,CalibrationShape s) {
    // Bound vector/DMA iterations per physical AIV even when the model runner
    // presents a 16K prefill. This changes neither the global count nor when
    // normalization occurs; each tile accumulates into the same moments.
    int rows_per_core=(s.hk*s.d+31)/32,group=s.hq/s.hk;
    int limit=4096/(rows_per_core*group);limit=limit<1?1:(limit>128?128:limit);
    for(int begin=0;begin<s.n;begin+=limit) {
        CalibrationShape tile=s;tile.n=s.n-begin<limit?s.n-begin:limit;
        auto ptr=(GM_ADDR)q+int64_t(begin)*s.qs0*2;
        oscar_calib_q_moments_kernel<<<32,nullptr,stream>>>(ptr,(GM_ADDR)m,(GM_ADDR)c,OSCAR_LAUNCH_ARG(tile));
    }
}
extern "C" void oscar_calib_q_cov_launch(void* stream,const void* m,const void* c,void* p,void* g,int h,int d,int divisor) {
    oscar_calib_q_cov_kernel<<<32,nullptr,stream>>>((GM_ADDR)m,(GM_ADDR)c,(GM_ADDR)p,(GM_ADDR)g,h,d,divisor);
}
extern "C" void oscar_calib_sst_moments_launch(void* stream,const void* k,const void* v,const void* cq,void* m,void* den,void* ws,CalibrationShape s) {
    int rows_per_core=(s.hk*s.d+31)/32,limit=4096/rows_per_core;limit=limit<1?1:(limit>128?128:limit);
    for(int begin=0;begin<s.n;begin+=limit) {
        CalibrationShape tile=s;tile.n=s.n-begin<limit?s.n-begin:limit;
        auto kp=(GM_ADDR)k+int64_t(begin)*s.ks0*2,vp=(GM_ADDR)v+int64_t(begin)*s.vs0*2;
        auto wp=(GM_ADDR)ws+int64_t(begin)*s.hk*4;
        oscar_calib_weights_kernel<<<32,nullptr,stream>>>(kp,(GM_ADDR)cq,wp,OSCAR_LAUNCH_ARG(tile));
        oscar_calib_sst_moments_kernel<<<32,nullptr,stream>>>(vp,wp,(GM_ADDR)m,(GM_ADDR)den,OSCAR_LAUNCH_ARG(tile));
    }
}
extern "C" void oscar_calib_sst_cov_launch(void* stream,const void* m,const void* den,void* out,int h,int d,int divisor) {
    oscar_calib_sst_cov_kernel<<<32,nullptr,stream>>>((GM_ADDR)m,(GM_ADDR)den,(GM_ADDR)out,h,d,divisor);
}
extern "C" void oscar_calib_fingerprint_launch(void* stream,const void* q,const void* k,const void* v,void* fp,void* partial,CalibrationShape s,int64_t offset) {
    int limit=(32768*32)/(s.hq*s.d);limit=limit<1?1:(limit>128?128:limit);
    for(int begin=0;begin<s.n;begin+=limit) {
        CalibrationShape tile=s;tile.n=s.n-begin<limit?s.n-begin:limit;
        auto qp=(GM_ADDR)q+int64_t(begin)*s.qs0*2,kp=(GM_ADDR)k+int64_t(begin)*s.ks0*2,vp=(GM_ADDR)v+int64_t(begin)*s.vs0*2;
        oscar_calib_fingerprint_partial_kernel<<<32,nullptr,stream>>>(qp,kp,vp,(GM_ADDR)partial,OSCAR_LAUNCH_ARG(tile),offset+begin);
        oscar_calib_fingerprint_reduce_kernel<<<6,nullptr,stream>>>((GM_ADDR)partial,(GM_ADDR)fp);
    }
}

namespace {
__aicore__ inline void Pair(int d,int round,int pair,int& p,int& q) {
    if(pair==0) {p=d-1;q=round%(d-1);}
    else {p=(round+pair)%(d-1);q=(round+d-1-pair)%(d-1);}
    if(p>q) {int t=p;p=q;q=t;}
}
__aicore__ inline void Coeff(GlobalTensor<float> a,int d,int p,int q,LocalTensor<float> tmp,float& c,float& s) {
    float app=a.GetValue(p*d+p),aqq=a.GetValue(q*d+q),apq=a.GetValue(p*d+q);
    if(apq==0) {c=1;s=0;return;}
    float delta=0.5f*(aqq-app),scale=Max(AbsScalar(delta),AbsScalar(apq));
    if(scale==0) {c=1;s=0;return;}
    float x=delta/scale,y=apq/scale;
    float hyp=scale*Root(tmp,x*x+y*y);
    float denominator=delta+(delta<0?-hyp:hyp);
    float t=apq/denominator;c=1.0f/Root(tmp,1.0f+t*t);s=t*c;
}
}

extern "C" __global__ __aicore__ void oscar_calib_jacobi_init_kernel(
 GM_ADDR covariance,GM_ADDR vectors,GM_ADDR workspace,GM_ADDR diagnostics,int d,float tolerance) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if(GetBlockIdx()!=0||GetSubBlockIdx()!=0)return;
    TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,2*kMaxDim*4+512);auto f=buf.Get<float>();
    GlobalTensor<float> cov,u,a,diag;
    cov.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(covariance));u.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vectors));
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));diag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diagnostics));
    float norm=0,off=0,maxdiag=0;int status=0;
    for(int i=0;i<8;++i)diag.SetValue(i,0);
    for(int row=0;row<d;++row) {
        for(int col=0;col<d;++col) {
            float value=0.5f*(cov.GetValue(row*d+col)+cov.GetValue(col*d+row));
            if(!Finite(value))status=2;
            f.SetValue(col,value);norm+=value*value;if(row!=col)off+=value*value;
            else maxdiag=Max(maxdiag,AbsScalar(value));
        }
        Save(a[row*d],f,d);Duplicate(f,0.0f,d);PipeBarrier<PIPE_ALL>();f.SetValue(row,1.0f);Save(u[row*d],f,d);
    }
    if(!Finite(norm)||!Finite(off))status=2;else if(norm==0)status=3;
    float relative=status==0?Root(f[2*kMaxDim],off/norm):__builtin_inff();
    diag.SetValue(0,status==0&&relative<=tolerance?1:0);diag.SetValue(2,relative);diag.SetValue(3,maxdiag);diag.SetValue(7,status);
}

// Parallel round-robin Jacobi: disjoint column pairs first write B=A*J and
// U=U*J. A remains untouched until the following kernel, so every coefficient
// comes from the same matrix. Each kernel is bounded to one round, not a whole
// D^3*sweeps solve, avoiding a single long device watchdog interval.
extern "C" __global__ __aicore__ void oscar_calib_jacobi_right_kernel(
 GM_ADDR workspace,GM_ADDR vectors,GM_ADDR diagnostics,int d,int round) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> a,b,u,diag;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)+d*d);
    u.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vectors));diag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diagnostics));
    if(diag.GetValue(0)==1||diag.GetValue(7)!=0)return;
    TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,9*kMaxDim*4+512);auto f=buf.Get<float>();
    for(int pair=GetBlockIdx();pair<d/2;pair+=GetBlockNum()) {
        int p,q;Pair(d,round,pair,p,q);float c,s;Coeff(a,d,p,q,f[9*kMaxDim],c,s);
        for(int row=0;row<d;++row) {
            f.SetValue(row,a.GetValue(row*d+p));f[kMaxDim].SetValue(row,a.GetValue(row*d+q));
            f[4*kMaxDim].SetValue(row,u.GetValue(row*d+p));f[5*kMaxDim].SetValue(row,u.GetValue(row*d+q));
        }
        PipeBarrier<PIPE_ALL>();
        for(int side=0;side<2;++side) {
            auto x=f[side*4*kMaxDim],y=f[(side*4+1)*kMaxDim];
            auto xp=f[(side*4+2)*kMaxDim],yq=f[(side*4+3)*kMaxDim],tmp=f[8*kMaxDim];
            Muls(xp,x,c,d);Muls(tmp,y,-s,d);PipeBarrier<PIPE_V>();Add(xp,xp,tmp,d);PipeBarrier<PIPE_V>();
            Muls(yq,x,s,d);Muls(tmp,y,c,d);PipeBarrier<PIPE_V>();Add(yq,yq,tmp,d);PipeBarrier<PIPE_ALL>();
            for(int row=0;row<d;++row) {
                (side==0?b:u).SetValue(row*d+p,xp.GetValue(row));
                (side==0?b:u).SetValue(row*d+q,yq.GetValue(row));
            }
        }
    }
}
extern "C" __global__ __aicore__ void oscar_calib_jacobi_left_kernel(
 GM_ADDR workspace,GM_ADDR diagnostics,int d,int round) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> a,b,diag;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)+d*d);
    diag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diagnostics));if(diag.GetValue(0)==1||diag.GetValue(7)!=0)return;
    TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,5*kMaxDim*4+512);auto f=buf.Get<float>();
    for(int pair=GetBlockIdx();pair<d/2;pair+=GetBlockNum()) {
        int p,q;Pair(d,round,pair,p,q);float c,s;Coeff(a,d,p,q,f[5*kMaxDim],c,s);
        // Only this pair writes A rows p and q; its three old coefficient
        // entries cannot be overwritten by another pair while Coeff reads them.
        Load(f,b[p*d],d);Load(f[kMaxDim],b[q*d],d);
        Muls(f[2*kMaxDim],f,c,d);Muls(f[4*kMaxDim],f[kMaxDim],-s,d);PipeBarrier<PIPE_V>();
        Add(f[2*kMaxDim],f[2*kMaxDim],f[4*kMaxDim],d);PipeBarrier<PIPE_V>();
        Muls(f[3*kMaxDim],f,s,d);Muls(f[4*kMaxDim],f[kMaxDim],c,d);PipeBarrier<PIPE_V>();
        Add(f[3*kMaxDim],f[3*kMaxDim],f[4*kMaxDim],d);PipeBarrier<PIPE_ALL>();
        f[2*kMaxDim].SetValue(q,0.0f);f[3*kMaxDim].SetValue(p,0.0f);
        Save(a[p*d],f[2*kMaxDim],d);Save(a[q*d],f[3*kMaxDim],d);
    }
}
extern "C" __global__ __aicore__ void oscar_calib_jacobi_check_kernel(
 GM_ADDR workspace,GM_ADDR diagnostics,int d,int sweep,float tolerance) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(GetBlockIdx()!=0||GetSubBlockIdx()!=0)return;
    GlobalTensor<float> a,diag;a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    diag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diagnostics));if(diag.GetValue(0)==1||diag.GetValue(7)!=0)return;
    TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,512);auto tmp=buf.Get<float>();
    float norm=0,off=0,maxdiag=0;int status=0;
    for(int row=0;row<d;++row) {
        float value=a.GetValue(row*d+row);norm+=value*value;maxdiag=Max(maxdiag,AbsScalar(value));
        for(int col=row+1;col<d;++col) {
            float x=0.5f*(a.GetValue(row*d+col)+a.GetValue(col*d+row));
            a.SetValue(row*d+col,x);a.SetValue(col*d+row,x);off+=2*x*x;
        }
    }
    norm+=off;if(!Finite(norm)||!Finite(off))status=2;else if(norm==0)status=3;
    float relative=status==0?Root(tmp,off/norm):__builtin_inff();
    diag.SetValue(0,status==0&&relative<=tolerance?1:0);diag.SetValue(1,sweep);diag.SetValue(2,relative);diag.SetValue(3,maxdiag);diag.SetValue(7,status);
}

extern "C" __global__ __aicore__ void oscar_calib_eigen_sort_kernel(
 GM_ADDR workspace,GM_ADDR vectors,GM_ADDR eigenvalues,GM_ADDR diagnostics,int d) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(GetBlockIdx()!=0||GetSubBlockIdx()!=0)return;
    GlobalTensor<float> a,b,u,ev,diag;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)+d*d);
    u.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vectors));ev.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eigenvalues));
    diag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diagnostics));if(diag.GetValue(7)!=0)return;
    TPipe pipe;TBuf<TPosition::VECCALC> buf,ibuf;pipe.InitBuffer(buf,3*kMaxDim*4);pipe.InitBuffer(ibuf,kMaxDim*4);
    auto f=buf.Get<float>();auto idx=ibuf.Get<int32_t>();
    for(int i=0;i<d;++i){f.SetValue(i,a.GetValue(i*d+i));idx.SetValue(i,i);}
    // Stable ascending eigenvalues, as torch.linalg.eigh's documented order.
    for(int i=1;i<d;++i) {
        int selected=idx.GetValue(i),j=i-1;
        while(j>=0&&f.GetValue(idx.GetValue(j))>f.GetValue(selected)) {idx.SetValue(j+1,idx.GetValue(j));--j;}
        idx.SetValue(j+1,selected);
    }
    for(int col=0;col<d;++col) {
        int old=idx.GetValue(col);ev.SetValue(col,f.GetValue(old));float maximum=0,sign=1;
        for(int row=0;row<d;++row) {float value=u.GetValue(row*d+old);if(AbsScalar(value)>maximum){maximum=AbsScalar(value);sign=value<0?-1:1;}}
        // This deterministic sign convention is recorded in artifact metadata;
        // the paper's eigensolver signs are unspecified, not byte-identical.
        for(int row=0;row<d;++row)b.SetValue(row*d+col,sign*u.GetValue(row*d+old));
    }
    for(int row=0;row<d;++row){Load(f,b[row*d],d);Save(u[row*d],f,d);}
    // Pbr = I[:,perm], perm[bitreverse(i)] = argsort(lambda,descending)[i].
    // Descending ties retain their ascending column index for reproducibility.
    for(int i=0;i<d;++i)idx.SetValue(i,i);
    for(int i=1;i<d;++i){int selected=idx.GetValue(i),j=i-1;while(j>=0&&ev.GetValue(idx.GetValue(j))<ev.GetValue(selected)){idx.SetValue(j+1,idx.GetValue(j));--j;}idx.SetValue(j+1,selected);}
    int bits=0;for(int size=d;size>1;size>>=1)++bits;
    for(int i=0;i<d;++i){int reverse=0,value=i;for(int bit=0;bit<bits;++bit){reverse=(reverse<<1)|(value&1);value>>=1;}a.SetValue(reverse,float(idx.GetValue(i)));}
}
extern "C" __global__ __aicore__ void oscar_calib_rhp_kernel(
 GM_ADDR vectors,GM_ADDR workspace,GM_ADDR rotation,GM_ADDR diagnostics,int d) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> u,perm,out,diag;u.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vectors));
    perm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotation));
    diag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diagnostics));if(diag.GetValue(7)!=0)return;
    TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,2*kMaxDim*4+512);auto f=buf.Get<float>();
    float normalization=1.0f/Root(f[2*kMaxDim],float(d));
    for(int row=GetBlockIdx();row<d;row+=GetBlockNum()) {
        Load(f,u[row*d],d);
        for(int step=1;step<d;step<<=1)for(int base=0;base<d;base+=2*step)for(int j=0;j<step;++j){float a=f.GetValue(base+j),b=f.GetValue(base+j+step);f.SetValue(base+j,a+b);f.SetValue(base+j+step,a-b);}
        PipeBarrier<PIPE_ALL>();Muls(f,f,normalization,d);PipeBarrier<PIPE_ALL>();
        for(int col=0;col<d;++col)f[kMaxDim].SetValue(col,f.GetValue(static_cast<int>(perm.GetValue(col))));
        Save(out[row*d],f[kMaxDim],d);
    }
}

extern "C" __global__ __aicore__ void oscar_calib_validate_rows_kernel(
 GM_ADDR covariance,GM_ADDR vectors,GM_ADDR eigenvalues,GM_ADDR rotation,GM_ADDR workspace,GM_ADDR diagnostics,int d) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> cov,u,ev,r,rows,diag;
    cov.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(covariance));u.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vectors));
    ev.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eigenvalues));r.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotation));
    rows.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));diag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diagnostics));if(diag.GetValue(7)!=0)return;
    TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,9*kMaxDim*4+512);auto f=buf.Get<float>();
    for(int row=GetBlockIdx();row<d;row+=GetBlockNum()) {
        for(int col=0;col<d;++col)f.SetValue(col,0.5f*(cov.GetValue(row*d+col)+cov.GetValue(col*d+row)));
        PipeBarrier<PIPE_ALL>();RotateRow(f[kMaxDim],f[2*kMaxDim],f[3*kMaxDim],u,f,d,false);
        Load(f[4*kMaxDim],u[row*d],d);Load(f[5*kMaxDim],ev,d);Load(f[6*kMaxDim],r[row*d],d);
        Mul(f[2*kMaxDim],f[4*kMaxDim],f[5*kMaxDim],d);PipeBarrier<PIPE_V>();
        Sub(f[kMaxDim],f[kMaxDim],f[2*kMaxDim],d);PipeBarrier<PIPE_V>();
        Mul(f[kMaxDim],f[kMaxDim],f[kMaxDim],d);Mul(f[2*kMaxDim],f,f,d);PipeBarrier<PIPE_V>();
        float residual=Sum(f[kMaxDim],f[8*kMaxDim],d),norm=Sum(f[2*kMaxDim],f[8*kMaxDim],d),uo=0,ro=0;
        for(int other=0;other<d;++other) {
            Load(f[7*kMaxDim],u[other*d],d);Mul(f[kMaxDim],f[4*kMaxDim],f[7*kMaxDim],d);PipeBarrier<PIPE_V>();
            uo=Max(uo,AbsScalar(Sum(f[kMaxDim],f[8*kMaxDim],d)-(row==other?1.0f:0.0f)));
            Load(f[7*kMaxDim],r[other*d],d);Mul(f[kMaxDim],f[6*kMaxDim],f[7*kMaxDim],d);PipeBarrier<PIPE_V>();
            ro=Max(ro,AbsScalar(Sum(f[kMaxDim],f[8*kMaxDim],d)-(row==other?1.0f:0.0f)));
        }
        rows.SetValue(row*4,residual);rows.SetValue(row*4+1,norm);rows.SetValue(row*4+2,uo);rows.SetValue(row*4+3,ro);
    }
}
extern "C" __global__ __aicore__ void oscar_calib_validate_reduce_kernel(
 GM_ADDR workspace,GM_ADDR eigenvalues,GM_ADDR diagnostics,int d,float tolerance) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(GetBlockIdx()!=0||GetSubBlockIdx()!=0)return;
    GlobalTensor<float> rows,ev,diag;rows.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    ev.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eigenvalues));diag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diagnostics));if(diag.GetValue(7)!=0)return;
    TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,512);auto tmp=buf.Get<float>();
    float residual=0,norm=0,uo=0,ro=0;
    for(int row=0;row<d;++row){residual+=rows.GetValue(row*4);norm+=rows.GetValue(row*4+1);uo=Max(uo,rows.GetValue(row*4+2));ro=Max(ro,rows.GetValue(row*4+3));}
    float rel=norm>0?Root(tmp,residual/norm):__builtin_inff();int status=0;
    if(diag.GetValue(0)!=1)status=1;
    if(!Finite(rel)||!Finite(uo)||!Finite(ro))status=2;
    else if(rel>Max(8*tolerance,2e-4f))status=4;
    else if(uo>1e-3f||ro>1e-3f)status=5;
    else if(ev.GetValue(0)<-1e-4f*diag.GetValue(3))status=6;
    diag.SetValue(4,rel);diag.SetValue(5,uo);diag.SetValue(6,ro);diag.SetValue(7,status);
}

extern "C" void oscar_calib_eigh_rhp_launch(void* stream,const void* covariance,void* rotation,void* eigenvalues,
 void* vectors,void* workspace,void* diagnostics,int d,int sweeps,float tolerance) {
    oscar_calib_jacobi_init_kernel<<<1,nullptr,stream>>>((GM_ADDR)covariance,(GM_ADDR)vectors,(GM_ADDR)workspace,(GM_ADDR)diagnostics,d,tolerance);
    for(int sweep=0;sweep<sweeps;++sweep) {
        for(int round=0;round<d-1;++round) {
            oscar_calib_jacobi_right_kernel<<<32,nullptr,stream>>>((GM_ADDR)workspace,(GM_ADDR)vectors,(GM_ADDR)diagnostics,d,round);
            oscar_calib_jacobi_left_kernel<<<32,nullptr,stream>>>((GM_ADDR)workspace,(GM_ADDR)diagnostics,d,round);
        }
        oscar_calib_jacobi_check_kernel<<<1,nullptr,stream>>>((GM_ADDR)workspace,(GM_ADDR)diagnostics,d,sweep+1,tolerance);
    }
    oscar_calib_eigen_sort_kernel<<<1,nullptr,stream>>>((GM_ADDR)workspace,(GM_ADDR)vectors,(GM_ADDR)eigenvalues,(GM_ADDR)diagnostics,d);
    oscar_calib_rhp_kernel<<<32,nullptr,stream>>>((GM_ADDR)vectors,(GM_ADDR)workspace,(GM_ADDR)rotation,(GM_ADDR)diagnostics,d);
    oscar_calib_validate_rows_kernel<<<32,nullptr,stream>>>((GM_ADDR)covariance,(GM_ADDR)vectors,(GM_ADDR)eigenvalues,(GM_ADDR)rotation,(GM_ADDR)workspace,(GM_ADDR)diagnostics,d);
    oscar_calib_validate_reduce_kernel<<<1,nullptr,stream>>>((GM_ADDR)workspace,(GM_ADDR)eigenvalues,(GM_ADDR)diagnostics,d,tolerance);
}
