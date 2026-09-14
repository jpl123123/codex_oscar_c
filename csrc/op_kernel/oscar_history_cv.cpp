#include "oscar_common.h"
#include "../include/oscar_launch.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace oscar;
using namespace matmul;
using OscarMatA=matmul::MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t>;
using OscarMatBT=matmul::MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t,true>;
using OscarMatB=matmul::MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t>;
using OscarMatC=matmul::MatmulType<TPosition::GM,CubeFormat::ND,float>;

// The only INT2 read site is LoadKvHalf. Each AIV owns 16 distinct token rows,
// together filling a 32-token bridge for one Cube. All 16 query rows consume
// that bridge for QK and PV before either AIV is allowed to overwrite it.
class HistoryCV {
public:
    __aicore__ inline void Init(GM_ADDR q,GM_ADDR cache,GM_ADDR bt,GM_ADDR qsl,
      GM_ADDR hs,GM_ADDR he,GM_ADDR qpos,GM_ADDR workspace,const HistoryPlan& plan,TPipe* pipe) {
        p=plan; pipe_=pipe;
        qg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(q));
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cache));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bt));
        qsg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qsl));
        hsg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(hs));
        heg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(he));
        qpg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qpos));
        if ASCEND_IS_AIC { core=GetBlockIdx(); }
        if ASCEND_IS_AIV { core=GetBlockIdx()/2; sub=GetSubBlockIdx(); }
        auto* base=workspace+core*p.tile_bytes;
        uint64_t off=0;
        tq.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(base+off)); off+=16*p.d*2;
        tk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(base+off)); off+=32*p.d*2;
        tv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(base+off)); off+=32*p.d*2;
        prob.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(base+off)); off+=16*32*2;
        scores.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base+off)); off+=16*32*4;
        pv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base+off));
        partial.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace+p.partial_offset));
        if ASCEND_IS_AIC {
            mmqk.Init(&p.qk,pipe_); mmpv.Init(&p.pv,pipe_);
        }
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(packbuf,512); pipe_->InitBuffer(halfbuf,32);
            pipe_->InitBuffer(kvbuf,2*16*kMaxDim*2);
            pipe_->InitBuffer(qbuf,8*kMaxDim*2);
            pipe_->InitBuffer(scorebuf,8*32*4);
            pipe_->InitBuffer(pbuf,8*32*2);
            pipe_->InitBuffer(accbuf,8*kMaxDim*4);
            pipe_->InitBuffer(pvbuf,8*kMaxDim*4);
            pipe_->InitBuffer(scratchbuf,512);
            pipe_->InitBuffer(unpackbuf,kMaxDim*4);
        }
    }
    __aicore__ inline void SetWindow(GM_ADDR key,GM_ADDR value,GM_ADDR wk,GM_ADDR wv,GM_ADDR pos,GM_ADDR map) {
        rawk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(key));
        rawv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(value));
        wink.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wk));
        winv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wv));
        winpos.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pos));
        winmap.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(map));
    }
    __aicore__ inline void Process() {
        int tasks=p.b*p.hq*p.query_tiles*p.splits;
        for(int task=core;task<tasks;task+=p.cores) {
            int si=task%p.splits;
            int qt=(task/p.splits)%p.query_tiles;
            int head=(task/p.splits/p.query_tiles)%p.hq;
            int req=task/p.splits/p.query_tiles/p.hq;
            int qs=qsg.GetValue(req)+qt*16, qe=qsg.GetValue(req+1);
            if(qs>=qe) continue;
            int hstart=0,hend=0;
            if(p.mode==0) {hstart=hsg.GetValue(req);hend=heg.GetValue(req);}
            else {
                // No query in this tile can see a later raw token. Keep the
                // per-row causal mask, but avoid loading those future tiles.
                int visible_raw=Min(qe-qsg.GetValue(req),(qt+1)*16);
                hend=p.window_capacity+visible_raw;
            }
            int length=Max(0,hend-hstart), per=(length+p.splits-1)/p.splits;
            int begin=hstart+si*per,end=Min(hend,begin+per);
            if ASCEND_IS_AIV {
                auto acc=accbuf.Get<float>(); Duplicate(acc,0.0f,8*p.d);
                for(int i=0;i<8;++i) { maxima[i]=kNegInf; sums[i]=0; }
                auto qb=qbuf.Get<bfloat16_t>(); Duplicate(qb,static_cast<bfloat16_t>(0),8*p.d);
                for(int i=0;i<8;++i) {
                    int token=qs+sub*8+i;
                    if(token<qe) Load(qb[i*p.d],qg[(int64_t(token)*p.hq+head)*p.d],p.d);
                }
                Save(tq[sub*8*p.d],qb,8*p.d);
            }
            for(int start=begin;start<end;start+=32) {
                if ASCEND_IS_AIV {
                    LoadKvHalf(req,head,start,end);
                    CrossCoreSetFlag<0x2,PIPE_MTE3>(1);
                    CrossCoreWaitFlag(2);
                    SoftmaxHalf(req,qs,qe,start,end);
                    CrossCoreSetFlag<0x2,PIPE_MTE3>(3);
                    CrossCoreWaitFlag(4);
                    AccumulateHalf();
                }
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag(1);
                    mmqk.SetOrgShape(16,32,p.d); mmqk.SetSingleShape(16,32,p.d);
                    mmqk.SetTensorA(tq); mmqk.SetTensorB(tk,true); mmqk.IterateAll(scores); mmqk.End();
                    CrossCoreSetFlag<0x2,PIPE_FIX>(2);
                    CrossCoreWaitFlag(3);
                    mmpv.SetOrgShape(16,p.d,32); mmpv.SetSingleShape(16,p.d,32);
                    mmpv.SetTensorA(prob); mmpv.SetTensorB(tv); mmpv.IterateAll(pv); mmpv.End();
                    CrossCoreSetFlag<0x2,PIPE_FIX>(4);
                }
            }
            if ASCEND_IS_AIV {
                auto acc=accbuf.Get<float>(); auto scratch=scratchbuf.Get<float>();
                for(int i=0;i<8;++i) {
                    int token=qs+sub*8+i; if(token>=qe) continue;
                    int64_t off=((int64_t(token)*p.hq+head)*p.splits+si)*(p.d+1);
                    if(sums[i]>0) {
                        Muls(acc[i*p.d],acc[i*p.d],1.0f/sums[i],p.d);
                        Save(partial[off],acc[i*p.d],p.d);
                        partial.SetValue(off+p.d,maxima[i]+VLog(scratch,sums[i]));
                    } else {
                        Duplicate(acc[i*p.d],0.0f,p.d); Save(partial[off],acc[i*p.d],p.d);
                        partial.SetValue(off+p.d,kNegInf);
                    }
                }
                // Fence partial writes and consumption of final PV before the
                // paired Cube starts another task that reuses its tile bridge.
                PipeBarrier<PIPE_ALL>(); CrossCoreSetFlag<0x2,PIPE_MTE3>(5);
            }
            if ASCEND_IS_AIC { CrossCoreWaitFlag(5); }
        }
    }
private:
    __aicore__ inline int WindowPosition(int req,int logical) {
        if(logical<p.window_capacity) {
            int row=winmap.GetValue(req);
            if(row<0 || row>=p.window_rows) return -1;
            int pos=winpos.GetValue(row*p.window_capacity+logical);
            int first=qpg.GetValue(qsg.GetValue(req));
            return pos>=0 && pos<first?pos:-1;
        }
        int token=qsg.GetValue(req)+logical-p.window_capacity;
        if(token>=qsg.GetValue(req+1)) return -1;
        return qpg.GetValue(token);
    }
    __aicore__ inline void LoadKvHalf(int req,int head,int start,int end) {
        auto packed=packbuf.Get<uint8_t>(); auto halfs=halfbuf.Get<half>(); auto bits=halfs.ReinterpretCast<uint16_t>();
        auto buf=kvbuf.Get<bfloat16_t>(); auto ku=buf,vu=buf[16*p.d];
        int bytes=p.d/4,region=bytes+4,slotbytes=2*region,kh=head/(p.hq/p.hk);
        Duplicate(ku,static_cast<bfloat16_t>(0),16*p.d);
        Duplicate(vu,static_cast<bfloat16_t>(0),16*p.d); PipeBarrier<PIPE_ALL>();
        for(int t=0;t<16;++t) {
            int pos=start+sub*16+t; if(pos>=end) continue;
            if(p.mode==1) {
                if(WindowPosition(req,pos)<0) continue;
                bool window=pos<p.window_capacity;
                int64_t row=window?int64_t(winmap.GetValue(req))*p.window_capacity+pos:
                    qsg.GetValue(req)+pos-p.window_capacity;
                Load(ku[t*p.d],(window?wink:rawk)[(row*p.hk+kh)*p.d],p.d);
                Load(vu[t*p.d],(window?winv:rawv)[(row*p.hk+kh)*p.d],p.d);
                continue;
            }
            if(pos<0 || pos/p.table_block_size>=p.pages) continue;
            int blk=bg.GetValue(req*p.pages+pos/p.table_block_size);
            int64_t slot=int64_t(blk)*p.table_block_size+pos%p.table_block_size;
            if(slot<0 || slot>=int64_t(p.num_blocks)*p.block_size) continue;
            int64_t off=(slot/p.block_size)*p.cache_block_stride+(slot%p.block_size)*p.hk*slotbytes+kh*slotbytes;
            Load(packed,cg[off],slotbytes);
            // Scalar float->bfloat16 conversion is not supported by the device
            // backend; stage one row in FP32 and use the vector Cast.
            auto fs=unpackbuf.Get<float>();
            for(int side=0;side<2;++side) {
                int b=side*region;
                bits.SetValue(0,uint16_t(packed.GetValue(b+bytes))|(uint16_t(packed.GetValue(b+bytes+1))<<8));
                bits.SetValue(1,uint16_t(packed.GetValue(b+bytes+2))|(uint16_t(packed.GetValue(b+bytes+3))<<8));
                float sc=static_cast<float>(halfs.GetValue(0)),z=static_cast<float>(halfs.GetValue(1));
                auto out=side==0?ku:vu;
                for(int j=0;j<p.d;++j) {
                    int q=(packed.GetValue(b+j/4)>>(2*(j%4)))&3;
                    fs.SetValue(j,q*sc+z);
                }
                PipeBarrier<PIPE_ALL>();
                Cast(out[t*p.d],fs,RoundMode::CAST_RINT,p.d);
                PipeBarrier<PIPE_ALL>();
            }
        }
        Save(tk[sub*16*p.d],ku,16*p.d); Save(tv[sub*16*p.d],vu,16*p.d);
    }
    __aicore__ inline void SoftmaxHalf(int req,int qs,int qe,int start,int end) {
        auto s=scorebuf.Get<float>(); auto pb=pbuf.Get<bfloat16_t>(); auto scratch=scratchbuf.Get<float>();
        Load(s,scores[sub*8*32],8*32);
        Muls(s,s,p.scale,8*32); PipeBarrier<PIPE_ALL>();
        for(int i=0;i<8;++i) {
            int token=qs+sub*8+i; int qp=token<qe?qpg.GetValue(token):-1;
            float m=maxima[i];
            for(int j=0;j<32;++j) {
                bool valid=token<qe && start+j<end;
                int pos=start+j;
                if(p.mode==1) {
                    int absolute=WindowPosition(req,pos);
                    valid=valid && absolute>=0 && absolute<=qp;
                } else if(pos<0 || pos/p.table_block_size>=p.pages) valid=false;
                else {
                    int64_t slot=int64_t(bg.GetValue(req*p.pages+pos/p.table_block_size))*p.table_block_size+pos%p.table_block_size;
                    valid=valid && pos<=qp && slot>=0 && slot<int64_t(p.num_blocks)*p.block_size;
                }
                float x=valid?s.GetValue(i*32+j):kNegInf;
                s.SetValue(i*32+j,x); m=Max(m,x);
            }
            if(m==kNegInf) {
                alphas[i]=0; for(int j=0;j<32;++j) s.SetValue(i*32+j,0);
            } else {
                alphas[i]=maxima[i]==kNegInf?0:VExp(scratch,maxima[i]-m);
                Adds(s[i*32],s[i*32],-m,32); PipeBarrier<PIPE_V>();
                Exp(s[i*32],s[i*32],32); PipeBarrier<PIPE_ALL>();
                float denom=0; for(int j=0;j<32;++j) denom+=s.GetValue(i*32+j);
                sums[i]=sums[i]*alphas[i]+denom; maxima[i]=m;
            }
        }
        Cast(pb,s,RoundMode::CAST_RINT,8*32); Save(prob[sub*8*32],pb,8*32);
    }
    __aicore__ inline void AccumulateHalf() {
        auto a=accbuf.Get<float>(),v=pvbuf.Get<float>(); Load(v,pv[sub*8*p.d],8*p.d);
        for(int i=0;i<8;++i) {
            Muls(a[i*p.d],a[i*p.d],alphas[i],p.d); PipeBarrier<PIPE_V>();
            Add(a[i*p.d],a[i*p.d],v[i*p.d],p.d); PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
    }
    HistoryPlan p; TPipe* pipe_; int core=0,sub=0;
    matmul::MatmulImpl<OscarMatA,OscarMatBT,OscarMatC> mmqk; matmul::MatmulImpl<OscarMatA,OscarMatB,OscarMatC> mmpv;
    GlobalTensor<bfloat16_t> qg,tq,tk,tv,prob; GlobalTensor<uint8_t> cg;
    GlobalTensor<bfloat16_t> rawk,rawv,wink,winv;
    GlobalTensor<int32_t> winpos,winmap;
    GlobalTensor<int32_t> bg,qsg,hsg,heg,qpg; GlobalTensor<float> scores,pv,partial;
    TBuf<TPosition::VECCALC> packbuf,halfbuf,kvbuf,qbuf,scorebuf,pbuf,accbuf,pvbuf,scratchbuf,unpackbuf;
    float maxima[8],sums[8],alphas[8];
};

extern "C" __global__ __aicore__ void oscar_history_cv_kernel(
    GM_ADDR q,GM_ADDR cache,GM_ADDR bt,GM_ADDR qsl,GM_ADDR hs,GM_ADDR he,GM_ADDR qpos,GM_ADDR ws,HistoryPlan plan) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe; HistoryCV op; op.Init(q,cache,bt,qsl,hs,he,qpos,ws,plan,&pipe); op.Process();
}

extern "C" __global__ __aicore__ void oscar_window_cv_kernel(
    GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR wk,GM_ADDR wv,GM_ADDR pos,GM_ADDR map,GM_ADDR qsl,GM_ADDR qp,GM_ADDR ws,HistoryPlan plan) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;HistoryCV op;
    op.Init(q,nullptr,nullptr,qsl,nullptr,nullptr,qp,ws,plan,&pipe);
    op.SetWindow(k,v,wk,wv,pos,map);op.Process();
}

extern "C" __global__ __aicore__ void oscar_history_reduce_kernel(GM_ADDR ws,GM_ADDR output,GM_ADDR lse,GM_ADDR qsl,HistoryPlan plan) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe p; TBuf<TPosition::VECCALC> buf; p.InitBuffer(buf,3*kMaxDim*4+512); auto b=buf.Get<float>();
    GlobalTensor<float> partial,out,lg;
    partial.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws+plan.partial_offset));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output)); lg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
    GlobalTensor<int32_t> qs; qs.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qsl));
    int actual_tokens=qs.GetValue(plan.b);
    for(int row=GetBlockIdx();row<plan.n*plan.hq;row+=GetBlockNum()) {
        if(row/plan.hq>=actual_tokens) {
            Duplicate(b,0.0f,plan.d);Save(out[int64_t(row)*plan.d],b,plan.d);lg.SetValue(row,kNegInf);continue;
        }
        float m=kNegInf;
        int64_t base=int64_t(row)*plan.splits*(plan.d+1);
        for(int s=0;s<plan.splits;++s) m=Max(m,partial.GetValue(base+s*(plan.d+1)+plan.d));
        Duplicate(b,0.0f,plan.d); float sum=0;
        if(m!=kNegInf) for(int s=0;s<plan.splits;++s) {
            float v=partial.GetValue(base+s*(plan.d+1)+plan.d); if(v==kNegInf) continue;
            float w=VExp(b[3*kMaxDim],v-m); sum+=w;
            Load(b[kMaxDim],partial[base+s*(plan.d+1)],plan.d);
            Muls(b[kMaxDim],b[kMaxDim],w,plan.d); PipeBarrier<PIPE_V>();
            Add(b,b,b[kMaxDim],plan.d); PipeBarrier<PIPE_V>();
        }
        if(sum>0) { Muls(b,b,1.0f/sum,plan.d); lg.SetValue(row,m+VLog(b[3*kMaxDim],sum)); }
        else lg.SetValue(row,kNegInf);
        Save(out[int64_t(row)*plan.d],b,plan.d);
    }
}
// The framework generates host triple-chevron stubs that receive struct
// parameters by pointer (aclrtlaunch_triple_chevrons_func.h), while device
// passes of this TU type-check the by-value call against the kernel. Both
// modes must compile; a wrong assumption here fails loudly at build time.
#if defined(__CCE_AICORE__)
#define OSCAR_LAUNCH_PLAN plan
#else
#define OSCAR_LAUNCH_PLAN &plan
#endif
extern "C" void oscar_history_launch(void* stream,const void* q,const void* cache,const void* bt,const void* qsl,
  const void* hs,const void* he,const void* qpos,void* out,void* lse,void* ws,HistoryPlan plan) {
    oscar_history_cv_kernel<<<plan.cores,nullptr,stream>>>((GM_ADDR)q,(GM_ADDR)cache,(GM_ADDR)bt,(GM_ADDR)qsl,(GM_ADDR)hs,(GM_ADDR)he,(GM_ADDR)qpos,(GM_ADDR)ws,OSCAR_LAUNCH_PLAN);
    oscar_history_reduce_kernel<<<plan.cores*2,nullptr,stream>>>((GM_ADDR)ws,(GM_ADDR)out,(GM_ADDR)lse,(GM_ADDR)qsl,OSCAR_LAUNCH_PLAN);
}
extern "C" void oscar_window_cv_launch(void* stream,const void* q,const void* k,const void* v,const void* wk,
 const void* wv,const void* pos,const void* map,const void* qsl,const void* qp,void* out,void* lse,void* ws,HistoryPlan plan) {
    oscar_window_cv_kernel<<<plan.cores,nullptr,stream>>>((GM_ADDR)q,(GM_ADDR)k,(GM_ADDR)v,(GM_ADDR)wk,(GM_ADDR)wv,(GM_ADDR)pos,(GM_ADDR)map,(GM_ADDR)qsl,(GM_ADDR)qp,(GM_ADDR)ws,OSCAR_LAUNCH_PLAN);
    oscar_history_reduce_kernel<<<plan.cores*2,nullptr,stream>>>((GM_ADDR)ws,(GM_ADDR)out,(GM_ADDR)lse,(GM_ADDR)qsl,OSCAR_LAUNCH_PLAN);
}
