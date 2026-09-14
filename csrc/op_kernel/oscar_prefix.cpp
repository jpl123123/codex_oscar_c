// Bounded recovery and prefix-cache window reconstruction kernels.
//
// dequant: explicit bounded row-count recovery used by probes and reference
//   alignment. It is never part of the decode/verify hot path.
// stage: mirrors the upstream PR's BF16 staging cache. Raw K/V of tokens that
//   live in a Sink/Recent window at write time are stored, keyed by the same
//   virtual slot namespace as native slot_mapping. Owner tags are published
//   before the data, so a reader never observes a torn entry.
// restore: on a prefix-cache hit (new request identity with computed tokens),
//   rebuild the BF16 Sink/Recent ring from staging where the tag still
//   matches, falling back to INT2 history for evicted rows. Lossy rows are
//   counted; they are never reported as lossless.
#include "oscar_common.h"
#include "oscar_launch.h"
using namespace AscendC;
using namespace oscar;

namespace {
__aicore__ inline int UnpackHalf(LocalTensor<uint8_t>& packed, LocalTensor<half>& halfs,
                                 int region, int bytes, int side, float& sc, float& z) {
    auto bits = halfs.ReinterpretCast<uint16_t>();
    bits.SetValue(0, uint16_t(packed.GetValue(side * region + bytes)) |
                    (uint16_t(packed.GetValue(side * region + bytes + 1)) << 8));
    bits.SetValue(1, uint16_t(packed.GetValue(side * region + bytes + 2)) |
                    (uint16_t(packed.GetValue(side * region + bytes + 3)) << 8));
    sc = static_cast<float>(halfs.GetValue(0));
    z = static_cast<float>(halfs.GetValue(1));
    return side * region;
}
}

// slots[i] selects one packed row; outputs are dequantized and rotated back
// (transpose) into the original BF16 coordinate frame, one row per head.
extern "C" __global__ __aicore__ void oscar_dequant_kernel(
 GM_ADDR cache,GM_ADDR slots,GM_ADDR rk,GM_ADDR rv,GM_ADDR kout,GM_ADDR vout,
 int n,int hk,int d,int bs,int numblocks,int64_t block_stride) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; TBuf<TPosition::VECCALC> fb,bb,packb,halfb;
    pipe.InitBuffer(fb,4*kMaxDim*sizeof(float));
    pipe.InitBuffer(bb,kMaxDim*sizeof(bfloat16_t));
    pipe.InitBuffer(packb,Align(2*(kMaxDim/4+4)));
    pipe.InitBuffer(halfb,32);
    auto f=fb.Get<float>(); auto bf=bb.Get<bfloat16_t>();
    auto packed=packb.Get<uint8_t>(); auto halfs=halfb.Get<half>();
    GlobalTensor<uint8_t> cg; GlobalTensor<int64_t> sg; GlobalTensor<float> kr,vr;
    GlobalTensor<bfloat16_t> ko,vo;
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cache));
    sg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
    kr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rk));
    vr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rv));
    ko.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(kout));
    vo.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(vout));
    int bytes=d/4,region=bytes+4,slot_bytes=2*region;
    for(int row=GetBlockIdx();row<n*hk;row+=GetBlockNum()) {
        int token=row/hk,head=row%hk; int64_t slot=sg.GetValue(token);
        if(slot<0 || slot>=int64_t(numblocks)*bs) continue;
        int64_t off=(slot/bs)*block_stride+(slot%bs)*hk*slot_bytes+head*slot_bytes;
        Load(packed,cg[off],slot_bytes);
        for(int side=0;side<2;++side) {
            float sc,z; int b=UnpackHalf(packed,halfs,region,bytes,side,sc,z);
            for(int j=0;j<d;++j) {
                int q=(packed.GetValue(b+j/4)>>(2*(j%4)))&3;
                f.SetValue(j,q*sc+z);
            }
            PipeBarrier<PIPE_ALL>();
            RotateRow(f[kMaxDim],f[2*kMaxDim],f[3*kMaxDim],side==0?kr:vr,f,d,true);
            Cast(bf,f[kMaxDim],RoundMode::CAST_RINT,d);
            Save((side==0?ko:vo)[int64_t(row)*d],bf,d);
        }
    }
}

// Stage raw K/V rows of the current forward whose absolute positions belong to
// Sink or the final Recent window. Keyed by the native virtual slot namespace.
extern "C" __global__ __aicore__ void oscar_stage_kernel(
 GM_ADDR k,GM_ADDR v,GM_ADDR seq,GM_ADDR qsl,GM_ADDR slots,
 GM_ADDR sk,GM_ADDR sv,GM_ADDR owner,int batch,int n,int hk,int d,
 int sink,int recent,int srows,int bs) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf,kMaxDim*2);
    GlobalTensor<bfloat16_t> kg,vg,skg,svg;
    GlobalTensor<int32_t> sg,qsg; GlobalTensor<int64_t> slg,og;
    kg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
    vg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(v));
    sg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seq));
    qsg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qsl));
    slg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
    skg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sk));
    svg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sv));
    og.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(owner));
    auto tmp=buf.Get<bfloat16_t>();
    for(int req=GetBlockIdx();req<batch;req+=GetBlockNum()) {
        int begin=qsg.GetValue(req),end=qsg.GetValue(req+1);
        int seqlen=sg.GetValue(req);
        for(int t=begin;t<end;++t) {
            int pos=seqlen-(end-begin)+(t-begin);
            if(!(pos<sink || (recent>0 && pos>=Max(sink,seqlen-recent)))) continue;
            int64_t slot=slg.GetValue(t);
            if(slot<0) continue;
            int row=int((slot/bs)%srows),off=int(slot%bs);
            int64_t entry=int64_t(row)*bs+off;
            // Publish ownership before data: a competing reader either sees the
            // previous owner (mismatch) or this owner with completed contents.
            og.SetValue(entry,slot/bs);
            for(int side=0;side<2;++side) {
                auto src=side==0?kg:vg, dst=side==0?skg:svg;
                for(int h=0;h<hk;++h) {
                    Load(tmp,src[int64_t(t)*hk*d+h*d],d);
                    Save(dst[entry*hk*d+h*d],tmp,d);
                }
            }
        }
    }
}

// Rebuild the window ring for prefix-hit rows: positions of Sink [0,min(S,C))
// and Recent [max(S,C-R),C) are restored from staging when the owner tag still
// matches, otherwise from bounded INT2 dequantization and counted as lossy.
extern "C" __global__ __aicore__ void oscar_restore_kernel(
 GM_ADDR seq,GM_ADDR qsl,GM_ADDR rowmap,GM_ADDR epochs,GM_ADDR bt,GM_ADDR pos,
 GM_ADDR wk,GM_ADDR wv,GM_ADDR state,GM_ADDR sk,GM_ADDR sv,GM_ADDR owner,
 GM_ADDR cache,GM_ADDR rk,GM_ADDR rv,GM_ADDR lossy,
 int batch,int hk,int d,int w,int pages,int sink,int recent,int mp,
 int srows,int bs,int table_bs,int num_blocks,int64_t cache_stride) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; TBuf<TPosition::VECCALC> fb,bb,packb,halfb;
    pipe.InitBuffer(fb,4*kMaxDim*sizeof(float));
    pipe.InitBuffer(bb,kMaxDim*sizeof(bfloat16_t));
    pipe.InitBuffer(packb,Align(2*(kMaxDim/4+4)));
    pipe.InitBuffer(halfb,32);
    auto f=fb.Get<float>(); auto bf=bb.Get<bfloat16_t>();
    auto packed=packb.Get<uint8_t>(); auto halfs=halfb.Get<half>();
    GlobalTensor<int32_t> sg,qsg,rg,bg,posg; GlobalTensor<int64_t> eg,st,og;
    GlobalTensor<bfloat16_t> wkg,wvg,skg,svg; GlobalTensor<uint8_t> cg;
    GlobalTensor<float> kr,vr; GlobalTensor<int32_t> lz;
    sg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seq));
    qsg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qsl));
    rg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowmap));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bt));
    posg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pos));
    eg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(epochs));
    st.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(state));
    og.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(owner));
    wkg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wk));
    wvg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wv));
    skg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sk));
    svg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sv));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cache));
    kr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rk));
    vr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rv));
    lz.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lossy));
    int cap=sink+recent+mp,bytes=d/4,region=bytes+4,slot_bytes=2*region;
    for(int req=GetBlockIdx();req<batch;req+=GetBlockNum()) {
        int row=rg.GetValue(req);
        int qlen=qsg.GetValue(req+1)-qsg.GetValue(req);
        if(row<0 || row>=w || qlen==0) continue;
        int64_t epoch=eg.GetValue(req);
        if(st.GetValue(row*4)==epoch) continue;  // no prefix hit on this row
        for(int j=0;j<cap;++j) posg.SetValue(row*cap+j,-1);
        int committed=sg.GetValue(req)-qlen;
        if(committed<=0) continue;  // fresh identity: phase-0 reset handles it
        st.SetValue(row*4,epoch);
        st.SetValue(row*4+1,committed);
        st.SetValue(row*4+2,0);
        st.SetValue(row*4+3,0);
        int bound=Max(sink,committed-recent);
        for(int p=0;p<committed;++p) {
            bool sink_row=p<sink,recent_row=recent>0 && p>=bound;
            if(!sink_row && !recent_row) continue;
            int index=sink_row?p:sink+(p-sink)%recent;
            if(p/table_bs>=pages) { st.SetValue(row*4+3,6); break; }
            int block=bg.GetValue(req*pages+p/table_bs);
            if(block<0) { st.SetValue(row*4+3,6); break; }
            int64_t slot=int64_t(block)*table_bs+p%table_bs;
            int srow=int((slot/bs)%srows),soff=int(slot%bs);
            int64_t entry=int64_t(srow)*bs+soff;
            bool staged=og.GetValue(entry)==slot/bs;
            if(staged) {
                for(int h=0;h<hk;++h) {
                    Load(bf,skg[entry*hk*d+h*d],d); Save(wkg[(int64_t(row)*cap+index)*hk*d+h*d],bf,d);
                    Load(bf,svg[entry*hk*d+h*d],d); Save(wvg[(int64_t(row)*cap+index)*hk*d+h*d],bf,d);
                }
            } else {
                // Bounded lossy fallback: only this window row is recovered.
                if(slot/int64_t(bs)>=num_blocks || slot<0) { st.SetValue(row*4+3,6); break; }
                int64_t base=(slot/bs)*cache_stride+(slot%bs)*hk*slot_bytes;
                for(int h=0;h<hk;++h) {
                    Load(packed,cg[base+h*slot_bytes],slot_bytes);
                    for(int side=0;side<2;++side) {
                        float sc,z; int b=UnpackHalf(packed,halfs,region,bytes,side,sc,z);
                        for(int j=0;j<d;++j) {
                            int q=(packed.GetValue(b+j/4)>>(2*(j%4)))&3;
                            f.SetValue(j,q*sc+z);
                        }
                        PipeBarrier<PIPE_ALL>();
                        RotateRow(f[kMaxDim],f[2*kMaxDim],f[3*kMaxDim],side==0?kr:vr,f,d,true);
                        Cast(bf,f[kMaxDim],RoundMode::CAST_RINT,d);
                        Save((side==0?wkg:wvg)[(int64_t(row)*cap+index)*hk*d+h*d],bf,d);
                    }
                }
                lz.AtomicAdd(0,1);
            }
            posg.SetValue(row*cap+index,p);
        }
    }
}

extern "C" void oscar_dequant_launch(void* stream,const void* cache,const void* slots,
 const void* rk,const void* rv,void* kout,void* vout,
 int n,int hk,int d,int bs,int nb,int64_t stride) {
    oscar_dequant_kernel<<<32,nullptr,stream>>>((GM_ADDR)cache,(GM_ADDR)slots,(GM_ADDR)rk,(GM_ADDR)rv,
        (GM_ADDR)kout,(GM_ADDR)vout,n,hk,d,bs,nb,stride);
}
extern "C" void oscar_stage_launch(void* stream,const void* k,const void* v,const void* seq,
 const void* qsl,const void* slots,void* sk,void* sv,void* owner,
 int batch,int n,int hk,int d,int sink,int recent,int srows,int bs) {
    oscar_stage_kernel<<<32,nullptr,stream>>>((GM_ADDR)k,(GM_ADDR)v,(GM_ADDR)seq,(GM_ADDR)qsl,(GM_ADDR)slots,
        (GM_ADDR)sk,(GM_ADDR)sv,(GM_ADDR)owner,batch,n,hk,d,sink,recent,srows,bs);
}
extern "C" void oscar_restore_launch(void* stream,const void* seq,const void* qsl,const void* rowmap,
 const void* epochs,const void* bt,void* pos,void* wk,void* wv,void* state,
 const void* sk,const void* sv,const void* owner,const void* cache,
 const void* rk,const void* rv,void* lossy,
 int batch,int hk,int d,int w,int pages,int sink,int recent,int mp,
 int srows,int bs,int table_bs,int num_blocks,int64_t cache_stride) {
    oscar_restore_kernel<<<32,nullptr,stream>>>((GM_ADDR)seq,(GM_ADDR)qsl,(GM_ADDR)rowmap,(GM_ADDR)epochs,
        (GM_ADDR)bt,(GM_ADDR)pos,(GM_ADDR)wk,(GM_ADDR)wv,(GM_ADDR)state,(GM_ADDR)sk,(GM_ADDR)sv,(GM_ADDR)owner,
        (GM_ADDR)cache,(GM_ADDR)rk,(GM_ADDR)rv,(GM_ADDR)lossy,
        batch,hk,d,w,pages,sink,recent,mp,srows,bs,table_bs,num_blocks,cache_stride);
}
