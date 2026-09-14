#include "oscar_common.h"
#include "../include/oscar_launch.h"
using namespace AscendC;
using namespace oscar;

class WindowState {
public:
    __aicore__ inline void Init(GM_ADDR key,GM_ADDR value,GM_ADDR seq,GM_ADDR qsl,GM_ADDR rowmap,
      GM_ADDR epochs,GM_ADDR bt,GM_ADDR slots,GM_ADDR prefill,GM_ADDR wk,GM_ADDR wv,GM_ADDR pos,
      GM_ADDR state,GM_ADDR hs,GM_ADDR he,GM_ADDR qp,GM_ADDR mk,GM_ADDR mv,GM_ADDR ms,GM_ADDR cs,
      int batch,int heads,int dim,int windows,int pages,int phase,int sink,int recent,int pending,int bs) {
        b=batch;hk=heads;d=dim;w=windows;npages=pages;ph=phase;s=sink;r=recent;mp=pending;blocksize=bs;
        cap=s+r+mp; mcap=r+mp;
        kg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(key)); vg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(value));
        sg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seq)); qsg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qsl));
        rg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowmap)); eg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(epochs));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bt)); slg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
        pg.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(prefill));
        wkg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wk)); wvg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wv));
        posg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pos)); st.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(state));
        hsg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(hs)); heg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(he));
        qpg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qp)); mkg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(mk));
        mvg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(mv)); msg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ms));
        csg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cs));
        pipe.InitBuffer(buf,kMaxDim*2);
    }
    __aicore__ inline void Process() {
        for(int req=GetBlockIdx();req<b;req+=GetBlockNum()) {
            int row=rg.GetValue(req),qb=qsg.GetValue(req),qe=qsg.GetValue(req+1),qlen=qe-qb;
            for(int j=0;j<mcap;++j) msg.SetValue(req*mcap+j,-1);
            for(int j=qb;j<qe;++j) csg.SetValue(j,-1);
            if(ph==0) for(int j=qb;j<qe;++j) qpg.SetValue(j,-1);
            if(row<0 || qlen==0) { hsg.SetValue(req,0); heg.SetValue(req,0); continue; }
            if(row>=w) continue; // host validates W against row assignment capacity
            int64_t epoch=eg.GetValue(req);
            if(ph==0 && st.GetValue(row*4)!=epoch) {
                for(int j=0;j<cap;++j) posg.SetValue(row*cap+j,-1);
                st.SetValue(row*4,epoch); st.SetValue(row*4+1,0); st.SetValue(row*4+2,0); st.SetValue(row*4+3,0);
            }
            if(st.GetValue(row*4+3)!=0) continue;
            int old=static_cast<int>(st.GetValue(row*4+1));
            int pending=static_cast<int>(st.GetValue(row*4+2));
            if(ph==0) {
                int committed=sg.GetValue(req)-qlen;
                if(committed<old || committed>old+pending) { Error(row,1); continue; }
                bool ok=true;int migrated=0;
                for(int pos=Max(s,old-r);pos<Max(s,committed-r);++pos) {
                    int wi=pos>=old?s+r+(pos-old):Ring(pos);
                    if(!Migrate(req,row,wi,pos,migrated++)) {ok=false;break;}
                }
                if(!ok) continue;
                for(int pos=old;pos<committed;++pos) {
                    int source=s+r+(pos-old);
                    if(posg.GetValue(row*cap+source)!=pos) {Error(row,2);ok=false;break;}
                    if(pos<s || pos>=Max(s,committed-r)) CopyWindow(row,source,Ring(pos),pos);
                }
                if(!ok) continue;
                for(int j=s+r;j<cap;++j) posg.SetValue(row*cap+j,-1);
                for(int j=s;j<s+r;++j) {
                    int pos=posg.GetValue(row*cap+j);
                    if(pos<Max(s,committed-r) || pos>=committed) posg.SetValue(row*cap+j,-1);
                }
                st.SetValue(row*4+1,committed); st.SetValue(row*4+2,0);
                int start=Min(s,committed); hsg.SetValue(req,start); heg.SetValue(req,Max(start,committed-r));
                for(int j=0;j<qlen;++j) qpg.SetValue(qb+j,committed+j);
            } else if(pg.GetValue(req)) {
                int committed=old+qlen,bound=Max(s,committed-r),migrated=0;bool ok=true;
                for(int pos=Max(s,old-r);pos<Min(old,bound);++pos) {
                    if(!Migrate(req,row,Ring(pos),pos,migrated++)) {ok=false;break;}
                }
                if(!ok) continue;
                for(int j=0;j<qlen;++j) {
                    int pos=old+j,token=qb+j;
                    if(pos>=s && pos<bound) csg.SetValue(token,slg.GetValue(token));
                    else CopyRaw(token,row,Ring(pos),pos);
                }
                for(int j=0;j<cap;++j) {
                    int pos=posg.GetValue(row*cap+j);
                    bool keep=pos>=0 && pos<committed && (pos<s || pos>=bound);
                    if(!keep || j>=s+r) posg.SetValue(row*cap+j,-1);
                }
                st.SetValue(row*4+1,committed); st.SetValue(row*4+2,0);
            } else {
                if(qlen>mp) { Error(row,3);continue; }
                for(int j=0;j<qlen;++j) CopyRaw(qb+j,row,s+r+j,old+j);
                st.SetValue(row*4+2,qlen);
            }
        }
    }
private:
    __aicore__ inline int Ring(int pos) {
        // With R=0 every committed non-sink token goes straight to History;
        // no caller may ask for a recent ring slot in that configuration.
        return pos<s?pos:(r>0?s+(pos-s)%r:-1);
    }
    __aicore__ inline void Error(int row,int code) { st.SetValue(row*4+3,code); }
    __aicore__ inline void CopyRow(GlobalTensor<bfloat16_t> dst,GlobalTensor<bfloat16_t> src) {
        auto tmp=buf.Get<bfloat16_t>(); for(int h=0;h<hk;++h) { Load(tmp,src[h*d],d); Save(dst[h*d],tmp,d); }
    }
    __aicore__ inline void CopyWindow(int row,int src,int dst,int pos) {
        if(src!=dst) {
            CopyRow(wkg[(int64_t(row)*cap+dst)*hk*d],wkg[(int64_t(row)*cap+src)*hk*d]);
            CopyRow(wvg[(int64_t(row)*cap+dst)*hk*d],wvg[(int64_t(row)*cap+src)*hk*d]);
        }
        posg.SetValue(row*cap+dst,pos);
    }
    __aicore__ inline void CopyRaw(int token,int row,int dst,int pos) {
        CopyRow(wkg[(int64_t(row)*cap+dst)*hk*d],kg[int64_t(token)*hk*d]);
        CopyRow(wvg[(int64_t(row)*cap+dst)*hk*d],vg[int64_t(token)*hk*d]);
        posg.SetValue(row*cap+dst,pos);
    }
    __aicore__ inline bool Migrate(int req,int row,int wi,int pos,int index) {
        if(index>=mcap || wi<0 || wi>=cap || posg.GetValue(row*cap+wi)!=pos) {Error(row,2);return false;}
        if(pos/blocksize>=npages) {Error(row,4);return false;}
        int64_t block=bg.GetValue(req*npages+pos/blocksize);
        if(block<0) {Error(row,4);return false;}
        CopyRow(mkg[(int64_t(req)*mcap+index)*hk*d],wkg[(int64_t(row)*cap+wi)*hk*d]);
        CopyRow(mvg[(int64_t(req)*mcap+index)*hk*d],wvg[(int64_t(row)*cap+wi)*hk*d]);
        msg.SetValue(req*mcap+index,block*blocksize+pos%blocksize); return true;
    }
    TPipe pipe;TBuf<TPosition::VECCALC> buf;
    GlobalTensor<bfloat16_t> kg,vg,wkg,wvg,mkg,mvg;
    GlobalTensor<int32_t> sg,qsg,rg,bg,posg,hsg,heg,qpg;
    GlobalTensor<int64_t> eg,slg,st,msg,csg; GlobalTensor<bool> pg;
    int b,hk,d,w,npages,ph,s,r,mp,blocksize,cap,mcap;
};
extern "C" __global__ __aicore__ void oscar_window_state_kernel(
 GM_ADDR key,GM_ADDR value,GM_ADDR seq,GM_ADDR qsl,GM_ADDR rowmap,GM_ADDR epochs,GM_ADDR bt,GM_ADDR slots,
 GM_ADDR prefill,GM_ADDR wk,GM_ADDR wv,GM_ADDR pos,GM_ADDR state,GM_ADDR hs,GM_ADDR he,GM_ADDR qp,
 GM_ADDR mk,GM_ADDR mv,GM_ADDR ms,GM_ADDR cs,int b,int hk,int d,int w,int pages,int phase,int s,int r,int mp,int bs) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    WindowState op;op.Init(key,value,seq,qsl,rowmap,epochs,bt,slots,prefill,wk,wv,pos,state,hs,he,qp,mk,mv,ms,cs,b,hk,d,w,pages,phase,s,r,mp,bs);op.Process();
}
extern "C" void oscar_window_state_launch(void* stream,const void* k,const void* v,const void* seq,const void* qsl,
 const void* map,const void* epoch,const void* bt,const void* slots,const void* prefill,void* wk,void* wv,void* pos,void* state,
 void* hs,void* he,void* qp,void* mk,void* mv,void* ms,void* cs,
 int b,int hk,int d,int w,int pages,int phase,int s,int r,int mp,int bs) {
    oscar_window_state_kernel<<<32,nullptr,stream>>>((GM_ADDR)k,(GM_ADDR)v,(GM_ADDR)seq,(GM_ADDR)qsl,(GM_ADDR)map,(GM_ADDR)epoch,(GM_ADDR)bt,(GM_ADDR)slots,(GM_ADDR)prefill,(GM_ADDR)wk,(GM_ADDR)wv,(GM_ADDR)pos,(GM_ADDR)state,(GM_ADDR)hs,(GM_ADDR)he,(GM_ADDR)qp,(GM_ADDR)mk,(GM_ADDR)mv,(GM_ADDR)ms,(GM_ADDR)cs,b,hk,d,w,pages,phase,s,r,mp,bs);
}
