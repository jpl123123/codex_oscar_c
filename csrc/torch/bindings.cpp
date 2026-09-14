#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <cmath>
#include <limits>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "oscar_launch.h"
#include "../op_host/tiling.h"

namespace {
using at::Tensor;
void Check(const Tensor& t,at::ScalarType dtype,int rank,const char* name,bool contiguous=true) {
    TORCH_CHECK(t.device().type()==c10::DeviceType::PrivateUse1,name," must be on NPU");
    TORCH_CHECK(t.scalar_type()==dtype && t.dim()==rank,name," dtype/rank mismatch");
    TORCH_CHECK(!contiguous || t.is_contiguous(),name," must be contiguous");
}
void Same(std::initializer_list<Tensor> tensors) {
    auto d=tensors.begin()->device();
    for(const auto& t:tensors) TORCH_CHECK(t.device()==d,"all OSCAR operands must share one NPU device");
}
int Dim(int64_t d) { TORCH_CHECK(d==64||d==128||d==256,"OSCAR v1 supports D=64,128,256");return d; }
void Qkv(const Tensor& q,const char* name) {
    Check(q,at::kBFloat16,3,name);Dim(q.size(2));
    TORCH_CHECK(q.size(0)<=std::numeric_limits<int>::max() && q.size(1)>0,"invalid QKV geometry");
}
void Rotation(const Tensor& r,int d) { Check(r,at::kFloat,2,"rotation");TORCH_CHECK(r.size(0)==d&&r.size(1)==d,"rotation shape must be [D,D]"); }
void Cache(const Tensor& h,int d,int hk) {
    Check(h,at::kByte,4,"history",false);
    int c=2*(d/4+4);
    TORCH_CHECK(h.size(0)>0&&h.size(1)>0&&h.size(2)==hk&&h.size(3)==c,"history shape mismatch");
    TORCH_CHECK(h.stride(3)==1&&h.stride(2)==c&&h.stride(1)==hk*c&&h.stride(0)>=h.size(1)*hk*c,"invalid history strides");
}
void Vec(const Tensor& t,at::ScalarType type,int64_t n,const char* name) {
    Check(t,type,1,name);TORCH_CHECK(t.numel()==n,name," length mismatch");
}
void Partial(const Tensor& out,const Tensor& lse,const Tensor& q) {
    Check(out,at::kFloat,3,"output");Check(lse,at::kFloat,2,"lse");
    TORCH_CHECK(out.sizes()==q.sizes()&&lse.size(0)==q.size(0)&&lse.size(1)==q.size(1),"partial output shape mismatch");
}
template<class F> void Launch(const char* name,F fn) {
    at_npu::native::OpCommand cmd;cmd.Name(name);cmd.SetCustomHandler(fn);cmd.Run();
}
int64_t Workspace(int64_t n,int64_t hq,int64_t d,int64_t splits) {
    Dim(d);TORCH_CHECK(n>=0&&n<=(1<<24)&&hq>0&&hq<=256&&splits>0&&splits<=64,"invalid workspace dimensions");
    return oscar::WorkspaceBytes(n,hq,d,splits,oscar::CoreCount());
}
void Rotate(const Tensor& x,const Tensor& rotation,Tensor out,bool transpose) {
    Qkv(x,"x");Rotation(rotation,x.size(2));Qkv(out,"out");
    TORCH_CHECK(x.sizes()==out.sizes(),"rotate output shape mismatch");Same({x,rotation,out});
    c10::DeviceGuard guard(x.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto xp=x.data_ptr(),rp=rotation.data_ptr(),op=out.data_ptr();int rows=x.size(0)*x.size(1),d=x.size(2);
    Launch("oscar_rotate_out",[=](){oscar_rotate_launch(stream,xp,rp,op,rows,d,transpose);return 0;});
}
void Store(const Tensor& k,const Tensor& v,const Tensor& rk,const Tensor& rv,const Tensor& slots,Tensor h,double kc,double vc) {
    Qkv(k,"key");Qkv(v,"value");TORCH_CHECK(k.sizes()==v.sizes(),"K/V shape mismatch");
    int n=k.size(0),hk=k.size(1),d=k.size(2),bs=h.size(1);
    Rotation(rk,d);Rotation(rv,d);Vec(slots,at::kLong,n,"slots");Cache(h,d,hk);
    TORCH_CHECK(std::isfinite(kc)&&kc>=0&&kc<=1&&std::isfinite(vc)&&vc>=0&&vc<=1,"clip ratios must be in [0,1]");
    Same({k,v,rk,rv,slots,h});c10::DeviceGuard guard(k.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto kp=k.data_ptr(),vp=v.data_ptr(),kr=rk.data_ptr(),vr=rv.data_ptr(),sp=slots.data_ptr(),hp=h.data_ptr();int64_t stride=h.stride(0);int nb=h.size(0);
    Launch("oscar_store_int2",[=](){oscar_store_launch(stream,kp,vp,kr,vr,sp,hp,n,hk,d,bs,nb,stride,kc,vc);return 0;});
}
void History(const Tensor& q,const Tensor& h,const Tensor& bt,const Tensor& qsl,const Tensor& hs,const Tensor& he,
             const Tensor& qp,Tensor out,Tensor lse,Tensor ws,double scale,int64_t splits,int64_t maxq,int64_t tablebs) {
    Qkv(q,"query_rotated");Check(h,at::kByte,4,"history",false);Cache(h,q.size(2),h.size(2));
    Check(bt,at::kInt,2,"block_table");int b=bt.size(0);Vec(qsl,at::kInt,b+1,"qsl");
    Vec(hs,at::kInt,b,"history_start");Vec(he,at::kInt,b,"history_end");Vec(qp,at::kInt,q.size(0),"query_positions");
    Partial(out,lse,q);Check(ws,at::kByte,1,"workspace");
    TORCH_CHECK(q.size(1)%h.size(2)==0,"query heads must be divisible by KV heads");
    TORCH_CHECK(std::isfinite(scale)&&scale>0&&splits>0&&splits<=64&&maxq>0,"invalid attention scalars");
    if(tablebs==0)tablebs=h.size(1);
    TORCH_CHECK(tablebs>0&&h.size(1)%tablebs==0,"block table page size must divide physical history page size");
    Same({q,h,bt,qsl,hs,he,qp,out,lse,ws});c10::DeviceGuard guard(q.device());
    auto plan=oscar::MakePlan(q.size(0),q.size(1),h.size(2),q.size(2),b,bt.size(1),h.size(1),splits,maxq,h.stride(0),scale,tablebs);
    plan.num_blocks=h.size(0);
    TORCH_CHECK(ws.numel()>=oscar::WorkspaceBytes(plan.n,plan.hq,plan.d,plan.splits,plan.cores),"history workspace too small");
    auto stream=c10_npu::getCurrentNPUStream().stream();
    auto qptr=q.data_ptr(),hp=h.data_ptr(),bp=bt.data_ptr(),qsp=qsl.data_ptr(),hsp=hs.data_ptr(),hep=he.data_ptr(),qpp=qp.data_ptr();
    auto op=out.data_ptr(),lp=lse.data_ptr(),wp=ws.data_ptr();
    Launch("oscar_history_attention_cv",[=](){oscar_history_launch(stream,qptr,hp,bp,qsp,hsp,hep,qpp,op,lp,wp,plan);return 0;});
}
void Window(const Tensor& q,const Tensor& k,const Tensor& v,const Tensor& wk,const Tensor& wv,const Tensor& pos,
            const Tensor& map,const Tensor& qsl,const Tensor& qp,Tensor out,Tensor lse,double scale,Tensor ws,int64_t maxq) {
    Qkv(q,"query");Qkv(k,"key");Qkv(v,"value");TORCH_CHECK(k.sizes()==v.sizes()&&q.size(0)==k.size(0)&&q.size(2)==k.size(2),"raw QKV mismatch");
    Check(wk,at::kBFloat16,4,"window_k");Check(wv,at::kBFloat16,4,"window_v");Check(pos,at::kInt,2,"window_positions");
    TORCH_CHECK(wk.sizes()==wv.sizes()&&wk.size(2)==k.size(1)&&wk.size(3)==k.size(2)&&pos.size(0)==wk.size(0)&&pos.size(1)==wk.size(1),"window geometry mismatch");
    Check(map,at::kInt,1,"row_to_window");int b=map.numel();Vec(qsl,at::kInt,b+1,"qsl");Vec(qp,at::kInt,q.size(0),"query_positions");
    TORCH_CHECK(q.size(1)%k.size(1)==0&&std::isfinite(scale)&&scale>0&&maxq>0,"invalid GQA, max_query_len or scale");Partial(out,lse,q);Check(ws,at::kByte,1,"workspace");
    Same({q,k,v,wk,wv,pos,map,qsl,qp,out,lse,ws});c10::DeviceGuard guard(q.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto a=q.data_ptr(),bp=k.data_ptr(),c=v.data_ptr(),e=wk.data_ptr(),f=wv.data_ptr(),g=pos.data_ptr(),j=map.data_ptr(),l=qsl.data_ptr(),m=qp.data_ptr(),o=out.data_ptr(),z=lse.data_ptr();
    int n=q.size(0),hq=q.size(1),hk=k.size(1),d=q.size(2),cap=wk.size(1);
    auto plan=oscar::MakePlan(n,hq,hk,d,b,0,1,1,maxq,0,scale,1);
    plan.mode=1;plan.window_capacity=cap;plan.window_rows=wk.size(0);
    TORCH_CHECK(ws.numel()>=oscar::WorkspaceBytes(n,hq,d,1,plan.cores),"window workspace too small");auto wsp=ws.data_ptr();
    Launch("oscar_window_attention_cv",[=](){oscar_window_cv_launch(stream,a,bp,c,e,f,g,j,l,m,o,z,wsp,plan);return 0;});
}
void Merge(const Tensor& h,const Tensor& hl,const Tensor& w,const Tensor& wl,const Tensor& rv,Tensor out) {
    Qkv(out,"output");Partial(h,hl,out);Partial(w,wl,out);Rotation(rv,out.size(2));Same({h,hl,w,wl,rv,out});
    c10::DeviceGuard guard(out.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto hp=h.data_ptr(),lp=hl.data_ptr(),wp=w.data_ptr(),wlp=wl.data_ptr(),rp=rv.data_ptr(),op=out.data_ptr();
    int rows=out.size(0)*out.size(1),d=out.size(2);
    Launch("oscar_attention_merge",[=](){oscar_merge_launch(stream,hp,lp,wp,wlp,rp,op,rows,d);return 0;});
}
void Zero(Tensor h,const Tensor& ids) {
    Check(h,at::kByte,4,"history",false);Check(ids,at::kLong,1,"block_ids");
    TORCH_CHECK(h.stride(3)==1&&h.stride(2)==h.size(3)&&h.stride(1)==h.size(2)*h.size(3)&&h.stride(0)>=h.size(1)*h.size(2)*h.size(3),"zero requires row-major pages");
    TORCH_CHECK(uint64_t(h.storage_offset()+h.size(0)*h.stride(0))<=h.storage().nbytes(),"zero needs backing storage for whole final padded page");
    Same({h,ids});c10::DeviceGuard guard(h.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto hp=h.data_ptr(),ip=ids.data_ptr();int count=ids.numel();int64_t stride=h.stride(0),blocks=h.size(0);
    Launch("oscar_zero_blocks",[=](){oscar_zero_launch(stream,hp,ip,count,stride,blocks);return 0;});
}
void Dequant(const Tensor& h,const Tensor& slots,const Tensor& rk,const Tensor& rv,Tensor k,Tensor v) {
    Qkv(k,"key_out");Qkv(v,"value_out");TORCH_CHECK(k.sizes()==v.sizes(),"K/V output mismatch");
    int n=k.size(0),hk=k.size(1),d=k.size(2);Rotation(rk,d);Rotation(rv,d);
    Cache(h,d,hk);Vec(slots,at::kLong,n,"slots");
    Same({h,slots,rk,rv,k,v});c10::DeviceGuard guard(k.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto hp=h.data_ptr(),sp=slots.data_ptr(),kr=rk.data_ptr(),vr=rv.data_ptr(),kp=k.data_ptr(),vp=v.data_ptr();
    int64_t stride=h.stride(0);int nb=h.size(0),bs=h.size(1);
    Launch("oscar_dequant_history",[=](){oscar_dequant_launch(stream,hp,sp,kr,vr,kp,vp,n,hk,d,bs,nb,stride);return 0;});
}
void Stage(const Tensor& k,const Tensor& v,const Tensor& seq,const Tensor& qsl,const Tensor& slots,
           Tensor sk,Tensor sv,Tensor owner,int64_t sink,int64_t recent) {
    Qkv(k,"key");Qkv(v,"value");TORCH_CHECK(k.sizes()==v.sizes(),"stage K/V mismatch");
    int n=k.size(0),hk=k.size(1),d=k.size(2);Check(seq,at::kInt,1,"seq_lens");
    int batch=seq.numel();Vec(qsl,at::kInt,batch+1,"qsl");Vec(slots,at::kLong,n,"slots");
    Check(sk,at::kBFloat16,4,"staging_k");Check(sv,at::kBFloat16,4,"staging_v");Check(owner,at::kLong,2,"staging_owner");
    TORCH_CHECK(sk.sizes()==sv.sizes()&&sk.size(2)==hk&&sk.size(3)==d&&owner.size(0)==sk.size(0)&&owner.size(1)==sk.size(1),
                "staging pool geometry mismatch");
    TORCH_CHECK(sink>=0&&recent>=0,"staging window bounds must be nonnegative");
    Same({k,v,seq,qsl,slots,sk,sv,owner});c10::DeviceGuard guard(k.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto kp=k.data_ptr(),vp=v.data_ptr(),ep=seq.data_ptr(),fp=qsl.data_ptr(),lp=slots.data_ptr();
    auto sp=sk.data_ptr(),up=sv.data_ptr(),op=owner.data_ptr();
    int rows=sk.size(0),bs=sk.size(1);
    Launch("oscar_stage_window",[=](){oscar_stage_launch(stream,kp,vp,ep,fp,lp,sp,up,op,batch,n,hk,d,int(sink),int(recent),rows,bs);return 0;});
}
void Restore(const Tensor& seq,const Tensor& qsl,const Tensor& map,const Tensor& epochs,const Tensor& bt,
             Tensor pos,Tensor wk,Tensor wv,Tensor state,const Tensor& sk,const Tensor& sv,const Tensor& owner,
             const Tensor& h,const Tensor& rk,const Tensor& rv,Tensor lossy,
             int64_t sink,int64_t recent,int64_t mp,int64_t tablebs) {
    int d=wk.size(3),hk=wk.size(2),w=wk.size(0),cap=wk.size(1);
    Dim(d);Check(wk,at::kBFloat16,4,"window_k");Check(wv,at::kBFloat16,4,"window_v");Check(pos,at::kInt,2,"window_positions");
    TORCH_CHECK(wk.sizes()==wv.sizes()&&pos.size(0)==w&&pos.size(1)==cap&&state.size(0)==w&&state.size(1)==4,"window geometry mismatch");
    Check(sk,at::kBFloat16,4,"staging_k");Check(sv,at::kBFloat16,4,"staging_v");Check(owner,at::kLong,2,"staging_owner");
    TORCH_CHECK(sk.sizes()==sv.sizes()&&sk.size(2)==hk&&sk.size(3)==d&&owner.size(0)==sk.size(0)&&owner.size(1)==sk.size(1),"staging pool geometry mismatch");
    Cache(h,d,hk);Rotation(rk,d);Rotation(rv,d);Vec(lossy,at::kInt,w,"lossy_rows");
    int batch=seq.numel();Check(seq,at::kInt,1,"seq_lens");Vec(qsl,at::kInt,batch+1,"qsl");Vec(map,at::kInt,batch,"row_to_window");
    Vec(epochs,at::kLong,batch,"epochs");Check(bt,at::kInt,2,"block_table");TORCH_CHECK(bt.size(0)==batch,"blocktable batch mismatch");
    TORCH_CHECK((sink>=0&&recent>=0&&mp>0&&mp<=16&&cap==sink+recent+mp&&tablebs>0),"invalid restore configuration");
    Same({seq,qsl,map,epochs,bt,pos,wk,wv,state,sk,sv,owner,h,rk,rv,lossy});
    c10::DeviceGuard guard(wk.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto ep=seq.data_ptr(),fp=qsl.data_ptr(),gp=map.data_ptr(),ip=epochs.data_ptr(),jp=bt.data_ptr();
    auto pp=pos.data_ptr(),op=wk.data_ptr(),vp=wv.data_ptr(),sp=state.data_ptr();
    auto kp=sk.data_ptr(),up=sv.data_ptr(),np=owner.data_ptr(),hp=h.data_ptr();
    auto kr=rk.data_ptr(),vr=rv.data_ptr(),zp=lossy.data_ptr();
    int rows=sk.size(0),sbs=sk.size(1),pages=bt.size(1),blocks=h.size(0),hbs=h.size(1);
    TORCH_CHECK(hbs==sbs,"staging and history must share one page granularity");
    int64_t stride=h.stride(0);
    Launch("oscar_prefix_restore",[=](){oscar_restore_launch(stream,ep,fp,gp,ip,jp,pp,op,vp,sp,kp,up,np,hp,kr,vr,zp,
        batch,hk,d,w,pages,int(sink),int(recent),int(mp),rows,sbs,int(tablebs),blocks,stride);return 0;});
}
void State(const Tensor& k,const Tensor& v,const Tensor& seq,const Tensor& qsl,const Tensor& map,const Tensor& epochs,
 const Tensor& bt,const Tensor& slots,const Tensor& prefill,Tensor wk,Tensor wv,Tensor pos,Tensor state,Tensor hs,Tensor he,
 Tensor qp,Tensor mk,Tensor mv,Tensor ms,Tensor cs,int64_t phase,int64_t s,int64_t r,int64_t mp,int64_t bs) {
    Qkv(k,"key");Qkv(v,"value");TORCH_CHECK(k.sizes()==v.sizes(),"state K/V mismatch");Check(seq,at::kInt,1,"seq_lens");
    int b=seq.numel(),n=k.size(0),hk=k.size(1),d=k.size(2);Vec(qsl,at::kInt,b+1,"qsl");Vec(map,at::kInt,b,"row_to_window");
    Vec(epochs,at::kLong,b,"epochs");Check(bt,at::kInt,2,"block_table");TORCH_CHECK(bt.size(0)==b,"blocktable batch mismatch");
    Vec(slots,at::kLong,n,"native_slots");Vec(prefill,at::kBool,b,"is_prefill");
    Check(wk,at::kBFloat16,4,"window_k");Check(wv,at::kBFloat16,4,"window_v");Check(pos,at::kInt,2,"window_positions");Check(state,at::kLong,2,"state");
    TORCH_CHECK((phase==0||phase==1)&&s>=0&&r>=0&&mp>0&&mp<=16&&bs>0,"invalid window configuration");
    TORCH_CHECK(wk.sizes()==wv.sizes()&&wk.size(1)==s+r+mp&&wk.size(2)==hk&&wk.size(3)==d,"window shape mismatch");
    int w=wk.size(0);TORCH_CHECK(pos.size(0)==w&&pos.size(1)==s+r+mp&&state.size(0)==w&&state.size(1)==4,"window state shape mismatch");
    Vec(hs,at::kInt,b,"history_start");Vec(he,at::kInt,b,"history_end");Vec(qp,at::kInt,n,"query_positions");
    Qkv(mk,"migration_k");Qkv(mv,"migration_v");
    TORCH_CHECK(mk.sizes()==mv.sizes()&&mk.size(0)==b*(r+mp)&&mk.size(1)==hk&&mk.size(2)==d,"migration buffer shape mismatch");
    Vec(ms,at::kLong,b*(r+mp),"migration_slots");Vec(cs,at::kLong,n,"current_slots");
    Same({k,v,seq,qsl,map,epochs,bt,slots,prefill,wk,wv,pos,state,hs,he,qp,mk,mv,ms,cs});c10::DeviceGuard guard(k.device());
    auto stream=c10_npu::getCurrentNPUStream().stream();
    auto a=k.data_ptr(),c=v.data_ptr(),e=seq.data_ptr(),f=qsl.data_ptr(),g=map.data_ptr(),i=epochs.data_ptr(),j=bt.data_ptr(),l=slots.data_ptr(),m=prefill.data_ptr();
    auto o=wk.data_ptr(),u=wv.data_ptr(),z=pos.data_ptr(),sta=state.data_ptr(),hsp=hs.data_ptr(),hep=he.data_ptr(),qpp=qp.data_ptr(),mkp=mk.data_ptr(),mvp=mv.data_ptr(),msp=ms.data_ptr(),csp=cs.data_ptr();
    int pages=bt.size(1);
    Launch("oscar_window_state",[=](){oscar_window_state_launch(stream,a,c,e,f,g,i,j,l,m,o,u,z,sta,hsp,hep,qpp,mkp,mvp,msp,csp,b,hk,d,w,pages,phase,s,r,mp,bs);return 0;});
}
}

TORCH_LIBRARY(oscar_ascend,m) {
    m.def("abi_version() -> int",[]()->int64_t{return 2;});
    m.def("workspace_size(int n, int hq, int d, int splits) -> int",Workspace);
    m.def("rotate_out(Tensor x, Tensor rotation, Tensor(a!) out, bool transpose=False) -> ()");
    m.def("store_int2(Tensor key, Tensor value, Tensor rk, Tensor rv, Tensor slots, Tensor(a!) history, float kclip, float vclip) -> ()");
    m.def("history_attention_out(Tensor qrot, Tensor history, Tensor block_table, Tensor qsl, Tensor history_start, Tensor history_end, Tensor query_positions, Tensor(a!) output_rot, Tensor(b!) lse, Tensor(c!) workspace, float scale, int splits, int max_query_len, int block_table_block_size=0) -> ()");
    m.def("window_attention_out(Tensor query, Tensor key, Tensor value, Tensor window_k, Tensor window_v, Tensor window_positions, Tensor row_to_window, Tensor qsl, Tensor query_positions, Tensor(a!) output, Tensor(b!) lse, float scale, Tensor(c!) workspace, int max_query_len) -> ()");
    m.def("merge_out(Tensor history_output, Tensor history_lse, Tensor window_output, Tensor window_lse, Tensor rv, Tensor(a!) output) -> ()");
    m.def("zero_blocks_out(Tensor(a!) history, Tensor block_ids) -> ()");
    m.def("dequant_history_out(Tensor history, Tensor slots, Tensor rk, Tensor rv, Tensor(a!) key, Tensor(b!) value) -> ()");
    m.def("stage_window_out(Tensor key, Tensor value, Tensor seq_lens, Tensor qsl, Tensor slots, Tensor(a!) staging_k, Tensor(b!) staging_v, Tensor(c!) owner, int sink, int recent) -> ()");
    m.def("prefix_restore_out(Tensor seq_lens, Tensor qsl, Tensor row_to_window, Tensor epochs, Tensor block_table, Tensor(a!) window_positions, Tensor(b!) window_k, Tensor(c!) window_v, Tensor(d!) state, Tensor staging_k, Tensor staging_v, Tensor owner, Tensor history, Tensor rk, Tensor rv, Tensor(e!) lossy_rows, int sink, int recent, int max_pending, int block_table_block_size) -> ()");
    m.def("window_state_out(Tensor key, Tensor value, Tensor seq_lens, Tensor qsl, Tensor row_to_window, Tensor epochs, Tensor block_table, Tensor native_slots, Tensor is_prefill, Tensor(a!) window_k, Tensor(b!) window_v, Tensor(c!) window_positions, Tensor(d!) state, Tensor(e!) history_start, Tensor(f!) history_end, Tensor(g!) query_positions, Tensor(h!) migration_k, Tensor(i!) migration_v, Tensor(j!) migration_slots, Tensor(k!) current_slots, int phase, int sink, int recent, int max_pending, int block_size) -> ()");
}
TORCH_LIBRARY_IMPL(oscar_ascend,PrivateUse1,m) {
    m.impl("rotate_out",TORCH_FN(Rotate));m.impl("store_int2",TORCH_FN(Store));
    m.impl("history_attention_out",TORCH_FN(History));m.impl("window_attention_out",TORCH_FN(Window));
    m.impl("merge_out",TORCH_FN(Merge));m.impl("zero_blocks_out",TORCH_FN(Zero));m.impl("window_state_out",TORCH_FN(State));
    m.impl("dequant_history_out",TORCH_FN(Dequant));m.impl("stage_window_out",TORCH_FN(Stage));
    m.impl("prefix_restore_out",TORCH_FN(Restore));
}
