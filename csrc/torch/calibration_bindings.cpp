#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../include/oscar_calibration.h"

namespace {
using at::Tensor;
void Check(const Tensor& t,at::ScalarType dtype,int rank,const char* name,bool contiguous=true) {
    TORCH_CHECK(t.device().type()==c10::DeviceType::PrivateUse1,name," must be on NPU");
    TORCH_CHECK(t.scalar_type()==dtype&&t.dim()==rank,name," dtype/rank mismatch");
    TORCH_CHECK(!contiguous||t.is_contiguous(),name," must be contiguous");
}
void Same(std::initializer_list<Tensor> tensors) {
    const auto device=tensors.begin()->device();for(const auto& t:tensors)TORCH_CHECK(t.device()==device,"calibration tensors must share one NPU");
}
int Dim(int64_t d) {TORCH_CHECK(d>=2&&d<=256&&(d&(d-1))==0,"calibration D must be a power of two in [2,256]");return d;}
void Raw(const Tensor& x,const char* name) {
    Check(x,at::kBFloat16,3,name,false);Dim(x.size(2));
    TORCH_CHECK(x.size(0)>=0&&x.size(0)<=(1<<24)&&x.size(1)>0&&x.size(1)<=256,"calibration raw shape exceeds ABI limits");
    for(int i=0;i<3;++i)TORCH_CHECK(x.stride(i)>0,"calibration supports positive raw strides only");
}
void Vector(const Tensor& x,at::ScalarType type,int64_t n,const char* name) {Check(x,type,1,name);TORCH_CHECK(x.numel()==n,name," length mismatch");}
void Matrix(const Tensor& x,int d,const char* name) {Check(x,at::kFloat,2,name);TORCH_CHECK(x.size(0)==d&&x.size(1)==d,name," must be [D,D]");}
void Moments(const Tensor& x,int h,int d,const char* name) {Check(x,at::kFloat,3,name);TORCH_CHECK(x.size(0)==h&&x.size(1)==d&&x.size(2)==d,name," must be [Hkv,D,D]");}
int Divisor(int64_t value,int heads) {if(value==0)value=heads;TORCH_CHECK(value>=heads&&value<=4096,"global KV head divisor must cover local heads");return value;}
void NoOverlap(std::initializer_list<Tensor> tensors) {
    // Called only on contiguous small statistic/constant buffers, not raw KV.
    for(auto a=tensors.begin();a!=tensors.end();++a)for(auto b=a+1;b!=tensors.end();++b) {
        auto ab=reinterpret_cast<uintptr_t>(a->data_ptr()),bb=reinterpret_cast<uintptr_t>(b->data_ptr());
        auto ae=ab+a->numel()*a->element_size(),be=bb+b->numel()*b->element_size();
        TORCH_CHECK(ae<=bb||be<=ab,"calibration input/output buffers must not overlap");
    }
}
template<class F>void Launch(const char* name,F fn) {at_npu::native::OpCommand cmd;cmd.Name(name);cmd.SetCustomHandler(fn);cmd.Run();}
oscar::CalibrationShape QShape(const Tensor& q,int hk) {
    oscar::CalibrationShape s{};s.n=q.size(0);s.hq=q.size(1);s.hk=hk;s.d=q.size(2);
    s.qs0=q.stride(0);s.qs1=q.stride(1);s.qs2=q.stride(2);return s;
}
void SetKV(oscar::CalibrationShape& s,const Tensor& k,const Tensor& v) {
    s.ks0=k.stride(0);s.ks1=k.stride(1);s.ks2=k.stride(2);s.vs0=v.stride(0);s.vs1=v.stride(1);s.vs2=v.stride(2);
}
void QMoments(const Tensor& q,Tensor moments,Tensor counts) {
    Raw(q,"q");Check(moments,at::kFloat,3,"moments");int hk=moments.size(0),d=q.size(2);
    TORCH_CHECK(hk>0&&q.size(1)%hk==0,"calibration Q heads must divide into KV groups");
    Moments(moments,hk,d,"moments");Vector(counts,at::kLong,hk,"counts");Same({q,moments,counts});NoOverlap({moments,counts});
    c10::DeviceGuard guard(q.device());auto stream=c10_npu::getCurrentNPUStream().stream();auto s=QShape(q,hk);
    auto qp=q.data_ptr(),mp=moments.data_ptr(),cp=counts.data_ptr();
    Launch("oscar_calib_q_moments",[=](){oscar_calib_q_moments_launch(stream,qp,mp,cp,s);return 0;});
}
void QCov(const Tensor& moments,const Tensor& counts,Tensor per,Tensor global,int64_t total_heads) {
    Check(moments,at::kFloat,3,"moments");int h=moments.size(0),d=Dim(moments.size(1));TORCH_CHECK(h>0,"no calibration heads");
    Moments(moments,h,d,"moments");Vector(counts,at::kLong,h,"counts");Moments(per,h,d,"per_head");Matrix(global,d,"global_cov");
    Same({moments,counts,per,global});NoOverlap({moments,counts,per,global});int divisor=Divisor(total_heads,h);
    c10::DeviceGuard guard(moments.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto mp=moments.data_ptr(),cp=counts.data_ptr(),pp=per.data_ptr(),gp=global.data_ptr();
    Launch("oscar_calib_q_cov",[=](){oscar_calib_q_cov_launch(stream,mp,cp,pp,gp,h,d,divisor);return 0;});
}
void SSTMoments(const Tensor& k,const Tensor& v,const Tensor& per,Tensor weighted,Tensor sums,Tensor weights) {
    Raw(k,"k");Raw(v,"v");TORCH_CHECK(k.sizes()==v.sizes(),"calibration K/V shape mismatch");
    int h=k.size(1),d=k.size(2);Moments(per,h,d,"per_head_q_cov");Moments(weighted,h,d,"weighted_moments");Vector(sums,at::kFloat,h,"weight_sum");
    Check(weights,at::kFloat,2,"weights_workspace");TORCH_CHECK(weights.size(0)==k.size(0)&&weights.size(1)==h,"weights workspace shape mismatch");
    Same({k,v,per,weighted,sums,weights});NoOverlap({per,weighted,sums,weights});
    oscar::CalibrationShape s{};s.n=k.size(0);s.hk=h;s.d=d;SetKV(s,k,v);
    c10::DeviceGuard guard(k.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto kp=k.data_ptr(),vp=v.data_ptr(),pp=per.data_ptr(),mp=weighted.data_ptr(),sp=sums.data_ptr(),wp=weights.data_ptr();
    Launch("oscar_calib_sst_moments",[=](){oscar_calib_sst_moments_launch(stream,kp,vp,pp,mp,sp,wp,s);return 0;});
}
void SSTCov(const Tensor& weighted,const Tensor& sums,Tensor global,int64_t total_heads) {
    Check(weighted,at::kFloat,3,"weighted_moments");int h=weighted.size(0),d=Dim(weighted.size(1));TORCH_CHECK(h>0,"no calibration heads");
    Moments(weighted,h,d,"weighted_moments");Vector(sums,at::kFloat,h,"weight_sum");Matrix(global,d,"global_cov");
    Same({weighted,sums,global});NoOverlap({weighted,sums,global});int divisor=Divisor(total_heads,h);
    c10::DeviceGuard guard(weighted.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto mp=weighted.data_ptr(),sp=sums.data_ptr(),gp=global.data_ptr();
    Launch("oscar_calib_sst_cov",[=](){oscar_calib_sst_cov_launch(stream,mp,sp,gp,h,d,divisor);return 0;});
}
void Eigh(const Tensor& cov,Tensor rotation,Tensor evals,Tensor vectors,Tensor workspace,Tensor diag,int64_t sweeps,double tolerance) {
    Check(cov,at::kFloat,2,"covariance");int d=Dim(cov.size(0));Matrix(cov,d,"covariance");Matrix(rotation,d,"rotation");Matrix(vectors,d,"eigenvectors");Vector(evals,at::kFloat,d,"eigenvalues");
    Check(workspace,at::kFloat,3,"eigensolver workspace");TORCH_CHECK(workspace.size(0)==2&&workspace.size(1)==d&&workspace.size(2)==d,"eigensolver workspace must be [2,D,D]");
    Vector(diag,at::kFloat,8,"diagnostics");TORCH_CHECK(sweeps>0&&sweeps<=64&&std::isfinite(tolerance)&&tolerance>=1e-8&&tolerance<=1e-5,"invalid eigensolver convergence configuration");
    Same({cov,rotation,evals,vectors,workspace,diag});NoOverlap({cov,rotation,evals,vectors,workspace,diag});
    c10::DeviceGuard guard(cov.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto cp=cov.data_ptr(),rp=rotation.data_ptr(),ep=evals.data_ptr(),up=vectors.data_ptr(),wp=workspace.data_ptr(),dp=diag.data_ptr();
    Launch("oscar_calib_eigh_rhp",[=](){oscar_calib_eigh_rhp_launch(stream,cp,rp,ep,up,wp,dp,d,sweeps,tolerance);return 0;});
}
void Fingerprint(const Tensor& q,const Tensor& k,const Tensor& v,Tensor fp,Tensor partial,int64_t offset) {
    Raw(q,"q");Raw(k,"k");Raw(v,"v");TORCH_CHECK(k.sizes()==v.sizes()&&q.size(0)==k.size(0)&&q.size(2)==k.size(2)&&q.size(1)%k.size(1)==0,"fingerprint Q/K/V shape mismatch");
    Vector(fp,at::kLong,6,"fingerprint");Check(partial,at::kLong,2,"fingerprint partial");TORCH_CHECK(partial.size(0)==6&&partial.size(1)==32&&offset>=0,"fingerprint requires partial[6,32] and nonnegative token offset");
    Same({q,k,v,fp,partial});NoOverlap({fp,partial});auto s=QShape(q,k.size(1));SetKV(s,k,v);
    c10::DeviceGuard guard(q.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto qp=q.data_ptr(),kp=k.data_ptr(),vp=v.data_ptr(),fpv=fp.data_ptr(),pp=partial.data_ptr();
    Launch("oscar_calib_fingerprint",[=](){oscar_calib_fingerprint_launch(stream,qp,kp,vp,fpv,pp,s,offset);return 0;});
}
}
TORCH_LIBRARY_FRAGMENT(oscar_ascend,m) {
    m.def("calib_q_moments_out(Tensor q, Tensor(a!) moments, Tensor(b!) counts) -> ()");
    m.def("calib_q_cov_out(Tensor moments, Tensor counts, Tensor(a!) per_head, Tensor(b!) global_cov, int global_kv_heads=0) -> ()");
    m.def("calib_sst_moments_out(Tensor k, Tensor v, Tensor per_head_q_cov, Tensor(a!) weighted_moments, Tensor(b!) weight_sum, Tensor(c!) weights_workspace) -> ()");
    m.def("calib_sst_cov_out(Tensor weighted_moments, Tensor weight_sum, Tensor(a!) global_cov, int global_kv_heads=0) -> ()");
    m.def("calib_eigh_rhp_out(Tensor covariance, Tensor(a!) rotation, Tensor(b!) eigenvalues, Tensor(c!) eigenvectors, Tensor(d!) workspace, Tensor(e!) diagnostics, int max_sweeps=16, float tolerance=1e-6) -> ()");
    m.def("calib_fingerprint_out(Tensor q, Tensor k, Tensor v, Tensor(a!) fingerprint, Tensor(b!) partial, int token_offset) -> ()");
}
TORCH_LIBRARY_IMPL(oscar_ascend,PrivateUse1,m) {
    m.impl("calib_q_moments_out",TORCH_FN(QMoments));m.impl("calib_q_cov_out",TORCH_FN(QCov));
    m.impl("calib_sst_moments_out",TORCH_FN(SSTMoments));m.impl("calib_sst_cov_out",TORCH_FN(SSTCov));
    m.impl("calib_eigh_rhp_out",TORCH_FN(Eigh));m.impl("calib_fingerprint_out",TORCH_FN(Fingerprint));
}
