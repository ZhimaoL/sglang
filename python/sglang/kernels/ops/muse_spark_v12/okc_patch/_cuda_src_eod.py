"""CUDA source of the EOD prefill MoE path (OKC_EOD; muse-opt docs/serving/review-1031/EOD.md 5.2).

eod_gemm is pg_gemm<1,5,0,3,1,0> (the wide build's gg1) with two additions behind its EOD template bit set 11:
  - copiers: warps 1-3 of each CTA (idle in pg_gemm) push the own SP shard's route rows into every rank's symm-mem token
    images through peer pointers, in (local slot, route id) order; per slot each warp fences (fence.acq_rel.sys) and
    counts in a local epoch-monotonic counter, and the call's last warp for a slot adds 1 to that slot's counter on
    every rank (fence.acq_rel.sys, then red.relaxed.sys);
  - gates: before an item's first k-tile the TMA thread waits until its slot's counter reaches 8 x epoch (relaxed .sys
    loads, then fence.acq_rel.sys and fence.proxy.async.global). The epoch is a device word bumped once per call by the
    global align, so a captured graph re-reads it on every replay; counters never reset.
eod_galign: every rank derives every destination's layout from the gathered ids, the 8 ranks' replica tables and the
served replica cut (owner = pg_rep_split's rule), deterministically: routes sit in route-id order inside each
(rank, slot) bucket; outputs are the own layout in align_local's formats and this rank's push list.
eod_rep_cut: pg_rep_count + pg_rep_cut of split_replicas, without its split pass.
eod_source(base) builds the module from pg's _CUDA_SRC: pg's text up to its host section, eod_gemm (an edited copy of
pg_gemm) and these host entry points; anchors are matched as token sequences (any whitespace between tokens).
"""
import re

_PG_HEAD = 'template<int GATED,int NST,int MODE,int WSET=3,int PAIR64=1,int SPL=0>__global__ void __launch_bounds__(384,1)pg_gemm('
_EOD_HEAD = ('template<int GATED,int NST,int MODE,int WSET=3,int PAIR64=1,int SPL=0,int EOD=0>__global__ void __launch_bounds__(384,1)'
     'eod_gemm(')
_EDITS = (
    ('int NWT,int numel,int snake,__nv_bfloat16*m1,unsigned*m1_ep){',
     'int NWT,int numel,int snake,__nv_bfloat16*m1,unsigned*m1_ep,const __grid_constant__ EodArgs eod){'),
    ('const int wg=tid>>7;',
     'const int wg=((EOD&2)!=0)?__shfl_sync(0xFFFFFFFFu,tid>>7,0):(tid>>7);'),
    ('if(tid==0){int sl=0,rnd=0;',
     ('if constexpr((EOD&2)!=0){const int wu=__shfl_sync(0xFFFFFFFFu,tid>>5,0);if(wu>=1){eod_copy<EOD>(eod,wu);return;}}if(tid='
     '=0){int sl=0,rnd=0;unsigned eod_mask=0u;if constexpr((EOD&4)!=0){if(eod.trace!=nullptr){eod.trace[blockIdx.x]=eod_now();'
     'if(blockIdx.x==0){for(int q=0;q<8;q++)eod.trace[236 + q]=eod.chunk!=nullptr?(unsigned long long)eod_ld(eod.chunk + q):0u'
     'll;eod.trace[236 + 8]=eod.peer!=nullptr?(unsigned long long)eod_ld(eod.peer):0ull;eod.trace[236 + 9]=(unsigned long long'
     ')eod_ld(eod.done);}}}')),
    ('const char*ws=wsf +((size_t)e*NWT + wt)*(size_t)NKT*WSB;',
     'const char*ws=wsf +((size_t)e*NWT + wt)*(size_t)NKT*WSB;if constexpr((EOD&1)!=0){eod_gate<EOD>(eod,e,eod_mask);}'),
)
_CUDA_SRC_EOD_DEV = r"""
struct EodArgs{int*done;const int*peer;const int*rank_of;const int*needv;const int*chunk;int done_target;int nidx;
const int4*ent;const int*woff;const char*src;char*dst[8];const unsigned*rank_t0;unsigned long long*trace;
int*rcnt[8];unsigned*lcnt;int epoch;int ndst;const unsigned*ep;};
DINL int eod_ld(const int*p){int v;asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");return v;}
DINL unsigned long long eod_now(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));return t;}
DINL int eod_rl(const int*p){int v;asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");return v;}
DINL int4 eod_rl4(const int*p){int4 v;asm volatile("ld.relaxed.gpu.global.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p) : "memory");return v;}
DINL void eod_acq(){asm volatile("fence.acq_rel.gpu;\n\tfence.proxy.async.global;" ::: "memory");}
DINL void eod_acq_sys(){asm volatile("fence.acq_rel.sys;\n\tfence.proxy.async.global;" ::: "memory");}
DINL int eod_rls(const int*p){int v;asm volatile("ld.relaxed.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");return v;}
template<int EOD>DINL bool eod_ready(const EodArgs&a,int e){const int r=a.rank_of[e];bool ok=true;
{const int tgt=(a.ep!=nullptr)?a.ndst*(int)__ldg(a.ep):a.done_target;if(tgt>0)ok=ok&&(((EOD&8)!=0)?eod_rls(a.done + r):eod_rl(a.done + r))>=tgt;}if(a.peer!=nullptr)ok=ok&&eod_rl(a.peer)>=r + 1;
if(a.needv!=nullptr){const int4 n0=reinterpret_cast<const int4*>(a.needv)[2*e],n1=reinterpret_cast<const int4*>(a.needv)[2*e + 1];
const int4 c0=eod_rl4(a.chunk),c1=eod_rl4(a.chunk + 4);
ok=ok&&c0.x>=n0.x&&c0.y>=n0.y&&c0.z>=n0.z&&c0.w>=n0.w&&c1.x>=n1.x&&c1.y>=n1.y&&c1.z>=n1.z&&c1.w>=n1.w;}return ok;}
template<int EOD>DINL void eod_gate(const EodArgs&a,int e,unsigned&mask){if(e<32&&((mask>>e)&1u))return;
unsigned long long t0=0;if constexpr((EOD&4)!=0){t0=eod_now();}
while(!eod_ready<EOD>(a,e))__nanosleep(32);if constexpr((EOD&8)!=0)eod_acq_sys();else eod_acq();if(e<32)mask|=1u<<e;
if constexpr((EOD&4)!=0){if(a.trace!=nullptr){const unsigned long long t1=eod_now();if(t1 - t0>200ull){
const unsigned i=atomicAdd(reinterpret_cast<unsigned*>(a.trace + 256),1u);if(i<8192u){unsigned long long*q=a.trace + 256 + 1 + 3*i;
q[0]=((unsigned long long)blockIdx.x<<32)|(unsigned)e;q[1]=t0;q[2]=t1;}}}}}
template<int EOD>DINL void eod_signal(const EodArgs&a,int r,int lane,int w){
if constexpr((EOD&8)!=0){asm volatile("fence.acq_rel.sys;" ::: "memory");__syncwarp();
if(lane==0){unsigned old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(a.lcnt + r) : "memory");
if(old==(unsigned)(3*gridDim.x)*(a.ep!=nullptr?__ldg(a.ep):(unsigned)a.epoch) - 1u){asm volatile("fence.acq_rel.sys;" ::: "memory");
for(int d=0;d<a.ndst;d++)asm volatile("red.relaxed.sys.global.add.u32 [%0], 1;" :: "l"(a.rcnt[d] + r) : "memory");}
if constexpr((EOD&4)!=0){if(a.trace!=nullptr)a.trace[24833 + w*32 + r]=eod_now();}}}
else{__threadfence();__syncwarp();
if(lane==0){atomicAdd(a.done + r,1);if constexpr((EOD&4)!=0){if(a.trace!=nullptr)a.trace[24833 + w*32 + r]=eod_now();}}}}
DINL void eod_row(const EodArgs&a,const int4 en,int lane){const int t=en.z&0xFF;
char*b=a.dst[((unsigned)en.y)>>24] +(((unsigned)en.y&0xFFFFFFu)*(unsigned)(NKT*XB)+(unsigned)(t*128 +(((lane&7)^(t&7))<<4)));
const bool ok=en.x>=0;const char*s=a.src +((unsigned)(ok?en.x:0)*8192u +(unsigned)(lane*16));
#pragma unroll 1
for(int q0=0;q0<16;q0 +=4){uint4 v[4];
#pragma unroll
for(int u=0;u<4;u++)v[u]=ok?__ldg(reinterpret_cast<const uint4*>(s +(q0 + u)*512)):make_uint4(0u,0u,0u,0u);
#pragma unroll
for(int u=0;u<4;u++){const unsigned kt=(unsigned)((q0 + u)*4 +(lane>>3));*reinterpret_cast<uint4*>(b + kt*(unsigned)XB)=v[u];}}}
template<int EOD>DINL void eod_copy(const EodArgs&a,int wp){const int lane=threadIdx.x&31;const int w=blockIdx.x*3 + wp - 1;
const int k0=__shfl_sync(0xFFFFFFFFu,a.woff[w],0),k1=__shfl_sync(0xFFFFFFFFu,a.woff[w + 1],0);const unsigned long long t0=eod_now();int cur=0;
for(int k=k0;k<k1;k++){const int4 ev=a.ent[k];int4 en;en.x=__shfl_sync(0xFFFFFFFFu,ev.x,0);en.y=__shfl_sync(0xFFFFFFFFu,ev.y,0);
en.z=__shfl_sync(0xFFFFFFFFu,ev.z,0);en.w=0;const int r=(en.z>>8)&0xFF;for(;cur<r;cur++)eod_signal<EOD>(a,cur,lane,w);
if(a.rank_t0!=nullptr){const unsigned long long due=t0 + a.rank_t0[r];while(!__all_sync(0xFFFFFFFFu,eod_now()>=due)){}}
if(a.chunk!=nullptr){const int nd=(en.z>>16)&0xFF;if(nd>0){const int*f=a.chunk +(((unsigned)en.z)>>24);
while(!__all_sync(0xFFFFFFFFu,eod_ld(f)>=nd))__nanosleep(64);}}
eod_row(a,en,lane);}for(;cur<a.nidx;cur++)eod_signal<EOD>(a,cur,lane,w);}
"""
_CUDA_SRC_EOD = r"""
static int pg_num_sms(){static int n=0;if(!n)n=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;return n;}
template<typename... KArgs,typename... Args>static void pg_launch(void(*k)(KArgs...),dim3 grid,dim3 block,size_t smem,Args... args){
cudaLaunchConfig_t cfg={};cfg.gridDim=grid;cfg.blockDim=block;cfg.dynamicSmemBytes=smem;cfg.stream=at::cuda::getCurrentCUDAStream();
cfg.numAttrs=0;C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,k,static_cast<KArgs>(args)...));}
template<int EOD>static void eod_launch(const torch::Tensor&xg,const torch::Tensor&wsf,const float*gs,void*out,const torch::Tensor&be,
const torch::Tensor&live,int grid,int snake,const pgk::EodArgs&a){auto fn=pgk::eod_gemm<1,5,0,3,1,0,EOD>;constexpr int smem=pgk::smem_bytes<1,0>(5);
static bool set=false;if(!set){C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));set=true;}
const int NWT=wsf.size(1);TORCH_CHECK(wsf.size(2)==pgk::NKT&&wsf.size(3)==pgk::WSB,"eod: wsf layout");
pg_launch(fn,dim3(grid>0?grid:pg_num_sms()),dim3(384),smem,reinterpret_cast<const char*>(xg.data_ptr()),reinterpret_cast<const char*>(wsf.data_ptr()),
gs,out,(const int*)nullptr,be.data_ptr<int>(),live.data_ptr<int>(),NWT,0,snake,(__nv_bfloat16*)nullptr,(unsigned*)nullptr,a);}
void eod_gg1(torch::Tensor xg,torch::Tensor wsf,torch::Tensor gs,torch::Tensor inter,torch::Tensor block_experts,torch::Tensor live,
int64_t numel,int64_t mode,torch::Tensor args){TORCH_CHECK(args.device().is_cpu()&&args.scalar_type()==at::kLong&&args.numel()>=32,"eod: int64 cpu args");
const int64_t*v=args.data_ptr<int64_t>();pgk::EodArgs a;a.done=reinterpret_cast<int*>(v[0]);a.peer=reinterpret_cast<const int*>(v[1]);
a.rank_of=reinterpret_cast<const int*>(v[2]);a.needv=reinterpret_cast<const int*>(v[3]);a.chunk=reinterpret_cast<const int*>(v[4]);
a.done_target=(int)v[5];a.nidx=(int)v[6];a.ent=reinterpret_cast<const int4*>(v[7]);a.woff=reinterpret_cast<const int*>(v[8]);
a.src=reinterpret_cast<const char*>(v[9]);for(int i=0;i<8;i++)a.dst[i]=reinterpret_cast<char*>(v[10 + i]);
a.rank_t0=reinterpret_cast<const unsigned*>(v[18]);a.trace=reinterpret_cast<unsigned long long*>(v[19]);const int grid=(int)v[20];
for(int i=0;i<8;i++)a.rcnt[i]=reinterpret_cast<int*>(v[22 + i]);a.lcnt=reinterpret_cast<unsigned*>(v[30]);a.epoch=(int)(v[31]&0x7fffffff);
a.ndst=(int)(v[31]>>32);a.ep=reinterpret_cast<const unsigned*>(v[21]);
TORCH_CHECK(grid==0||grid==pg_num_sms(),"eod: copier lists assume one CTA per SM");
const int snake=(numel/8<=4096)?1:0;const float*gs_p=gs.data_ptr<float>();void*o=inter.data_ptr();
TORCH_CHECK(mode==11,"eod: the served build has mode 11 only");eod_launch<11>(xg,wsf,gs_p,o,block_experts,live,grid,snake,a);}
void eod_rep_cut(torch::Tensor ids,torch::Tensor tab,torch::Tensor cnt,torch::Tensor bc,torch::Tensor cut,int64_t sb){
const int n_rep=(int)tab.size(0)- 256,topk=(int)ids.size(1),n_routes=(int)ids.numel(),nb=(int)((ids.size(0)+ sb - 1)/sb);
const int smem=n_rep*nb*4;TORCH_CHECK(n_rep>=1&&n_rep<=64&&nb>=1&&smem<=48*1024&&cnt.numel()==256&&cut.numel()==256&&bc.numel()>=n_rep*nb&&ids.is_contiguous(),"eod_rep_cut: table / counts");
auto st=at::cuda::getCurrentCUDAStream();C10_CUDA_CHECK(cudaMemsetAsync(cnt.data_ptr<int>(),0,256*4,st));
pgk::pg_rep_count<<<nb,256,0,st>>>(ids.data_ptr<int>(),n_routes,topk,tab.data_ptr<int>(),n_rep,(int)sb,nb,cnt.data_ptr<int>(),bc.data_ptr<int>());
pgk::pg_rep_cut<<<1,1024,smem,st>>>(tab.data_ptr<int>(),n_rep,(int)sb,nb,cnt.data_ptr<int>(),bc.data_ptr<int>(),cut.data_ptr<int>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
namespace ga2b{
constexpr int NT=512,NWP=NT/32,CH=4096,PER=CH/NWP,NBMAX=512,NLMAX=64;
__device__ __forceinline__ int owner_of(const int*tb,const int*ct,int e,int t){const int v=tb[e];
return(((v>>11)&1)&&t<ct[e])?((v>>12)&7):((v>>8)&7);}
__global__ void __launch_bounds__(NT)ga2b_count(const int*__restrict__ ids,int numel,const int*__restrict__ tabs,
const int*__restrict__ cut,int NL,int*__restrict__ hist){__shared__ int tb[8*256],ct[256],h[NBMAX];const int t=threadIdx.x;
for(int i=t;i<8*256;i +=NT)tb[i]=tabs[i];if(t<256)ct[t]=cut[t];if(t<NBMAX)h[t]=0;__syncthreads();
const int lo=blockIdx.x*CH,hi=min(numel,lo + CH);
for(int i=lo + t;i<hi;i +=NT){const int e=ids[i];if((unsigned)e>=256u)continue;const int d=owner_of(tb,ct,e,i>>3);
atomicAdd(&h[d*NL +(tb[d*256 + e]&0xFF)],1);}__syncthreads();if(t<8*NL)hist[blockIdx.x*NBMAX + t]=h[t];}
// one CTA per 4096-route chunk: its prefix / totals / offsets / push bases from hist (every CTA redundantly), CTA 0 also
// writes the own layout's n_padded, block_experts and pads and bumps the epoch word; then the stable ranks and the writes
__global__ void __launch_bounds__(NT)ga2b_scatter(const int*__restrict__ ids,int numel,const int*__restrict__ tabs,
const int*__restrict__ cut,int NL,const int*__restrict__ hist,int nch,int me,int c0,int nown,int row0,int BM,int nblocks,
int W,int N,int*__restrict__ sid,int4*__restrict__ ent,int*__restrict__ npad,int*__restrict__ be,unsigned*__restrict__ ep){
extern __shared__ int sm[];const int B=8*NL;int*tb=sm;int*ct=tb + 8*256;int*hw=ct + 256;int*hr=hw + NWP*B;int*pre=hr + NWP*NL;
int*tot=pre + B;int*so=tot + B;int*pb=so + 8*(NL + 1);int*tos=pb + NL;const int t=threadIdx.x,w=t>>5,lane=t&31,c=blockIdx.x;
for(int i=t;i<8*256;i +=NT)tb[i]=tabs[i];if(t<256)ct[t]=cut[t];for(int i=t;i<NWP*B;i +=NT)hw[i]=0;for(int i=t;i<NWP*NL;i +=NT)hr[i]=0;
for(int b=t;b<B;b +=NT){int p=0,q=0;
#pragma unroll 8
for(int k=0;k<nch;k++){const int v=hist[k*NBMAX + b];q +=v;p +=k<c?v:0;}pre[b]=p;tot[b]=q;}
const bool own=c>=c0&&c<c0 + nown;
if(own&&t<NL){int a=0,q=0;for(int k=0;k<nown;k++){int v=0;for(int d=0;d<8;d++)v +=hist[(c0 + k)*NBMAX + d*NL + t];
if(k<c - c0)a +=v;q +=v;}pb[t]=a;tos[t]=q;}
__syncthreads();
if(t<8){int acc=0;for(int q=0;q<NL;q++){so[t*(NL + 1)+ q]=acc;acc +=(tot[t*NL + q]+ BM - 1)/BM*BM;}so[t*(NL + 1)+ NL]=acc;}
if(own&&t==32){int acc=0;for(int q=0;q<NL;q++){const int v=tos[q];pb[q]+=acc;acc +=v;}}
__syncthreads();
if(c==0){const int*som=so + me*(NL + 1);if(t==0){*npad=som[NL];if(ep!=nullptr)*ep +=1u;}
for(int b=t;b<nblocks;b +=NT){const int pos=b*BM;int e= -1;for(int q=0;q<NL;q++)if(pos>=som[q]&&pos<som[q + 1])e=q;be[b]=e;}
for(int q=0;q<NL;q++){const int n=tot[me*NL + q];for(int p=som[q]+ n + t;p<som[q + 1];p +=NT)sid[p]=numel;}}
const int lo=c*CH + w*PER;const unsigned lt=(1u<<lane)- 1u;int rg[PER/32],rr[PER/32],bb[PER/32];
#pragma unroll
for(int it=0;it<PER/32;it++){const int i=lo + it*32 + lane;const int e=i<numel?ids[i]: -1;int b= -1,s= -1;
if((unsigned)e<256u){const int d=owner_of(tb,ct,e,i>>3);s=tb[d*256 + e]&0xFF;b=d*NL + s;}bb[it]=b;
const unsigned m=__match_any_sync(0xFFFFFFFFu,b);const int base=b>=0?hw[w*B + b]:0;__syncwarp();rg[it]=base + __popc(m&lt);
if(b>=0&&lane==__ffs(m)- 1)hw[w*B + b]=base + __popc(m);__syncwarp();rr[it]=0;
if(own){const unsigned m2=__match_any_sync(0xFFFFFFFFu,s);const int b2=s>=0?hr[w*NL + s]:0;__syncwarp();rr[it]=b2 + __popc(m2&lt);
if(s>=0&&lane==__ffs(m2)- 1)hr[w*NL + s]=b2 + __popc(m2);__syncwarp();}}
__syncthreads();for(int b=t;b<B;b +=NT){int acc=0;for(int q=0;q<NWP;q++){const int v=hw[q*B + b];hw[q*B + b]=acc;acc +=v;}}
if(own&&t<NL){int acc=0;for(int q=0;q<NWP;q++){const int v=hr[q*NL + t];hr[q*NL + t]=acc;acc +=v;}}
__syncthreads();const int nq=N/W,nm=N%W;
#pragma unroll
for(int it=0;it<PER/32;it++){const int b=bb[it];if(b<0)continue;const int i=lo + it*32 + lane;const int d=b/NL,s=b - d*NL;
const int pos=so[d*(NL + 1)+ s]+ pre[b]+ hw[w*B + b]+ rg[it];if(d==me)sid[pos]=i;
if(own){const int j=pb[s]+ hr[w*NL + s]+ rr[it];const int q=j%W;
ent[j/W + q*nq + min(q,nm)]=make_int4((i>>3)- row0,(pos/BM)|(d<<24),(pos%BM)|(s<<8),0);}}}
}
void eod_galign(torch::Tensor ids,torch::Tensor tabs,torch::Tensor cut,int64_t me,int64_t S,int64_t BM,int64_t W,int64_t NL,
torch::Tensor hist,torch::Tensor npad,torch::Tensor be,torch::Tensor sid,torch::Tensor ent,torch::Tensor ep){
TORCH_CHECK(ids.is_cuda()&&ids.scalar_type()==torch::kInt32&&ids.is_contiguous()&&tabs.numel()==8*256&&cut.numel()>=256,"galign2b: inputs");
const int numel=(int)ids.numel(),nch=numel/ga2b::CH,nblocks=(int)be.numel(),N=(int)(S*8),c0=(int)(me*S*8/ga2b::CH),nown=N/ga2b::CH;
TORCH_CHECK(numel%ga2b::CH==0&&N%ga2b::CH==0&&nown>=1&&nown<=8&&NL>=1&&NL<=ga2b::NLMAX,"galign2b: shapes");
TORCH_CHECK(hist.numel()>=(long)nch*ga2b::NBMAX&&ent.numel()>=(long)N*4&&sid.numel()>=(long)nblocks*BM,"galign2b: buffers");
const int B=8*(int)NL;const size_t smem=(size_t)(8*256 + 256 + ga2b::NWP*B + ga2b::NWP*(int)NL + 2*B + 8*((int)NL + 1)+ 2*(int)NL)*4;
static bool set=false;if(!set){C10_CUDA_CHECK(cudaFuncSetAttribute(ga2b::ga2b_scatter,cudaFuncAttributeMaxDynamicSharedMemorySize,64*1024));set=true;}
auto st=at::cuda::getCurrentCUDAStream();unsigned*epp=ep.numel()?reinterpret_cast<unsigned*>(ep.data_ptr<int>()):nullptr;
ga2b::ga2b_count<<<nch,ga2b::NT,0,st>>>(ids.data_ptr<int>(),numel,tabs.data_ptr<int>(),cut.data_ptr<int>(),(int)NL,hist.data_ptr<int>());
ga2b::ga2b_scatter<<<nch,ga2b::NT,smem,st>>>(ids.data_ptr<int>(),numel,tabs.data_ptr<int>(),cut.data_ptr<int>(),(int)NL,
hist.data_ptr<int>(),nch,(int)me,c0,nown,(int)(me*S),(int)BM,nblocks,(int)W,N,sid.data_ptr<int>(),
reinterpret_cast<int4*>(ent.data_ptr<int>()),npad.data_ptr<int>(),be.data_ptr<int>(),epp);C10_CUDA_KERNEL_LAUNCH_CHECK();}
"""


_GAP = r"(?:\s|//[^\n]*\n|/\*(?:.|\n)*?\*/)"


def _pat(anchor):
    """Token pattern of an anchor: between tokens any whitespace or comments, and at least one between two words."""
    toks = re.findall(r"[A-Za-z0-9_]+|\S", anchor)
    out = []
    for k, t in enumerate(toks):
        if k:
            word = (toks[k - 1][-1].isalnum() or toks[k - 1][-1] == "_") and (t[0].isalnum() or t[0] == "_")
            out.append(_GAP + ("+" if word else "*"))
        out.append(re.escape(t))
    return "".join(out)


def _span(text, anchor, what):
    m = list(re.finditer(_pat(anchor), text))
    if len(m) != 1:
        raise RuntimeError(f"EOD source: anchor {what} matched {len(m)} times")
    return m[0].span()


def eod_source(base):
    """pg's _CUDA_SRC -> the EOD module source (pg text to its host section, eod_gemm, the EOD host entry points)."""
    i, _ = _span(base, _PG_HEAD, "pg_gemm head")
    j, _ = _span(base, "constexpr int AL_MAX=64;", "AL_MAX")
    h, _ = _span(base, "static int pg_num_sms(){", "host section")
    if not i < j < h:
        raise RuntimeError("EOD source: anchors out of order")
    gemm = base[i:j]
    for old, new in _EDITS:
        a, b = _span(gemm, old, old[:24])
        gemm = gemm[:a] + new + gemm[b:]
    a, b = _span(gemm, _PG_HEAD, "eod_gemm head")
    return base[:j] + _CUDA_SRC_EOD_DEV + gemm[:a] + _EOD_HEAD + gemm[b:] + base[j:h] + _CUDA_SRC_EOD
