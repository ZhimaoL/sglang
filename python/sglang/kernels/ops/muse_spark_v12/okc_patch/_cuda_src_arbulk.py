"""CUDA source of the OKC bulk-copy 2-shot all-reduce (custom_all_reduce_v2 under OKC_SW_AR_BULK; lane-moe ARBULK,
muse-opt serving/kernels/moe-graft/bulk_ar_src.py): the verify ARs at 1.5-4 MiB (bs 12-32) as TMA bulk copies into the
peers' symmetric workspace with a release flag per piece, instead of push2's 16 B Lamport stores. Same sum as push2
(clear_pos_zero on every input word, fp32 over source ranks 0..7, bf16 RNE, clear_pos_zero): bit-identical outputs.
Epoch = workspace gen + 1 (graph-safe, no reset); needs all G CTAs of every rank co-resident (G <= SMs)."""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

namespace bar2 {
constexpr int W = 8;
constexpr int NTHR = 512;
constexpr int NB = 4;          // RS ring slots
constexpr int D = 2;           // RS stores in flight before the oldest is signalled

struct Params {
  const char* in;
  char* out;
  char* ws;
  char* peer_ws[W];
  long long S;                 // bytes, = W * NP * P
  long long cmax;              // slot stride (bytes)
  int rank, P, npmax, pdl;
};

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(unsigned bar, unsigned cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(unsigned bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned bar, unsigned parity) {
  asm volatile("{\n .reg .pred P1;\n LAB_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               " @P1 bra DONE;\n bra LAB_WAIT;\n DONE:\n}\n" :: "r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(unsigned dst, const void* src, unsigned bytes, unsigned bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* dst, unsigned src, unsigned bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst), "r"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_cg(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned cpz(unsigned w) { return w == 0u ? 0x8000u : w; }   // Lamport clear_pos_zero, kAtom 4
__device__ __forceinline__ float2 bf2f(unsigned w) {
  __nv_bfloat162 h = *reinterpret_cast<__nv_bfloat162*>(&w);
  return __bfloat1622float2(h);
}
__device__ __forceinline__ unsigned f2bf(float2 f) {
  __nv_bfloat162 h = __float22bfloat162_rn(f);
  return *reinterpret_cast<unsigned*>(&h);
}

__global__ void __launch_bounds__(NTHR, 1) ar2_bulk(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) unsigned long long bars[NB];
  __shared__ unsigned s_epoch, s_last;
  const int tid = threadIdx.x, r = p.rank, G = (int)gridDim.x, c = (int)blockIdx.x;
  if (p.pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // input and gen come from earlier kernels
  const long long C = p.S / W;
  const int P = p.P, NP = (int)(C / P);
  unsigned* lfl = reinterpret_cast<unsigned*>(p.ws + 2 * W * p.cmax);   // [rs W x npmax][ag W x npmax][gen][done]
  unsigned* lgen = lfl + 2 * W * p.npmax;
  if (tid == 0) {
    s_epoch = *reinterpret_cast<volatile unsigned*>(lgen) + 1u;
    for (int i = 0; i < NB; i++) mbar_init(smem_u32(&bars[i]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const unsigned epoch = s_epoch;

  // ---- RS: my partial of chunk q -> rank q's rs slot [r]
  if (tid == 0) {
    const int nt = ((W - 1) * NP - c + G - 1) / G;     // my tasks: t = c + i * G
    auto task = [&](int i, int& q, int& k) {
      const int t = c + i * G;
      q = (r + 1 + t % (W - 1)) % W;
      k = t / (W - 1);
    };
    for (int i = 0; i < NB && i < nt; i++) {
      int q, k; task(i, q, k);
      const unsigned b = smem_u32(&bars[i]);
      mbar_expect_tx(b, P);
      bulk_g2s(smem_u32(smem + i * P), p.in + q * C + (long long)k * P, P, b);
    }
    for (int i = 0; i < nt; i++) {
      const int s = i % NB;
      int q, k; task(i, q, k);
      mbar_wait(smem_u32(&bars[s]), (i / NB) & 1);
      bulk_s2g(p.peer_ws[q] + r * p.cmax + (long long)k * P, smem_u32(smem + s * P), P);
      bulk_commit();
      if (i >= D) {
        bulk_wait<D>();                                  // groups <= i - D complete: written, smem read
        asm volatile("fence.proxy.async;" ::: "memory");
        int q2, k2; task(i - D, q2, k2);
        st_release_sys(reinterpret_cast<unsigned*>(p.peer_ws[q2] + 2 * W * p.cmax) + r * p.npmax + k2, epoch);
        const int j = i - D + NB;                       // the freed slot ((i - D) % NB) takes task j
        if (j < nt) {
          int q3, k3; task(j, q3, k3);
          const int s3 = j % NB;
          const unsigned b = smem_u32(&bars[s3]);
          mbar_expect_tx(b, P);
          bulk_g2s(smem_u32(smem + s3 * P), p.in + q3 * C + (long long)k3 * P, P, b);
        }
      }
    }
    bulk_wait<0>();
    asm volatile("fence.proxy.async;" ::: "memory");
    for (int i = (nt > D ? nt - D : 0); i < nt; i++) {
      int q, k; task(i, q, k);
      st_release_sys(reinterpret_cast<unsigned*>(p.peer_ws[q] + 2 * W * p.cmax) + r * p.npmax + k, epoch);
    }
  }
  __syncthreads();

  // ---- reduce my chunk's pieces, keep them, fan them out
  unsigned char* stage = smem;                           // the RS ring is free now
  for (int k = c; k < NP; k += G) {
    if (tid < W && tid != r)
      while (ld_acquire_sys(lfl + tid * p.npmax + k) != epoch) {}
    __syncthreads();
    const long long off = r * C + (long long)k * P;
    for (int v = tid; v < P / 16; v += NTHR) {
      uint4 x[W];
#pragma unroll
      for (int i = 0; i < W; i++)
        x[i] = i == r ? ld_cg(p.in + off + 16 * v) : ld_cg(p.ws + i * p.cmax + (long long)k * P + 16 * v);
      unsigned o[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 acc;
#pragma unroll
        for (int i = 0; i < W; i++) {
          const unsigned w = cpz(j == 0 ? x[i].x : j == 1 ? x[i].y : j == 2 ? x[i].z : x[i].w);
          const float2 f = bf2f(w);
          acc.x = i == 0 ? f.x : acc.x + f.x;
          acc.y = i == 0 ? f.y : acc.y + f.y;
        }
        o[j] = cpz(f2bf(acc));
      }
      const uint4 ov = make_uint4(o[0], o[1], o[2], o[3]);
      *reinterpret_cast<uint4*>(p.out + off + 16 * v) = ov;
      *reinterpret_cast<uint4*>(stage + 16 * v) = ov;
    }
    __syncthreads();
    if (tid == 0) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      for (int i = 1; i < W; i++) {
        const int q = (r + i) % W;
        bulk_s2g(p.peer_ws[q] + W * p.cmax + r * p.cmax + (long long)k * P, smem_u32(stage), P);
      }
      bulk_commit();
      bulk_wait<0>();
      asm volatile("fence.proxy.async;" ::: "memory");
      for (int i = 1; i < W; i++) {
        const int q = (r + i) % W;
        st_release_sys(reinterpret_cast<unsigned*>(p.peer_ws[q] + 2 * W * p.cmax) + W * p.npmax + r * p.npmax + k, epoch);
      }
    }
    __syncthreads();
  }

  // ---- AG: the peers' reduced pieces from my ag slots to the output
  for (int t = c; t < (W - 1) * NP; t += G) {
    const int q = (r + 1 + t % (W - 1)) % W, k = t / (W - 1);
    if (tid == 0)
      while (ld_acquire_sys(lfl + W * p.npmax + q * p.npmax + k) != epoch) {}
    __syncthreads();
    const char* src = p.ws + W * p.cmax + q * p.cmax + (long long)k * P;
    char* dst = p.out + q * C + (long long)k * P;
    for (int v = tid; v < P / 16; v += NTHR)
      *reinterpret_cast<uint4*>(dst + 16 * v) = ld_cg(src + 16 * v);
    __syncthreads();
  }

  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(lgen + 1, 1u) == (unsigned)(G - 1);
    if (s_last) {
      lgen[1] = 0u;
      *reinterpret_cast<volatile unsigned*>(lgen) = epoch;
    }
  }
  if (p.pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
}  // namespace bar2

void ar2_bulk_launch(int64_t in, int64_t out, int64_t ws, std::vector<int64_t> peer_ws, int64_t rank, int64_t S,
                     int64_t P, int64_t cmax, int64_t npmax, int64_t G, int64_t pdl, int64_t stream) {
  static int configured = 0;
  const int smem = bar2::NB * (int)P;
  if (configured < smem) {
    cudaFuncSetAttribute(bar2::ar2_bulk, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = smem;
  }
  TORCH_CHECK(peer_ws.size() == bar2::W && S % (bar2::W * P) == 0 && P % 16 == 0 && S / bar2::W <= cmax &&
              S / bar2::W / P <= npmax, "ar2_bulk: bad shape");
  bar2::Params p;
  p.in = reinterpret_cast<const char*>(in);
  p.out = reinterpret_cast<char*>(out);
  p.ws = reinterpret_cast<char*>(ws);
  for (int i = 0; i < bar2::W; i++) p.peer_ws[i] = reinterpret_cast<char*>(peer_ws[i]);
  p.S = S; p.cmax = cmax; p.rank = (int)rank; p.P = (int)P; p.npmax = (int)npmax; p.pdl = pdl ? 1 : 0;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)G);
  cfg.blockDim = dim3(bar2::NTHR);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, bar2::ar2_bulk, p));
}

int64_t ar2_ws_bytes(int64_t cmax, int64_t npmax) { return 2 * bar2::W * cmax + (2 * bar2::W * npmax + 2) * 4; }
"""
