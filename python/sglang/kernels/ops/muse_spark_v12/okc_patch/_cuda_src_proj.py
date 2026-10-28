"""muse_bf16_proj v5 CUDA source (the certified kernel-track lane): the m == 1 GEMV and the deterministic router (the
T = 1 GEMV stores per-K-slice partials, reduce_k<2> sums them in a fixed order). The lane's m 2..8 MMA-pack and 13-bit
streams are not carried: smallm serves those m off the resident weight."""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

// Programmatic dependent launch: no-ops unless launched with the programmatic-serialization attribute (proj_set_pdl).
// A kernel waits before reading its activations and before any global store; constant weights may be read earlier.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// bf16 pair (packed in a u32) -> two fp32, 1 int op each.
__device__ __forceinline__ float blo(unsigned u) { return __int_as_float(u << 16); }
__device__ __forceinline__ float bhi(unsigned u) { return __int_as_float(u & 0xffff0000u); }

__device__ __forceinline__ float dot8(uint4 w, uint4 x, float acc) {
  acc = fmaf(blo(w.x), blo(x.x), acc);
  acc = fmaf(bhi(w.x), bhi(x.x), acc);
  acc = fmaf(blo(w.y), blo(x.y), acc);
  acc = fmaf(bhi(w.y), bhi(x.y), acc);
  acc = fmaf(blo(w.z), blo(x.z), acc);
  acc = fmaf(bhi(w.z), bhi(x.z), acc);
  acc = fmaf(blo(w.w), blo(x.w), acc);
  acc = fmaf(bhi(w.w), bhi(x.w), acc);
  return acc;
}

// out[i, n] = sum_k x[i, k] * W[n, k];  warp owns rows [wid*R, wid*R+R), full K.
// U = k-steps per load group (all U*R W loads + U*MT x loads issued back-to-back
// for latency hiding); requires kvec % (32*U) == 0 (python-guarded).
template <int MT, int R, int U>
__global__ void gemv_ffma_k(const uint4* __restrict__ W, const uint4* __restrict__ X,
                            __nv_bfloat16* __restrict__ out, int m, int N, int kvec) {
  pdl_wait();
  pdl_trigger();
  int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  size_t row0 = (size_t)wid * R;
  if (row0 >= (size_t)N) return;
  const uint4* wp = W + row0 * kvec;
  float acc[2][R][MT];
#pragma unroll
  for (int p = 0; p < 2; p++)
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
      for (int i = 0; i < MT; i++) acc[p][r][i] = 0.f;

  for (int kv = lane; kv < kvec; kv += 32 * U) {
    uint4 wv[U][R], xf[U][MT];
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int r = 0; r < R; r++) wv[u][r] = __ldcs(wp + (size_t)r * kvec + kv + 32 * u);
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int i = 0; i < MT; i++)
        xf[u][i] = (MT == 1 || i < m) ? __ldg(X + (size_t)i * kvec + kv + 32 * u)
                                      : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int i = 0; i < MT; i++)
          acc[u & 1][r][i] = dot8(wv[u][r], xf[u][i], acc[u & 1][r][i]);
  }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int i = 0; i < MT; i++) {
      float v = acc[0][r][i] + acc[1][r][i];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == i && (MT == 1 || i < m))
        out[(size_t)i * N + row0 + r] = __float2bfloat16(v);
    }
}

// Router: Wp = [2*NR, K] bf16, rows 2j / 2j+1 = hi_j / lo_j. Warp (j, sk) owns the
// pair over K-slice sk and stores its fp32 partial to out = P[SK][m][NR]; reduce_k<2>
// sums the slices in a fixed order (no atomics: every TP rank gets the same logits).
template <int MT, int U>
__global__ void gemv_router_k(const uint4* __restrict__ Wp, const uint4* __restrict__ X,
                              float* __restrict__ out, int m, int NR, int kvec, int SK) {
  int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  int j = wid / SK, sk = wid - j * SK;
  if (j >= NR) return;
  int kchunk = kvec / SK;
  int k0 = sk * kchunk;
  const uint4* hp = Wp + (size_t)(2 * j) * kvec + k0;
  const uint4* lp = hp + kvec;
  float acc[2][MT];
#pragma unroll
  for (int p = 0; p < 2; p++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[p][i] = 0.f;

  for (int kv = lane; kv < kchunk; kv += 32 * U) {
    uint4 hv[U], lv[U], xf[U][MT];
#pragma unroll
    for (int u = 0; u < U; u++) {
      hv[u] = __ldcs(hp + kv + 32 * u);
      lv[u] = __ldcs(lp + kv + 32 * u);
    }
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int i = 0; i < MT; i++)
        xf[u][i] = (MT == 1 || i < m) ? __ldg(X + (size_t)i * kvec + k0 + kv + 32 * u)
                                      : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int i = 0; i < MT; i++)
        acc[u & 1][i] = dot8(lv[u], xf[u][i], dot8(hv[u], xf[u][i], acc[u & 1][i]));
  }
#pragma unroll
  for (int i = 0; i < MT; i++) {
    float v = acc[0][i] + acc[1][i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == i && (MT == 1 || i < m)) out[((size_t)sk * m + i) * NR + j] = v;
  }
}

// FOLD=0: out bf16 [m,NOUT] = sum_sk P. FOLD=1 (router): out fp32 [m,NOUT/2],
// out[i][j] = sum_sk P[sk][i][j] + P[sk][i][j+NOUT/2]. FOLD=2 (router): out fp32
// [m,NOUT] = sum_sk P, sk = 0, 1, ... in order.
template <int FOLD>
__global__ void reduce_k(const float* __restrict__ P, void* __restrict__ out, int m, int NOUT,
                         int SK) {
  pdl_wait();
  pdl_trigger();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t plane = (size_t)m * NOUT;
  if (FOLD == 0) {
    int total = (int)(plane >> 1);
    if (tid >= total) return;
    const float2* p2 = reinterpret_cast<const float2*>(P);
    float2 s = make_float2(0.f, 0.f);
    for (int si = 0; si < SK; si++) {
      float2 v = p2[(plane >> 1) * si + tid];
      s.x += v.x;
      s.y += v.y;
    }
    reinterpret_cast<__nv_bfloat162*>(out)[tid] = __float22bfloat162_rn(s);
  } else if (FOLD == 1) {
    int half = NOUT >> 1;
    int total = m * half;
    if (tid >= total) return;
    int i = tid / half, j = tid - i * half;
    float s = 0.f;
    for (int si = 0; si < SK; si++) {
      const float* p = P + plane * si + (size_t)i * NOUT;
      s += p[j] + p[j + half];
    }
    reinterpret_cast<float*>(out)[(size_t)i * half + j] = s;
  } else {
    int total = m * NOUT;
    if (tid >= total) return;
    float s = 0.f;
#pragma unroll 16
    for (int si = 0; si < SK; si++) s += P[plane * si + tid];
    reinterpret_cast<float*>(out)[tid] = s;
  }
}

static bool g_pdl = false;
void proj_set_pdl(int64_t on) { g_pdl = on != 0; }

template <typename... KArgs, typename... Args>
static void plaunch(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = g_pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...));
}

void gemv_bf16_launch(at::Tensor W, at::Tensor X, at::Tensor out, int64_t m, int64_t n,
                      int64_t k, int64_t MT, int64_t R, int64_t U, int64_t WPC) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int kvec = (int)(k >> 3);
  dim3 grid((unsigned)(n / R / WPC)), block((unsigned)(WPC * 32));
  const uint4* w = reinterpret_cast<const uint4*>(W.data_ptr());
  const uint4* x = reinterpret_cast<const uint4*>(X.data_ptr());
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define GCASE(mt, r, u)                                                               \
  if (MT == mt && R == r && U == u) {                                                 \
    plaunch(gemv_ffma_k<mt, r, u>, grid, block, 0, stream, w, x, o, (int)m, (int)n, kvec); \
    return;                                                                           \
  }
  GCASE(1, 1, 2) GCASE(1, 1, 4) GCASE(1, 1, 8)
  TORCH_CHECK(false, "no gemv instantiation");
}

void reduce_launch(at::Tensor P, at::Tensor out, int64_t m, int64_t nout, int64_t SK,
                   int64_t FOLD) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int total = (int)(FOLD == 1 ? m * (nout >> 1) : FOLD == 2 ? m * nout : (m * nout) >> 1);
  dim3 grid((unsigned)((total + 255) / 256)), block(256u);
  const float* p = reinterpret_cast<const float*>(P.data_ptr());
  if (FOLD == 2)
    plaunch(reduce_k<2>, grid, block, 0, stream, p, out.data_ptr(), (int)m, (int)nout, (int)SK);
  else if (FOLD)
    plaunch(reduce_k<1>, grid, block, 0, stream, p, out.data_ptr(), (int)m, (int)nout, (int)SK);
  else
    plaunch(reduce_k<0>, grid, block, 0, stream, p, out.data_ptr(), (int)m, (int)nout, (int)SK);
}

void gemv_router_launch(at::Tensor Wp, at::Tensor X, at::Tensor out, int64_t m, int64_t k,
                        int64_t MT, int64_t SK, int64_t U, int64_t WPC) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int kvec = (int)(k >> 3);
  int NR = (int)(Wp.size(0) / 2);
  dim3 grid((unsigned)((int64_t)NR * SK / WPC)), block((unsigned)(WPC * 32));
  const uint4* w = reinterpret_cast<const uint4*>(Wp.data_ptr());
  const uint4* x = reinterpret_cast<const uint4*>(X.data_ptr());
  auto* o = reinterpret_cast<float*>(out.data_ptr());
#define RCASE(mt, u)                                                                        \
  if (MT == mt && U == u) {                                                                 \
    gemv_router_k<mt, u><<<grid, block, 0, stream>>>(w, x, o, (int)m, NR, kvec, (int)SK);   \
    return;                                                                                 \
  }
  RCASE(1, 1) RCASE(1, 2) RCASE(1, 4)
  TORCH_CHECK(false, "no router instantiation");
}
"""
