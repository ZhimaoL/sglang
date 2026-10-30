"""OKC first-turn norm kernels (FIRSTTURN #1; okc_patch/ft_norm.py). Both reproduce the launches they replace operation
for operation, so their outputs are bit-identical.

qk_fold (OKC_QK_FOLD): the attention prep of an extend in one pass over the raw QKV projection output. It replaces
FusedQKNorm's Triton _qk_rms_norm_fwd_kernel (per-head RMS norm of q and k, optional interleaved rope, bf16 out)
followed by fused_fp8_qkv_kv_cache (E4M3 q, E4M3 K / V rows into the paged cache at cache_loc). The arithmetic is the
Triton kernel's as compiled by Triton 3.8 for sm_90 (PTX read 10-30):
- Per (token, head) row of 64, L = 64 / TREE lanes, lane t holding columns [TREE t, TREE t + TREE).
- Lane sum: f_i = fma(x_2i, x_2i, x_2i+1 * x_2i+1); TREE 8: (f0 + f1) + (f2 + f3), TREE 4: f0 + f1, TREE 2: f0.
- Then xor butterflies L / 2 .. 1 (own + partner), div.full.f32 by 64, + eps, rsqrt.approx.ftz, rstd * x.
- TREE follows Triton's layout for the launch's BLOCK_M: 8 for n_rows = 18 T >= 8192, 4 for >= 1024, else 2.
- Rope (is_neox False): both halves bf16-rounded, p = x_r[c ^ 1] * sin, out = fma(x_r[c], cos, -p) for even c and
  fma(x_r[c], cos, p) for odd c.
- Then bf16, then E4M3 RN satfinite (x 1.0 scales). v rows go straight from bf16 to E4M3.

shard_prenorm (OKC_SHARD_PRENORM): F.rms_norm(x, (4096,), bf16 weight, eps) on [rows, 4096] bf16, four rows per
512-thread CTA. Each row's 128 threads follow torch's vectorized rms_norm reduction (as _cuda_src_glue.torch_rstd):
fma over the bf16x4 vectors l + 128 j in order, shfl_down 16 .. 1, warps (0 + 2) + (1 + 3), / 4096, rsqrtf(. + eps),
out = bf16(w * (rstd * x)).
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace okcf {

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float rsqrt_ftz(float a) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ unsigned char e4m3(float x) {
  return (unsigned char)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

template <int TREE>
__device__ __forceinline__ float lane_sum(const float (&x)[TREE]) {
  float f[TREE / 2];
#pragma unroll
  for (int i = 0; i < TREE / 2; i++) f[i] = __fmaf_rn(x[2 * i], x[2 * i], __fmul_rn(x[2 * i + 1], x[2 * i + 1]));
  if constexpr (TREE == 8) return __fadd_rn(__fadd_rn(f[0], f[1]), __fadd_rn(f[2], f[3]));
  else if constexpr (TREE == 4) return __fadd_rn(f[0], f[1]);
  else return f[0];
}

template <int N> struct Vec;
template <> struct Vec<8> { using in = uint4; using out = uint2; };
template <> struct Vec<4> { using in = uint2; using out = unsigned; };
template <> struct Vec<2> { using in = unsigned; using out = unsigned short; };

// rows = T x 20 (q 16, k 2, v 2 heads of 64); RW rows per warp, grid-stride over warps
template <int TREE, bool ROPE, bool QFP8, typename IdxT>
__global__ void __launch_bounds__(256) qkfold_k(
    const __nv_bfloat16* __restrict__ qkv, int ld, int T, const int64_t* __restrict__ pos,
    const float* __restrict__ cs, float eps, void* __restrict__ q_out, unsigned char* __restrict__ kc, long long kcs,
    unsigned char* __restrict__ vc, long long vcs, const IdxT* __restrict__ loc) {
  constexpr int L = 64 / TREE, RW = 32 / L;
  using VI = typename Vec<TREE>::in;
  using VO = typename Vec<TREE>::out;
  const int lane = threadIdx.x & 31, sub = lane / L, t = lane % L, c0 = TREE * t;
  const int nwarp = gridDim.x * (blockDim.x >> 5);
  const int nrows = T * 20;
  for (int r0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RW; r0 < nrows; r0 += nwarp * RW) {
    const int row = r0 + sub;
    const bool live = row < nrows;
    const int tok = live ? row / 20 : 0, h = live ? row - 20 * (row / 20) : 0;
    float x[TREE];
    {
      VI v = {};
      if (live) v = *reinterpret_cast<const VI*>(qkv + (size_t)tok * ld + h * 64 + c0);
      const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
      for (int i = 0; i < TREE; i++) x[i] = __bfloat162float(b[i]);
    }
    float s = lane_sum<TREE>(x);
#pragma unroll
    for (int off = L / 2; off > 0; off >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, off));
    if (!live) continue;
    const float rstd = rsqrt_ftz(__fadd_rn(eps, div_full(s, 64.0f)));
    float o[TREE];
    if (h < 18) {
#pragma unroll
      for (int i = 0; i < TREE; i++) o[i] = __fmul_rn(rstd, x[i]);
      if constexpr (ROPE) {
        const long long p = pos[tok];
        const float* cp = cs + p * 64 + c0 / 2;
        float xr[TREE];
#pragma unroll
        for (int i = 0; i < TREE; i++) xr[i] = bfr(o[i]);
#pragma unroll
        for (int i = 0; i < TREE; i += 2) {
          const float cv = cp[i / 2], sv = cp[32 + i / 2];
          o[i] = __fmaf_rn(xr[i], cv, -__fmul_rn(xr[i + 1], sv));
          o[i + 1] = __fmaf_rn(xr[i + 1], cv, __fmul_rn(xr[i], sv));
        }
      }
    }
    if (h < 16) {
      if constexpr (QFP8) {
        VO w;
        unsigned char* wb = reinterpret_cast<unsigned char*>(&w);
#pragma unroll
        for (int i = 0; i < TREE; i++) wb[i] = e4m3(bfr(o[i]));
        *reinterpret_cast<VO*>(static_cast<unsigned char*>(q_out) + (size_t)tok * 1024 + h * 64 + c0) = w;
      } else {
        VI w;
        __nv_bfloat16* wb = reinterpret_cast<__nv_bfloat16*>(&w);
#pragma unroll
        for (int i = 0; i < TREE; i++) wb[i] = __float2bfloat16_rn(o[i]);
        *reinterpret_cast<VI*>(static_cast<__nv_bfloat16*>(q_out) + (size_t)tok * 1024 + h * 64 + c0) = w;
      }
    } else {
      const long long slot = (long long)loc[tok];
      VO w;
      unsigned char* wb = reinterpret_cast<unsigned char*>(&w);
      if (h < 18) {
#pragma unroll
        for (int i = 0; i < TREE; i++) wb[i] = e4m3(bfr(o[i]));
        *reinterpret_cast<VO*>(kc + slot * kcs + (h - 16) * 64 + c0) = w;
      } else {
#pragma unroll
        for (int i = 0; i < TREE; i++) wb[i] = e4m3(x[i]);
        *reinterpret_cast<VO*>(vc + slot * vcs + (h - 18) * 64 + c0) = w;
      }
    }
  }
}

// F.rms_norm(x, (4096,), w, eps): 4 rows per 512-thread CTA, each row on 128 threads as torch's kernel
__global__ void __launch_bounds__(512) prenorm_k(const __nv_bfloat16* __restrict__ xin,
                                                 const __nv_bfloat16* __restrict__ w, float eps,
                                                 __nv_bfloat16* __restrict__ out, int rows) {
  constexpr int NJ = 8, N = 4096;
  __shared__ float red[4][4];
  const int g = threadIdx.x >> 7, l = threadIdx.x & 127;
  const int row = blockIdx.x * 4 + g;
  const bool live = row < rows;
  float v[NJ][4];
  const uint2* xr = reinterpret_cast<const uint2*>(xin + (size_t)(live ? row : 0) * N);
#pragma unroll
  for (int j = 0; j < NJ; j++) {
    const uint2 a = live ? xr[l + 128 * j] : make_uint2(0u, 0u);
    v[j][0] = __uint_as_float(a.x << 16);
    v[j][1] = __uint_as_float(a.x & 0xFFFF0000u);
    v[j][2] = __uint_as_float(a.y << 16);
    v[j][3] = __uint_as_float(a.y & 0xFFFF0000u);
  }
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < NJ; j++)
#pragma unroll
    for (int i = 0; i < 4; i++) s = __fmaf_rn(v[j][i], v[j][i], s);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xFFFFFFFFu, s, off));
  if ((l & 31) == 0) red[g][l >> 5] = s;
  __syncthreads();
  const float m2 = __fdiv_rn(__fadd_rn(__fadd_rn(red[g][0], red[g][2]), __fadd_rn(red[g][1], red[g][3])), (float)N);
  const float rstd = rsqrtf(__fadd_rn(m2, eps));
  if (!live) return;
  const uint2* wr = reinterpret_cast<const uint2*>(w);
  uint2* orow = reinterpret_cast<uint2*>(out + (size_t)row * N);
#pragma unroll
  for (int j = 0; j < NJ; j++) {
    const uint2 wv = wr[l + 128 * j];
    const float wf[4] = {__uint_as_float(wv.x << 16), __uint_as_float(wv.x & 0xFFFF0000u),
                         __uint_as_float(wv.y << 16), __uint_as_float(wv.y & 0xFFFF0000u)};
    __nv_bfloat16 r[4];
#pragma unroll
    for (int i = 0; i < 4; i++) r[i] = __float2bfloat16_rn(__fmul_rn(wf[i], __fmul_rn(rstd, v[j][i])));
    orow[l + 128 * j] = *reinterpret_cast<const uint2*>(r);
  }
}

}  // namespace okcf

// qkv: the projection output [T, ld] (q 16 heads, k 2, v 2 of 64 from column 0); pos int64 [T] and cos_sin fp32
// [max_pos, 64] for rope (empty: none); q_out [T, 1024] E4M3 or bf16; k / v caches [slots, 128] E4M3 rows
void qk_fold(torch::Tensor qkv, int64_t ld, torch::Tensor pos, torch::Tensor cos_sin, double eps, torch::Tensor q_out,
             torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor loc, int64_t tree) {
  const int T = (int)loc.numel();
  if (T == 0) return;
  const bool rope = pos.numel() > 0, qfp8 = q_out.scalar_type() == at::kFloat8_e4m3fn;
  const bool l64 = loc.scalar_type() == at::kLong;
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && (qfp8 || q_out.scalar_type() == at::kBFloat16) &&
                  q_out.is_contiguous() && q_out.numel() == (int64_t)T * 1024 && (l64 || loc.scalar_type() == at::kInt) &&
                  k_cache.scalar_type() == at::kFloat8_e4m3fn && v_cache.scalar_type() == at::kFloat8_e4m3fn &&
                  k_cache.dim() == 2 && v_cache.dim() == 2 && k_cache.size(1) == 128 && v_cache.size(1) == 128 &&
                  k_cache.stride(1) == 1 && v_cache.stride(1) == 1 && ld % 8 == 0 &&
                  (!rope || (pos.scalar_type() == at::kLong && pos.numel() >= T && cos_sin.scalar_type() == at::kFloat &&
                             cos_sin.dim() == 2 && cos_sin.size(1) == 64 && cos_sin.is_contiguous())) &&
                  (tree == 8 || tree == 4 || tree == 2),
              "qk_fold: layout");
  const int rw = 32 / (64 / (int)tree), nrows = T * 20;
  const int grid = (nrows + 8 * rw - 1) / (8 * rw);
  auto st = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* q = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  const int64_t* p = rope ? pos.data_ptr<int64_t>() : nullptr;
  const float* c = rope ? cos_sin.data_ptr<float>() : nullptr;
  auto* kc = reinterpret_cast<unsigned char*>(k_cache.data_ptr());
  auto* vc = reinterpret_cast<unsigned char*>(v_cache.data_ptr());
#define OKCF_QK(TR, RO, QF, IT)                                                                                     \
  okcf::qkfold_k<TR, RO, QF, IT><<<grid, 256, 0, st>>>(q, (int)ld, T, p, c, (float)eps, q_out.data_ptr(), kc,       \
                                                       k_cache.stride(0), vc, v_cache.stride(0),                     \
                                                       reinterpret_cast<const IT*>(loc.data_ptr()))
#define OKCF_QK3(RO, QF, IT)                \
  if (tree == 8) OKCF_QK(8, RO, QF, IT);    \
  else if (tree == 4) OKCF_QK(4, RO, QF, IT); \
  else OKCF_QK(2, RO, QF, IT)
  if (l64) {
    if (rope) { if (qfp8) { OKCF_QK3(true, true, int64_t); } else { OKCF_QK3(true, false, int64_t); } }
    else { if (qfp8) { OKCF_QK3(false, true, int64_t); } else { OKCF_QK3(false, false, int64_t); } }
  } else {
    if (rope) { if (qfp8) { OKCF_QK3(true, true, int); } else { OKCF_QK3(true, false, int); } }
    else { if (qfp8) { OKCF_QK3(false, true, int); } else { OKCF_QK3(false, false, int); } }
  }
#undef OKCF_QK3
#undef OKCF_QK
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shard_prenorm(torch::Tensor x, torch::Tensor w, double eps, torch::Tensor out) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 &&
                  x.dim() == 2 && x.size(1) == 4096 && x.is_contiguous() && w.numel() == 4096 && w.is_contiguous() &&
                  out.is_contiguous() && out.sizes() == x.sizes(),
              "shard_prenorm: bf16 [rows, 4096]");
  const int rows = (int)x.size(0);
  if (rows == 0) return;
  okcf::prenorm_k<<<(rows + 3) / 4, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
      (float)eps, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP_SRC = (
    "void qk_fold(torch::Tensor qkv, int64_t ld, torch::Tensor pos, torch::Tensor cos_sin, double eps, "
    "torch::Tensor q_out, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor loc, int64_t tree);\n"
    "void shard_prenorm(torch::Tensor x, torch::Tensor w, double eps, torch::Tensor out);\n"
)
