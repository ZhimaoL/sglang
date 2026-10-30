"""OKC_QK_FOLD (ft_norm.py): FusedQKNorm's Triton _qk_rms_norm_fwd_kernel + fused_fp8_qkv_kv_cache in one pass, bit-identical:
the Triton 3.8 sm_90 PTX's per-BLOCK_M reduction tree, div.full, rsqrt.approx.ftz, the rope fma pattern, then bf16
and E4M3 satfinite (serving/kernels/ftnorm/LOG.md F1)."""

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

"""

_CPP_SRC = (
    "void qk_fold(torch::Tensor qkv, int64_t ld, torch::Tensor pos, torch::Tensor cos_sin, double eps, "
    "torch::Tensor q_out, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor loc, int64_t tree);\n"
)
