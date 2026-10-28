"""Fused small-T routing (router.route_small): route_k sums each token's split-K partials in reduce_k<2>'s order, then
runs routing.cuh's routing_kernel (sigmoid, E 256, top-8) op for op: bit-identical to reduce_launch(FOLD=2) ->
FusedMuseSparkRouting's JIT kernel (no --use_fast_math, as both). serving/kernels/smallt-moe LOG E23.
dconv_k: the DFlash draft's grouped 2-tap conv (dflash.DFlashGroupedConv._convolve) with inductor's exact arithmetic
(fp32 coefficient add, fma(c0, x, c1 * x_prev), one bf16 rounding), 8 channels per thread (LOG E25).
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace okcr {

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ float ex2_approx(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
template <typename X>
__device__ __forceinline__ void xchg(X& a, X& b) {
  const X t = a;
  a = b;
  b = t;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float triton_sigmoid(float x) {
  return div_full(1.0f, ex2_approx((0.0f - x) * kLog2e) + 1.0f);
}
// routing.cuh's round in fewer steps: max as a redux over an order-preserving uint map (heads compare as floats:
// +0 / -0 tie as there), the lowest head expert as a redux min (= its group-ballot owner), its lane from the id
__device__ __forceinline__ unsigned ord_u(float f) {
  const unsigned u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float ord_f(unsigned u) {
  return __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
}

// every partial plane's 16 B in flight at once (ptxas sinks register loads of .nc data to their adds)
__device__ __forceinline__ void cp16(void* sdst, const float* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((unsigned)__cvta_generic_to_shared(sdst)), "l"(gsrc)
               : "memory");
}

// CTA per token: warp v sums experts v * 128 + 4 l + j over the planes in order; warp 0 then holds them at slot 4 v + j
// (monotone in the slot, as routing.cuh relies on) and routes
template <int SK>
__global__ void __launch_bounds__(64) route_k(const float* __restrict__ P, int T, const float* __restrict__ bias,
                                              float* __restrict__ tw, int* __restrict__ ti, int* __restrict__ tp,
                                              float mult, float scale, float eps) {
  constexpr int EPL = 8, TOPK = 8, G = 128;
  __shared__ float4 s_sum[2][32];
  __shared__ float4 s_p[SK][2][32];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, token = blockIdx.x;
  pdl_wait();
  pdl_trigger();
  {
    const float* p = P + (size_t)token * 256 + w * G + lane * 4;
    const size_t plane = (size_t)T * 256;
#pragma unroll
    for (int i = 0; i < SK; i++) cp16(&s_p[i][w][lane], p + plane * i);
    asm volatile("cp.async.wait_all;" ::: "memory");   // each thread reads back only its own copies
    float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < SK; i++)
#pragma unroll
      for (int j = 0; j < 4; j++) s[j] += reinterpret_cast<const float*>(&s_p[i][w][lane])[j];
    s_sum[w][lane] = make_float4(s[0], s[1], s[2], s[3]);
  }
  __syncthreads();
  if (w) return;
  float biased[EPL], score[EPL];
  int expert[EPL];
#pragma unroll
  for (int s = 0; s < EPL; s++) {
    const int e = (s >> 2) * G + lane * 4 + (s & 3);
    biased[s] = bias[e];
    score[s] = reinterpret_cast<const float*>(&s_sum[s >> 2][lane])[s & 3];
    expert[s] = e;
  }

  // routing.cuh routing_kernel from here
#pragma unroll
  for (int s = 0; s < EPL; s++) score[s] *= mult;
#pragma unroll
  for (int s = 0; s < EPL; s++) score[s] = triton_sigmoid(score[s]);
  float key[EPL];
#pragma unroll
  for (int s = 0; s < EPL; s++) key[s] = biased[s] + score[s];

#pragma unroll
  for (int r = 0; r < EPL; r++) {
#pragma unroll
    for (int i = r % 2; i + 1 < EPL; i += 2)
      if (key[i + 1] > key[i]) {
        xchg(key[i], key[i + 1]);
        xchg(expert[i], expert[i + 1]);
        xchg(score[i], score[i + 1]);
      }
  }

  float score_sum = 0.0f;
  int out_id = 0;
  float out_score = 0.0f;
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    const float round_max = ord_f(__reduce_max_sync(0xFFFFFFFFu, ord_u(key[0])));
    const bool is_head = key[0] == round_max;
    const int winner = (int)__reduce_min_sync(0xFFFFFFFFu, is_head ? (unsigned)expert[0] : 0xFFFFFFFFu);
    const float selected = __shfl_sync(0xFFFFFFFFu, score[0], (winner >> 2) & 31);
    const bool pop = expert[0] == winner;
#pragma unroll
    for (int s = 0; s + 1 < EPL; ++s)
      if (pop && s + k <= TOPK - 2) {
        key[s] = key[s + 1];
        expert[s] = expert[s + 1];
        score[s] = score[s + 1];
      }
    score_sum += selected;
    if (lane == k) {
      out_id = winner;
      out_score = selected;
    }
  }
  score_sum += eps;
  const float inv_sum_scaled = div_full(scale, score_sum);
  if (lane < TOPK) {
    const float weight = out_score * inv_sum_scaled;
    const int o = token * TOPK + lane;
    ti[o] = out_id;
    tw[o] = weight;
    const __nv_bfloat16 wb = __float2bfloat16_rn(weight);
    tp[o] = (out_id << 16) | (int)__bfloat16_as_ushort(wb);
  }
}

}  // namespace okcr

namespace okcr {
// out[t, c] = bf16(fma(b0[c] + d0[t, c / gs], x[t, c], (b1[c] + d1[t, c / gs]) * (t's place in its block >= 1 ? x[t - 1,
// c] : 0))), in fp32 (inductor's triton_poi for _grouped_conv: add.f32, add.f32, mul.f32, fma.rn.f32, cvt.rn.bf16).
// d: the side's [T, 2, G] coefficient view (row stride ds, tap stride G); base [2, H]. 8 channels per thread (gs % 8 == 0).
__global__ void __launch_bounds__(256) dconv_k(const __nv_bfloat16* __restrict__ x, long long xs,
                                               const __nv_bfloat16* __restrict__ d, long long ds, int G,
                                               const __nv_bfloat16* __restrict__ base, __nv_bfloat16* __restrict__ out,
                                               int T, int H, int gs, int bsz, int pdl) {
  if (pdl) pdl_wait();
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int nv = H >> 3;
  if (i >= (long long)T * nv) return;
  const int t = (int)(i / nv), c = (int)(i - (long long)t * nv) << 3, g = c / gs;
  const int pos = (bsz & (bsz - 1)) == 0 ? (t & (bsz - 1)) : (t % bsz);
  const uint4 xv = *reinterpret_cast<const uint4*>(x + t * xs + c);
  uint4 pv = make_uint4(0u, 0u, 0u, 0u);
  if (pos >= 1 && t >= 1) pv = *reinterpret_cast<const uint4*>(x + (t - 1) * xs + c);
  const uint4 b0 = *reinterpret_cast<const uint4*>(base + c), b1 = *reinterpret_cast<const uint4*>(base + H + c);
  const float d0 = __bfloat162float(d[t * ds + g]), d1 = __bfloat162float(d[t * ds + G + g]);
  const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xv);
  const __nv_bfloat162* ph = reinterpret_cast<const __nv_bfloat162*>(&pv);
  const __nv_bfloat162* b0h = reinterpret_cast<const __nv_bfloat162*>(&b0);
  const __nv_bfloat162* b1h = reinterpret_cast<const __nv_bfloat162*>(&b1);
  uint4 ov;
  __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float2 xf = __bfloat1622float2(xh[j]), pf = __bfloat1622float2(ph[j]);
    const float2 c0 = __bfloat1622float2(b0h[j]), c1 = __bfloat1622float2(b1h[j]);
    const float p0 = pos >= 1 ? pf.x : 0.0f, p1 = pos >= 1 ? pf.y : 0.0f;
    const float lo = __fmaf_rn(__fadd_rn(c0.x, d0), xf.x, __fmul_rn(__fadd_rn(c1.x, d1), p0));
    const float hi = __fmaf_rn(__fadd_rn(c0.y, d0), xf.y, __fmul_rn(__fadd_rn(c1.y, d1), p1));
    oh[j] = __floats2bfloat162_rn(lo, hi);
  }
  *reinterpret_cast<uint4*>(out + (long long)t * H + c) = ov;
  if (pdl) pdl_trigger();
}
}  // namespace okcr

// x [T, H] bf16 (row stride xs), d [T, 2, G] bf16 view of one side's coefficients, base [2, H] bf16 -> out [T, H]
void dconv(torch::Tensor x, torch::Tensor d, torch::Tensor base, torch::Tensor out, int64_t gs, int64_t bsz,
           int64_t pdl) {
  const int T = (int)x.size(0), H = (int)x.size(1), G = (int)d.size(2);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && d.scalar_type() == at::kBFloat16 && base.scalar_type() == at::kBFloat16 &&
              x.stride(1) == 1 && x.stride(0) % 8 == 0 && H % 8 == 0 && gs % 8 == 0 && H == G * gs && d.size(1) == 2 &&
              d.stride(2) == 1 && d.stride(1) == G && base.is_contiguous() && base.numel() == 2 * H &&
              out.is_contiguous() && out.sizes() == x.sizes() && bsz > 0 &&
              ((uintptr_t)x.data_ptr() | (uintptr_t)base.data_ptr() | (uintptr_t)out.data_ptr()) % 16 == 0,
              "okc dconv: bf16 x [T, H], d [T, 2, H / gs], base [2, H], gs % 8 == 0");
  if (T == 0) return;
  const unsigned nb = (unsigned)(((long long)T * (H / 8) + 255) / 256);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(nb);
  cfg.blockDim = dim3(256u);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, okcr::dconv_k, (const __nv_bfloat16*)x.data_ptr(), (long long)x.stride(0),
                                    (const __nv_bfloat16*)d.data_ptr(), (long long)d.stride(0), G,
                                    (const __nv_bfloat16*)base.data_ptr(), (__nv_bfloat16*)out.data_ptr(), T, H,
                                    (int)gs, (int)bsz, (int)pdl));
}

// P [SK, T, 256] fp32 split-K partials -> tw [T, 8] fp32, ti / tp [T, 8] int32
void route(torch::Tensor P, torch::Tensor bias, torch::Tensor tw, torch::Tensor ti, torch::Tensor tp, int64_t T,
           int64_t SK, double mult, double scale, double eps, int64_t pdl) {
  // router.route_small allocates the outputs; it checks bias (fp32 [256], contiguous)
  TORCH_CHECK(P.is_contiguous() && P.numel() == SK * T * 256 && (SK == 16 || SK == 32), "okc route: P [16|32, T, 256]");
  if (T == 0) return;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)T);
  cfg.blockDim = dim3(64u);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  auto k = SK == 16 ? okcr::route_k<16> : okcr::route_k<32>;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, (const float*)P.data_ptr<float>(), (int)T,
                                    (const float*)bias.data_ptr<float>(), tw.data_ptr<float>(), ti.data_ptr<int>(),
                                    tp.data_ptr<int>(), (float)mult, (float)scale, (float)eps));
}
"""
