"""Fused small-T routing (router.route_small): route_k sums each token's split-K partials in reduce_k<2>'s order, then
runs routing.cuh's routing_kernel (sigmoid, E 256, top-8) op for op: bit-identical to reduce_launch(FOLD=2) ->
FusedMuseSparkRouting's JIT kernel (no --use_fast_math, as both). serving/kernels/smallt-moe LOG E23.
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
                                              float mult, float scale, float eps, const char* pf0, int n0,
                                              const char* pf1, int n1) {
  constexpr int EPL = 8, TOPK = 8, G = 128;
  __shared__ float4 s_sum[2][32];
  __shared__ float4 s_p[SK][2][32];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, token = blockIdx.x;
  if (token == 0)   // the MoE's static per-layer inputs into L2 ahead of st (its first loads were HBM latency)
    for (int i = threadIdx.x * 128; i < n0 + n1; i += 64 * 128)
      asm volatile("prefetch.global.L2 [%0];" :: "l"(i < n0 ? pf0 + i : pf1 + (i - n0)));
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

// P [SK, T, 256] fp32 split-K partials -> tw [T, 8] fp32, ti / tp [T, 8] int32
void route(torch::Tensor P, torch::Tensor bias, torch::Tensor tw, torch::Tensor ti, torch::Tensor tp, int64_t T,
           int64_t SK, double mult, double scale, double eps, int64_t pdl, torch::Tensor pf0, torch::Tensor pf1) {
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
                                    tp.data_ptr<int>(), (float)mult, (float)scale, (float)eps,
                                    (const char*)pf0.data_ptr(), (int)(pf0.numel() * pf0.element_size()),
                                    (const char*)pf1.data_ptr(), (int)(pf1.numel() * pf1.element_size())));
}
"""
