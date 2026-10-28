"""CUDA source for the OKC verify-step glue kernels (serving/kernels/verify-glue).

tail_boundary: the last decoder layer's FFN boundary and the model's final norm in one launch, reproducing the torch
chain they replace operation for operation (MuseSpark1xRMSNormWithInputScale -> MuseSpark1xNormPreservingResidualConnection
-> stream.to(bf16) -> MuseSpark1xRMSNorm), so the outputs are bit-identical:
  y  = bf16(w_post * x)                        (bf16 multiply, fp32 opmath)
  nb = bf16(rstd(y) * y)                       (F.rms_norm, no weight)
  ns = (alpha * stream) + (beta * float(nb))   (three fp32 kernels: two products, then the sum)
  xs = bf16(ns)                                (stream.to(bf16): the aux capture value and the final norm input)
  out = bf16(w_final * (rstd(xs) * xs))        (F.rms_norm with weight)
rstd follows torch's vectorized rms_norm reduction (block 32 x 4): thread l of 128 sums v * v by fma over the bf16x4
vectors l + 128 j in order, shfl_down 16 .. 1 inside each warp, warps (0 + 2) + (1 + 3), / N, rsqrtf(. + eps).
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace okcg {

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float lo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi(unsigned u) { return __uint_as_float(u & 0xFFFF0000u); }

template <int NJ>
__device__ __forceinline__ float torch_rstd(const float (&v)[NJ][4], float eps, float* red) {
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < NJ; j++)
#pragma unroll
    for (int i = 0; i < 4; i++) s = __fmaf_rn(v[j][i], v[j][i], s);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xFFFFFFFFu, s, off));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  const float m2 = __fdiv_rn(__fadd_rn(__fadd_rn(red[0], red[2]), __fadd_rn(red[1], red[3])), (float)(NJ * 512));
  __syncthreads();   // red is reused by the next reduction
  return rsqrtf(__fadd_rn(m2, eps));
}

template <int NJ>
__global__ void __launch_bounds__(128) tail_boundary_k(
    const __nv_bfloat16* __restrict__ branch, const float* __restrict__ stream, const __nv_bfloat16* __restrict__ w_post,
    float eps_post, const float* __restrict__ alpha, const float* __restrict__ beta, int alpha_n, int beta_n,
    const __nv_bfloat16* __restrict__ w_final, float eps_final, __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ stream_bf16) {
  __shared__ float red[4];
  constexpr int N = NJ * 512;
  const int row = blockIdx.x, l = threadIdx.x;
  pdl_wait();
  pdl_trigger();
  const uint2* br = reinterpret_cast<const uint2*>(branch + (size_t)row * N);
  const uint2* wp = reinterpret_cast<const uint2*>(w_post);
  float y[NJ][4];
#pragma unroll
  for (int j = 0; j < NJ; j++) {
    const uint2 a = br[l + 128 * j], w = wp[l + 128 * j];
    y[j][0] = bf(__fmul_rn(lo(w.x), lo(a.x)));
    y[j][1] = bf(__fmul_rn(hi(w.x), hi(a.x)));
    y[j][2] = bf(__fmul_rn(lo(w.y), lo(a.y)));
    y[j][3] = bf(__fmul_rn(hi(w.y), hi(a.y)));
  }
  const float r1 = torch_rstd<NJ>(y, eps_post, red);
  const float4* sp = reinterpret_cast<const float4*>(stream + (size_t)row * N);
  float xs[NJ][4];
#pragma unroll
  for (int j = 0; j < NJ; j++) {
    const float4 s4 = sp[l + 128 * j];
    const float sv[4] = {s4.x, s4.y, s4.z, s4.w};
    const int c0 = 4 * (l + 128 * j);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float a = alpha_n == 1 ? alpha[0] : alpha[c0 + i];
      const float b = beta_n == 1 ? beta[0] : beta[c0 + i];
      const float nb = bf(__fmul_rn(r1, y[j][i]));
      xs[j][i] = bf(__fadd_rn(__fmul_rn(a, sv[i]), __fmul_rn(b, nb)));
    }
    __nv_bfloat162* sd = reinterpret_cast<__nv_bfloat162*>(stream_bf16 + (size_t)row * N + c0);
    sd[0] = __floats2bfloat162_rn(xs[j][0], xs[j][1]);
    sd[1] = __floats2bfloat162_rn(xs[j][2], xs[j][3]);
  }
  const float r2 = torch_rstd<NJ>(xs, eps_final, red);
  const uint2* wf = reinterpret_cast<const uint2*>(w_final);
#pragma unroll
  for (int j = 0; j < NJ; j++) {
    const uint2 g = wf[l + 128 * j];
    const int c0 = 4 * (l + 128 * j);
    __nv_bfloat162* od = reinterpret_cast<__nv_bfloat162*>(out + (size_t)row * N + c0);
    od[0] = __floats2bfloat162_rn(__fmul_rn(lo(g.x), __fmul_rn(r2, xs[j][0])), __fmul_rn(hi(g.x), __fmul_rn(r2, xs[j][1])));
    od[1] = __floats2bfloat162_rn(__fmul_rn(lo(g.y), __fmul_rn(r2, xs[j][2])), __fmul_rn(hi(g.y), __fmul_rn(r2, xs[j][3])));
  }
}

}  // namespace okcg

static bool g_glue_pdl = false;
void glue_set_pdl(int64_t on) { g_glue_pdl = on != 0; }

void tail_boundary(torch::Tensor branch, torch::Tensor stream, torch::Tensor w_post, double eps_post,
                   torch::Tensor alpha, torch::Tensor beta, torch::Tensor w_final, double eps_final,
                   torch::Tensor out, torch::Tensor stream_bf16) {
  const int64_t m = branch.size(0), n = branch.size(1);
  TORCH_CHECK(n == 8192, "tail_boundary: hidden 8192");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)m);
  cfg.blockDim = dim3(128);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = g_glue_pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, okcg::tail_boundary_k<16>, reinterpret_cast<const __nv_bfloat16*>(branch.data_ptr()),
      static_cast<const float*>(stream.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(w_post.data_ptr()),
      (float)eps_post, static_cast<const float*>(alpha.data_ptr()), static_cast<const float*>(beta.data_ptr()),
      (int)alpha.numel(), (int)beta.numel(), reinterpret_cast<const __nv_bfloat16*>(w_final.data_ptr()),
      (float)eps_final, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(stream_bf16.data_ptr())));
}
"""
