"""OKC draft-chain L2 prefetch kernel (OKC_DRAFT_PF, draft_pf.py; serving/kernels/draft LOG D7). Issue and exit: a few
CTAs hand L2 prefetch requests for up to four strided byte ranges (rows x seg bytes at base + r * stride) to the memory
system and retire. Nothing waits on them and nothing is written, so the GEMMs that read those bytes next produce the same
bits; only where their loads hit moves.
- mode bit 0 = 0: per-thread prefetch.global.L2, one warp per row (lanes take the row's 128 B lines);
  mode bit 0 = 1: cp.async.bulk.prefetch.L2 from one thread per CTA, one instruction per row (rows of <= 1 MB).
- mode bit 1: L2::evict_last instead of evict_normal (the lines then outlive their single use)."""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace okcp {

constexpr int MAXB = 4;
struct Boxes {
  const char* base[MAXB];
  long long stride[MAXB];
  int rows[MAXB];
  int seg[MAXB];  // bytes per row, a multiple of 128 (16 for the bulk form)
  int n;
};

template <int EL>
__device__ __forceinline__ void pf_line(const char* a) {
  if (EL) asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(a));
  else asm volatile("prefetch.global.L2 [%0];" :: "l"(a));
}

// one warp per row, grid-stride over rows; the boxes run back to back (unrolled: the box fields stay in param space)
template <int EL>
__global__ void __launch_bounds__(256) dpf_lines_k(const Boxes b) {
  const int lane = threadIdx.x & 31, nw = gridDim.x * 8;
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
#pragma unroll
  for (int k = 0; k < MAXB; k++) {
    if (k >= b.n) break;
    const int lines = b.seg[k] >> 7;
    for (int r = w; r < b.rows[k]; r += nw) {
      const char* p = b.base[k] + (long long)r * b.stride[k];
      for (int l = lane; l < lines; l += 32) pf_line<EL>(p + l * 128);
    }
  }
}

// thread 0 of each CTA, grid-stride over rows: one bulk prefetch per row
template <int EL>
__global__ void __launch_bounds__(32) dpf_bulk_k(const Boxes b) {
  if (threadIdx.x != 0) return;
  unsigned long long pol = 0;
  if (EL) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
#pragma unroll
  for (int k = 0; k < MAXB; k++) {
    if (k >= b.n) break;
    for (int r = blockIdx.x; r < b.rows[k]; r += gridDim.x) {
      const char* p = b.base[k] + (long long)r * b.stride[k];
      if (EL)
        asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(p), "r"(b.seg[k]), "l"(pol)
                     : "memory");
      else
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(b.seg[k]) : "memory");
    }
  }
}

}  // namespace okcp

// d = n x (base address, row stride, rows, bytes per row); launched on the current stream
void dpf_issue(std::vector<int64_t> d, int64_t ctas, int64_t mode) {
  TORCH_CHECK(d.size() % 4 == 0 && d.size() / 4 <= okcp::MAXB && ctas >= 1 && ctas <= 1024, "dpf: <= 4 boxes");
  okcp::Boxes b = {};
  b.n = (int)(d.size() / 4);
  for (int k = 0; k < b.n; k++) {
    b.base[k] = reinterpret_cast<const char*>(d[4 * k]);
    b.stride[k] = d[4 * k + 1];
    b.rows[k] = (int)d[4 * k + 2];
    b.seg[k] = (int)d[4 * k + 3];
    TORCH_CHECK(b.rows[k] >= 0 && b.seg[k] > 0 && b.seg[k] % ((mode & 1) ? 16 : 128) == 0 &&
                    (!(mode & 1) || b.seg[k] <= (1 << 20)),
                "dpf: rows >= 0, bytes per row a multiple of 128 (bulk: 16, <= 1 MB)");
  }
  auto st = at::cuda::getCurrentCUDAStream();
  if (mode & 1) {
    if (mode & 2) okcp::dpf_bulk_k<1><<<(int)ctas, 32, 0, st>>>(b);
    else okcp::dpf_bulk_k<0><<<(int)ctas, 32, 0, st>>>(b);
  } else {
    if (mode & 2) okcp::dpf_lines_k<1><<<(int)ctas, 256, 0, st>>>(b);
    else okcp::dpf_lines_k<0><<<(int)ctas, 256, 0, st>>>(b);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP_SRC = "void dpf_issue(std::vector<int64_t> d, int64_t ctas, int64_t mode);\n"
