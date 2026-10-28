"""OKC DFlash draft glue kernels (serving/kernels/draft), each reproducing the torch chain it replaces bit for bit:
cand_pack (merged candidate gather input), cand_merge (rank-major view, torch.topk = sbtopk gatherTopK + SmallBitonicSort
with ties, id gather, .long(), DFlash2 unary softcap as torch's float ops), kv_write (ConvertToFloat8E4M3fnOp + the
prefix-valid KV write of every draft layer)."""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace okcd {

// at::native::TopKTypeConfig<float>::convert
__device__ __forceinline__ unsigned radix_key(float v) {
  unsigned x = __float_as_uint(v);
  unsigned mask = (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return (v == v) ? (x ^ mask) : 0xffffffffu;
}

// GTOp<float, true>
__device__ __forceinline__ bool gt_op(float a, float b) { return (isnan(a) && !isnan(b)) || (a > b); }

template <typename I>
__global__ void cand_pack_k(const __nv_bfloat16* __restrict__ vals, const I* __restrict__ ids, int* __restrict__ out,
                            int rows, int k, int offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * k) return;
  int r = i / k, j = i % k;
  out[r * 2 * k + j] = __float_as_int(__bfloat162float(vals[i]));
  out[r * 2 * k + k + j] = static_cast<int>(ids[i]) + offset;
}

// One warp per row; W * K == 128 (8 ranks x top-16), K == 16.
template <int W, int K>
__global__ void cand_merge_k(const int* __restrict__ ag, int rows, float mult, float inv_cap, float cap,
                             float* __restrict__ out_vals, int64_t* __restrict__ out_ids) {
  constexpr int M = W * K;
  constexpr int PER = M / 32;
  constexpr int SORT = 32;  // SmallBitonicSort: sort_size 32 = block_x 16 x items_per_thread 2
  static_assert(M % 32 == 0 && K <= SORT, "shape");
  __shared__ unsigned skey[M];
  __shared__ float sk[SORT];
  __shared__ int64_t sv[SORT];
  __shared__ bool sval[SORT];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  float v[PER];
  unsigned c[PER];
#pragma unroll
  for (int t = 0; t < PER; ++t) {
    int idx = lane + 32 * t;  // rank-major position r * K + j in the permuted view
    int r = idx / K, j = idx % K;
    v[t] = __int_as_float(ag[((size_t)r * rows + row) * 2 * K + j]);
    c[t] = radix_key(v[t]);
    skey[idx] = c[t];
  }
  __syncwarp();
  // k-th largest radix key: the element with (#greater <= K - 1) and (#greater + #equal >= K)
  unsigned kth = 0u;
  bool have = false;
#pragma unroll
  for (int t = 0; t < PER; ++t) {
    int gtr = 0, eq = 0;
    for (int q = 0; q < M; ++q) {
      unsigned o = skey[q];
      gtr += o > c[t];
      eq += o == c[t];
    }
    if (gtr <= K - 1 && gtr + eq >= K) { kth = c[t]; have = true; }
  }
  unsigned any = __ballot_sync(0xffffffffu, have);
  int src = __ffs(any) - 1;
  kth = __shfl_sync(0xffffffffu, kth, src);
  // gatherTopK: strictly greater in index order, then equal ones in index order until K are written
  if (lane < SORT) { sval[lane] = false; sk[lane] = 0.f; sv[lane] = 0; }
  __syncwarp();
  int base = 0;
  const unsigned lt = (1u << lane) - 1u;
#pragma unroll
  for (int t = 0; t < PER; ++t) {
    bool g = c[t] > kth;
    unsigned b = __ballot_sync(0xffffffffu, g);
    if (g) { int w = base + __popc(b & lt); sk[w] = v[t]; sv[w] = lane + 32 * t; sval[w] = true; }
    base += __popc(b);
  }
  int remaining = K - base;
#pragma unroll
  for (int t = 0; t < PER; ++t) {
    bool e = c[t] == kth;
    unsigned b = __ballot_sync(0xffffffffu, e);
    int pos = __popc(b & lt);
    if (e && pos < remaining) { int w = base + pos; sk[w] = v[t]; sv[w] = lane + 32 * t; sval[w] = true; }
    int carry = __popc(b);
    base += carry < remaining ? carry : remaining;
    remaining -= carry < remaining ? carry : remaining;
  }
  __syncwarp();
  // bitonicSort<32> with 16 threads (tid = lane < 16), descending comparator GTOp
  const unsigned tid = lane;
  auto swap_step = [&](unsigned stride, bool dir_flag) {
    if (tid < SORT / 2) {
      unsigned pos = 2 * tid - (tid & (stride - 1));
      float kA = sk[pos], kB = sk[pos + stride];
      int64_t vA = sv[pos], vB = sv[pos + stride];
      bool aA = sval[pos], aB = sval[pos + stride];
      bool sw = (gt_op(kA, kB) && aA) || !aB;
      if (sw == dir_flag) {
        sk[pos] = kB; sk[pos + stride] = kA;
        sv[pos] = vB; sv[pos + stride] = vA;
        sval[pos] = aB; sval[pos + stride] = aA;
      }
    }
    __syncwarp();
  };
#pragma unroll
  for (unsigned size = 2; size < SORT; size *= 2) {
    bool flag = ((tid & (size / 2)) != 0);
#pragma unroll
    for (unsigned stride = size / 2; stride > 0; stride /= 2) swap_step(stride, flag);
  }
#pragma unroll
  for (unsigned stride = SORT / 2; stride > 0; stride /= 2) swap_step(stride, false);
  if (lane < K) {
    int idx = static_cast<int>(sv[lane]);
    int r = idx / K, j = idx % K;
    int id = ag[((size_t)r * rows + row) * 2 * K + K + j];
    float y = sk[lane];
    y = mult * y;
    y = y * inv_cap;
    y = ::tanhf(y);
    y = cap * y;
    out_vals[(size_t)row * K + lane] = y;
    out_ids[(size_t)row * K + lane] = static_cast<int64_t>(id);
  }
}

// blockIdx.x = row (b * block + j), blockIdx.y = layer; threads over the row's elements (k and v).
__global__ void kv_write_k(const __nv_bfloat16* __restrict__ k_all, const __nv_bfloat16* __restrict__ v_all,
                           int64_t layer_stride_k, int64_t layer_stride_v, int64_t row_stride_k, int64_t row_stride_v,
                           const int64_t* __restrict__ bufs, const int64_t* __restrict__ loc,
                           const int* __restrict__ commit_lens, int block, int row_dim) {
  const int row = blockIdx.x, layer = blockIdx.y;
  if (commit_lens != nullptr && (row % block) >= commit_lens[row / block]) return;
  const int64_t dst = loc[row];
  __nv_fp8_storage_t* kb = reinterpret_cast<__nv_fp8_storage_t*>(bufs[2 * layer]) + dst * row_dim;
  __nv_fp8_storage_t* vb = reinterpret_cast<__nv_fp8_storage_t*>(bufs[2 * layer + 1]) + dst * row_dim;
  const __nv_bfloat16* ks = k_all + layer * layer_stride_k + row * row_stride_k;
  const __nv_bfloat16* vs = v_all + layer * layer_stride_v + row * row_stride_v;
  for (int e = threadIdx.x; e < row_dim; e += blockDim.x) {
    kb[e] = __nv_cvt_bfloat16raw_to_fp8(static_cast<__nv_bfloat16_raw>(ks[e]), __NV_SATFINITE, __NV_E4M3);
    vb[e] = __nv_cvt_bfloat16raw_to_fp8(static_cast<__nv_bfloat16_raw>(vs[e]), __NV_SATFINITE, __NV_E4M3);
  }
}

}  // namespace okcd

#define BF(t) reinterpret_cast<const __nv_bfloat16*>(t.data_ptr())
// Callers (draft_glue.py) pass contiguous tensors of the documented dtypes.
void cand_pack(torch::Tensor vals, torch::Tensor ids, torch::Tensor out, int64_t offset) {
  int rows = vals.size(0), k = vals.size(1), n = rows * k;
  if (n == 0) return;
  auto st = at::cuda::getCurrentCUDAStream();
  if (ids.scalar_type() == at::kInt)
    okcd::cand_pack_k<int><<<(n + 127) / 128, 128, 0, st>>>(BF(vals), ids.data_ptr<int>(), out.data_ptr<int>(), rows, k, (int)offset);
  else
    okcd::cand_pack_k<int64_t><<<(n + 127) / 128, 128, 0, st>>>(BF(vals), ids.data_ptr<int64_t>(), out.data_ptr<int>(), rows, k, (int)offset);
}

void cand_merge(torch::Tensor ag, int64_t rows, int64_t world, int64_t k, double mult, double inv_cap, double cap,
                torch::Tensor out_vals, torch::Tensor out_ids) {
  TORCH_CHECK(world == 8 && k == 16, "cand_merge: 8 ranks x top-16");
  if (rows == 0) return;
  okcd::cand_merge_k<8, 16><<<(int)rows, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      ag.data_ptr<int>(), (int)rows, (float)mult, (float)inv_cap, (float)cap, out_vals.data_ptr<float>(), out_ids.data_ptr<int64_t>());
}

void kv_write(torch::Tensor k_all, torch::Tensor v_all, torch::Tensor bufs, torch::Tensor loc,
              c10::optional<torch::Tensor> commit_lens, int64_t block) {
  TORCH_CHECK(k_all.stride(2) == 1 && v_all.stride(2) == 1 && bufs.numel() == 2 * k_all.size(0), "kv_write: layout");
  int layers = k_all.size(0), rows = k_all.size(1);
  if (rows == 0) return;
  okcd::kv_write_k<<<dim3(rows, layers), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      BF(k_all), BF(v_all), k_all.stride(0), v_all.stride(0), k_all.stride(1), v_all.stride(1), bufs.data_ptr<int64_t>(),
      loc.data_ptr<int64_t>(), commit_lens.has_value() ? commit_lens->data_ptr<int>() : nullptr, (int)block,
      (int)k_all.size(2));
}
"""

_CPP_SRC = (
    "void cand_pack(torch::Tensor vals, torch::Tensor ids, torch::Tensor out, int64_t offset);\n"
    "void cand_merge(torch::Tensor ag, int64_t rows, int64_t world, int64_t k, double mult, double inv_cap, "
    "double cap, torch::Tensor out_vals, torch::Tensor out_ids);\n"
    "void kv_write(torch::Tensor k_all, torch::Tensor v_all, torch::Tensor bufs, torch::Tensor loc, "
    "c10::optional<torch::Tensor> commit_lens, int64_t block);\n"
)
