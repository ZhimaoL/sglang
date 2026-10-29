// bf16 all-gather of a [M, N / world] column slice into [M, N] (+ b) (+ c),
// through the CustomAllReduceV2 push workspace (same phase / Lamport protocol
// as all_reduce_1shot_push). Each rank multicast-stores its slice into every
// peer's workspace, then spins its full rows together. Any TP world size.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/stl.h>

#include <algorithm>
#include <cstdint>
#include <optional>

namespace sglang {

namespace muse_all_gather {

using device::distributed::Counter;

// OKC: 512 (was 128): each thread polls one or a few vectors instead of ~8 in series (H200 x8 S9 vg-micro, us per
// gather 128 -> 512 at N 8192: m 8 2.57 -> 2.60, m 32 3.97 -> 3.73, m 64 6.13 -> 5.24, m 128 10.18 -> 8.43,
// m 256 16.58 -> 14.14; bit-identical; 1024 costs +0.28 at m 8)
constexpr uint32_t kBlock = 512;
constexpr uint32_t kVec = 16 / sizeof(bf16_t);  // 8 bf16 (16B) per thread

template <uint32_t kWorld>
struct Params {
  uint8_t* ws_local;      // LOCAL VA of the push workspace base (poll + reset)
  uint8_t* ws_mc;         // multicast VA of the push workspace base (push)
  Counter* counter;       // per-block phase counters
  uint32_t num_counters;  // PushPlane::num_blocks
  uint32_t half_bytes;    // world_size * slot_bytes
  uint32_t rank;
  const bf16_t* local;  // [M, N / world]
  const bf16_t* b;      // [M, N] or null
  const bf16_t* c;      // [M, N] or null
  bf16_t* out;          // [M, N]
  uint32_t num_rows;
  const bf16_t* gamma;  // kNorm: the pre-expert RMSNorm weight [N]
  float eps;
};

// OKC kNorm: the gathered row's RMSNorm in torch's vectorized rms_norm order (128 lanes, an fma chain over bf16x4
// vectors l + 128 j, shfl_down 16 .. 1, warps (0 + 2) + (1 + 3), / N, rsqrtf(. + eps)); the row is
// bf16(gamma * (rstd * x)), the values F.rms_norm returns and st's norm fold rebuilds (_cuda_src_st rms_rstd).
template <uint32_t N>
__device__ __forceinline__ float okc_row_rstd(const bf16_t* row, float eps, float* s_w) {
  const uint32_t tx = threadIdx.x;
  if (tx < 128) {
    const uint2* r4 = reinterpret_cast<const uint2*>(row);
    float sq = 0.f;
#pragma unroll
    for (uint32_t j = 0; j < N / 512; ++j) {
      const uint2 v = r4[tx + 128 * j];
      const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
      const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
      sq = __fmaf_rn(lo.x, lo.x, sq);
      sq = __fmaf_rn(lo.y, lo.y, sq);
      sq = __fmaf_rn(hi.x, hi.x, sq);
      sq = __fmaf_rn(hi.y, hi.y, sq);
    }
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) sq = __fadd_rn(sq, __shfl_down_sync(0xffffffffu, sq, o));
    if ((tx & 31) == 0) s_w[tx >> 5] = sq;
  }
  __syncthreads();
  const float m2 = __fdiv_rn(__fadd_rn(__fadd_rn(s_w[0], s_w[2]), __fadd_rn(s_w[1], s_w[3])), float(N));
  return rsqrtf(__fadd_rn(m2, eps));
}

template <uint32_t N>
__device__ __forceinline__ void okc_row_norm(uint4& raw, const bf16_t* gamma, uint32_t col, float rs) {
  const uint4 g = *reinterpret_cast<const uint4*>(gamma + col);
  const uint32_t* xw = reinterpret_cast<const uint32_t*>(&raw);
  const uint32_t* gw = reinterpret_cast<const uint32_t*>(&g);
  uint32_t* ow = reinterpret_cast<uint32_t*>(&raw);
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xw[j]));
    const float2 w = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gw[j]));
    const __nv_bfloat162 y = __floats2bfloat162_rn(__fmul_rn(w.x, __fmul_rn(rs, x.x)), __fmul_rn(w.y, __fmul_rn(rs, x.y)));
    ow[j] = *reinterpret_cast<const uint32_t*>(&y);
  }
}

// Slot layout inside a phase half: dense [world][M][N / world] bf16.
template <uint32_t N, uint32_t kWorld, bool kHasB, bool kHasC, bool kUsePDL, bool kNorm = false>
__global__ __launch_bounds__(kBlock) void all_gather_kernel(const __grid_constant__ Params<kWorld> params) {
  using namespace device;
  using vec_t = AlignedVector<bf16x2_t, kVec / 2>;
  constexpr uint32_t kNLocal = N / kWorld;
  static_assert(N % kVec == 0 && kNLocal % kVec == 0, "a vector must never cross a rank block");
  static_assert(!kNorm || (N == kBlock * kVec && !kHasB && !kHasC), "kNorm: one row per CTA pass, no b / c");
  __shared__ __align__(16) bf16_t s_row[kNorm ? N : 8];
  __shared__ float s_w[4];
  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const uint32_t tid = bx * kBlock + tx;
  PDLWaitPrimary<kUsePDL>();
  const uint32_t phase = params.counter[bx].get() & 1;
  const uint32_t half_off = phase * params.half_bytes;

  // the last block flips its own counter and every one past the grid
  if (const auto num_blocks = gridDim.x; bx == num_blocks - 1) {
    [[unlikely]];
    PDLTriggerSecondary<kUsePDL>();
    __syncthreads();
    for (uint32_t i = num_blocks - 1 + tx; i < params.num_counters; i += kBlock) {
      params.counter[i].set(phase ^ 1);
    }
    return void();
  }

  // Bound the grid to the existing AR counters; larger batches use a stride.
  const uint32_t stride = (gridDim.x - 1) * kBlock * kVec;
  for (uint32_t local_elem = tid * kVec; local_elem < params.num_rows * kNLocal; local_elem += stride) {
    uint4 raw = *reinterpret_cast<const uint4*>(params.local + local_elem);
    // +0.0 is the "not arrived" marker; -0.0 is numerically identical
    uint32_t* w = reinterpret_cast<uint32_t*>(&raw);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      if ((w[j] & 0xffffu) == 0) w[j] |= 0x8000u;
      if ((w[j] >> 16) == 0) w[j] |= 0x80000000u;
    }
    const uint32_t byte_off = half_off + (params.rank * params.num_rows * kNLocal + local_elem) * sizeof(bf16_t);
    ptx::st_multimem_16B(raw, params.ws_mc + byte_off, 0);
  }
  PDLTriggerSecondary<kUsePDL>();

  // Every producer finishes its local pushes before polling output vectors.
  for (uint32_t elem = tid * kVec; elem < params.num_rows * N; elem += stride) {
    const auto row = elem / N;
    const auto col = elem % N;
    const auto base = reinterpret_cast<bf16_t*>(params.ws_local + half_off);
    const auto src = base + ((col / kNLocal) * params.num_rows + row) * kNLocal + col % kNLocal;
    vec_t b_vec, c_vec;
    if constexpr (kHasB) b_vec.load(params.b + elem);
    if constexpr (kHasC) c_vec.load(params.c + elem);
    uint4 raw;
    do {
      ptx::ld_relaxed_16B(raw, src, 0);
    } while ((raw.x & 0xffffu) == 0 || (raw.x >> 16) == 0 || (raw.y & 0xffffu) == 0 || (raw.y >> 16) == 0 ||
             (raw.z & 0xffffu) == 0 || (raw.z >> 16) == 0 || (raw.w & 0xffffu) == 0 || (raw.w >> 16) == 0);
    if constexpr (kNorm) {   // the whole row is this CTA's (one vector per thread): stage it, then normalize
      *reinterpret_cast<uint4*>(s_row + col) = raw;
      __syncthreads();
      const float rs = okc_row_rstd<N>(s_row, params.eps, s_w);
      okc_row_norm<N>(raw, params.gamma, col, rs);
      __syncthreads();   // s_row / s_w are reused by the next row
    }
    const auto& gathered = *reinterpret_cast<const vec_t*>(&raw);
    vec_t out_vec;
#pragma unroll
    for (uint32_t j = 0; j < kVec / 2; ++j) {
      using Trait = DTypeTrait<bf16x2_t>;
      out_vec[j] = gathered[j];
      if constexpr (kHasB) out_vec[j] = Trait::add(out_vec[j], b_vec[j]);
      if constexpr (kHasC) out_vec[j] = Trait::add(out_vec[j], c_vec[j]);
    }
    out_vec.store(params.out + elem);
    AlignedVector<uint32_t, 4> zero;
    zero.fill(0);
    zero.store(src);
  }
  __syncthreads();
  if (tx == 0) params.counter[bx].set(phase ^ 1);
}

// OKC: 1-GPU check of the kNorm row math (one CTA per row, the same helpers)
template <uint32_t N>
__global__ __launch_bounds__(kBlock) void okc_row_norm_test_kernel(const bf16_t* x, const bf16_t* gamma, float eps,
                                                                    bf16_t* out) {
  __shared__ __align__(16) bf16_t s_row[N];
  __shared__ float s_w[4];
  const uint32_t col = threadIdx.x * kVec;
  uint4 raw = *reinterpret_cast<const uint4*>(x + size_t(blockIdx.x) * N + col);
  *reinterpret_cast<uint4*>(s_row + col) = raw;
  __syncthreads();
  const float rs = okc_row_rstd<N>(s_row, eps, s_w);
  okc_row_norm<N>(raw, gamma, col, rs);
  *reinterpret_cast<uint4*>(out + size_t(blockIdx.x) * N + col) = raw;
}

}  // namespace muse_all_gather

using host::distributed::CommunicatorRef;

template <uint32_t N, uint32_t kWorld, bool kUsePDL>
struct MuseSparkAllGather {
  using TensorView = tvm::ffi::TensorView;
  static constexpr uint32_t kNLocal = N / kWorld;

  template <bool kHasB, bool kHasC>
  static constexpr auto kernel = muse_all_gather::all_gather_kernel<N, kWorld, kHasB, kHasC, kUsePDL>;

  static void
  run(CommunicatorRef ref, TensorView local, std::optional<TensorView> b, std::optional<TensorView> c, TensorView out) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();

    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, kNLocal}).with_dtype<bf16_t>().with_device(device).verify(local);
    if (b.has_value()) TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(b.value());
    if (c.has_value()) TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(c.value());
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(num_tokens >= 1);
    CHECK_HOST(push.world_size == kWorld) << "the kernel is compiled for TP" << kWorld;
    CHECK_HOST(int64_t(num_tokens) * N * 2 <= push.slot_bytes)
        << "gathered rows exceed the push slot size " << push.slot_bytes;
    CHECK_HOST(push.mc_workspace != nullptr) << "requires a multicast-capable push plane";

    muse_all_gather::Params<kWorld> params{
        .ws_local = push.workspaces[push.rank],
        .ws_mc = push.mc_workspace,
        .counter = push.counter,
        .num_counters = push.num_blocks,
        .half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size),
        .rank = push.rank,
        .local = static_cast<const bf16_t*>(local.data_ptr()),
        .b = b.has_value() ? static_cast<const bf16_t*>(b.value().data_ptr()) : nullptr,
        .c = c.has_value() ? static_cast<const bf16_t*>(c.value().data_ptr()) : nullptr,
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .num_rows = num_tokens,
        .gamma = nullptr,
        .eps = 0.f,
    };
    const auto num_vecs = num_tokens * N / muse_all_gather::kVec;
    CHECK_HOST(push.num_blocks >= 2) << "gather needs a work block and a counter-update block";
    const auto num_work_blocks = std::min(host::div_ceil(num_vecs, muse_all_gather::kBlock), push.num_blocks - 1);
    const auto fn =
        b.has_value() ? (c.has_value() ? kernel<1, 1> : kernel<1, 0>) : (c.has_value() ? kernel<0, 1> : kernel<0, 0>);
    host::LaunchKernel(num_work_blocks + 1, muse_all_gather::kBlock, device.unwrap()).enable_pdl(kUsePDL)(fn, params);
  }
};

// OKC: the gather with the pre-expert RMSNorm fused (kNorm, N == 4096 only).
template <uint32_t N, uint32_t kWorld, bool kUsePDL>
struct MuseSparkAllGatherNorm {
  using TensorView = tvm::ffi::TensorView;
  static constexpr uint32_t kNLocal = N / kWorld;

  static void run(CommunicatorRef ref, TensorView local, TensorView gamma, double eps, TensorView out) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, kNLocal}).with_dtype<bf16_t>().with_device(device).verify(local);
    TensorMatcher({N}).with_dtype<bf16_t>().with_device(device).verify(gamma);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(num_tokens >= 1);
    CHECK_HOST(push.world_size == kWorld) << "the kernel is compiled for TP" << kWorld;
    CHECK_HOST(int64_t(num_tokens) * N * 2 <= push.slot_bytes)
        << "gathered rows exceed the push slot size " << push.slot_bytes;
    CHECK_HOST(push.mc_workspace != nullptr) << "requires a multicast-capable push plane";
    muse_all_gather::Params<kWorld> params{
        .ws_local = push.workspaces[push.rank],
        .ws_mc = push.mc_workspace,
        .counter = push.counter,
        .num_counters = push.num_blocks,
        .half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size),
        .rank = push.rank,
        .local = static_cast<const bf16_t*>(local.data_ptr()),
        .b = nullptr,
        .c = nullptr,
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .num_rows = num_tokens,
        .gamma = static_cast<const bf16_t*>(gamma.data_ptr()),
        .eps = static_cast<float>(eps),
    };
    const auto num_vecs = num_tokens * N / muse_all_gather::kVec;
    CHECK_HOST(push.num_blocks >= 2) << "gather needs a work block and a counter-update block";
    const auto num_work_blocks = std::min(host::div_ceil(num_vecs, muse_all_gather::kBlock), push.num_blocks - 1);
    host::LaunchKernel(num_work_blocks + 1, muse_all_gather::kBlock, device.unwrap())
        .enable_pdl(kUsePDL)(muse_all_gather::all_gather_kernel<N, kWorld, false, false, kUsePDL, true>, params);
  }

  static void test(TensorView x, TensorView gamma, double eps, TensorView out) {   // 1 GPU: the row math only
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N}).with_dtype<bf16_t>().with_device(device).verify(gamma);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    host::LaunchKernel(static_cast<uint32_t>(M.unwrap()), muse_all_gather::kBlock, device.unwrap())(
        muse_all_gather::okc_row_norm_test_kernel<N>, static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(gamma.data_ptr()), static_cast<float>(eps), static_cast<bf16_t*>(out.data_ptr()));
  }
};

}  // namespace sglang
