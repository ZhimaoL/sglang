// OKC: muse_spark_all_gather_norm_residual_row_kernel at 256 threads (its slots, phases, math), run by one CTA of
// H/8 threads per row: one 16 B vector per thread, so the poll and the loads of the row are all in flight at once.
// Thread t is the row kernel's thread t % 256, pack t / 256; each fmaf chain is passed pack to pack through smem
// and reduced by the warps holding pack 3, as cta::reduce_sum<256> does. Bit-identical to the row kernel.
#include "all_gather_norm_residual.cuh"

#include <array>
#include <utility>

namespace sglang {
namespace okc_ag_norm {

template <int64_t H, uint32_t W, bool PDL, bool S, bool P, bool G>
__global__ void __launch_bounds__(H / 8, 1) kern(const __grid_constant__ muse_ag_norm::Params params) {
  using namespace device;
  using namespace device::norm_residual;
  constexpr int N = H / kVec, R = 256, NW = R / kWarpThreads;
  constexpr uint32_t kNLocal = static_cast<uint32_t>(H) / W;
  using BV = AlignedVector<bf16x2_t, kPairs>;
  using FV = AlignedVector<fp32x2_t, kFPairs>;
  __shared__ float chain[R], red[2][NW];
  const auto& norm = params.norm;
  const int t = threadIdx.x, r0 = t % R, pk = t / R, lane = t % kWarpThreads;
  const bool cw = G && norm.gate_channel_wise;
  const bool hq = norm.has_pre_weight;
  // the row kernel's cta::reduce_sum<256>: pack 3's threads are its 256, every warp reads the 8 partials
  const auto reduce = [&](float v, float* sm) {
    if (pk == N / R - 1) {
      v = warp::reduce_sum(v);
      if (lane == 0) sm[r0 / kWarpThreads] = v;
    }
    __syncthreads();
    return warp::reduce_sum(lane < NW ? sm[lane] : 0.0f);
  };
  // the row kernel's per-thread fmaf chain over its packs, pack after pack
  const auto chained = [&](const fp32x2_t (&v)[kPairs]) {
    float s = 0.0f;
#pragma unroll
    for (int p = 0; p < N / R; ++p) {
      if (pk == p) {
        if (p > 0) s = chain[r0];
#pragma unroll
        for (int q = 0; q < kPairs; ++q) s = fmaf(v[q].x, v[q].x, s), s = fmaf(v[q].y, v[q].y, s);
        if (p < N / R - 1) chain[r0] = s;
      }
      __syncthreads();
    }
    return s;
  };
  PDLWaitPrimary<PDL>();
  const uint32_t phase = muse_ag_norm::begin_phase<N>(params);
  const uint32_t half_off = phase * params.half_bytes;
  muse_ag_norm::push_local_slice<kNLocal, N>(params, half_off);
  PDLTriggerSecondary<PDL>();
  const auto gathered = reinterpret_cast<bf16_t*>(params.ws_local + half_off);
  const auto pw = static_cast<const bf16_t*>(norm.post_weight);
  const auto qw = static_cast<const bf16_t*>(norm.pre_weight);
  const auto al = static_cast<const fp32_t*>(norm.alpha);
  const auto be = static_cast<const fp32_t*>(norm.beta);
  for (uint32_t row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    const int64_t base = static_cast<int64_t>(row) * H;
    FV sv[kFParts], av[kFParts], bv[kFParts];
    BV qv;
    fp32x2_t w[kPairs], val[kPairs];
#pragma unroll
    for (int j = 0; j < kFParts; ++j) {
      sv[j].load(static_cast<const fp32_t*>(norm.stream) + base, t * kFParts + j);
      if (cw) av[j].load(al, t * kFParts + j), bv[j].load(be, t * kFParts + j);
    }
    if (hq) qv.load(qw, t);
    if constexpr (P) {
      BV wv;
      wv.load(pw, t);
#pragma unroll
      for (int q = 0; q < kPairs; ++q) w[q] = cast<fp32x2_t>(wv[q]);
    }
    const uint4 raw = muse_ag_norm::take_gathered<kNLocal>(gathered, params.num_rows, row, t * kVec);
    const auto& br = *reinterpret_cast<const BV*>(&raw);
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      auto v = cast<fp32x2_t>(br[q]);
      if constexpr (S && P) {
        v = muse_ag_norm::mul_pair(v, w[q]);
        v = round_bf16(v);
      }
      val[q] = v;
    }
    const float pr = math::rsqrt(reduce(chained(val), red[0]) / H + norm.post_eps);
    fp32x2_t ga = {1.0f, 1.0f}, gb = {1.0f, 1.0f};
    if (G && !cw) ga.x = ga.y = *al, gb.x = gb.y = *be;
    FV nv[kFParts];
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      const int h = q / kFPairs, l = q % kFPairs;
      fp32x2_t nb = muse_ag_norm::mul_pair(val[q], {pr, pr});
      if constexpr (!S && P) nb = muse_ag_norm::mul_pair(nb, w[q]);
      nb = round_bf16(nb);
      const fp32x2_t s = sv[h][l];
      fp32x2_t ns;
      if constexpr (G) {
        const fp32x2_t a = cw ? av[h][l] : ga;
        const fp32x2_t b = cw ? bv[h][l] : gb;
        ns.x = a.x * s.x + b.x * nb.x;
        ns.y = a.y * s.y + b.y * nb.y;
      } else {
        ns.x = s.x + nb.x;
        ns.y = s.y + nb.y;
      }
      nv[h][l] = ns;
      val[q] = ns;
    }
#pragma unroll
    for (int j = 0; j < kFParts; ++j) nv[j].store(static_cast<fp32_t*>(norm.new_stream) + base, t * kFParts + j);
    const float qr = math::rsqrt(reduce(chained(val), red[1]) / H + norm.pre_eps);
    BV ov;
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      fp32x2_t v = muse_ag_norm::mul_pair(val[q], {qr, qr});
      if (hq) v = muse_ag_norm::mul_pair(v, cast<fp32x2_t>(qv[q]));
      ov[q] = cast<bf16x2_t>(v);
    }
    ov.store(static_cast<bf16_t*>(norm.output) + base, t);
    __syncthreads();
  }
  __syncthreads();
  if (t == 0) params.counter[blockIdx.x].set(phase ^ 1);
}
}  // namespace okc_ag_norm

template <int64_t H, uint32_t W, bool PDL>
struct OkcAgNorm {
  using TV = tvm::ffi::TensorView;
  using OTV = tvm::ffi::Optional<TV>;
  using Fn = void (*)(muse_ag_norm::Params);
  template <size_t... I>
  static constexpr std::array<Fn, 8> tab(std::index_sequence<I...>) {
    return {okc_ag_norm::kern<H, W, PDL, (I & 4) != 0, (I & 2) != 0, (I & 1) != 0>...};
  }
  static void launch(muse_ag_norm::Params p, TV local, TV st, TV out, TV ns, OTV pw, OTV qw, OTV al, OTV be,
                     double peps, double qeps, bool scale, int64_t grid) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({M, H / W}).with_dtype<bf16_t>().with_device(dev).verify(local);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(dev).verify(st).verify(ns);
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(dev).verify(out);
    CHECK_HOST(al.has_value() == be.has_value());
    const auto ptr = [](OTV v) { return v.has_value() ? v.value().data_ptr() : nullptr; };
    p.num_rows = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(int64_t{p.num_rows} * H * 2 * W <= int64_t{p.half_bytes}) << "gathered rows exceed the push slot";
    p.local = static_cast<const bf16_t*>(local.data_ptr());
    p.norm = NormResidualParams{nullptr, st.data_ptr(), out.data_ptr(), ns.data_ptr(), nullptr, ptr(pw), ptr(qw), ptr(al),
                                ptr(be), static_cast<float>(peps), static_cast<float>(qeps),
                                static_cast<int>(p.num_rows), al.has_value() && al.value().numel() == H,
                                qw.has_value(), false};
    const auto fn = tab(std::make_index_sequence<8>{})[scale * 4 + pw.has_value() * 2 + al.has_value()];
    const auto g = grid > 0 ? static_cast<uint32_t>(grid) : std::min(p.num_rows, p.num_counters);
    LaunchKernel(g, H / 8, dev.unwrap()).enable_pdl(PDL)(fn, p);
  }
  static void run(CommunicatorRef ref, TV local, TV st, TV out, TV ns, OTV pw, OTV qw, OTV al, OTV be, double peps,
                  double qeps, bool scale) {
    const auto& push = ref.get()->get_push_obj();
    CHECK_HOST(push.world_size == W && push.mc_workspace != nullptr);
    muse_ag_norm::Params p{};
    p.ws_local = push.workspaces[push.rank];
    p.ws_mc = push.mc_workspace;
    p.counter = push.counter;
    p.num_counters = push.num_blocks;
    p.half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size);
    p.rank = push.rank;
    launch(p, local, st, out, ns, pw, qw, al, be, peps, qeps, scale, 0);
  }
};
}  // namespace sglang
