// OKC: all_reduce_2shot_push (its slots, sentinels, epochs, sum order) + muse_spark_norm_residual_cluster (its math
// and C-CTA reduction tree, in one CTA per row). Rank c owns rows [c M/W, (c+1) M/W). Bit-identical to the pair.
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include "norm_residual.cuh"

#include <algorithm>
#include <array>
#include <utility>

namespace sglang {
namespace okc_ar_norm {
using device::distributed::Counter;

template <uint32_t W>
struct Params {
  device::distributed::PushWorkSpace<W> ws;
  uint32_t ncnt, rank, rows;
  const bf16_t* x;
  const fp32_t* st;
  bf16_t* out;
  fp32_t* ns;
  const bf16_t *pw, *qw;
  const fp32_t *al, *be;
  float peps, qeps;
  bool cw, hq;
};

template <int64_t H, uint32_t W, int C, bool PDL, bool S, bool P, bool G>
__global__ void __launch_bounds__(H / 8, 1) kern(const __grid_constant__ Params<W> p) {
  using namespace device;
  using namespace device::norm_residual;
  constexpr int N = H / kVec, T = N / C, NW = T / kWarpThreads;
  constexpr float kInv = 1.0f / static_cast<float>(H);
  using BV = AlignedVector<bf16x2_t, kPairs>;
  using FV = AlignedVector<fp32x2_t, kFPairs>;
  using L = distributed::LamportTrait<bf16_t, kVec, 4>;
  __shared__ float wsum[4][N / kWarpThreads];
  __shared__ alignas(16) float part[C][4];
  const int t = threadIdx.x, lane = t % kWarpThreads, vc = t / T;
  const uint32_t r = p.rank, rpr = p.rows / W, chunk = rpr * N;
  const bool cw = G && p.cw;
  PDLWaitPrimary<PDL>();
  const auto ep = distributed::PushEpoch<W>{p.ws};
  for (uint32_t row = blockIdx.x; row < p.rows; row += gridDim.x) {
    const uint32_t o = row / rpr;
    if (o == r) continue;
    BV v;
    v.load(p.x + int64_t(row) * H, t);
    L::clear_pos_zero(v.data());
    ptx::st_relaxed_16B(v, ep.slot_ptr(o, r), (row - o * rpr) * N + t);
  }
  BV zero;
  L::fill_pos_zero(zero.data());
  for (uint32_t row = blockIdx.x; row < p.rows; row += gridDim.x) {
    const uint32_t o = row / rpr, j = (row - o * rpr) * N + t;
    const int64_t base = int64_t(row) * H;
    BV pw, qw, br;
    FV sv[kFParts], av[kFParts], bv[kFParts];
    if constexpr (P) pw.load(p.pw, t);
    if (p.hq) qw.load(p.qw, t);
#pragma unroll
    for (int k = 0; k < kFParts; ++k) {
      sv[k].load(p.st + base, t * kFParts + k);
      if (cw) av[k].load(p.al, t * kFParts + k), bv[k].load(p.be, t * kFParts + k);
    }
    fp32x2_t ga = {1.0f, 1.0f}, gb = {1.0f, 1.0f};
    if (G && !cw) ga.x = ga.y = *p.al, gb.x = gb.y = *p.be;
    if (o == r) {
      BV v[W];
      v[r].load(p.x + base, t);
      L::clear_pos_zero(v[r].data());
      bool pending;
      do {
        pending = false;
#pragma unroll
        for (uint32_t i = 0; i < W; ++i)
          if (i != r) ptx::ld_relaxed_16B(v[i], ep.slot_ptr(r, i), j);
#pragma unroll
        for (uint32_t i = 0; i < W; ++i)
          if (i != r) pending |= L::has_pos_zero(v[i].data());
      } while (pending);
      br = reduce_vec(v);
      L::clear_pos_zero(br.data());
#pragma unroll
      for (uint32_t i = 0; i < W; ++i)
        if (i != r) ptx::st_relaxed_16B(br, ep.slot_ptr(i, r), chunk + j);
#pragma unroll
      for (uint32_t i = 0; i < W; ++i)
        if (i != r) ptx::st_global_16B(zero, ep.slot_ptr(r, i), j);
    } else {
      const auto src = ep.slot_ptr(r, o);
      do ptx::ld_relaxed_16B(br, src, chunk + j);
      while (L::has_pos_zero(br.data()));
      ptx::st_global_16B(zero, src, chunk + j);
    }
    fp32x2_t val[kPairs], w[kPairs], res[kPairs], bet[kPairs];
    float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      const int k = q / kFPairs, l = q % kFPairs;
      auto v = cast<fp32x2_t>(br[q]);
      if constexpr (P) {
        w[q] = cast<fp32x2_t>(pw[q]);
        if constexpr (S) {
          v.x *= w[q].x;
          v.y *= w[q].y;
          v = round_bf16(v);
        }
      }
      auto m = v;
      if constexpr (!S && P) m.x *= w[q].x, m.y *= w[q].y;
      const auto x = sv[k][l];
      auto a = ga, b = gb;
      if (cw) a = av[k][l], b = bv[k][l];
      const fp32x2_t rs = {a.x * x.x, a.y * x.y}, sb = {b.x * m.x, b.y * m.y};
      val[q] = v, res[q] = rs, bet[q] = b;
      s[0] = fmaf(v.x, v.x, s[0]);
      s[0] = fmaf(v.y, v.y, s[0]);
      s[1] = fmaf(rs.x, rs.x, s[1]);
      s[1] = fmaf(rs.y, rs.y, s[1]);
      s[2] = fmaf(rs.x, sb.x, s[2]);
      s[2] = fmaf(rs.y, sb.y, s[2]);
      s[3] = fmaf(sb.x, sb.x, s[3]);
      s[3] = fmaf(sb.y, sb.y, s[3]);
    }
    PDLTriggerSecondary<PDL>();
    for (int k = 0; k < 4; ++k) {
      s[k] = warp::reduce_sum(s[k]);
      if (lane == 0) wsum[k][t / kWarpThreads] = s[k];
    }
    __syncthreads();
    for (int k = 0; k < 4; ++k) s[k] = warp::reduce_sum(lane < NW ? wsum[k][vc * NW + lane] : 0.0f);
    if (t % T == 0)
      for (int k = 0; k < 4; ++k) part[vc][k] = s[k];
    __syncthreads();
    const int slot = lane & (C - 1);
    const float4 rm = slot != vc ? reinterpret_cast<const float4*>(part)[slot] : make_float4(0.f, 0.f, 0.f, 0.f);
    s[0] += warp::reduce_sum<C>(rm.x);
    s[1] += warp::reduce_sum<C>(rm.y);
    s[2] += warp::reduce_sum<C>(rm.z);
    s[3] += warp::reduce_sum<C>(rm.w);
    const float pr = math::rsqrt(s[0] * kInv + p.peps);
    const float qs = fmaf(pr, fmaf(pr, s[3], 2.0f * s[2]), s[1]);
    const float qr = math::rsqrt(fmaxf(qs, 0.0f) * kInv + p.qeps);
    BV ov;
    FV nv[kFParts];
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      auto n = val[q];
      n.x *= pr;
      n.y *= pr;
      if constexpr (!S && P) n.x *= w[q].x, n.y *= w[q].y;
      n = round_bf16(n);
      const fp32x2_t u = {fmaf(bet[q].x, n.x, res[q].x), fmaf(bet[q].y, n.y, res[q].y)};
      nv[q / kFPairs][q % kFPairs] = u;
      fp32x2_t z = {u.x * qr, u.y * qr};
      if (p.hq) {
        const auto g = cast<fp32x2_t>(qw[q]);
        z.x *= g.x;
        z.y *= g.y;
      }
      ov[q] = cast<bf16x2_t>(z);
    }
    ov.store(p.out + base, t);
#pragma unroll
    for (int k = 0; k < kFParts; ++k) nv[k].store(p.ns + base, t * kFParts + k);
    __syncthreads();
  }
  ep.flip();  // all counters flip once per call
  if (blockIdx.x == 0) ep.unsafe_flip_range(gridDim.x, p.ncnt);
}
}  // namespace okc_ar_norm

template <int64_t H, uint32_t W, bool PDL>
struct OkcArNorm {
  using TV = tvm::ffi::TensorView;
  using OTV = tvm::ffi::Optional<TV>;
  using Pa = okc_ar_norm::Params<W>;
  using Fn = void (*)(Pa);
  template <int C, size_t... I>
  static constexpr std::array<Fn, 8> tab(std::index_sequence<I...>) {
    return {okc_ar_norm::kern<H, W, C, PDL, (I & 4) != 0, (I & 2) != 0, (I & 1) != 0>...};
  }

  static void launch(Pa p, TV x, TV st, TV out, TV ns, OTV pw, OTV qw, OTV al, OTV be, double peps, double qeps,
                     bool scale, int64_t c, int64_t grid) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(dev).verify(x).verify(out);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(dev).verify(st).verify(ns);
    p.rows = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(p.rows >= W && p.rows % W == 0 && (c == 4 || c == 8 || c == 16));
    CHECK_HOST(al.has_value() == be.has_value());
    CHECK_HOST(2 * int64_t{p.rows} / W * (H / 8) * 16 <= p.ws.slot_bytes) << "push slot too small";
    const auto ptr = [](OTV v) { return v.has_value() ? v.value().data_ptr() : nullptr; };
    p.x = static_cast<const bf16_t*>(x.data_ptr());
    p.st = static_cast<const fp32_t*>(st.data_ptr());
    p.out = static_cast<bf16_t*>(out.data_ptr());
    p.ns = static_cast<fp32_t*>(ns.data_ptr());
    p.pw = static_cast<const bf16_t*>(ptr(pw));
    p.qw = static_cast<const bf16_t*>(ptr(qw));
    p.al = static_cast<const fp32_t*>(ptr(al));
    p.be = static_cast<const fp32_t*>(ptr(be));
    p.peps = static_cast<float>(peps);
    p.qeps = static_cast<float>(qeps);
    p.cw = al.has_value() && al.value().numel() == H;
    p.hq = qw.has_value();
    const auto seq = std::make_index_sequence<8>{};
    const auto i = scale * 4 + pw.has_value() * 2 + al.has_value();
    const Fn fn = (c == 16 ? tab<16>(seq) : c == 8 ? tab<8>(seq) : tab<4>(seq))[i];
    const auto g = grid > 0 ? static_cast<uint32_t>(grid) : std::min(p.rows, p.ncnt);
    LaunchKernel(g, H / 8, dev.unwrap()).enable_pdl(PDL)(fn, p);
  }

  static void run(host::distributed::CommunicatorRef ref, TV x, TV st, TV out, TV ns, OTV pw, OTV qw, OTV al, OTV be,
                  double peps, double qeps, bool scale, int64_t c) {
    const auto& push = ref.get()->get_push_obj();
    CHECK_HOST(push.world_size == W);
    Pa p{};
    p.ws = push.get_workspace<W>(push.slot_bytes);
    p.ncnt = push.num_blocks;
    p.rank = push.rank;
    launch(p, x, st, out, ns, pw, qw, al, be, peps, qeps, scale, c, 0);
  }
};
}  // namespace sglang
