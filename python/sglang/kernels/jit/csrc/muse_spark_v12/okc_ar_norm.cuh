/**
 * \brief OKC: TP all-reduce (lamport 2-shot push) fused with the post-attention norm_residual boundary.
 *
 * x: [M, H] bf16, this rank's partial (the o-proj GEMM output before its all-reduce).
 * stream / output / new_stream / weights / gates: as muse_spark_norm_residual (norm_residual.cuh).
 *
 * M % world == 0, so all_reduce_2shot_push_kernel's chunk c is exactly rows [c * M / world, (c + 1) * M / world):
 * every row has an owner rank. One CTA of H / 8 threads handles one row (rows strided over the grid), one 16 B
 * vector per thread. Not owned: push the partial to the owner's slot, poll the owner's reduced vector. Owned: poll the
 * peers' partials, reduce in rank order, fan the sum out. Slots, sentinels, epochs and the reduce order are
 * all_reduce_2shot_push_kernel's, so the sum is bit-identical to it, and the push plane stays shared with the other
 * push-family kernels (all-reduces, the muse all-gather) in one graph. The norm then runs on the reduced row with
 * muse_spark_norm_residual_cluster's arithmetic and reduction tree reproduced inside the CTA (virtual cluster of kC
 * CTAs of H / 8 / kC threads), so the outputs are bit-identical to 2shot_push -> cluster norm_residual.
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "norm_residual.cuh"

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace okc_ar_norm {

using device::distributed::Counter;
using device::distributed::PushEpoch;
using device::distributed::PushWorkSpace;

template <uint32_t kWorld>
struct Params {
  PushWorkSpace<kWorld> ws;
  uint32_t num_counters;  // the push plane's counter array: every one is flipped once per call
  uint32_t rank;
  uint32_t num_rows;
  const bf16_t* x;
  const fp32_t* stream;
  bf16_t* output;
  fp32_t* new_stream;
  const bf16_t* post_weight;
  const bf16_t* pre_weight;
  const fp32_t* alpha;
  const fp32_t* beta;
  float post_eps;
  float pre_eps;
  bool gate_channel_wise;
  bool has_pre_weight;
};

template <
    int64_t kHidden,
    uint32_t kWorld,
    int kC,
    bool kUsePDL,
    bool kPostNormIsInputScale,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kHidden / device::norm_residual::kVec, 1)
    ar_norm_kernel(const __grid_constant__ Params<kWorld> params) {
  using namespace device;
  using namespace device::norm_residual;

  constexpr int kThreads = kHidden / kVec;  // one 16 B vector of the row per thread
  constexpr int kT = kThreads / kC;         // the cluster kernel's CTA width
  constexpr int kW = kT / kWarpThreads;     // and its warps
  constexpr float kInvHidden = 1.0f / static_cast<float>(kHidden);
  static_assert(kC == 4 || kC == 8 || kC == 16);
  static_assert(kT % kWarpThreads == 0 && kW <= kWarpThreads);

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;
  using Lamport = distributed::LamportTrait<bf16_t, kVec, /*kAtom=*/4>;

  __shared__ float warp_sums[4][kThreads / kWarpThreads];
  __shared__ alignas(16) float partials[kC][4];

  const int tid = threadIdx.x;
  const int lane = tid % kWarpThreads;
  const int warp_index = tid / kWarpThreads;
  const int vc = tid / kT;  // virtual cluster rank of this thread
  const uint32_t r = params.rank;
  const uint32_t rows_per_rank = params.num_rows / kWorld;
  const uint32_t chunk_vecs = rows_per_rank * kThreads;
  const bool channel_wise = kGate && params.gate_channel_wise;
  const bool has_pre_weight = params.has_pre_weight;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = PushEpoch<kWorld>{params.ws};

  // reduce-scatter: my partial of every row I do not own, to its owner
  for (uint32_t row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    const uint32_t owner = row / rows_per_rank;
    if (owner == r) continue;
    BF16Vec vec;
    vec.load(params.x + static_cast<int64_t>(row) * kHidden, tid);
    Lamport::clear_pos_zero(vec.data());
    ptx::st_relaxed_16B(vec, epoch.slot_ptr(/*dst=*/owner, /*src=*/r), (row - owner * rows_per_rank) * kThreads + tid);
  }

  BF16Vec pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());

  for (uint32_t row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    const uint32_t owner = row / rows_per_rank;
    const uint32_t j = (row - owner * rows_per_rank) * kThreads + tid;
    const int64_t base = static_cast<int64_t>(row) * kHidden;

    // the row's other operands first, so their latency overlaps the wait for peer data
    BF16Vec post_weights;
    BF16Vec pre_weights;
    F32Vec stream_values[kFParts];
    F32Vec alpha_values[kFParts];
    F32Vec beta_vector[kFParts];
    if constexpr (kPostWeight) post_weights.load(params.post_weight, tid);
    if (has_pre_weight) pre_weights.load(params.pre_weight, tid);
#pragma unroll
    for (int part = 0; part < kFParts; ++part) {
      const int float_index = tid * kFParts + part;
      stream_values[part].load(params.stream + base, float_index);
      if constexpr (kGate) {
        if (channel_wise) {
          alpha_values[part].load(params.alpha, float_index);
          beta_vector[part].load(params.beta, float_index);
        }
      }
    }
    fp32x2_t gate_a = {1.0f, 1.0f};
    fp32x2_t gate_b = {1.0f, 1.0f};
    if constexpr (kGate) {
      if (!channel_wise) {
        gate_a.x = gate_a.y = *params.alpha;
        gate_b.x = gate_b.y = *params.beta;
      }
    }

    // ---- all-reduce of this vector (all_reduce_2shot_push_kernel's arithmetic and slots)
    BF16Vec branch_values;
    if (owner == r) {
      BF16Vec vec[kWorld];
      vec[r].load(params.x + base, tid);
      Lamport::clear_pos_zero(vec[r].data());
      do {
        bool has_zero = false;
#pragma unroll
        for (uint32_t i = 0; i < kWorld; ++i) {
          if (i != r) ptx::ld_relaxed_16B(vec[i], epoch.slot_ptr(/*dst=*/r, /*src=*/i), j);
        }
#pragma unroll
        for (uint32_t i = 0; i < kWorld; ++i) {
          if (i != r) has_zero |= Lamport::has_pos_zero(vec[i].data());
        }
        if (!has_zero) break;
      } while (true);
      branch_values = reduce_vec(vec);
      Lamport::clear_pos_zero(branch_values.data());
#pragma unroll
      for (uint32_t i = 0; i < kWorld; ++i) {
        if (i != r) ptx::st_relaxed_16B(branch_values, epoch.slot_ptr(/*dst=*/i, /*src=*/r), chunk_vecs + j);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorld; ++i) {
        if (i != r) ptx::st_global_16B(pos_zero_vec, epoch.slot_ptr(/*dst=*/r, /*src=*/i), j);
      }
    } else {
      const auto poll_ptr = epoch.slot_ptr(/*dst=*/r, /*src=*/owner);
      do {
        ptx::ld_relaxed_16B(branch_values, poll_ptr, chunk_vecs + j);
      } while (Lamport::has_pos_zero(branch_values.data()));
      ptx::st_global_16B(pos_zero_vec, poll_ptr, chunk_vecs + j);
    }

    // ---- norm_residual (muse_spark_norm_residual_cluster, first pass)
    fp32x2_t values[kPairs];
    fp32x2_t post_w[kPairs];
    fp32x2_t residual_terms[kPairs];
    fp32x2_t beta_values[kPairs];
    float branch_sum = 0.0f;
    float residual_sum = 0.0f;
    float cross_sum = 0.0f;
    float scaled_branch_sum = 0.0f;
#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      const int part = pair / kFPairs;
      const int lane_in_part = pair % kFPairs;
      auto value = cast<fp32x2_t>(branch_values[pair]);
      if constexpr (kPostWeight) {
        const auto weight = cast<fp32x2_t>(post_weights[pair]);
        post_w[pair] = weight;
        if constexpr (kPostNormIsInputScale) {
          value.x *= weight.x;
          value.y *= weight.y;
          value = round_bf16(value);
        }
      }
      auto moment_value = value;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        moment_value.x *= post_w[pair].x;
        moment_value.y *= post_w[pair].y;
      }
      const auto stream_value = stream_values[part][lane_in_part];
      auto alpha_value = gate_a;
      auto beta_value = gate_b;
      if constexpr (kGate) {
        if (channel_wise) {
          alpha_value = alpha_values[part][lane_in_part];
          beta_value = beta_vector[part][lane_in_part];
        }
      }
      const fp32x2_t residual = {alpha_value.x * stream_value.x, alpha_value.y * stream_value.y};
      const fp32x2_t scaled_branch = {beta_value.x * moment_value.x, beta_value.y * moment_value.y};
      values[pair] = value;
      residual_terms[pair] = residual;
      beta_values[pair] = beta_value;
      branch_sum = fmaf(value.x, value.x, branch_sum);
      branch_sum = fmaf(value.y, value.y, branch_sum);
      residual_sum = fmaf(residual.x, residual.x, residual_sum);
      residual_sum = fmaf(residual.y, residual.y, residual_sum);
      cross_sum = fmaf(residual.x, scaled_branch.x, cross_sum);
      cross_sum = fmaf(residual.y, scaled_branch.y, cross_sum);
      scaled_branch_sum = fmaf(scaled_branch.x, scaled_branch.x, scaled_branch_sum);
      scaled_branch_sum = fmaf(scaled_branch.y, scaled_branch.y, scaled_branch_sum);
    }
    PDLTriggerSecondary<kUsePDL>();

    // ---- the cluster kernel's reduction tree: local_reduce per virtual CTA, then dsmem_reduce
    branch_sum = warp::reduce_sum(branch_sum);
    residual_sum = warp::reduce_sum(residual_sum);
    cross_sum = warp::reduce_sum(cross_sum);
    scaled_branch_sum = warp::reduce_sum(scaled_branch_sum);
    if (lane == 0) {
      warp_sums[0][warp_index] = branch_sum;
      warp_sums[1][warp_index] = residual_sum;
      warp_sums[2][warp_index] = cross_sum;
      warp_sums[3][warp_index] = scaled_branch_sum;
    }
    __syncthreads();
    const int first_warp = vc * kW;
    branch_sum = warp::reduce_sum(lane < kW ? warp_sums[0][first_warp + lane] : 0.0f);
    residual_sum = warp::reduce_sum(lane < kW ? warp_sums[1][first_warp + lane] : 0.0f);
    cross_sum = warp::reduce_sum(lane < kW ? warp_sums[2][first_warp + lane] : 0.0f);
    scaled_branch_sum = warp::reduce_sum(lane < kW ? warp_sums[3][first_warp + lane] : 0.0f);
    if (tid % kT == 0) {
      partials[vc][0] = branch_sum;
      partials[vc][1] = residual_sum;
      partials[vc][2] = cross_sum;
      partials[vc][3] = scaled_branch_sum;
    }
    __syncthreads();
    {
      const int slot = lane & (kC - 1);
      float total0 = 0.0f;
      float total1 = 0.0f;
      float total2 = 0.0f;
      float total3 = 0.0f;
      if (slot != vc) {
        const auto remote = reinterpret_cast<const float4*>(partials)[slot];
        total0 = remote.x;
        total1 = remote.y;
        total2 = remote.z;
        total3 = remote.w;
      }
      branch_sum += warp::reduce_sum<kC>(total0);
      residual_sum += warp::reduce_sum<kC>(total1);
      cross_sum += warp::reduce_sum<kC>(total2);
      scaled_branch_sum += warp::reduce_sum<kC>(total3);
    }
    const float post_rstd = math::rsqrt(branch_sum * kInvHidden + params.post_eps);
    const float pre_sum = fmaf(post_rstd, fmaf(post_rstd, scaled_branch_sum, 2.0f * cross_sum), residual_sum);
    const float pre_rstd = math::rsqrt(fmaxf(pre_sum, 0.0f) * kInvHidden + params.pre_eps);

    // ---- second pass
    BF16Vec output_values;
    F32Vec new_stream_values[kFParts];
#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      const int part = pair / kFPairs;
      const int lane_in_part = pair % kFPairs;
      auto normed = values[pair];
      normed.x *= post_rstd;
      normed.y *= post_rstd;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        normed.x *= post_w[pair].x;
        normed.y *= post_w[pair].y;
      }
      normed = round_bf16(normed);
      const auto beta_value = beta_values[pair];
      const auto residual = residual_terms[pair];
      const fp32x2_t updated = {
          fmaf(beta_value.x, normed.x, residual.x),
          fmaf(beta_value.y, normed.y, residual.y),
      };
      new_stream_values[part][lane_in_part] = updated;
      fp32x2_t normalized = {updated.x * pre_rstd, updated.y * pre_rstd};
      if (has_pre_weight) {
        const auto weight = cast<fp32x2_t>(pre_weights[pair]);
        normalized.x *= weight.x;
        normalized.y *= weight.y;
      }
      output_values[pair] = cast<bf16x2_t>(normalized);
    }
    output_values.store(params.output + base, tid);
#pragma unroll
    for (int part = 0; part < kFParts; ++part) {
      new_stream_values[part].store(params.new_stream + base, tid * kFParts + part);
    }
    __syncthreads();  // warp_sums / partials are reused by the next row
  }

  // every counter flips once per call, also the ones past this grid (all push-family kernels share them)
  epoch.flip();
  if (blockIdx.x == 0) epoch.unsafe_flip_range(gridDim.x, params.num_counters);
}

}  // namespace okc_ar_norm

using host::distributed::CommunicatorRef;

template <int64_t kHidden, uint32_t kWorld, bool kUsePDL>
struct OkcArNorm {
  using TensorView = tvm::ffi::TensorView;
  static constexpr int kThreads = kHidden / device::norm_residual::kVec;

  template <int kC, bool kScale, bool kPostWeight, bool kGate>
  static constexpr auto kernel =
      okc_ar_norm::ar_norm_kernel<kHidden, kWorld, kC, kUsePDL, kScale, kPostWeight, kGate>;

  static void launch(
      okc_ar_norm::Params<kWorld> params,
      DLDevice device,
      uint32_t grid,
      int64_t cluster,
      bool scale,
      bool has_post_weight,
      bool gate) {
    using namespace host;
    const auto go = [&]<int kC>() {
      const auto fn = scale ? (has_post_weight ? (gate ? kernel<kC, 1, 1, 1> : kernel<kC, 1, 1, 0>)
                                               : (gate ? kernel<kC, 1, 0, 1> : kernel<kC, 1, 0, 0>))
                            : (has_post_weight ? (gate ? kernel<kC, 0, 1, 1> : kernel<kC, 0, 1, 0>)
                                               : (gate ? kernel<kC, 0, 0, 1> : kernel<kC, 0, 0, 0>));
      LaunchKernel(grid, kThreads, device).enable_pdl(kUsePDL)(fn, params);
    };
    if (cluster == 16)
      go.template operator()<16>();
    else if (cluster == 8)
      go.template operator()<8>();
    else if (cluster == 4)
      go.template operator()<4>();
    else
      CHECK_HOST(false) << "virtual cluster must be 4, 8 or 16, got " << cluster;
  }

  static okc_ar_norm::Params<kWorld> make_params(
      TensorView x,
      TensorView stream,
      TensorView output,
      TensorView new_stream,
      tvm::ffi::Optional<TensorView> post_weight,
      tvm::ffi::Optional<TensorView> pre_weight,
      tvm::ffi::Optional<TensorView> alpha,
      tvm::ffi::Optional<TensorView> beta,
      double post_eps,
      double pre_eps,
      DLDevice& device_out) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto G = SymbolicSize{"gate_dim"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({M, kHidden}).with_dtype<fp32_t>().with_device(device).verify(stream);
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device).verify(output);
    TensorMatcher({M, kHidden}).with_dtype<fp32_t>().with_device(device).verify(new_stream);
    if (post_weight.has_value())
      TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(post_weight.value());
    if (pre_weight.has_value())
      TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(pre_weight.value());
    const bool gate = alpha.has_value();
    CHECK_HOST(gate == beta.has_value()) << "alpha and beta must both be present or both absent";
    bool channel_wise = false;
    if (gate) {
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device).verify(alpha.value());
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device).verify(beta.value());
      CHECK_HOST(G.unwrap() == 1 || G.unwrap() == kHidden) << "alpha/beta must have 1 or hidden_size elements";
      channel_wise = G.unwrap() == kHidden;
    }
    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(num_rows >= kWorld && num_rows % kWorld == 0) << "rows must be a positive multiple of the world size";
    device_out = device.unwrap();
    return okc_ar_norm::Params<kWorld>{
        .ws = {},
        .num_counters = 0,
        .rank = 0,
        .num_rows = num_rows,
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .stream = static_cast<const fp32_t*>(stream.data_ptr()),
        .output = static_cast<bf16_t*>(output.data_ptr()),
        .new_stream = static_cast<fp32_t*>(new_stream.data_ptr()),
        .post_weight = post_weight.has_value() ? static_cast<const bf16_t*>(post_weight.value().data_ptr()) : nullptr,
        .pre_weight = pre_weight.has_value() ? static_cast<const bf16_t*>(pre_weight.value().data_ptr()) : nullptr,
        .alpha = gate ? static_cast<const fp32_t*>(alpha.value().data_ptr()) : nullptr,
        .beta = gate ? static_cast<const fp32_t*>(beta.value().data_ptr()) : nullptr,
        .post_eps = static_cast<float>(post_eps),
        .pre_eps = static_cast<float>(pre_eps),
        .gate_channel_wise = channel_wise,
        .has_pre_weight = pre_weight.has_value(),
    };
  }

  // The server path: the CustomAllReduceV2 communicator's push plane.
  static void
  run(CommunicatorRef ref,
      TensorView x,
      TensorView stream,
      TensorView output,
      TensorView new_stream,
      tvm::ffi::Optional<TensorView> post_weight,
      tvm::ffi::Optional<TensorView> pre_weight,
      tvm::ffi::Optional<TensorView> alpha,
      tvm::ffi::Optional<TensorView> beta,
      double post_eps,
      double pre_eps,
      bool post_norm_is_input_scale,
      int64_t cluster) {
    using namespace host;
    DLDevice device;
    auto params =
        make_params(x, stream, output, new_stream, post_weight, pre_weight, alpha, beta, post_eps, pre_eps, device);
    const auto& push = ref.get()->get_push_obj();
    CHECK_HOST(push.world_size == kWorld) << "the kernel is compiled for TP" << kWorld;
    const auto chunk_vecs = int64_t{params.num_rows} / kWorld * kThreads;
    params.ws = push.get_workspace<kWorld>(2 * chunk_vecs * 16);
    params.num_counters = push.num_blocks;
    params.rank = push.rank;
    const auto grid = std::min(params.num_rows, push.num_blocks);
    launch(params, device, grid, cluster, post_norm_is_input_scale, post_weight.has_value(), alpha.has_value());
  }

  // Test path: explicit per-rank workspaces (a CPU int64 tensor of kWorld device pointers, each 2 * kWorld slots of
  // slot_bytes) and counters, so kWorld emulated ranks can run on one GPU on separate streams.
  static void run_emulated(
      TensorView workspaces,
      TensorView counter,
      int64_t slot_bytes,
      int64_t rank,
      int64_t grid,
      TensorView x,
      TensorView stream,
      TensorView output,
      TensorView new_stream,
      tvm::ffi::Optional<TensorView> post_weight,
      tvm::ffi::Optional<TensorView> pre_weight,
      tvm::ffi::Optional<TensorView> alpha,
      tvm::ffi::Optional<TensorView> beta,
      double post_eps,
      double pre_eps,
      bool post_norm_is_input_scale,
      int64_t cluster) {
    using namespace host;
    DLDevice device;
    auto params =
        make_params(x, stream, output, new_stream, post_weight, pre_weight, alpha, beta, post_eps, pre_eps, device);
    CHECK_HOST(workspaces.device().device_type == kDLCPU && workspaces.numel() == kWorld);
    const auto ptrs = static_cast<const int64_t*>(workspaces.data_ptr());
    for (uint32_t i = 0; i < kWorld; ++i) params.ws.workspaces[i] = reinterpret_cast<uint8_t*>(ptrs[i]);
    params.ws.counter = static_cast<okc_ar_norm::Counter*>(counter.data_ptr());
    params.ws.mc_workspace = nullptr;
    params.ws.slot_bytes = static_cast<uint32_t>(slot_bytes);
    params.num_counters = static_cast<uint32_t>(counter.numel());
    params.rank = static_cast<uint32_t>(rank);
    CHECK_HOST(2 * (int64_t{params.num_rows} / kWorld) * kThreads * 16 <= slot_bytes) << "slot too small";
    launch(params, device, static_cast<uint32_t>(grid), cluster, post_norm_is_input_scale, post_weight.has_value(),
           alpha.has_value());
  }
};

}  // namespace sglang
