"""The pg chain for the NVFP4 Marlin experts of Muse Spark (prefill chunks only).

At T >= OKC_MOE_PG_MIN_TOKENS a captured layer runs pg_moe.grouped_pg and then the same finalize as
_forward_marlin; every other call (follow-up prefills, verify steps) stays on stock Marlin.
The pg packs are re-encodings of the raw NVFP4 expert tensors, so maybe_capture_nvfp4_packs must run
right before prepare_moe_nvfp4_layer_for_marlin repacks them in place.
"""

import logging

import torch

from sglang.kernels.ops.muse_spark_v12 import okc_patch as _sw

logger = logging.getLogger(__name__)

_captured_layers = 0
_logged_first_call = False
_pack_streams = None


def _streams():
    global _pack_streams
    if _pack_streams is None:
        _pack_streams = (torch.cuda.Stream(), torch.cuda.Stream())
    return _pack_streams


def build_resident(build) -> dict:
    """Run build() -> {name: tensor} on the side stream and copy each result into a block of its own on the
    keep-stream: the allocator discipline of _build_packs, for the load-time packs of the dense projections and the
    router (proj.prepare_weight, router.prepare_gate). Temporaries never share a segment with a resident copy."""
    side, keep = _streams()
    cur = torch.cuda.current_stream()
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        built = build()
    keep.wait_stream(side)
    with torch.cuda.stream(keep):
        out = {name: torch.empty_like(t).copy_(t) for name, t in built.items()}
    cur.wait_stream(keep)
    keep.synchronize()
    del built
    return out


def _build_packs(layer) -> dict | None:
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    sf13 = layer.w13_weight_scale.data
    sf2 = layer.w2_weight_scale.data
    gs13 = layer.w13_weight_scale_2.data
    if w13.dtype != torch.uint8 or w2.dtype != torch.uint8 or not w13.is_cuda:
        return None
    if w13.dim() != 3 or w2.dim() != 3 or sf13.dim() != 3 or sf2.dim() != 3:
        return None
    E, I2, HB = w13.shape
    H = w2.shape[1]
    # The pg kernels are built for I = H = 4096 (K = 4096 in both GEMMs).
    if H != 4096 or HB * 2 != H or I2 != 2 * H or w2.shape[0] != E or w2.shape[2] * 2 != H:
        return None
    if gs13.dim() != 1 or gs13.numel() != E:
        return None
    if sf13.shape[:2] != (E, I2) or sf2.shape[:2] != (E, H):
        return None
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe

    if _sw.MOE_PG_ONLY:
        # A pg-only layer takes pg on every forward, CUDA-graph capture included, so its
        # extension has to be built here rather than lazily inside the first captured forward.
        pg_moe.pg_ext()
        if _sw.PG_NARROW_MIN_T <= _sw.PG_NARROW_MAX_T:
            pg_moe.pg_narrow_ext()

    # pack_wsf wants the raw stacked [gate; up] rows and raw e4m3 scale bytes; gs13 is the
    # per-expert scale Marlin also uses (its constant gain is cancelled by the post-expert norm).
    # Built in place, the packs land in split blocks of freed temporaries and raw weights, and the
    # caching allocator can never return those segments: 11.6 GiB/rank of KV pool lost over 62 layers
    # (serving-patch/smokes/okc_pack_mem_probe.py). So the temporaries live on a side stream (their
    # blocks are reused only there and are all free afterwards) and each pack is copied into a block of
    # its own on a keep-stream that never frees anything.
    side, keep = _streams()
    cur = torch.cuda.current_stream()
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        built = {
            "w13_gs32": gs13.float().contiguous(),
            "pg_wsf13": pg_moe.pack_wsf(w13.contiguous(), sf13.contiguous().view(torch.uint8), gated=True),
            "pg_wsf2": pg_moe.pack_wsf(w2.contiguous(), sf2.contiguous().view(torch.uint8), gated=False),
        }
    keep.wait_stream(side)
    packs = {"H": H}
    with torch.cuda.stream(keep):
        for name, t in built.items():
            packs[name] = torch.empty_like(t).copy_(t)
    cur.wait_stream(keep)
    keep.synchronize()
    del built
    return packs


def maybe_capture_nvfp4_packs(layer) -> bool:
    """Build the pg packs of this layer while the raw NVFP4 layout still exists (layer budget permitting).
    True means the layer runs pg only: the caller skips the Marlin repack, and the raw tensors are freed."""
    global _captured_layers
    if not _sw.SW_MOE_PG or _captured_layers >= _sw.MOE_PACK_MAX_LAYERS:
        return False
    from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts

    if not isinstance(layer, MuseSpark1xFp4Experts):
        return False
    state = _build_packs(layer)
    if state is None:
        logger.warning("OKC pg pack capture declined (shape/dtype mismatch); layer stays on Marlin.")
        return False
    state["pg_only"] = _sw.MOE_PG_ONLY and layer.num_local_experts == 32 and layer.top_k == 8
    layer._okc_moe_state = state
    _captured_layers += 1
    if state["pg_only"]:
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            old = getattr(layer, name)
            setattr(layer, name, torch.nn.Parameter(old.data.new_empty((0,)), requires_grad=False))
    logger.info(
        "OKC pg packs captured for MoE layer %d (budget %d, %s).",
        _captured_layers,
        _sw.MOE_PACK_MAX_LAYERS,
        "pg only" if state["pg_only"] else "Marlin kept",
    )
    return state["pg_only"]


def maybe_forward(layer, hidden_states, topk_scores, topk_ids, post_expert_norm, pre_expert_norm=None):
    """Run the pg chain for this batch, or None to fall back to stock Marlin. Same contract as
    _forward_marlin: applies pre_expert_norm itself, returns the EP-partial [T, H] combined output."""
    global _logged_first_call
    state = layer._okc_moe_state
    if state is None:
        return None
    tokens = hidden_states.shape[0]
    if not state["pg_only"] and (tokens < _sw.MOE_PG_MIN_TOKENS or torch.compiler.is_compiling()):
        return None
    if tokens == 0:
        return hidden_states.new_empty((0, hidden_states.shape[1]))
    # A pg-only layer has no Marlin copy to fall back to, so its declines are errors.
    decline = _decline_reason(layer, hidden_states, topk_ids, post_expert_norm, state)
    if decline is not None:
        if state["pg_only"]:
            raise RuntimeError(f"OKC pg-only MoE layer cannot run this batch: {decline}")
        return None
    if pre_expert_norm is not None:
        hidden_states = pre_expert_norm(hidden_states)
    if not _logged_first_call:
        _logged_first_call = True
        logger.info("OKC pg MoE active (first T=%d).", tokens)
    return _moe_pg(layer, hidden_states.contiguous(), topk_scores, topk_ids, state)


def _decline_reason(layer, hidden_states, topk_ids, post_expert_norm, state):
    if torch.compiler.is_compiling():
        return "inside torch.compile"
    if hidden_states.shape[1] != state["H"] or hidden_states.dtype is not torch.bfloat16:
        return f"hidden {tuple(hidden_states.shape)} {hidden_states.dtype}"
    # The pg kernels hard-code topk 8 and 32 local experts (the TP8/EP8 shape).
    if topk_ids.shape[1] != 8 or layer.num_local_experts != 32:
        return f"topk {topk_ids.shape[1]}, {layer.num_local_experts} local experts"
    if post_expert_norm is None:
        return "no post-expert norm"
    if layer._post_expert_norm is not post_expert_norm:
        layer.set_post_expert_norm(post_expert_norm)
    if layer._post_expert_norm_weight is None:
        return "post-expert norm without a weight"
    return None


def _moe_pg(layer, x, topk_scores, topk_ids, state):
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    num_tokens, size_k = x.shape
    local_ids = topk_ids.to(torch.int32)
    if layer._expert_map is not None:
        local_ids = layer._expert_map[local_ids]
    local_ids = local_ids.contiguous()
    # Marlin's alignment convention: global expert count with -1 for non-local routes.
    align_num_experts = layer.global_num_experts if layer._expert_map is not None else layer.num_local_experts
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, pg_moe.block_m(num_tokens), align_num_experts)
    y = pg_moe.grouped_pg(
        x, sorted_ids, block_experts, n_padded,
        wsf13=state["pg_wsf13"], gs13=state["w13_gs32"], wsf2=state["pg_wsf2"],
    )
    return fused_post_expert_norm_weighted_sum(
        y.view(num_tokens, 8, size_k),
        topk_scores,
        norm_weight=layer._post_expert_norm_weight,
        eps=layer._post_expert_norm_eps,
        weight_first=layer._post_expert_norm_weight_first,
        slot_expert_ids=local_ids,
        # Below 32 tokens the wrapper's torch chain is 14 launches, ~36 us per layer at T = 8 (verify bs 1) against
        # ~4 us for the kernel (session SC); the kernel's math is the same fp32 norm and weighted sum.
        min_tokens_for_fused=1,
    )
