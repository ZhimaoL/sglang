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

    # pack_wsf wants the raw stacked [gate; up] rows and raw e4m3 scale bytes; gs13 is the
    # per-expert scale Marlin also uses (its constant gain is cancelled by the post-expert norm).
    return {
        "H": H,
        "w13_gs32": gs13.float().contiguous(),
        "pg_wsf13": pg_moe.pack_wsf(w13.contiguous(), sf13.contiguous().view(torch.uint8), gated=True),
        "pg_wsf2": pg_moe.pack_wsf(w2.contiguous(), sf2.contiguous().view(torch.uint8), gated=False),
    }


def maybe_capture_nvfp4_packs(layer) -> None:
    """Build the pg packs of this layer while the raw NVFP4 layout still exists (layer budget permitting)."""
    global _captured_layers
    if not _sw.SW_MOE_PG or _captured_layers >= _sw.MOE_PACK_MAX_LAYERS:
        return
    from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts

    if not isinstance(layer, MuseSpark1xFp4Experts):
        return
    state = _build_packs(layer)
    if state is None:
        logger.warning("OKC pg pack capture declined (shape/dtype mismatch); layer stays on Marlin.")
        return
    layer._okc_moe_state = state
    _captured_layers += 1
    logger.info("OKC pg packs captured for MoE layer %d (budget %d).", _captured_layers, _sw.MOE_PACK_MAX_LAYERS)


def maybe_forward(layer, hidden_states, topk_scores, topk_ids, post_expert_norm, pre_expert_norm=None):
    """Run the pg chain for this batch, or None to fall back to stock Marlin. Same contract as
    _forward_marlin: applies pre_expert_norm itself, returns the EP-partial [T, H] combined output."""
    global _logged_first_call
    state = layer._okc_moe_state
    if state is None or torch.compiler.is_compiling():
        return None
    tokens = hidden_states.shape[0]
    if tokens < _sw.MOE_PG_MIN_TOKENS or hidden_states.shape[1] != state["H"] or post_expert_norm is None:
        return None
    # The pg kernels hard-code topk 8 and 32 local experts (the TP8/EP8 shape).
    if topk_ids.shape[1] != 8 or layer.num_local_experts != 32:
        return None
    if layer._post_expert_norm is not post_expert_norm:
        layer.set_post_expert_norm(post_expert_norm)
    if layer._post_expert_norm_weight is None:
        return None
    if pre_expert_norm is not None:
        hidden_states = pre_expert_norm(hidden_states)
    if hidden_states.dtype is not torch.bfloat16:
        return None
    if not _logged_first_call:
        _logged_first_call = True
        logger.info("OKC pg MoE active (first T=%d).", tokens)
    return _moe_pg(layer, hidden_states.contiguous(), topk_scores, topk_ids, state)


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
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, pg_moe._BLOCK_M, align_num_experts)
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
    )
