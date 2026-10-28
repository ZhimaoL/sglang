"""The pg chain for the NVFP4 Marlin experts of Muse Spark (prefill chunks only).

At T >= OKC_MOE_PG_MIN_TOKENS a captured layer runs pg_moe.grouped_pg and then the same finalize as
_forward_marlin; every other call (follow-up prefills, verify steps) stays on stock Marlin. A pg-only layer (the
default) runs the chain dispatch_path picks from T (st / pg's small-T front end / align / align_local, pg's narrow or
shipped build, the replica balancers).
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
    keep-stream: the allocator discipline of _build_packs, for the router's load-time hi / lo copies
    (router.prepare_gate). Temporaries never share a segment with a resident copy."""
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
        if _sw.SW_MOE_ST:
            from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import st_ext

            st_ext()

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
    state["pg_only"] = _sw.MOE_PG_ONLY and layer.num_local_experts < 255 and layer.top_k == 8
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
    if not _logged_first_call:
        _logged_first_call = True
        logger.info("OKC pg MoE active (first T=%d).", tokens)
    path = dispatch_path(layer, tokens, pre_expert_norm)
    if path.startswith("small"):
        small = _small_front(hidden_states, topk_ids, pre_expert_norm)
        if small is not None:
            return _moe_pg_small(layer, hidden_states, topk_scores, topk_ids, state, *small)
    if path.startswith("st") and _sw.SW_ST_NORM_FOLD:
        fold = _norm_fold(hidden_states, pre_expert_norm)
        if fold is not None:
            return _moe_st(layer, hidden_states, topk_scores, topk_ids, state, *fold)
    if pre_expert_norm is not None:
        hidden_states = pre_expert_norm(hidden_states)
    if path.startswith("st"):
        return _moe_st(layer, hidden_states.contiguous(), topk_scores, topk_ids, state)
    return _moe_pg(layer, hidden_states.contiguous(), topk_scores, topk_ids, state)


def dispatch_path(layer, tokens, pre_expert_norm=None):
    """The chain a pg-only layer runs for a T-token forward, from host-side sizes only (one path per graph bucket):
    "st" (T <= OKC_MOE_ST_MAX_TOKENS), "small" (pg_route / pg_gather_norm, <= 2048 routes; it may still decline at
    run time), "local" (align_local, >= OKC_PG_LOCAL_ALIGN_MIN_TOKENS) or "align" (moe_align_block_size); "-narrow"
    for pg's narrow build; "+rep" (assign_replicas) / "+split" (split_replicas) with replicas."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe
    from sglang.srt.layers.muse_spark_layers import MuseSpark1xRMSNorm

    state = layer._okc_moe_state
    if _sw.SW_MOE_ST and state["pg_only"] and tokens <= _sw.MOE_ST_MAX_TOKENS:
        name = "st"
    else:
        small = tokens * layer.top_k <= pg_moe.SMALL_MAX_ROUTES and type(pre_expert_norm) is MuseSpark1xRMSNorm
        if _sw.SW_PG_SMALL and small:
            name = "small"
        elif _sw.PG_LOCAL_ALIGN and tokens >= _sw.PG_LOCAL_ALIGN_MIN_TOKENS and layer.num_local_experts <= 64:
            name = "local"
        else:
            name = "align"
        if pg_moe.block_m(tokens) != pg_moe._BLOCK_M:
            name += "-narrow"
    mode = _rep_mode(layer, tokens)
    # split_replicas is per route: the small front end's per-expert map (route_map) is static above the limit
    if mode == "assign":
        name += "+rep"
    elif mode == "split" and not name.startswith("small"):
        name += "+split"
    return name


def _moe_st(layer, x, topk_scores, topk_ids, state, norm_weight=None, norm_eps=0.0):
    """The st pair (okc_patch/st_moe.py) on pre-normed x, or on un-normed x with the pre-expert RMSNorm folded into
    its gate/up kernel (norm_weight, norm_eps from _norm_fold)."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import st_moe

    local_ids = local_routes(layer, topk_ids)
    y = st_moe.grouped_st(
        x, local_ids, wsf13=state["pg_wsf13"], gs13=state["w13_gs32"], wsf2=state["pg_wsf2"], pdl=_sw.MOE_ST_PDL,
        norm_weight=norm_weight, norm_eps=norm_eps,
    )
    return finalize_routes(layer, y.view(x.shape[0], 8, x.shape[1]), topk_scores, local_ids)


def _norm_fold(hidden_states, pre_expert_norm):
    """(norm weight, eps) when st can fold the pre-expert norm (F.rms_norm(x, (H,), bf16 weight, eps) reproduced
    exactly in its gate/up kernel), else None (the norm keeps its own launch)."""
    from sglang.srt.layers.muse_spark_layers import MuseSpark1xRMSNorm

    if type(pre_expert_norm) is not MuseSpark1xRMSNorm or not hidden_states.is_contiguous():
        return None
    weight = pre_expert_norm._get_effective_weight()
    if weight is None or weight.dtype is not torch.bfloat16 or not weight.is_contiguous():
        return None
    if weight.numel() != hidden_states.shape[1]:
        return None
    return weight, float(pre_expert_norm.eps)


def _small_front(hidden_states, topk_ids, pre_expert_norm):
    """(norm weight, eps) for the pg_route / pg_gather_norm front end, or None for the moe_align chain."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe
    from sglang.srt.layers.muse_spark_layers import MuseSpark1xRMSNorm

    if not _sw.SW_PG_SMALL or topk_ids.numel() > pg_moe.SMALL_MAX_ROUTES:
        return None
    if topk_ids.dtype is not torch.int32 or not topk_ids.is_contiguous() or not hidden_states.is_contiguous():
        return None
    # The fold reproduces F.rms_norm(x, (H,), bf16 weight, eps) exactly; any other norm keeps its own launch.
    if type(pre_expert_norm) is not MuseSpark1xRMSNorm:
        return None
    weight = pre_expert_norm._get_effective_weight()
    if weight is None or weight.dtype is not torch.bfloat16 or not weight.is_contiguous():
        return None
    if weight.numel() != hidden_states.shape[1]:
        return None
    return weight, float(pre_expert_norm.eps)


def _decline_reason(layer, hidden_states, topk_ids, post_expert_norm, state):
    if torch.compiler.is_compiling():
        return "inside torch.compile"
    if hidden_states.shape[1] != state["H"] or hidden_states.dtype is not torch.bfloat16:
        return f"hidden {tuple(hidden_states.shape)} {hidden_states.dtype}"
    # The pg kernels hard-code topk 8; local slots are 8-bit in the replica table (255 = not held).
    if topk_ids.shape[1] != 8 or layer.num_local_experts >= 255:
        return f"topk {topk_ids.shape[1]}, {layer.num_local_experts} local experts"
    if post_expert_norm is None:
        return "no post-expert norm"
    if layer._post_expert_norm is not post_expert_norm:
        layer.set_post_expert_norm(post_expert_norm)
    if layer._post_expert_norm_weight is None:
        return "post-expert norm without a weight"
    return None


def _rep_mode(layer, tokens):
    """The replica balancer a T-token forward uses: "assign" (<= OKC_MOE_REP_MAX_TOKENS), "split" (>=
    OKC_MOE_REP_SPLIT_MIN_TOKENS: its 3 kernels cost ~9 us per layer, more than they save at ~1K extends), else None
    (the static home map; replicas idle)."""
    if getattr(layer, "_okc_rep_table", None) is None:
        return None
    if tokens <= _sw.MOE_REP_MAX_TOKENS:
        return "assign"
    return "split" if _sw.MOE_REP_PREFILL and tokens >= _sw.MOE_REP_SPLIT_MIN_TOKENS else None


def local_routes(layer, topk_ids):
    """[T, 8] global top-k ids -> [T, 8] int32 local expert slots on this rank, -1 where another rank computes the
    route: the static home map, or with hot-expert replicas a per-forward balancer that every rank computes alike from
    the same ids (T and the ids are the same on every rank: a TP batch, or the SP gathered chunk)."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe

    mode = _rep_mode(layer, topk_ids.shape[0])
    if mode == "assign":
        # verify-sized: each active replicated expert to the holder with less work (st passes on the st chain)
        st = _sw.SW_MOE_ST and topk_ids.shape[0] <= _sw.MOE_ST_MAX_TOKENS
        return pg_moe.assign_replicas(topk_ids, layer._okc_rep_table, layer.replica_plan["me"], passes=st)
    if mode == "split":
        # prefill chunks: every expert is active; each replicated expert's routes split to level routed tokens
        return pg_moe.split_replicas(topk_ids, layer._okc_rep_table, layer.replica_plan["me"])
    local_ids = topk_ids.to(torch.int32)
    if layer._expert_map is not None:
        local_ids = layer._expert_map[local_ids]
    return local_ids.contiguous()


def route_map(layer, topk_ids):
    """local_routes' decision as a [256] global -> slot map, for pg_route: the static map or the balancer's."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe

    rep_table = getattr(layer, "_okc_rep_table", None)
    if rep_table is not None and topk_ids.shape[0] <= _sw.MOE_REP_MAX_TOKENS:
        return pg_moe.assign_replicas(topk_ids, rep_table, layer.replica_plan["me"], as_map=True)
    return layer._expert_map


def _moe_pg_small(layer, x, topk_scores, topk_ids, state, norm_weight, norm_eps):
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe

    y, local_ids = pg_moe.grouped_pg_small(
        x, topk_ids, route_map(layer, topk_ids),
        norm_weight=norm_weight, norm_eps=norm_eps,
        wsf13=state["pg_wsf13"], gs13=state["w13_gs32"], wsf2=state["pg_wsf2"], pdl=_sw.SW_PDL,
        bm=pg_moe.block_m(x.shape[0]),
    )
    return finalize_routes(layer, y.view(x.shape[0], 8, x.shape[1]), topk_scores, local_ids)


def pg_prep(layer, topk_ids):
    """For the SP MoE pipeline (okc_patch/sp_moe_ovl.py): the route alignment of pg, which needs only the top-k ids,
    so it can run while the token rows are still in flight -> (local slot ids [T, 8], -1 = not this rank's expert,
    and moe_align_block_size's sorted ids / block experts / padded count)."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    emap = layer._expert_map
    bm = pg_moe.block_m(topk_ids.shape[0])
    if (_sw.PG_LOCAL_ALIGN and topk_ids.shape[0] >= _sw.PG_LOCAL_ALIGN_MIN_TOKENS
            and (emap is None or emap.dtype is torch.int32) and layer.num_local_experts <= 64):
        # Large chunks: one count of the routes into the local-slot counters (map and int32 cast folded in)
        # instead of the cast + map gather + 256-expert moe_align / count_and_sort chain; same values downstream.
        if _rep_mode(layer, topk_ids.shape[0]) is not None:
            return pg_moe.align_local(local_routes(layer, topk_ids), None, layer.num_local_experts, bm)
        return pg_moe.align_local(topk_ids, emap, layer.num_local_experts, bm)
    local_ids = local_routes(layer, topk_ids)
    # Marlin's alignment convention: global expert count with -1 for non-local routes.
    align_num_experts = layer.global_num_experts if layer._expert_map is not None else layer.num_local_experts
    return (local_ids, *moe_align_block_size(local_ids, bm, align_num_experts))


def pg_run(layer, x, prep, post_expert_norm):
    """pg on x [T, H] (pre-expert-normed) with pg_prep's alignment -> y [T, 8, H] per-route outputs; finalize_routes
    then combines any row range. pg-only layers (the pipeline checks), so a decline is an error."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe

    state = layer._okc_moe_state
    local_ids, sorted_ids, block_experts, n_padded = prep
    decline = _decline_reason(layer, x, local_ids, post_expert_norm, state)
    if decline is not None:
        raise RuntimeError(f"OKC pg-only MoE layer cannot run this batch: {decline}")
    y = pg_moe.grouped_pg(
        x.contiguous(), sorted_ids, block_experts, n_padded,
        wsf13=state["pg_wsf13"], gs13=state["w13_gs32"], wsf2=state["pg_wsf2"], bm=pg_moe.block_m(x.shape[0]),
    )
    return y.view(x.shape[0], 8, x.shape[1])


def finalize_routes(layer, y, topk_scores, local_ids):
    """Post-expert norm and top-k weighted sum of pg's per-route rows (rows of non-local routes are masked)."""
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )

    return fused_post_expert_norm_weighted_sum(
        y,
        topk_scores,
        norm_weight=layer._post_expert_norm_weight,
        eps=layer._post_expert_norm_eps,
        weight_first=layer._post_expert_norm_weight_first,
        slot_expert_ids=local_ids,
        # Below 32 tokens the wrapper's torch chain is 14 launches, ~36 us per layer at T = 8 (verify bs 1) against
        # ~4 us for the kernel (session SC); the kernel's math is the same fp32 norm and weighted sum.
        min_tokens_for_fused=1,
    )


def _moe_pg(layer, x, topk_scores, topk_ids, state):
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe

    prep = pg_prep(layer, topk_ids)
    y = pg_moe.grouped_pg(
        x, *prep[1:],
        wsf13=state["pg_wsf13"], gs13=state["w13_gs32"], wsf2=state["pg_wsf2"], bm=pg_moe.block_m(x.shape[0]),
    )
    return finalize_routes(layer, y.view(x.shape[0], 8, x.shape[1]), topk_scores, prep[0])
