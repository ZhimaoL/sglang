"""OKC U3 (OKC_U3_DRAFT_SWA, default off): the DFlash draft KV on the target's sliding-window slots
(docs/serving/review-1031/U3-DRAFTKV.md).

Every DFlash2 draft layer is sliding-window attention with the target's window (window_left 2047 in both), and the
draft block's queries are the next verify's positions, so the draft reads only rows the target's SWA layers keep for
that verify. Today's draft pool still holds a row per FULL slot. Under U3 it holds a row per SWA slot: an SWAKVPool with
no full layers, every draft layer in its SWA sub-pool, sized at the target's SWA capacity and indexed by the target's
SWA ids through the target allocator's full -> SWA mapping (shared, as a hybrid-SWA MTP draft's pool already is in
KVCacheConfigurator._build_token_to_kv_pool_allocator). Being an SWAKVPool, nothing else changes: fa3 reads it through
its SWA page table and writes the draft block at the plan's SWA ids, HiCache backs it up as a DRAFT_SWA sidecar on the
target's SWA indices, and the draft runner's translator gives the DFlash worker the SWA write ids for the context rows
(dflash_worker_v2._draft_write_ids / _draft_kv_write_pool). Same values at other addresses; frees (full - SWA tokens)
x 1,280 B per rank, which the hybrid pool configurator then charges per SWA token (active()).

Eligibility is checked once, at the draft pool build; anything else declines with a log line and today's pool.
"""

import logging
import os

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("OKC_U3_DRAFT_SWA", "0") == "1"
_target_runner = None   # armed by the DFlash worker just before its draft pool is built
_active = False


class U3DraftPool(SWAKVPool):
    """The DFlash draft pool on the target's SWA slots: no full layers, every draft layer in the SWA sub-pool."""

    def get_v_head_dim(self):
        return self.swa_kv_pool.v_head_dim


def arm(target_runner) -> None:
    """The DFlash worker, before it allocates the draft pool: the target whose SWA slots the draft would share."""
    global _target_runner
    _target_runner = target_runner if ENABLED else None


def active() -> bool:
    """Whether the DFlash draft KV lives on the target's SWA slots (a pool sizing then charges it per SWA token)."""
    return _active


def is_u3_pool(pool) -> bool:
    return isinstance(pool, U3DraftPool)


def _decline_reason(kvc, target_runner):
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel, get_schedule

    if not kvc.spec_algorithm.is_dflash_family():
        return "not a DFlash draft"
    alloc = kvc.token_to_kv_pool_allocator
    if type(alloc) is not SWATokenToKVPoolAllocator or alloc.swa_req_ring:
        return f"target allocator {type(alloc).__name__}"
    target = alloc.get_kvcache()
    if type(target) is not SWAKVPool or target is not target_runner.token_to_kv_pool:
        return f"target pool {type(target).__name__}"
    if target.full_layer_nums == 0 or target.swa_layer_nums == 0 or target.full_to_swa_index_mapping is None:
        return "target pool is not a full + SWA pool with a mapping"
    if (get_memory().enable_unified_memory or get_parallel().attn_dcp_size != 1
            or get_disagg().disaggregation_mode != "null" or get_schedule().prefill_only_disable_kv_cache):
        return "unified memory / DCP / disaggregation / prefill-only"
    if kvc.pool_page_size != 1 or alloc.page_size != 1:
        return f"page size {kvc.pool_page_size}"
    if (kvc.kv_cache_dtype is not torch.float8_e4m3fn or kvc.kv_cache_dtype_str == "mxfp8"
            or target.swa_kv_pool.dtype is not kvc.kv_cache_dtype):
        return f"KV dtype {kvc.kv_cache_dtype} (target SWA {target.swa_kv_pool.dtype})"
    if bool(kvc.post_capture_kv_active) != bool(target.post_capture_active):
        return "post-capture backing differs from the target's"
    attns = [m for m in kvc.model.modules() if isinstance(m, RadixAttention)]
    ids = sorted(a.layer_id for a in attns)
    info = kvc.layer_info
    if (not attns or info.start_layer != 0 or ids != list(range(info.end_layer))
            or len(ids) != info.num_effective_layers):
        return f"draft layer ids {ids} (layers [{info.start_layer}, {info.end_layer}))"
    window, target_window = kvc.sliding_window_size, target_runner.sliding_window_size
    if window is None or target_window is None or not 0 <= window <= target_window:
        return f"draft window {window} vs target window {target_window}"
    if any(a.sliding_window_size != window or a.is_cross_attention for a in attns):
        return "a draft layer is not sliding-window at the runner's window"
    return None


def build_pool(kvc, mha_pool_class):
    """KVCacheConfigurator._build_token_to_kv_pool, draft worker: the draft pool on the target's SWA slots, or None
    (today's pool)."""
    global _target_runner, _active
    target_runner, _target_runner = _target_runner, None
    if not ENABLED or target_runner is None or not kvc.is_draft_worker:
        return None
    reason = _decline_reason(kvc, target_runner)
    if reason is not None:
        logger.info("OKC U3 draft KV on SWA slots declined: %s", reason)
        return None
    from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel, get_spec

    target = kvc.token_to_kv_pool_allocator.get_kvcache()
    n = kvc.layer_info.num_effective_layers
    pool = U3DraftPool(
        size=target.size,
        size_swa=target.size_swa,
        page_size=kvc.pool_page_size,
        dtype=kvc.kv_cache_dtype,
        head_num=kvc.model_config.get_num_kv_heads(get_parallel().attn_tp_size, get_parallel().attn_dcp_size),
        head_dim=kvc.model_config.head_dim,
        swa_attention_layer_ids=list(range(n)),
        full_attention_layer_ids=[],
        device=kvc.device,
        token_to_kv_pool_class=mha_pool_class,
        enable_memory_saver=get_exec().features.enable_memory_saver,
        v_head_dim=kvc.model_config.v_head_dim,
        enable_alt_stream=not get_disagg().enable_pdmux,
        enable_kv_cache_copy=(get_spec().speculative_algorithm is not None),
        post_capture_active=kvc.post_capture_kv_active,
    )
    # The target allocator's tensor (stable: resize / clear refill it in place), so draft ids follow every SWA alloc
    # and free; its trailing -1 keeps a -1 loc at -1.
    pool.register_mapping(target.full_to_swa_index_mapping)
    _active = True
    sub = pool.swa_kv_pool
    row = (sub.row_dim + sub.v_row_dim) * sub.store_dtype.itemsize * n
    logger.info(
        "OKC U3 draft KV on SWA slots: %d layers x %d SWA tokens, the target's SWA ids (%d B per token; full slots "
        "would be %d tokens)", n, target.size_swa, row, target.size)
    return pool
