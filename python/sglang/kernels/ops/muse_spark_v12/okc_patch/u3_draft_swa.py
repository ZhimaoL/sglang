"""OKC U3 (OKC_U3_DRAFT_SWA, default off; docs/serving/review-1031/U3-DRAFTKV.md): the DFlash draft KV on the target's
SWA slots. The draft's layers all slide at the target's window and its queries are the next verify's, so it reads only
rows the target's SWA layers keep. Its pool is then an SWAKVPool with no full layers, sized at the target's SWA capacity
and indexed by the target's SWA ids (the target allocator's mapping, shared as a hybrid-SWA MTP draft's is): fa3 reads it
through its SWA page table, HiCache backs it up as a DRAFT_SWA sidecar, and the DFlash worker writes context rows into
its SWA sub-pool at the plan's SWA ids. Same values at other addresses; (full - SWA) x 1,280 B per rank freed."""

import logging
import os

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("OKC_U3_DRAFT_SWA", "0") == "1"
_target_runner = None   # armed by the DFlash worker just before it allocates the draft pool
_active = False


class U3DraftPool(SWAKVPool):
    def get_v_head_dim(self):
        return self.swa_kv_pool.v_head_dim


def arm(target_runner):
    global _target_runner
    _target_runner = target_runner if ENABLED else None


def active():
    """The draft KV is on the SWA slots: pool sizing charges it per SWA token."""
    return _active


def is_u3_pool(pool):
    return isinstance(pool, U3DraftPool)


def _decline_reason(kvc, target_runner):
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel, get_schedule

    alloc = kvc.token_to_kv_pool_allocator
    if not kvc.spec_algorithm.is_dflash_family() or type(alloc) is not SWATokenToKVPoolAllocator or alloc.swa_req_ring:
        return f"draft {kvc.spec_algorithm}, allocator {type(alloc).__name__}"
    target = alloc.get_kvcache()
    if (type(target) is not SWAKVPool or target is not target_runner.token_to_kv_pool or not target.full_layer_nums
            or not target.swa_layer_nums or target.full_to_swa_index_mapping is None):
        return f"target pool {type(target).__name__}"
    if (get_memory().enable_unified_memory or get_parallel().attn_dcp_size != 1 or kvc.pool_page_size != 1
            or alloc.page_size != 1 or get_disagg().disaggregation_mode != "null"
            or get_schedule().prefill_only_disable_kv_cache):
        return "unified memory / DCP / page size / disaggregation / prefill-only"
    if (kvc.kv_cache_dtype is not torch.float8_e4m3fn or kvc.kv_cache_dtype_str == "mxfp8"
            or target.swa_kv_pool.dtype is not kvc.kv_cache_dtype
            or bool(kvc.post_capture_kv_active) != bool(target.post_capture_active)):
        return f"KV dtype {kvc.kv_cache_dtype} / post-capture backing"
    attns = [m for m in kvc.model.modules() if isinstance(m, RadixAttention)]
    ids, info = sorted(a.layer_id for a in attns), kvc.layer_info
    if not attns or info.start_layer != 0 or ids != list(range(info.num_effective_layers)):
        return f"draft layer ids {ids}"
    window, target_window = kvc.sliding_window_size, target_runner.sliding_window_size
    if (window is None or target_window is None or not 0 <= window <= target_window
            or any(a.sliding_window_size != window or a.is_cross_attention for a in attns)):
        return f"draft window {window} vs the target's {target_window}"
    return None


def build_pool(kvc, mha_pool_class):
    """The draft worker's pool on the target's SWA slots, or None (today's pool)."""
    global _target_runner, _active
    target_runner, _target_runner = _target_runner, None
    if target_runner is None or not kvc.is_draft_worker:
        return None
    reason = _decline_reason(kvc, target_runner)
    if reason is not None:
        logger.info("OKC U3 draft KV on SWA slots declined: %s", reason)
        return None
    from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel

    target, n = kvc.token_to_kv_pool_allocator.get_kvcache(), kvc.layer_info.num_effective_layers
    pool = U3DraftPool(
        size=target.size, size_swa=target.size_swa, page_size=1, dtype=kvc.kv_cache_dtype,
        head_num=kvc.model_config.get_num_kv_heads(get_parallel().attn_tp_size, get_parallel().attn_dcp_size),
        head_dim=kvc.model_config.head_dim, v_head_dim=kvc.model_config.v_head_dim,
        swa_attention_layer_ids=list(range(n)), full_attention_layer_ids=[], device=kvc.device,
        token_to_kv_pool_class=mha_pool_class, enable_memory_saver=get_exec().features.enable_memory_saver,
        enable_alt_stream=not get_disagg().enable_pdmux, enable_kv_cache_copy=True,
        post_capture_active=kvc.post_capture_kv_active)
    # The target allocator's tensor: stable (resize / clear refill it in place), its trailing -1 keeps -1 at -1.
    pool.register_mapping(target.full_to_swa_index_mapping)
    _active = True
    sub = pool.swa_kv_pool
    logger.info("OKC U3 draft KV on SWA slots: %d layers x %d SWA tokens (%d B per token), not %d full tokens",
                n, target.size_swa, (sub.row_dim + sub.v_row_dim) * sub.store_dtype.itemsize * n, target.size)
    return pool
