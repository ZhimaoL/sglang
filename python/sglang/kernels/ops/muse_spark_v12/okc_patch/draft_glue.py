"""OKC DFlash draft glue (serving/kernels/draft), bit-identical single launches (_cuda_src_draft.py)."""

import logging

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import draft_ext

logger = logging.getLogger(__name__)
_logged = set()


def prepare():
    draft_ext()


def cand_pack(vals, ids, offset):
    """torch.cat([vals.float().view(int32), ids.to(int32) + offset], -1); None = other dtypes."""
    if vals.dtype is not torch.bfloat16 or ids.dtype not in (torch.int32, torch.int64) or ids.shape != vals.shape:
        return None
    vals, ids = vals.contiguous(), ids.contiguous()
    out = torch.empty((vals.shape[0], 2 * vals.shape[1]), dtype=torch.int32, device=vals.device)
    draft_ext().cand_pack(vals, ids, out, int(offset))
    return out


def cand_merge(gathered, rows, world, k, mult, cap, ids_out=None):
    """gathered [world * rows, 2k] int32 -> (ids int64, unary fp32) [rows, k]; ids_out: optional contiguous int64
    buffer of rows * k the ids go to. torch's div_(cap) multiplies by float(1.0 / cap)."""
    if world != 8 or k != 16 or not gathered.is_contiguous():
        return None
    vals = torch.empty((rows, k), dtype=torch.float32, device=gathered.device)
    if ids_out is not None and ids_out.dtype is torch.int64 and ids_out.is_contiguous() and ids_out.numel() == rows * k:
        ids = ids_out.view(rows, k)
    else:
        ids = torch.empty((rows, k), dtype=torch.int64, device=gathered.device)
    draft_ext().cand_merge(gathered, rows, world, k, float(mult), 1.0 / float(cap), float(cap), vals, ids)
    if "cand" not in _logged:
        _logged.add("cand")
        logger.info("OKC DFlash candidate pack / merge active.")
    return ids, vals


def kv_commit_reason(pool, attns):
    """None if one launch reproduces pool.set_kv_buffer_prefix_valid (unscaled k.to(fp8) path) for these layers."""
    from sglang.srt.mem_cache import memory_pool as mp

    if type(pool).set_kv_buffer_prefix_valid is not mp.MHATokenToKVPool.set_kv_buffer_prefix_valid:
        return f"pool {type(pool).__name__}"
    if pool.dtype is not torch.float8_e4m3fn or pool.store_dtype is not torch.uint8 or pool.v_row_dim != pool.row_dim:
        return "pool dtype / rows"
    for attn in attns:
        if attn.k_scale is not None or attn.v_scale is not None:
            return "k / v scales"
        lid = attn.layer_id - pool.start_layer
        for buf in (pool.k_buffer[lid], pool.v_buffer[lid]):
            if buf.dim() != 3 or not buf.is_contiguous() or buf.shape[1] * buf.shape[2] != pool.row_dim:
                return f"buffer {tuple(buf.shape)}"
    return None


def kv_commit_for(pool, attns):
    from sglang.kernels.ops.muse_spark_v12 import okc_patch

    reason = "switch off" if not okc_patch.SW_DRAFT_KV_FUSE else kv_commit_reason(pool, attns)
    if reason is None:
        return KVCommit(pool, attns)
    logger.info("OKC DFlash fused draft-KV commit declined: %s", reason)
    return None


class KVCommit:
    """cache_k / cache_v [layers, rows, heads, dim] bf16 -> every layer's prefix-valid rows, cast as k.to(fp8)."""

    def __init__(self, pool, attns):
        ptrs = []
        for attn in attns:
            lid = attn.layer_id - pool.start_layer
            ptrs += [pool.k_buffer[lid].data_ptr(), pool.v_buffer[lid].data_ptr()]
        self.bufs = torch.tensor(ptrs, dtype=torch.int64, device=pool.k_buffer[0].device)
        self.row_dim = pool.row_dim
        draft_ext()
        logger.info("OKC DFlash fused draft-KV commit active (%d layers).", len(attns))

    def __call__(self, cache_k, cache_v, loc_2d, commit_lens):
        shape = (cache_k.shape[0], cache_k.shape[1], self.row_dim)
        draft_ext().kv_write(cache_k.reshape(shape), cache_v.reshape(shape), self.bufs,
                             loc_2d.reshape(-1).to(torch.int64).contiguous(),
                             commit_lens.to(torch.int32).contiguous(), int(loc_2d.shape[1]))
