"""OKC_QK_FOLD (FIRSTTURN #1, _cuda_src_ftnorm.py): a marked layer's extend hands q / k raw to the FA3 backend, which folds
the norm into _okc_fused_fp8_kv's launch or applies FusedQKNorm as the model did; both sides decide by applies().
q / k / v are used only within their own extents: the SP attention break's replay hands them over as weak-ref views
whose storage ends at their own last element (QKF: a [T, 2304] row view of q overran it by 1,280 elements).
"""

import logging

import torch

from sglang.kernels.ops.muse_spark_v12 import okc_patch as _sw

logger = logging.getLogger(__name__)
_LOGGED = set()


def _log_once(what, tokens):
    """one INFO line per process the first time the fold runs or declines, so an arm's server log shows which"""
    if what not in _LOGGED:
        _LOGGED.add(what)
        logger.info("OKC QK fold %s (first T=%d).", what, tokens)


class QKFoldSpec:
    """What the backend needs of a MuseSpark1xAttention (a plain object: never registered as a submodule)."""

    def __init__(self, attn):
        self.norm = attn._fused_qk_norm
        self.rope = attn.rotary_emb
        self.eps = float(attn.norm_eps)


def mark(attn):
    """At load: this layer's extends reach the FA3 backend with raw q / k (16 q + 2 kv heads of 64 per rank)."""
    if (_sw.QK_FOLD and attn._fused_qk_norm is not None and attn.head_dim == 64 and attn.num_heads == 16
            and attn.num_kv_heads == 2):
        object.__setattr__(attn.attn, "okc_qk_fold", QKFoldSpec(attn))


def applies(layer, forward_batch):
    """The spec when this extend's q / k arrive raw (model side: skip the norm; backend side: fold or apply it)."""
    spec = getattr(layer, "okc_qk_fold", None)
    if spec is None:
        return None
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.forward_context import get_attn_backend

    if forward_batch.forward_mode != ForwardMode.EXTEND:
        return None
    if type(get_attn_backend()).forward_extend is not FlashAttentionBackend.forward_extend:
        return None
    return spec


def _view(x, t, width):
    """x ([t, width] or [t, heads, 64]) as a [t, width] view with unit column stride and 16-byte rows, or None"""
    if x.shape[0] != t or x.numel() != t * width or x.dtype is not torch.bfloat16:
        return None
    x2 = x.reshape(t, width)
    if x2.data_ptr() != x.data_ptr() or x2.stride(1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
        return None
    return x2


def _qk_rows(q, k):
    """FusedQKNorm's input, the [T, 1152] q / k rows: the projection output itself when q's storage holds k right after
    q in every row, else a copy (the same values, so the same output)"""
    t = q.shape[0]
    q2, k2 = q.reshape(t, 1024), k.reshape(t, 128)
    ld, es = q2.stride(0), q2.element_size()
    if (t and q2.stride(1) == 1 and k2.stride(1) == 1 and (t == 1 or k2.stride(0) == ld)
            and k2.data_ptr() == q2.data_ptr() + 1024 * es
            and (q2.storage_offset() + (t - 1) * ld + 1152) * es <= q2.untyped_storage().nbytes()):
        return q2.as_strided((t, 1152), (ld, 1))
    return torch.cat((q2, k2), 1)


def _apply_norm(spec, q, k, positions):
    """FusedQKNorm on the q / k projection rows, as the model calls it (no KV store on an extend), in q / k's shapes."""
    qk = _qk_rows(q, k)
    if spec.rope is not None:
        qn, kn = spec.norm(qk, positions, spec.rope.cos_sin_cache)
    else:
        qn, kn = spec.norm(qk)
    return qn.view(q.shape), kn.view(k.shape)


def extend(backend, q, k, v, layer, forward_batch, save_kv_cache, cp_active):
    """forward_extend's entry: (q, k, save_kv_cache); save_kv_cache False = the fold wrote the KV cache and q."""
    spec = applies(layer, forward_batch)
    if spec is None:
        return q, k, save_kv_cache
    if save_kv_cache and k is not None and not backend.fa_skip_kv_cache and not backend.use_mla and not cp_active:
        out = backend._okc_fused_fp8_kv(q, k, v, layer, forward_batch, True, qk_fold=spec)
        if out is not None:
            _log_once("active", q.shape[0])
            return out, k, False
    _log_once("declined, FusedQKNorm in the backend", q.shape[0])
    qn, kn = _apply_norm(spec, q, k, forward_batch.positions)
    return qn, kn, save_kv_cache


def _tree(tokens):
    """the Triton kernel's reduction layout for its BLOCK_M (FusedQKNorm._triton_config on 18 rows per token)"""
    n = tokens * 18
    return 8 if n >= 8192 else (4 if n >= 1024 else 2)


def fold(q, k, v, k_cache, v_cache, loc, quant_q, spec, forward_batch):
    """_okc_fused_fp8_kv's kernel call with the norm folded in: FP8 (or bf16) q, or None (the caller applies the norm)."""
    t = q.shape[0]
    q2, k2, v2 = _view(q, t, 1024), _view(k, t, 128), _view(v, t, 128)
    if q2 is None or k2 is None or v2 is None:
        return None
    rope = spec.rope is not None
    pos = forward_batch.positions if rope else None
    if rope and (pos is None or pos.dtype is not torch.int64 or not pos.is_contiguous() or pos.numel() < t):
        return None
    from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import ftnorm_ext

    q_out = torch.empty((t, 1024), dtype=torch.float8_e4m3fn if quant_q else torch.bfloat16, device=q.device)
    empty = torch.empty((0,), dtype=torch.int64, device=q.device)
    ftnorm_ext().qk_fold(q2, k2, v2, pos if rope else empty, spec.rope.cos_sin_cache if rope else empty.float(),
                         spec.eps, q_out, k_cache.view(-1, 128), v_cache.view(-1, 128), loc, _tree(t))
    return q_out.view(q.shape)
