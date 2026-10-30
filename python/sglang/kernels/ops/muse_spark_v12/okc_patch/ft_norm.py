"""OKC_QK_FOLD (FIRSTTURN #1, _cuda_src_ftnorm.py): a marked layer's extend hands q / k raw to the FA3 backend, which folds
the norm into _okc_fused_fp8_kv's launch or applies FusedQKNorm as the model did; both sides decide by applies().
"""

import torch

from sglang.kernels.ops.muse_spark_v12 import okc_patch as _sw


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


def _rows(q):
    """the projection output [T, row stride] that q ([T, 1024] or [T, 16, 64], a view of its first columns) starts"""
    q2 = q.reshape(q.shape[0], -1)
    if q2.data_ptr() != q.data_ptr() or q2.stride(1) != 1:
        return None
    return q2.as_strided((q2.shape[0], q2.stride(0)), (q2.stride(0), 1))


def _apply_norm(spec, q, k, positions):
    """FusedQKNorm on the projection rows, as the model calls it (no KV store on an extend), in q / k's shapes."""
    qkv = _rows(q)
    if qkv is None:
        raise RuntimeError("OKC_QK_FOLD: q is not a view of the QKV projection output")
    if spec.rope is not None:
        qn, kn = spec.norm(qkv, positions, spec.rope.cos_sin_cache)
    else:
        qn, kn = spec.norm(qkv)
    return qn.view(q.shape), kn.view(k.shape)


def extend(backend, q, k, v, layer, forward_batch, save_kv_cache, cp_active):
    """forward_extend's entry: (q, k, save_kv_cache); save_kv_cache False = the fold wrote the KV cache and q."""
    spec = applies(layer, forward_batch)
    if spec is None:
        return q, k, save_kv_cache
    if save_kv_cache and k is not None and not backend.fa_skip_kv_cache and not backend.use_mla and not cp_active:
        out = backend._okc_fused_fp8_kv(q, k, v, layer, forward_batch, True, qk_fold=spec)
        if out is not None:
            return out, k, False
    qn, kn = _apply_norm(spec, q, k, forward_batch.positions)
    return qn, kn, save_kv_cache


def _tree(tokens):
    """the Triton kernel's reduction layout for its BLOCK_M (FusedQKNorm._triton_config on 18 rows per token)"""
    n = tokens * 18
    return 8 if n >= 8192 else (4 if n >= 1024 else 2)


def fold(q, k_cache, v_cache, loc, quant_q, spec, forward_batch):
    """_okc_fused_fp8_kv's kernel call with the norm folded in: FP8 (or bf16) q, or None (the caller applies the norm)."""
    qkv = _rows(q)
    t = q.shape[0]
    if qkv is None or q.numel() != t * 1024 or qkv.stride(0) % 8 or q.data_ptr() % 16:
        return None
    ld = qkv.stride(0)
    rope = spec.rope is not None
    pos = forward_batch.positions if rope else None
    if rope and (pos is None or pos.dtype is not torch.int64 or not pos.is_contiguous() or pos.numel() < t):
        return None
    from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import ftnorm_ext

    q_out = torch.empty((t, 1024), dtype=torch.float8_e4m3fn if quant_q else torch.bfloat16, device=q.device)
    empty = torch.empty((0,), dtype=torch.int64, device=q.device)
    ftnorm_ext().qk_fold(qkv, ld, pos if rope else empty, spec.rope.cos_sin_cache if rope else empty.float(),
                         spec.eps, q_out, k_cache.view(-1, 128), v_cache.view(-1, 128), loc, _tree(t))
    return q_out.view(q.shape)
