"""OKC attention dispatch for MuseSpark1xAttention.gated_attention.

Two env-gated paths, both consuming the FlashAttention backend's per-step metadata
(page tables, cache_seqlens, SWA write locations) and the paged KV pool directly:

  OKC_SW_ATTN  sliding layers at q_len 8 (TP8 heads): attn-v3's prep + split-with-tail
               verify, replacing fused QK norm + RoPE + KV store + FA3 + gated output norm.
  OKC_SW_WST   full layers at q_len 8, bs >= OKC_WST_MIN_BS: the attention lane's verify-full kernels (wst_ct14 with the
               row-sliced in-kernel merge, wst_ct14b balanced over the batch from OKC_WST_BAL_MIN_BS up), one
               launch per layer for QK norm + new KV rows + FP8 attention + merge + output norm and gate.

Every eligibility miss returns None and the caller falls through to the stock path.
Decisions depend only on config, forward mode, and batch shape, so a decision taken
during CUDA-graph capture holds for every replay of that graph.

Operating points (kernel court, H200): sliding v1/v16/v32 1.25/1.21/1.14x per call. Only
TARGET_VERIFY dispatches (the target model never decodes under DFLASH); decode steps and
every other shape run the stock path.
"""

import logging
import math
from typing import Optional

import torch
import triton

logger = logging.getLogger(__name__)
_FIRED = set()


def log_once(path: str, detail: str = "") -> None:
    """One INFO line the first time each OKC path dispatches, so a serving check's
    stdout proves which paths actually fired (PLAN.md assumption 3)."""
    if path not in _FIRED:
        _FIRED.add(path)
        logger.info("OKC path active: %s %s", path, detail)

from sglang.kernels.ops.muse_spark_v12.okc_patch import (
    SW_ATTN,
    SW_ATTN_ACC16,
    SW_ATTN_V3,
    SW_PDL,
    SW_VFG,
    SW_VFG_FULL1,
    SW_WST,
    WST_ACC16,
    WST_BAL_MIN_BS,
    WST_MIN_BS,
)
from sglang.kernels.ops.muse_spark_v12.okc_patch import attn_kernels as ak

LOG2E = 1.4426950408889634

_FA_CLS = None


def _fa_backend(backend):
    global _FA_CLS
    if _FA_CLS is None:
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        _FA_CLS = FlashAttentionBackend
    return isinstance(backend, _FA_CLS)


def _plain_rope_theta(attn) -> Optional[float]:
    """The layer's RoPE base when it is the plain interleaved rotary the kernels
    implement; None when the layer has no RoPE; -1.0 to decline (scaled variants)."""
    rope = attn.rotary_emb
    if rope is None:
        return None
    if type(rope).__name__ != "RotaryEmbedding" or rope.is_neox_style:
        return -1.0
    return float(rope.base)


def _step_ctx(forward_batch) -> dict:
    ctx = forward_batch.okc_fs_meta
    if ctx is None:
        ctx = {}
        forward_batch.okc_fs_meta = ctx
    return ctx


def _seq_pre(ctx: dict, cache_seqlens, q_len: int):
    """Per-request cached-position count EXCLUDING the block's own q_len tokens (the
    court kernels' seq_lens convention); built once per step."""
    key = ("seq_pre", cache_seqlens.data_ptr(), q_len)
    seq = ctx.get(key)
    if seq is None:
        seq = (cache_seqlens - q_len).contiguous()
        ctx[key] = seq
    return seq


def _layer_eps(attn) -> Optional[float]:
    """The layer's norm eps when the kernels' fused contract holds for it, else None."""
    # The kernels fuse per-head QK RMSNorm, the output RMSNorm and the sigmoid gate;
    # a layer without any of the three computes something else.
    if not (attn.use_qk_norm and attn.use_o_norm and attn.use_o_gate):
        return None
    # Both norms take the layer's norm_eps (FusedQKNorm and the output RMSNorm are built
    # with it); the kernels apply that same EPS to q, k and the output.
    eps = getattr(attn, "norm_eps", None)
    if eps is None or attn.attn.logit_cap:
        return None
    return float(eps)


def _pool_contract(attn, qkv, k_cache):
    """(hq, hkv, d, is_fp8, kv_scale) when qkv and the layer's pool fit the kernels, else None."""
    hq, hkv, d = attn.num_heads, attn.num_kv_heads, attn.head_dim
    if hq % hkv or d & (d - 1) or qkv.shape[1] != (2 * hq + 2 * hkv) * d:
        return None
    if qkv.dtype is not torch.bfloat16 or not qkv.is_contiguous():
        return None
    if k_cache.dtype not in (torch.float8_e4m3fn, torch.bfloat16):
        return None
    if k_cache.dim() != 3 or k_cache.shape[1] != hkv or k_cache.shape[2] != d:
        return None
    is_fp8 = k_cache.dtype is torch.float8_e4m3fn
    k_scale = attn.attn.k_scale_float
    v_scale = attn.attn.v_scale_float
    if is_fp8:
        kv_scale = 1.0 if k_scale is None else float(k_scale)
        if (1.0 if v_scale is None else float(v_scale)) != kv_scale:
            return None
    else:
        if k_scale is not None and float(k_scale) != 1.0:
            return None
        kv_scale = 1.0
    return hq, hkv, d, is_fp8, kv_scale


def try_attention(attn, qkv, forward_batch) -> Optional[torch.Tensor]:
    """Dispatch gated_attention to an OKC path; None = stock path."""
    if torch.compiler.is_compiling():
        return None
    # Verify blocks only: under DFLASH the target model runs TARGET_VERIFY and EXTEND, never DECODE.
    if not forward_batch.forward_mode.is_target_verify():
        return None
    eps = _layer_eps(attn)
    if eps is None:
        return None
    radix = attn.attn
    if forward_batch.dcp_kv_mask is not None:
        return None
    from sglang.srt.layers.cp.utils import is_cp_active

    if is_cp_active(forward_batch):
        return None
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )

    backend = get_attn_backend()
    if not _fa_backend(backend) or backend.use_mla:
        return None
    # Cascade (spec topk > 1) and prefill-aware SWA rewrite the tables these kernels
    # read; DFLASH is chain mode, so both stay off in the judge config.
    if forward_batch.spec_info is not None and backend.topk > 1:
        return None
    meta = backend.forward_metadata
    if meta is None or meta.cache_seqlens_int32 is None:
        return None
    if backend.is_prefill_aware_swa and meta.pa_swa_page_table is not None:
        return None
    tokens = qkv.shape[0]
    bs = forward_batch.batch_size
    q_len = meta.max_seq_len_q
    if not q_len or q_len > 8 or tokens != bs * q_len:
        return None

    pool = get_token_to_kv_pool()
    k_cache = pool.get_key_buffer(attn.layer_idx)
    v_cache = pool.get_value_buffer(attn.layer_idx)
    contract = _pool_contract(attn, qkv, k_cache)
    if contract is None:
        return None
    hq, hkv, d, is_fp8, kv_scale = contract

    window = radix.sliding_window_size
    is_swa_layer = window is not None and window > -1
    if is_swa_layer and backend.use_sliding_window_kv_pool:
        cache_loc = meta.swa_out_cache_loc
        page_table = meta.swa_page_table
    else:
        cache_loc = forward_batch.out_cache_loc
        page_table = meta.page_table
    if cache_loc is None or page_table is None:
        return None
    if cache_loc.numel() != tokens:
        return None

    theta = _plain_rope_theta(attn)
    if theta is not None and theta < 0:
        return None
    ctx = _step_ctx(forward_batch)
    seq_pre = _seq_pre(ctx, meta.cache_seqlens_int32, q_len)
    sm_scale = d**-0.5
    qk_factor = attn.scale_query_by

    if is_swa_layer:
        if not SW_ATTN or q_len > 8 or theta is None:
            return None
        return _sliding(
            qkv, k_cache, v_cache, cache_loc, seq_pre, page_table,
            bs=bs, q_len=q_len, hq=hq, hkv=hkv, d=d,
            window=window + 1, page_size=backend.page_size,
            kv_scale=kv_scale, theta=theta, sm_scale=sm_scale, qk_factor=qk_factor, eps=eps,
        )
    if theta is not None or backend.page_size != 1:
        return None
    return _full_rows(
        qkv, k_cache, v_cache, cache_loc, seq_pre, page_table,
        bs=bs, q_len=q_len, hq=hq, hkv=hkv, d=d, is_fp8=is_fp8,
        kv_scale=kv_scale, sm_scale=sm_scale, qk_factor=qk_factor, eps=eps,
    )


def verify_full_rows(attn, qkv, k_cache, v_cache, cache_loc, seq_pre, page_table, *,
                     bs, q_len=8) -> Optional[torch.Tensor]:
    """A full-attention layer's verify blocks on explicit rows (ride-along riders inside an EXTEND forward): the
    same eligibility and routing as a TARGET_VERIFY step in try_attention (wst from OKC_WST_MIN_BS).

    qkv: the blocks' pre-norm projection rows [bs * q_len, (2 hq + 2 hkv) d] (row slices of the forward's qkv are
    fine); k_cache / v_cache: the layer's pool [rows, hkv, d] (page size 1); cache_loc: the blocks' reserved slots
    [bs * q_len] (their new k / v rows are (re)written there); seq_pre: int32 [bs] committed lengths (cache_seqlens -
    q_len); page_table: int32 rows >= bs (any width, any pitch), positions [0, seq_pre + q_len) of request b in row b.
    Returns the gated output
    [bs * q_len, hq * d] bf16, or None when not handled (the caller keeps its own path)."""
    if torch.compiler.is_compiling() or q_len != 8:
        return None
    eps = _layer_eps(attn)
    if eps is None or attn.rotary_emb is not None:
        return None
    window = attn.attn.sliding_window_size
    if window is not None and window > -1:
        return None
    contract = _pool_contract(attn, qkv, k_cache)
    if contract is None:
        return None
    hq, hkv, d, is_fp8, kv_scale = contract
    if qkv.shape[0] != bs * q_len or cache_loc.numel() != bs * q_len:
        return None
    if seq_pre.dtype is not torch.int32 or seq_pre.numel() != bs or page_table.shape[0] < bs:
        return None
    return _full_rows(
        qkv, k_cache, v_cache, cache_loc, seq_pre.contiguous(), page_table,
        bs=bs, q_len=q_len, hq=hq, hkv=hkv, d=d, is_fp8=is_fp8,
        kv_scale=kv_scale, sm_scale=d**-0.5, qk_factor=attn.scale_query_by, eps=eps,
    )


def _full_rows(qkv, k_cache, v_cache, cache_loc, seq_pre, page_table, *,
               bs, q_len, hq, hkv, d, is_fp8, kv_scale, sm_scale, qk_factor, eps):
    """Full-layer routing (decode and verify); None = stock path."""
    if (q_len == 8 and bs == 1 and SW_VFG_FULL1 and is_fp8 and hq == 16 and hkv == 2 and d == 64
            and page_table.dtype is torch.int32 and page_table.stride(1) == 1):
        log_once("full_vfg[q_len=8]", "(bs=1)")
        return _vfg_attn(
            qkv, k_cache, v_cache, cache_loc, seq_pre, page_table,
            bs=bs, window=0, kv_scale=kv_scale, theta=None, sm_scale=sm_scale, qk_factor=qk_factor, eps=eps,
        )
    # The wst kernels hard-code the TP8 head layout (16 q / 2 kv heads of 64) and read e4m3 cache bytes.
    if (q_len == 8 and bs >= max(1, WST_MIN_BS) and SW_WST and is_fp8 and hq == 16 and hkv == 2 and d == 64
            and page_table.dtype is torch.int32 and page_table.stride(1) == 1):
        log_once(f"wst_full[q_len={q_len}]", f"(first bs={bs})")
        return _wst_full(
            qkv, k_cache, v_cache, cache_loc, seq_pre, page_table,
            bs=bs, q_len=q_len, kv_scale=kv_scale, sm_scale=sm_scale, qk_factor=qk_factor, eps=eps,
        )
    return None


_CNT = {}


def _counter(kind, n, S, device):
    """Persistent int32 ticket counters (attn-v3): allocated zero once per (kind,
    owners, S). Every sliding call takes exactly S tickets per owner and the last
    arriver (ticket S - 1) resets its counter to zero in-kernel, so the count is back
    at zero after every call, eager or graph replay (capture executes nothing), and
    never wraps. Keyed per (n, S), so each captured decode graph replays against its
    own counter buffer."""
    key = (kind, n, S, str(device))
    c = _CNT.get(key)
    if c is None:
        c = _CNT[key] = torch.zeros((n,), dtype=torch.int32, device=device)
    return c


# attn-v3 verify launch knobs per (q_len, bs), bench_v3 sweeps; unlisted bs take the
# lane defaults (BN 128 / 4 warps / sc 4).
_V3_TUNE_BY = {
    (8, 1): (64, 4, 4),
    (8, 16): (64, 4, 4),
    (8, 32): (64, 4, 4),
}


def _sliding(qkv, k_cache, v_cache, cache_loc, seq_pre, page_table, *,
             bs, q_len, hq, hkv, d, window, page_size, kv_scale, theta,
             sm_scale, qk_factor, eps):
    """Sliding dispatch: verify blocks on the bank's VFG / VP2 kernels (FP8 pool, TP8 shape, page size 1), else the
    attn-v3 fused verify kernels on the TP8 shape, else the stock path."""
    if (SW_VFG and q_len == 8 and hq == 16 and hkv == 2 and d == 64 and page_size == 1 and 0 < window <= 2048
            and k_cache.dtype is torch.float8_e4m3fn and page_table.dtype is torch.int32
            and page_table.stride(1) == 1):
        log_once("sliding_vfg[q_len=8]", f"(first bs={bs})")
        return _vfg_attn(
            qkv, k_cache, v_cache, cache_loc, seq_pre, page_table,
            bs=bs, window=window, kv_scale=kv_scale, theta=theta,
            sm_scale=sm_scale, qk_factor=qk_factor, eps=eps,
        )
    if SW_ATTN_V3 and hq == 16 and hkv == 2 and d == 64 and q_len == 8:
        log_once(f"sliding_v3[q_len={q_len}]", f"(first bs={bs})")
        return _sliding_v3(
            qkv, k_cache, v_cache, cache_loc, seq_pre, page_table,
            bs=bs, q_len=q_len, hq=hq, hkv=hkv, d=d,
            window=window, page_size=page_size, kv_scale=kv_scale, theta=theta,
            sm_scale=sm_scale, qk_factor=qk_factor, eps=eps,
        )
    return None   # no v1 fallback: the stock path


def _sliding_v3(qkv, k_cache, v_cache, cache_loc, seq_pre, page_table, *,
                bs, q_len, hq, hkv, d, window, page_size, kv_scale, theta,
                sm_scale, qk_factor, eps):
    """attn-v3 fused sliding verify (q_len 8): prep + split-with-tail (_sl_verify_f2); the
    last-arriver program folds the combine, output RMSNorm and sigmoid gate into the same
    launch. Split-KV partials are fp32 unless OKC_SW_ATTN_ACC16=1 (bf16 partials, the
    combine still in fp32)."""
    device = qkv.device
    acc_dtype = torch.bfloat16 if SW_ATTN_ACC16 else torch.float32
    B, L = bs, q_len
    group = hq // hkv
    kc = k_cache.view(k_cache.shape[0], -1)
    vc = v_cache.view(v_cache.shape[0], -1)
    out = torch.empty((B * L, hq * d), dtype=torch.bfloat16, device=device)
    theta_coef = (-2.0 * math.log2(theta) / d) if theta else 0.0
    alpha = sm_scale * kv_scale * LOG2E
    gate_col0 = (hq + 2 * hkv) * d
    # prep (q norm+RoPE -> ws, own k/v -> cache) + split-with-tail.
    BN, warps, sc = _V3_TUNE_BY.get((8, B), (128, 4, 4))
    span = window - 1 + L
    S = max(1, min(math.ceil(span / BN), math.ceil(256 / (B * hkv)), 16))
    chunk = BN * math.ceil(span / (S * BN))
    LG = L * group
    P = B * hkv * S
    q_ws = torch.empty((B * hkv * LG, d), dtype=torch.bfloat16, device=device)
    ws_acc = torch.empty((P * LG, d), dtype=acc_dtype, device=device)
    ws_m = torch.empty((P * LG,), dtype=torch.float32, device=device)
    ws_l = torch.empty((P * LG,), dtype=torch.float32, device=device)
    counter = _counter("sv", B * hkv, S, device)
    br = triton.next_power_of_2(hq + 2 * hkv)
    ak._sliding_prep[(B * L,)](
        qkv, q_ws, kc, vc,
        cache_loc, seq_pre,
        qkv.stride(0),
        1.0 / kv_scale, 1.0 / kv_scale, qk_factor, theta_coef,
        EPS=eps,
        L=L, GROUP=group, HQ=hq, HKV=hkv,
        GROUP_Q=group, HKV_Q=hkv,
        HAS_ROPE=bool(theta), D=d, BLOCK_R=br, PDL=SW_PDL,
        num_warps=4 if br <= 64 else 8, num_stages=1, launch_pdl=SW_PDL,
    )
    ak._sl_verify_f2[(P,)](
        q_ws, kc, vc,
        seq_pre, page_table,
        ws_acc, ws_m, ws_l, counter, qkv, out,
        page_table.stride(0), qkv.stride(0), hq * d, gate_col0,
        page_table.shape[1] * page_size - L,
        alpha, kv_scale,
        EPS=eps,
        L=L, GROUP=group, HKV=hkv,
        WINDOW=window, PAGE=page_size,
        S=S, CHUNK=chunk, SC=sc, ACC16=SW_ATTN_ACC16,
        D=d,
        BLOCK_M=max(16, triton.next_power_of_2(LG)),
        BLOCK_N=BN, PDL=SW_PDL,
        num_warps=warps, num_stages=3, launch_pdl=SW_PDL,
    )
    return out


_WST_WS = {}
_SMS = {}
def verify_core_rows(q, k_cache, v_cache, page_table, cache_seqlens, out, *, bs, q_len=8, sm_scale, kv_scale) -> bool:
    """The attention core of a full layer's verify blocks on the wst kernels, for callers that run QK norm, the KV store
    and the output norm / gate themselves (ride-along riders inside the FA3 backend's eager attention break).

    q: [bs * q_len, 16, 64] float8_e4m3fn, the blocks' rows in order, QK-normed (full layers: no RoPE), not multiplied by
    the layer's query factor; k_cache / v_cache: the layer's e4m3 pool [rows, 2, 64] (page size 1) already holding the
    blocks' new rows; page_table: int32, >= bs rows, stride(1) == 1, any pitch; cache_seqlens: int32 [bs] including the
    q_len block tokens; out: bf16, bs * q_len rows of 16 x 64 (contiguous), receives the pre-norm attention output (what
    FA3 returns); sm_scale: the layer's softmax scale (d^-0.5 x scale_query_by); kv_scale: the pool's scalar scale.
    Same routes as _wst_full (wst_ct14 below OKC_WST_BAL_MIN_BS, wst_ct14b from it; cooperative launches), with their
    own workspace and tickets. Returns False (nothing launched) when not handled."""
    if torch.compiler.is_compiling() or not SW_WST or q_len != 8:
        return False
    if q.dtype is not torch.float8_e4m3fn or q.dim() != 3 or tuple(q.shape) != (bs * q_len, 16, 64) or not q.is_contiguous():
        return False
    if k_cache.dtype is not torch.float8_e4m3fn or k_cache.dim() != 3 or k_cache.shape[1:] != (2, 64):
        return False
    if page_table.dtype is not torch.int32 or page_table.stride(1) != 1 or page_table.shape[0] < bs:
        return False
    if cache_seqlens.dtype is not torch.int32 or cache_seqlens.numel() != bs:
        return False
    if out.dtype is not torch.bfloat16 or out.numel() != bs * q_len * 16 * 64 or not out.is_contiguous():
        return False
    from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import wst_ext

    device = q.device
    dkey = str(device)
    G = _SMS.get(dkey)
    if G is None:
        G = _SMS[dkey] = torch.cuda.get_device_properties(device).multi_processor_count
    B, L = bs, q_len
    LG = L * 8
    bal = B >= WST_BAL_MIN_BS and G <= 256
    S = 0 if bal else G // B
    nrow = (G + B) * 2 * LG if bal else B * 2 * S * LG
    ws = _WST_WS.get(("core_ws", dkey))
    if ws is None or ws["acc"].shape[0] < nrow:
        n = max(nrow, (G + 32) * 2 * LG)
        ws = {
            "acc": torch.empty((n, 64), dtype=torch.bfloat16 if WST_ACC16 else torch.float32, device=device),
            "m": torch.empty((n,), dtype=torch.float32, device=device),
            "l": torch.empty((n,), dtype=torch.float32, device=device),
        }
        _WST_WS.setdefault(("core_all", dkey), []).append(ws)
        _WST_WS[("core_ws", dkey)] = ws
    key = ("core", bal, B, L, S, dkey)
    tick = _WST_WS.get(key)
    if tick is None:   # per (b, kv head), +256 per launch
        tick = _WST_WS[key] = torch.zeros((2 * B,), dtype=torch.int32, device=device)
    skey = ("scale", float(kv_scale), dkey)
    vs = _WST_WS.get(skey)
    if vs is None:
        vs = _WST_WS[skey] = torch.tensor([float(kv_scale)], dtype=torch.float32, device=device)
    seq_pre = (cache_seqlens - L).to(torch.int32)
    ext = wst_ext()
    smqk = sm_scale * LOG2E
    q8, kc, vc = q.view(torch.uint8), k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    if bal:
        ext.wst_core_bal(q8, kc, vc, page_table, seq_pre, ws["acc"], ws["m"], ws["l"], vs, smqk, B, L,
                         page_table.stride(0), G, out, tick, vs)
    else:
        ext.wst_core_fqbt(q8, kc, vc, page_table, seq_pre, ws["acc"], ws["m"], ws["l"], vs, smqk, B, L,
                          page_table.stride(0), S, out, tick, vs)
    log_once("wst_core[q_len=8]", f"(first bs={bs})")
    return True


_VFG_WS = {}


def _vfg_attn(qkv, k_cache, v_cache, cache_loc, seq_pre, page_table, *,
              bs, window, kv_scale, theta, sm_scale, qk_factor, eps):
    """Verify blocks (q_len 8) on the attention lane's bank v16c_m7g_rlx3 routes (`_vph` / `_vfg`):
      sliding, bs 1:  VP2 (two 16-CTA clusters, one per kv head; window slices merged in-cluster, no second kernel)
      sliding, bs > 1 and full, bs 1: VFG, SMs // bs splits per request + the vfgm (>= 32 splits) / vfgm2 merge kernel
    All of them build q (RMSNorm, RoPE from the fp32 inv_freq table on sliding layers, e4m3) and the block's new k / v
    cache rows in-kernel and write RMSNorm(O) x sigmoid(gate). Split-KV partials are fp32 (global records, VP2 owner
    blocks). qkv rows must be the contiguous [bs * 8, 2304] projection (the kernels' row stride is the TP8 width).
    No ticket counters: every launch is self-contained, so graph capture and replays need no state."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import vfg_ext

    device = qkv.device
    dkey = str(device)
    G = _SMS.get(dkey)
    if G is None:
        G = _SMS[dkey] = torch.cuda.get_device_properties(device).multi_processor_count
    if cache_loc.dtype != torch.int64:
        cache_loc = cache_loc.to(torch.int64)
    freq = 0
    if window:
        fkey = ("freq", float(theta), dkey)
        f = _VFG_WS.get(fkey)
        if f is None:   # exactly the reference's fp32 inv_freq table
            f = _VFG_WS[fkey] = (1.0 / (float(theta) ** (torch.arange(0, 64, 2, device=device).float() / 64))).contiguous()
        freq = f.data_ptr()
    out = torch.empty((bs * 8, 16 * 64), dtype=torch.bfloat16, device=device)
    c2 = qk_factor * sm_scale * kv_scale * LOG2E
    stream = torch.cuda.current_stream().cuda_stream
    ext = vfg_ext()
    if window and bs == 1:
        ext.muse_vp2_launch(qkv.data_ptr(), page_table.data_ptr(), seq_pre.data_ptr(), cache_loc.data_ptr(),
                            k_cache.data_ptr(), v_cache.data_ptr(), freq, out.data_ptr(), window, c2,
                            1.0 / kv_scale, 1.0 / kv_scale, kv_scale, eps, 0, stream)
        return out
    ns = max(1, G // bs)
    wkey = ("ws", dkey)
    ws = _VFG_WS.get(wkey)
    if ws is None:   # bs * ns <= SMs partial rows of 128 (8 tokens x 16 heads) x 72 floats; shared by every bs / graph
        ws = _VFG_WS[wkey] = torch.empty((G * 128 * 72,), dtype=torch.float32, device=device)
    ext.muse_vfg_launch(qkv.data_ptr(), page_table.data_ptr(), page_table.stride(0), seq_pre.data_ptr(),
                        cache_loc.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), freq, out.data_ptr(), bs, ns,
                        window, c2, 1.0 / kv_scale, 1.0 / kv_scale, kv_scale, eps, ws.data_ptr(), 0, stream)
    return out


def _wst_full(qkv, k_cache, v_cache, cache_loc, seq_pre, page_table, *,
              bs, q_len, kv_scale, sm_scale, qk_factor, eps):
    """Full-layer verify on the attention lane's wst kernels (bank v16c_m7g_rlx3 `_full_wst`, wst_cfg 54, no RoPE):
    every CTA builds its e4m3 q fragments from qkv (RMSNorm), the split holding the block's new positions writes their
    k (RMSNorm) / v rows to the cache before staging them, and the (b, kv head) partials (O fp32, bf16 with
    OKC_WST_ACC16=1; m / l fp32) are folded in-kernel with RMSNorm x sigmoid(gate). Splits come from the device-side lengths only (never the table width).

      bs < WST_BAL_MIN_BS: wst_ct14, grid (SMs // bs, bs), uniform split of the batch's longest span, row-sliced
                           K-waiter merge (cooperative launch: SMs // bs * bs <= SMs CTAs at 1 CTA / SM).
      otherwise          : wst_ct14b, SM-count CTAs (cooperative launch), all tiles of the batch balanced over them
                           (default from bs 5: equal at 4, ahead from 6, robust to spread lengths).

    Ticket counters persist per (route, bs): every launch moves each counter by exactly 256 (both routes), so they
    are graph-safe without a reset and wrap harmlessly. The partial workspace (5.4 MB fp32) is shared.
    PDL: both launches are plain (no programmatic serialization, no early trigger), so every earlier kernel on the
    stream has finished before the first CTA starts, and a PDL-launched successor (the o projection) leaves its
    griddepcontrol.wait when the whole grid is done."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import wst_ext

    device = qkv.device
    B, L = bs, q_len
    LG = L * 8
    dkey = str(device)
    G = _SMS.get(dkey)
    if G is None:
        G = _SMS[dkey] = torch.cuda.get_device_properties(device).multi_processor_count
    bal = B >= WST_BAL_MIN_BS and G <= 256
    S = 0 if bal else G // B
    # Partial slots: bal (c + b, kv head), fqbt (b, kv head, split) with S * B <= G. One buffer serves every bs and
    # graph (all launches are stream-ordered); it is sized for bs 32 up front and never freed, since captured graphs
    # keep its address.
    nrow = (G + B) * 2 * LG if bal else B * 2 * S * LG
    ws = _WST_WS.get(("ws", dkey))
    if ws is None or ws["acc"].shape[0] < nrow:
        n = max(nrow, (G + 32) * 2 * LG)
        ws = {
            "acc": torch.empty((n, 64), dtype=torch.bfloat16 if WST_ACC16 else torch.float32, device=device),
            "m": torch.empty((n,), dtype=torch.float32, device=device),
            "l": torch.empty((n,), dtype=torch.float32, device=device),
        }
        _WST_WS.setdefault(("all", dkey), []).append(ws)
        _WST_WS[("ws", dkey)] = ws
    key = (bal, B, L, S, dkey)
    tick = _WST_WS.get(key)
    if tick is None:   # per (b, kv head), +256 per launch
        tick = _WST_WS[key] = torch.zeros((2 * B,), dtype=torch.int32, device=device)
    skey = ("scale", float(kv_scale), dkey)
    vs = _WST_WS.get(skey)
    if vs is None:
        vs = _WST_WS[skey] = torch.tensor([float(kv_scale)], dtype=torch.float32, device=device)
    if cache_loc.dtype != torch.int64:
        cache_loc = cache_loc.to(torch.int64)
    out = torch.empty((B * L, 16 * 64), dtype=torch.bfloat16, device=device)
    smqk = sm_scale * qk_factor * LOG2E
    ext = wst_ext()
    kc, vc = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    pt_pitch = page_table.stride(0)
    if bal:
        ext.wst_attn_bal(qkv, kc, vc, page_table, seq_pre, cache_loc, ws["acc"], ws["m"], ws["l"], vs, smqk,
                         B, L, pt_pitch, G, 0, 3, 1.0 / kv_scale, 1.0 / kv_scale, eps, out, tick, vs)
    else:
        ext.wst_attn_fqbt(qkv, kc, vc, page_table, seq_pre, cache_loc, ws["acc"], ws["m"], ws["l"], vs, smqk,
                          B, L, pt_pitch, 0, S, 0, 1.0 / kv_scale, 1.0 / kv_scale, eps, out, tick, vs)
    return out

