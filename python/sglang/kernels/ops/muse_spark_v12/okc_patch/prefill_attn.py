"""OKC prefill attention: warp-specialized FP8 FMHA on full layers, run as an
eager graph break (ws_attn graft, v2 seat).

Why a break. Under the judge launch every prefill batch up to 16384 tokens replays
a BREAKABLE prefill CUDA graph that was captured with ONE synthetic request at the
bucket's token count. Whatever is captured inside a segment sees the batch size and
the token count as host constants. The v1 hook launched the ws kernels inside the
segment, so a replay whose real token count differed from the bucket (the 8624-token
tail chunk of a 74160-token first turn runs in the 8704 bucket) shifted the causal
limit, and a replay carrying several co-batched requests was treated as one sequence
reading page_table[0] (serving audit B1). The stock attention op is an eager_on_graph
break for exactly this reason; this module takes the same seat and replaces that
break for the layers it serves (no extra break per layer).

Replay semantics (mirrors radix_attention.unified_attention_with_output):
  * the break reads the LIVE ForwardBatch from the tc-piecewise context, narrows the
    static qkv / out buffers and out_cache_loc / positions to the real token count,
    then runs either the ws pipeline (one request, real tokens >= OKC_WS_MIN_QLEN) or
    the stock sequence (fused QK norm -> backend extend attention with KV store), and
    zeroes the padded tail of the output;
  * outside breakable graphs (eager EXTEND) the same body runs directly.
The model's FusedRMSNormGate (output norm + sigmoid gate) stays in the captured
segment on the bucket rows, like every other elementwise op there.

ws pipeline per call (replaces fused QK norm + KV store + FA3):
  1. _prefill_prep (Triton): per-head QK RMSNorm, k/v scaled into the fp8 cache rows
     at out_cache_loc, q -> fp8 [T, 16, 64].
  2. ws gather (CUDA): paged fp8 cache -> per-(kv head, 128-key block) 16KB stage
     images over the real context extent (zero-filled past len), V pi-permuted.
  3. ws fmha2 (CUDA, nst 35 = NST4/EXPM5 = certified): 2 consumer warpgroups,
     f16-QK/lazy-max/hybrid-exp2 softmax, fp8 PV; causal over the chunk tail.
Numerics: the certified recipe's rel-L2 is 1.3-1.7x FA3-fp8's own error (court
checker band). The ws kernel pair is fp8-only and 16q/2kv/64d-only by design.

Evidence (audit B7): "OKC path active" fires at capture too. OKC_PATH_STATS=1 logs the
cumulative replay/eager hit counters (ws vs stock fallback) every 256 ws hits.
"""

import logging
import os
from typing import Optional

import torch
import triton

from sglang.kernels.ops.muse_spark_v12.okc_patch import WS_MIN_QLEN
from sglang.kernels.ops.muse_spark_v12.okc_patch import attn_kernels as ak
from sglang.kernels.ops.muse_spark_v12.okc_patch.attention import (
    RMS_EPS,
    _fa_backend,
    log_once,
)
from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import ws_ext
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)

logger = logging.getLogger(__name__)

LOG2E = 1.4426950408889634
# ws_fmha kernel selector: 35 = ws_fmha2<NST=4, EXPM=5>, the certified config.
WS_NST = 35
PATH_STATS = os.environ.get("OKC_PATH_STATS", "0") == "1"

_SCALES = {}
# RadixAttention.layer_id -> MuseSpark1xAttention; the break only receives static
# tensors + the layer id (eager_on_graph keeps capture-time argument identity).
_LAYERS = {}
# ws = fused path ran; stock = stock sequence ran; declined = stock ran although the batch had >= WS_MIN_QLEN
# real tokens (ragged co-batch, cross-request page table, ...). OKC_PATH_STATS=1 logs at power-of-two counts.
HITS = {"ws": 0, "stock": 0, "declined": 0, "ws_tokens": 0}


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _scale_tensor(value: float, device) -> torch.Tensor:
    """fp32 [] device scalars for the CUDA launchers; cached, so graph-safe."""
    key = (value, str(device))
    t = _SCALES.get(key)
    if t is None:
        t = _SCALES[key] = torch.full((), value, dtype=torch.float32, device=device)
    return t


def _layer_ok(attn) -> bool:
    """Static per-layer eligibility: only module attributes and backend facts, so it
    is safe to evaluate at capture time (no batch facts)."""
    if not (attn.use_qk_norm and attn.use_o_norm and attn.use_o_gate):
        return False
    radix = attn.attn
    if radix.logit_cap:
        return False
    window = radix.sliding_window_size
    if window is not None and window > -1:
        return False  # sliding layers keep FA3 (ws is a full-attention kernel)
    if attn.rotary_emb is not None:
        return False  # muse full layers are NoPE; a roped full layer is a config drift
    if attn.norm_eps != RMS_EPS:
        return False
    if attn.num_heads != 16 or attn.num_kv_heads != 2 or attn.head_dim != 64:
        return False  # the ws kernel pair is specialized to the TP8 muse shape
    backend = get_attn_backend()
    if not _fa_backend(backend) or backend.use_mla:
        return False
    if backend.page_size != 1:
        return False  # extend metadata carries a token-level table only at page 1
    return True


def try_prefill_attention(attn, qkv, forward_batch) -> Optional[torch.Tensor]:
    """Dispatch an EXTEND gated_attention through the OKC break; None = stock path.

    Called inside the captured segment (or eagerly). Decides ONLY on static facts;
    the real batch is read inside the break at execution time."""
    if not _layer_ok(attn):
        return None
    if forward_batch.dcp_kv_mask is not None:
        return None
    from sglang.srt.layers.cp.utils import is_cp_active

    if is_cp_active(forward_batch):
        return None
    if forward_batch.spec_info is not None and get_attn_backend().topk > 1:
        return None
    hq, hkv, d = attn.num_heads, attn.num_kv_heads, attn.head_dim
    if qkv.dtype is not torch.bfloat16 or not qkv.is_contiguous():
        return None
    if qkv.shape[1] != (2 * hq + 2 * hkv) * d:
        return None
    pool = get_token_to_kv_pool()
    k_cache = pool.get_key_buffer(attn.layer_idx)
    if k_cache.dtype is not torch.float8_e4m3fn:
        return None  # ws reads raw e4m3 bytes
    if k_cache.dim() != 3 or k_cache.shape[1] != hkv or k_cache.shape[2] != d:
        return None
    layer_id = attn.attn.layer_id
    _LAYERS[layer_id] = attn
    tokens = qkv.shape[0]
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=qkv.device)
    if is_in_breakable_cuda_graph():
        # Capture: ends the segment, runs once, records the replay call that re-reads
        # the live batch from the context. Eager: the wrapper calls straight through.
        _breakable_attention(qkv, out, layer_id)
    else:
        _attention(qkv, out, layer_id, forward_batch)
    o_gate = qkv[:, (hq + 2 * hkv) * d:]
    return attn._fused_o_norm_gate(out, o_gate)


def _attention_break(qkv: torch.Tensor, out: torch.Tensor, layer_id: int) -> None:
    context = get_tc_piecewise_forward_context()
    _attention(qkv, out, layer_id, context.forward_batch)


_breakable_attention = eager_on_graph(True)(_attention_break)


def _attention(qkv, out, layer_id, forward_batch) -> None:
    """The break body: narrow to the real tokens, run ws or the stock sequence into
    `out[:real]`, zero the padded tail."""
    attn = _LAYERS[layer_id]
    total = out.shape[0]
    real = getattr(forward_batch, "global_num_token_non_padded_cpu", None)
    real = total if real is None else min(int(real), total)
    if real <= 0:
        out.zero_()
        return
    qkv_r = qkv[:real]
    out_r = out[:real]
    orig_loc = forward_batch.out_cache_loc
    orig_pos = forward_batch.positions
    if orig_loc is not None:
        forward_batch.out_cache_loc = orig_loc[:real]
    if orig_pos is not None:
        forward_batch.positions = orig_pos[:real]
    try:
        if _ws(attn, qkv_r, out_r, forward_batch):
            HITS["ws"] += 1
            HITS["ws_tokens"] += real
            if PATH_STATS and _pow2(HITS["ws"]):
                logger.info("OKC prefill_fmha hits: %s (layer %d, tokens %d)", HITS, layer_id, real)
        else:
            HITS["stock"] += 1
            if real >= WS_MIN_QLEN:
                HITS["declined"] += 1
                if PATH_STATS and _pow2(HITS["declined"]):
                    logger.info("OKC prefill_fmha declined long batch: %s (layer %d, tokens %d, total %d)", HITS, layer_id, real, total)
            _stock(attn, qkv_r, out_r, forward_batch)
    finally:
        forward_batch.out_cache_loc = orig_loc
        forward_batch.positions = orig_pos
    if real < total:
        out[real:].zero_()


def _ws(attn, qkv, out, forward_batch) -> bool:
    """ws pipeline on the real rows; False = decline (caller runs the stock sequence)."""
    tokens = qkv.shape[0]
    if tokens < WS_MIN_QLEN:
        return False
    backend = get_attn_backend()
    meta = backend.forward_metadata
    if meta is None or meta.cache_seqlens_int32 is None or meta.page_table is None:
        return False
    cu_q = meta.cu_seqlens_q
    # One request only: the kernels take a uniform q_len per sequence, and a ragged
    # co-batch (several 753-token later turns) is not that. Ragged -> stock FA3.
    if cu_q is None or cu_q.numel() != 2 or int(meta.max_seq_len_q) != tokens:
        return False
    lens = meta.cache_seqlens_int32
    page_table = meta.page_table
    if page_table.dtype is not torch.int32 or lens.numel() != 1:
        return False
    max_len = int(meta.max_seq_len_k)
    if max_len <= 0 or page_table.shape[1] < max_len:
        return False
    cache_loc = forward_batch.out_cache_loc
    if cache_loc is None or cache_loc.numel() != tokens:
        return False
    radix = attn.attn
    hq, hkv, d = attn.num_heads, attn.num_kv_heads, attn.head_dim
    pool = get_token_to_kv_pool()
    k_cache = pool.get_key_buffer(attn.layer_idx)
    v_cache = pool.get_value_buffer(attn.layer_idx)
    device = qkv.device
    k_scale = 1.0 if radix.k_scale_float is None else float(radix.k_scale_float)
    v_scale = 1.0 if radix.v_scale_float is None else float(radix.v_scale_float)
    kst = _scale_tensor(k_scale, device)
    vst = _scale_tensor(v_scale, device)
    bs = 1
    # Stage images over the real context extent (the break sees the live metadata).
    cap = (max_len + 255) & ~255
    capblks = cap // 128

    log_once("prefill_fmha", f"(first q_len={tokens}, max_len={max_len})")
    ext = ws_ext()
    kc = k_cache.view(k_cache.shape[0], -1)
    vc = v_cache.view(v_cache.shape[0], -1)

    q_fp8 = torch.empty((tokens, hq, d), dtype=torch.float8_e4m3fn, device=device)
    # ROPE=False, so pos_ptr and inv_freq_ptr are never read: pass stand-ins.
    ak._prefill_prep[(triton.cdiv(tokens, 8), hq + 2 * hkv)](
        qkv, q_fp8, kc, vc,
        cache_loc, cache_loc, kst, vst, vst,
        qkv.stride(0), tokens,
        HQ=hq, HKV=hkv, D=d, ROPE=False, EPS=RMS_EPS, BT=8,
        num_warps=2,
    )
    kvt = torch.empty((bs * 2 * capblks, 16384), dtype=torch.uint8, device=device)
    ext.gather(
        kc.view(torch.uint8), vc.view(torch.uint8), kvt,
        page_table, lens, page_table.stride(0),
        0, capblks, bs,
    )
    sm_scale = d**-0.5
    ext.fmha(
        q_fp8.view(torch.uint8), kvt, out.view(tokens, hq, d), lens, kst, vst,
        sm_scale * attn.scale_query_by * LOG2E,
        bs, tokens, capblks, WS_NST,
    )
    return True


def _stock(attn, qkv, out, forward_batch) -> None:
    """The stock gated_attention middle for a full layer on FA3: fused QK norm (no
    RoPE, no fused store on this backend) -> backend extend attention with the KV
    store, written into `out`. Same calls as MuseSpark1xAttention.gated_attention +
    radix_attention._unified_attention_with_output_impl, minus the nested break."""
    radix = attn.attn
    q_size, kv_size = attn.q_size, attn.kv_size
    v = qkv[:, q_size + kv_size : q_size + 2 * kv_size]
    q, k = attn._fused_qk_norm(qkv, kv_store=None)
    k = k.view(-1, radix.tp_k_head_num, radix.qk_head_dim)
    v = v.view(-1, radix.tp_v_head_num, radix.v_head_dim)
    forward_batch._attn_output = out
    ret = get_attn_backend().forward(q, k, v, radix, forward_batch, True)
    if isinstance(ret, tuple):
        ret = ret[0]
    if ret.data_ptr() != out.data_ptr():
        out.view(ret.shape).copy_(ret)
