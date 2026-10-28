"""OKC prefill attention: the prefill lane's FP8 FMHA (fmha7) in place of FA3's extend call.

Seat. FlashAttentionBackend.forward_extend calls try_extend right where it would call FA3 on the paged KV
cache: after the stock KV store and the stock FP8 cast of q, inside the eager attention break of the breakable
prefill CUDA graph (the break reads the live batch, so host shapes are the real ones). Everything around the
call stays stock (QK norm, KV store, output norm and gate); try_extend computes the same attention FA3 would
and returns None whenever it does not apply, so the backend calls FA3.

Applies to: Muse target layers (MuseSpark1xAttention marks its RadixAttention), forward mode EXTEND, one
request (uniform q_len, the 16K chunks of a first turn and single follow-up extends), q / KV e4m3, page size 1,
16 q heads / 2 kv heads / head dim 64 per rank, causal, no softcap / sinks / score mod, not under capture.
Full layers attend [0, len); sliding layers (OKC_PFMHA_SLIDING) the 2048-key window. Below
OKC_PFMHA_MIN_TOKENS real tokens FA3 keeps the call.

Kernel (okc_patch/_cuda_src_fmha.py): gather_kp copies the request's paged K/V rows into per (kv head,
128-key block) 16 KB stage images (zero-filled past the context), fmha7 streams them by TMA into 3 or 4
consumer warpgroups (f16 QK, lazy-max hybrid exp2, e4m3 P, fp8 PV, tensor-core row sums), split-KV + combine
when one request's grid under-fills the GPU (pick_cfg below = the bank's pick_cfg4).
"""

import logging
import os
from typing import Optional

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch import (
    PFMHA_MIN_TOKENS,
    PFMHA_SLIDING,
    PFMHA_SLIDING_MIN_TOKENS,
)
from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import fmha_ext

logger = logging.getLogger(__name__)

LOG2E = 1.4426950408889634
HQ, HKV, D = 16, 2, 64
SLOTS4 = 132  # 1 CTA per SM
IT_US = 1.41  # bank cost model: us per 128-key iteration per CTA
EFF4 = 0.9  # 4-consumer-warpgroup kernel's per-token cost relative to the 3-warpgroup one
PATH_STATS = os.environ.get("OKC_PATH_STATS", "0") == "1"
# fmha = ran here; fa3 = declined (FA3 ran) although the call had >= the token threshold
HITS = {"fmha": 0, "fmha_tokens": 0, "fa3": 0}
_FIRED = set()
_SCALES = {}


def _log_once(path: str, detail: str) -> None:
    if path not in _FIRED:
        _FIRED.add(path)
        logger.info("OKC path active: %s %s", path, detail)


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _scale_tensor(value: float, device) -> torch.Tensor:
    key = (value, str(device))
    t = _SCALES.get(key)
    if t is None:
        t = _SCALES[key] = torch.full((), value, dtype=torch.float32, device=device)
    return t


def pick_cfg(L: int, ntf: int):
    """(consumer warpgroups, split) for one request of L query tokens over ntf 128-key blocks: the bank's
    pick_cfg4 (waves x (blocks / split + 20) iterations, plus the partials' HBM round trip when split)."""
    rows = L * 16
    best = (float("inf"), 3, 1)
    for nc in (3, 4):
        units = ((L + 8 * nc - 1) // (8 * nc)) * 2
        for s in range(1, 17):
            waves = -(-units * s // SLOTS4)
            cost = waves * (ntf / s + 20.0) * IT_US * 0.75 * (nc / 3) * (EFF4 if nc == 4 else 1.0)
            if s > 1:
                cost += s * rows * 136 * 2 / 3e12 * 1e6 + 3.0
            if cost < best[0] - 1e-9:
                best = (cost, nc, s)
    return best[1], best[2]


def _decline(tokens: int, threshold: int) -> None:
    if tokens >= threshold:
        HITS["fa3"] += 1
        if PATH_STATS and _pow2(HITS["fa3"]):
            logger.info("OKC prefill_fmha declined: %s (tokens %d)", HITS, tokens)
    return None


def try_extend(backend, q, key_cache, value_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q,
               layer, forward_batch, window_size, causal, kwargs, out) -> Optional[torch.Tensor]:
    """fmha7 for this extend call, or None (the backend then calls FA3 with the same arguments).
    q: e4m3 [T, HQ * D] after prepare_paged_mha_query; key_cache / value_cache: the backend's paged views."""
    tokens = q.shape[0]
    is_swa = window_size != (-1, -1)
    threshold = PFMHA_SLIDING_MIN_TOKENS if is_swa else PFMHA_MIN_TOKENS
    if tokens < threshold or (is_swa and not PFMHA_SLIDING):
        return None
    if torch.cuda.is_current_stream_capturing():
        return _decline(tokens, threshold)
    if not causal or layer.logit_cap or backend.page_size != 1 or backend.fa_impl_ver != 3:
        return _decline(tokens, threshold)
    if set(kwargs) - {"k_descale", "v_descale"}:
        return _decline(tokens, threshold)
    if is_swa and window_size[1] != 0:
        return _decline(tokens, threshold)
    if forward_batch.batch_size != 1 or cu_seqlens_q is None or cu_seqlens_q.numel() != 2:
        return _decline(tokens, threshold)
    if int(max_seqlen_q) != tokens:
        return _decline(tokens, threshold)
    if (layer.tp_q_head_num, layer.tp_k_head_num, layer.head_dim, layer.v_head_dim) != (HQ, HKV, D, D):
        return _decline(tokens, threshold)
    fp8 = torch.float8_e4m3fn
    if q.dtype is not fp8 or key_cache.dtype is not fp8 or value_cache.dtype is not fp8:
        return _decline(tokens, threshold)
    if not (key_cache.is_contiguous() and value_cache.is_contiguous()):
        return _decline(tokens, threshold)
    if page_table is None or page_table.dtype is not torch.int32 or page_table.stride(1) != 1:
        return _decline(tokens, threshold)
    if cache_seqlens is None or cache_seqlens.dtype is not torch.int32 or cache_seqlens.numel() != 1:
        return _decline(tokens, threshold)
    max_len = page_table.shape[1]  # = metadata.max_seq_len_k, the request's prefix + chunk (host int)
    if max_len < tokens:
        return _decline(tokens, threshold)

    device = q.device
    ks = kwargs.get("k_descale")
    vs = kwargs.get("v_descale")
    if ks is None or vs is None:
        ks = vs = _scale_tensor(1.0, device)
    elif ks.dtype is not torch.float32 or vs.dtype is not torch.float32:
        return _decline(tokens, threshold)
    # (expanded [bs, hkv] descale views of the layer's [] scale: data_ptr()[0] is that scale)

    ext = fmha_ext()
    L = tokens
    win = window_size[0] + 1 if is_swa else 0
    cap = ((min(win - 1 + L, max_len) if win else max_len) + 255) & ~255
    capblks = cap // 128
    kc = key_cache.view(torch.uint8)
    vc = value_cache.view(torch.uint8)
    kvt = torch.empty((2 * capblks, 16384), dtype=torch.uint8, device=device)
    if win:
        seq = cache_seqlens - L  # the request's cached prefix, on device
        lens = torch.empty_like(cache_seqlens)
        ext.gatherp(kc, vc, kvt, page_table, cache_seqlens, seq, lens, page_table.stride(0), 0, capblks, 1, L, win)
        ntf = min(capblks, (win + 23) // 128 + 2)
    else:
        lens = cache_seqlens
        none = cache_seqlens[:0]
        ext.gatherp(kc, vc, kvt, page_table, cache_seqlens, none, none, page_table.stride(0), 0, capblks, 1, L, 0)
        ntf = capblks
    nc, split = pick_cfg(L, ntf)
    if out is None or out.dtype is not torch.bfloat16 or not out.is_contiguous() or out.numel() != L * HQ * D:
        out = torch.empty((L, HQ, D), dtype=torch.bfloat16, device=device)
    o = out.view(L, HQ, D)
    smqk = float(layer.scaling) * LOG2E
    q8 = q.contiguous().view(torch.uint8)
    if split <= 1:
        ext.fmha4n_split(q8, kvt, o, o, o, lens, ks, vs, smqk, 1, L, capblks, 1, win, nc)
    else:
        rows = L * HQ
        po = torch.empty((split, rows, D), dtype=torch.float16, device=device)
        pml = torch.empty((split, rows, 2), dtype=torch.float32, device=device)
        ext.fmha4n_split(q8, kvt, o, po, pml, lens, ks, vs, smqk, 1, L, capblks, split, win, nc)
        ext.combine(po, pml, o, rows, split)
    HITS["fmha"] += 1
    HITS["fmha_tokens"] += L
    _log_once("prefill_fmha" + ("_swa" if win else ""), f"(first tokens={L}, len={max_len}, nc={nc}, split={split})")
    if PATH_STATS and _pow2(HITS["fmha"]):
        logger.info("OKC prefill_fmha hits: %s (layer %d, tokens %d, len %d)", HITS, layer.layer_id, L, max_len)
    return o
