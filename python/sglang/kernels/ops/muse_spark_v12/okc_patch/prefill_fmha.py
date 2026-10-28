"""OKC prefill attention: the prefill lane's FP8 FMHA (fmha7) in place of FA3's extend call.

Seat. FlashAttentionBackend.forward_extend calls try_extend right where it would call FA3 on the paged KV
cache: after the stock KV store and the stock FP8 cast of q, inside the eager attention break of the breakable
prefill CUDA graph (the break reads the live batch, so host shapes are the real ones). Everything around the
call stays stock (QK norm, KV store, output norm and gate); try_extend computes the same attention FA3 would
and returns None whenever it does not apply, so the backend calls FA3.

Applies to: Muse target layers (MuseSpark1xAttention marks its RadixAttention), forward mode EXTEND, one
request (the 16K chunks of a first turn and single follow-up extends; a ragged co-batch with OKC_PFMHA_RAGGED,
one call per request), q / KV e4m3, page size 1,
16 q heads / 2 kv heads / head dim 64 per rank, causal, no softcap / sinks / score mod, not under capture.
Full layers attend [0, len); sliding layers (OKC_PFMHA_SLIDING) the 2048-key window. A request below the
layer's token threshold (OKC_PFMHA_MIN_TOKENS / OKC_PFMHA_SLIDING_MIN_TOKENS) keeps its batch on FA3.

Kernel (okc_patch/_cuda_src_fmha.py): gather_kp copies the request's paged K/V rows into per (kv head,
128-key block) 16 KB stage images (zero-filled past the context), fmha7 streams them by TMA into 3 or 4
consumer warpgroups (f16 QK, lazy-max hybrid exp2, e4m3 P, fp8 PV, tensor-core row sums), split-KV + combine
when one request's grid under-fills the GPU (pick_cfg below = the bank's pick_cfg4). OKC_PREFILL_FMHA_EXACT=1 runs
every exp2 on MUFU (no piecewise-linear exp). Row blocks (OKC rsfold O1): see defer_row_blocks.
"""

import contextlib
import logging
import os
from typing import Optional

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch import (
    PFMHA_EXACT,
    PFMHA_MIN_TOKENS,
    PFMHA_PREFETCH,
    PFMHA_RAGGED,
    PFMHA_RAGGED_MAX_BS,
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
_DEFER = (0, 0)
_DEFERRED = None  # (output, blocks, run)
# PFMHA_PREFETCH state: the side stream, two persistent stage-image slots, the one pending prefetch (the next full
# layer's prefix blocks), the learned order of full layers (layer id -> next full layer id of the same forward) and
# the last full call (layer id, forward batch).
_PF = {"stream": None, "bufs": [None, None], "pending": None, "next": {}, "last": None}


@contextlib.contextmanager
def defer_row_blocks(rows: int, blocks: int):
    """A one-request call of exactly rows * blocks tokens inside gathers and leaves its output unwritten;
    take_deferred(output) gives run(j, n): fmha7 at the whole call's (nc, split) over rows [j rows, (j + n) rows) with
    that range's length. Blocks are whole 8 * nc query tiles, so every CTA computes as in the whole call (bit-equal)."""
    global _DEFER
    _DEFER = (rows, blocks)
    try:
        yield
    finally:
        _DEFER = (0, 0)


def take_deferred(output: torch.Tensor):
    """run of the deferred call whose output `output` is, else None (another deferred call runs whole first)."""
    global _DEFERRED
    d, _DEFERRED = _DEFERRED, None
    if d is None:
        return None
    if output.data_ptr() == d[0].data_ptr() and output.numel() == d[0].numel():
        return d[2]
    d[2](0, d[1])
    return None


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
    bs = forward_batch.batch_size
    if cu_seqlens_q is None or cu_seqlens_q.numel() != bs + 1 or (bs > 1 and not PFMHA_RAGGED):
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
    if page_table.shape[0] != bs or cache_seqlens is None or cache_seqlens.dtype is not torch.int32:
        return _decline(tokens, threshold)
    if cache_seqlens.numel() != bs:
        return _decline(tokens, threshold)
    if bs == 1:
        if int(max_seqlen_q) != tokens:
            return _decline(tokens, threshold)
        # page_table = req_to_token[req, :max_seq_len_k]: its width is the request's prefix + chunk (host int)
        reqs = [(0, tokens, page_table.shape[1])]
    else:
        # A ragged co-batch (follow-up extends, a chunk beside them): one call per request, each over the threshold.
        q_lens = forward_batch.extend_seq_lens_cpu
        seq_cpu = forward_batch.seq_lens_cpu
        if q_lens is None or seq_cpu is None or len(q_lens) != bs or sum(q_lens) != tokens:
            return _decline(tokens, threshold)
        if min(q_lens) < threshold or bs > PFMHA_RAGGED_MAX_BS:
            return _decline(tokens, threshold)
        reqs, off = [], 0
        for i, n in enumerate(q_lens):
            reqs.append((off, int(n), int(seq_cpu[i])))
            off += int(n)
    if any(max_len < n or max_len > page_table.shape[1] for _, n, max_len in reqs):
        return _decline(tokens, threshold)

    device = q.device
    ks = kwargs.get("k_descale")
    vs = kwargs.get("v_descale")
    if ks is None or vs is None:
        ks = vs = _scale_tensor(1.0, device)
    elif ks.dtype is not torch.float32 or vs.dtype is not torch.float32:
        return _decline(tokens, threshold)
    # (expanded [bs, hkv] descale views of the layer's [] scale: data_ptr()[0] is that scale)

    ext = fmha_ext(PFMHA_EXACT)
    win = window_size[0] + 1 if is_swa else 0
    usable = out is not None and out.dtype is torch.bfloat16 and out.is_contiguous() and out.numel() == tokens * HQ * D
    # row blocks: one request of exactly the blocks, and no caller copy of an unwritten result (out is used)
    R = _DEFER[0] if bs == 1 and tokens == _DEFER[0] * _DEFER[1] > 0 and (out is None or usable) else 0
    if not usable:
        out = torch.empty((tokens, HQ, D), dtype=torch.bfloat16, device=device)
    o = out.view(tokens, HQ, D)
    smqk = float(layer.scaling) * LOG2E
    q8 = q.contiguous().view(torch.uint8).view(tokens, HQ * D)
    kc = key_cache.view(torch.uint8)
    vc = value_cache.view(torch.uint8)
    pf = (layer.layer_id, forward_batch, backend) if PFMHA_PREFETCH and bs == 1 and not win else None
    for i, (off, L, max_len) in enumerate(reqs):
        nc, split = _one(ext, q8[off:off + L], kc, vc, page_table[i:i + 1], cache_seqlens[i:i + 1], L, max_len,
                         win, ks, vs, smqk, o[off:off + L], device, R, pf)
    HITS["fmha"] += 1
    HITS["fmha_tokens"] += tokens
    _log_once("prefill_fmha" + ("_swa" if win else "") + ("_ragged" if bs > 1 else ""),
              f"(first bs={bs}, tokens={tokens}, len={reqs[-1][2]}, nc={nc}, split={split})")
    if PATH_STATS and _pow2(HITS["fmha"]):
        logger.info("OKC prefill_fmha hits: %s (layer %d, bs %d, tokens %d)", HITS, layer.layer_id, bs, tokens)
    return o


def _one(ext, q8, kc, vc, page_table, cache_seqlens, L, max_len, win, ks, vs, smqk, o, device, R=0, pf=None):
    """One request: gather its keys into stage images, then fmha7 (+ combine), or with R the deferred R-row blocks.
    pf = (layer id, forward batch, backend): a full layer of a one-request call (PFMHA_PREFETCH). Returns (nc, split)."""
    global _DEFERRED
    cap = ((min(win - 1 + L, max_len) if win else max_len) + 255) & ~255
    capblks = cap // 128
    if win:
        kvt = torch.empty((2 * capblks, 16384), dtype=torch.uint8, device=device)
        seq = cache_seqlens - L  # the request's cached prefix, on device
        lens = torch.empty_like(cache_seqlens)
        ext.gatherp(kc, vc, kvt, page_table, cache_seqlens, seq, lens, page_table.stride(0), 0, capblks, 1, L, win,
                    0, 0)
        ntf = min(capblks, (win + 23) // 128 + 2)
    else:
        lens = cache_seqlens
        kvt = _full_images(ext, kc, vc, page_table, cache_seqlens, L, max_len, capblks, device, pf)
        ntf = capblks
    nc, split = pick_cfg(L, ntf)
    if R and R % (8 * nc) == 0:
        blens = lens - torch.arange(L // R - 1, -1, -1, dtype=torch.int32, device=device) * R   # block ends

        def run(j, n):
            r = slice(j * R, (j + n) * R)
            _attend(ext, q8[r], kvt, o[r], blens[j + n - 1:j + n], ks, vs, smqk, n * R, capblks, win, nc, split, device)

        _DEFERRED = (o, L // R, run)
    else:
        _attend(ext, q8, kvt, o, lens, ks, vs, smqk, L, capblks, win, nc, split, device)
    return nc, split


def _full_images(ext, kc, vc, page_table, cache_seqlens, L, max_len, capblks, device, pf):
    """A full layer's stage images [2 * capblks, 16 KB]. With pf, blocks [0, old) (prefix rows only) may already be
    staged in a slot by the previous full layer's call on _PF's side stream: this call then gathers only blocks
    [old, capblks) (its new rows, after its KV store), and stages the next full layer's prefix blocks into the other
    slot beside this layer's fmha7. The side stream starts after everything this stream has queued so far (so after
    the fmha7 that last read that slot) and takes the next layer's K / V buffers in its own context (a HiCache
    layer-wise load then gates that stream, not this one)."""
    none = cache_seqlens[:0]
    stride = page_table.stride(0)
    old = (max_len - L) >> 7
    if pf is None:
        kvt = torch.empty((2 * capblks, 16384), dtype=torch.uint8, device=device)
        ext.gatherp(kc, vc, kvt, page_table, cache_seqlens, none, none, stride, 0, capblks, 1, L, 0, 0, 0)
        return kvt
    lid, fb, backend = pf
    key = (max_len, L, capblks, page_table.data_ptr(), stride, cache_seqlens.data_ptr())
    ent, _PF["pending"] = _PF["pending"], None
    main = torch.cuda.current_stream()
    slot = None
    if ent is not None and ent["layer"] == lid and ent["fb"] is fb and ent["key"] == key:
        slot = ent["slot"]
        kvt = _PF["bufs"][slot][: 2 * capblks]
        main.wait_event(ent["done"])
        ext.gatherp(kc, vc, kvt, page_table, cache_seqlens, none, none, stride, 0, capblks, 1, L, 0, old, 0)
    else:
        if ent is not None:   # an unused prefetch: order its slot's later reuse after its writes
            main.wait_event(ent["done"])
        kvt = torch.empty((2 * capblks, 16384), dtype=torch.uint8, device=device)
        ext.gatherp(kc, vc, kvt, page_table, cache_seqlens, none, none, stride, 0, capblks, 1, L, 0, 0, 0)
    last = _PF["last"]
    if last is not None and last[1] is fb and last[0] != lid:
        _PF["next"][last[0]] = lid
    _PF["last"] = (lid, fb)
    nxt = _PF["next"].get(lid)
    if nxt is None or old <= 0:
        return kvt
    ns = 1 - slot if slot is not None else 0
    buf = _PF["bufs"][ns]
    if buf is None or buf.shape[0] < 2 * capblks:
        buf = _PF["bufs"][ns] = torch.empty((2 * capblks, 16384), dtype=torch.uint8, device=device)
    side = _PF["stream"]
    if side is None or side.device != main.device:
        side = _PF["stream"] = torch.cuda.Stream(device=main.device)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        k2, v2 = backend.token_to_kv_pool.get_kv_buffer(nxt)
        if not (k2.is_contiguous() and v2.is_contiguous() and k2.element_size() == 1 and v2.element_size() == 1):
            return kvt
        ext.gatherp(k2.view(torch.uint8), v2.view(torch.uint8), buf[: 2 * capblks], page_table, cache_seqlens, none,
                    none, stride, 0, capblks, 1, L, 0, 0, old)
        done = torch.cuda.Event()
        done.record(side)
    _PF["pending"] = {"layer": nxt, "fb": fb, "key": key, "slot": ns, "done": done}
    return kvt


def _attend(ext, q8, kvt, o, lens, ks, vs, smqk, L, capblks, win, nc, split, device):
    if split <= 1:
        ext.fmha4n_split(q8, kvt, o, o, o, lens, ks, vs, smqk, 1, L, capblks, 1, win, nc)
    else:
        rows = L * HQ
        po = torch.empty((split, rows, D), dtype=torch.float16, device=device)
        pml = torch.empty((split, rows, 2), dtype=torch.float32, device=device)
        ext.fmha4n_split(q8, kvt, o, po, pml, lens, ks, vs, smqk, 1, L, capblks, split, win, nc)
        ext.combine(po, pml, o, rows, split)
