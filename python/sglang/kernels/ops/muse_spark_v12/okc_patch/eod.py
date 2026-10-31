"""OKC_EOD: the SP prefill MoE section without its rows all-gather (muse-opt docs/serving/review-1031/EOD.md 5.2).

Every rank keeps its own (pre-normalized) SP shard rows. After the scores / ids gather, every rank derives every
destination's layout from the gathered ids, the 8 ranks' replica tables and the served replica cut (eod_galign,
deterministic, so all ranks agree): each destination's slots sit in its layout by (route count ascending, slot), and a
route's push key / counter is its slot's position there (rank_of: this rank's slot -> position, written per call). Its
gg1 (eod_gemm: the pg wide build's gg1 plus copier warps and per-slot gates) pushes the own rows into every rank's
symm-mem token images in key order while it computes; a slot's items start once all 8 sources have released its
position. gg2 and the M1-RS combine are today's (gg2 reads block_experts' weight slots, so the order is free there).

Image reuse needs no handshake: EOD runs only where M1-RS runs for the same call, every rank's gg1 of a layer precedes
its gg2 releases on the stream, and every rank's M1-RS receive of that layer waits for all world x SMs releases before
the rank's next EOD call pushes again. Counters are epoch-monotonic: eod_galign bumps a device epoch word once per call
before gg1, so a captured graph re-reads it on every replay, and every rank runs the same calls in the same order.
Buffers (images, counters) are symmetric, made on the first eager call (never inside a capture) and agreed over the TP
ranks like okc_patch.m1's; a rank where setup fails takes the gathered path with all the others.
"""
import logging
import os
import sys

import torch

from sglang.kernels.ops.muse_spark_v12 import okc_patch as _sw

logger = logging.getLogger(__name__)

_BM = 256        # the pg wide build's token block (the only build eod_gemm instantiates)
_CHUNK = 4096    # eod_galign's route chunk: T * 8 and S * 8 must be multiples
_NAN = 0x7FC0    # bf16 quiet NaN (OKC_EOD_POISON)
_STATE = {}


def _blocks(tokens, nl):
    """align_local's block count for `tokens` gathered tokens over nl local slots."""
    return (tokens * 8 + nl * (_BM - 1) + _BM - 1) // _BM


def _setup(device, nl):
    key = str(device)
    st = _STATE.get(key, False)
    if st is not False:
        return st
    if torch.cuda.is_current_stream_capturing():
        return None
    from sglang.kernels.ops.muse_spark_v12.okc_patch.ext_coll import _symm_ok, agreed

    st = None
    if _symm_ok():
        import torch.distributed._symmetric_memory as symm_mem

        from sglang.srt.distributed.parallel_state import get_tp_group

        g = get_tp_group()
        name = g.device_group.group_name
        nblocks = _blocks(_sw.M1RS_MAX_T, 64)   # every layer's slot count fits (+64 MiB / rank over 33 slots)
        img = agreed(lambda: symm_mem.empty(nblocks * 64 * _BM * 64, dtype=torch.bfloat16, device=device))
        hi = img is not None and agreed(lambda: symm_mem.rendezvous(img, name))
        cnt = hi and agreed(lambda: symm_mem.empty(64, dtype=torch.int32, device=device))
        hc = cnt is not None and cnt is not False and agreed(lambda: symm_mem.rendezvous(cnt, name))
        if hc and g.world_size == 8:
            img.view(torch.int16).fill_(_NAN if _sw.EOD_POISON else 0)
            cnt.zero_()
            torch.cuda.synchronize()
            torch.distributed.barrier(group=g.cpu_group)   # every rank's counters are zero before any release
            st = {"img": img, "img_ptrs": [int(p) for p in hi.buffer_ptrs], "cnt": cnt,
                  "cnt_ptrs": [int(p) for p in hc.buffer_ptrs], "nblocks": nblocks, "nl": nl,
                  "lcnt": torch.zeros((64,), dtype=torch.int32, device=device),
                  "ep": torch.zeros((1,), dtype=torch.int32, device=device),
                  "rank_of": torch.zeros((64,), dtype=torch.int32, device=device),   # eod_galign writes it per call
                  "W": 3 * torch.cuda.get_device_properties(device).multi_processor_count,
                  "rank": g.rank_in_group, "group": g, "woff": {}, "calls": 0}
            logger.info("OKC EOD setup: images %.3f GiB (%d blocks), %.2f GiB free after.",
                        img.numel() * 2 / 2**30, nblocks, torch.cuda.mem_get_info(device)[0] / 2**30)   # MEM-F5
    _STATE[key] = st
    if st is not None and _sw.EOD_EPOCH_DIR:
        _epoch_probe(st, device)
    return st


def _epoch_probe(st, device):
    """OKC_EOD_EPOCH_DIR (identity harness only): log the device epoch word whenever <dir>/epoch's mtime moves."""
    import threading
    import time

    host = torch.empty((1,), dtype=torch.int32).pin_memory()

    def run():
        torch.cuda.set_device(device)
        s, seen = torch.cuda.Stream(device=device), 0.0
        while True:
            time.sleep(0.5)
            try:
                m = os.path.getmtime(os.path.join(_sw.EOD_EPOCH_DIR, "epoch"))
            except OSError:
                continue
            if m > seen:
                seen = m
                with torch.cuda.stream(s):
                    host.copy_(st["ep"], non_blocking=True)
                s.synchronize()
                logger.info("OKC EOD device epoch: %d (rank %d).", int(host[0]), st["rank"])

    threading.Thread(target=run, daemon=True, name="okc-eod-epoch").start()


def _p1ss_pass():
    """True inside a P1-SS two-child pass (the w1b family marks each stage in muse_spark._OKC_P1SS["child"] and
    clears it after the pass), and whenever P1-SS is switched on but that mark is missing (fail closed). One image
    set, counter set and epoch word per device serve one forward at a time, and a gated gg1 beside the other child's
    kernels could wait on a peer that waits on it (qa-eod E3 item 2)."""
    st = getattr(sys.modules.get("sglang.srt.models.muse_spark"), "_OKC_P1SS", None)
    if isinstance(st, dict):
        return st.get("child") is not None
    return bool(getattr(_sw, "SW_P1SS", False)) or os.environ.get("OKC_SW_P1SS", "0") == "1"


def active(layer, tokens, device):
    """Whether this SP call (`tokens` gathered tokens) runs EOD; host-side sizes and agreed setup only, so every rank
    decides alike. The caller skips the rows gather when it does. Never inside a P1-SS two-child pass, never with
    adaptive replicas (they rebuild the tables in place after the once-per-layer gather)."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import m1 as okc_m1
    from sglang.kernels.ops.muse_spark_v12.okc_patch import moe as okc_moe
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe

    if _sw.ADAPT_REPLICAS or _p1ss_pass():
        return False
    state = getattr(layer, "_okc_moe_state", None)
    if (state is None or not state["pg_only"] or tokens % _CHUNK or pg_moe.block_m(tokens) != _BM
            or layer.num_local_experts > 64 or okc_moe._rep_mode(layer, tokens) not in (None, "split")
            or okc_m1.rs_slot(layer, tokens, device) is None):
        return False
    st = _setup(device, layer.num_local_experts)
    if st is None or _blocks(tokens, layer.num_local_experts) > st["nblocks"]:
        return False
    if getattr(layer, "_okc_eod_tabs", None) is None or tokens not in st["woff"]:
        if torch.cuda.is_current_stream_capturing():   # a collective / a host copy: eager calls only (shared state)
            return False
        if getattr(layer, "_okc_eod_tabs", None) is None:
            _tables(layer, st, device)
        _woff(st, tokens, device)                       # S * 8 = T entries per rank
    return True


def _tables(layer, st, device):
    """The 8 ranks' [256] replica-table words (slot | home << 8 | replicated << 11 | replica rank << 12), gathered once
    per layer: the replica table's head when the layer has one, else made from the 8 expert maps. Only the slot field is
    per rank; every rank must hold the same home / replica fields and the same replicated list (rep[256:]), since the
    cut every rank computes alone rests on them (qa-eod E3 addition a): checked here, on every rank alike."""
    g = st["group"]
    rep = getattr(layer, "_okc_rep_table", None)
    if rep is not None:
        mine = rep[:256].to(device=device, dtype=torch.int32).contiguous()
    else:
        mine = layer._expert_map.to(device=device, dtype=torch.int32).contiguous()
    allm = torch.empty((8, 256), dtype=torch.int32, device=device)
    torch.distributed.all_gather_into_tensor(allm, mine, group=g.device_group)
    if rep is None:   # expert maps (-1 = not here) -> slot | home << 8
        held = allm >= 0
        home = held.int().argmax(0)
        allm = torch.where(held, allm, torch.full_like(allm, 255)) | (home[None, :] << 8)
    ok = bool(((allm >> 8) == (allm[:1] >> 8)).all())
    if rep is not None:
        tail = rep[256:].to(device=device, dtype=torch.int32).contiguous()
        ns = torch.empty((8,), dtype=torch.int32, device=device)
        torch.distributed.all_gather_into_tensor(ns, torch.tensor([tail.numel()], dtype=torch.int32, device=device),
                                                 group=g.device_group)
        if bool((ns == ns[:1]).all()) and tail.numel():
            tails = torch.empty((8, tail.numel()), dtype=torch.int32, device=device)
            torch.distributed.all_gather_into_tensor(tails, tail, group=g.device_group)
            ok = ok and bool((tails == tails[:1]).all())
        else:
            ok = ok and bool((ns == ns[:1]).all())
    if not ok:
        raise RuntimeError("OKC_EOD: the ranks' replica tables disagree on the home / replica fields or the replicated "
                           "list; every rank's global align would place routes differently")
    layer._okc_eod_tabs = allm.contiguous()
    return layer._okc_eod_tabs


def _woff(st, n, device):
    """Copier list offsets: entry j goes to warp j % W, so warp q holds n // W entries, one more for q < n % W."""
    w = st["woff"].get(n)
    if w is None:
        W = st["W"]
        q = torch.arange(W + 1, dtype=torch.int64)
        w = st["woff"][n] = (q * (n // W) + torch.clamp(q, max=n % W)).to(device=device, dtype=torch.int32)
    return w


def forward(layer, x, topk_scores, topk_ids):
    """x: this rank's pre-normalized shard [S, 4096]; topk_scores / topk_ids: the gathered [T, 8] -> this rank's
    combined shard [S, 4096] (M1-RS receive)."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import cuda_ext, pg_moe
    from sglang.kernels.ops.muse_spark_v12.okc_patch import m1 as okc_m1
    from sglang.kernels.ops.muse_spark_v12.okc_patch import moe as okc_moe

    assert not _p1ss_pass(), "OKC_EOD inside a P1-SS two-child pass"
    st = _STATE[str(x.device)]
    state = layer._okc_moe_state
    T, S, nl = topk_ids.shape[0], x.shape[0], layer.num_local_experts
    numel = T * 8
    dev = x.device
    ids = topk_ids.to(torch.int32).contiguous()
    rs = okc_m1.rs_slot(layer, T, dev)
    ext = cuda_ext.pg_eod_ext()
    tabs = layer._okc_eod_tabs
    cut = torch.zeros((256,), dtype=torch.int32, device=dev)
    if okc_moe._rep_mode(layer, T) == "split":   # the served split's cut (split_replicas without its split pass)
        rep = layer._okc_rep_table
        n_rep, sb = rep.numel() - 256, 64
        while n_rep * -(-T // sb) > 12288:
            sb *= 2
        cnt = torch.empty((256,), dtype=torch.int32, device=dev)
        bc = torch.empty((max(1, n_rep * -(-T // sb)),), dtype=torch.int32, device=dev)
        ext.eod_rep_cut(ids, rep, cnt, bc, cut, sb)
    nblocks = _blocks(T, nl)
    hist = torch.empty((numel // _CHUNK, 512), dtype=torch.int32, device=dev)
    npad = torch.empty((1,), dtype=torch.int32, device=dev)
    be = torch.empty((nblocks,), dtype=torch.int32, device=dev)
    sid = torch.empty((nblocks * _BM,), dtype=torch.int32, device=dev)
    ent = torch.empty((S * 8, 4), dtype=torch.int32, device=dev)
    ext.eod_galign(ids, tabs, cut, st["rank"], S, _BM, st["W"], nl, hist, npad, be, sid, ent, st["ep"],
                   st["rank_of"])
    live = torch.empty((nblocks + 1,), dtype=torch.int32, device=dev)
    pg = pg_moe._ext(_BM)
    pg.tiles(be, npad, sid, live, nblocks, _BM, numel, pg_moe._WMASK, state["pg_wsf13"].shape[0])
    inter_t = torch.empty((nblocks * 64 * _BM * 64,), dtype=torch.bfloat16, device=dev)
    x = x.contiguous()
    woff = st["woff"][S * 8]
    if _sw.EOD_DROP_RANK == st["rank"]:   # identity-harness control: this rank pushes nothing, the gates stay on
        woff = st.setdefault("woff0", torch.zeros_like(woff))
    args = ([st["cnt"].data_ptr(), 0, st["rank_of"].data_ptr(), 0, 0, 0, nl, ent.data_ptr(),
             woff.data_ptr(), x.data_ptr()] + st["img_ptrs"] + [0, 0, 0, st["ep"].data_ptr()]
            + st["cnt_ptrs"] + [st["lcnt"].data_ptr(), ((0 if _sw.EOD_UNGATED else 8) << 32) | 0x7FFFFFFF])
    ext.eod_gg1(st["img"], state["pg_wsf13"], state["w13_gs32"], inter_t, be, live, numel, 11,
                torch.tensor(args, dtype=torch.int64))
    if _sw.EOD_POISON:   # this layer's rows are consumed and no peer pushes the next layer's before gg2's releases
        st["img"].view(torch.int16).fill_(_NAN)
    y = torch.empty((1, 4096), dtype=torch.bfloat16, device=dev)
    cuda_ext.pg_rs_ext(False).gg2(inter_t, state["pg_wsf2"], y, sid, be, live, numel, pg_moe._CFG, 0, 0,
                                  rs[1].data_ptr(), rs[3])
    st["calls"] += 1
    if st["calls"] == 1 or st["calls"] % 4096 == 0:
        logger.info("OKC EOD active: %d calls (T=%d, %d slots)%s.", st["calls"], T, nl,
                    (", poison" if _sw.EOD_POISON else "") + (", ungated" if _sw.EOD_UNGATED else ""))
    return okc_m1.rs_receive(layer, rs, topk_scores)
