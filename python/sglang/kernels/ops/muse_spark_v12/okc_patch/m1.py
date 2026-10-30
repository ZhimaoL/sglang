"""OKC_SW_M1: the verify MoE combine issued from gg2's epilogue (lane-moe M1).

Every rank's gg2 multicasts its route rows into every rank's slot buffer [T * 8, 4096] (NVLS) and releases once per
CTA; a receiver on every rank sums the 8 slots of each token (post-expert norm, scores) into the combined output, so
neither the finalize nor the combine all-reduce runs.

Two slot buffers plus epoch words, symmetric with multicast mappings, are made on the first eager call (never inside a
capture) and agreed over the TP ranks. MoE layers alternate them by layer parity, so a buffer is written again only two
MoE layers later, after the all-reduce in between has ordered every rank's receiver before any rank's next gg2.

OKC_SW_M1RS (lane-extend E57) shares the buffers and epoch words: the SP prefill / extend combine in the reduce-scatter
form. Every rank's gg2 stores each route row (route r = token * 8 + slot of the T gathered tokens) into the slot buffer
of the token's home rank (r / T, the rank whose SP shard holds the token) at row r - home * T, through a table of the
ranks' slot pointers, and releases once per CTA. The home rank's receiver sums its T / tp tokens' 8 slots into its shard,
so neither the finalize nor the reduce-scatter runs. Each route is computed by exactly one rank (the home map or the
replica balancer: split_replicas / the extend balancer), so each slot is written once. The parity argument below holds
with the SP attention's collectives in place of the o-proj all-reduce, and layer L + 1's own epoch wait also orders it
(every rank's L + 1 gg2 follows its L receiver).
"""
import logging

import torch

from sglang.kernels.ops.muse_spark_v12 import okc_patch as _sw

logger = logging.getLogger(__name__)

# Why two buffers by layer parity are enough (no per-use handshake):
# - A slot buffer (and its epoch word) is reused two MoE layers later: layer L writes buffer L % 2, so rank A's gg2 of
#   layer L + 2 is the next writer after layer L.
# - Between them, layer L + 1 runs the attention o-proj all-reduce (custom push AR) on every rank. A rank pushes into
#   that all-reduce only after its own earlier kernels on the stream are done, so rank B's layer-L receiver (and its
#   epoch reset) has finished before B's push.
# - Rank A completes that all-reduce only after B's push arrives, and A's layer L + 2 gg2 runs after it.
# - So no rank overwrites buffer L % 2 or bumps its epoch while any rank still reads it for layer L.
# - The same holds across verify steps (the last MoE layer of a step and the next step's first are two or more
#   all-reduces apart) and between graphs.
_STATE = {}


def _mc(t, name):
    import torch.distributed._symmetric_memory as symm_mem

    h = symm_mem.rendezvous(t, name)
    return h if int(getattr(h, "multicast_ptr", 0) or 0) else None


def _setup(device):
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
        rows = max(_sw.M1_MAX_T * 8, -(-_sw.M1RS_MAX_T // g.world_size) * 8 if _sw.SW_M1RS else 0)
        bufs, tables = [], []
        for _ in range(2):
            slot = agreed(lambda: symm_mem.empty(rows, 4096, dtype=torch.bfloat16, device=device))
            sh = slot is not None and agreed(lambda: _mc(slot, name))
            ep = sh and agreed(lambda: symm_mem.empty(16, dtype=torch.int32, device=device))
            eh = ep is not None and ep is not False and agreed(lambda: _mc(ep, name))
            if not eh:
                break
            ep.zero_()
            bufs.append((slot, int(sh.multicast_ptr), ep, int(eh.multicast_ptr)))
            tables.append(torch.tensor(list(sh.buffer_ptrs), dtype=torch.int64, device=device))   # M1-RS: home -> slots
        if len(bufs) == 2:
            torch.cuda.synchronize()
            torch.distributed.barrier(group=g.cpu_group)   # every rank's epoch words are zero before any release
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            st = {"bufs": bufs, "want": g.world_size * sms, "tables": tables, "rank": g.rank_in_group,
                  "world": g.world_size, "rows": rows}
    _STATE[key] = st
    return st


def slot(layer, tokens, device):
    """(slot buffer, slot multicast, epoch words, epoch multicast, releases to wait for) for this layer's call, or None."""
    if not (_sw.M1_MIN_T <= tokens <= _sw.M1_MAX_T) or getattr(layer, "_post_expert_norm_weight", None) is None:
        return None
    st = _setup(device)
    if st is None:
        return None
    s, smc, e, emc = st["bufs"][getattr(layer, "_okc_layer_idx", 0) & 1]
    return s, smc, e, emc, st["want"]


def rs_slot(layer, tokens, device):
    """M1-RS: (this rank's slot rows, home table, epoch words, epoch multicast, releases to wait for, rank, world) for
    an SP call of `tokens` gathered tokens, or None."""
    from sglang.srt.runtime_context import get_forward

    if not (_sw.SW_M1RS and _sw.M1RS_MIN_T <= tokens <= _sw.M1RS_MAX_T and getattr(get_forward(), "sp_active", False)):
        return None
    if getattr(layer, "_post_expert_norm_weight", None) is None:
        return None
    st = _setup(device)
    if st is None or "tables" not in st or tokens % st["world"] or tokens // st["world"] * 8 > st["rows"]:
        return None
    par = getattr(layer, "_okc_layer_idx", 0) & 1
    s, _, e, emc = st["bufs"][par]
    return s[: tokens // st["world"] * 8], st["tables"][par], e, emc, st["want"], st["rank"], st["world"]


def rs_receive(layer, m, topk_scores):
    """M1-RS: the home rank's shard [T / tp, 4096] from its slot rows (m from rs_slot)."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import cuda_ext

    rows, _, e, _, want, rank, world = m
    S = rows.shape[0] // 8
    out = torch.empty((S, 4096), dtype=torch.bfloat16, device=rows.device)
    cuda_ext.pg_rs_ext(False).m1_recv(rows, topk_scores[rank * S:(rank + 1) * S].float().contiguous(),
                                      layer._post_expert_norm_weight, out, float(layer._post_expert_norm_eps),
                                      1 if layer._post_expert_norm_weight_first else 0, e.data_ptr(), want)
    layer._okc_m1rs_shard = True
    if not _STATE.get("rs_logged"):
        _STATE["rs_logged"] = True
        logger.info("OKC M1-RS active (first T=%d, %d home rows per rank).", S * world, rows.shape[0])
    return out
