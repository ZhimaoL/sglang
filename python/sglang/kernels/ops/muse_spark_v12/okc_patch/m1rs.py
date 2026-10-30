"""OKC_SW_M1RS: the SP prefill / extend MoE combine issued from gg2's epilogue, in the reduce-scatter form (lane-extend
E57).

Under SP every rank runs its experts on the whole gathered chunk of T tokens; today the finalize sums each token's
local routes into a [T, 4096] partial and a reduce-scatter hands each rank its shard of T / tp tokens. With M1-RS, gg2
stores every route row (route r = token * 8 + slot) into the slot buffer of the token's home rank (r / T: the rank whose
SP shard holds the token) at row r - home * T, and releases once per CTA on every rank's epoch word. The home rank's
receiver (pg's m1_recv) sums its T / tp tokens' 8 slots (post-expert norm, scores) into the shard, so neither the
finalize nor the reduce-scatter runs. Each route is computed by exactly one rank (replicas included), so each slot is
written once.

Two slot buffers [M1RS_MAX_T / tp * 8, 4096] plus epoch words, symmetric (the epochs with multicast), are made on the
first eager call (never inside a capture) and agreed over the TP ranks. MoE layers alternate them by layer parity:
buffer L % 2 is written again only two MoE layers later, and the attention between them runs collectives that every rank
enters only after its layer-L receiver (and epoch reset) is done, so no rank overwrites a buffer another still reads
(m1.py's argument, with the SP attention's AG / RS in place of the verify all-reduce).
"""
import torch

from sglang.kernels.ops.muse_spark_v12 import okc_patch as _sw

_STATE = {}


def _setup(device):
    key = str(device)
    st = _STATE.get(key, False)
    if st is not False:
        return st
    if torch.cuda.is_current_stream_capturing():
        return None
    from sglang.kernels.ops.muse_spark_v12.okc_patch.ext_coll import _symm_ok, agreed
    from sglang.kernels.ops.muse_spark_v12.okc_patch.m1 import _mc

    st = None
    if _symm_ok():
        import torch.distributed._symmetric_memory as symm_mem

        from sglang.srt.distributed.parallel_state import get_tp_group

        g = get_tp_group()
        name, world = g.device_group.group_name, g.world_size
        rows = -(-_sw.M1RS_MAX_T // world) * 8
        bufs = []
        for _ in range(2):
            slot = agreed(lambda: symm_mem.empty(rows, 4096, dtype=torch.bfloat16, device=device))
            sh = slot is not None and agreed(lambda: symm_mem.rendezvous(slot, name))
            ep = sh and agreed(lambda: symm_mem.empty(16, dtype=torch.int32, device=device))
            eh = ep is not None and ep is not False and agreed(lambda: _mc(ep, name))
            if not eh:
                break
            ep.zero_()
            table = torch.tensor(list(sh.buffer_ptrs), dtype=torch.int64, device=device)   # home rank -> its slots
            bufs.append((slot, table, ep, int(eh.multicast_ptr)))
        if len(bufs) == 2:
            torch.cuda.synchronize()
            torch.distributed.barrier(group=g.cpu_group)   # every rank's epoch words are zero before any release
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            st = {"bufs": bufs, "want": world * sms, "rank": g.rank_in_group, "world": world}
    _STATE[key] = st
    return st


def slot(layer, tokens, device):
    """(rank's slot rows, home table, epoch words, epoch multicast, releases to wait for, rank, world) for this SP
    call of `tokens` gathered tokens, or None."""
    from sglang.srt.layers.layer_boundary.context import get_forward

    if not (_sw.SW_M1RS and _sw.M1RS_MIN_T <= tokens <= _sw.M1RS_MAX_T and getattr(get_forward(), "sp_active", False)):
        return None
    if getattr(layer, "_post_expert_norm_weight", None) is None:
        return None
    st = _setup(device)
    if st is None or tokens % st["world"]:
        return None
    s, table, e, emc = st["bufs"][getattr(layer, "_okc_layer_idx", 0) & 1]
    return s[: tokens // st["world"] * 8], table, e, emc, st["want"], st["rank"], st["world"]


def receive(layer, m, topk_scores):
    """The home rank's shard [T / tp, 4096] from its slot rows (m from slot())."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import cuda_ext

    rows, table, e, _, want, rank, world = m
    S = rows.shape[0] // 8
    out = torch.empty((S, 4096), dtype=torch.bfloat16, device=rows.device)
    cuda_ext.pg_rs_ext(False).m1_recv(rows, topk_scores[rank * S:(rank + 1) * S].float().contiguous(),
                                      layer._post_expert_norm_weight, out, float(layer._post_expert_norm_eps),
                                      1 if layer._post_expert_norm_weight_first else 0, e.data_ptr(), want)
    layer._okc_m1rs_shard = True
    return out
