"""OKC_SW_M1: the verify MoE combine issued from gg2's epilogue (lane-moe M1).

Every rank's gg2 multicasts its route rows into every rank's slot buffer [T * 8, 4096] (NVLS) and releases once per
CTA; a receiver on every rank sums the 8 slots of each token (post-expert norm, scores) into the combined output, so
neither the finalize nor the combine all-reduce runs.

Two slot buffers plus epoch words, symmetric with multicast mappings, are made on the first eager call (never inside a
capture) and agreed over the TP ranks. MoE layers alternate them by layer parity, so a buffer is written again only two
MoE layers later, after the all-reduce in between has ordered every rank's receiver before any rank's next gg2.
"""
import torch

from sglang.kernels.ops.muse_spark_v12 import okc_patch as _sw

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
        bufs = []
        for _ in range(2):
            slot = agreed(lambda: symm_mem.empty(_sw.M1_MAX_T * 8, 4096, dtype=torch.bfloat16, device=device))
            sh = slot is not None and agreed(lambda: _mc(slot, name))
            ep = sh and agreed(lambda: symm_mem.empty(16, dtype=torch.int32, device=device))
            eh = ep is not None and ep is not False and agreed(lambda: _mc(ep, name))
            if not eh:
                break
            ep.zero_()
            bufs.append((slot, int(sh.multicast_ptr), ep, int(eh.multicast_ptr)))
        if len(bufs) == 2:
            torch.cuda.synchronize()
            torch.distributed.barrier(group=g.cpu_group)   # every rank's epoch words are zero before any release
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            st = {"bufs": bufs, "want": g.world_size * sms}
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
