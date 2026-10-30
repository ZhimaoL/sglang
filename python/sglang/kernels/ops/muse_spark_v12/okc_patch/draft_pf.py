"""OKC DFlash draft-graph L2 prefetch (OKC_DRAFT_PF; serving/kernels/draft LOG D7). Bitwise: the GEMMs and the
collectives are unchanged, only where their weight loads hit moves.

At a fork point of the captured draft graph, a side-stream kernel (_cuda_src_dpf.py, issue and exit) asks L2 for the
bytes the next GEMMs read first while HBM idles behind a collective, the attention and the glue kernels. The side
stream joins the capture after the layer loop (join()). Per layer, from the bs1 / bs4 traces (rank 0):
- layer start (bit 1): the attention conv's kernel_projection slice (4.2 MB, read ~4 us later) + qkv whole (21 MB, ~15 us);
- after qkv (bit 2): o_proj whole (16.8 MB, read after the attention, ~21 us);
- after o_proj's all-reduce (bit 4): the MLP conv's kernel_projection slice (4.2 MB, ~4 us) + the first DRAFT_PF_GU_COLS
  columns of every gate_up row (29 MB at 2048, ~16 us). nvjet runs gate_up split-K 1 in one wave (56 / 112 CTAs at
  m 8 / 32), so every CTA streams its rows k-band by k-band and that band is what the GEMM reads first.
down_proj follows gate_up's stream directly (no idle window) and is not prefetched.
Graph capture only: an eager launch would add CPU time on the scheduler's critical path.
Generic: issue(ranges) takes whole(t) / band(t, cols) / rows(t, r0, r1) ranges, so another chain can reuse it.
"""

import torch

from sglang.kernels.ops.muse_spark_v12 import okc_patch as _sw

_SIDE = {}  # device index -> side stream
_PENDING = set()  # device indices whose side stream holds work not yet joined
_LSU_ROW = 4096  # whole ranges as rows of 4 KB: one warp per row, 32 lines per pass
_BULK_ROW = 1 << 20  # the bulk form's rows: one cp.async.bulk.prefetch.L2 each


def _ok(t):
    return (isinstance(t, torch.Tensor) and t.is_cuda and t.dim() == 2 and t.dtype is torch.bfloat16
            and t.is_contiguous())


def whole(t):
    """every byte of a contiguous bf16 matrix"""
    return [(t.data_ptr(), t.numel() * 2, 1, t.numel() * 2)] if _ok(t) else []


def rows(t, r0, r1):
    """rows [r0, r1) of a contiguous bf16 matrix (a column-split slice)"""
    if not _ok(t) or not 0 <= r0 < r1 <= t.shape[0]:
        return []
    k = t.shape[1] * 2
    return [(t.data_ptr() + r0 * k, (r1 - r0) * k, 1, (r1 - r0) * k)]


def band(t, cols):
    """the first cols columns of every row of a contiguous bf16 [N, K] matrix (what a split-K 1 GEMM reads first)"""
    if not _ok(t):
        return []
    n, k = t.shape
    cols = min(cols, k) // 64 * 64
    return [(t.data_ptr(), k * 2, n, cols * 2)] if cols > 0 else []


def _boxes(ranges, mode):
    """(base, row stride, rows, bytes per row) ranges -> dpf_issue's flat box list; whole ranges split into rows"""
    out, chunk = [], _BULK_ROW if mode & 1 else _LSU_ROW
    for base, stride, nrows, seg in ranges:
        if base % 128 or seg % 128 or (nrows > 1 and stride % 128):
            continue  # never prefetch past a range
        if nrows == 1 and seg > chunk:
            full = seg // chunk
            out.append((base, chunk, full, chunk))
            if seg - full * chunk:
                out.append((base + full * chunk, 0, 1, seg - full * chunk))
        elif nrows > 0 and (not mode & 1 or seg <= _BULK_ROW):
            out.append((base, stride, nrows, seg))
    return [v for b in out[:4] for v in b]


def issue(ranges):
    """fork the side stream off the current (capturing) stream and prefetch ranges there; no-op outside capture"""
    if not ranges or not torch.cuda.is_current_stream_capturing():
        return
    d = _boxes(ranges, _sw.DRAFT_PF_MODE)
    if not d:
        return
    from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import dpf_ext

    cur = torch.cuda.current_stream()
    dev = cur.device.index
    side = _SIDE.get(dev)
    if side is None:
        side = _SIDE[dev] = torch.cuda.Stream(device=cur.device)
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        dpf_ext().dpf_issue(d, _sw.DRAFT_PF_CTAS, _sw.DRAFT_PF_MODE)
    _PENDING.add(dev)


def begin(_=None):
    """start of a draft forward: nothing pending (a capture that died mid-forward leaves no stale join)"""
    _PENDING.discard(torch.cuda.current_device())


def join(_=None):
    """after the layer loop: the current stream waits for the side stream (the capture's join)"""
    cur = torch.cuda.current_stream()
    dev = cur.device.index
    if dev in _PENDING:
        _PENDING.discard(dev)
        cur.wait_stream(_SIDE[dev])


def _kp_slice(weight):
    """this rank's row block of a replicated kernel_projection, as gemm_ag's gemm_local reads it"""
    if not _sw.SW_DFLASH_CONV_AG or not _ok(weight):
        return []
    from sglang.srt.distributed import get_tp_group

    comm = getattr(get_tp_group(), "ca_comm", None)
    world = getattr(comm, "world_size", 1)
    if comm is None or world < 2 or weight.shape[0] % world:
        return []
    n = weight.shape[0] // world
    return rows(weight, comm.rank * n, (comm.rank + 1) * n)


def layer_start(layer):
    """bit 1: the attention conv's kernel_projection slice + qkv"""
    if not _sw.DRAFT_PF & 1:
        return
    r = _kp_slice(layer.attention_conv.kernel_projection.weight) if layer.attention_conv is not None else []
    issue(r + whole(getattr(layer.self_attn.qkv_proj, "weight", None)))


def after_qkv(attn):
    """bit 2: o_proj, read after the attention"""
    if _sw.DRAFT_PF & 2:
        issue(whole(getattr(attn.o_proj, "weight", None)))


def after_attn(layer):
    """bit 4: the MLP conv's kernel_projection slice + gate_up's first k-band"""
    if not _sw.DRAFT_PF & 4:
        return
    r = _kp_slice(layer.mlp_conv.kernel_projection.weight) if layer.mlp_conv is not None else []
    issue(r + band(getattr(layer.mlp.gate_up_proj, "weight", None), _sw.DRAFT_PF_GU_COLS))
