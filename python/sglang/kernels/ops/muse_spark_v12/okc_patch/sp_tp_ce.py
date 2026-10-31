"""OKC: the SP participant linears' collectives on copy-engine pushes with few barriers.

Under OKC Hopper SP (layers/layernorm_sp.py) the qkv projection all-gathers the token shards before its GEMM and
o_proj reduce-scatters its partial sums after it. At >= OKC_SP_SYMM_MIN_TOKENS tokens these run as torch's symm-mem
fused ops (async TP). Their matmul + reduce-scatter pipelines GEMM -> barrier -> pull -> barrier per peer on two
alternating streams over a two-chunk buffer: 16 barriers per op, every step gated on the slowest rank (session SD,
16K chunk: 953 us per o_proj against ~800 us for its GEMMs and copies; serving/kernels/moe-overlap, E10).

Here both ops push through the copy engines (Tensor.copy_ into the peers' torch symmetric-memory buffers: no SMs)
and synchronize rarely:
  matmul_rs   per destination rank j (peers first, rank-staggered, own last) one cuBLAS GEMM of j's rows, pushed into
              slot `rank` of j's receive buffer as soon as it is done; ONE barrier after the last push; the fp32 sum of
              the tp slots, one bf16 rounding (what torch's reduce_partials computes, in another order).
  ag_matmul   pushes of the shard into every peer's gather buffer in rank-staggered order, a barrier after every
              GROUP push steps, and the GEMM of each source block as soon as its group is published (the own block
              first, from the shard itself), into its rows of the output.
Buffer reuse across layers is ordered by the data: a peer pushes into my buffers for the next op only after a
collective that needed my contribution made after my reads (o_proj slots: the MoE section and the next AG follow
my reduce; gather slots: the next o_proj / MoE reduce-scatter follows my qkv GEMMs). The two ops keep separate
buffers for that reason. Every branch depends only on shapes and switches, so all ranks agree.
Inside defer_slot_sum() (OKC_SW_RS_NORM_FOLD, the decoder layer around its attention) matmul_rs skips the slot sum and
returns an unwritten output; take_slots(out) hands the slots to the post-attention norm, which sums them itself in
torch's order (bit-equal). It reads them where the sum did, before the MoE section, so the reuse order holds.
"""

import contextlib
import logging
import os

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch import LT_PREFILL_O

logger = logging.getLogger(__name__)

# Session SI's 8-GPU bench (16K / 8K chunks, ms per layer vs torch's fused ops): matmul + RS -0.055..-0.065 at 16K and
# -0.044 at 8K at any AG group; AG + matmul -0.052 at 16K with a barrier per push step (GROUP 1), worse with fewer
# barriers and at 8K; a second comm stream loses everywhere. So: matmul + RS on every SP forward the fused ops would
# take, AG + matmul only from OKC_SP_CE_AG_MIN_TOKENS tokens (full 16K chunks), one stream.
# AG push steps per barrier (1 = a barrier per source block, tp-1 = one barrier for all of them).
GROUP = int(os.environ.get("OKC_SP_CE_AG_GROUP", "1"))
AG_MIN_TOKENS = int(os.environ.get("OKC_SP_CE_AG_MIN_TOKENS", "16384"))
STREAMS = int(os.environ.get("OKC_SP_CE_STREAMS", "1"))
MAX_TOKENS = int(os.environ.get("OKC_SP_CE_MAX_TOKENS", "16384"))

_state = None
_failed = False
_logged = set()
# rsfold: whether the next matmul_rs defers its slot sum, and (out, slots) of the one that did
_defer = False
_deferred = None


@contextlib.contextmanager
def defer_slot_sum(on: bool = True):
    global _defer
    _defer = on
    try:
        yield
    finally:
        _defer = False


def defer_slots(out, slots=None):
    """out stands for `slots` (default: the pending ones) whose sum is never taken (rsfold O1)."""
    global _deferred
    _deferred = (out, _deferred[1] if slots is None else slots)


class RowBlocks:
    """o-proj input as W blocks of R rows made on demand, block(j) -> [R, K] once per j in matmul_rs's order."""

    def __init__(self, W, R, K, dtype, device, block):
        self.shape, self.dtype, self.device, self.block = (W * R, K), dtype, device, block


def take_slots(out):
    """The [W, S, N] slots whose dim-0 sum `out` stands for (a deferred matmul_rs output), else None."""
    global _deferred
    d, _deferred = _deferred, None
    if d is None:
        return None
    if d[0] is out:
        return d[1]
    torch.sum(d[1], dim=0, out=d[0])   # not the caller's tensor: write the sum, nothing reads it unwritten
    return None


class _Buffers:
    """One symmetric allocation per rank: [gather: W*R*H bf16 | receive: W*R*H bf16], rank-major slots of R rows."""

    def __init__(self, buf, group_name: str, world: int, rows: int, hidden: int):
        from torch.distributed import _symmetric_memory as symm_mem

        n = world * rows * hidden
        self.buf = buf
        self.hdl = symm_mem.rendezvous(self.buf, group_name)
        self.rank, self.world = self.hdl.rank, self.hdl.world_size
        if self.world != world:
            raise RuntimeError(f"symm-mem world {self.world} != TP world {world}")
        self.rows, self.hidden = rows, hidden
        self.g = [self.hdl.get_buffer(p, (n,), torch.bfloat16, 0) for p in range(world)]
        self.o = [self.hdl.get_buffer(p, (n,), torch.bfloat16, n) for p in range(world)]
        # high priority: a barrier kernel takes the next free SM ahead of queued GEMM CTAs
        self.streams = [torch.cuda.Stream(priority=-1) for _ in range(max(1, STREAMS))]

    def barrier(self):
        self.hdl.barrier(channel=0)


def _buffers(tp_group, hidden: int, device):
    """Allocate and rendezvous once, outside graph capture (a collective: every rank reaches the first eligible call
    together, in a warmup forward). None if symm-mem is unusable (every rank fails alike)."""
    global _state, _failed
    if _state is not None or _failed:
        return _state
    if torch.cuda.is_current_stream_capturing():
        return None
    from torch.distributed import _symmetric_memory as symm_mem

    from sglang.kernels.ops.muse_spark_v12.okc_patch.ext_coll import agreed

    world = tp_group.world_size
    rows = -(-MAX_TOKENS // world)
    # agreed steps (allocate, then rendezvous): every rank takes the copy-engine path or none does
    buf = agreed(lambda: symm_mem.empty(2 * 2 * world * rows * hidden, dtype=torch.uint8, device=device))
    _state = None if buf is None else agreed(lambda: _Buffers(buf, tp_group.device_group.group_name, world, rows, hidden))
    if _state is None:
        logger.warning("OKC SP copy-engine TP collectives unavailable (symm-mem buffers); using the fused / NCCL path.")
        _failed = True
    else:
        logger.info("OKC SP copy-engine TP collectives: AG barrier every %d steps, %d comm streams, %.0f MiB symmetric.",
                    GROUP, len(_state.streams), _state.buf.numel() / 2**20)
    return _state


def _order(me, W):
    return [(me + s) % W for s in range(1, W)]


def ag_matmul_steps(st, x_shard, weight):
    """Generator (yields the stream that needs a barrier over all ranks; returns [W*S, N]): AG of x_shard [S, K] and
    the GEMM with weight [N, K]. st: rank, world, g (flat views of every rank's gather buffer), streams."""
    S, K = x_shard.shape
    W, me = st.world, st.rank
    N = weight.shape[0]
    wt = weight.t()
    cur = torch.cuda.current_stream()
    X = st.streams[0]
    for s in st.streams:
        s.wait_stream(cur)
    x_shard = x_shard.contiguous()
    out = torch.empty((W * S, N), dtype=x_shard.dtype, device=x_shard.device)
    # the own block from the shard, while the copy engines push it to the peers
    torch.mm(x_shard, wt, out=out[me * S:(me + 1) * S])
    steps = _order(me, W)
    g = max(1, GROUP)
    for g0 in range(0, len(steps), g):
        group = steps[g0:g0 + g]
        for s in st.streams[1:]:
            s.wait_stream(X)
        for j, dest in enumerate(group):
            with torch.cuda.stream(st.streams[j % len(st.streams)]):
                st.g[dest][me * S * K:(me + 1) * S * K].copy_(x_shard.view(-1))
        for s in st.streams[1:]:
            X.wait_stream(s)
        yield X
        ev = torch.cuda.Event()
        ev.record(X)
        cur.wait_event(ev)
        # pushed this group: dest = me + s for its steps s, so the blocks of sources me - s have arrived
        for step in range(g0 + 1, g0 + len(group) + 1):
            src = (me - step) % W
            torch.mm(st.g[me][src * S * K:(src + 1) * S * K].view(S, K), wt, out=out[src * S:(src + 1) * S])
    return out


def _chunk_mm(x, weight, wt, out=None):
    """torch.mm(x, wt) for one matmul_rs chunk; OKC_LT_PREFILL_O: okc_patch.lt's claimed config for the shape, whose
    rows are the same bytes (the call falls back to torch.mm whenever lt declines)."""
    if LT_PREFILL_O:
        from sglang.kernels.ops.muse_spark_v12.okc_patch import lt as _lt

        y = _lt.lt_prefill_mm(x, weight, out)
        if y is not None:
            return y
    return torch.mm(x, wt) if out is None else torch.mm(x, wt, out=out)


def matmul_rs_steps(st, x, weight):
    """Generator (yields the stream that needs a barrier; returns [S, N]): x [W*S, K] @ weight [N, K]^T, reduce-scattered
    over the ranks. st: rank, world, o (flat views of every rank's receive buffer), streams."""
    global _deferred
    M, K = x.shape
    W, me = st.world, st.rank
    S, N = M // W, weight.shape[0]
    wt = weight.t()
    cur = torch.cuda.current_stream()
    X = st.streams[0]
    for s in st.streams:
        s.wait_stream(cur)
    if isinstance(x, RowBlocks):
        rows = x.block
    else:
        x = x.contiguous()
        rows = lambda j: x[j * S:(j + 1) * S]   # noqa: E731
    keep = []
    for j, dest in enumerate(_order(me, W)):
        part = _chunk_mm(rows(dest), weight, wt)
        keep.append(part)
        ev = torch.cuda.Event()
        ev.record(cur)
        s = st.streams[j % len(st.streams)]
        s.wait_event(ev)
        with torch.cuda.stream(s):
            st.o[dest][me * S * N:(me + 1) * S * N].copy_(part.view(-1))
    _chunk_mm(rows(me), weight, wt, out=st.o[me][me * S * N:(me + 1) * S * N].view(S, N))
    ev = torch.cuda.Event()
    ev.record(cur)
    X.wait_event(ev)
    for s in st.streams[1:]:
        X.wait_stream(s)
    yield X
    ev = torch.cuda.Event()
    ev.record(X)
    cur.wait_event(ev)
    slots = st.o[me][:W * S * N].view(W, S, N)
    if _defer:
        out = torch.empty((S, N), dtype=x.dtype, device=x.device)
        _deferred = (out, slots)
    else:
        out = slots.sum(dim=0)
    # the pushed partials stay referenced until the barrier is joined (the copy engines read them)
    del keep
    return out


def _drive(st, gen):
    try:
        while True:
            stream = next(gen)
            with torch.cuda.stream(stream):
                st.barrier()
    except StopIteration as stop:
        return stop.value


def _eligible(st, rows, width, dtype):
    return st is not None and rows <= st.rows and width <= st.hidden and dtype == torch.bfloat16


def ag_matmul(tp_group, x_shard, weight):
    """[S, K] shard -> [W*S, N] = all-gather(x_shard) @ weight^T, or None (keep torch's fused op)."""
    if x_shard.shape[0] * tp_group.world_size < AG_MIN_TOKENS:
        return None
    st = _buffers(tp_group, max(x_shard.shape[1], weight.shape[0]), x_shard.device)
    if not _eligible(st, x_shard.shape[0], x_shard.shape[1], x_shard.dtype) or weight.dtype != torch.bfloat16:
        return None
    if "ag" not in _logged:
        _logged.add("ag")
        logger.info("OKC SP copy-engine all-gather + matmul active (%d-row shard).", x_shard.shape[0])
    return _drive(st, ag_matmul_steps(st, x_shard, weight))


def matmul_rs(tp_group, x, weight):
    """[W*S, K] @ weight [N, K]^T reduce-scattered -> [S, N], or None (keep torch's fused op)."""
    W = tp_group.world_size
    if x.shape[0] % W:
        return None
    st = _buffers(tp_group, max(x.shape[1], weight.shape[0]), x.device)
    if not _eligible(st, x.shape[0] // W, weight.shape[0], x.dtype) or weight.dtype != torch.bfloat16:
        return None
    if "rs" not in _logged:
        _logged.add("rs")
        logger.info("OKC SP copy-engine matmul + reduce-scatter active (%d-row shard).", x.shape[0] // W)
    return _drive(st, matmul_rs_steps(st, x, weight))
