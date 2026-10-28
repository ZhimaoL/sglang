"""OKC: the sequence-parallel prefill MoE section as a k-micro-batch pipeline on the copy engines.

Under OKC Hopper SP (layers/layernorm_sp.py) every rank holds 1/tp of a chunk's tokens. The MoE section
(MuseSpark1xMoE._forward_gathered_experts) all-gathers the pooled rows and their top-k, runs the local experts (pg)
on the whole chunk and reduce-scatters the partial sums back. On NCCL that is ~375 us of AG and ~370 us of RS per
layer on a 16K chunk, all exposed: NCCL's kernels need SMs and pg's persistent grid holds every SM (one 384 x 168
register CTA per SM) for ~2.3 ms, so nothing can run beside it (session SD trace, serving/kernels/moe-overlap).

Here the shard's rows are cut into k micro-batches (the same row range of every rank's shard) that move by
copy-engine P2P pushes into torch symmetric-memory buffers of the peers (the transport of the async-TP linears),
so the bytes move without SMs while pg computes:
    comm stream:  AG(0) | AG(1) ...               | RS(0)              | RS(1)
    main stream:        | pg(0)        | pg(1)    | reduce(0)          | reduce(1)
Each push group is published by a symm-mem barrier (a one-CTA kernel; pg's grid is capped FREE_SMS short of the
SM count while the pipeline runs so the barrier finds an SM). AG: every rank pushes its rows (pre-expert-normed,
the norm is per row) and top-k (fp32 scores and int32 ids in one int32 row) into slot `rank` of every peer's
buffer. RS: every rank pushes the partial outputs of each peer's rows to that peer, which sums the tp slots in fp32
and rounds once to bf16 (NCCL rounds per ring hop): the same expert math, only the summation order differs.

Buffer reuse across layers and forwards is ordered by the data: a peer writes my buffers for the next use only
after it has passed a barrier that I enter after my reads of this use (AG buffers: my RS barrier follows my pg;
RS buffers: the next AG barrier follows my reduce). One set of buffers serves every layer (the layers run in
sequence). Every decision here depends only on the shard shape and the switches, so all ranks agree.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

SPLITS = int(os.environ.get("OKC_MOE_OVL_SPLITS", "2"))
# SMs left out of pg's persistent grid while the pipeline runs (for the comm stream's barrier kernels).
FREE_SMS = int(os.environ.get("OKC_MOE_OVL_FREE_SMS", "1"))
# Comm streams for the pushes of one group (2+: peers in parallel on more copy engines).
STREAMS = int(os.environ.get("OKC_MOE_OVL_STREAMS", "1"))
# Chunk token range that takes the pipeline (a ~1K extend's collectives are latency-bound: nothing to hide).
MIN_TOKENS = int(os.environ.get("OKC_MOE_OVL_MIN_TOKENS", "4096"))
MAX_TOKENS = int(os.environ.get("OKC_MOE_OVL_MAX_TOKENS", "16384"))
# Run the unpooler per micro-batch inside the pipeline (its first slices hide under the last RS push).
UNPOOL = os.environ.get("OKC_MOE_OVL_UNPOOL", "1") == "1"
# Run the pooler per micro-batch inside the pipeline (AG(0) starts after the first slice, beside the router).
POOLER = os.environ.get("OKC_MOE_OVL_POOLER", "1") == "1"
TK = 16   # int32 words of top-k per row: 8 fp32 scores (bit patterns) + 8 int32 expert ids

_state = None
_failed = False
_logged = False


class _Buffers:
    """One symmetric allocation per rank: [x: W*R*H bf16 | rs: W*R*H bf16 | tk: W*R*TK int32], R = max shard rows."""

    def __init__(self, group_name: str, world: int, rows: int, hidden: int, device):
        from torch.distributed import _symmetric_memory as symm_mem

        n = world * rows
        nbytes = 2 * (2 * n * hidden) + 4 * n * TK
        self.buf = symm_mem.empty(nbytes, dtype=torch.uint8, device=device)
        self.hdl = symm_mem.rendezvous(self.buf, group_name)
        self.rank = self.hdl.rank
        self.world = self.hdl.world_size
        if self.world != world:
            raise RuntimeError(f"symm-mem world {self.world} != TP world {world}")
        self.rows, self.hidden = rows, hidden
        nx = n * hidden
        tk_off = (2 * 2 * nx) // 4
        self.x = [self.hdl.get_buffer(p, (nx,), torch.bfloat16, 0) for p in range(world)]
        self.rs = [self.hdl.get_buffer(p, (nx,), torch.bfloat16, nx) for p in range(world)]
        self.tk = [self.hdl.get_buffer(p, (n * TK,), torch.int32, tk_off) for p in range(world)]
        # high priority: a barrier kernel takes the next free SM ahead of queued CTAs of the main stream
        self.streams = [torch.cuda.Stream(priority=-1) for _ in range(max(1, STREAMS))]
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count


def _buffers(tp_group, hidden: int, device):
    """Allocate and rendezvous once (a collective over the TP group: every rank reaches the first eligible SP MoE
    call together, in a warmup forward before graph capture). None if symm-mem is unusable."""
    global _state, _failed
    if _state is not None or _failed:
        return _state
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        world = tp_group.world_size
        rows = -(-MAX_TOKENS // world)
        _state = _Buffers(tp_group.device_group.group_name, world, rows, hidden, device)
        logger.info(
            "OKC SP MoE overlap: %d micro-batches, %d free SMs, %d comm streams, %.0f MiB symmetric buffers.",
            SPLITS, FREE_SMS, len(_state.streams), _state.buf.numel() / 2**20,
        )
    except Exception as e:  # every rank fails the same way (same allocation) -> all keep NCCL
        logger.warning("OKC SP MoE overlap unavailable (symm-mem buffers): %r", e)
        _failed = True
        _state = None
    return _state


def eligible(num_shard_rows: int, world: int) -> bool:
    tokens = num_shard_rows * world
    return SPLITS >= 1 and MIN_TOKENS <= tokens <= MAX_TOKENS and num_shard_rows >= SPLITS


def run(tp_group, x_shard, scores, ids, experts_fn, post=None, rows=None, routed=None):
    """x_shard [S, H] bf16 (pre-expert-normed), scores [S, 8] fp32, ids [S, 8] int32 -> [S, H] bf16 combined
    expert output of this rank's rows, or None (caller keeps the NCCL path). experts_fn(x, scores, ids) runs the
    local experts on W * m gathered rows and returns their [W * m, H] partial sums. post = (fn, width): fn(rows,
    out) writes the [m, width] image of each micro-batch's combined rows into out, and run returns [S, width].
    Pooler split: x_shard = (producer, S) with producer(a, m) -> the [m, H] normed rows a .. a + m on the current
    stream (enqueued per micro-batch, so AG(0) starts after a fraction of the pooler); routed = an event after which
    scores / ids are valid (the router's side stream, not yet joined: the router runs beside the pooler)."""
    global _logged
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe

    producer = None
    if isinstance(x_shard, tuple):
        producer, S = x_shard
        H = rows
    else:
        S, H = x_shard.shape
    st = _buffers(tp_group, H, scores.device)
    if st is None or S > st.rows or H != st.hidden:
        return None
    W, me = st.world, st.rank
    k = SPLITS
    off = [(i * S) // k for i in range(k + 1)]
    order = [(me + j) % W for j in range(1, W + 1)]   # peers first (staggered), own slot last
    cur = torch.cuda.current_stream()
    X = st.streams[0]
    for s in st.streams:
        s.wait_stream(cur)

    def push(dst_views, width, src_rows_of, a, m):
        """Copy src_rows_of(p) ([m, width] rows meant for peer p) into slot `me` of micro-batch (a, m) of every
        peer's dst_views, on the comm streams; X waits for all of them."""
        base = W * a
        for s in st.streams[1:]:
            s.wait_stream(X)
        for j, p in enumerate(order):
            s = st.streams[j % len(st.streams)]
            with torch.cuda.stream(s):
                dst_views[p][(base + me * m) * width:(base + (me + 1) * m) * width].copy_(src_rows_of(p))
        for s in st.streams[1:]:
            X.wait_stream(s)

    xs_all, ev_ag, ev_rs, outs = [], [], [], []
    tk = None
    for i in range(k):
        a, m = off[i], off[i + 1] - off[i]
        if producer is None:
            xs = x_shard[a:a + m]
        else:
            xs = producer(a, m)
            X.wait_stream(cur)
        xs = xs.contiguous().view(-1)
        xs_all.append(xs)
        push(st.x, H, lambda p: xs, a, m)
        if tk is None:
            # top-k rows of the shard as int32 [S, 16] (fp32 score bits | expert ids), packed on the comm stream
            # once the router is done (it may still run beside the first pooler slice)
            if routed is not None:
                X.wait_event(routed)
            with torch.cuda.stream(X):
                tk = torch.cat((scores.float().contiguous().view(torch.int32), ids.to(torch.int32)), dim=1)
        ts = tk[a:a + m].reshape(-1)
        push(st.tk, TK, lambda p: ts, a, m)
        with torch.cuda.stream(X):
            st.hdl.barrier(channel=0)
            ev = torch.cuda.Event()
            ev.record(X)
        ev_ag.append(ev)

    cap = max(1, st.num_sms - FREE_SMS)
    for i in range(k):
        a, m = off[i], off[i + 1] - off[i]
        base = W * a
        cur.wait_event(ev_ag[i])
        full_x = st.x[me][base * H:(base + W * m) * H].view(W * m, H)
        full_tk = st.tk[me][base * TK:(base + W * m) * TK].view(W * m, TK)
        prev = pg_moe.GRID_CAP
        pg_moe.GRID_CAP = cap
        try:
            out = experts_fn(full_x, full_tk[:, :8].contiguous().view(torch.float32), full_tk[:, 8:].contiguous())
        finally:
            pg_moe.GRID_CAP = prev
        out = out.contiguous()
        outs.append(out)
        ev = torch.cuda.Event()
        ev.record(cur)
        X.wait_event(ev)
        flat = out.view(W, m * H)
        push(st.rs, H, lambda p: flat[p], a, m)
        with torch.cuda.stream(X):
            st.hdl.barrier(channel=0)
            ev = torch.cuda.Event()
            ev.record(X)
        ev_rs.append(ev)

    result = torch.empty((S, H if post is None else post[1]), dtype=torch.bfloat16, device=scores.device)
    for i in range(k):
        a, m = off[i], off[i + 1] - off[i]
        base = W * a
        cur.wait_event(ev_rs[i])
        slots = st.rs[me][base * H:(base + W * m) * H].view(W, m, H)
        if post is None:
            torch.sum(slots, dim=0, out=result[a:a + m])
        else:
            post[0](slots.sum(dim=0), result[a:a + m])
    # Every comm-stream op is joined (the last reduce waited for the last barrier, which followed every push and the
    # router event); outs / tk / the pushed rows stay referenced until here, so their blocks are not handed out
    # while the copy engines still read them.
    del outs, tk, xs_all
    if not _logged:
        _logged = True
        logger.info("OKC SP MoE overlap active: %d-row shard, %d micro-batches%s.", S, k,
                    ", pooler split" if producer is not None else "")
    return result
