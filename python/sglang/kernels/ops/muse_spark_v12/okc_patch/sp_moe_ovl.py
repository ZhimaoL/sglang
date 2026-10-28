"""OKC: the sequence-parallel prefill MoE block with its collectives on the copy engines, sliced at the edges.

Under OKC Hopper SP (layers/layernorm_sp.py) every rank holds 1/tp of a chunk's tokens. The MoE block pools its shard
(pooler GEMM), all-gathers the pooled rows and their top-k, runs the local experts (pg) on the whole chunk,
reduce-scatters the partial sums back and unpools (MuseSpark1xMoE._forward_gathered_experts). On NCCL a 16K chunk
spends ~410 us in the AGs and ~370 us in the RS per layer, all exposed (session SD traces, serving/kernels/moe-overlap).

pg itself is not split: every extra token micro-batch adds a partial 256-row block per expert, and pg re-streams a
weight tile per block (+0.25..+0.4 ms per 16K section for two micro-batches, about what they would hide). Only the
per-row work around it is sliced into k row ranges of the shard (the same ranges on every rank):
    main stream:  pooler(0) pooler(1) .. | pg (whole chunk) | fin(0) fin(1) .. | reduce+unpool(0) .. reduce+unpool(k-1)
    comm stream:        AG(0)   AG(1) .. |                  |   RS(0)   RS(1) ..
The transfers are copy-engine P2P pushes (Tensor.copy_ into the peers' torch symmetric-memory buffers, the transport
of the async-TP linears): no SMs, so they run under the pooler / finalize / unpooler slices. AG: every rank pushes
each slice of its pooled, pre-expert-normed rows (the norm is per row) into slot `rank` of every peer's buffer, and
its top-k once (fp32 scores and int32 ids as one int32 row) when the router's side stream is done; one symm-mem
barrier (a one-CTA kernel) publishes them before pg. RS: per slice, every rank pushes the finalized partial rows of
each peer's tokens to that peer, a barrier publishes them, and the peer sums the tp slots in fp32 and rounds once to
bf16 (NCCL's ring rounds per hop): the same expert math, only the summation order differs.

Buffer reuse across layers and forwards is ordered by the data: a peer writes my buffers for the next use only after
it has passed a barrier that I enter after my reads of this use (x / top-k: my first RS barrier follows my pg; RS
slots: the next AG barrier follows my reduces). One set of buffers serves every layer (the layers run in sequence).
Every branch depends only on the shard shape and the switches, so all ranks agree.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# Row slices of the shard at the edges (pooler / AG pushes, finalize / RS pushes / reduce / unpooler).
SPLITS = int(os.environ.get("OKC_MOE_OVL_SPLITS", "4"))
# Comm streams for the pushes of one slice (2+: peers in parallel on more copy engines).
STREAMS = int(os.environ.get("OKC_MOE_OVL_STREAMS", "2"))
# Chunk token range that takes the pipeline (a ~1K extend's collectives are latency-bound: nothing to hide).
MIN_TOKENS = int(os.environ.get("OKC_MOE_OVL_MIN_TOKENS", "4096"))
MAX_TOKENS = int(os.environ.get("OKC_MOE_OVL_MAX_TOKENS", "16384"))
# Pooler / unpooler per slice inside the pipeline (else: whole-shard GEMMs before / after it).
POOLER = os.environ.get("OKC_MOE_OVL_POOLER", "1") == "1"
UNPOOL = os.environ.get("OKC_MOE_OVL_UNPOOL", "1") == "1"
TK = 16   # int32 words of top-k per row: 8 fp32 scores (bit patterns) + 8 int32 expert ids

_state = None
_failed = False
_logged = False


class _Buffers:
    """One symmetric allocation per rank: [x: W*R*H bf16 | rs: W*R*H bf16 | tk: W*R*TK int32], R = max shard rows.
    x and rs are slice-major (slice i holds W blocks of its m_i rows, block r = rank r's), tk is rank-major."""

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
            "OKC SP MoE overlap: %d row slices, %d comm streams, %.0f MiB symmetric buffers.",
            SPLITS, len(_state.streams), _state.buf.numel() / 2**20,
        )
    except Exception as e:  # every rank fails the same way (same allocation) -> all keep NCCL
        logger.warning("OKC SP MoE overlap unavailable (symm-mem buffers): %r", e)
        _failed = True
        _state = None
    return _state


def eligible(num_shard_rows: int, world: int) -> bool:
    tokens = num_shard_rows * world
    return SPLITS >= 1 and MIN_TOKENS <= tokens <= MAX_TOKENS and num_shard_rows >= SPLITS


def pipeline(st, x_shard, scores, ids, pg_fn, fin_fn, post=None, rows=None, routed=None):
    """Generator: enqueues the block on the current stream and st.streams, yielding where the comm stream
    (st.streams[0], current inside the yield) needs a barrier over all ranks; returns the [S, H] combined rows (or
    [S, post width] with post). st: rank, world, rows, x / rs / tk (flat views of every rank's buffers), streams.
      x_shard   [S, H] bf16 pre-expert-normed rows, or (producer, S): producer(a, m) -> rows a .. a + m, normed
      scores    [S, 8] fp32, ids [S, 8] int32 (valid on the current stream, or after the event `routed`)
      pg_fn(x [W*S, H], ids [W*S, 8]) -> (y [W*S, 8, H] per-route outputs, local slot ids [W*S, 8])
      fin_fn(y rows, scores rows, local rows) -> [n, H] partial sums of those rows
      post = (fn, width): fn(rows [m, H], out [m, width]) per slice (the unpooler)"""
    producer = None
    if isinstance(x_shard, tuple):
        producer, S = x_shard
        H = rows
    else:
        S, H = x_shard.shape
    W, me, k = st.world, st.rank, SPLITS
    off = [(i * S) // k for i in range(k + 1)]
    sl = [(off[i], off[i + 1] - off[i]) for i in range(k)]
    order = [(me + j) % W for j in range(1, W + 1)]   # peers first (staggered), own slot last
    cur = torch.cuda.current_stream()
    X = st.streams[0]
    for s in st.streams:
        s.wait_stream(cur)

    def push(dst_views, width, src_rows_of, at, m):
        """Copy src_rows_of(p) (m rows of `width` meant for peer p) to element offset (at + me * m) * width of every
        peer's dst_views, spread over the comm streams; X then waits for all of them."""
        for s in st.streams[1:]:
            s.wait_stream(X)
        for j, p in enumerate(order):
            with torch.cuda.stream(st.streams[j % len(st.streams)]):
                dst_views[p][(at + me * m) * width:(at + (me + 1) * m) * width].copy_(src_rows_of(p))
        for s in st.streams[1:]:
            X.wait_stream(s)

    # AG: pooled rows per slice (slice-major), then the top-k once (rank-major), one barrier
    keep = []
    for a, m in sl:
        xs = x_shard[a:a + m] if producer is None else producer(a, m)
        if producer is not None:
            X.wait_stream(cur)
        xs = xs.contiguous().view(-1)
        keep.append(xs)
        push(st.x, H, lambda p: xs, W * a, m)
    if routed is not None:
        X.wait_event(routed)
    with torch.cuda.stream(X):
        tk = torch.cat((scores.float().contiguous().view(torch.int32), ids.to(torch.int32)), dim=1).view(-1)
    keep.append(tk)
    push(st.tk, TK, lambda p: tk, 0, S)
    with torch.cuda.stream(X):
        yield
    ev = torch.cuda.Event()
    ev.record(X)
    cur.wait_event(ev)

    # pg on the whole chunk; top-k reordered to the slice-major row order of x
    tk_all = st.tk[me][:W * S * TK].view(W, S, TK)
    tk_sm = torch.cat([tk_all[:, a:a + m].reshape(W * m, TK) for a, m in sl]) if k > 1 else tk_all.view(W * S, TK)
    g_scores = tk_sm[:, :8].contiguous().view(torch.float32)
    g_ids = tk_sm[:, 8:].contiguous()
    full_x = st.x[me][:W * S * H].view(W * S, H)
    y, local = pg_fn(full_x, g_ids)

    # RS per slice: finalize the slice's rows (W blocks of m), push block p to peer p, barrier
    ev_rs = []
    for a, m in sl:
        r0, r1 = W * a, W * (a + m)
        out = fin_fn(y[r0:r1], g_scores[r0:r1], local[r0:r1]).contiguous()
        keep.append(out)
        ev = torch.cuda.Event()
        ev.record(cur)
        X.wait_event(ev)
        flat = out.view(W, m * H)
        push(st.rs, H, lambda p: flat[p], W * a, m)
        with torch.cuda.stream(X):
            yield
        ev = torch.cuda.Event()
        ev.record(X)
        ev_rs.append(ev)

    result = torch.empty((S, H if post is None else post[1]), dtype=torch.bfloat16, device=scores.device)
    for (a, m), ev in zip(sl, ev_rs):
        cur.wait_event(ev)
        slots = st.rs[me][W * a * H:W * (a + m) * H].view(W, m, H)
        if post is None:
            torch.sum(slots, dim=0, out=result[a:a + m])
        else:
            post[0](slots.sum(dim=0), result[a:a + m])
    # Every comm-stream op is joined (the last reduce waited for the last barrier, which followed every push and the
    # router event); the pushed tensors stay referenced until here, so their blocks are not handed out while the
    # copy engines still read them.
    del keep, y, local
    return result


def run(tp_group, x_shard, scores, ids, pg_fn, fin_fn, post=None, rows=None, routed=None):
    """The block on the TP group's symmetric buffers (see pipeline); None = keep the NCCL path."""
    global _logged
    S = x_shard[1] if isinstance(x_shard, tuple) else x_shard.shape[0]
    H = rows if isinstance(x_shard, tuple) else x_shard.shape[1]
    st = _buffers(tp_group, H, scores.device)
    if st is None or S > st.rows or H != st.hidden:
        return None
    gen = pipeline(st, x_shard, scores, ids, pg_fn, fin_fn, post=post, rows=rows, routed=routed)
    try:
        while True:
            next(gen)
            with torch.cuda.stream(st.streams[0]):
                st.hdl.barrier(channel=0)
    except StopIteration as stop:
        result = stop.value
    if not _logged:
        _logged = True
        logger.info("OKC SP MoE overlap active: %d-row shard, %d slices%s.", S, SPLITS,
                    ", pooler sliced" if isinstance(x_shard, tuple) else "")
    return result
