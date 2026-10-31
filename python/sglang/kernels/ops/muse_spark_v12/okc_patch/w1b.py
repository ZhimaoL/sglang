"""OKC W1b: the SP MoE section's gathered inputs (rows [S, 4096] bf16 + router words [S, 16] int32) as zero-SM copy-engine
pushes with memop readiness flags (docs/serving/review-1031/W1B-ORDER.md; STRUCTURAL §3.1a "W1b after K2b2").

Each rank pushes its shard into its own slot of every peer's parity buffer, on a dedicated push stream: words first (pg_prep
needs only them), then the rows, each followed by a stream-memop flag write into the peer's pad. The consumer waits the
flags with stream memops on the compute stream, so nothing on the comm path needs an SM (no barrier kernel, no NCCL; K1).
Flags follow x-agflags' Flags helper: cuStreamWriteValue32 DEFAULT (the writer's memory barrier orders the copies before
the flag), cuStreamWaitValue32 EQ, reset to 0 on the consumer's stream. Captured memops keep constant values, so 1 / 0.

Order (W1B-ORDER §2-§4): every produce is enqueued before any wait that depends on it; the own block is a same-device copy
on a side compute stream (never the push stream: a same-device copy is an SM kernel and would gate the peer copies behind
it); the pushes read the caller's x, which is held until the next same-parity produce, which first waits the parity's
push-done event (the own-source fence; also what keeps the own slots safe without the qkv barrier).
"""

import logging

import torch

logger = logging.getLogger(__name__)

_IDS, _ROWS = 0, 1


def _ok(cu, res):
    if res[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {res[0]}")
    return res[1] if len(res) > 1 else None


class Flags:
    """x-agflags' memop helper. pads[p]: rank p's int32 pad [P][2][W] as mapped here; flag (par, kind, src) of rank r is
    written by rank src after its pushes into r and waited / reset by r."""

    def __init__(self, pads, me, world, parities):
        from cuda.bindings import driver as cu

        self.cu, self.me, self.W, self.P = cu, me, world, parities
        self.pads = pads
        self.write_flags = cu.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT
        self.wait_flags = int(cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ)
        dev = _ok(cu, cu.cuCtxGetDevice())
        if _ok(cu, cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, dev)):
            self.wait_flags |= int(cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_FLUSH)
        self._cs = {}

    def _stream(self, stream):
        cs = self._cs.get(stream.cuda_stream)
        if cs is None:
            cs = self._cs[stream.cuda_stream] = self.cu.CUstream(stream.cuda_stream)
        return cs

    def _addr(self, rank, par, kind, src):
        return self.cu.CUdeviceptr(self.pads[rank] + 4 * ((par * 2 + kind) * self.W + src))

    def signal(self, stream, dest, par, kind):
        """On `stream`, after its copies: dest's flag (par, kind, me) = 1."""
        _ok(self.cu, self.cu.cuStreamWriteValue32(self._stream(stream), self._addr(dest, par, kind, self.me), 1,
                                                  self.write_flags))

    def wait(self, stream, src, par, kind):
        """On `stream`: wait until my flag (par, kind, src) == 1, then reset it to 0."""
        cs = self._stream(stream)
        a = self._addr(self.me, par, kind, src)
        _ok(self.cu, self.cu.cuStreamWaitValue32(cs, a, 1, self.wait_flags))
        _ok(self.cu, self.cu.cuStreamWriteValue32(cs, a, 0, self.write_flags))


class State:
    """One symmetric allocation per rank: per parity [rows W x S_max x H bf16 | words W x S_max x 16 int32], then the flag
    pad [P][2][W] int32; the push stream and side compute stream; per-parity events and the held x."""

    def __init__(self, hdl, buf, world, rank, s_max, hidden, parities):
        self.hdl, self.buf = hdl, buf
        self.W, self.me, self.S_max, self.H, self.P = world, rank, s_max, hidden, parities
        rows_b = world * s_max * hidden * 2
        words_b = world * s_max * 16 * 4
        self.per_par = rows_b + words_b
        pad_off = parities * self.per_par
        # get_buffer's offset counts elements of the view's dtype, not bytes
        self.rows = [[hdl.get_buffer(p, (world * s_max, hidden), torch.bfloat16, par * self.per_par // 2)
                      for p in range(world)] for par in range(parities)]
        self.words = [[hdl.get_buffer(p, (world * s_max, 16), torch.int32, (par * self.per_par + rows_b) // 4)
                       for p in range(world)] for par in range(parities)]
        pads = [hdl.get_buffer(p, (parities * 2 * world,), torch.int32, pad_off // 4).data_ptr() for p in range(world)]
        self.flags = Flags(pads, rank, world, parities)
        self.push = torch.cuda.Stream(priority=-1)
        self.side = torch.cuda.Stream()
        self.ce_done = [None] * parities
        self.own_done = [None] * parities
        self.held = [None] * parities
        self.pending = [None] * parities


def nbytes(world, s_max, hidden, parities):
    return parities * (world * s_max * hidden * 2 + world * s_max * 16 * 4) + parities * 2 * world * 4


def setup(group_name, world, rank, device, s_max, hidden=4096, parities=2, agreed=None):
    """Allocate, zero and rendezvous (outside capture; every rank together). agreed: ext_coll.agreed in the server, so a
    failure on any rank leaves every rank on the old path; None runs the steps directly (tests)."""
    from torch.distributed import _symmetric_memory as symm_mem

    run = agreed or (lambda step: step())
    n = nbytes(world, s_max, hidden, parities)

    def alloc():
        buf = symm_mem.empty(n, dtype=torch.uint8, device=device)
        buf.zero_()
        torch.cuda.synchronize(device)   # the pads are zero before any peer returns from the rendezvous
        return buf

    buf = run(alloc)
    if buf is None:
        return None
    hdl = run(lambda: symm_mem.rendezvous(buf, group_name))
    if hdl is None:
        return None
    if hdl.world_size != world or hdl.rank != rank:
        raise RuntimeError(f"W1b: symm-mem world / rank {hdl.world_size} / {hdl.rank} != {world} / {rank}")
    st = State(hdl, buf, world, rank, s_max, hidden, parities)
    logger.info("OKC W1b push transport active (%d parities, S_max %d, %.0f MiB symmetric).", parities, s_max, n / 2**20)
    return st


# The own-source fence (QA-P1SS §3 b): the next same-parity stage waits this parity's push-done event before it writes
# the own slots or drops x. Off = rely on the transitive argument (every peer's M, which consumed my flags, precedes the
# next A's qkv barrier on every rank); the unit runs both. join_push (required in graph form) discharges it either way.
GUARD = True

# Timing trace for one profiled pass (QA-P1SS §3; W1B-ORDER-A2; okc_patch/w1b_trace.py): None = off. Otherwise a callable
# (tag, stream, par) run at each trace point: the gate record ("gate"), the push stream after its waits ("go" / "go_g"
# gated) and after each peer's words ("w1".."w7") and rows ("r1".."r7"), the side stream around the own copy ("side0" /
# "side1"), the compute stream around the consumer's waits ("words0" / "words1", "rows0" / "rows1"). Never on in a
# served pass.
TRACE = None


class Gate:
    """The push-window gate (STRUCTURAL §3.1a amendment 3; QA-P1SS §3 c): an event plus a generation counter. A stream wait
    binds to the event's most recent record at call time, so push() asserts the generation its caller expects: a gate
    recorded after the push was enqueued (or not at all) fails loudly instead of silently not gating."""

    def __init__(self):
        self.ev = torch.cuda.Event()
        self.gen = 0

    def record(self, stream=None):
        stream = stream or torch.cuda.current_stream()
        self.ev.record(stream)
        self.gen += 1
        if TRACE is not None:
            TRACE("gate", stream, None)
        return self.gen


def stage(st, x, scores, ids, par):
    """Compute side of A_x(j)'s produce, on the compute stream: x [S, H] bf16 (the shard prenorm), scores [S, 8] fp32,
    ids [S, 8] int32. Words into my own slot (cat(out=)), the own rows by a side-stream copy (never on the push stream: a
    same-device copy is an SM kernel). push() enqueues the peer copies; until then they are pending."""
    if st.pending[par] is not None:
        # a second same-slot stage before its push would drop the first child's pushes and M would hang (QA-P1SS §3)
        raise RuntimeError(f"W1b: slot {par} is staged and not pushed; push() or discard() it first")
    S, me = x.shape[0], st.me
    cur = torch.cuda.current_stream()
    if GUARD and st.ce_done[par] is not None:     # the own-source fence, unless join_push already took it
        cur.wait_event(st.ce_done[par])
    words = st.words[par][me][me * S:(me + 1) * S]
    torch.cat([scores.contiguous().view(torch.int32), ids.to(torch.int32).contiguous()], dim=1, out=words)
    x = x.contiguous()
    ready = torch.cuda.Event()
    ready.record(cur)
    st.side.wait_event(ready)
    if TRACE is not None:
        TRACE("side0", st.side, par)
    with torch.cuda.stream(st.side):
        st.rows[par][me][me * S:(me + 1) * S].copy_(x)
    own = torch.cuda.Event()
    own.record(st.side)
    if TRACE is not None:
        TRACE("side1", st.side, par)
    st.own_done[par] = own
    st.held[par] = (x, scores, ids)               # the push stream reads x; dropped after the next same-parity fence
    st.pending[par] = (S, words, x, ready)


def push(st, par, gate=None, gate_gen=None):
    """Enqueue the staged words + rows into every peer's slot `me` on the push stream, each followed by its flag, after
    the stage's ready event and, when gated, after `gate` (a Gate whose generation must equal gate_gen: the record the
    caller made for this push). Must be called before any wait that depends on these pushes (host order; QA a)."""
    if gate is not None and gate.gen != gate_gen:
        raise RuntimeError(f"W1b gate generation {gate.gen} != expected {gate_gen}: the wait would bind to another record")
    S, words, x, ready = st.pending[par]
    st.pending[par] = None
    st.push.wait_event(ready)
    if gate is not None:
        st.push.wait_event(gate.ev)
    W, me, fl = st.W, st.me, st.flags
    if TRACE is not None:
        TRACE("go_g" if gate is not None else "go", st.push, par)
    with torch.cuda.stream(st.push):
        for s in range(1, W):
            p = (me + s) % W
            st.words[par][p][me * S:(me + 1) * S].copy_(words)
            fl.signal(st.push, p, par, _IDS)
            if TRACE is not None:
                TRACE(f"w{s}", st.push, par)
            st.rows[par][p][me * S:(me + 1) * S].copy_(x)
            fl.signal(st.push, p, par, _ROWS)
            if TRACE is not None:
                TRACE(f"r{s}", st.push, par)
    done = torch.cuda.Event()
    done.record(st.push)
    st.ce_done[par] = done


def discard(st, par):
    """Drop a staged, never-pushed slot (tests; an aborted pass). Its own-slot writes stay; nothing was sent."""
    st.pending[par] = None


def produce(st, x, scores, ids, par, gate=None, gate_gen=None):
    """stage + push in one step (chunk a, single-chunk forwards, and P1-SS's "ungated" switch)."""
    stage(st, x, scores, ids, par)
    push(st, par, gate, gate_gen)


def wait_words(st, par, S):
    """Compute stream: the 7 peers' words are in -> (scores [W S, 8] fp32, ids [W S, 8] int32), contiguous copies (as
    W1a-P's ids gather returns; pg_prep / pg_route take no strides): ~1 MiB of local copy at S 2048 (QA-P1SS n3)."""
    cur = torch.cuda.current_stream()
    if TRACE is not None:
        TRACE("words0", cur, par)
    for s in range(1, st.W):
        st.flags.wait(cur, (st.me - s) % st.W, par, _IDS)
    if TRACE is not None:
        TRACE("words1", cur, par)
    w = st.words[par][st.me][: st.W * S]
    return w[:, :8].contiguous().view(torch.float32), w[:, 8:].contiguous()


def wait_rows(st, par, S):
    """Compute stream: the 7 peers' rows and my own block are in -> rows [W S, H] bf16, a view of my buffer."""
    cur = torch.cuda.current_stream()
    if TRACE is not None:
        TRACE("rows0", cur, par)
    for s in range(1, st.W):
        st.flags.wait(cur, (st.me - s) % st.W, par, _ROWS)
    cur.wait_event(st.own_done[par])
    if TRACE is not None:
        TRACE("rows1", cur, par)
    return st.rows[par][st.me][: st.W * S]


def join_push(st, par):
    """Compute stream: my parity-`par` pushes are done. Call it before the graph segment that issued them ends (a captured
    fork must rejoin its origin stream), at the end of the MoE section; it costs nothing once the pushes have drained.
    It also discharges the next same-parity produce's fence (an event from an earlier capture cannot be waited in a
    later one)."""
    if st.ce_done[par] is not None:
        torch.cuda.current_stream().wait_event(st.ce_done[par])
        st.ce_done[par] = None
