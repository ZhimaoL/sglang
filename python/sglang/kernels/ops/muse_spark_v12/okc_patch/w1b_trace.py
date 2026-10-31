"""OKC W1b timing trace for profiled P1-SS passes (QA-P1SS §3; docs/serving/review-1031/W1B-ORDER-A2.md).

OKC_W1B_TRACE names P1-SS pass ordinals of the boot ("3" or "3,9"); the executor calls pass_begin() / pass_end() around
each pass, and a named pass sets w1b.TRACE to a Trace, so each of w1b's trace points records a timing event. Each
consumer wait group also starts with a compute-stream wait on a host counter that the host releases only after the
whole group is enqueued (the host-ahead guard), so a group's span is GPU time (flag waits), never host enqueue time.
In a GPU-bound stretch the counter is released before the stream gets there; in a host-bound one the stream would have
idled for the same enqueue time anyway. Eager passes only: P1-SS never runs captured. Never on in a served pass.

Per gated push (b(j)), on this rank's clock: slack = arrival at M_b(j)'s first wait - the push's last peer rows done;
excess = the two wait groups' span minus this pass's floor (the minimum over all its consumers); gate lag = push start
- gate record (>= 0: the push waited for a(j+1)'s qkv GEMM); CE busy = last rows done - push start."""

import json
import logging
import os

import torch

from . import w1b

logger = logging.getLogger(__name__)

PASSES = frozenset(int(x) for x in os.environ.get("OKC_W1B_TRACE", "").split(",") if x.strip())
OUT_DIR = os.environ.get("OKC_W1B_TRACE_DIR", "/tmp")
_N = [0]


class Trace:
    def __init__(self):
        from cuda.bindings import driver as cu

        self.cu = cu
        self.recs = []                                    # (tag, par, event) in host order
        self.hc = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.hv = self.hc.numpy()
        self.addr = cu.CUdeviceptr(self.hc.data_ptr())    # pinned host memory is device-mapped under UVA
        self.k = 0
        self.geq = int(cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ)
        self._cs = {}

    def _ev(self, tag, stream, par):
        e = torch.cuda.Event(enable_timing=True)
        e.record(stream)
        self.recs.append((tag, par, e))

    def __call__(self, tag, stream, par):
        if tag in ("words0", "rows0"):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("W1b trace: eager passes only")
            self._ev("a" + tag[0], stream, par)          # the stream reaches the group's first wait
            self.k += 1
            cs = self._cs.get(stream.cuda_stream)
            if cs is None:
                cs = self._cs[stream.cuda_stream] = self.cu.CUstream(stream.cuda_stream)
            w1b._ok(self.cu, self.cu.cuStreamWaitValue32(cs, self.addr, self.k, self.geq))
        self._ev(tag, stream, par)
        if tag in ("words1", "rows1"):
            self.release()

    def release(self):
        """Let every enqueued host-ahead wait pass (also on an aborted pass: no stream stays blocked)."""
        self.hv[0] = self.k

    def times(self):
        """[(tag, par, us since the first event)] after a synchronize."""
        torch.cuda.synchronize()
        if not self.recs:
            return []
        t0 = self.recs[0][2]
        return [(tag, par, round(t0.elapsed_time(e) * 1e3, 2)) for tag, par, e in self.recs]


def summarize(T):
    """Pair each consumer with its slot's push and stage (FIFO per slot) -> (uses, pass summary)."""
    pushes, stages, cons, gate = {}, {}, [], None
    cur_p, cur_c = {}, {}
    for tag, par, t in T:
        if tag == "gate":
            gate = t
        elif tag in ("go", "go_g"):
            d = {"par": par, "gated": tag == "go_g", "go": t, "gate": gate if tag == "go_g" else None}
            pushes.setdefault(par, []).append(d)
            cur_p[par] = d
        elif tag[0] in "wr" and tag[1:].isdigit():
            cur_p[par][tag] = t
        elif tag == "side0":
            stages.setdefault(par, []).append({"side0": t})
        elif tag == "side1":
            stages[par][-1]["side1"] = t
        elif tag in ("aw", "ar"):
            if tag == "aw":
                cur_c[par] = {"par": par}
                cons.append(cur_c[par])
            cur_c[par][tag] = t
        elif tag in ("words0", "words1", "rows0", "rows1"):
            cur_c[par][tag] = t
    k = {}
    uses = []
    for c in cons:
        par = c["par"]
        i = k[par] = k.get(par, -1) + 1
        p = pushes[par][i]
        s = stages[par][i]
        rl = max((int(n[1:]) for n in p if n[0] == "r" and n[1:].isdigit()), default=None)
        r_last = p["r%d" % rl] if rl else None
        u = {"par": par, "gated": p["gated"],
             "slack": None if r_last is None else round(c["aw"] - r_last, 2),
             "slack_rows": None if r_last is None else round(c["ar"] - r_last, 2),
             "span_w": round(c["words1"] - c["words0"], 2), "span_r": round(c["rows1"] - c["rows0"], 2),
             "gate_lag": None if p["gate"] is None else round(p["go"] - p["gate"], 2),
             "busy": None if r_last is None else round(r_last - p["go"], 2),
             "own": round(s["side1"] - s["side0"], 2)}
        uses.append(u)
    if not uses:
        return uses, None
    fw = min(u["span_w"] for u in uses)
    fr = min(u["span_r"] for u in uses)
    for u in uses:
        u["excess"] = round(u["span_w"] - fw + u["span_r"] - fr, 2)
    g = [u for u in uses if u["gated"]]
    sl = [u["slack"] for u in g if u["slack"] is not None]
    s = {"uses": len(uses), "gated": len(g), "floor_w": fw, "floor_r": fr,
         "min_slack": min(sl) if sl else None,
         "min_slack_at": [u["slack"] for u in g].index(min(sl)) if sl else None,   # gated-use ordinal (b(j): j)
         "max_excess_gated": max((u["excess"] for u in g), default=None),
         "min_gate_lag": min((u["gate_lag"] for u in g if u["gate_lag"] is not None), default=None),
         "ce_busy": round(sum(u["busy"] or 0 for u in uses), 2),
         "exposed": round(sum(u["excess"] for u in uses), 2),
         "own_max": max(u["own"] for u in uses)}
    s["rows_before_first_wait"] = bool(sl) and len(sl) == len(g) and min(sl) >= 0 and s["max_excess_gated"] <= 10.0
    return uses, s


def pass_begin():
    """The executor, at each P1-SS pass start: a Trace if OKC_W1B_TRACE names this pass, else None."""
    if not PASSES:
        return None
    _N[0] += 1
    if _N[0] not in PASSES:
        return None
    tr = Trace()
    w1b.TRACE = tr
    return tr


def pass_end(tr, ok=True):
    """The executor, after the pass (also on its error path): detach, release, and on success dump this rank's events
    and the summary (one synchronize)."""
    w1b.TRACE = None
    tr.release()
    if not ok:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    T = tr.times()
    uses, s = summarize(T)
    path = os.path.join(OUT_DIR, f"w1b_trace_p{_N[0]}_r{rank}.json")
    with open(path, "w") as f:
        json.dump({"pass": _N[0], "rank": rank, "events": T, "uses": uses, "summary": s}, f)
    logger.info("OKC W1b trace pass %d rank %d: %s (%s)", _N[0], rank, json.dumps(s), path)
