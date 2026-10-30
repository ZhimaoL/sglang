"""OKC runtime-adaptive replicas (OKC_MOE_ADAPTIVE_REPLICAS, default 1; lane-moe P2-5, muse-opt
serving/kernels/moe-graft LOG "P2-5"). Homes stay as placed (contiguous by default); which experts get the R replica slots per rank is
decided from the run's OWN routing counts and refreshed between forwards.

Counting (graph-safe, every rank in step): per pg layer with replicas, a verify counter [256] (route_k adds its picks
when given it: T <= 256, rank 0 only) and a ring of the last ADAPT_RING prefill / extend forwards' counts [K, 256]
(T > 256: every rank under SP, whose ids are its own shard's; rank 0 only otherwise), advanced by a device-side slot.
Refresh (ModelRunner.forward entry, target worker, never while capturing): when ADAPT_FIRST_TOKENS prefill tokens and
ADAPT_FIRST_VERIFY verify tokens (then ADAPT_EVERY_TOKENS prefill tokens) have been forwarded since the last one, every
rank all-reduces the counters (so all
hold the exact global counts), halves its verify counters, and decides its share of the layers (layer i on rank i %
world) in a background process with decide() below: placement_joint's objective greedy on those counts (verify
stragglers, prefill straggler, extend; homes fixed, R per rank, never on the home rank). Every ADAPT_CHECK forwards
the ranks all-gather readiness and, once all are done, the decisions; then apply(): each new (expert, rank) gets the
expert's pg packs by NCCL send / recv from its home rank straight into a free replica slot and its global scale from
an all-gathered table; the replica table is rebuilt and copied in place (captured graphs keep their addresses). A
replica slot is a byte copy of the home's packs, so every expert's output is bit-identical on any holder.
"""
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

EP, NE, TOPK = 8, 256, 8
C0, C1 = 13.5, 7.6                       # st cost: c0 + c1 x passes (moe-int ep_balance_sim)
BM, NT = 256, 96
FT_MS = 2109.0
STRAG_CHUNKS = 4.9                       # 16K forwards per first turn (joint_search's calibration on session M)
W_X = 0.217                              # % official per ms per extend
RUNGS = {"c4": (0.37, 0.45, 3.25, 0.28, 0.11, 0.7), "c16": (0.53, 0.26, 13.3, 0.27, 0.11, 0.5),
         "c32": (0.59, 0.20, 20.9, 0.45, 0.0, 0.0)}


def _step_ms(bs):
    return float(np.interp(bs, [1, 4, 16, 32], [9.77, 14.66, 24.5, 31.0]))


W_P = sum(100 * w * F / FT_MS for F, _, _, w, _, _ in RUNGS.values())
_W_RUNG = {r: 100 * (w * V + wt * k) / _step_ms(bs) for r, (F, V, bs, w, wt, k) in RUNGS.items()}
GROUPS = [(3, _W_RUNG["c4"] / 2, "passes", True), (4, _W_RUNG["c4"] / 2, "passes", True),
          (13, _W_RUNG["c16"], "distinct", False), (21, _W_RUNG["c32"], "distinct", False)]


def route_counts(p, bs, beta, F, rng):
    out = np.zeros((F, NE), int)
    for f in range(F):
        for _ in range(bs):
            q = rng.dirichlet(beta * p * NE) if beta > 0 else p
            g = rng.gumbel(size=(8, NE)) + np.log(q + 1e-30)[None, :]
            np.add.at(out[f], np.argpartition(-g, TOPK, axis=-1)[:, :TOPK].reshape(-1), 1)
    return out


def rank_costs(cnt, home, pairs, metric, cost_passes):
    F = cnt.shape[0]
    passes = (cnt + 7) // 8
    work = passes if cost_passes else (cnt > 0).astype(int)
    lw = passes if metric == "passes" else (cnt > 0).astype(int)
    nonrep = np.ones(NE, bool)
    for e, _ in pairs:
        nonrep[e] = False
    oh = np.zeros((NE, EP)); oh[np.arange(NE), home] = 1
    load = (lw * nonrep) @ oh
    cost = (work * nonrep) @ oh
    rows = np.arange(F)
    for e, r in pairs:
        h = home[e]
        a = lw[:, e] > 0
        pick = np.where(load[:, r] < load[:, h], r, h)
        load[rows[a], pick[a]] += lw[a, e]
        cost[rows[a], pick[a]] += work[a, e]
    return np.where(cost > 0, C0 + C1 * cost, 0.0)


def _block_us(n):
    n = np.asarray(n, dtype=np.float64)
    full = np.floor(n / BM)
    tail = n - full * BM
    tail_w = np.where(tail <= 0, 0, np.where(tail <= 64, 64, np.where(tail <= 128, 128, 256)))
    item = lambda w: np.where(w > 0, 8 + 0.125 * w, 0.0)
    return 96 * (full * item(256) + item(tail_w))


def strag_us(c, home, pairs):
    po = sorted(pairs, key=lambda x: -c[:, x[0]].sum())
    F = c.shape[0]
    rep = {e for e, _ in po}
    fixed = np.array([e not in rep for e in range(NE)])
    oh = np.zeros((NE, EP)); oh[np.arange(NE), home] = 1
    tok = (c * fixed) @ oh
    pg = (_block_us(c) * fixed) @ oh
    rows = np.arange(F)
    for e, r in po:
        holders = [home[e], r]
        left = c[:, e].astype(np.float64).copy()
        share = np.zeros((F, 2))
        base = tok[:, holders].copy()
        while True:
            act = left > 0
            if not act.any():
                break
            j = np.argmin(base + share, axis=1)
            give = np.minimum(left, BM)
            share[rows[act], j[act]] += give[act]
            left -= np.where(act, give, 0)
        for k, h in enumerate(holders):
            tok[:, h] += share[:, k]
            pg[:, h] += _block_us(share[:, k])
    pg = pg / 131.0
    return (pg.max(1) - pg.mean(1)).mean()


def _pg_us(n):
    n = np.asarray(n, float)
    full = np.floor(n / 128); tail = n - 128 * full
    tw = np.where(tail <= 0, 0, np.where(tail <= 16, 16, np.where(tail <= 64, 64, 128)))
    item = lambda w: np.where(w > 0, 8 + 0.125 * w, 0.0)
    return NT * (full * item(128) + item(tw)) / 131.0


def extend_slowest(cnt, home, pairs):
    cost = _pg_us(cnt)
    rep = dict(pairs)
    oh = np.zeros((NE, EP)); oh[np.arange(NE), home] = 1
    fixed = np.array([e not in rep for e in range(NE)], float)
    load = (cost * fixed) @ oh
    for f in range(cnt.shape[0]):
        for e in sorted(rep, key=lambda x: (-cnt[f, x], x)):
            if cnt[f, e] <= 0:
                continue
            h, r = home[e], rep[e]
            j = r if load[f, r] < load[f, h] else h
            load[f, j] += cost[f, e]
    return load.max(1).mean()


def _draw_ext(q, T, kappa, F, rng):
    out = np.zeros((F, NE))
    for f in range(F):
        qq = rng.dirichlet(kappa * NE * q + 1e-6)
        g = rng.gumbel(size=(T, NE)) + np.log(qq + 1e-30)[None, :]
        np.add.at(out[f], np.argpartition(-g, TOPK, axis=1)[:, :TOPK].ravel(), 1)
    return out


ROUTES_16K = 16384 * 8


def live_groups(vhist):
    """{verify bs: forwards} -> [(bs, forwards, metric, cost_passes)]: the st regime (T = 8 bs <= 64, passes) and the pg
    regime (distinct experts) apart, each as at most 2 groups of adjacent batch sizes (the split with the least
    count-weighted bs variance), each group at its count-weighted mean bs."""
    out = []
    for small in (True, False):
        items = sorted((int(b), float(c)) for b, c in vhist.items() if c > 0 and (8 * int(b) <= 64) == small)
        if not items:
            continue
        if len(items) <= 2:
            bins = [[it] for it in items]
        else:   # the 2-way split of adjacent sizes with the least count-weighted bs variance
            def sse(g):
                n = sum(c for _, c in g)
                m = sum(b * c for b, c in g) / n
                return sum(c * (b - m) ** 2 for b, c in g)
            k = min(range(1, len(items)), key=lambda j: sse(items[:j]) + sse(items[j:]))
            bins = [items[:k], items[k:]]
        for g in bins:
            n = sum(c for _, c in g)
            bs = max(1, int(round(sum(b * c for b, c in g) / n)))
            bs = min(bs, 8) if small else max(bs, 9)
            out.append((bs, n, "passes", True) if small else (bs, n, "distinct", False))
    return out


def _context(home, verify_counts, prefill_counts, seed, prefill_chunks, F_v, F_p, F_x, live=None):
    """(J, p, q, candidates): the objective on this run's counts. prefill_chunks rows (per-forward route counts) are
    scaled to one 16K forward each, the unit STRAG_CHUNKS counts. live = {"vhist", "n_pre", "n_ext"} (OKC_ADAPT_OBJ=live):
    the costs are weighted by the run's own forwards instead of the per-load constants, J in ms of slowest-rank time."""
    rng = np.random.default_rng(seed)
    p = np.asarray(verify_counts, float) + 1e-3; p /= p.sum()
    q = np.asarray(prefill_counts, float) + 1e-3; q /= q.sum()
    if live is not None:
        groups = live_groups(live["vhist"])
        # st-regime groups (passes metric, a few routes per draw) need more draws than pg's for the same accuracy
        vtr = [(route_counts(p, bs, 0.2, 30 if cp else 10, rng), n, met, cp) for bs, n, met, cp in groups]
        w_p, w_x = float(live["n_pre"]), float(live["n_ext"])
    else:
        vtr = [(route_counts(p, bs, 0.2, F, rng), w, met, cp) for (bs, w, met, cp), F in zip(GROUPS, F_v)]
        w_p, w_x = W_P * STRAG_CHUNKS, W_X
    if prefill_chunks is not None and len(prefill_chunks):
        ptr = np.asarray(prefill_chunks, float)
        ptr = np.round(ptr * (ROUTES_16K / np.maximum(ptr.sum(1, keepdims=True), 1.0)))
    else:
        ptr = rng.multinomial(ROUTES_16K, q, size=F_p).astype(float)
    xtr = _draw_ext(q, 1024, 8.49, F_x, rng)

    def J(pairs):
        v = sum(w * rank_costs(c, home, pairs, met, cp).max(1).mean() / 1e3 for c, w, met, cp in vtr)
        return v + w_p * strag_us(ptr, home, pairs) / 1e3 + w_x * extend_slowest(xtr, home, pairs) / 1e3

    cand = sorted(set(np.argsort(-p)[:32].tolist()) | set(np.argsort(-q)[:32].tolist()))
    return J, p, q, cand


def _order(pairs, p):
    return [(int(e), int(r)) for e, r in sorted(pairs, key=lambda x: (-p[x[0]], x[0]))]


def _greedy(J, p, cand, home, R):
    chosen, slots = {}, np.full(EP, R)
    for _ in range(EP * R):
        best = None
        for e in cand:
            if e in chosen:
                continue
            for r in range(EP):
                if r == home[e] or slots[r] == 0:
                    continue
                trial = dict(chosen); trial[e] = r
                val = J(_order(trial.items(), p))
                if best is None or val < best[0] - 1e-12:
                    best = (val, e, r)
        if best is None:
            break
        chosen[best[1]] = best[2]; slots[best[2]] -= 1
    return _order(chosen.items(), p)


def _refine(J, p, cand, home, cur, theta, max_swaps):
    """Local search from the current plan: one move = the expert in one (expert, rank) slot replaced by a candidate
    not replicated yet (same rank, never its home); a move is taken only if it lowers J by more than theta (its copy
    cost); at most max_swaps moves."""
    cur = list(cur)
    val = J(_order(cur, p))
    for _ in range(max_swaps):
        used = {e for e, _ in cur}
        best = None
        for i, (e0, r) in enumerate(cur):
            for e in cand:
                if e in used or home[e] == r:
                    continue
                trial = cur[:i] + [(e, r)] + cur[i + 1:]
                v = J(_order(trial, p))
                if best is None or v < best[0] - 1e-12:
                    best = (v, trial)
        if best is None or best[0] > val - theta:
            break
        val, cur = best
    return _order(cur, p), val


def decide(home, verify_counts, prefill_counts, R, seed=0, prefill_chunks=None, F_v=(20, 20, 10, 10), F_p=16, F_x=8,
           current=None, theta=0.0, max_swaps=4, full=True, live=None):
    """home [256] rank, the run's verify / prefill route counts [256] (+ per-forward prefill rows) -> R x 8
    (expert, rank) pairs in balancing order. With current pairs: the incremental search (theta = objective units per
    expert copy, at most max_swaps copies) and, if full, the from-scratch greedy too; the plan with the lower
    J + theta x copies (vs current) wins, current itself included."""
    home = np.asarray(home)
    J, p, q, cand = _context(home, verify_counts, prefill_counts, seed, prefill_chunks, F_v, F_p, F_x, live)
    if current is None:
        return _greedy(J, p, cand, home, R)
    cur = _order(current, p)
    copies = lambda pl: len(set(pl) - set(cur))
    best = (J(cur), cur)
    inc, vi = _refine(J, p, cand, home, cur, theta, max_swaps)
    if vi + theta * copies(inc) < best[0] - 1e-12:
        best = (vi + theta * copies(inc), inc)
    if full:
        g = _greedy(J, p, cand, home, R)
        vg = J(g) + theta * copies(g)
        if vg < best[0] - 1e-12:
            best = (vg, g)
    return best[1]


# ----------------------------------------------------------------------------------------------- runtime (serving)
def _knobs():
    from sglang.kernels.ops.muse_spark_v12 import okc_patch as sw

    return sw


def attach_buffers(layer, device=None):
    """Register the counters on a MuseSpark1xFp4Experts with replicas (called from its __init__)."""
    K = _knobs().ADAPT_RING
    layer.register_buffer("_okc_adapt_vcnt", torch.zeros(256, dtype=torch.int32, device=device), persistent=False)
    layer.register_buffer("_okc_adapt_ring", torch.zeros((K, 256), dtype=torch.int32, device=device), persistent=False)
    layer.register_buffer("_okc_adapt_slot", torch.zeros(1, dtype=torch.int64, device=device), persistent=False)


def verify_hist(experts, ep_rank):
    """route_k's counter for this layer (rank 0 only: the verify ids are rank-identical), else None."""
    v = getattr(experts, "_okc_adapt_vcnt", None)
    return v if (v is not None and ep_rank == 0) else None


def count_prefill(experts, topk_ids, local_ids, ep_rank):
    """T > 256 forwards: this forward's route counts into the ring's next slot. local_ids: SP shard ids (count on every
    rank), else rank-identical ids (rank 0 counts)."""
    ring = getattr(experts, "_okc_adapt_ring", None)
    if ring is None:
        return
    slot = experts._okc_adapt_slot
    ring.index_fill_(0, slot, 0)    # the slot's row holds one forward: clear what it held K forwards ago
    if local_ids or ep_rank == 0:   # every rank still advances the slot, so ring rows stay aligned across ranks
        idx = topk_ids.reshape(-1).long() + slot * 256
        ring.view(-1).index_add_(0, idx, torch.ones(idx.shape, dtype=torch.int32, device=idx.device))
    slot.add_(1).remainder_(ring.shape[0])


def _decide_many(jobs):
    """Background worker: [(layer, home, verify counts, ring counts, R, seed, current pairs, theta, max_swaps, full)]
    -> {layer: pairs}."""
    out = {}
    for job in jobs:
        li, home, v, ring, R, seed, cur, theta, swaps, full = job[:10]
        live = job[10] if len(job) > 10 else None
        ring = np.asarray(ring, float)
        chunks = ring[ring.sum(1) > 0]
        pre = chunks.sum(0) if len(chunks) else np.ones(256)
        out[li] = decide(np.asarray(home), np.asarray(v, float), pre, R, seed=seed,
                         prefill_chunks=chunks if len(chunks) >= 4 else None,
                         current=[tuple(x) for x in cur], theta=theta, max_swaps=swaps, full=full, live=live)
    return out


class _State:
    def __init__(self):
        self.layers = None          # [(layer_idx, experts module)] with replicas, layer order
        self.tokens = 0             # prefill tokens forwarded since the last trigger
        self.vtokens = 0            # verify tokens forwarded (host count; gates the first refresh)
        self.seen = 0               # prefill tokens forwarded since the start
        self.seen_full = 0          # ... at the last full refresh
        self.full = False           # the pending refresh is a full one
        self.refreshes = 0
        self.pending = None         # AsyncResult of this rank's decisions
        self.fwd = 0
        self.gs = None              # {layer: [256] float global scales of every expert}
        self.last_counts = None
        self.queue = []             # [(layer, pairs)] decided, not applied yet (trickled: ADAPT_TRICKLE copies per forward)
        self.rec = None             # the refresh being applied (audit record)
        self.warm = False
        self.vhist = {}             # OKC_ADAPT_OBJ=live: verify forwards per batch size since the last refresh (halved)
        self.n_pre = 0.0            # ... prefill forwards (T >= ADAPT_MIN_T) in 16K-chunk equivalents
        self.n_ext = 0              # ... extend forwards below ADAPT_MIN_T


_S = _State()


def _collect(model):
    from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts

    out = []
    for m in model.modules():
        if isinstance(m, MuseSpark1xFp4Experts) and getattr(m, "_okc_adapt_ring", None) is not None \
                and getattr(m, "_okc_rep_table", None) is not None and getattr(m, "_okc_moe_state", None) is not None:
            out.append((int(m._okc_layer_idx), m))
    return sorted(out, key=lambda x: x[0])


def _placement_ranks(li):
    from sglang.srt.layers import muse_spark_expert_placement as pl

    return pl._placement()["layers"][str(li)]


def maybe_refresh(model_runner, forward_batch):
    """ModelRunner.forward entry (target worker). Host-side counters only, except at a trigger / check point."""
    sw = _knobs()
    if torch.cuda.is_current_stream_capturing():
        return
    st = _S
    st.fwd += 1
    mode = forward_batch.forward_mode
    if mode.is_target_verify():
        st.vtokens += int(forward_batch.input_ids.numel())
        if sw.ADAPT_OBJ == "live":
            b = int(forward_batch.batch_size)
            st.vhist[b] = st.vhist.get(b, 0) + 1
    elif mode.is_extend():
        n = getattr(forward_batch, "extend_num_tokens", None)
        n = int(n if n is not None else forward_batch.input_ids.numel())
        st.tokens += n
        st.seen += n
        if sw.ADAPT_OBJ == "live":
            if n >= sw.ADAPT_MIN_T:
                st.n_pre += n / 16384.0
            else:
                st.n_ext += 1
    if st.layers is None:
        st.layers = _collect(model_runner.model)
        if not st.layers:
            logger.info("OKC adaptive replicas: no pg layer with replicas; off.")
        else:
            logger.info("OKC adaptive replicas on for %d layers (first after %d prefill + %d verify tokens, then every "
                        "%d prefill tokens).", len(st.layers), sw.ADAPT_FIRST_TOKENS, sw.ADAPT_FIRST_VERIFY,
                        sw.ADAPT_EVERY_TOKENS)
    if not st.layers:
        return
    if not st.warm:   # NCCL's P2P connections between every pair, paid here (server warm-up), not at a refresh
        st.warm = True
        _p2p_warmup()
    if st.queue:
        _trickle(sw.ADAPT_TRICKLE)
    elif st.pending is None:
        if st.refreshes == 0:
            if st.tokens >= sw.ADAPT_FIRST_TOKENS and st.vtokens >= sw.ADAPT_FIRST_VERIFY:
                _trigger()
        elif st.tokens >= sw.ADAPT_EVERY_TOKENS:
            _trigger()
    elif st.fwd % sw.ADAPT_CHECK == 0:
        _check()


def _groups():
    from sglang.srt.distributed.parallel_state import get_tp_group

    g = get_tp_group()
    return g.device_group, g.cpu_group, g.rank_in_group, g.world_size


def full_refresh(refreshes, seen, seen_full, growth):
    """Refresh 1 is full; later ones when the prefill tokens seen have grown growth x since the last full one."""
    return refreshes == 0 or (growth > 0 and seen >= growth * seen_full)


def _trigger():
    sw = _knobs()
    st = _S
    dev_g, cpu_g, rank, world = _groups()
    import torch.distributed as dist

    V = torch.stack([m._okc_adapt_vcnt for _, m in st.layers])
    Rg = torch.stack([m._okc_adapt_ring for _, m in st.layers])
    dist.all_reduce(V, group=dev_g)
    dist.all_reduce(Rg, group=dev_g)
    if st.gs is None:   # every expert's global scale: each rank's home slots, gathered once
        G = torch.stack([m._okc_moe_state["w13_gs32"][: m.num_home_experts].float() for _, m in st.layers])
        allg = [torch.empty_like(G) for _ in range(world)]
        dist.all_gather(allg, G, group=dev_g)
        allg = [a.cpu().numpy() for a in allg]
        st.gs = {}
        for i, (li, m) in enumerate(st.layers):
            g = np.zeros(256, np.float32)
            for r, ids in enumerate(_placement_ranks(li)):
                g[np.asarray(ids)] = allg[r][i][: len(ids)]
            st.gs[li] = g
    Vh, Rh = V.cpu().numpy(), Rg.cpu().numpy()
    st.last_counts = {"layers": [li for li, _ in st.layers], "verify": Vh.tolist(), "prefill_ring": Rh.tolist()}
    for _, m in st.layers:
        m._okc_adapt_vcnt.floor_divide_(2)
    full = full_refresh(st.refreshes, st.seen, st.seen_full, sw.ADAPT_FULL_GROWTH)
    if full:
        st.seen_full = st.seen
    live = None
    if sw.ADAPT_OBJ == "live":   # every rank saw the same forwards, so the weights agree without a collective
        live = {"vhist": dict(st.vhist), "n_pre": st.n_pre, "n_ext": st.n_ext}
        st.vhist = {b: c // 2 for b, c in st.vhist.items() if c // 2}
        st.n_pre /= 2.0
        st.n_ext //= 2
    theta = sw.ADAPT_THETA_MS if live is not None else sw.ADAPT_THETA
    jobs = []
    for i, (li, m) in enumerate(st.layers):
        if i % world != rank:
            continue
        R = len(m.replica_plan["order"]) // world
        cur = [list(x) for x in zip(m.replica_plan["order"], m.replica_plan["rank"])]
        jobs.append((li, list(m.replica_plan["home"]), Vh[i].tolist(), Rh[i].tolist(), R, 1000 + li, cur,
                     theta, sw.ADAPT_MAX_SWAPS, full, live))
    # a plain child interpreter (not multiprocessing: the scheduler process may be daemonic), numpy single-threaded
    import pickle
    import subprocess
    import sys
    import tempfile

    f = tempfile.NamedTemporaryFile(prefix="okc_adapt_", suffix=".pkl", delete=False)
    pickle.dump(jobs, f); f.close()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    proc = subprocess.Popen([sys.executable, "-m", __name__, f.name, f.name + ".out"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=lambda: os.nice(10))
    st.pending = (proc, f.name, f.name + ".out")
    st.tokens = 0
    st.full = full
    logger.info("OKC adaptive replicas: refresh %d (%s, %d prefill tokens seen) decided in the background (%d layers "
                "here).", st.refreshes + 1, "full" if full else "incremental", st.seen, len(jobs))


def _check():
    import torch.distributed as dist

    st = _S
    _, cpu_g, rank, world = _groups()
    flags = [None] * world
    proc, fin, fout = st.pending
    dist.all_gather_object(flags, proc.poll() is not None, group=cpu_g)
    if not all(flags):
        return
    import pickle

    mine = {}
    if proc.returncode == 0 and os.path.exists(fout):
        with open(fout, "rb") as f:
            mine = pickle.load(f)
    else:
        logger.warning("OKC adaptive replicas: this rank's decision failed (rc %s); its layers keep their sets.", proc.returncode)
    for x in (fin, fout):
        try:
            os.remove(x)
        except OSError:
            pass
    parts = [None] * world
    dist.all_gather_object(parts, mine, group=cpu_g)
    plan = {}
    for p in parts:
        plan.update(p)
    st.pending = None
    st.refreshes += 1
    import hashlib
    import json

    changed = []
    for li, m in st.layers:
        new = plan.get(li)
        cur = [(int(e), int(r)) for e, r in zip(m.replica_plan["order"], m.replica_plan["rank"])]
        if new and [tuple(x) for x in new] != cur:
            changed.append((li, [tuple(map(int, x)) for x in new], len(set(map(tuple, new)) - set(cur))))
    st.queue = [(li, pl) for li, pl, _ in changed]
    full = {str(k): [list(map(int, x)) for x in v] for k, v in sorted(plan.items())}
    digest = hashlib.md5(json.dumps(full, sort_keys=True).encode()).hexdigest()[:12]
    ncopy = sum(c for _, _, c in changed)
    st.rec = {"refresh": st.refreshes, "full": st.full, "seen": st.seen,
              "forward": st.fwd, "layers_changed": len(changed), "copies": ncopy,
              "apply_ms": 0.0, "steps": 0, "md5": digest, "plan": full}
    logger.info("OKC adaptive replicas: refresh %d decided (plan %s): %d layers change, %d expert copies, trickled.",
                st.refreshes, digest, len(changed), ncopy)
    d = os.environ.get("OKC_ADAPT_LOG_DIR")
    if d and rank == 0:   # the audit trail: every refresh's plan and the counts it was decided on
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"refresh_{st.refreshes:03d}.json"), "w") as f:
            json.dump(dict(st.rec, counts=st.last_counts), f)
    if not st.queue:
        st.rec = None


def _trickle(budget):
    """Apply queued layers, whole layers only, up to `budget` expert copies this forward (at least one layer): every
    rank pops the same layers (the queue is identical everywhere), so the P2P pairs match."""
    st = _S
    take, n = {}, 0
    while st.queue:
        li, pl = st.queue[0]
        m = dict(st.layers)[li]
        cur = set(zip(m.replica_plan["order"], m.replica_plan["rank"]))
        c = len(set(pl) - cur)
        if take and n + c > budget:
            break
        take[li] = pl; n += c
        st.queue.pop(0)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    apply(take, _p2p_transfer)
    e1.record()
    if st.rec is not None:
        st.rec.setdefault("_ev", []).append((e0, e1))
        st.rec["steps"] += 1
        if not st.queue:
            e1.synchronize()
            st.rec["apply_ms"] = round(sum(a.elapsed_time(b) for a, b in st.rec.pop("_ev")), 2)
            logger.info("OKC adaptive replicas: refresh %d applied (%d expert copies over %d forwards, %.1f ms in all, "
                        "plan %s).", st.rec["refresh"], st.rec["copies"], st.rec["steps"], st.rec["apply_ms"],
                        st.rec["md5"])
            st.rec = None


def _p2p_warmup():
    import torch.distributed as dist

    dev_g, _, rank, world = _groups()
    dev = torch.cuda.current_device()
    ops = []
    for q in range(world):
        if q == rank:
            continue
        ops.append(dist.P2POp(dist.isend, torch.zeros(256, dtype=torch.uint8, device=dev), q, group=dev_g))
        ops.append(dist.P2POp(dist.irecv, torch.empty(256, dtype=torch.uint8, device=dev), q, group=dev_g))
    for w in dist.batch_isend_irecv(ops):
        w.wait()
    torch.cuda.current_stream().synchronize()


def _p2p_transfer(copies):
    """copies: [(src_rank, dst_rank, src tensors, dst tensors)] in one global order -> one NCCL message per (src, dst)
    (the pair's packs concatenated), unpacked into the slots on arrival."""
    import torch.distributed as dist

    dev_g, _, rank, _ = _groups()
    groups = {}
    for src, dst, s_tens, d_tens in copies:
        g = groups.setdefault((src, dst), [[], []])
        g[0] += s_tens; g[1] += d_tens
    ops, unpack = [], []
    for (src, dst), (s_tens, d_tens) in groups.items():
        if rank == src and s_tens:
            buf = torch.cat([t.reshape(-1).view(torch.uint8) for t in s_tens])
            ops.append(dist.P2POp(dist.isend, buf, dst, group=dev_g))
            unpack.append(buf)                      # keep alive until the send completes
        if rank == dst and d_tens:
            n = sum(t.numel() * t.element_size() for t in d_tens)
            buf = torch.empty(n, dtype=torch.uint8, device=d_tens[0].device)
            ops.append(dist.P2POp(dist.irecv, buf, src, group=dev_g))
            unpack.append((buf, d_tens))
    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()
    for u in unpack:
        if isinstance(u, tuple):
            buf, d_tens = u
            off = 0
            for t in d_tens:
                k = t.numel() * t.element_size()
                t.view(-1).view(torch.uint8).copy_(buf[off:off + k])
                off += k


def plan_slots(old_order, old_rank, new_pairs, world, R, nh):
    """-> (new local replica list per rank [world][R] (slot nh + i holds entry i), copies [(expert, rank, slot)])
    keeping every (expert, rank) that survives in its slot."""
    old_local = [[e for e, r in zip(old_order, old_rank) if r == q] for q in range(world)]
    new_by = [[e for e, r in new_pairs if r == q] for q in range(world)]
    local, copies = [], []
    for q in range(world):
        keep = set(old_local[q]) & set(new_by[q])
        slots = [e if e in keep else None for e in old_local[q]] + [None] * (R - len(old_local[q]))
        for e in new_by[q]:
            if e in keep:
                continue
            i = slots.index(None)
            slots[i] = e
            copies.append((e, q, nh + i))
        local.append(slots)
    return local, copies


def apply(plan, transfer, rank=None, world=None):
    """Install the decided replica sets: plan {layer: [(expert, replica rank)] in balancing order}. transfer(copies)
    moves pack bytes (NCCL in serving; local copies in tests). Returns the number of expert copies."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch.pg_moe import rep_table

    st = _S
    if rank is None:
        _, _, rank, world = _groups()
    copies_all, todo = [], []
    for li, m in st.layers:
        pairs = plan.get(li)
        if not pairs:
            continue
        R = len(m.replica_plan["order"]) // world
        home = m.replica_plan["home"]
        if any(home[e] == r for e, r in pairs) or any(sum(1 for _, r in pairs if r == q) != R for q in range(world)) \
                or len({e for e, _ in pairs}) != len(pairs):
            logger.warning("OKC adaptive replicas: layer %d plan rejected (bad shape); kept.", li)
            continue
        nh = m.num_home_experts
        local, copies = plan_slots(m.replica_plan["order"], m.replica_plan["rank"], pairs, world, R, nh)
        ranks = _placement_ranks(li)
        state = m._okc_moe_state
        for e, q, slot in copies:
            h = home[e]
            src_slot = ranks[h].index(e)
            s = [state["pg_wsf13"][src_slot], state["pg_wsf2"][src_slot]] if rank == h else []
            d = [state["pg_wsf13"][slot], state["pg_wsf2"][slot]] if rank == q else []
            copies_all.append((h, q, s, d))
        todo.append((li, m, pairs, local, copies))
    transfer(copies_all)
    for li, m, pairs, local, copies in todo:
        nh = m.num_home_experts
        mine = local[rank]
        gs = st.gs[li] if st.gs is not None else None
        for e, q, slot in copies:
            if q == rank and gs is not None:
                m._okc_moe_state["w13_gs32"][slot] = float(gs[e])
        ids = list(_placement_ranks(li)[rank]) + [e for e in mine if e is not None]
        slot_of = [-1] * 256
        for i, e in enumerate(ids):
            slot_of[e] = i
        order, holder = [e for e, _ in pairs], [r for _, r in pairs]
        m._okc_rep_table.copy_(rep_table(m.replica_plan["home"], slot_of, order, holder).to(m._okc_rep_table.device))
        m.replica_plan = dict(m.replica_plan, order=order, rank=holder, local=[e for e in mine if e is not None])
        if getattr(m, "local_expert_ids", None) is not None:
            m.local_expert_ids = ids
    return len(copies_all)


if __name__ == "__main__":   # the background decision: python -m ...okc_patch.adaptive <jobs.pkl> <out.pkl>
    import pickle
    import sys

    with open(sys.argv[1], "rb") as f:
        _jobs = pickle.load(f)
    # one process per layer job (this rank's ~8 layers): every layer's search is self-contained (its own seed), so the
    # plans equal the serial loop's, ~8x sooner; capped by OKC_ADAPT_WORKERS and by the host's cores over the 8 ranks
    _n = max(1, min(len(_jobs), int(os.environ.get("OKC_ADAPT_WORKERS", "8")), (os.cpu_count() or 8) // 8))
    if _n > 1:
        import multiprocessing as mp

        _res = {}
        with mp.get_context("fork").Pool(_n) as _pool:
            for _part in _pool.map(_decide_many, [[j] for j in _jobs]):
                _res.update(_part)
    else:
        _res = _decide_many(_jobs)
    with open(sys.argv[2] + ".tmp", "wb") as f:
        pickle.dump(_res, f)
    os.replace(sys.argv[2] + ".tmp", sys.argv[2])
