"""Expert placement for Muse Spark under expert parallelism.

Default: the homes in muse_spark_expert_placement.json (per MoE layer, the experts each rank holds; derived from routed-token
counts on agentic coding traffic; any expert set per rank is exact, only which rank computes an expert changes) plus R
hot-expert replicas per rank and layer whose sets come from the run's own routing counts (okc_patch/adaptive.py, refreshed
between forwards; the first plan is neutral: rank q replicates the first R experts of rank q + 1). A per-forward balancer
that every rank computes alike from the top-k ids picks which of the two holders computes each replicated expert.

OKC_MOE_HOMES=contiguous: the contiguous split (rank r holds experts [r * n, (r + 1) * n)), no table. Experiments only:
OKC_MOE_PLACEMENT_FILE (another table of the same form) and OKC_MOE_ADAPT_INIT=table with OKC_MOE_REPLICA_FILE.
"""

import functools
import json
import os
from typing import List, Optional

# OKC_MOE_PLACEMENT_FILE / OKC_MOE_REPLICA_FILE: other tables of the same form (experiments; unset in serving)
_PLACEMENT_PATH = os.environ.get("OKC_MOE_PLACEMENT_FILE") or os.path.join(
    os.path.dirname(__file__), "muse_spark_expert_placement.json"
)
_REPLICA_PATH = os.environ.get("OKC_MOE_REPLICA_FILE", "")
# OKC_MOE_HOMES=contiguous: stock homes (rank r holds experts [r * n, (r + 1) * n)) and a neutral initial replica plan
# (rank q replicates the first R experts of rank q + 1), no fitted table at all; with OKC_MOE_ADAPTIVE_REPLICAS the
# replica sets then come from the run's own counts (okc_patch/adaptive.py).
_CONTIGUOUS = os.environ.get("OKC_MOE_HOMES", "table") == "contiguous"
# OKC_MOE_ADAPT_INIT (adaptive replicas only): "neutral" (default) starts from the neutral plan on whatever homes are
# active (rank q replicates the first R experts of rank q + 1), so the replica sets come from the run's own counts and
# no fitted R table is used; "table" starts from the replica file (experiments only).
_NEUTRAL_INIT = _CONTIGUOUS or (os.environ.get("OKC_MOE_ADAPTIVE_REPLICAS", "1") == "1"
                                and os.environ.get("OKC_MOE_ADAPT_INIT", "neutral") == "neutral")


_SHAPE = []  # (num_experts, ep_size) of the served model, recorded by the first local_expert_ids / replica_plan call


class _EveryLayer(dict):
    """The contiguous split's per-rank expert lists, the same for every layer key."""

    def __init__(self, ranks):
        super().__init__()
        self._ranks = ranks

    def get(self, key, default=None):
        return self._ranks

    def __getitem__(self, key):
        return self._ranks


@functools.cache
def _placement() -> dict:
    if _CONTIGUOUS:
        num_experts, ep_size = _SHAPE[0]
        n = num_experts // ep_size
        return {"ep_size": ep_size, "num_experts": num_experts,
                "layers": _EveryLayer([list(range(r * n, (r + 1) * n)) for r in range(ep_size)])}
    with open(_PLACEMENT_PATH) as f:
        placement = json.load(f)
    if "owner" in placement:
        # Compact form: owner[layer][e] is the rank holding expert e, one digit per expert (ep_size <= 10). Expanded to
        # the list form, each rank's experts ascending, so every reader sees the same table either way.
        n = placement["ep_size"]
        placement["layers"] = {
            k: [[e for e, r in enumerate(s) if int(r) == rank] for rank in range(n)]
            for k, s in placement.pop("owner").items()
        }
    return placement


def local_expert_ids(
    *, layer_idx: int, num_experts: int, ep_size: int, ep_rank: int
) -> Optional[List[int]]:
    """Global ids of the experts ``ep_rank`` holds in ``layer_idx``, or None to keep the contiguous split."""
    if not _SHAPE:
        _SHAPE.append((num_experts, ep_size))
    placement = _placement()
    if placement["ep_size"] != ep_size or placement["num_experts"] != num_experts:
        return None
    ranks = placement["layers"].get(str(layer_idx))
    return None if ranks is None else list(ranks[ep_rank])


@functools.cache
def _replicas() -> dict:
    if _NEUTRAL_INIT:   # the neutral first plan replaces every table row, so no file is read
        num_experts, ep_size = _SHAPE[0]
        return {"ep_size": ep_size, "num_experts": num_experts, "replicas": {}}
    with open(_REPLICA_PATH) as f:
        return json.load(f)


def replica_plan(
    *, layer_idx: int, num_experts: int, ep_size: int, ep_rank: int, count: int
) -> Optional[dict]:
    """The replicas of ``layer_idx`` with ``count`` per rank, or None (no replicas): ``home`` [num_experts] home rank of
    every expert, ``order`` the replicated experts in balancing order, ``rank`` the rank holding each one's replica,
    ``local`` the experts ``ep_rank`` holds as replicas (its local slots after the home experts), ``me`` = ep_rank."""
    if count <= 0:
        return None
    if not _SHAPE:
        _SHAPE.append((num_experts, ep_size))
    placement, plan = _placement(), _replicas()
    if (placement["ep_size"], placement["num_experts"]) != (ep_size, num_experts) or (
        plan["ep_size"], plan["num_experts"]
    ) != (ep_size, num_experts):
        return None
    ranks = placement["layers"].get(str(layer_idx))
    layer = plan["replicas"].get(str(count), {}).get(str(layer_idx))
    if _NEUTRAL_INIT and ranks is not None:
        layer = " ".join(f"{ranks[(q + 1) % ep_size][j]}:{q}" for j in range(count) for q in range(ep_size))
    if ranks is None or layer is None:
        return None
    home = [-1] * num_experts
    for r, ids in enumerate(ranks):
        for e in ids:
            home[e] = r
    pairs = [tuple(map(int, x.split(":"))) for x in layer.split()]
    order, holder = [e for e, _ in pairs], [r for _, r in pairs]
    if (
        min(home) < 0
        or len(set(order)) != len(order)
        or any(home[e] == r for e, r in zip(order, holder))
        or any(holder.count(r) != count for r in range(ep_size))
    ):
        raise ValueError(f"replica plan: bad plan for layer {layer_idx}, R={count}")
    local = [e for e, r in zip(order, holder) if r == ep_rank]
    return {"home": home, "order": order, "rank": holder, "local": local, "me": ep_rank}
