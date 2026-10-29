"""Static expert placement for Muse Spark under expert parallelism.

The contiguous split (rank r holds experts [r * n, (r + 1) * n)) leaves some ranks with far more routed tokens
than others in a layer, and every rank waits for the slowest one at the MoE all-reduce. The placement file lists,
per MoE layer, the experts each rank holds instead; it was fitted to routed-token counts measured on agentic
coding traffic, and any expert set per rank is exact (only which rank computes an expert changes).

The replica file adds R hot experts of other ranks per rank and layer; a per-forward balancer that every rank computes
alike from the top-k ids picks which of the two holders computes each (again only the computing rank changes).
"""

import functools
import json
import os
from typing import List, Optional

# OKC_MOE_PLACEMENT_FILE / OKC_MOE_REPLICA_FILE: other tables of the same form (experiments; unset in serving)
_PLACEMENT_PATH = os.environ.get("OKC_MOE_PLACEMENT_FILE") or os.path.join(
    os.path.dirname(__file__), "muse_spark_expert_placement.json"
)
_REPLICA_PATH = os.environ.get("OKC_MOE_REPLICA_FILE") or os.path.join(
    os.path.dirname(__file__), "muse_spark_expert_replicas.json"
)
# OKC_MOE_HOMES=contiguous: stock homes (rank r holds experts [r * n, (r + 1) * n)) and a neutral initial replica plan
# (rank q replicates the first R experts of rank q + 1), no fitted table at all; with OKC_MOE_ADAPTIVE_REPLICAS the
# replica sets then come from the run's own counts (okc_patch/adaptive.py).
_CONTIGUOUS = os.environ.get("OKC_MOE_HOMES", "") == "contiguous"
# OKC_MOE_ADAPT_INIT (adaptive replicas only): "neutral" (default) starts from the neutral plan on whatever homes are
# active (rank q replicates the first R experts of rank q + 1), so the replica sets come from the run's own counts and
# no fitted R table is used; "table" starts from the replica file (experiments only).
_NEUTRAL_INIT = _CONTIGUOUS or (os.environ.get("OKC_MOE_ADAPTIVE_REPLICAS", "0") == "1"
                                and os.environ.get("OKC_MOE_ADAPT_INIT", "neutral") == "neutral")


@functools.cache
def _placement() -> dict:
    with open(_PLACEMENT_PATH) as f:
        placement = json.load(f)
    if _CONTIGUOUS:
        n = placement["num_experts"] // placement["ep_size"]
        placement["layers"] = {k: [list(range(r * n, (r + 1) * n)) for r in range(placement["ep_size"])]
                               for k in placement.get("layers", placement.get("owner", {}))}
        placement.pop("owner", None)
        return placement
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
    placement = _placement()
    if placement["ep_size"] != ep_size or placement["num_experts"] != num_experts:
        return None
    ranks = placement["layers"].get(str(layer_idx))
    return None if ranks is None else list(ranks[ep_rank])


@functools.cache
def _replicas() -> dict:
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
        raise ValueError(f"muse_spark_expert_replicas.json: bad plan for layer {layer_idx}, R={count}")
    local = [e for e, r in zip(order, holder) if r == ep_rank]
    return {"home": home, "order": order, "rank": holder, "local": local, "me": ep_rank}
