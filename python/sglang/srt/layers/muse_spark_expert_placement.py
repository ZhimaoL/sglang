"""Static expert placement for Muse Spark under expert parallelism.

The contiguous split (rank r holds experts [r * n, (r + 1) * n)) leaves some ranks with far more routed tokens
than others in a layer, and every rank waits for the slowest one at the MoE all-reduce. The placement file lists,
per MoE layer, the experts each rank holds instead; it was fitted to routed-token counts measured on agentic
coding traffic, and any expert set per rank is exact (only which rank computes an expert changes).
"""

import functools
import json
import os
from typing import List, Optional

_PLACEMENT_PATH = os.path.join(
    os.path.dirname(__file__), "muse_spark_expert_placement.json"
)


@functools.cache
def _placement() -> dict:
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
    placement = _placement()
    if placement["ep_size"] != ep_size or placement["num_experts"] != num_experts:
        return None
    ranks = placement["layers"].get(str(layer_idx))
    return None if ranks is None else list(ranks[ep_rank])
