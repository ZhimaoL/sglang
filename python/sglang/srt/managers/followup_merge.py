"""Bounded merge of follow-up extends (OKC_SW_FU_MERGE, court C30 / C31).

At c16 every follow-up turn arrives on its own and the scheduler, prefill first, runs it as its own ~1.1K-token
extend (S14: 323 extends for 361 follow-ups per rep). An extend has a fixed cost of ~37 ms (the layers' weight
streams at T ~1K) and stalls every decoding request for its length. Holding a follow-up for a few scheduler passes,
while the running batch keeps decoding, lets the next ones join it in one extend: fewer extends, the same tokens.

The hold is bounded in passes (one per scheduler iteration, i.e. one forward), not in wall time, so every TP rank,
which sees the same requests in the same iteration, decides the same. It applies only when:
- the load (decoding + waiting requests) is >= min_load, which keeps c4 / c1's between-chunk admission and TTFT gate
  untouched;
- requests are decoding (the GPU never idles for it);
- no first turn is being chunked, and every waiting request is a follow-up (< 4096 new tokens over its cached
  prefix). A waiting first turn goes as small-prefill-first decides, and the folds during a chunked first turn are
  follow-up admission's.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)

# A request with this many new tokens or more is a first turn (or a follow-up that lost its prefix): never held.
_SMALL_NEW_TOKENS = 4096
# Prefix matches are capped like small-prefill-first's: above this many waiting requests the hold is skipped.
_MAX_WAITING = 16
_LOG_EVERY = 200


class FollowupMerge:
    def __init__(self, *, enabled: bool, passes: int, min_load: int):
        self.enabled = enabled and passes > 0
        self.passes = passes
        self.min_load = min_load
        self._pass = 0
        # rid -> (pass it was first seen waiting, new tokens over its cached prefix)
        self._seen: Dict[str, Tuple[int, int]] = {}
        self._num_holds = 0

    def should_hold(
        self,
        *,
        waiting_queue: List[Req],
        tree_cache: BasePrefixCache,
        chunked_req: Optional[Req],
        num_running: int,
    ) -> bool:
        """True = run a decode step now and keep the waiting follow-ups for a later, merged extend."""
        if not self.enabled:
            return False
        self._pass += 1
        if not waiting_queue:
            self._seen.clear()
            return False
        if (
            chunked_req is not None
            or num_running == 0
            or num_running + len(waiting_queue) < self.min_load
            or len(waiting_queue) > _MAX_WAITING
        ):
            return False
        from sglang.srt.managers.small_prefill_first import SmallPrefillFirst

        live = {req.rid for req in waiting_queue}
        for rid in [r for r in self._seen if r not in live]:
            del self._seen[rid]
        oldest = None
        for req in waiting_queue:
            seen = self._seen.get(req.rid)
            if seen is None:
                seen = self._seen[req.rid] = (self._pass, SmallPrefillFirst._new_tokens(req, tree_cache))
            if seen[1] >= _SMALL_NEW_TOKENS:
                return False
            oldest = seen[0] if oldest is None else min(oldest, seen[0])
        hold = self._pass - oldest < self.passes
        if hold:
            self._num_holds += 1
            if self._num_holds % _LOG_EVERY == 0:
                logger.info("Follow-up merge: %d passes held follow-ups for a merged extend.", self._num_holds)
        return hold
