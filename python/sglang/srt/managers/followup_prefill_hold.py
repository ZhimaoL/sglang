"""Under load, let a small follow-up prefill wait a few decode steps for company.

A follow-up turn of a long conversation prefills only its new tokens, but every prefill batch still pays a fixed
cost and stalls every decoding request; follow-ups that arrive within a few decode steps of each other can share
one batch instead. The hold counts decode steps, not wall time, so that every TP rank makes the same choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from sglang.srt.mem_cache.common import match_kv_cache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

# Below this many decoding requests a prefill stalls little, so low-concurrency traffic is never held.
_MIN_RUNNING = 8
# A queue this deep already makes a batch.
_MAX_QUEUED = 4
# Only small prefills wait: a first turn or an evicted conversation goes at once.
_MAX_NEW_TOKENS = 4096


class FollowupPrefillHold:
    def __init__(self, *, hold_steps: int):
        self._hold_steps = hold_steps
        self._held = 0

    def should_hold(
        self,
        *,
        waiting_queue: List[Req],
        tree_cache: BasePrefixCache,
        num_running: int,
        has_chunked_req: bool,
    ) -> bool:
        if not waiting_queue:
            self._held = 0
            return False
        if (
            self._hold_steps == 0
            or self._held >= self._hold_steps
            or has_chunked_req
            or num_running < _MIN_RUNNING
            or len(waiting_queue) >= _MAX_QUEUED
            or not tree_cache.supports_fast_match_prefix()
            or _new_tokens(waiting_queue=waiting_queue, tree_cache=tree_cache)
            >= _MAX_NEW_TOKENS
        ):
            return False
        self._held += 1
        return True

    def on_prefill_scheduled(self) -> None:
        self._held = 0


def _new_tokens(*, waiting_queue: List[Req], tree_cache: BasePrefixCache) -> int:
    # The same fast match calc_priority runs on every waiting request each scheduling pass.
    total = 0
    for req in waiting_queue:
        match_kv_cache(req, tree_cache)
        total += (
            len(req.origin_input_ids)
            + len(req.output_ids)
            - req.num_matched_prefix_tokens
        )
    return total
