"""Serve small prefills (follow-up turns) ahead of large ones (first turns) and decode them right away.

A reasoning model shows its first token only after a decode step, and the scheduler runs every pending prefill
before any decode, so a follow-up that arrives together with a 74K-token first turn waits for that whole prefill
(~5.6 s on 8x H200) to show anything. Prefilling the small requests alone first and then decoding twice costs one
small prefill batch and two decode steps. Decisions count scheduler passes, so every TP rank makes the same choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from sglang.srt.mem_cache.common import match_kv_cache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

# A follow-up turn adds ~1K tokens to a cached conversation; a first turn is ~74K.
_SMALL_NEW_TOKENS = 4096
# Bounds how long a large request can be passed over by a stream of small ones.
_MAX_CONSECUTIVE_SMALL_BATCHES = 4
# Decode steps after a small prefill before the next prefill (the first visible token can take two).
_DECODE_STEPS_AFTER_SMALL = 2


class SmallPrefillFirst:
    def __init__(self, *, enabled: bool):
        self._enabled = enabled
        self._consecutive_small_batches = 0
        self._decode_steps_left = 0
        # The request being chunked; its final (possibly short) chunk is still a first turn.
        self._chunked_rid: Optional[str] = None

    def small_first_order(
        self,
        *,
        waiting_queue: List[Req],
        tree_cache: BasePrefixCache,
        has_chunked_req: bool,
    ) -> Optional[Tuple[List[Req], int]]:
        """(queue with the small requests first, token budget that admits only them), or None."""
        if (
            not self._enabled
            or has_chunked_req
            or len(waiting_queue) < 2
            or self._consecutive_small_batches >= _MAX_CONSECUTIVE_SMALL_BATCHES
            or not tree_cache.supports_fast_match_prefix()
        ):
            self._consecutive_small_batches = 0
            return None
        small, large, budget = [], [], 0
        for req in waiting_queue:
            match_kv_cache(req, tree_cache)
            new_tokens = (
                len(req.origin_input_ids)
                + len(req.output_ids)
                - req.num_matched_prefix_tokens
            )
            if new_tokens < _SMALL_NEW_TOKENS:
                small.append(req)
                budget += new_tokens
            else:
                large.append(req)
        if not small or not large:
            self._consecutive_small_batches = 0
            return None
        self._consecutive_small_batches += 1
        return small + large, max(budget, 1)

    def on_prefill_scheduled(
        self, *, batch: ScheduleBatch, chunked_req: Optional[Req], has_waiting: bool
    ) -> None:
        completes_small = any(
            req is not chunked_req
            and req.rid != self._chunked_rid
            and req.extend_end - req.prefix_len < _SMALL_NEW_TOKENS
            for req in batch.reqs
        )
        self._chunked_rid = chunked_req.rid if chunked_req is not None else None
        more_prefill = chunked_req is not None or has_waiting
        self._decode_steps_left = (
            _DECODE_STEPS_AFTER_SMALL
            if self._enabled and completes_small and more_prefill
            else 0
        )

    def should_decode_first(self, *, num_running: int) -> bool:
        if self._decode_steps_left == 0 or num_running == 0:
            self._decode_steps_left = 0
            return False
        self._decode_steps_left -= 1
        return True
