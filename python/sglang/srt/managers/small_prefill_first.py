"""Serve small prefills (follow-up turns) ahead of large ones (first turns) and decode them right away.

A reasoning model shows its first token only after a decode step, and the scheduler runs every pending prefill
before any decode, so a follow-up that arrives together with a 74K-token first turn waits for that whole prefill
(~5.6 s on 8x H200) to show anything. Prefilling the small requests alone first and then decoding twice costs one
small prefill batch and two decode steps. A follow-up that arrives while a first turn is being chunked joins the
next chunk's batch whole (the chunk shrinks to make room) at low concurrency, then gets the same two decode steps.
Between-chunk admission (OKC_SW_BETWEEN_CHUNK) keeps the chunk whole instead: the follow-up is prefilled on its own
between two chunks, and the scheduler picks what follows a chunk only once that chunk's target forward is done.
Follow-up admission (OKC_SW_FOLLOWUP_ADMIT) is between-chunk admission at low load and, above it, folds the small
newest arrivals into every chunk pass (the chunk shrinks to make room): at c32 under ride-along a follow-up otherwise
waits FCFS behind the remaining first turns (~40 s) and outlives the device SWA cache's ~8 s retention of its
conversation tail, so it re-prefills the whole prompt (serving/kernels/court/LOG.md C12).
Decisions count scheduler passes, so every TP rank makes the same choice.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from sglang.srt.mem_cache.common import match_kv_cache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)

# A follow-up turn adds ~1K tokens to a cached conversation; a first turn is ~74K.
_SMALL_NEW_TOKENS = 4096
# Bounds how long a large request can be passed over by a stream of small ones.
_MAX_CONSECUTIVE_SMALL_BATCHES = 4
# Decode steps after a small prefill before the next prefill (the first visible token can take two).
_DECODE_STEPS_AFTER_SMALL = 2
# Both orderings match every waiting request against the prefix tree to learn its uncached length, which
# upstream only does on caches whose fast-match flag is set (the serving UnifiedRadixCache's is not). These
# policies only matter at low concurrency, so the work is capped by queue length instead of by that flag.
_MAX_MATCH_REQS = 16
# Slack per request in a between-chunk budget: the adder may count a token or so more than the prefix match (the
# last cached token is re-extended), and a request that does not fit would be chunked.
_BETWEEN_MARGIN_TOKENS = 64
# A progress line every this many fold passes.
_FOLD_LOG_EVERY = 10


class SmallPrefillFirst:
    def __init__(
        self,
        *,
        enabled: bool,
        between_chunk_enabled: bool = False,
        between_chunk_max_reqs: int = 8,
        fold_enabled: bool = False,
    ):
        self._enabled = enabled
        # Follow-up admission is between-chunk admission up to the load cap.
        self._between_enabled = between_chunk_enabled or fold_enabled
        self._between_max_reqs = between_chunk_max_reqs
        self._fold_enabled = fold_enabled
        self._num_folds = 0
        self._num_folded = 0
        # One small batch per chunk boundary: set when it is scheduled, cleared when the next chunk is.
        self._between_done = False
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
            or len(waiting_queue) > _MAX_MATCH_REQS
        ):
            self._consecutive_small_batches = 0
            return None
        small, large, budget = [], [], 0
        for req in waiting_queue:
            new_tokens = self._new_tokens(req, tree_cache)
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

    def fold_order(
        self,
        *,
        chunked_req: Req,
        waiting_queue: List[Req],
        tree_cache: BasePrefixCache,
        budget: int,
        page_size: int,
        num_running: int,
    ) -> Optional[Tuple[List[Req], Optional[int]]]:
        """Follow-up admission above the between-chunk load cap: (queue with the small newest arrivals first, cap on
        the chunked request's next chunk or None) that folds them into this chunk pass whole, or None. Only the newest
        _MAX_MATCH_REQS waiting requests are matched (follow-ups arrive after the first turns they queue behind). The
        chunk keeps at least half the budget; when the chunked request finishes in this pass they take the leftover
        ahead of the next first turn."""
        if (
            not self._fold_enabled
            or not waiting_queue
            or num_running + len(waiting_queue) <= self._between_max_reqs
            or budget < 4 * page_size
        ):
            return None
        remaining = len(chunked_req.full_untruncated_fill_ids) - chunked_req.prefix_len
        page = lambda n: -(-n // page_size) * page_size
        finishing = remaining <= budget
        room = budget - page(remaining) if finishing else budget // 2
        small, reserved = [], 0
        for req in waiting_queue[-_MAX_MATCH_REQS:]:
            new_tokens = self._new_tokens(req, tree_cache)
            charge = page(new_tokens)
            if new_tokens < _SMALL_NEW_TOKENS and reserved + charge <= room:
                small.append(req)
                reserved += charge
        if not small:
            return None
        picked = set(map(id, small))
        queue = small + [req for req in waiting_queue if id(req) not in picked]
        return queue, None if finishing else (budget - reserved) // page_size * page_size

    def between_chunk_order(
        self,
        *,
        waiting_queue: List[Req],
        tree_cache: BasePrefixCache,
        num_running: int,
        chunk_size: int,
    ) -> Optional[Tuple[List[Req], int]]:
        """(the small waiting requests, a token budget that admits exactly them) to prefill on their own before the
        chunked request's next chunk, or None (the chunk goes next). The chunked request is not in this batch, so
        none of these may be chunked: they are taken while they fit the chunk size with a margin."""
        if (
            not self._between_enabled
            or self._between_done
            or not waiting_queue
            or num_running + len(waiting_queue) > self._between_max_reqs
            or len(waiting_queue) > _MAX_MATCH_REQS
        ):
            return None
        small, budget = [], 0
        for req in waiting_queue:
            new_tokens = self._new_tokens(req, tree_cache)
            charge = new_tokens + _BETWEEN_MARGIN_TOKENS
            if new_tokens < _SMALL_NEW_TOKENS and budget + charge <= chunk_size:
                small.append(req)
                budget += charge
        if not small:
            return None
        return small, budget

    def wait_for_chunk_boundary(self, *, num_running: int, num_waiting: int) -> bool:
        """Pick what follows an in-flight chunk only once its target forward is done: a follow-up that arrives during
        the chunk can still go next (the scheduler otherwise runs a batch ahead and has queued the next chunk)."""
        return self._between_enabled and num_running + num_waiting <= self._between_max_reqs

    @staticmethod
    def _new_tokens(req: Req, tree_cache: BasePrefixCache) -> int:
        match_kv_cache(req, tree_cache)
        return len(req.origin_input_ids) + len(req.output_ids) - req.num_matched_prefix_tokens

    def on_prefill_scheduled(
        self,
        *,
        batch: ScheduleBatch,
        chunked_req: Optional[Req],
        has_waiting: bool,
        between_chunk: bool = False,
        folded: bool = False,
    ) -> None:
        if self._between_enabled:
            self._between_done = between_chunk
        num_small = sum(
            1
            for req in batch.reqs
            if req is not chunked_req
            and req.rid != self._chunked_rid
            and req.extend_end - req.prefix_len < _SMALL_NEW_TOKENS
        )
        self._chunked_rid = chunked_req.rid if chunked_req is not None else None
        more_prefill = chunked_req is not None or has_waiting
        self._decode_steps_left = (
            _DECODE_STEPS_AFTER_SMALL
            if (self._enabled or between_chunk)
            and num_small
            and more_prefill
            else 0
        )
        if folded and num_small:
            self._num_folds += 1
            self._num_folded += num_small
            if self._num_folds % _FOLD_LOG_EVERY == 0:
                logger.info(
                    "Follow-up admission: %d chunk passes folded in %d follow-ups.",
                    self._num_folds,
                    self._num_folded,
                )

    def should_decode_first(self, *, num_running: int) -> bool:
        if self._decode_steps_left == 0 or num_running == 0:
            self._decode_steps_left = 0
            return False
        self._decode_steps_left -= 1
        return True
