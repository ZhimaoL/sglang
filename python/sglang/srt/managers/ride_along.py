"""Let the running batch's verify step ride along in a prefill / extend forward.

The scheduler runs every pending prefill before any decode, so a 16K-token first-turn
chunk (~0.5 s on 8x H200) or a ~1K-token follow-up extend (~80 ms) stalls every
running request for its whole length, and the verify step that follows reads all the
weights again. A DFLASH verify row is an extend row (the block's tokens at positions
L..L+7 over the request's own KV), so the running requests' blocks join the prefill
forward as 8-row extend segments and their verify costs a small share of a forward
that streams the weights anyway. The prefill batch carries the prepared decode batch
(`ScheduleBatch.ride_along_batch`); the speculative worker runs both in one forward
(`forward_ride_along`) and returns one ordinary result per batch. Decisions use
scheduler state only, so every TP rank makes the same choice.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)

# OKC_SW_RIDE_ALONG values.
MODE_OFF = 0
# One EXTEND forward over the prefill rows and the riders' verify rows.
MODE_MERGED = 1
# Diagnostic: the same scheduling, but the prefill forward and the riders' ordinary
# verify step run back to back.
MODE_SEQUENTIAL = 2
# A progress line every this many carried verify steps.
_LOG_EVERY = 500


def plain_greedy(sampling_info: Optional[SamplingBatchInfo]) -> bool:
    """Whether a verify reduces to the argmax of the raw target logits (what the
    merged forward computes)."""
    if sampling_info is None:
        return True
    penalizer = sampling_info.penalizer_orchestrator
    return (
        sampling_info.is_all_greedy
        and not sampling_info.has_custom_logit_processor
        and sampling_info.logit_bias is None
        and sampling_info.grammar_mask is None
        and sampling_info.sampling_mask_batch_indices is None
        and (penalizer is None or not penalizer.is_required)
    )


class RideAlong:
    def __init__(self, *, mode: int, block_size: int, max_tokens: Optional[int]):
        self.mode = mode
        self._block_size = block_size
        # Largest forward the prefill CUDA graph replays; a merged forward must fit it.
        self._max_tokens = max_tokens
        self._num_rides = 0
        self._num_riders = 0

    @property
    def enabled(self) -> bool:
        return self.mode != MODE_OFF

    def _can_carry(self, running_batch: ScheduleBatch) -> bool:
        return (
            self.enabled
            and not running_batch.is_empty()
            and not running_batch.is_prefill_only
        )

    def reserve_tokens(
        self, running_batch: ScheduleBatch, chunk_budget: Optional[int]
    ) -> int:
        """Prefill tokens to hold back so the chunk plus the riders' rows still fit one
        graph bucket; a budget that already leaves room (a small-first batch) is not
        shrunk."""
        if (
            not self._can_carry(running_batch)
            or chunk_budget is None
            or self._max_tokens is None
        ):
            return 0
        rows = self._block_size * running_batch.batch_size()
        return max(0, chunk_budget + rows - self._max_tokens)

    def should_ride(
        self, prefill_batch: ScheduleBatch, running_batch: ScheduleBatch
    ) -> bool:
        """Whether the running batch's next verify step can join this prefill batch's
        forward."""
        if not self._can_carry(running_batch):
            return False
        if (
            prefill_batch.forward_mode != ForwardMode.EXTEND
            or prefill_batch.is_prefill_only
            or prefill_batch.input_embeds is not None
            or prefill_batch.replace_embeds is not None
            or any(mm is not None for mm in prefill_batch.multimodal_inputs or ())
        ):
            return False
        for batch in (prefill_batch, running_batch):
            if batch.return_logprob or batch.has_grammar or batch.return_hidden_states:
                return False
        if not plain_greedy(running_batch.sampling_info):
            return False
        rows = self._block_size * running_batch.batch_size()
        return (
            self._max_tokens is None
            or prefill_batch.extend_num_tokens + rows <= self._max_tokens
        )

    def note_ride(self, num_riders: int) -> None:
        self._num_rides += 1
        self._num_riders += num_riders
        if self._num_rides % _LOG_EVERY == 0:
            logger.info(
                "Ride-along: %d prefill forwards carried a verify step "
                "(%.1f riders each).",
                self._num_rides,
                self._num_riders / self._num_rides,
            )
