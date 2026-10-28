"""The merged EXTEND forward of a prefill batch and the DFLASH verify blocks riding
along in it.

Request r with committed length L verifies its block (bonus token + drafts) at
positions L..L+B-1: the block writes the KV slots reserved for those positions and
attends causally to [0, L+i]. An EXTEND forward computes exactly that for a request
with prefix L and extend length B, so each rider joins the prefill batch as a B-row
extend segment behind the prefill requests. The forward returns logits for each
prefill request's last row and for every rider row.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def ride_along_logits_rows(
    prefill_extend_lens: List[int],
    num_riders: int,
    block_size: int,
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[List[int], torch.Tensor]:
    """Flat token rows of the merged forward whose logits are needed: each prefill
    request's last row, then every row of the riders' blocks (they follow the prefill
    tokens). A list, or an int64 tensor on ``device`` (pinned host-to-device copy)."""
    rows, end = [], 0
    for extend_len in prefill_extend_lens:
        end += extend_len
        rows.append(end - 1)
    rows.extend(range(end, end + num_riders * block_size))
    if device is None:
        return rows
    return torch.tensor(
        rows, dtype=torch.int64, pin_memory=is_pin_memory_available(device)
    ).to(device, non_blocking=True)


def build_ride_along_batch(
    prefill_batch: ScheduleBatch,
    decode_batch: ScheduleBatch,
    *,
    draft_tokens: torch.Tensor,
    verify_out_cache_loc: torch.Tensor,
) -> ScheduleBatch:
    """A ScheduleBatch view of the merged EXTEND forward for `ForwardBatch.init_new`;
    both inputs stay as they are. ``draft_tokens`` is the riders' [bs, B] block,
    ``verify_out_cache_loc`` its reserved slots (row-major); the riders' committed
    lengths are ``decode_batch.seq_lens`` (device) and ``seq_lens_cpu`` (host, exact
    after the overlap resolve)."""
    num_riders, block_size = draft_tokens.shape
    committed = decode_batch.seq_lens
    committed_cpu = decode_batch.seq_lens_cpu
    rider_seq_lens = committed + block_size
    rider_seq_lens_cpu = committed_cpu + block_size

    merged = copy.copy(prefill_batch)

    def cat(prefill: torch.Tensor, riders: torch.Tensor) -> torch.Tensor:
        return torch.cat([prefill, riders.to(prefill.dtype)])

    merged.reqs = prefill_batch.reqs + decode_batch.reqs
    merged.forward_mode = ForwardMode.EXTEND
    merged.input_ids = cat(prefill_batch.input_ids, draft_tokens.reshape(-1))
    merged.req_pool_indices = cat(
        prefill_batch.req_pool_indices, decode_batch.req_pool_indices
    )
    merged.req_pool_indices_cpu = cat(
        prefill_batch.req_pool_indices_cpu, decode_batch.req_pool_indices_cpu
    )
    merged.seq_lens = cat(prefill_batch.seq_lens, rider_seq_lens)
    merged.seq_lens_cpu = cat(prefill_batch.seq_lens_cpu, rider_seq_lens_cpu)
    merged.seq_lens_sum = int(merged.seq_lens_cpu.sum())
    if prefill_batch.orig_seq_lens is not None:
        merged.orig_seq_lens = cat(prefill_batch.orig_seq_lens, rider_seq_lens)
    merged.out_cache_loc = cat(prefill_batch.out_cache_loc, verify_out_cache_loc)
    merged.prefix_lens = prefill_batch.prefix_lens + [
        int(x) for x in committed_cpu.tolist()
    ]
    merged.extend_lens = prefill_batch.extend_lens + [int(block_size)] * num_riders
    merged.extend_num_tokens = prefill_batch.extend_num_tokens + num_riders * block_size
    if prefill_batch.extend_logprob_start_lens is not None:
        merged.extend_logprob_start_lens = (
            prefill_batch.extend_logprob_start_lens + [0] * num_riders
        )
    if prefill_batch.multimodal_inputs is not None:
        merged.multimodal_inputs = prefill_batch.multimodal_inputs + [None] * num_riders
    merged.spec_info = None
    merged.decoding_reqs = None
    merged.ride_along_batch = None
    return merged
