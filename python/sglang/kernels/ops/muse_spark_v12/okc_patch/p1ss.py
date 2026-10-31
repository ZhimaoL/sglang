"""OKC_SW_P1SS (docs/serving/review-1031/STRUCTURAL.md §3.1 / §3.1c): a first-turn pass above P1SS_CUT tokens (the
scheduler's two-chunk budget) runs as two children cut at P1SS_CUT tokens, so each child is exactly one stock chunk,
through the decoder layers on one stream in a static stage order (MuseSpark1xModel._forward_p1ss). This module holds
the children's attention backends (built beside the target worker's own) and cuts the pass's ForwardBatch."""
import torch

_BACKENDS = []
_FENCE = {}


def set_backends(backends):
    _BACKENDS[:] = backends


def split(fb, cut):
    """The pass's two children (ForwardBatches over tokens [0, cut) and [cut, n); a sequence that straddles the cut
    continues in child b with child a's tokens as its prefix) with their attention metadata on the child backends,
    and the forward contexts that select those backends."""
    from dataclasses import replace

    from sglang.srt.batch_overlap.two_batch_overlap import (
        TboForwardBatchPreparer,
        _update_device_and_sum_field_from_cpu_field,
    )
    from sglang.srt.model_executor.forward_context import get_forward_context

    n = fb.input_ids.shape[0]
    lens = list(fb.extend_seq_lens_cpu)
    assert len(_BACKENDS) == 2 and fb.tbo_children is None and sum(lens) == n and 0 < cut < n, (cut, n, lens)
    k = done = 0
    while done + lens[k] <= cut:
        done += lens[k]
        k += 1
    left = cut - done   # child a's tokens of sequence k (0: the cut is a sequence boundary)
    nnp = torch.tensor([cut, n - cut], dtype=torch.int32).to(fb.input_ids.device, non_blocking=True)
    a, b = kids = [
        TboForwardBatchPreparer.filter_batch(
            fb, start_token_index=t0, end_token_index=t1, start_seq_index=s0, end_seq_index=s1,
            out_num_token_non_padded=nnp[i], out_num_token_non_padded_cpu=t1 - t0)
        for i, (t0, t1, s0, s1) in enumerate(((0, cut, 0, k + (left > 0)), (cut, n, k, fb.batch_size)))
    ]
    if left:
        a.extend_seq_lens_cpu = list(a.extend_seq_lens_cpu)
        a.extend_seq_lens_cpu[-1] = left
        a.seq_lens_cpu = a.seq_lens_cpu.clone()
        a.seq_lens_cpu[-1] = a.extend_prefix_lens_cpu[-1] + left
        b.extend_seq_lens_cpu = list(b.extend_seq_lens_cpu)
        b.extend_seq_lens_cpu[0] -= left
        b.extend_prefix_lens_cpu = list(b.extend_prefix_lens_cpu)
        b.extend_prefix_lens_cpu[0] += left
        for c, cpu, dev, total in ((a, "extend_seq_lens_cpu", "extend_seq_lens", "extend_num_tokens"),
                                   (a, "seq_lens_cpu", "seq_lens", "seq_lens_sum"),
                                   (b, "extend_seq_lens_cpu", "extend_seq_lens", "extend_num_tokens"),
                                   (b, "extend_prefix_lens_cpu", "extend_prefix_lens", None)):
            _update_device_and_sum_field_from_cpu_field(batch=c, cpu_field=cpu, device_field=dev, sum_field=total)
    for c in kids:   # each child's token array starts at 0
        c.extend_start_loc = torch.zeros_like(c.extend_seq_lens)
        c.extend_start_loc[1:] = torch.cumsum(c.extend_seq_lens[:-1], 0)
    for c, backend in zip(kids, _BACKENDS):
        backend.init_forward_metadata(c)
    fb.tbo_children = kids
    ctx = get_forward_context()
    return kids, [replace(ctx, attn_backend=backend) for backend in _BACKENDS]


def pdl_fence(device):
    """A plain (non-programmatic) launch: the next kernel, even one launched with PDL, starts only after everything
    before it on the stream has finished. The deferred M1-RS receiver launches with PDL and never waits on its
    predecessor, which under P1-SS is the other child's gg2 or rows gather."""
    t = _FENCE.get(device)
    if t is None:
        t = _FENCE[device] = torch.zeros(1, dtype=torch.int32, device=device)
    t.zero_()
