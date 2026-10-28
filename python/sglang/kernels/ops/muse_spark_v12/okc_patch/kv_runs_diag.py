"""Diagnostic, off by default (OKC_DIAG_KV_RUNS=N > 0 enables it): every N-th decode / verify batch with at least
OKC_DIAG_KV_RUNS_MIN_BS requests, TP rank 0 logs how the batch's KV slots (its req_to_token rows, page size 1) split
into runs of consecutive slots: runs, mean run length, the token share in runs of 1 / 2-15 / 16-63 / 64-255 /
256-1023 / >= 1024 slots, and the share of whole 128-token tiles whose 128 slots are consecutive (what a TMA box
producer for the full-layer verify attention could load in one copy). Read-only; one host sync per logged batch."""

import logging

import torch

logger = logging.getLogger(__name__)

_BOUNDS = (2, 16, 64, 256, 1024)
_LABELS = ("1", "2-15", "16-63", "64-255", "256-1023", ">=1024")
_SEEN = [0]


def kv_runs_stats(rows, lens):
    """rows [B, L] int slots (row b valid up to lens[b]); returns (tokens, runs, token share per bucket, tile share)."""
    B, L = rows.shape
    dev = rows.device
    pos = torch.arange(L, device=dev)
    valid = pos[None, :] < lens[:, None]
    start = torch.ones((B, L), dtype=torch.bool, device=dev)
    if L > 1:
        start[:, 1:] = (rows[:, 1:] - rows[:, :-1]) != 1
    runs = torch.bincount(torch.cumsum(start[valid].to(torch.int64), 0) - 1)   # run lengths, rows in order
    bucket = torch.bucketize(runs, torch.tensor(_BOUNDS, device=dev), right=True)
    share = torch.zeros(len(_LABELS), dtype=torch.float64, device=dev).index_add_(0, bucket, runs.double())
    nt = L // 128
    if nt:
        brk = start[:, :nt * 128].reshape(B, nt, 128)[:, :, 1:].any(-1)              # a break inside the tile
        full = (torch.arange(1, nt + 1, device=dev)[None, :] * 128) <= lens[:, None]
        tiles = (full & ~brk).sum() / full.sum().clamp_min(1)
    else:
        tiles = torch.zeros((), device=dev)
    tok = int(valid.sum())
    return tok, int(runs.numel()), (share / max(tok, 1)).tolist(), float(tiles)


def kv_runs(batch, req_to_token, every, min_bs):
    if batch is None or batch.forward_mode is None or batch.forward_mode.is_extend() or batch.batch_size() < min_bs:
        return
    _SEEN[0] += 1
    if _SEEN[0] % every:
        return
    lens = (batch.seq_lens_cpu if batch.seq_lens_cpu is not None else batch.seq_lens).to(req_to_token.device)
    rows = req_to_token[batch.req_pool_indices.to(req_to_token.device).long(), : int(lens.max())]
    tok, nruns, share, tiles = kv_runs_stats(rows, lens.long())
    logger.info(
        "OKC kv runs: bs %d, %d tokens in %d runs (mean %.0f); token share in runs of %s slots: %s; whole 128-token "
        "tiles on consecutive slots %.1f %%", batch.batch_size(), tok, nruns, tok / max(nruns, 1), " / ".join(_LABELS),
        " / ".join(f"{100 * s:.1f} %" for s in share), 100 * tiles)
