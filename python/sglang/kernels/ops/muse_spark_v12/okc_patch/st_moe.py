"""OKC st: the small-T MoE pair for verify / decode forwards (T <= OKC_MOE_ST_MAX_TOKENS, default 32) on pg's packs.

pg's gg1 / gg2 tile tokens 64 wide, so at T <= 64 a round of items costs the n64 wgmma time whatever the token
count (per-call durations 4 + 16 x ceil(64 n_act / 132) us for gg1 on the 8xH200 traces). st reads the same blocks
(pg_moe.pack_wsf, no other copy of the weights) with mma.sync m16n8k16, whose A fragment is the layout of one warp's
16 rows of a pg block, so the decode is pg's (dq3 + 8-bit scale codes) and the math is the same BF16 x BF16 -> fp32:
  * no moe_align / pg_tiles / pg_gather: every CTA builds the call's routing table from local_ids (deterministic, route
    order inside each expert); an expert with c routes runs as ceil(c / 8) passes of <= 8 tokens;
  * each CTA takes an equal contiguous range of 16-row slabs of the active experts; its 16 warps split each slab's K
    and stream their 576 B pieces straight into registers; the 16 partials are summed in warp order (no atomics);
  * epilogues are pg's (gg1: silu(g) u with the expert's global scale and the 2^8 decode compensation; gg2: x 2^8),
    so y has pg's per-route scaling and the caller's finalize applies unchanged.
CUDA-graph capturable (no host sync; buffer sizes come from T only). OKC_MOE_ST_PDL launches the down kernel with
programmatic dependent launch (its first W2 k-tiles are in flight before it waits for the intermediate).
"""

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import st_ext

MAX_TOKENS = 32     # the kernels' table holds 256 routes (top-8)


def h_rows(numel: int) -> int:
    # intermediate rows: 8 per pass; passes = sum over experts of ceil(c / 8) <= 32 + numel / 8
    return (32 + numel // 8) * 8


def grouped_st(x, local_ids, *, wsf13, gs13, wsf2, pdl=False):
    """x [T, 4096] bf16 (T <= 32), local_ids [T, 8] int32 (-1 = not local) -> y [T*8, 4096] bf16 per-route expert
    outputs with pg's scaling (rows of non-local routes stay unwritten): the contract of pg_moe.grouped_pg."""
    numel = local_ids.numel()
    h = torch.empty((h_rows(numel), 4096), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((numel, 4096), dtype=torch.bfloat16, device=x.device)
    st_ext().st_moe(x, local_ids.reshape(-1), wsf13, gs13, wsf2, h, y, 1 if pdl else 0)
    return y
