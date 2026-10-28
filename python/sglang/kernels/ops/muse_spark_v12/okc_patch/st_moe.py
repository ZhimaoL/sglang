"""OKC st: the small-T MoE pair for verify / decode forwards (T <= OKC_MOE_ST_MAX_TOKENS, default 72) on pg's packs.

pg tiles tokens 64 wide, so at T <= 64 a round of items costs the n64 wgmma time whatever the token count. st reads
the same blocks (pg_moe.pack_wsf) with mma.sync m16n8k16 (one warp's 16 rows of a pg block are its A fragment), pg's
decode and the same BF16 x BF16 -> fp32 math: every CTA builds the routing table from local_ids (32 home + replica
slots; an expert with c routes runs ceil(c / 8) passes), takes an equal range of 16-row slabs and splits K over its
16 warps (partials summed in warp order: deterministic); the epilogues are pg's, so y has pg's per-route scaling and
the finalize applies unchanged. Graph-capturable; OKC_MOE_ST_PDL launches the down kernel with PDL.
"""

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import st_ext

MAX_TOKENS = 128    # the kernels' table holds 1024 routes (top-8)


def h_rows(numel: int, n_local: int = 32) -> int:
    # intermediate rows: 8 per pass; passes = sum over local slots of ceil(c / 8) <= n_local + numel / 8
    return (n_local + numel // 8) * 8


_NO_NORM = {}


def grouped_st(x, local_ids, *, wsf13, gs13, wsf2, pdl=False, norm_weight=None, norm_eps=0.0):
    """x [T, 4096] bf16 (T <= 128), local_ids [T, 8] int32 (-1 = not local) -> y [T*8, 4096] bf16 per-route expert
    outputs with pg's scaling (rows of non-local routes stay unwritten): the contract of pg_moe.grouped_pg."""
    numel = local_ids.numel()
    h = torch.empty((h_rows(numel, wsf13.shape[0]), 4096), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((numel, 4096), dtype=torch.bfloat16, device=x.device)
    if norm_weight is None:                            # x already normed
        norm_weight = _NO_NORM.get(x.device)
        if norm_weight is None:
            norm_weight = _NO_NORM[x.device] = torch.empty(0, dtype=torch.bfloat16, device=x.device)
    st_ext().st_moe(x, local_ids.reshape(-1), wsf13, gs13, wsf2, h, y, 1 if pdl else 0, norm_weight, float(norm_eps))
    return y
