"""OKC small-m dense GEMM straight off the resident row-major bf16 weight (no copy): out = x @ W^T at verify m.

Kernel (_cuda_src_smallm.py, from muse-opt serving/kernels/smallm-dense/rmw.py): a 2-D TMA tensor map on the [N, K]
weight itself (box 64 k x 64 rows, SWIZZLE_128B lands each tile in the K-major layout wgmma reads), x by the same kind
of map, one consumer warpgroup running wgmma m64nNTk16 (W rows as A, the m tokens as B, NT = m rounded up to 8 / 16 /
32 / 64) beside a TMA producer warp, and split-K over the S CTAs of a cluster reduced through distributed shared memory
in a fixed order: fp32 accumulation, bf16 out, deterministic.

Claims only the (N, K, m) the dispatch table was measured at (1xH200, lt_final.py's harness): qkv_gate [2304, 8192] at
m <= 16 (14.7 / 15.2 us vs F.linear 16.7 and the MMA-pack route 16.7), the gemm_ag pooler slice [512, 8192] at m <= 64
(7.7-9.5 vs 10.1-10.5) and the unpooler slice [1024, 4096] at m <= 32 (7.9-9.0 vs 10.1-10.2). With it the qkv_gate MMA
pack (proj.prepare_weight, 2.2 GiB per rank) is not built. Everything else returns None.
"""

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_smallm import TABLE
from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import smallm_ext

_NTS = (8, 16, 32, 64)


def claims(n: int, k: int, m: int) -> bool:
    row = TABLE.get((n, k))
    return row is not None and 1 <= m <= max(row)


def prepare() -> None:
    """Build the extension at weight load (outside any CUDA-graph capture)."""
    smallm_ext()


def gemm(x, weight):
    """[m, K] @ [N, K]^T for a claimed (N, K, m); None = declined."""
    if torch.compiler.is_compiling():
        return None
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16 or not (x.is_cuda and weight.is_cuda):
        return None
    k = x.shape[-1]
    m = x.numel() // k if k else 0
    n = weight.shape[0]
    if weight.dim() != 2 or weight.shape[1] != k or not claims(n, k, m):
        return None
    if not weight.is_contiguous() or weight.data_ptr() % 16:
        return None
    x2 = x.reshape(m, k)
    if not x2.is_contiguous() or x2.data_ptr() % 16:
        return None
    nt = next(t for t in _NTS if t >= m)
    cfg = TABLE[(n, k)].get(nt)
    if cfg is None:
        return None
    NT, NST, S, KT, RT, NWG, PIPE = cfg
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    smallm_ext().rmw_launch(weight, x2, out, NT, NST, S, KT, RT, NWG, PIPE)
    return out.view(*x.shape[:-1], n)
