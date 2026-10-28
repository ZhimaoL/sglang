"""OKC small-m dense projections (muse_bf16_proj v5 graft).

m == 1 runs the FFMA streaming GEMV straight off the resident [N, K] bf16 weight (no
copies); m = 2..8 runs the tuned Triton dot where that won (smallm.py takes qkv_gate and
the gemm_ag slices at those m first). Only the court-certified (N, K) shapes are
claimed; everything else falls through to the stock dispatcher.

Certified per-call wins (kernel court, H200, graph replay, cold L2): the small-T CUDA
stack is where decode1 1.46x / verify1 1.27-1.29x per layer came from. Nothing is copied.

"""

import torch
import triton
import triton.language as tl
from triton.language.extra import cuda as tlc

from sglang.kernels.ops.muse_spark_v12.okc_patch import SW_PDL
from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import proj_ext


@triton.jit
def _dot(x_ptr, w_ptr, out_ptr, M, N,
         K: tl.constexpr, BLOCK_M: tl.constexpr,
         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:   # programmatic dependent launch: wait for the producer of x before any load or store
        tlc.gdc_wait()
        tlc.gdc_launch_dependents()
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x_t = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                      mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
        acc += tl.dot(x_t, tl.trans(w), out_dtype=tl.float32)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),
             mask=m_mask[:, None])


# Court-certified winners keyed by the dense site's (N, K) at Muse Spark TP8:
# qkv_gate (2304, 8192), o (8192, 1024), pooler (4096, 8192), unpooler (8192, 4096).
_CU_CFG = {
    (2304, 8192): (1, 1, 2, 4),
    (8192, 1024): (1, 1, 2, 4),
    (4096, 8192): (1, 1, 2, 4),
    (8192, 4096): (1, 1, 2, 4),
}
_TRI_CFG = {
    (8192, 1024): (32, 128, 4, 4),
}

def small_m_gemm(x, weight):
    """Claim a [m <= 8, K] @ [N, K]^T bf16 GEMM at a certified shape; None = declined."""
    if torch.compiler.is_compiling():
        return None
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return None
    if not (x.is_cuda and weight.is_cuda):
        return None
    k = x.shape[-1]
    m = x.numel() // k
    if m > 8 or m == 0:
        return None
    n = weight.shape[0]
    if weight.shape[1] != k or not weight.is_contiguous():
        return None
    x2 = x.view(-1, k)
    if not x2.is_contiguous():
        return None
    out_shape = (*x.shape[:-1], n)
    if m == 1:
        cfg = _CU_CFG.get((n, k))
        if cfg is not None:
            MT, R, U, WPC = cfg
            if n % (R * WPC) == 0 and k % (256 * U) == 0:
                out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
                proj_ext().gemv_bf16_launch(weight, x2, out, 1, n, k, MT, R, U, WPC)
                return out.view(out_shape)
        return None
    cfg = _TRI_CFG.get((n, k))
    if cfg is not None:
        BN, BK, warps, stages = cfg
        if n % BN == 0 and k % BK == 0:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            _dot[(n // BN,)](x2, weight, out, m, n, K=k,
                             BLOCK_M=max(16, triton.next_power_of_2(m)),
                             BLOCK_N=BN, BLOCK_K=BK, PDL=SW_PDL,
                             num_warps=warps, num_stages=stages, launch_pdl=SW_PDL)
            return out.view(out_shape)
    return None
