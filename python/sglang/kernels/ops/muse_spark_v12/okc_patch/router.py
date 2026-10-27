"""OKC fused hi/lo router logits at T <= 8 (layer_s2 S2 router boundary graft).

The fp32 router weight is split once per gate into a bf16 hi/lo pair (the same
decomposition the pin itself uses above 8 tokens); T == 1 runs both dots in one CUDA
GEMV over interleaved hi/lo rows, T = 2..8 runs the fused Triton split-K. Output is
fp32 logits [T, E], bit-compatible with what FusedMuseSparkRouting consumes.

Measured at the kernel court: T=1 pair GEMV 8.3 us vs 13.5 us fp32 F.linear; the
Triton split-K was the proj lane's 3.19x m8 winner. Cache cost: hi + lo + interleaved
pairs = 3 bf16 copies of the [256, 8192] router = 12 MiB per MoE layer (~0.7 GiB per
rank over 61 MoE layers).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import proj_ext

MAX_TOKENS = 8


@triton.jit
def _router_fused(x_ptr, hi_ptr, lo_ptr, out_ptr, M,
                  N: tl.constexpr, K: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_per = K // SPLIT_K
    k_start = pid_k * k_per
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_start + k_per, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x_t = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                      mask=m_mask[:, None], other=0.0)
        wh = tl.load(hi_ptr + offs_n[:, None] * K + offs_k[None, :])
        wl = tl.load(lo_ptr + offs_n[:, None] * K + offs_k[None, :])
        acc += tl.dot(x_t, tl.trans(wh), out_dtype=tl.float32)
        acc += tl.dot(x_t, tl.trans(wl), out_dtype=tl.float32)
    tl.atomic_add(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
                  mask=m_mask[:, None])


# gate weight (data_ptr, shape) -> (hi [N,K], lo [N,K], pairs [2N,K]) bf16 row-major.
_SPLITS = {}


def _hi_lo_pairs(weight):
    key = (weight.data_ptr(), tuple(weight.shape))
    entry = _SPLITS.get(key)
    if entry is None:
        with torch.no_grad():
            hi = weight.to(torch.bfloat16)
            lo = (weight - hi.to(torch.float32)).to(torch.bfloat16)
            hi = hi.contiguous()
            lo = lo.contiguous()
            pairs = torch.empty(
                (hi.shape[0] * 2, hi.shape[1]), dtype=torch.bfloat16, device=hi.device
            )
            pairs[0::2] = hi
            pairs[1::2] = lo
            entry = (hi, lo, pairs.contiguous())
            _SPLITS[key] = entry
    return entry


def router_logits_small(gate, hidden_states) -> Optional[torch.Tensor]:
    """fp32 logits for T <= 8 from the bf16 hidden states; None = declined."""
    if torch.compiler.is_compiling():
        return None
    weight = gate.weight
    if weight.dtype is not torch.float32 or hidden_states.dtype is not torch.bfloat16:
        return None
    if not hidden_states.is_cuda or hidden_states.dim() != 2:
        return None
    tokens, hidden = hidden_states.shape
    if tokens == 0 or tokens > MAX_TOKENS:
        return None
    ne = weight.shape[0]
    if weight.shape[1] != hidden:
        return None
    x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    if tokens == 1 and hidden % (16 * 256 * 2) == 0:
        hi, lo, pairs = _hi_lo_pairs(weight)
        logits = torch.zeros((tokens, ne), dtype=torch.float32, device=x.device)
        proj_ext().gemv_router_launch(pairs, x, logits, tokens, hidden, 1, 16, 2, 2)
        return logits
    if ne % 32 == 0 and hidden % (16 * 256) == 0:
        hi, lo, pairs = _hi_lo_pairs(weight)
        logits = torch.zeros((tokens, ne), dtype=torch.float32, device=x.device)
        _router_fused[(ne // 32, 16)](
            x, hi, lo, logits,
            tokens, N=ne, K=hidden,
            BLOCK_M=max(16, triton.next_power_of_2(tokens)),
            BLOCK_N=32, BLOCK_K=256, SPLIT_K=16,
            num_warps=4, num_stages=3)
        return logits
    return None
