"""OKC fused hi/lo router logits at T <= 256 (layer_s2 S2 router boundary graft + verify / small-extend T).

The fp32 router weight is split once per gate into a bf16 hi/lo pair (the same
decomposition the pin itself uses above 8 tokens); T == 1 runs both dots in one CUDA
GEMV over interleaved hi/lo rows, T = 2..256 runs the fused Triton split-K (T > 8: a token-tile grid dimension and
the per-bucket tiles in _MID; stock runs forward_split2 there, ~2x slower). Output is
fp32 logits [T, E], bit-compatible with what FusedMuseSparkRouting consumes.

Both split-K kernels store one fp32 partial per K slice and reduce_launch(FOLD=2) sums
the slices in a fixed order, so the logits are deterministic: every TP rank routes on
its own copy, and atomics could resolve a top-8 near-tie differently across ranks.
route_small runs that sum and the routing in one kernel (_cuda_src_route.py).

Measured at the kernel court: T=1 pair GEMV 8.3 us vs 13.5 us fp32 F.linear; the
Triton split-K was the proj lane's 3.19x m8 winner. Cache cost: hi + lo + interleaved
pairs = 3 bf16 copies of the [256, 8192] router = 12 MiB per MoE layer (~0.7 GiB per
rank over 61 MoE layers), built at weight load by prepare_gate (before KV-pool sizing,
outside any graph capture); a gate without them declines to the stock router.
"""

from typing import Optional

import os

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import proj_ext

MAX_TOKENS = 256
_SPLIT_K = 16  # K slices of the T <= 8 kernels (= fp32 partial planes reduced in order)
# T 9..256 (OKC_SW_ROUTER_MID=0 declines them to forward_split2): T upper edge -> (BLOCK_M, BLOCK_N, BLOCK_K,
# SPLIT_K, SWAP, warps, stages); best of serving/kernels/router/LOG.md E1-E2 per bucket, H200, cold L2.
_SW_MID = os.environ.get("OKC_SW_ROUTER_MID", "1") == "1"
_MID = ((16, (16, 64, 64, 32, 1, 4, 4)), (32, (32, 64, 64, 32, 1, 4, 4)), (64, (64, 32, 128, 16, 0, 4, 4)),
        (128, (64, 64, 128, 16, 1, 4, 3)), (256, (128, 64, 64, 16, 0, 8, 4)))


@triton.jit
def _router_fused(x_ptr, hi_ptr, lo_ptr, part_ptr, M,
                  N: tl.constexpr, K: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  SPLIT_K: tl.constexpr, SWAP: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_per = K // SPLIT_K
    k_start = pid_k * k_per
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    # SWAP: experts are the dot's M (wgmma M = 64 while the token tile is small)
    acc = tl.zeros((BLOCK_N, BLOCK_M) if SWAP else (BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_start + k_per, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x_t = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                      mask=m_mask[:, None], other=0.0)
        wh = tl.load(hi_ptr + offs_n[:, None] * K + offs_k[None, :])
        wl = tl.load(lo_ptr + offs_n[:, None] * K + offs_k[None, :])
        if SWAP:
            acc += tl.dot(wh, tl.trans(x_t), out_dtype=tl.float32)
            acc += tl.dot(wl, tl.trans(x_t), out_dtype=tl.float32)
        else:
            acc += tl.dot(x_t, tl.trans(wh), out_dtype=tl.float32)
            acc += tl.dot(x_t, tl.trans(wl), out_dtype=tl.float32)
    # this K slice's partial -> part[pid_k, m, n]; reduce_launch sums the slices in order
    if SWAP:
        tl.store(part_ptr + (pid_k * M + offs_m[None, :]) * N + offs_n[:, None], acc,
                 mask=m_mask[None, :])
    else:
        tl.store(part_ptr + (pid_k * M + offs_m[:, None]) * N + offs_n[None, :], acc,
                 mask=m_mask[:, None])


# gate weight (data_ptr, shape) -> (hi [N,K], lo [N,K], pairs [2N,K]) bf16 row-major.
_SPLITS = {}


def prepare_gate(gate) -> None:
    """Split the fp32 gate weight into its resident bf16 hi/lo copies at load time (the
    caller is MuseSpark1xForCausalLM.post_load_weights); router_logits_small only looks
    them up."""
    weight = gate.weight
    if weight.dtype is not torch.float32 or not weight.is_cuda or weight.dim() != 2:
        return
    if weight.shape[1] % (_SPLIT_K * 256):
        return  # neither kernel claims this K
    key = (weight.data_ptr(), tuple(weight.shape))
    if key in _SPLITS:
        return
    from sglang.kernels.ops.muse_spark_v12.okc_patch.moe import build_resident

    w = weight.data

    def build():
        hi = w.to(torch.bfloat16)
        lo = (w - hi.to(torch.float32)).to(torch.bfloat16)
        pairs = torch.empty((w.shape[0] * 2, w.shape[1]), dtype=torch.bfloat16, device=w.device)
        pairs[0::2] = hi
        pairs[1::2] = lo
        return {"hi": hi.contiguous(), "lo": lo.contiguous(), "pairs": pairs}

    with torch.no_grad():
        built = build_resident(build)
    _SPLITS[key] = (built["hi"], built["lo"], built["pairs"])


def _partials(gate, hidden_states):
    """(fp32 split-K partials [SK, T, E], T, E, SK); None = declined."""
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
    entry = _SPLITS.get((weight.data_ptr(), (ne, hidden)))
    if entry is None:
        return None
    hi, lo, pairs = entry
    x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    if tokens == 1 and hidden % (_SPLIT_K * 256 * 2) == 0:
        part = torch.empty((_SPLIT_K, tokens, ne), dtype=torch.float32, device=x.device)
        proj_ext().gemv_router_launch(pairs, x, part, tokens, hidden, 1, _SPLIT_K, 2, 2)
        return part, tokens, ne, _SPLIT_K
    if tokens <= 8:
        cfg = (max(16, triton.next_power_of_2(tokens)), 32, 256, _SPLIT_K, 0, 4, 3)
    elif _SW_MID:
        cfg = next(c for edge, c in _MID if tokens <= edge)
    else:
        return None
    bm, bn, bk, sk, swap, warps, stages = cfg
    if ne % bn or hidden % (sk * bk):
        return None
    part = torch.empty((sk, tokens, ne), dtype=torch.float32, device=x.device)
    _router_fused[(ne // bn, sk, triton.cdiv(tokens, bm))](
        x, hi, lo, part,
        tokens, N=ne, K=hidden,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, SWAP=swap,
        num_warps=warps, num_stages=stages)
    return part, tokens, ne, sk


def router_logits_small(gate, hidden_states) -> Optional[torch.Tensor]:
    """fp32 logits for T <= 256 from the bf16 hidden states; None = declined."""
    got = _partials(gate, hidden_states)
    if got is None:
        return None
    part, tokens, ne, sk = got
    logits = torch.empty((tokens, ne), dtype=torch.float32, device=part.device)
    proj_ext().reduce_launch(part, logits, tokens, ne, sk, 2)
    return logits


def route_small(gate, hidden_states, op, top_k, hist=None):
    """op's (topk_scores, topk_ids, packed ids) from the partials in one launch, bit-identical; None = declined."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import SW_PDL
    from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import route_ext

    bias = op.expert_bias
    if (top_k != 8 or gate.weight.shape[0] != 256 or op.router_score_normalization != "sigmoid"
            or op.local_expert_mapping is not None or op.num_valid_tokens is not None or op.sol_topk_out is not None
            or bias.dtype is not torch.float32 or not bias.is_contiguous() or bias.numel() != 256):
        return None
    got = _partials(gate, hidden_states)
    if got is None:
        return None
    part, tokens, _, sk = got                          # sk 16 or 32
    tw =torch.empty((tokens, 8), dtype=torch.float32, device=part.device)
    ti = torch.empty((tokens, 8), dtype=torch.int32, device=part.device)
    tp = torch.empty((tokens, 8), dtype=torch.int32, device=part.device)
    mult = op.router_output_multiplier
    eps = op.router_score_division_eps
    route_ext().route(part, bias, tw, ti, tp, tokens, sk, float(mult if mult is not None else 1.0),
                      float(op.route_scale), float(eps if eps is not None else 0.0), 1 if SW_PDL else 0,
                      hist if hist is not None else part.new_empty((0,), dtype=torch.int32))
    return tw, ti, tp
