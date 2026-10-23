"""OKC serving patch: opt-in Muse Spark decode/verify kernels (kernel-track S2 graft).

Every path is behind an env kill switch, default OFF, so the unpatched behavior is
bit-identical until a switch is set. Switches are read once at import (before server
start), so dispatch decisions are stable across CUDA-graph capture and replay.

  OKC_SW_ATTN=1      fused sliding-window attention (qk norm + RoPE + KV store +
                     flash-decode + gated output norm) on sliding layers at
                     q_len <= 8; certified 1.44-1.48x per sliding call at the
                     kernel court. Seam: MuseSpark1xAttention.gated_attention.
  OKC_SW_FS=1        physical-order full-attention streamer for decode/verify-bs1
                     on full layers; the phys->(req,rank) counting-sort metadata is
                     built in-graph once per step by the first full layer and
                     consumed by the rest (16 full layers share one build).
                     Seam: MuseSpark1xAttention.gated_attention.
  OKC_SW_PROJ=1      m == 1 FFMA streaming GEMV for the dense projections
                     (qkv_gate / o / pooler / unpooler shapes); no weight copies.
                     Seam: unquant bf16 GEMM dispatcher.
  OKC_SW_PROJ_MMA=1  m = 2..8 tensor-core/Triton paths for the same sites; packs a
                     B-fragment copy of each tuned weight (see proj.py for the
                     per-layer memory cost). Requires OKC_SW_PROJ=1.
  OKC_SW_ROUTER=1    fused hi/lo router logits at T <= 8 (CUDA GEMV at T == 1,
                     Triton split-K at T = 2..8) in place of fp32 F.linear.
                     Seam: MuseSpark1xMoE._forward_experts.
  OKC_SW_MOE_CUDA=1  v14 CUDA W4A16 chain (K0f -> k1 -> k2 -> K3c) at T <= 8 on the
                     NVFP4-Marlin experts. Needs OKC packs captured at weight prep.
  OKC_SW_MOE_TMA=1   v14 Triton TMA chain at T > 32 (verify bs >= 5). Same packs
                     precondition. 9 <= T <= 32 always stays on Marlin (measured tie).
  OKC_MOE_PACK_MAX_LAYERS=N  cap on how many MoE layers build the extra packs
                     (first N eligible layers in load order); default 0 = none,
                     "all" = every layer. See moe.py for the per-layer GiB cost.

Expected regime (judge config TP8/EP8, Marlin MoE, DFLASH 8 drafts, decode bs <= 32):
target verify runs T = 8 * bs, draft decode runs T = bs, so the small-T paths fire at
verify bs 1 and decode bs <= 8, and the TMA path at verify bs >= 5.
"""

import os


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


SW_ATTN = _flag("OKC_SW_ATTN")
SW_FS = _flag("OKC_SW_FS")
SW_PROJ = _flag("OKC_SW_PROJ")
SW_PROJ_MMA = _flag("OKC_SW_PROJ_MMA")
SW_ROUTER = _flag("OKC_SW_ROUTER")
SW_MOE_CUDA = _flag("OKC_SW_MOE_CUDA")
SW_MOE_TMA = _flag("OKC_SW_MOE_TMA")

_pack_layers = os.environ.get("OKC_MOE_PACK_MAX_LAYERS", "0")
MOE_PACK_MAX_LAYERS = 1 << 30 if _pack_layers == "all" else int(_pack_layers)

ANY_ATTN = SW_ATTN or SW_FS
ANY_MOE = SW_MOE_CUDA or SW_MOE_TMA

__all__ = [
    "SW_ATTN",
    "SW_FS",
    "SW_PROJ",
    "SW_PROJ_MMA",
    "SW_ROUTER",
    "SW_MOE_CUDA",
    "SW_MOE_TMA",
    "MOE_PACK_MAX_LAYERS",
    "ANY_ATTN",
    "ANY_MOE",
]
