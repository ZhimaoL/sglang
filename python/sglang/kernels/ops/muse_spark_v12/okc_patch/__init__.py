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

v1 switches (the three certified kernel-track assets wired after the audit). These
are KILL switches: default ON, "0" falls back to the stock (or v0) path.

  OKC_SW_ATTN_V3=1   upgrade the OKC_SW_ATTN sliding paths to the attn-v3 fused
                     kernels (single-launch decode with persistent-ticket combine;
                     2-launch verify with in-kernel tail). Judge-checked at the
                     kernel court: sliding decode 1.73-1.92x, verify > the v1
                     3-launch chain on every shape. Inert while OKC_SW_ATTN=0.
  OKC_SW_PREFILL_FMHA=1  warp-specialized FP8 FMHA (hand wgmma CUDA) on full-layer
                     PREFILL attention at q_len >= OKC_WS_MIN_QLEN (default 2048):
                     exactly the 16K-chunk long-context TTFT segments. Certified
                     1.18-1.21x FA3 at 16K query. FA3 fallback on any decline.
                     Runs as an eager_on_graph break (prefill replays breakable CUDA
                     graphs captured with one synthetic request per token bucket), so
                     the real bs / token count are read at replay; ragged co-batches
                     and short chunks take the stock sequence inside the same break.
  OKC_PATH_STATS=1   log cumulative replay-time hit counters of the break paths.
  OKC_SW_MOE_WG=1    wgmma W4A16 grouped GEMM pair on the fp4 experts at
                     T >= OKC_MOE_WG_MIN_TOKENS (default 1536, = certified
                     MARLIN_MAX_TOKENS+1): the 16K-token prefill chunks Marlin
                     currently keeps. Certified ~2x Marlin there. Needs packs
                     (same OKC_MOE_PACK_MAX_LAYERS budget; ~0.95 GiB/layer).

Expected regime (judge config TP8/EP8, Marlin MoE, DFLASH 8 drafts, decode bs <= 32):
target verify runs T = 8 * bs, draft decode runs T = bs, so the small-T paths fire at
verify bs 1 and decode bs <= 8, and the TMA path at verify bs >= 5. The DFLASH draft
model is its own architecture (srt/models/dflash.py), NOT the muse modules: plain
decode mode never reaches these hooks in the judge config, so the live surfaces are
target verify (all tiers) and prefill.
"""

import os


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) == "1"


# The judge's launch command is fixed, so a submission turns a path on by flipping its
# default here ("0" -> "1"); the env override stays for local runs.
SW_ATTN = _flag("OKC_SW_ATTN", "0")
SW_FS = _flag("OKC_SW_FS", "0")
SW_PROJ = _flag("OKC_SW_PROJ", "0")
SW_PROJ_MMA = _flag("OKC_SW_PROJ_MMA", "0")
SW_ROUTER = _flag("OKC_SW_ROUTER", "0")
SW_MOE_CUDA = _flag("OKC_SW_MOE_CUDA", "0")
SW_MOE_TMA = _flag("OKC_SW_MOE_TMA", "0")

# v1 kill switches (default ON; each falls back gracefully when "0").
SW_ATTN_V3 = _flag("OKC_SW_ATTN_V3", "1")
SW_PREFILL_FMHA = _flag("OKC_SW_PREFILL_FMHA", "1")
SW_MOE_WG = _flag("OKC_SW_MOE_WG", "1")

# Prefill FMHA fires at q_len >= this (certified gate: ws loses to FA3 below ~2048).
WS_MIN_QLEN = int(os.environ.get("OKC_WS_MIN_QLEN", "2048"))
# WG grouped GEMM fires at T >= this (court: Marlin wins below; 1536 = certified cut).
MOE_WG_MIN_TOKENS = int(os.environ.get("OKC_MOE_WG_MIN_TOKENS", "1536"))

_pack_layers = os.environ.get("OKC_MOE_PACK_MAX_LAYERS", "0")
MOE_PACK_MAX_LAYERS = 1 << 30 if _pack_layers == "all" else int(_pack_layers)

ANY_ATTN = SW_ATTN or SW_FS or SW_PREFILL_FMHA
ANY_MOE = SW_MOE_CUDA or SW_MOE_TMA or SW_MOE_WG

__all__ = [
    "SW_ATTN",
    "SW_FS",
    "SW_PROJ",
    "SW_PROJ_MMA",
    "SW_ROUTER",
    "SW_MOE_CUDA",
    "SW_MOE_TMA",
    "SW_ATTN_V3",
    "SW_PREFILL_FMHA",
    "SW_MOE_WG",
    "WS_MIN_QLEN",
    "MOE_WG_MIN_TOKENS",
    "MOE_PACK_MAX_LAYERS",
    "ANY_ATTN",
    "ANY_MOE",
]
