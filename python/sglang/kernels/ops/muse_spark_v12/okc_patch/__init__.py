"""OKC serving patch for Muse Spark: the pg grouped-GEMM MoE pair on prefill chunks.

  OKC_SW_MOE_PG=1    pg grouped GEMM pair (prefill bank plt_c3, BF16 token images and
                     intermediate, NVFP4 weights decoded in registers) on the fp4 experts at
                     T >= OKC_MOE_PG_MIN_TOKENS: the long prefill chunks. Kill switch, default on.
  OKC_MOE_PG_MIN_TOKENS=1536
  OKC_MOE_PACK_MAX_LAYERS=N  how many MoE layers (first N in load order) build the pg packs,
                     0.94 GiB per layer per rank taken from the KV pool; "all" = every layer.
  OKC_MOE_PG_ONLY=1  (default) packed layers drop their Marlin copy and run pg at every token count
                     (verify steps included), so every layer fits: +0.1 GiB per layer over Marlin.
                     0 keeps Marlin and uses pg on the prefill chunks of the first N layers only.

Switches are read once at import, so dispatch is stable across CUDA-graph capture and replay.
The judge's launch command is fixed, so the submitted defaults are the ones below.
"""

import os


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) == "1"


SW_MOE_PG = _flag("OKC_SW_MOE_PG", "1")
# Measured on 8x H200: the pg pair beats Marlin on 16K-token chunks; Marlin keeps smaller prefills.
MOE_PG_MIN_TOKENS = int(os.environ.get("OKC_MOE_PG_MIN_TOKENS", "1536"))
MOE_PG_ONLY = _flag("OKC_MOE_PG_ONLY", "1")
_pack_layers = os.environ.get("OKC_MOE_PACK_MAX_LAYERS", "all" if MOE_PG_ONLY else "16")
MOE_PACK_MAX_LAYERS = 1 << 30 if _pack_layers == "all" else int(_pack_layers)


# Verify-side kernel stack (kernel-track grafts): fused sliding-window / full attention for decode and verify,
# the m<=8 dense projections and the small-T router. All four modules stage BF16 activations only
# (docs/COMPLIANCE-FP16-1025.md) and measured c1 +5.6% / c4 +2.1% with consistency 0.943-0.949 (session Q).
SW_ATTN = _flag("OKC_SW_ATTN", "1")
SW_FS = _flag("OKC_SW_FS", "1")
SW_PROJ = _flag("OKC_SW_PROJ", "1")
SW_PROJ_MMA = _flag("OKC_SW_PROJ_MMA", "1")
SW_ROUTER = _flag("OKC_SW_ROUTER", "1")
# Kill switches of the v1 attention paths (default on; "0" falls back to the older chain).
SW_ATTN_V3 = _flag("OKC_SW_ATTN_V3", "1")
# Prefill FMHA hook: measured neutral on the ladder (session K), off so the patch carries no prefill attention.
SW_PREFILL_FMHA = _flag("OKC_SW_PREFILL_FMHA", "0")
WS_MIN_QLEN = int(os.environ.get("OKC_WS_MIN_QLEN", "2048"))
# Sequence-parallel prefill on Hopper (layers/layernorm_sp.py okc_hopper_sp): the residual stream, norms, router,
# pooler and unpooler run on 1/tp of the tokens; the experts see every token through an all-gather and return a
# reduce-scatter, so the bytes on the wire match the two all-reduces they replace.
SW_SP = _flag("OKC_SW_SP", "1")
# Measurement switch: SP participant linears through torch's symmetric-memory fused all-gather+matmul /
# matmul+reduce-scatter (async TP, copy-engine overlap) instead of separate NCCL collectives.
SP_SYMM = _flag("OKC_SP_SYMM", "0")

ANY_ATTN = SW_ATTN or SW_FS

__all__ = ["SW_MOE_PG", "MOE_PG_MIN_TOKENS", "MOE_PG_ONLY", "MOE_PACK_MAX_LAYERS",
           "SW_ATTN", "SW_FS", "SW_PROJ", "SW_PROJ_MMA", "SW_ROUTER", "SW_ATTN_V3",
           "SW_PREFILL_FMHA", "WS_MIN_QLEN", "SW_SP", "SP_SYMM", "ANY_ATTN"]
