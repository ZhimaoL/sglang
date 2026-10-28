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
# Split-KV partial accumulators of the attn-v3 sliding kernels: "0" (default) keeps them fp32 like FA3; "1" stores
# them bf16 (the certified attn-v3 setting: half the last-arriver combine's reads).
SW_ATTN_ACC16 = _flag("OKC_SW_ATTN_ACC16", "0")
# Sequence-parallel prefill on Hopper (layers/layernorm_sp.py okc_hopper_sp): the residual stream, norms, router,
# pooler and unpooler run on 1/tp of the tokens; the experts see every token through an all-gather and return a
# reduce-scatter, so the bytes on the wire match the two all-reduces they replace.
SW_SP = _flag("OKC_SW_SP", "1")
# SP only above this many tokens (a prefill CUDA-graph bucket boundary, so a batch and its padded bucket agree): tiny
# extends lost ~100 ms to collective waits under SP (sessions SB / SC), a ~1K-token extend is 75 ms with SP vs 80 without.
SP_MIN_TOKENS = int(os.environ.get("OKC_SP_MIN_TOKENS", "512"))
# Replicated pooler / unpooler at decode and verify M (<= 512 tokens) on Hopper: column-parallel slice + the
# multicast all-gather (muse_spark_v12/gemm_ag.py), which upstream enables above 8 tokens on B300 only. Every
# rank otherwise streams the full 128 MiB of the pair per layer and step (session SC: ~2 ms of a bs=1 step).
SW_GEMM_AG = _flag("OKC_SW_GEMM_AG", "1")
# Up to this many tokens (verify bs <= 16): session SE measured the per-step saving shrinking with M (bs 1 / 4 / 16:
# -2.6 / -2.2 / -0.6 ms) and the c32 rung 3.4% below the replicated GEMM.
GEMM_AG_MAX_TOKENS = int(os.environ.get("OKC_GEMM_AG_MAX_TOKENS", "128"))
# SP participant linears through torch's symmetric-memory fused all-gather+matmul / matmul+reduce-scatter
# (async TP, copy-engine overlap) on forwards of >= OKC_SP_SYMM_MIN_TOKENS tokens, NCCL below: session SB measured
# the 74K first turn -6.0% but cached follow-up TTFT +54 ms (bs 1) / +70 ms (bs 4) with it on every forward.
SP_SYMM = _flag("OKC_SP_SYMM", "1")
SP_SYMM_MIN_TOKENS = int(os.environ.get("OKC_SP_SYMM_MIN_TOKENS", "4096"))
# SP prefill MoE block with its collectives on the copy engines (okc_patch/sp_moe_ovl.py): pooled AG and RS move by
# symm-mem P2P pushes under the row-sliced pooler / finalize / unpooler (pg whole) instead of exposed NCCL collectives.
SW_MOE_OVL = _flag("OKC_SW_MOE_OVL", "0")
# NCCL SP MoE section: the pre-expert RMSNorm (per row) on the shard before the all-gather, not on the gathered chunk.
SP_PRENORM_SHARD = _flag("OKC_SP_PRENORM_SHARD", "1")

# Prefill attention (prefill_fmha.py): the prefill lane's FP8 FMHA (fmha7) in place of FA3's extend call. Per request:
# full layers from OKC_PFMHA_MIN_TOKENS new tokens, sliding layers from OKC_PFMHA_SLIDING_MIN_TOKENS (1xH200 over a 75K
# prefix: full 0.85x FA3 at 32 tokens, 1.07x at 64, 1.39x at 128 .. 2.6x at 2048; sliding 0.85x at 512, 1.16x at 640,
# 1.17x at 753, 1.45-1.68x from 1024).
# A co-batch (bs <= OKC_PFMHA_RAGGED_MAX_BS, every request over the threshold) runs one call per request (1.55-1.93x
# FA3's batched call at bs 2-8 of 753-1500-token follow-ups).
SW_PREFILL_FMHA = _flag("OKC_SW_PREFILL_FMHA", "1")
PFMHA_MIN_TOKENS = int(os.environ.get("OKC_PFMHA_MIN_TOKENS", "128"))
PFMHA_SLIDING = _flag("OKC_PFMHA_SLIDING", "1")
PFMHA_SLIDING_MIN_TOKENS = int(os.environ.get("OKC_PFMHA_SLIDING_MIN_TOKENS", "640"))
PFMHA_RAGGED = _flag("OKC_PFMHA_RAGGED", "1")
PFMHA_RAGGED_MAX_BS = int(os.environ.get("OKC_PFMHA_RAGGED_MAX_BS", "32"))
# Every exp2 of the softmax on MUFU ex2.f16x2 (no piecewise-linear exp; the scores stay f16): a cleaner-numerics
# build of the same kernels, compiled separately when on.
PFMHA_EXACT = _flag("OKC_PREFILL_FMHA_EXACT", "0")

# Verify-step glue (serving/kernels/verify-glue): at <= 2048 routes a pg-only layer runs one routing launch (pg_route:
# local ids + block list, in place of the int32 cast / expert-map gather / moe_align / pg_tiles) and folds the
# pre-expert RMSNorm into the token gather (bit-identical to F.rms_norm).
SW_PG_SMALL = _flag("OKC_SW_PG_SMALL", "1")
# Programmatic dependent launch on the OKC kernels inside verify / decode graphs (the next kernel's launch and
# prologue overlap the previous one's tail; every kernel waits on griddepcontrol before reading its inputs).
SW_PDL = _flag("OKC_SW_PDL", "1")
# DFlash context projection fc (5 x 8192 -> 8192, replicated 640 MiB bf16, 159 us per verify step) as a column-split
# GEMM + the multicast all-gather (gemm_ag) at <= GEMM_AG_MAX_TOKENS tokens.
SW_DFLASH_FC_AG = _flag("OKC_SW_DFLASH_FC_AG", "1")
# The same column split for the DFlash grouped convs' kernel_projection (replicated nn.Linear, 2 per draft layer, ~11 us
# each per verify step); 8-GPU check (session SI vg-micro): slices bit-identical, [3072, 8192] 16.5 -> 10.4 us at m 8.
SW_DFLASH_CONV_AG = _flag("OKC_SW_DFLASH_CONV_AG", "1")
# BF16 expert layers (0 and 61 of the NVFP4 export): local ids by one table lookup and the fused finalize kernel at
# every token count (the wrapper's torch chain below 32 tokens is 11 launches).
SW_BF16_MOE_GLUE = _flag("OKC_SW_BF16_MOE_GLUE", "1")
# The last layer's FFN boundary (post-FFN norm, gated residual) + the model's final norm as one bit-identical kernel
# (okc_patch.glue.tail_boundary) instead of 9 torch launches.
SW_TAIL_NORM = _flag("OKC_SW_TAIL_NORM", "1")
# Target verify argmax over the vocab shard with one program per 4096-column block (softcap_row_argmax_split) instead of
# one per row; same first-maximum result after the gather.
SW_ARGMAX_SPLIT = _flag("OKC_SW_ARGMAX_SPLIT", "1")
# DFlash candidate_topk: one all-gather of packed value bits + ids instead of two.
SW_CAND_AG_MERGE = _flag("OKC_SW_CAND_AG_MERGE", "1")

# Hopper cluster cap for sglang's fused norm-residual boundary (norm_residual.py): upstream runs the one-CTA-per-row
# kernel unless capability >= 10; > 1 allows the cluster kernel (GB300 policy C = 16 / 8 / 4 / 2 at M <= 8 / 16 / 32
# / 64) up to this size on sm_90 at M <= 32 (dev2 VG4, SI vg-micro: -1.0 .. -1.8 us per boundary). 1 = upstream.
HOPPER_NORM_MAX_CLUSTER = int(os.environ.get("OKC_HOPPER_NORM_MAX_CLUSTER", "16"))
# sglang's fused multicast all-gather + FFN norm-residual boundary (all_gather_norm_residual.py) on sm_90: the
# unpooler's column slices are gathered inside the boundary kernel instead of a separate all_gather launch (upstream
# gates it to capability >= 10). Cluster size follows the Hopper rule above.
SW_AG_NORM_HOPPER = _flag("OKC_SW_AG_NORM_HOPPER", "1")
# FA3 backend (the DFlash draft's attention; the target's full-attention layers at bs >= 2): FP8 casts of q / k / v and
# the paged KV write as one fused_fp8_qkv_kv_cache launch (the TRT-LLM backend's kernel) instead of three casts +
# store_kvcache, unscaled FP8 pools only.
SW_FA3_FUSED_KV = _flag("OKC_SW_FA3_FUSED_KV", "1")

# TP8 graph-mode all-reduce: multicast 2-shot (copy in / out of the workspace) only from this size; below it the
# in-place 2-shot pull (configs/custom_all_reduce_v2.py, SI vg-micro). Upstream 524288.
AR_TP8_GRAPH_MC_MIN_BYTES = int(os.environ.get("OKC_AR_TP8_GRAPH_MC_MIN_BYTES", str(2 << 20)))

ANY_ATTN = SW_ATTN or SW_FS

__all__ = ["SW_MOE_PG", "MOE_PG_MIN_TOKENS", "MOE_PG_ONLY", "MOE_PACK_MAX_LAYERS",
           "SW_ATTN", "SW_FS", "SW_PROJ", "SW_PROJ_MMA", "SW_ROUTER", "SW_ATTN_V3", "SW_ATTN_ACC16",
           "SW_PREFILL_FMHA", "PFMHA_MIN_TOKENS", "PFMHA_SLIDING", "PFMHA_SLIDING_MIN_TOKENS", "PFMHA_RAGGED",
           "PFMHA_RAGGED_MAX_BS", "PFMHA_EXACT",
           "SW_SP", "SP_MIN_TOKENS", "SW_GEMM_AG", "GEMM_AG_MAX_TOKENS", "SP_SYMM", "SP_SYMM_MIN_TOKENS", "SW_MOE_OVL",
           "SP_PRENORM_SHARD", "SW_PG_SMALL", "SW_PDL", "SW_DFLASH_FC_AG", "SW_DFLASH_CONV_AG", "SW_BF16_MOE_GLUE",
           "SW_TAIL_NORM", "SW_ARGMAX_SPLIT", "SW_CAND_AG_MERGE", "HOPPER_NORM_MAX_CLUSTER",
           "SW_AG_NORM_HOPPER", "SW_FA3_FUSED_KV",
           "AR_TP8_GRAPH_MC_MIN_BYTES", "ANY_ATTN"]
