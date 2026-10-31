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
  OKC_MOE_REPLICAS=1 hot-expert replicas per rank per pg-only layer (0..3), 1.8 GiB per rank per replica.

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
# Hot-expert replicas against the verify EP straggler (every rank waits at the MoE all-reduce for the one with the
# most active experts): forwards of <= OKC_MOE_REP_MAX_TOKENS tokens give each active replicated expert to the holder
# with fewer active experts (pg_moe.assign_replicas), longer ones (OKC_MOE_REP_PREFILL) split its routes between the
# two holders at a token boundary that levels routed tokens (pg_moe.split_replicas).
MOE_REPLICAS = int(os.environ.get("OKC_MOE_REPLICAS", "1"))
# runtime-adaptive replicas (okc_patch/adaptive.py): replica sets decided from the run's own routing counts, refreshed
# between forwards; first after ADAPT_FIRST_TOKENS prefill tokens and ADAPT_FIRST_VERIFY verify tokens (so the first,
# full decision sees verify counts; prefill-only traffic such as max_new_tokens 1 never fires it), then every
# ADAPT_EVERY_TOKENS; ring of the last ADAPT_RING prefill forwards (SP shards or T >= ADAPT_MIN_T; 64 ~ 14 first turns:
# the decisions price the prefill straggler on these rows, and 16 left them on the last ~3); readiness checked every
# ADAPT_CHECK forwards.
ADAPT_REPLICAS = _flag("OKC_MOE_ADAPTIVE_REPLICAS", "0")
ADAPT_FIRST_TOKENS = int(os.environ.get("OKC_ADAPT_FIRST_TOKENS", "65536"))
ADAPT_EVERY_TOKENS = int(os.environ.get("OKC_ADAPT_EVERY_TOKENS", "524288"))
ADAPT_FIRST_VERIFY = int(os.environ.get("OKC_ADAPT_FIRST_VERIFY", "4096"))
ADAPT_RING = int(os.environ.get("OKC_ADAPT_RING", "64"))
ADAPT_MIN_T = int(os.environ.get("OKC_ADAPT_MIN_T", "4096"))
ADAPT_CHECK = int(os.environ.get("OKC_ADAPT_CHECK", "32"))
# hysteresis: a slot change must lower the layer objective (% official) by more than ADAPT_THETA per expert copy;
# at most ADAPT_MAX_SWAPS incremental changes per layer per refresh; a full refresh may also take the from-scratch
# greedy plan: refresh 1, and every refresh at which the prefill tokens seen have grown ADAPT_FULL_GROWTH x since the
# last full one (0: refresh 1 only); the applied changes trickle in at <= ADAPT_TRICKLE expert copies per forward.
ADAPT_THETA = float(os.environ.get("OKC_ADAPT_THETA", "0.001"))
ADAPT_MAX_SWAPS = int(os.environ.get("OKC_ADAPT_MAX_SWAPS", "4"))
ADAPT_TRICKLE = int(os.environ.get("OKC_ADAPT_TRICKLE", "8"))
ADAPT_FULL_GROWTH = float(os.environ.get("OKC_ADAPT_FULL_GROWTH", "4"))
MOE_REP_MAX_TOKENS = int(os.environ.get("OKC_MOE_REP_MAX_TOKENS", "256"))
MOE_REP_PREFILL = _flag("OKC_MOE_REP_PREFILL", "1")
# on the st chain the verify balancer weighs an expert by its st passes ceil(routes / 8); 0 = every active expert 1
MOE_REP_PASSES = _flag("OKC_MOE_REP_PASSES", "1")
# the split from this size up: its kernels cost 9-15 us per layer, which at prefill routing it repays only on 16K
# chunks (S8 arm C ~1K extends: <= 1 us of straggler saved; 1xH200 bench: +5..+21 us at T 753..5632, -20 at 16384)
MOE_REP_SPLIT_MIN_TOKENS = int(os.environ.get("OKC_MOE_REP_SPLIT_MIN_TOKENS", "8192"))
# Between them, from OKC_MOE_REP_EXTEND_MIN_TOKENS (extends): the balancer inside the local-slot align (pg_moe.align_local,
# pg cost weights, count order). Slowest rank per layer at R1: T 1024 -9, 4096 -31 us; T 512 no gain (serving-replicas M29).
MOE_REP_EXTEND = _flag("OKC_MOE_REP_EXTEND", "1")
MOE_REP_EXTEND_MIN_TOKENS = int(os.environ.get("OKC_MOE_REP_EXTEND_MIN_TOKENS", "1024"))
# pg's narrow build (-DPGN_WSET: 128-token blocks, widths 16 / 64 / 128) for calls of OKC_PG_NARROW_MIN_T..MAX_T
# tokens (serving/kernels/extend E5-E7: 0.77-0.80x at verify T 8..256, 0.86-0.97x at extends of 753..3072); the
# shipped build (256-token blocks) outside. Same outputs for every real token.
PG_NARROW_MIN_T = int(os.environ.get("OKC_PG_NARROW_MIN_T", "1"))
PG_NARROW_MAX_T = int(os.environ.get("OKC_PG_NARROW_MAX_T", "3072"))
# ... and inside that range, calls of <= OKC_PG64_MAX_T tokens on its 64-token-block instance (-DPGN_XNT=64, widths
# 16 / 32 / 64; serving/kernels/extend LOG E19-E20: -5..-7 % of the slowest rank's MoE at verify T 136-256,
# bit-identical). 0 = off.
PG64_MAX_T = int(os.environ.get("OKC_PG64_MAX_T", "256"))
# ... and that instance's gg1 on four consumer warpgroups on the small front's calls (cuda_ext.pg_n4_ext / _cuda_src_pgn4.py:
# the prefill lane's pg_n4, 256-row items over shared X k-tiles, bit-identical); 0 = the 2-warpgroup kernel.
PG64_N4 = _flag("OKC_PG64_N4", "1")
# pg_gather_norm (the small front end's normed token images) splits each block's token rows over this many CTAs: the
# per-warp row loop was the kernel's latency chain (LOG E21: 11-14 -> 7-10 us at T 96-256, outputs unchanged).
PG_GN_SPLIT = int(os.environ.get("OKC_PG_GN_SPLIT", "2"))
# st pair (okc_patch/st_moe.py: the same packs read by mma.sync n8 tiles) on pg-only layers at T <=
# OKC_MOE_ST_MAX_TOKENS (verify bs 1..9); OKC_MOE_ST_PDL launches its down kernel with PDL.
SW_MOE_ST = _flag("OKC_SW_MOE_ST", "1")
# 72: st beats the narrow small front by 6-15 % at T 40..72, 3 % at 80, loses from 96 (smallt-moe LOG E17 / E21).
MOE_ST_MAX_TOKENS = min(128, int(os.environ.get("OKC_MOE_ST_MAX_TOKENS", "72")))
MOE_ST_PDL = _flag("OKC_MOE_ST_PDL", "1")
# st folds the pre-expert RMSNorm into its gate/up kernel (bit-identical to F.rms_norm, as pg_gather_norm), saving the
# separate norm launch on every st layer.
SW_ST_NORM_FOLD = _flag("OKC_SW_ST_NORM_FOLD", "1")
# M1 (okc_patch/m1.py): the verify MoE combine from gg2's epilogue (multimem rows + a receiver; no finalize / all-reduce)
# for pg-small calls of M1_MIN_T..M1_MAX_T tokens (st below)
SW_M1 = _flag("OKC_SW_M1", "0")
M1_MIN_T = int(os.environ.get("OKC_M1_MIN_T", "73"))
M1_MAX_T = int(os.environ.get("OKC_M1_MAX_T", "256"))
# M1-RS (okc_patch/m1.py rs_*): the SP prefill / extend MoE combine from gg2's epilogue in the reduce-scatter form (each row
# to its token's home rank + a receiver there; no finalize / reduce-scatter) for gathered calls of M1RS_MIN_T..MAX_T tokens
SW_M1RS = _flag("OKC_SW_M1RS", "1")
M1RS_MIN_T = int(os.environ.get("OKC_M1RS_MIN_T", "512"))
M1RS_MAX_T = int(os.environ.get("OKC_M1RS_MAX_T", "16384"))
# with the fold on, the pooler's gather emits the pre-expert norm (once per token, same values; court C39)
SW_AG_PRENORM = _flag("OKC_SW_AG_PRENORM", "1")
# st runs pg_rep_assign's balancer (or the home map) in its own CTAs, saving that launch (bit-identical).
SW_ST_REP_FOLD = _flag("OKC_SW_ST_REP_FOLD", "1")


# Verify-side kernel stack (kernel-track grafts): fused sliding-window / full attention for decode and verify,
# the m<=8 dense projections and the small-T router. All four modules stage BF16 activations only
# (docs/COMPLIANCE-FP16-1025.md) and measured c1 +5.6% / c4 +2.1% with consistency 0.943-0.949 (session Q).
SW_ATTN = _flag("OKC_SW_ATTN", "1")
SW_PROJ = _flag("OKC_SW_PROJ", "1")
# Small-m dense GEMMs at verify m (okc_patch/smallm.py): TMA + wgmma straight off the resident row-major bf16 weight
# for qkv_gate (m <= 16) and the gemm_ag pooler / unpooler slices (m <= 64 / 32); off, those take F.linear (the old
# qkv_gate MMA-pack route measured the same 16.7 us at m 8 and is not carried).
SW_SMALLM = _flag("OKC_SW_SMALLM", "1")
# W-staged own GEMMs at verify m <= 32 (okc_patch/wsc2.py; muse-opt serving/kernels/megakernel LOG M5-M5.4): qkv_gate at
# m 17-32 (KQ3) and o_proj at m 1-32 (KO2b) load their weight before griddepcontrol.wait, beside the kernel in front (the
# FFN-boundary all-gather norm with a 100 % carveout; VP2 with an entry trigger + the carveout). Not bitwise (new math).
SW_WSC2 = _flag("OKC_SW_WSC2", "0")
# Pinned cuBLASLt configs (okc_patch/lt.py) at measured verify (N, K, m) entries outside smallm's range, each >= 5 %
# per call over F.linear in two runs (serving/kernels/lt/RESULTS.md); dispatched after smallm.
SW_LT = _flag("OKC_SW_LT", "1")
SW_ROUTER = _flag("OKC_SW_ROUTER", "1")
# router split-K reduce + routing in one launch (router.route_small, bit-identical)
SW_ROUTE_FUSED = _flag("OKC_SW_ROUTE_FUSED", "1")
# Full-layer verify (q_len 8) at bs >= OKC_WST_MIN_BS on the attention lane's wst kernels (_cuda_src_wst.py: QK norm,
# new KV rows, FP8 attention, merge, output norm and gate in one launch) instead of the stock FA3 chain (stock FA3 below
# OKC_WST_MIN_BS). From OKC_WST_BAL_MIN_BS up the tiles are balanced over the batch (wst_ct14b, cooperative launch).
SW_WST = _flag("OKC_SW_WST", "1")
WST_MIN_BS = int(os.environ.get("OKC_WST_MIN_BS", "1"))
# Dev box (serving/kernels/verify-attn LOG VA2 / VA3): wst_ct14 wins at bs 1-3, ties at 4, bal wins from 6 (by 18 % at
# bs 16 when the batch's lengths spread over 30K-86K; wst_ct14 splits every request by the longest span).
WST_BAL_MIN_BS = int(os.environ.get("OKC_WST_BAL_MIN_BS", "5"))
# Verify (q_len 8) on the attention lane's VFG / VP2 kernels (_cuda_src_vfg.py: QK norm + RoPE, new KV rows, FP8 attention,
# merge, output norm and gate; fp32 split-KV partials): sliding layers at every bs (VP2 up to OKC_VP2_MAX_BS, VFG + merge
# kernel above), and full layers at bs 1 in place of wst. "0": sliding layers on the stock path, full layers on wst.
SW_VFG = _flag("OKC_SW_VFG", "1")
SW_VFG_FULL1 = _flag("OKC_SW_VFG_FULL1", "1")
# Sliding verify batches up to this bs on VP2 (one 16-CTA cluster pair per request, merged in-cluster: no merge kernel)
# instead of VFG + vfgm / vfgm2: dev2 us / call VP2 vs VFG 5.7 / 9.6 (bs 2), 7.5 / 9.2 (4), 8.1 / 10.4 (7), 11.7 / 11.0
# (8; 7 clusters of 16 fit at 1 CTA / SM, the rest share SMs). "1" = the bank's routing (VP2 at bs 1 only).
VP2_MAX_BS = int(os.environ.get("OKC_VP2_MAX_BS", "7"))
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
# Per projection since (serving/kernels/smallm2 F4; S9B gemmag_m256, 8 x H200, us per call vs the replicated route with
# its lt pins, bit-exact slices): pooler m 136 / 160 / 192 / 224 / 256 -5.2 / -8.9 / -5.6 / -4.6 / -4.5 -> to 256;
# unpooler m 136 -0.5, then +0.2 / +1.4 / +1.9 / +1.4 at 160 / 192 / 224 / 256 -> to 136. GEMM_AG_MAX_TOKENS still caps
# the DFlash column splits, and an explicit OKC_GEMM_AG_MAX_TOKENS still sets all three (older arms keep their meaning).
_GEMM_AG_ENV = os.environ.get("OKC_GEMM_AG_MAX_TOKENS")
GEMM_AG_POOLER_MAX_TOKENS = int(os.environ.get("OKC_GEMM_AG_POOLER_MAX_TOKENS", _GEMM_AG_ENV or "256"))
GEMM_AG_UNPOOLER_MAX_TOKENS = int(os.environ.get("OKC_GEMM_AG_UNPOOLER_MAX_TOKENS", _GEMM_AG_ENV or "136"))
# SP participant linears through torch's symmetric-memory fused all-gather+matmul / matmul+reduce-scatter
# (async TP, copy-engine overlap) on forwards of >= OKC_SP_SYMM_MIN_TOKENS tokens, NCCL below: session SB measured
# the 74K first turn -6.0% but cached follow-up TTFT +54 ms (bs 1) / +70 ms (bs 4) with it on every forward.
SP_SYMM = _flag("OKC_SP_SYMM", "1")
SP_SYMM_MIN_TOKENS = int(os.environ.get("OKC_SP_SYMM_MIN_TOKENS", "4096"))
# SP forwards below that (the follow-up extends): the qkv-in all-gather as torch's multimem all-gather into a symmetric
# buffer of OKC_EXT_AG_MAX_BYTES instead of NCCL's ring (okc_patch/ext_coll.py, extend E44; exact, a copy; Lium XC
# step M: eager / graph / back-to-back / skewed layer pattern bitwise, 1.10-1.55x NCCL at 8-62 MiB)
SW_EXT_AG = _flag("OKC_SW_EXT_AG", "1")
EXT_AG_MAX_BYTES = int(os.environ.get("OKC_EXT_AG_MAX_BYTES", str(4096 * 8192 * 2)))
# ... and the extend's o-proj / MoE-out reduce-scatters as multimem.ld_reduce (fp32 in the switch) over a second
# symmetric buffer of that size: differs from NCCL's ring in ~half the elements by an ulp (equal to the fp32 sum), so
# off until a consistency read
SW_EXT_RS = _flag("OKC_SW_EXT_RS", "0")
EXT_RS_FENCE = _flag("OKC_EXT_RS_FENCE", "1")   # its buffer-reuse barrier after the ld_reduce (see ext_coll.py)
# Those two collectives on copy-engine pushes with one barrier per matmul + reduce-scatter and one per
# OKC_SP_CE_AG_GROUP all-gather steps (okc_patch/sp_tp_ce.py) instead of torch's fused ops (16 / 3 barriers per op):
# session SI bench -0.12 ms per 16K layer-chunk (~-30 ms per first turn); the AG side only on 16K chunks.
SP_CE_TP = _flag("OKC_SP_CE_TP", "1")
# The o-proj's slot sum there (torch .sum over the 8 pushed slots, 106 us per 16K layer-chunk) read inside the
# post-attention norm instead, in torch's fp32 order (bit-equal): the summed branch is never written or re-read and
# one kernel goes (serving/kernels/rsfold).
SW_RS_NORM_FOLD = _flag("OKC_SW_RS_NORM_FOLD", "1")
# Sliding layers there: fmha7 + output gate per destination block inside one eager break with the o-proj RS, so each
# GEMM + push starts under the attention (serving/kernels/rsfold R4-R7, bit-equal). Off until an 8-GPU arm prices it.
SW_RS_ATTN_BLOCKS = _flag("OKC_SW_RS_ATTN_BLOCKS", "1")
RS_ATTN_BLOCKS_MIN_ROWS = int(os.environ.get("OKC_RS_ATTN_BLOCKS_MIN_ROWS", "1152"))
RS_ATTN_BLOCKS_GROUP = int(os.environ.get("OKC_RS_ATTN_BLOCKS_GROUP", "2"))
# Full-attention layers too, in groups of this many destinations per fmha launch (0 = off; R4: 2-4 launches).
RS_ATTN_BLOCKS_FULL = int(os.environ.get("OKC_RS_ATTN_BLOCKS_FULL", "4"))
# ... and a padded bucket (the last chunk's real rows < rows x 8): blocks over the real rows, none past them (extend E39).
RS_ATTN_BLOCKS_PADDED = _flag("OKC_RS_ATTN_BLOCKS_PADDED", "1")
# NCCL SP MoE section: the pre-expert RMSNorm (per row) on the shard before the all-gather, not on the gathered chunk.
SP_PRENORM_SHARD = _flag("OKC_SP_PRENORM_SHARD", "1")
# ... issued before the router side-stream join, under the router's tail (extend E23).
SP_PRENORM_EARLY = _flag("OKC_SP_PRENORM_EARLY", "1")
# FIRSTTURN #1 (okc_patch/ft_norm.py, _cuda_src_ftnorm.py; bit-identical): the extend's per-head q / k norm (and
# rope) folded into the FP8 q / KV-cache write: one launch for FusedQKNorm + fused_fp8_qkv_kv_cache.
QK_FOLD = _flag("OKC_QK_FOLD", "1")
# pg calls of >= OKC_PG_LOCAL_ALIGN_MIN_TOKENS tokens align their routes with the prefill bank's local-slot align
# (pg_moe.align_local: one count into the local-slot counters) instead of the cast + map gather + 256-expert
# moe_align_block_size chain (1xH200: 16K-token call 127.7 -> 19.7 us GPU, bit-identical pg rows).
PG_LOCAL_ALIGN = _flag("OKC_PG_LOCAL_ALIGN", "1")
# 2049..24576 routes, static map: pg_route in place of the align + pg_tiles (extend E24).
PG_ROUTE = _flag("OKC_PG_ROUTE", "1")
# with replicas, pg_route runs the verify balancer itself (rep_owner, the same decision) instead of a pg_rep_assign launch
PG_ROUTE_REP_FOLD = _flag("OKC_PG_ROUTE_REP_FOLD", "1")
# with replicas at extend sizes (OKC_MOE_REP_EXTEND), pg_route runs the extend balancer itself (rep_align_scan's decision)
# instead of the rep_align count / scan / scatter + pg_tiles chain (extend E38)
PG_ROUTE_REPX = _flag("OKC_PG_ROUTE_REPX", "1")
PG_LOCAL_ALIGN_MIN_TOKENS = int(os.environ.get("OKC_PG_LOCAL_ALIGN_MIN_TOKENS", "1024"))
# NCCL SP MoE section: the pooled rows, top-k scores and top-k ids gathered in ONE NCCL group (one kernel) instead of
# three all-gathers. Same bytes; the two small gathers' fixed latency goes (cand7 ~1K extend: 17 + 12 us per layer).
SP_AG_GROUP = _flag("OKC_SP_AG_GROUP", "1")
# ... or as one push-plane launch (rows + top-k, gemm_ag.row_gather) up to this many gathered rows (extend E26).
SP_AG_PUSH = _flag("OKC_SP_AG_PUSH", "1")
SP_AG_PUSH_MAX_TOKENS = int(os.environ.get("OKC_SP_AG_PUSH_MAX_TOKENS", "2048"))
# ... or rows first (AG.md W1a; default 0): the rows' NCCL gather starts on its own stream at the shard prenorm's end,
# without waiting for the router; the top-k scores / ids follow as a symmetric-memory multimem gather (not the same NCCL
# comm, which would queue them behind the rows) and pg's routing prep runs under the rows. Gathered calls of at least
# SP_AG_ROWSFIRST_MIN_TOKENS..MAX_TOKENS rows; same kernels, same inputs.
SP_AG_ROWSFIRST = _flag("OKC_SP_AG_ROWSFIRST", "1")
SP_AG_ROWSFIRST_MIN_TOKENS = int(os.environ.get("OKC_SP_AG_ROWSFIRST_MIN_TOKENS", "4096"))
SP_AG_ROWSFIRST_MAX_TOKENS = int(os.environ.get("OKC_SP_AG_ROWSFIRST_MAX_TOKENS", "16384"))

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

# Verify-step glue (serving/kernels/verify-glue): at <= 2048 routes a pg-only layer runs one routing launch (pg_route:
# local ids + block list, in place of the int32 cast / expert-map gather / moe_align / pg_tiles) and folds the
# pre-expert RMSNorm into the token gather (bit-identical to F.rms_norm).
SW_PG_SMALL = _flag("OKC_SW_PG_SMALL", "1")
# Programmatic dependent launch on the OKC kernels inside verify / decode graphs (the next kernel's launch and
# prologue overlap the previous one's tail; every kernel waits on griddepcontrol before reading its inputs).
SW_PDL = _flag("OKC_SW_PDL", "1")
# Post-attention boundary (cluster norm_residual, M <= 32, TP > 1): load the fp32 stream and the weights before
# griddepcontrol.wait; only the all-reduced branch loads after it (the o-proj AR writes nothing else and triggers last).
NORM_PREFETCH = _flag("OKC_NORM_PREFETCH", "1")
# DFlash context projection fc (5 x 8192 -> 8192, replicated 640 MiB bf16, 159 us per verify step) as a column-split
# GEMM + the multicast all-gather (gemm_ag) at <= GEMM_AG_MAX_TOKENS tokens.
SW_DFLASH_FC_AG = _flag("OKC_SW_DFLASH_FC_AG", "1")
# Under SP prefill, the same fc + hidden_norm (row-wise) on each rank's shard of the packed taps before the exit gather:
# gather [N, 8192] instead of [N, 40960], and 1/tp of the GEMM (every rank ran fc over the whole chunk: 14.5 ms + a
# 1.34 GB all-gather per 16K chunk, serving/kernels/ft-glue).
SW_DFLASH_FC_SP = _flag("OKC_SW_DFLASH_FC_SP", "1")
# From this many shard rows (3072-token chunks) the shard GEMM is bit-equal to the whole-chunk one on the judge's
# cuBLAS (dev2, ft-glue E1); below it cuBLAS takes another kernel for the small shard (rel-L2 6e-4, bf16 ulps).
DFLASH_FC_SP_MIN_ROWS = int(os.environ.get("OKC_DFLASH_FC_SP_MIN_ROWS", "384"))
# Below it, from 64 rows: the shard GEMM at a pinned cuBLASLt config with the whole chunk's rows bit for bit (lt.lt_fc).
SW_DFLASH_FC_SP_PIN = _flag("OKC_SW_DFLASH_FC_SP_PIN", "1")
# Boot-time launch of every cuBLASLt variant the eager DFlash context GEMMs can pick (fc + fused KV materialization,
# M = 1..max prefill tokens): a variant's first launch blocks the CPU until the GPU drains (court C21). Outputs unchanged.
SW_WARM_GEMMS = _flag("OKC_SW_WARM_GEMMS", "1")
# The same column split for the DFlash grouped convs' kernel_projection (replicated nn.Linear, 2 per draft layer, ~11 us
# each per verify step); 8-GPU check (session SI vg-micro): slices bit-identical, [3072, 8192] 16.5 -> 10.4 us at m 8.
SW_DFLASH_CONV_AG = _flag("OKC_SW_DFLASH_CONV_AG", "1")
# BF16 expert layers (0 and 61 of the NVFP4 export): local ids by one table lookup and the fused finalize kernel at
# every token count (the wrapper's torch chain below 32 tokens is 11 launches).
SW_BF16_MOE_GLUE = _flag("OKC_SW_BF16_MOE_GLUE", "1")
# Their Triton fused_moe configs (okc_patch/moe_configs, E=32 N=4096 H200), retuned at EP routing (one local route per
# token): T 8 -7.5 %, 16 -11 %, 4096..16384 -12..-24 % per layer, bit-identical outputs (smallt-moe LOG E19).
SW_BF16_MOE_CFG = _flag("OKC_SW_BF16_MOE_CFG", "1")
# The last layer's FFN boundary (post-FFN norm, gated residual) + the model's final norm as one bit-identical kernel
# (okc_patch.glue.tail_boundary) instead of 9 torch launches.
SW_TAIL_NORM = _flag("OKC_SW_TAIL_NORM", "1")
# Target verify argmax over the vocab shard with one program per 4096-column block (softcap_row_argmax_split) instead of
# one per row; same first-maximum result after the gather.
SW_ARGMAX_SPLIT = _flag("OKC_SW_ARGMAX_SPLIT", "1")
# DFlash candidate_topk: one all-gather of packed value bits + ids instead of two.
SW_CAND_AG_MERGE = _flag("OKC_SW_CAND_AG_MERGE", "1")
# DFlash draft glue (okc_patch.draft_glue, bit-identical): candidate pack / merge; one-launch draft-KV commit.
SW_CAND_FUSE = _flag("OKC_SW_CAND_FUSE", "1")
SW_DRAFT_KV_FUSE = _flag("OKC_SW_DRAFT_KV_FUSE", "1")

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
# TP8 graph-mode all-reduce: the lamport 2-shot push (upstream tunes it on B300 only; SM90 had it off) from 128 KB up to
# this size (0 = off, the table above). H200 x8, us per AR heuristic -> 2-shot push, bitwise identical (session S8 vg-ar).
AR_TP8_GRAPH_2PUSH_MAX_BYTES = int(os.environ.get("OKC_AR_TP8_GRAPH_2PUSH_MAX_BYTES", str(4 << 20)))

# HiCache staged write-back: pages per relayout + DtoH call (upstream 64; pool_host/base.py). At page_size 1 that is
# tokens: 64 made a 16K-token prefill chunk 256 Python iterations x 3 host pools on the scheduler thread (1,792 CUDA
# calls, ~15 ms before the next chunk or the first decode step launches; SI arm A trace). 1024: ~1 ms of CPU and the
# copies at ~48 GB/s instead of ~19 (serving/tests/bench_staged_chunk.py), 17.6 MB of staging per rank taken before
# KV sizing. Same bytes to host; "64" restores upstream.
# HC-DRAIN (extend E40 / E42): a relayout kernel queued behind a DtoH copy on the write-through stream blocks a shared
# hardware queue's head, and a push stream aliased to it (8 connections) waits for the whole burst (~8 ms, twice per
# first turn). On: every staged call is one staging chunk (16384 pages: ~267 MiB of staging per rank, taken before KV
# sizing) and its relayout runs on a second stream that waits only on the submission's ready event, so no kernel waits
# on a copy. Same bytes to host. Default on (first-turn bundle x-ftb3: with OKC_SW_BOUNDARY_EARLY the next forward starts
# while the finished chunk's write-back is still queued, and this keeps the relayout off the copy's queue);
# OKC_SW_HICACHE_WB_SPLIT=0 restores the single-stream write-back and 1024 staging pages.
SW_HICACHE_WB_SPLIT = _flag("OKC_SW_HICACHE_WB_SPLIT", "1")
HICACHE_STAGING_PAGES = max(1, int(os.environ.get("OKC_HICACHE_STAGING_PAGES", "16384" if SW_HICACHE_WB_SPLIT else "1024")))

# DFLASH prefill -> verify hand-off without a host sync: the extend branch's extend / prefix lengths go to the device
# from pinned memory (dflash_worker_v2.host_ints_to_device) instead of torch.tensor(..., device=cuda), whose pageable
# copy synchronized the stream after every extend and prefill chunk (SI arm A: 18.2 ms blocked at a bs1 follow-up,
# ~400 ms per first-turn chunk; the GPU then idles behind each later launch: ~7.4 ms post-extend + ~3 ms in verify
# step 1 at bs1, serving/score/HOST-OVERHEAD.md). Same values. On in every confirmed 8xH200 arm since S8 C.
SW_PREFILL_HANDOFF = _flag("OKC_SW_PREFILL_HANDOFF", "1")
# Between-chunk admission (srt/managers/small_prefill_first.py): at low load (running + waiting <=
# OKC_BETWEEN_CHUNK_MAX_REQS) a follow-up that arrives while a first turn is being chunked is prefilled on its own between
# two full-size chunks, then gets the decode steps that show its first token. The scheduler picks what follows a chunk
# only once that chunk's target forward is done. A follow-up then waits <= one chunk (~0.45 s) instead of the whole first
# turn (~2.1 s at c4, which moved the c4 TTFT p90 past the 2x gate in S8). Same work, same chunk size. Default off.
SW_BETWEEN_CHUNK = _flag("OKC_SW_BETWEEN_CHUNK", "0")
BETWEEN_CHUNK_MAX_REQS = int(os.environ.get("OKC_BETWEEN_CHUNK_MAX_REQS", "8"))
# Follow-up admission (srt/managers/small_prefill_first.py fold_order): between-chunk admission at low load and, above
# the load cap, the small newest arrivals (< 4096 uncached tokens, newest 16 matched) fold into every chunk pass whole
# (the chunk shrinks, keeping >= half its budget; in a chunk's last pass they take the leftover before the next first
# turn). Under ride-along a follow-up otherwise waits FCFS behind the remaining first turns (c32: ~40 s), past the
# device SWA cache's ~8 s retention of its conversation tail, and re-prefills its whole prompt (session SI arm R: 30 of
# 32 first follow-ups, c32 p90 TTFT 42 s; serving/kernels/court/LOG.md C12). Folded follow-ups get small-prefill-first's two
# forced decode steps. Implies OKC_SW_BETWEEN_CHUNK. Same tokens per forward. Default off.
SW_FOLLOWUP_ADMIT = _flag("OKC_SW_FOLLOWUP_ADMIT", "1")
# ... and its wait at a chunk boundary on an event the forward records at the attention of layer N - OKC_BOUNDARY_EARLY_LAYERS
# (~13 ms before a 16K chunk's forward ends), not after the forward: the host's pick and launch of the next batch (~9 ms)
# then run under the last layers instead of leaving the GPU idle at every boundary (review-1030/FIRSTTURN.md §2f). Timing
# only; a follow-up arriving in those last ms waits one more chunk (c4 replay: worst TTFT 0.549 -> 0.547 s). Default on
# (x-ftb3; FTB2 with W1a + F1: prefill_bench -31 ms, identity equal); OKC_SW_BOUNDARY_EARLY=0 records after the forward.
SW_BOUNDARY_EARLY = _flag("OKC_SW_BOUNDARY_EARLY", "1")
BOUNDARY_EARLY_LAYERS = int(os.environ.get("OKC_BOUNDARY_EARLY_LAYERS", "2"))

# Bounded merge of follow-up extends (court C30 / C31): at load >= FU_MERGE_MIN_LOAD, with requests decoding and
# only follow-ups waiting, hold them up to FU_MERGE_PASSES scheduler passes so later ones join one extend.
SW_FU_MERGE = _flag("OKC_SW_FU_MERGE", "1")
FU_MERGE_PASSES = int(os.environ.get("OKC_FU_MERGE_PASSES", "6"))
FU_MERGE_MIN_LOAD = int(os.environ.get("OKC_FU_MERGE_MIN_LOAD", "9"))

ANY_ATTN = SW_ATTN or SW_WST

__all__ = ["SW_MOE_PG", "MOE_PG_MIN_TOKENS", "MOE_PG_ONLY", "MOE_PACK_MAX_LAYERS", "MOE_REPLICAS", "ADAPT_REPLICAS", "ADAPT_FIRST_TOKENS", "ADAPT_EVERY_TOKENS", "ADAPT_FIRST_VERIFY", "ADAPT_RING", "ADAPT_MIN_T", "ADAPT_CHECK", "ADAPT_THETA", "ADAPT_MAX_SWAPS", "ADAPT_TRICKLE", "ADAPT_FULL_GROWTH", "MOE_REP_MAX_TOKENS",
           "MOE_REP_PREFILL", "MOE_REP_PASSES", "MOE_REP_SPLIT_MIN_TOKENS", "MOE_REP_EXTEND", "MOE_REP_EXTEND_MIN_TOKENS", "PG_NARROW_MIN_T", "PG_NARROW_MAX_T", "PG64_MAX_T", "PG64_N4", "PG_GN_SPLIT", "SW_MOE_ST", "MOE_ST_MAX_TOKENS", "MOE_ST_PDL", "SW_ST_NORM_FOLD", "SW_AG_PRENORM", "SW_M1", "M1_MIN_T", "M1_MAX_T", "SW_M1RS", "M1RS_MIN_T", "M1RS_MAX_T", "SW_ST_REP_FOLD",
           "PG_LOCAL_ALIGN", "PG_LOCAL_ALIGN_MIN_TOKENS", "PG_ROUTE", "PG_ROUTE_REP_FOLD", "PG_ROUTE_REPX",
           "SW_ATTN", "SW_PROJ", "SW_SMALLM", "SW_LT", "SW_ROUTER", "SW_ROUTE_FUSED",
           "SW_PREFILL_FMHA", "PFMHA_MIN_TOKENS", "PFMHA_SLIDING", "PFMHA_SLIDING_MIN_TOKENS", "PFMHA_RAGGED",
           "PFMHA_RAGGED_MAX_BS", "SW_WST", "WST_MIN_BS", "WST_BAL_MIN_BS", "SW_VFG",
           "SW_VFG_FULL1", "VP2_MAX_BS",
           "SW_SP", "SP_MIN_TOKENS", "SW_GEMM_AG", "GEMM_AG_MAX_TOKENS", "SP_SYMM", "SP_SYMM_MIN_TOKENS", "SW_EXT_AG", "EXT_AG_MAX_BYTES", "SW_EXT_RS", "EXT_RS_FENCE", "SP_CE_TP",
           "SW_RS_NORM_FOLD", "SP_PRENORM_SHARD", "SP_PRENORM_EARLY", "QK_FOLD", "SP_AG_GROUP", "SP_AG_PUSH", "SP_AG_PUSH_MAX_TOKENS", "SP_AG_ROWSFIRST", "SP_AG_ROWSFIRST_MIN_TOKENS", "SP_AG_ROWSFIRST_MAX_TOKENS", "SW_PG_SMALL", "SW_PDL", "NORM_PREFETCH", "SW_DFLASH_FC_AG", "SW_DFLASH_CONV_AG", "SW_BF16_MOE_GLUE", "SW_BF16_MOE_CFG",
           "SW_DFLASH_FC_SP", "DFLASH_FC_SP_MIN_ROWS", "SW_DFLASH_FC_SP_PIN", "SW_WARM_GEMMS",
           "SW_TAIL_NORM", "SW_ARGMAX_SPLIT", "SW_CAND_AG_MERGE", "SW_CAND_FUSE", "SW_DRAFT_KV_FUSE", "HOPPER_NORM_MAX_CLUSTER",
           "SW_AG_NORM_HOPPER", "SW_FA3_FUSED_KV",
           "AR_TP8_GRAPH_MC_MIN_BYTES", "AR_TP8_GRAPH_2PUSH_MAX_BYTES", "HICACHE_STAGING_PAGES", "SW_HICACHE_WB_SPLIT", "SW_PREFILL_HANDOFF", "SW_BETWEEN_CHUNK", "BETWEEN_CHUNK_MAX_REQS", "SW_FOLLOWUP_ADMIT", "SW_BOUNDARY_EARLY", "BOUNDARY_EARLY_LAYERS",
           "SW_FU_MERGE", "FU_MERGE_PASSES", "FU_MERGE_MIN_LOAD", "GEMM_AG_POOLER_MAX_TOKENS", "GEMM_AG_UNPOOLER_MAX_TOKENS",
           "SW_RS_ATTN_BLOCKS", "RS_ATTN_BLOCKS_MIN_ROWS", "RS_ATTN_BLOCKS_GROUP", "RS_ATTN_BLOCKS_FULL",
           "RS_ATTN_BLOCKS_PADDED",
           "ANY_ATTN"]
SW_AG_NORM_WIDE = _flag("OKC_SW_AG_NORM_WIDE", "1")  # M 33..128: fused AG + norm, row kernel at 1024 threads
__all__ += ["SW_AG_NORM_WIDE"]
