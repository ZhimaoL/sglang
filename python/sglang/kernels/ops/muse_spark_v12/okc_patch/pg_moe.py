"""OKC pg grouped-GEMM MoE for prefill chunks (prefill-gemm graft from the prefill bank plt_c3.py).

BF16 end to end: the weights stay packed NVFP4 (pack_wsf reorders the e2m1 codes into the wgmma m64k16
A-fragment order and stores each e4m3 block scale exactly as bf16, a pure re-encoding of the same values,
the same legal class as the Marlin repack); per call pg_tiles lists the live 256-token blocks of the local
experts, pg_gather copies the bf16 token rows into swizzled bf16 images, gg1 runs gate+up with the
silu*mul epilogue into a bf16 intermediate, and gg2 runs the down GEMM and scatters bf16 rows by sorted_ids.
This replaces the v10 wg pair (wg_moe.py), which stages the token rows and the intermediate in fp16.

Scale handling (as in the bank): gg1 applies the expert's w13 global scale (and the 2^8 decode
compensation) before silu; the intermediate carries 2^-6 and gg2's rows keep a per-route positive constant
(2^-6 / the expert's w2 global scale) that the caller's post-expert RMSNorm cancels, as for the wg pair.
Memory per captured layer per rank (32 local experts, I = H = 4096): 640 + 320 MiB of W|sf blocks.
"""

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import pg_ext

_BLOCK_M = 256
_CFG = 350        # the bank's shipped pg configuration (widths 64 / 128 / 256, 5 stages)
_WMASK = 3        # width classes emitted by pg_tiles: 64 and 128 (256 always)
# Persistent grid: one CTA per SM (0) or capped at items-per-CTA x this (see _grid).
import os as _os
_CTA_ITEMS = int(_os.environ.get("OKC_PG_CTA_ITEMS", "0"))   # 0 = one CTA per SM


# v7 placement of the 8 codes of a weight word (see dq8r): magnitude bits at MAG[s] .. MAG[s] + 2, sign at SGN[s]
_MAG = (6, 0, 3, 11, 22, 16, 19, 27)    # bf16 placement (see dq8r)
_SGN = (15, 9, 10, 14, 31, 25, 26, 30)


def pack_wsf(w_u8: torch.Tensor, sf_u8: torch.Tensor, gated: bool):
    """e2m1 codes permuted into the wgmma m64k16 A-fragment order of the 256 consumer threads (gated: row r = gate
    feature f, row r+8 = up of the same f; the v3 / wg12 order) with each 32-bit word's 8 codes at the v7 bit
    positions (_MAG / _SGN: 4 bits per weight, same values), merged with the e4m3 scale bytes into one contiguous
    4608 B block per (expert, w-tile, k-tile); the scales are stored as (row r, row r + 8) pairs per k16 step in the
    order the consumer threads read them (8 B per thread pair of rows).
    w_u8 [E, R, K/2] uint8, sf_u8 [E, R, K/16] uint8 -> [E, R/128, K/64, 4608] uint8."""
    E, R, KB = w_u8.shape
    K = 2 * KB
    dev = w_u8.device
    NWT, NKT = R // 128, K // 64
    half = R // 2
    tid = torch.arange(256, device=dev)
    wg_, lw, l = tid >> 7, (tid >> 5) & 3, tid & 31
    g, t = l >> 2, l & 3
    if gated:
        f = wg_ * 32 + 8 * lw + g
        r0_base, r1_base, f_mult = f, half + f, 64
    else:
        r0 = wg_ * 64 + 16 * lw + g
        r0_base, r1_base, f_mult = r0, r0 + 8, 128
    kt_i = torch.arange(NKT, device=dev).view(NKT, 1, 1)
    q_i = torch.arange(4, device=dev).view(1, 1, 4)
    kb = kt_i * 64 + q_i * 16 + 2 * t.view(1, 256, 1)
    r0e = r0_base.view(1, 256, 1).expand(NKT, 256, 4)
    r1e = r1_base.view(1, 256, 1).expand(NKT, 256, 4)
    rows = torch.stack([r0e, r1e, r0e, r1e, r0e, r1e, r0e, r1e], dim=-1)
    ks = torch.stack([kb, kb, kb + 8, kb + 8, kb + 1, kb + 1, kb + 9, kb + 9], dim=-1)
    wt_i = torch.arange(NWT, device=dev).view(NWT, 1, 1, 1, 1)
    rows = rows.view(1, NKT, 256, 4, 8) + wt_i * f_mult
    ks = ks.view(1, NKT, 256, 4, 8).expand(NWT, NKT, 256, 4, 8)
    bidx = (rows.long() * KB + (ks >> 1)).reshape(-1, 8)
    shift = ((ks & 1) * 4).to(torch.uint8).reshape(-1, 8)
    wm = torch.empty((E, NWT, NKT, 256, 16), dtype=torch.uint8, device=dev)
    for e in range(E):
        nib = (w_u8[e].reshape(-1)[bidx] >> shift) & 0xF
        u32 = torch.zeros(bidx.shape[0], dtype=torch.int32, device=dev)
        for s in range(8):
            c = nib[:, s].to(torch.int32)
            u32 |= ((c & 7) << _MAG[s]) | ((c >> 3) << _SGN[s])
        wm[e] = u32.view(torch.uint8).view(NWT, NKT, 256, 16)
    del bidx, shift
    lr = torch.arange(128, device=dev)
    wgr, m = lr // 64, lr % 64
    lwr, gm = m >> 4, m & 15
    gr, hr = gm & 7, gm >> 3
    wt_r = torch.arange(NWT, device=dev).view(NWT, 1)
    if gated:
        fr = wgr * 32 + 8 * lwr + gr
        src_r = torch.where(hr.bool(), half + fr, fr).view(1, -1) + wt_r * 64
    else:
        rr = wgr * 64 + 16 * lwr + gr + 8 * hr
        src_r = rr.view(1, -1) + wt_r * 128
    sfp = (sf_u8[:, src_r.reshape(-1), :].view(E, NWT, 128, NKT, 4)
           .permute(0, 1, 3, 2, 4).contiguous())                     # [E, NWT, NKT, local row, q]
    # local row = 64 wg + 16 lw + 8 h + g (h = 0: the thread's row r0, 1: r0 + 8) -> pair (32 wg + 8 lw + g), value
    # 2q + h, stored as bf16 * 2^118 (exact: e4m3 needs 4 significant bits; 448 * 2^118 < bf16 max)
    sfp = (sfp.view(E, NWT, NKT, 2, 4, 2, 8, 4).permute(0, 1, 2, 3, 4, 6, 7, 5).contiguous())
    sbf = (sfp.view(torch.float8_e4m3fn).float() * 2.0 ** 118).to(torch.bfloat16)
    return torch.cat([wm.view(E, NWT, NKT, 4096), sbf.view(torch.uint8).reshape(E, NWT, NKT, 1024)],
                     dim=-1).contiguous()


def align_local(ids, expert_map=None, n_local=32, block_m=_BLOCK_M):
    """Large-T front end (the prefill bank's local-slot align): ids [T, 8] global expert ids (int32 / int64) mapped by
    expert_map ([E] int32, global -> local slot, -1 remote), or with expert_map None already local slot ids (-1 = not
    this rank's, e.g. a balancer's slots over 32 + R replicas) -> (local_ids [T, 8] int32, sorted_ids, block_experts,
    n_padded): moe_align_block_size(local_ids, block_m, n_local)'s layout without its 256-expert sort, sized for n_local
    slots (<= 64); block_m = the pg build's token block (256, or 128 for a narrow build). Routes inside a slot's range
    come in no fixed order; pg's per-route rows do not depend on it."""
    numel = ids.numel()
    device = ids.device
    nblocks = (numel + n_local * (block_m - 1) + block_m - 1) // block_m
    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=device)
    sorted_ids = torch.empty((nblocks * block_m,), dtype=torch.int32, device=device)
    block_experts = torch.empty((nblocks,), dtype=torch.int32, device=device)
    n_padded = torch.empty((1,), dtype=torch.int32, device=device)
    cnt = torch.zeros((n_local,), dtype=torch.int32, device=device)
    off = torch.empty((n_local,), dtype=torch.int32, device=device)
    cur = torch.empty((n_local,), dtype=torch.int32, device=device)
    emap = expert_map if expert_map is not None else local_ids.new_empty((0,))
    n_global = expert_map.shape[0] if expert_map is not None else n_local
    pg_ext().align(ids.contiguous(), emap, local_ids, cnt, off, cur, sorted_ids, block_experts, n_padded,
                   numel, block_m, n_local, n_global)
    return local_ids, sorted_ids, block_experts, n_padded


def grouped_pg(x, sorted_ids, block_experts, n_padded, *, wsf13, gs13, wsf2):
    """x [T, 4096] bf16 -> y [T*8, 4096] bf16 per-route expert outputs (rows of non-local routes stay
    unwritten; the caller's finalize masks them by slot_expert_ids). sorted_ids / block_experts / n_padded
    come from moe_align_block_size(local_ids, 256, ...) with -1 for non-local routes."""
    num_tokens = x.shape[0]
    numel = num_tokens * 8
    num_pid_m = block_experts.shape[0]
    ext = pg_ext()
    live = torch.empty((num_pid_m + 1,), dtype=torch.int32, device=x.device)
    ext.tiles(block_experts, n_padded, sorted_ids, live, num_pid_m, _BLOCK_M, numel, _WMASK)
    # Persistent grid: 0 = one CTA per SM (the bank's default). Capping the CTAs trades cross-item
    # pipelining for less per-CTA setup; the bound must come from host-side sizes only (a device read
    # here would be a sync inside the forward and break graph capture).
    grid = 0
    if _CTA_ITEMS:
        grid = min(4096, max(1, -(-(num_pid_m * (2 * 4096 // 128)) // _CTA_ITEMS)))
    xg_t = torch.empty((num_pid_m * 64 * 256 * 64,), dtype=torch.bfloat16, device=x.device)
    ext.gather(x, xg_t, sorted_ids, live, numel, 0)
    inter_t = torch.empty((num_pid_m * 64 * 256 * 64,), dtype=torch.bfloat16, device=x.device)
    ext.gg1(xg_t, wsf13, gs13, inter_t, block_experts, live, numel, _CFG, grid, 0)
    y = torch.empty((numel, 4096), dtype=torch.bfloat16, device=x.device)
    ext.gg2(inter_t, wsf2, y, sorted_ids, block_experts, live, numel, _CFG, grid, 0)
    return y


# Routes per call up to which grouped_pg_small runs (one CTA routes them all; verify bs 32 = 256 tokens = 2048).
SMALL_MAX_ROUTES = 2048


def grouped_pg_small(x, topk_ids, expert_map, *, norm_weight, norm_eps, wsf13, gs13, wsf2, pdl):
    """Verify / decode front end of grouped_pg: x [T, 4096] bf16 BEFORE the pre-expert RMSNorm, topk_ids
    [T, 8] int32 global ids, expert_map [E] int32 global -> local (-1 remote) or None. One pg_route launch
    replaces the int32 cast, the expert-map gather, moe_align_block_size and pg_tiles (blocks only for local
    experts: 32 + ceil(T * 8 / 256) at most, instead of moe_align's ~257 for 256 global experts), and
    pg_gather_norm applies the norm (F.rms_norm's exact fp32 sequence) while it builds the token images.
    -> (y [T*8, 4096] bf16 as grouped_pg, local_ids [T, 8] int32: the finalize's slot expert ids)."""
    num_tokens = x.shape[0]
    numel = num_tokens * 8
    nb = 32 + (numel + _BLOCK_M - 1) // _BLOCK_M
    dev = x.device
    ext = pg_ext()
    p = 1 if pdl else 0
    local_ids = torch.empty((num_tokens, 8), dtype=torch.int32, device=dev)
    sorted_ids = torch.empty((nb * _BLOCK_M,), dtype=torch.int32, device=dev)
    block_experts = torch.empty((nb,), dtype=torch.int32, device=dev)
    live = torch.empty((nb + 1,), dtype=torch.int32, device=dev)
    emap = expert_map if expert_map is not None else topk_ids.new_empty((0,))
    n_global = expert_map.shape[0] if expert_map is not None else 32
    ext.route(topk_ids, emap, local_ids, sorted_ids, block_experts, live, numel, n_global, p)
    xg_t = torch.empty((nb * 64 * 256 * 64,), dtype=torch.bfloat16, device=dev)
    ext.gather_norm(x, norm_weight, norm_eps, xg_t, sorted_ids, live, numel, p)
    inter_t = torch.empty((nb * 64 * 256 * 64,), dtype=torch.bfloat16, device=dev)
    ext.gg1(xg_t, wsf13, gs13, inter_t, block_experts, live, numel, _CFG, 0, p)
    y = torch.empty((numel, 4096), dtype=torch.bfloat16, device=dev)
    ext.gg2(inter_t, wsf2, y, sorted_ids, block_experts, live, numel, _CFG, 0, p)
    return y, local_ids
