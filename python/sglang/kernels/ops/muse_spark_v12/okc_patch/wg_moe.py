"""OKC wgmma W4A16 grouped-GEMM MoE for prefill chunks (muse_prefill_layer v10 graft).

The pair reads packed NVFP4 directly: pack_wg permutes the e2m1 nibbles and e4m3
scale bytes into the wgmma m64k16 A-fragment order (a pure reordering of the NVFP4
encoding, the same legal class as the Marlin repack), then per call a CUDA gather
builds fp16 tile images of x in sorted-slot order, gg1 runs the gate+up GEMM with a
fused silu*mul epilogue, and gg2 runs the down GEMM and scatters by sorted_ids.
Certified ~2x Marlin on the 16K-token court shapes (67%/61% of bf16 peak); Marlin
keeps small token counts (court cut at 1535 tokens).

Scale handling (court-certified): gs13 * 2^14 lands on gg1's fp32 accumulator
pre-silu (exact fp16 binade trick), the intermediate carries 2^-6, and gg2's output
keeps a per-row constant factor (2^20 and the expert's w2 global scale) that the
caller's post-expert RMSNorm cancels exactly -- identical to how Marlin's collapsed
per-expert global scale is cancelled.
"""

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import wg_ext

# Locked by the box descriptor sweep (bench_wg --sweep): B tile stored as
# [kc 0..7][tok 0..127] 16B chunks => trans-b 0, LBO/SBO derived in-kernel (0, 0).
_CFG_BIG = 52     # XN256/NST6: block_m 256 tiles (the certified 16K config)
_CFG_SMALL = 4    # XN128/NST4/2 CTAs-per-SM: block_m 128 tiles
_GSZ = 1
_BLOCK_M_BIG_MIN = 1536  # tokens >= this take 256-token tiles (court WG_T256)


def pack_wg(w_u8: torch.Tensor, sf_u8: torch.Tensor, gated: bool, mj: int = 1):
    """Permute packed NVFP4 into the wgmma A-fragment order (pure nibble/byte
    reordering - the NVFP4 encoding is unchanged).

    w_u8 [E, R, K/2] uint8 (lo nibble = even k), sf_u8 [E, R, K/16] uint8 (e4m3).
    Returns wm [E, NWT, NKT, mj*256, 16] uint8, sfp [E, NWT, NKT, mj*128, 4] uint8.

    Thread tid (0..255): wg=tid>>7, warp lw=(tid>>5)&3, lane l=tid&31, g=l>>2,
    t=l&3; m64 tile j (0..mj-1). Per (kt, q): u32 nibbles n0..n7 =
    (R0,kb),(R1,kb),(R0,kb+8),(R1,kb+8),(R0,kb+1),(R1,kb+1),(R0,kb+9),(R1,kb+9),
    kb = kt*64+q*16+2t, so dq8 emits the m64k16 A fragment a0..a3 directly
    (warp lw owns rows 16lw..16lw+15 of tile j; R0 row 16lw+g, R1 row +8).
    gated: wgmma row r = gate feature f, row r+8 = up of the SAME f:
      f = wt*(64*mj) + (wg*mj+j)*32 + 8*lw + g; R0 = f, R1 = R/2 + f.
    plain: R0 = wt*(128*mj) + (wg*mj+j)*64 + 16*lw + g, R1 = R0 + 8.
    sfp[(wt,kt)][lr][b] = sf(srcrow(lr), kt*4+b), lr = wg*(64*mj)+j*64+16lw+g(+8)."""
    E, R, KB = w_u8.shape
    K = 2 * KB
    dev = w_u8.device
    NWT, NKT = R // (128 * mj), K // 64
    half = R // 2

    tid = torch.arange(256, device=dev)
    wg, lw, l = tid >> 7, (tid >> 5) & 3, tid & 31
    g, t = l >> 2, l & 3
    j_i = torch.arange(mj, device=dev).view(mj, 1)
    if gated:
        f = (wg * mj).view(1, 256) * 32 + j_i * 32 + (8 * lw + g).view(1, 256)   # [mj, 256]
        r0_base, r1_base, f_mult = f, half + f, 64 * mj
    else:
        r0 = (wg * mj).view(1, 256) * 64 + j_i * 64 + (16 * lw + g).view(1, 256)
        r0_base, r1_base, f_mult = r0, r0 + 8, 128 * mj
    kt_i = torch.arange(NKT, device=dev).view(NKT, 1, 1, 1)
    q_i = torch.arange(4, device=dev).view(1, 1, 1, 4)
    kb = kt_i * 64 + q_i * 16 + 2 * t.view(1, 1, 256, 1)            # [NKT, 1, 256, 4]
    kb = kb.expand(NKT, mj, 256, 4)
    r0e = r0_base.view(1, mj, 256, 1).expand(NKT, mj, 256, 4)
    r1e = r1_base.view(1, mj, 256, 1).expand(NKT, mj, 256, 4)
    rows = torch.stack([r0e, r1e, r0e, r1e, r0e, r1e, r0e, r1e], dim=-1)
    ks = torch.stack([kb, kb, kb + 8, kb + 8, kb + 1, kb + 1, kb + 9, kb + 9], dim=-1)
    wt_i = torch.arange(NWT, device=dev).view(NWT, 1, 1, 1, 1, 1)
    rows = rows.view(1, NKT, mj, 256, 4, 8) + wt_i * f_mult          # [NWT, NKT, mj, 256, 4, 8]
    ks = ks.view(1, NKT, mj, 256, 4, 8).expand(NWT, NKT, mj, 256, 4, 8)
    bidx = rows.long() * KB + (ks >> 1)
    shift = ((ks & 1) * 4).to(torch.uint8)
    bidx = bidx.reshape(-1, 8)
    shift = shift.reshape(-1, 8)

    wm = torch.empty((E, NWT, NKT, mj * 256, 16), dtype=torch.uint8, device=dev)
    for e in range(E):
        nib = (w_u8[e].reshape(-1)[bidx] >> shift) & 0xF
        u32 = torch.zeros(bidx.shape[0], dtype=torch.int32, device=dev)
        for s in range(8):
            u32 |= nib[:, s].to(torch.int32) << (4 * s)
        # row order of bidx = (wt, kt, j, tid, q); LE bytes -> [NWT, NKT, mj*256, 16]
        wm[e] = u32.view(torch.uint8).view(NWT, NKT, mj * 256, 16)
    del bidx, shift

    # sf rows: local row lr -> source row
    lr = torch.arange(128 * mj, device=dev)
    wgr, rem = lr // (64 * mj), lr % (64 * mj)
    jr, m = rem >> 6, rem & 63
    lwr, gm = m >> 4, m & 15
    gr, hr = gm & 7, gm >> 3
    wt_r = torch.arange(NWT, device=dev).view(NWT, 1)
    if gated:
        fr = (wgr * mj + jr) * 32 + 8 * lwr + gr                      # [128*mj]
        src_r = torch.where(hr.bool(), half + fr, fr).view(1, -1) + wt_r * (64 * mj)
    else:
        rr = (wgr * mj + jr) * 64 + 16 * lwr + gr + 8 * hr
        src_r = rr.view(1, -1) + wt_r * (128 * mj)
    sfp = sf_u8[:, src_r.reshape(-1), :].view(E, NWT, 128 * mj, NKT, 4).permute(0, 1, 3, 2, 4).contiguous()
    return wm, sfp


def grouped_wg(x, sorted_ids, block_experts, n_padded, *, wm13, sfp13, gs13,
               wm2, sfp2, num_tokens, top_k, block_m):
    """x [T, K] bf16 -> y [T*top_k, K] bf16 expert outputs (rows of non-local routes
    stay unwritten; the caller's finalize masks them by slot_expert_ids)."""
    device = x.device
    size_k = x.shape[1]
    numel = num_tokens * top_k
    num_pid_m = block_experts.shape[0]
    cfg = _CFG_BIG if block_m == 256 else _CFG_SMALL
    ext = wg_ext()
    nkt = size_k // 64
    xg_t = torch.empty((num_pid_m * nkt * block_m * 64,), dtype=torch.float16, device=device)
    ext.gather(x, xg_t, sorted_ids, block_experts, n_padded, num_pid_m, numel, top_k, block_m)
    n_inter = size_k
    inter_t = torch.empty((num_pid_m * (n_inter // 64) * block_m * 64,),
                          dtype=torch.float16, device=device)
    ext.gg1(xg_t, wm13, sfp13, gs13, inter_t, block_experts, n_padded, num_pid_m,
            cfg, 0, 0, _GSZ)
    y = torch.empty((numel, size_k), dtype=torch.bfloat16, device=device)
    ext.gg2(inter_t, wm2, sfp2, y, sorted_ids, block_experts, n_padded, num_pid_m,
            numel, cfg, 0, 0, _GSZ)
    return y
