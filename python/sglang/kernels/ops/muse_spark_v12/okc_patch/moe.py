"""OKC W4A16 MoE chains for the NVFP4 Marlin experts (muse_moe_w4a16 + wg grafts).

Three chains behind their own switches, dispatched by token count T:
  T <= 8  (OKC_SW_MOE_CUDA): CUDA K0f (routing bookkeeping + bf16->fp16 x) -> k1
          (gemv1 + silu*mul) -> k2 (gemv2 * (1+w) + ssq) -> K3c (post-expert-norm
          finalize). Court: the decode1/verify1 MoE leg of the 1.46x/1.27x stack.
  T > 32  (OKC_SW_MOE_TMA): Triton TMA chain _k0z -> _k0b_planes -> _k1_gemm1 ->
          _k2_gemm2 -> _k3_final, CFG "large", 2-stream expert split. Court:
          verify16 1.30x / verify32 1.45x on the MoE load. Capped at
          OKC_MOE_TMA_MAX_TOKENS (256).
  T >= OKC_MOE_PG_MIN_TOKENS (1536, OKC_SW_MOE_PG): pg grouped pair (pg_moe.py,
          BF16 token images and intermediate) on the 16K-token prefill chunks.
  T >= OKC_MOE_WG_MIN_TOKENS (1536, OKC_SW_MOE_WG, default off): the older v10 wg
          pair (wg_moe.py, fp16 staging), kept as an A/B reference only.
9 <= T <= 32 and 256 < T < 1536 stay on stock Marlin (both measured Marlin wins).

All chains consume repacks of the raw NVFP4 expert weights, captured by
maybe_capture_nvfp4_packs right before Marlin repacks them in place
(modelopt_quant.process_weights_after_loading). Memory per captured Muse layer per
rank (E_local 32, I = H = 4096): CUDA packs w13c 512 + w2c 256 MiB (nibble-permuted
copies) + scale refs 96 MiB; TMA packs w13i 512 + sfw13i 128 + sfw2 64 MiB copies +
a kept ref to raw w2 256 MiB; WG packs wm13 512 + wm2 256 + sfp 96 MiB copies; PG packs
wsf13 640 + wsf2 320 MiB (W|sf blocks, bf16 scale pairs).
All three on: ~2.7 GiB/layer; budget with OKC_MOE_PACK_MAX_LAYERS (the N layers
captured first win proportionally; each layer dispatches independently). Flip
unneeded chain switches off to spend the budget on one regime.

The finalize kernels hard-code the post-expert-norm form the released checkpoint
uses: weight-first RMSNorm with effective weight (1 + raw), eps 1e-8; eligibility
checks both and declines otherwise.
"""

import logging
import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from sglang.kernels.ops.muse_spark_v12 import okc_patch as _sw
from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import moe_ext

logger = logging.getLogger(__name__)

_POST_NORM_EPS = 1e-8
# TMA-chain token ceiling: covers verify up to bs 32 x 8 drafts = 256 (the certified
# regime) and keeps 16K-token prefill chunks on Marlin, whose per-expert buffers the
# TMA chain's padded [E_local * ceil16(T), I] fp16 workspaces cannot afford there.
_TMA_MAX_TOKENS = int(os.environ.get("OKC_MOE_TMA_MAX_TOKENS", "256"))


_SIDE_STREAMS = []


def _side_streams(n):
    while len(_SIDE_STREAMS) < n:
        _SIDE_STREAMS.append(torch.cuda.Stream())
    return _SIDE_STREAMS[:n]


@triton.jit
def _k0z(ids_ptr, slots_ptr, tok_ptr, counts_ptr,
         n_routes, MP, TK, local_start, local_count,
         BLOCK: tl.constexpr):
    offs0 = tl.arange(0, BLOCK)
    zero = tl.zeros((BLOCK,), dtype=tl.int32)
    for c0 in range(0, local_count, BLOCK):
        tl.store(counts_ptr + c0 + offs0, zero, mask=c0 + offs0 < local_count)
    tl.debug_barrier()
    for r0 in range(0, n_routes, BLOCK):
        offs = r0 + offs0
        mask = offs < n_routes
        ids = tl.load(ids_ptr + offs, mask=mask, other=-1)
        e = ids - local_start
        is_local = (e >= 0) & (e < local_count) & mask
        e_safe = tl.where(is_local, e, 0)
        pos = tl.atomic_add(counts_ptr + e_safe, 1, mask=is_local)
        slot = e_safe * MP + pos
        tl.store(slots_ptr + offs, tl.where(is_local, slot, -1), mask=mask)
        tl.store(tok_ptr + slot, (offs // TK).to(tl.int32), mask=is_local)


@triton.jit
def _k0b_planes(x_ptr, tok_ptr, counts_ptr, xg_ptr,
                MP, H, SL,
                BT: tl.constexpr, BH: tl.constexpr):
    pid_e = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)
    cnt = tl.load(counts_ptr + pid_e)
    t0 = pid_t * BT
    if t0 < cnt:
        m_offs = t0 + tl.arange(0, BT)
        m_mask = m_offs < cnt
        toks = tl.load(tok_ptr + pid_e * MP + m_offs, mask=m_mask, other=0)
        h_offs = pid_h * BH + tl.arange(0, BH)
        xt = tl.load(x_ptr + toks[:, None] * H + h_offs[None, :],
                     mask=m_mask[:, None], other=0.0).to(tl.float16)
        x3 = tl.permute(tl.reshape(xt, (BT, BH // 8, 8)), (2, 1, 0))
        slots = pid_e * MP + m_offs
        optr = (xg_ptr + tl.arange(0, 8)[:, None, None] * ((H // 8) * SL)
                + (pid_h * (BH // 8) + tl.arange(0, BH // 8))[None, :, None] * SL
                + slots[None, None, :])
        tl.store(optr, x3, mask=m_mask[None, None, :])


@triton.jit
def _dq8(t):
    mm = 0x0E000E00
    ms = 0x80008000
    a0 = ((t << 9) & mm) | ((t << 12) & ms)
    a1 = ((t << 5) & mm) | ((t << 8) & ms)
    a2 = ((t << 1) & mm) | ((t << 4) & ms)
    a3 = ((t >> 3) & mm) | (t & ms)
    c0 = (a0 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    c4 = (a0 >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    c1 = (a1 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    c5 = (a1 >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    c2 = (a2 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    c6 = (a2 >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    c3 = (a3 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    c7 = (a3 >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    return c0, c1, c2, c3, c4, c5, c6, c7


@triton.jit
def _expand2(s8, A: tl.constexpr, W2: tl.constexpr):
    s16 = s8.to(tl.float16)
    return tl.reshape(tl.broadcast_to(s16[:, :, None], (A, W2, 2)), (A, 2 * W2))


@triton.jit
def _k1_gemm1(xdesc, xg_ptr, wdesc, sdesc, gs_ptr, counts_ptr, inter_ptr,
              MP, H, I, SL, local_start, e_off, e_mul,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              XTMA: tl.constexpr, DUAL: tl.constexpr, SFX: tl.constexpr,
              UF: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2) * e_mul + e_off
    cnt = tl.load(counts_ptr + pid_e)
    m0 = pid_m * BM
    if m0 < cnt:
        eg = pid_e + local_start
        row0 = eg * 2 * I + pid_n * BN          # interleaved rows: 2n=gate n, 2n+1=up n
        slot0 = pid_e * MP + m0
        HW = H // 8
        gsv = tl.load(gs_ptr + eg) * 16384.0
        acc = tl.zeros((BN, BM), dtype=tl.float32)
        acc2 = tl.zeros((BN, BM), dtype=tl.float32)
        KW: tl.constexpr = BK // 8
        m_off2 = m0 + tl.arange(0, BM)
        cmask = (m_off2 < MP)[None, :]
        xcols = (slot0 + tl.arange(0, BM))[None, :]
        xrows = tl.arange(0, KW)[:, None]
        for k0 in tl.range(0, H, BK, loop_unroll_factor=UF):
            kw = k0 // 8
            t = wdesc.load([row0, kw]).to(tl.uint32, bitcast=True)   # [BN, BK/8]
            if SFX:
                s = sdesc.load([row0, kw]).to(tl.float8e4nv, bitcast=True).to(tl.float16)
            else:
                s = _expand2(sdesc.load([row0, k0 // 16]), BN, BK // 16)  # [BN, BK/8] f16
            c0, c1, c2, c3, c4, c5, c6, c7 = _dq8(t)
            if XTMA:
                x0 = xdesc.load([0 * HW + kw, slot0])                 # [BK/8, BM]
                x1 = xdesc.load([1 * HW + kw, slot0])
                x2 = xdesc.load([2 * HW + kw, slot0])
                x3 = xdesc.load([3 * HW + kw, slot0])
                x4 = xdesc.load([4 * HW + kw, slot0])
                x5 = xdesc.load([5 * HW + kw, slot0])
                x6 = xdesc.load([6 * HW + kw, slot0])
                x7 = xdesc.load([7 * HW + kw, slot0])
            else:
                xb = xg_ptr + (kw + xrows) * SL + xcols
                x0 = tl.load(xb + (0 * HW) * SL, mask=cmask, other=0.0)
                x1 = tl.load(xb + (1 * HW) * SL, mask=cmask, other=0.0)
                x2 = tl.load(xb + (2 * HW) * SL, mask=cmask, other=0.0)
                x3 = tl.load(xb + (3 * HW) * SL, mask=cmask, other=0.0)
                x4 = tl.load(xb + (4 * HW) * SL, mask=cmask, other=0.0)
                x5 = tl.load(xb + (5 * HW) * SL, mask=cmask, other=0.0)
                x6 = tl.load(xb + (6 * HW) * SL, mask=cmask, other=0.0)
                x7 = tl.load(xb + (7 * HW) * SL, mask=cmask, other=0.0)
            if DUAL:
                acc = tl.dot(c0 * s, x0, acc)
                acc2 = tl.dot(c1 * s, x1, acc2)
                acc = tl.dot(c2 * s, x2, acc)
                acc2 = tl.dot(c3 * s, x3, acc2)
                acc = tl.dot(c4 * s, x4, acc)
                acc2 = tl.dot(c5 * s, x5, acc2)
                acc = tl.dot(c6 * s, x6, acc)
                acc2 = tl.dot(c7 * s, x7, acc2)
            else:
                acc = tl.dot(c0 * s, x0, acc)
                acc = tl.dot(c1 * s, x1, acc)
                acc = tl.dot(c2 * s, x2, acc)
                acc = tl.dot(c3 * s, x3, acc)
                acc = tl.dot(c4 * s, x4, acc)
                acc = tl.dot(c5 * s, x5, acc)
                acc = tl.dot(c6 * s, x6, acc)
                acc = tl.dot(c7 * s, x7, acc)
        if DUAL:
            acc += acc2
        h = acc * gsv
        g, u = tl.split(tl.permute(tl.reshape(h, (BN // 2, 2, BM)), (0, 2, 1)))
        it = (g * tl.sigmoid(g) * u).to(tl.float16)                  # [BN/2, BM]
        m_offs = m0 + tl.arange(0, BM)
        m_mask = m_offs < cnt
        n_offs = pid_n * (BN // 2) + tl.arange(0, BN // 2)
        slots = pid_e * MP + m_offs
        IW = I // 8
        optr = (inter_ptr + (n_offs % 8)[:, None].to(tl.int64) * (IW * SL)
                + (n_offs // 8)[:, None] * SL + slots[None, :])
        tl.store(optr, it, mask=m_mask[None, :])


@triton.jit
def _k2_gemm2(idesc, ig_ptr, wdesc, sdesc, pw_ptr, counts_ptr, z_ptr, ssp_ptr,
              MP, H, I, SL, NT2, local_start, e_off, e_mul,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              XTMA: tl.constexpr, DUAL: tl.constexpr, SFX: tl.constexpr,
              UF: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2) * e_mul + e_off
    cnt = tl.load(counts_ptr + pid_e)
    m0 = pid_m * BM
    if m0 < cnt:
        eg = pid_e + local_start
        row0 = eg * H + pid_n * BN
        slot0 = pid_e * MP + m0
        IW = I // 8
        acc = tl.zeros((BN, BM), dtype=tl.float32)
        acc2 = tl.zeros((BN, BM), dtype=tl.float32)
        KW: tl.constexpr = BK // 8
        m_off2 = m0 + tl.arange(0, BM)
        cmask = (m_off2 < MP)[None, :]
        xcols = (slot0 + tl.arange(0, BM))[None, :]
        xrows = tl.arange(0, KW)[:, None]
        for k0 in tl.range(0, I, BK, loop_unroll_factor=UF):
            kw = k0 // 8
            t = wdesc.load([row0, kw]).to(tl.uint32, bitcast=True)
            if SFX:
                s = sdesc.load([row0, kw]).to(tl.float8e4nv, bitcast=True).to(tl.float16)
            else:
                s = _expand2(sdesc.load([row0, k0 // 16]), BN, BK // 16)
            c0, c1, c2, c3, c4, c5, c6, c7 = _dq8(t)
            if XTMA:
                i0 = idesc.load([0 * IW + kw, slot0])
                i1 = idesc.load([1 * IW + kw, slot0])
                i2 = idesc.load([2 * IW + kw, slot0])
                i3 = idesc.load([3 * IW + kw, slot0])
                i4 = idesc.load([4 * IW + kw, slot0])
                i5 = idesc.load([5 * IW + kw, slot0])
                i6 = idesc.load([6 * IW + kw, slot0])
                i7 = idesc.load([7 * IW + kw, slot0])
            else:
                ib = ig_ptr + (kw + xrows) * SL + xcols
                i0 = tl.load(ib + (0 * IW) * SL, mask=cmask, other=0.0)
                i1 = tl.load(ib + (1 * IW) * SL, mask=cmask, other=0.0)
                i2 = tl.load(ib + (2 * IW) * SL, mask=cmask, other=0.0)
                i3 = tl.load(ib + (3 * IW) * SL, mask=cmask, other=0.0)
                i4 = tl.load(ib + (4 * IW) * SL, mask=cmask, other=0.0)
                i5 = tl.load(ib + (5 * IW) * SL, mask=cmask, other=0.0)
                i6 = tl.load(ib + (6 * IW) * SL, mask=cmask, other=0.0)
                i7 = tl.load(ib + (7 * IW) * SL, mask=cmask, other=0.0)
            if DUAL:
                acc = tl.dot(c0 * s, i0, acc)
                acc2 = tl.dot(c1 * s, i1, acc2)
                acc = tl.dot(c2 * s, i2, acc)
                acc2 = tl.dot(c3 * s, i3, acc2)
                acc = tl.dot(c4 * s, i4, acc)
                acc2 = tl.dot(c5 * s, i5, acc2)
                acc = tl.dot(c6 * s, i6, acc)
                acc2 = tl.dot(c7 * s, i7, acc2)
            else:
                acc = tl.dot(c0 * s, i0, acc)
                acc = tl.dot(c1 * s, i1, acc)
                acc = tl.dot(c2 * s, i2, acc)
                acc = tl.dot(c3 * s, i3, acc)
                acc = tl.dot(c4 * s, i4, acc)
                acc = tl.dot(c5 * s, i5, acc)
                acc = tl.dot(c6 * s, i6, acc)
                acc = tl.dot(c7 * s, i7, acc)
        if DUAL:
            acc += acc2
        n_offs = pid_n * BN + tl.arange(0, BN)
        pwv = tl.load(pw_ptr + n_offs)
        z16 = (acc * pwv[:, None]).to(tl.float16)
        zf = z16.to(tl.float32)
        part = tl.sum(zf * zf, axis=0)
        m_offs = m0 + tl.arange(0, BM)
        m_mask = m_offs < cnt
        slots = pid_e * MP + m_offs
        tl.store(ssp_ptr + slots * NT2 + pid_n, part, mask=m_mask)
        optr = z_ptr + slots[None, :].to(tl.int64) * H + n_offs[:, None]
        tl.store(optr, z16, mask=m_mask[None, :])


@triton.jit
def _k3_final(slots_ptr, tw_ptr, ssp_ptr, z_ptr, out_ptr,
              H, TK, NT2, inv_h, eps,
              BC: tl.constexpr, TKP: tl.constexpr, NT2P: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BC + tl.arange(0, BC)
    j = tl.arange(0, TKP)
    j_mask = j < TK
    slot = tl.load(slots_ptr + pid_t * TK + j, mask=j_mask, other=-1)
    valid = slot >= 0
    slot_s = tl.where(valid, slot, 0)
    tw = tl.load(tw_ptr + pid_t * TK + j, mask=j_mask, other=0.0)
    p_offs = tl.arange(0, NT2P)
    pm = valid[:, None] & (p_offs[None, :] < NT2)
    ssv = tl.sum(tl.load(ssp_ptr + slot_s[:, None] * NT2 + p_offs[None, :],
                         mask=pm, other=0.0), axis=1)
    coef = tl.where(valid, tw * tl.rsqrt(ssv * inv_h + eps), 0.0)
    z8 = tl.load(z_ptr + slot_s[:, None].to(tl.int64) * H + c_offs[None, :],
                 mask=valid[:, None], other=0.0)
    acc = tl.sum(coef[:, None] * z8.to(tl.float32), axis=0)
    tl.store(out_ptr + pid_t.to(tl.int64) * H + c_offs, acc.to(tl.bfloat16))


def _nibperm(b):
    """Per 4-byte group reorder nibbles [k0..k7] -> [k0,k2,k4,k6,k1,k3,k5,k7] so
    dq8's 4 half2 outputs are consecutive k-pairs."""
    v = b.contiguous().view(-1, 4)
    b0, b1, b2, b3 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    out = torch.empty_like(v)
    out[:, 0] = (b0 & 0xF) | ((b1 & 0xF) << 4)
    out[:, 1] = (b2 & 0xF) | ((b3 & 0xF) << 4)
    out[:, 2] = (b0 >> 4) | (b1 & 0xF0)
    out[:, 3] = (b2 >> 4) | (b3 & 0xF0)
    return out.view(b.shape)


def _moe_cuda(x, local_ids, tw, state):
    """v14 CUDA chain at T<=8 on pre-mapped LOCAL ids (local_start=0 inside K0f)."""
    dev = x.device
    M, H = x.shape
    I = H
    TK = local_ids.shape[1]
    EL = state["w13c"].shape[0]
    MP = (M + 15) // 16 * 16
    SL = EL * MP
    NT2 = 128
    counts = torch.empty(EL, dtype=torch.int32, device=dev)
    tok = torch.empty(SL, dtype=torch.int32, device=dev)
    slots = torch.empty(M * TK, dtype=torch.int32, device=dev)
    act = torch.empty(EL, dtype=torch.int32, device=dev)
    actn = torch.empty(1, dtype=torch.int32, device=dev)
    inter = torch.empty((SL, I), dtype=torch.float16, device=dev)
    zbuf = torch.empty((SL, H), dtype=torch.float16, device=dev)
    ssp = torch.empty((SL, NT2), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    xh = torch.empty((M, H), dtype=torch.float16, device=dev)
    ext = moe_ext()
    ext.k0f(local_ids, slots, tok, counts, act, actn, x, xh, M * TK, MP, TK, 0, EL,
            M * H // 8)
    ext.k1(state["w13c"], state["sf13c"], state["w13_gs32"], counts, act, actn,
           tok, xh, inter, MP, 0, M)
    ext.k2(state["w2c"], state["sf2c"], state["pw32"], counts, act, actn,
           inter, zbuf, ssp, MP, 0, M)
    ext.k3c(slots, tw, ssp, zbuf, out, M, TK, 1.0 / H)
    return out


_MOE_LARGE = dict(BM1=32, BN1=128, BK1=128, nw1=4, st1=3,
                  BM2=32, BN2=128, BK2=128, nw2=4, st2=3,
                  BT=64, BH=256, BC=1024, nw3=4, XTMA=1, DUAL=0, SFX=1, SPLIT=2, UF=1)


def _moe_tma(x, local_ids, tw, state):
    """v14 Triton TMA chain at T>32 (CFG "large"), local ids with local_start=0."""
    dev = x.device
    M, H = x.shape
    I = H
    TK = local_ids.shape[1]
    EL = state["w13i"].shape[0]
    MP = (M + 15) // 16 * 16
    SL = EL * MP
    c = _MOE_LARGE
    BM1, BN1, BK1, nw1, st1 = c["BM1"], c["BN1"], c["BK1"], c["nw1"], c["st1"]
    BM2, BN2, BK2, nw2, st2 = c["BM2"], c["BN2"], c["BK2"], c["nw2"], c["st2"]
    NT2 = triton.cdiv(H, BN2)

    counts = torch.empty(EL, dtype=torch.int32, device=dev)
    tok = torch.empty(SL, dtype=torch.int32, device=dev)
    slots = torch.empty(M * TK, dtype=torch.int32, device=dev)
    xg = torch.empty((H, SL), dtype=torch.float16, device=dev)
    inter = torch.empty((I, SL), dtype=torch.float16, device=dev)
    zbuf = torch.empty((SL, H), dtype=torch.float16, device=dev)
    ssp = torch.empty((SL, NT2), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)

    SFX = c["SFX"]
    w13f = state["w13i"].view(-1, H // 8)
    sf13f = state["sfw13i"].view(-1, H // 8)
    w2f = state["w2_32"].view(-1, I // 8)
    sf2f = state["sfw2"].view(-1, I // 8)
    wdesc1 = TensorDescriptor(w13f, shape=list(w13f.shape), strides=[w13f.stride(0), 1],
                              block_shape=[BN1, BK1 // 8])
    sdesc1 = TensorDescriptor(sf13f, shape=list(sf13f.shape), strides=[sf13f.stride(0), 1],
                              block_shape=[BN1, BK1 // (8 if SFX else 16)])
    xdesc = TensorDescriptor(xg, shape=[H, SL], strides=[SL, 1], block_shape=[BK1 // 8, BM1])
    wdesc2 = TensorDescriptor(w2f, shape=list(w2f.shape), strides=[w2f.stride(0), 1],
                              block_shape=[BN2, BK2 // 8])
    sdesc2 = TensorDescriptor(sf2f, shape=list(sf2f.shape), strides=[sf2f.stride(0), 1],
                              block_shape=[BN2, BK2 // (8 if SFX else 16)])
    idesc = TensorDescriptor(inter, shape=[I, SL], strides=[SL, 1], block_shape=[BK2 // 8, BM2])

    n_routes = M * TK
    _k0z[(1,)](local_ids, slots, tok, counts, n_routes, MP, TK, 0, EL,
               BLOCK=1024, num_warps=8)
    _k0b_planes[(EL, triton.cdiv(MP, c["BT"]), triton.cdiv(H, c["BH"]))](
        x, tok, counts, xg, MP, H, SL, BT=c["BT"], BH=c["BH"], num_warps=4)

    def run_k1(e_off, e_mul, e_cnt):
        _k1_gemm1[(triton.cdiv(2 * I, BN1), triton.cdiv(MP, BM1), e_cnt)](
            xdesc, xg, wdesc1, sdesc1, state["w13_gs32"], counts, inter,
            MP, H, I, SL, 0, e_off, e_mul, BM=BM1, BN=BN1, BK=BK1,
            XTMA=c["XTMA"], DUAL=c["DUAL"], SFX=SFX, UF=c["UF"],
            num_warps=nw1, num_stages=st1)

    def run_k2(e_off, e_mul, e_cnt):
        _k2_gemm2[(NT2, triton.cdiv(MP, BM2), e_cnt)](
            idesc, inter, wdesc2, sdesc2, state["pw32"], counts, zbuf, ssp,
            MP, H, I, SL, NT2, 0, e_off, e_mul, BM=BM2, BN=BN2, BK=BK2,
            XTMA=c["XTMA"], DUAL=c["DUAL"], SFX=SFX, UF=c["UF"],
            num_warps=nw2, num_stages=st2)

    NS = c["SPLIT"]
    if NS >= 2 and EL >= NS:
        main = torch.cuda.current_stream()
        sides = _side_streams(NS - 1)
        ev_pre = torch.cuda.Event()
        ev_pre.record(main)
        for sd in sides:
            sd.wait_event(ev_pre)
        for q in range(NS):
            cnt_q = (EL - q + NS - 1) // NS
            if q == 0:
                run_k1(0, NS, cnt_q)
            else:
                with torch.cuda.stream(sides[q - 1]):
                    run_k1(q, NS, cnt_q)
        for q in range(NS):
            cnt_q = (EL - q + NS - 1) // NS
            if q == 0:
                run_k2(0, NS, cnt_q)
            else:
                with torch.cuda.stream(sides[q - 1]):
                    run_k2(q, NS, cnt_q)
        for sd in sides:
            ev = torch.cuda.Event()
            ev.record(sd)
            main.wait_event(ev)
    else:
        run_k1(0, 1, EL)
        run_k2(0, 1, EL)
    _k3_final[(M, triton.cdiv(H, c["BC"]))](
        slots, tw, ssp, zbuf, out, H, TK, NT2, 1.0 / H, 1e-8,
        BC=c["BC"], TKP=triton.next_power_of_2(TK), NT2P=triton.next_power_of_2(NT2),
        num_warps=c["nw3"])
    return out


# --------------------------------------------------------------- serving glue

_captured_layers = 0


def _build_packs(layer) -> dict | None:
    """v14 weight layouts from the raw NVFP4 tensors (pre-Marlin-repack)."""
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    sf13 = layer.w13_weight_scale.data
    sf2 = layer.w2_weight_scale.data
    gs13 = layer.w13_weight_scale_2.data
    if w13.dtype != torch.uint8 or w2.dtype != torch.uint8 or not w13.is_cuda:
        return None
    if w13.dim() != 3 or w2.dim() != 3 or sf13.dim() != 3 or sf2.dim() != 3:
        return None
    E, I2, HB = w13.shape
    H = w2.shape[1]
    # The chains assume square experts (I == H); Muse Spark is 4096/4096.
    if HB * 2 != H or I2 != 2 * H or w2.shape[0] != E or w2.shape[2] * 2 != H:
        return None
    if gs13.dim() != 1 or gs13.numel() != E:
        return None
    if sf13.shape[:2] != (E, I2) or sf2.shape[:2] != (E, H):
        return None
    state = {"H": H, "w13_gs32": gs13.float().contiguous()}
    w13_32 = w13.contiguous().view(torch.int32)
    HW = w13_32.shape[-1]
    sf13c = sf13.contiguous()
    HS = sf13c.shape[-1]
    w13i = (w13_32.view(E, 2, I2 // 2, HW).permute(0, 2, 1, 3)
            .reshape(E, I2, HW).contiguous())
    sf13i = (sf13c.view(E, 2, I2 // 2, HS).permute(0, 2, 1, 3)
             .reshape(E, I2, HS).contiguous())
    w2_sf = sf2.contiguous()
    if _sw.SW_MOE_TMA and H % 1024 == 0:
        state["w13i"] = w13i
        state["w2_32"] = w2.contiguous().view(torch.int32)
        state["sfw13i"] = sf13i.view(torch.uint8).repeat_interleave(2, dim=-1).contiguous()
        state["sfw2"] = w2_sf.view(torch.uint8).repeat_interleave(2, dim=-1).contiguous()
    if _sw.SW_MOE_CUDA and HB == 2048 and w2.shape[-1] == 2048:
        state["w13c"] = _nibperm(w13i.view(torch.uint8))
        state["sf13c"] = sf13i.view(torch.uint8).contiguous()
        state["w2c"] = _nibperm(w2.contiguous().view(torch.int32).view(torch.uint8))
        state["sf2c"] = w2_sf.view(torch.uint8).contiguous()
    if _sw.SW_MOE_WG and HB == 2048 and w2.shape[-1] == 2048:
        from sglang.kernels.ops.muse_spark_v12.okc_patch import wg_moe

        # pack_wg wants the raw stacked [gate; up] rows and raw e4m3 scale bytes.
        state["wg_wm13"], state["wg_sfp13"] = wg_moe.pack_wg(
            w13.contiguous(), sf13c.view(torch.uint8), gated=True
        )
        state["wg_wm2"], state["wg_sfp2"] = wg_moe.pack_wg(
            w2.contiguous(), w2_sf.view(torch.uint8), gated=False
        )
    if _sw.SW_MOE_PG and HB == 2048 and w2.shape[-1] == 2048:
        from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe

        # pack_wsf wants the raw stacked [gate; up] rows and raw e4m3 scale bytes.
        state["pg_wsf13"] = pg_moe.pack_wsf(w13.contiguous(), sf13c.view(torch.uint8), gated=True)
        state["pg_wsf2"] = pg_moe.pack_wsf(w2.contiguous(), w2_sf.view(torch.uint8), gated=False)
    if not any(k in state for k in ("w13i", "w13c", "wg_wm13", "pg_wsf13")):
        return None
    return state


def maybe_capture_nvfp4_packs(layer) -> None:
    """Snapshot v14 packs before prepare_moe_nvfp4_layer_for_marlin destroys the raw
    layout. No-op unless an OKC MoE switch is on and the layer budget remains."""
    global _captured_layers
    if not _sw.ANY_MOE or _captured_layers >= _sw.MOE_PACK_MAX_LAYERS:
        return
    from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts

    if not isinstance(layer, MuseSpark1xFp4Experts):
        return
    state = _build_packs(layer)
    if state is None:
        logger.warning(
            "OKC MoE pack capture declined (shape/dtype mismatch); layer stays on Marlin."
        )
        return
    layer._okc_moe_state = state
    _captured_layers += 1
    logger.info(
        "OKC MoE packs captured for layer %d/%s (chains: %s).",
        _captured_layers,
        "all" if _sw.MOE_PACK_MAX_LAYERS >= (1 << 30) else _sw.MOE_PACK_MAX_LAYERS,
        "+".join(
            c
            for c, k in (("cuda", "w13c"), ("tma", "w13i"), ("wg", "wg_wm13"), ("pg", "pg_wsf13"))
            if k in state
        ),
    )


def _ensure_pw32(layer, state, post_expert_norm) -> bool:
    """Cache the fp32 effective post-expert-norm weight; decline on a norm form the
    finalize kernels do not implement (they hard-code weight-first, eps 1e-8)."""
    if layer._post_expert_norm is not post_expert_norm:
        layer.set_post_expert_norm(post_expert_norm)
    weight = layer._post_expert_norm_weight
    if weight is None or not layer._post_expert_norm_weight_first:
        return False
    if layer._post_expert_norm_eps != _POST_NORM_EPS:
        return False
    key = weight.data_ptr()
    if state.get("pw32_key") != key:
        state["pw32"] = weight.float().contiguous()
        state["pw32_key"] = key
    return True


def _moe_wg(layer, x, local_ids, tw, state):
    """wg grouped chain at prefill token counts: Marlin's own alignment convention
    (local ids with -1 non-local routes; worst-case grid guarded in-kernel), then
    gather -> gg1 -> gg2 -> the baseline finalize, exactly as _forward_marlin ends."""
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    from sglang.kernels.ops.muse_spark_v12.okc_patch import wg_moe
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    from sglang.kernels.ops.muse_spark_v12.okc_patch.attention import log_once

    num_tokens, size_k = x.shape
    top_k = local_ids.shape[1]
    log_once("moe_wg", f"(first T={num_tokens})")
    block_m = 256 if num_tokens >= wg_moe._BLOCK_M_BIG_MIN else 128
    align_num_experts = (
        layer.global_num_experts
        if layer._expert_map is not None
        else layer.num_local_experts
    )
    sorted_ids, block_experts, n_padded = moe_align_block_size(
        local_ids, block_m, align_num_experts
    )
    y = wg_moe.grouped_wg(
        x, sorted_ids, block_experts, n_padded,
        wm13=state["wg_wm13"], sfp13=state["wg_sfp13"], gs13=state["w13_gs32"],
        wm2=state["wg_wm2"], sfp2=state["wg_sfp2"],
        num_tokens=num_tokens, top_k=top_k, block_m=block_m,
    )
    return fused_post_expert_norm_weighted_sum(
        y.view(num_tokens, top_k, size_k),
        tw,
        norm_weight=layer._post_expert_norm_weight,
        eps=layer._post_expert_norm_eps,
        weight_first=layer._post_expert_norm_weight_first,
        slot_expert_ids=local_ids,
    )


def _moe_pg(layer, x, local_ids, tw, state):
    """pg grouped chain at prefill token counts: the same alignment (256-token blocks, -1 for
    non-local routes) and finalize as _moe_wg, with BF16 token images and intermediate."""
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    from sglang.kernels.ops.muse_spark_v12.okc_patch.attention import log_once

    num_tokens, size_k = x.shape
    top_k = local_ids.shape[1]
    log_once("moe_pg", f"(first T={num_tokens})")
    align_num_experts = (
        layer.global_num_experts
        if layer._expert_map is not None
        else layer.num_local_experts
    )
    sorted_ids, block_experts, n_padded = moe_align_block_size(
        local_ids, pg_moe._BLOCK_M, align_num_experts
    )
    y = pg_moe.grouped_pg(
        x, sorted_ids, block_experts, n_padded,
        wsf13=state["pg_wsf13"], gs13=state["w13_gs32"], wsf2=state["pg_wsf2"],
    )
    return fused_post_expert_norm_weighted_sum(
        y.view(num_tokens, top_k, size_k),
        tw,
        norm_weight=layer._post_expert_norm_weight,
        eps=layer._post_expert_norm_eps,
        weight_first=layer._post_expert_norm_weight_first,
        slot_expert_ids=local_ids,
    )


def maybe_forward(
    layer,
    hidden_states,
    topk_scores,
    topk_ids,
    post_expert_norm,
    pre_expert_norm=None,
):
    """Run an OKC chain for this batch, or None to fall back to stock Marlin.

    Same contract as _forward_marlin: applies pre_expert_norm itself, returns the
    EP-partial [T, H] combined output (the caller's all-reduce sums ranks)."""
    state = layer._okc_moe_state
    if state is None or torch.compiler.is_compiling():
        return None
    tokens = hidden_states.shape[0]
    if tokens == 0 or hidden_states.shape[1] != state["H"]:
        return None
    topk = topk_ids.shape[1]
    use_cuda = _sw.SW_MOE_CUDA and tokens <= 8 and topk <= 8 and "w13c" in state
    use_tma = (
        _sw.SW_MOE_TMA and 32 < tokens <= _TMA_MAX_TOKENS and "w13i" in state
    )
    # The pg and wg kernels hard-code topk 8 and 32 local experts (the TP8/EP8 shape).
    use_pg = (
        _sw.SW_MOE_PG
        and tokens >= _sw.MOE_PG_MIN_TOKENS
        and topk == 8
        and layer.num_local_experts == 32
        and "pg_wsf13" in state
    )
    use_wg = (
        _sw.SW_MOE_WG
        and tokens >= _sw.MOE_WG_MIN_TOKENS
        and topk == 8
        and layer.num_local_experts == 32
        and "wg_wm13" in state
    )
    if not (use_cuda or use_tma or use_pg or use_wg):
        return None
    if post_expert_norm is None or not _ensure_pw32(layer, state, post_expert_norm):
        return None
    if pre_expert_norm is not None:
        hidden_states = pre_expert_norm(hidden_states)
    if hidden_states.dtype is not torch.bfloat16:
        return None
    x = hidden_states.contiguous()
    local_ids = topk_ids.to(torch.int32)
    if layer._expert_map is not None:
        local_ids = layer._expert_map[local_ids]
    local_ids = local_ids.contiguous()
    weights = topk_scores.to(torch.float32).contiguous()
    if use_pg:
        return _moe_pg(layer, x, local_ids, weights, state)
    if use_wg:
        return _moe_wg(layer, x, local_ids, weights, state)
    if use_cuda:
        return _moe_cuda(x, local_ids, weights, state)
    return _moe_tma(x, local_ids, weights, state)
