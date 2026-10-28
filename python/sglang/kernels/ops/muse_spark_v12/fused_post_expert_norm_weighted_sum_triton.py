from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import cuda as tlc

from sglang.kernels.ops.muse_spark_v12.okc_patch import SW_PDL

_MIN_TOKENS_FOR_FUSED = 32


@triton.jit
def _ld_sys(p):   # peer-visible 32-bit load (lamport polling)
    return tl.inline_asm_elementwise("ld.relaxed.sys.global.b32 $0, [$1];", "=r,l", [p], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def _st_sys(p, v):   # peer-visible 32-bit store
    return tl.inline_asm_elementwise("st.relaxed.sys.global.b32 [$1], $2; mov.u32 $0, 0;", "=r,l,r", [p, v],
                                     dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _poll(p):
    y = _ld_sys(p)
    while tl.sum((y == 0).to(tl.int32)) > 0:
        y = _ld_sys(p)
    tl.store(p, tl.zeros_like(y))
    return y


@triton.jit
def _post_expert_norm_weighted_sum_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    ids_ptr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    eps,
    APPLY_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PDL: tl.constexpr = False,
    FUSE: tl.constexpr = False,   # OKC_SW_FINALIZE_AR: + the TP8 all-reduce, all_reduce_2shot_push_kernel's protocol
    ws_ptr=None,   # int64 [8]: every rank's push workspace base
    cnt_ptr=None,   # this rank's push counters (epochs)
    slot=0,   # push slot bytes
    rank=0,
    R=1,   # rows per rank = tokens / 8; rank c owns rows [c R, (c + 1) R)
    ncnt=0,
):
    if PDL:   # programmatic dependent launch: wait for the expert GEMM before any load or store
        tlc.gdc_wait()
        tlc.gdc_launch_dependents()
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in tl.static_range(TOPK):
        base = (tok * TOPK + k) * H
        live = True
        if HAS_IDS:
            live = tl.load(ids_ptr + tok * TOPK + k) >= 0
        if live:
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

            if APPLY_NORM:
                if HAS_WEIGHT and WEIGHT_FIRST:
                    y = x * w
                    var = tl.sum(y * y, axis=0) / H
                    y = y * tl.rsqrt(var + eps)
                elif HAS_WEIGHT:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps) * w
                else:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps)
            else:
                y = x

            s = tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
            acc += y * s

    if not FUSE:
        tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        # bf16 pairs as 32-bit atoms; an all-zero atom is the lamport not-arrived marker, so it is sent as (-0, +0)
        lo, hi = tl.split(tl.reshape(acc.to(tl.bfloat16).to(tl.int16, bitcast=True), (BLOCK_H // 2, 2)))
        v = (hi.to(tl.int32) << 16) | (lo.to(tl.int32) & 0xFFFF)
        v = tl.where(v == 0, 0x8000, v)
        e = (tl.load(cnt_ptr + tok) & 1).to(tl.int64)
        own = tok // R
        lane = (tok - own * R) * H * 2 + tl.arange(0, BLOCK_H // 2) * 4   # the row's bytes in a slot half
        mine = tl.load(ws_ptr + rank) + e * slot * 8
        if own != rank:   # push my partial to the owner, take the owner's sum back
            _st_sys((tl.load(ws_ptr + own) + e * slot * 8 + rank * slot + lane).to(tl.pointer_type(tl.int32)), v)
            v = _poll((mine + own * slot + R * H * 2 + lane).to(tl.pointer_type(tl.int32)))
        else:   # reduce the 8 partials in rank order (fp32), as reduce_vec, then fan the sum out
            sl = tl.zeros([BLOCK_H // 2], tl.float32)
            sh = tl.zeros([BLOCK_H // 2], tl.float32)
            for i in tl.static_range(8):
                if i == rank:
                    u = v
                else:
                    u = _poll((mine + i * slot + lane).to(tl.pointer_type(tl.int32)))
                fl = (u & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
                fh = (u >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
                if i == 0:   # reduce_vec starts from rank 0's value (keeps a -0), not from +0
                    sl, sh = fl, fh
                else:
                    sl, sh = sl + fl, sh + fh
            lo = sl.to(tl.bfloat16).to(tl.int16, bitcast=True)
            hi = sh.to(tl.bfloat16).to(tl.int16, bitcast=True)
            v = (hi.to(tl.int32) << 16) | (lo.to(tl.int32) & 0xFFFF)
            v = tl.where(v == 0, 0x8000, v)
            for i in tl.static_range(8):
                if i != rank:
                    _st_sys((tl.load(ws_ptr + i) + e * slot * 8 + rank * slot + R * H * 2 + lane).to(
                        tl.pointer_type(tl.int32)), v)
        tl.store((out_ptr + tok * H).to(tl.pointer_type(tl.int32), bitcast=True) + tl.arange(0, BLOCK_H // 2), v)
        tl.store(cnt_ptr + tok, (e ^ 1).to(tl.uint32))
        if tok == 0:   # every counter flips once per call, also the ones past this grid
            c = tl.num_programs(0) + tl.arange(0, 256)
            tl.store(cnt_ptr + c, (e ^ 1).to(tl.uint32), mask=c < ncnt)


def _okc_finalize_ar(out):
    """OKC_SW_FINALIZE_AR: the TP8 custom all-reduce the caller would run on `out`, to fuse into the finalize; None if
    it does not apply. Its peer table is built at the first eager call (the graph runner's warm-up), keyed on the push
    counters (reserve_push_capacity replaces the workspace)."""
    from sglang.kernels.ops.muse_spark_v12 import okc_patch
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import CustomAllReduceV2
    from sglang.srt.runtime_context import get_forward

    t = out.shape[0]
    if not okc_patch.SW_FINALIZE_AR or t % 8 or t > 128 or torch.compiler.is_compiling() or get_forward().sp_active:
        return None
    tp = get_tp_group()
    c = tp.ca_comm
    if (tp.world_size != 8 or not isinstance(c, CustomAllReduceV2) or c.disabled or t > c._push_counter.numel()
            or t * out.shape[1] // 2 > c.max_push_size or tp._can_use_flashinfer_allreduce(out)
            or tp._resolve_outplace_all_reduce_method(out) != "ca"):
        return None
    if getattr(c, "_okc_finar_key", None) != c._push_counter.data_ptr():
        if torch.cuda.is_current_stream_capturing():
            return None
        c._okc_finar_ptrs = torch.tensor([w.data_ptr() for w in c._push_workspaces], dtype=torch.int64,
                                         device=out.device)
        c._okc_finar_key = c._push_counter.data_ptr()
    return c


def fused_post_expert_norm_weighted_sum(
    expert_output: torch.Tensor,
    topk_scores: torch.Tensor,
    norm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    weight_first: bool = True,
    apply_norm: bool = True,
    slot_expert_ids: Optional[torch.Tensor] = None,
    min_tokens_for_fused: int = _MIN_TOKENS_FOR_FUSED,
    all_reduce: bool = False,
) -> torch.Tensor:
    """all_reduce (OKC): also run the TP all-reduce in the kernel when it applies; the result is then marked
    `_okc_reduced` and the caller skips its own all-reduce."""
    x = expert_output.contiguous()
    tokens, topk, hidden = x.shape
    out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    c = _okc_finalize_ar(out) if all_reduce and tokens >= min_tokens_for_fused else None
    ar = {} if c is None else dict(FUSE=True, ws_ptr=c._okc_finar_ptrs, cnt_ptr=c._push_counter,
                                   slot=c.max_push_size, rank=c.rank, R=tokens // 8, ncnt=c._push_counter.numel())
    if tokens < min_tokens_for_fused:
        y = x
        if slot_expert_ids is not None:
            y = torch.where(
                (slot_expert_ids.reshape(tokens, topk) >= 0).unsqueeze(-1),
                y,
                torch.zeros((), dtype=y.dtype, device=y.device),
            )
        if apply_norm:
            f = y.float()
            if norm_weight is not None and weight_first:
                f = f * norm_weight.float()
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
            elif norm_weight is not None:
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
                f = f * norm_weight.float()
            else:
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
            y = f
        else:
            y = y.float()
        y = y * topk_scores.float().unsqueeze(-1)
        return y.sum(dim=1).to(x.dtype)

    block_h = triton.next_power_of_2(hidden)
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        x,
        norm_weight if norm_weight is not None else x,
        topk_scores.contiguous(),
        out,
        slot_expert_ids if slot_expert_ids is not None else x,
        TOPK=topk,
        H=hidden,
        eps=eps,
        APPLY_NORM=apply_norm,
        HAS_WEIGHT=norm_weight is not None,
        WEIGHT_FIRST=weight_first,
        HAS_IDS=slot_expert_ids is not None,
        BLOCK_H=block_h,
        PDL=SW_PDL,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
        launch_pdl=SW_PDL,
        **ar,
    )
    if c is not None:
        out._okc_reduced = True
    return out


__all__ = ["fused_post_expert_norm_weighted_sum"]
