"""OKC_SW_WSC2: W-staged own GEMMs at verify m <= 32 (muse-opt serving/kernels/megakernel LOG M5-M5.4).

qkv_gate [2304, 8192] at m 17-32 runs KQ3 and o_proj [8192, 1024] at m 1-32 runs KO2b (_cuda_src_wsc2.py). Both are
PDL-launched and issue their weight before griddepcontrol.wait, so it streams while the kernel in front still runs --
when their CTAs fit beside that kernel's CTAs. So the kernel in front changes too, and only when it feeds a GEMM claimed
here: the FFN-boundary all-gather norm (it already triggers right after its wait) launches with a 100 % shared-memory
carveout from m 25; VP2 (sliding verify attention) triggers after its prologue and launches with the carveout. Each
marks the GEMM behind it with its own output's address (mark_q / mark_o), and gemm() claims only a GEMM whose x is that
tensor, once.
A dependent still waits for its predecessor's completion before it reads x, so the outputs do not depend on the overlap
(they are KQ3's / KO2b's, not cuBLAS's).
Dev2 micro, windows with the served footprints (MKW1b): qkv m 32 -5.2 us, o m 32 -3.4 us per layer.
"""

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch import SW_WSC2

QKV = (2304, 8192)
O_PROJ = (8192, 1024)
CARVEOUT = 100   # percent: the 228 KB split, which holds the kernel-in-front's CTA beside a staged GEMM CTA
_MARK = {"q": 0, "o": 0}   # data_ptr() of the predecessor's output that the next qkv / o GEMM must read (0 = none)


def claims_q(boundary, m: int) -> bool:
    """Whether the qkv GEMM behind the FFN-boundary all-gather norm (boundary._okc_wsc2_next = the next layer's qkv
    shape, set by the model) runs KQ3. The caller then marks the norm's output with mark_q()."""
    return bool(SW_WSC2) and getattr(boundary, "_okc_wsc2_next", None) == QKV and 17 <= m <= 32


def agnorm_carveout(boundary, m: int) -> int:
    """Carveout for that all-gather norm, -1 = unchanged launch. From m 25 only: at m 24 the stock norm in front of KQ3
    was 0.38 us faster (MKW2C)."""
    return CARVEOUT if claims_q(boundary, m) and m >= 25 else -1


def vp2_early(m: int) -> bool:
    """Whether VP2 (m = bs x 8 rows) triggers after its prologue with the carveout. The caller marks VP2's output with
    mark_o()."""
    return bool(SW_WSC2) and 1 <= m <= 32


def mark_q(t) -> None:
    _MARK["q"] = t.data_ptr()


def mark_o(t) -> None:
    _MARK["o"] = t.data_ptr()


def gemm(x, weight):
    """KQ3 / KO2b when x is the output the kernel in front marked; None = not claimed."""
    k = x.shape[-1]
    m = x.numel() // k if k else 0
    nk = tuple(weight.shape)
    p = x.data_ptr()
    if nk == QKV and p and _MARK["q"] == p and 17 <= m <= 32:
        _MARK["q"] = 0
        kind = 0
    elif nk == O_PROJ and p and _MARK["o"] == p and 1 <= m <= 32:
        _MARK["o"] = 0
        kind = 1
    else:
        return None
    x2 = x.reshape(m, k)
    if not (x2.is_contiguous() and weight.is_contiguous()) or x2.data_ptr() % 16 or weight.data_ptr() % 16:
        return None
    from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import wsc2_ext

    out = torch.empty((m, nk[0]), dtype=torch.bfloat16, device=x.device)
    wsc2_ext().wsc2_launch(kind, weight, x2, out)
    return out.view(*x.shape[:-1], nk[0])
