"""OKC_SW_WSC2: W-staged own GEMMs at verify m <= 32 (muse-opt serving/kernels/megakernel LOG M5-M5.4).

qkv_gate [2304, 8192] at m 17-32 runs KQ3 and o_proj [8192, 1024] at m 1-32 runs KO2b (_cuda_src_wsc2.py). Both are
PDL-launched and issue their weight before griddepcontrol.wait, so it streams while the kernel in front still runs --
when their CTAs fit beside that kernel's CTAs. So the kernel in front changes too, and only when it feeds a GEMM claimed
here: the FFN-boundary all-gather norm (it already triggers right after its wait) launches with a 100 % shared-memory
carveout; VP2 (sliding verify attention) triggers at entry and launches with the carveout. Each marks the GEMM behind it
(agnorm_carveout / vp2_early) and gemm() runs a marked GEMM once. A dependent still waits for its predecessor's
completion before it reads x, so the outputs do not depend on the overlap (they are KQ3's / KO2b's, not cuBLAS's).
Dev2 micro, windows with the served footprints (MKW1b): qkv m 32 -5.2 us, o m 32 -3.4 us per layer.
"""

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch import SW_WSC2

QKV = (2304, 8192)
O_PROJ = (8192, 1024)
CARVEOUT = 100   # percent: the 228 KB split, which holds the kernel-in-front's CTA beside a staged GEMM CTA
_MARK = {"q": False, "o": False}


def agnorm_carveout(boundary, m: int) -> int:
    """Carveout for the FFN-boundary all-gather norm (boundary._okc_wsc2_next = the next layer's qkv shape, set by the
    model); marks the qkv GEMM behind it. -1 = unchanged launch."""
    if not SW_WSC2 or getattr(boundary, "_okc_wsc2_next", None) != QKV or not 17 <= m <= 32:
        return -1
    _MARK["q"] = True
    return CARVEOUT


def vp2_early(m: int) -> bool:
    """Whether VP2 (m = bs x 8 rows) triggers at entry with the carveout; marks the o_proj GEMM behind it."""
    if not SW_WSC2 or not 1 <= m <= 32:
        return False
    _MARK["o"] = True
    return True


def gemm(x, weight):
    """KQ3 / KO2b for the GEMM the kernel in front marked; None = not claimed."""
    k = x.shape[-1]
    m = x.numel() // k if k else 0
    nk = tuple(weight.shape)
    if nk == QKV and _MARK["q"] and 17 <= m <= 32:
        _MARK["q"] = False
        kind = 0
    elif nk == O_PROJ and _MARK["o"] and 1 <= m <= 32:
        _MARK["o"] = False
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
