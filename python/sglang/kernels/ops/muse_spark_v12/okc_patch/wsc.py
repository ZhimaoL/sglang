"""OKC_SW_WSC: the launch options of the kernel in front of a W-staged small-m GEMM (okc_patch/smallm.py).

A W-staged GEMM is PDL-launched and loads its weight slice before griddepcontrol.wait, but its CTAs (75-198 KB of
shared memory) only land beside the CTAs of the kernel in front of it when that kernel has triggered its dependents and
runs with a big shared-memory split; without the split they land as it drains, and the staging hides ~nothing (muse-opt
serving/kernels/wstage LOG W4: -0.1..-0.8 us per GEMM without the carveout, -2.4..-4.3 with it). So the three kernels in
front of staged GEMMs -- the combine all-reduce before the unpooler slice, the FFN boundary norm (all-gather norm)
before qkv_gate, the post-attention boundary norm before the pooler slice -- launch with CARVEOUT and trigger right after
their own wait, and only when the GEMM behind them is staged. Outputs are unchanged: a dependent still waits for its
predecessor's completion before it reads anything that kernel wrote.
"""

import contextlib

from sglang.kernels.ops.muse_spark_v12.okc_patch import SW_WSC

CARVEOUT = 100   # percent: the 228 KB split, which holds the predecessor's CTA beside a staged GEMM CTA
NONE = (False, -1)
_AR = [NONE]


def opts(staged: bool):
    """(early_trigger, carveout) for a predecessor whose next GEMM is staged or not."""
    return (True, CARVEOUT) if staged else NONE


@contextlib.contextmanager
def ar_scope(staged: bool):
    """The all-reduce launched inside takes opts(staged) (push algorithms; custom_all_reduce_v2 reads ar_opts())."""
    prev = _AR[0]
    _AR[0] = opts(staged)
    try:
        yield
    finally:
        _AR[0] = prev


def ar_opts():
    """(early_trigger, carveout) for the all-reduce being launched now."""
    return _AR[0]


def boundary_opts(boundary, m: int):
    """(early_trigger, carveout) for a boundary norm whose output feeds the GEMM boundary._okc_wsc_next = (n, k) at m
    rows (set by the model; absent = no staged GEMM behind it)."""
    nk = getattr(boundary, "_okc_wsc_next", None)
    if not SW_WSC or nk is None:
        return NONE
    from sglang.kernels.ops.muse_spark_v12.okc_patch.smallm import wsc_staged

    return opts(wsc_staged(nk[0], nk[1], m))
