"""OKC pinned cuBLASLt configs for dense projections at verify M (serving/kernels/lt; RESULTS.md has the times).

Claims only measured (N, K, M) entries that beat the stock route by >= 5% per call in two runs (cuBLASLt 13.1.1;
other versions decline); reads the resident [N, K] bf16 weight, fp32 split-K partials. Dispatched after smallm;
prepare(weight) at weight load validates (AlgoCheck) and warms the entries of the weight and its row slices.
"""

import threading

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch import SW_LT
LT_VERSION = 130101

# (N, K) -> {M: (layout, tile, split-K, custom option, cluster shape)}; every entry is algo 66 (nvjet), stages
# 64xAUTO, no swizzle, fp32 split-K partials (lt_mktable.py --compact).
# BEGIN TABLE
_TABLE = {
    (2304, 8192): {  # qkv_gate
        144: (0, 315, 1, 1, 2),
        160: (0, 316, 1, 1, 2),
        176: (1, 217, 4, 0, 2),
        184: (1, 15, 1, 0, 2),
        192: (1, 15, 1, 0, 2),
        200: (0, 317, 1, 1, 2),
        208: (1, 317, 1, 1, 2),
        216: (0, 317, 1, 1, 2),
        232: (1, 318, 1, 1, 2),
        248: (0, 124, 1, 0, 2),
        256: (0, 124, 1, 0, 2),
    },
    (512, 8192): {  # pooler slice
        80: (0, 318, 8, 1, 4),
        88: (1, 16, 8, 5, 2),
        112: (1, 15, 8, 0, 2),
        120: (0, 15, 8, 0, 2),
        128: (0, 15, 8, 2, 5),
    },
    (1024, 4096): {  # unpooler slice
        40: (0, 15, 8, 0, 2),
        48: (0, 15, 8, 0, 2),
        56: (1, 15, 8, 2, 2),
        64: (1, 15, 8, 2, 2),
        72: (1, 317, 4, 1, 2),
        80: (0, 319, 8, 3, 2),
        88: (0, 315, 4, 1, 2),
        96: (1, 26, 8, 1, 2),
        104: (0, 15, 4, 0, 2),
        112: (1, 15, 4, 0, 2),
        120: (1, 15, 4, 0, 2),
        128: (1, 15, 4, 2, 5),
    },
    (4096, 8192): {  # pooler
        136: (0, 317, 1, 1, 5),
        144: (0, 317, 1, 1, 5),
        152: (1, 318, 1, 5, 2),
        168: (1, 319, 1, 3, 5),
        176: (0, 319, 1, 1, 2),
        184: (1, 27, 1, 0, 2),
        192: (1, 26, 1, 1, 2),
        200: (1, 398, 1, 1, 2),
        208: (1, 320, 1, 1, 5),
        216: (1, 398, 1, 1, 2),
        224: (0, 398, 1, 1, 2),
        232: (0, 322, 1, 1, 5),
        240: (1, 322, 1, 1, 2),
        248: (1, 18, 1, 0, 2),
        256: (1, 18, 1, 0, 2),
    },
    (8192, 4096): {  # unpooler
        176: (1, 401, 1, 5, 2),
        192: (0, 33, 1, 5, 2),
        208: (1, 402, 1, 5, 2),
        224: (0, 403, 1, 7, 5),
    },
    (8192, 1024): {  # o (target and draft)
        64: (1, 15, 1, 1, 2),
    },
    (1280, 8192): {  # draft qkv
        8: (0, 448, 8, 1, 4),
        24: (0, 193, 8, 6, 4),
        48: (1, 395, 4, 3, 5),
        56: (0, 134, 4, 2, 2),
        120: (0, 134, 4, 6, 5),
        144: (0, 397, 4, 5, 2),
        160: (0, 398, 4, 5, 2),
        168: (1, 398, 4, 5, 2),
        192: (1, 18, 4, 4, 2),
        200: (1, 399, 4, 5, 2),
        216: (1, 399, 4, 5, 2),
        232: (1, 193, 4, 4, 2),
        240: (1, 193, 4, 4, 2),
        248: (1, 135, 4, 6, 5),
    },
    (7168, 8192): {  # draft gate_up
        8: (0, 393, 1, 3, 4),
        16: (0, 394, 1, 7, 4),
        24: (0, 395, 1, 5, 4),
        192: (1, 33, 1, 5, 2),
        208: (1, 402, 1, 7, 5),
        224: (0, 403, 1, 7, 5),
        232: (0, 165, 1, 4, 2),
        240: (0, 165, 1, 4, 2),
    },
    (8192, 3584): {  # draft down
        144: (0, 399, 1, 5, 2),
        176: (1, 401, 1, 5, 2),
        192: (0, 33, 1, 7, 5),
        208: (0, 402, 1, 5, 2),
        224: (0, 403, 1, 5, 2),
    },
    (256, 8192): {  # draft conv kernel_projection slice
        8: (0, 104, 16, 2, 2),
        16: (0, 313, 16, 1, 2),
        24: (0, 313, 16, 1, 2),
        32: (1, 15, 16, 2, 2),
        40: (0, 12, 16, 4, 2),
        48: (1, 15, 16, 2, 2),
        56: (1, 12, 16, 4, 2),
        64: (1, 12, 16, 4, 3),
        72: (1, 314, 16, 3, 2),
        80: (1, 15, 16, 2, 2),
        88: (0, 15, 16, 2, 3),
        96: (1, 15, 16, 2, 2),
        104: (1, 15, 16, 2, 2),
        112: (1, 15, 16, 2, 2),
        120: (0, 15, 16, 2, 2),
        128: (0, 15, 16, 2, 3),
    },
}
# END TABLE

_WS_BYTES = 8 << 20  # >= every entry's AlgoCheck workspace (max 6.2 MiB); larger ones are rejected
_lock = threading.Lock()
_EXT = None
_OK = None
_IDX = {}  # (device, n, k, m) -> (index, workspace bytes, x align, w align); index < 0 = rejected
_WS = {}  # (device, stream) -> split-K workspace; per stream so streams never share one


def _ext():
    global _EXT, _OK
    if _OK is None:
        with _lock:
            if _OK is None:
                try:
                    from torch.utils.cpp_extension import load_inline

                    from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_lt import _CPP_SRC

                    _EXT = load_inline(name="okc_lt_v2", cpp_sources=[_CPP_SRC],
                                       functions=["lt_version", "lt_register", "lt_mm"], with_cuda=True,
                                       extra_cflags=["-O3"], extra_ldflags=["-lcublasLt"])
                    _OK = _EXT.lt_version() == LT_VERSION
                except Exception:  # noqa: BLE001
                    _OK = False
    return _EXT if _OK else None


def _register(dev, n, k, m):
    ent = _IDX.get((dev, n, k, m))
    if ent is None:
        ext = _ext()
        ent = (-1, 0, 1, 1)
        if ext is not None:
            layout, tile, sk, cu, cl = _TABLE[(n, k)][m]
            cfg = (66, tile, sk, 2 if sk > 1 else 0, 0, cu, 35, 0, cl)
            with torch.cuda.device(dev):
                idx, ws, al_a, al_b = ext.lt_register(m, n, k, layout, list(cfg), _WS_BYTES)
            ent = (idx, ws, max(al_a if layout else al_b, 1), max(al_b if layout else al_a, 1))
        _IDX[(dev, n, k, m)] = ent
    return ent


def _workspace(dev):
    key = (dev, torch.cuda.current_stream(dev).cuda_stream)
    if key not in _WS:  # a capture allocates it from the graph pool, as torch does for cuBLAS workspaces
        _WS[key] = torch.empty(_WS_BYTES, dtype=torch.uint8, device=f"cuda:{dev}")
    return _WS[key]


def lt_gemm(x, weight):
    """x [..., K] @ weight[N, K]^T (bf16) at a measured entry; None = declined."""
    if not SW_LT or torch.compiler.is_compiling() or weight.dim() != 2:
        return None
    n, k = weight.shape
    row = _TABLE.get((n, k))
    if row is None or x.shape[-1] != k or x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return None
    m = x.numel() // k
    if m not in row or not (x.is_cuda and x.is_contiguous() and weight.is_contiguous()):
        return None
    dev = x.device.index
    ent = _IDX.get((dev, n, k, m))
    if ent is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ent = _register(dev, n, k, m)
    idx, ws_bytes, al_x, al_w = ent
    if idx < 0 or x.data_ptr() % al_x or weight.data_ptr() % al_w:
        return None
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if not _EXT.lt_mm(x.view(m, k), weight, out, _workspace(dev) if ws_bytes else weight, idx):
        _IDX[(dev, n, k, m)] = (-1, 0, 1, 1)
        return None
    return out.view(*x.shape[:-1], n)


def prepare(weight):
    """At weight load: validate + warm the entries of weight's (N, K) and its row slices (N / d, K)."""
    if not SW_LT or _ext() is None or weight.dtype is not torch.bfloat16 or not weight.is_contiguous():
        return 0
    dev, (N, K), usable = weight.device.index, weight.shape, 0
    for (n, k), row in _TABLE.items():
        if k == K and N % n == 0:
            for m in row:
                if _register(dev, n, k, m)[0] >= 0:
                    usable += lt_gemm(torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device),
                                      weight[:n]) is not None
    return usable
