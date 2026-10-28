"""OKC pinned cuBLASLt configs for dense projections at verify M (serving/kernels/lt; RESULTS.md has the times).

Claims only measured (N, K, M) entries that beat the stock route by >= 5% per call in two runs (cuBLASLt 13.1.1;
other versions decline); reads the resident [N, K] bf16 weight, fp32 split-K partials. Dispatched after smallm;
prepare(weight) at weight load validates (AlgoCheck) and warms the entries of the weight and its row slices.
"""

import threading

import torch

from sglang.kernels.ops.muse_spark_v12.okc_patch import SW_LT
LT_VERSION = 130101

# (N, K) -> {M: (layout, [algo, tile, splitk, reduction, swizzle, custom, stages, inner, cluster], workspace bytes)}
# BEGIN TABLE
_TABLE = {
    (2304, 8192): {  # qkv_gate
        144: (0, (66, 315, 1, 0, 0, 1, 35, 0, 2), 0),
        160: (0, (66, 316, 1, 0, 0, 1, 35, 0, 2), 0),
        176: (1, (66, 217, 4, 2, 0, 0, 35, 0, 2), 6488080),
        184: (1, (66, 15, 1, 0, 0, 0, 35, 0, 2), 0),
        192: (1, (66, 15, 1, 0, 0, 0, 35, 0, 2), 0),
        200: (0, (66, 317, 1, 0, 0, 1, 35, 0, 2), 0),
        208: (1, (66, 317, 1, 0, 0, 1, 35, 0, 2), 0),
        216: (0, (66, 317, 1, 0, 0, 1, 35, 0, 2), 0),
        232: (1, (66, 318, 1, 0, 0, 1, 35, 0, 2), 0),
        248: (0, (66, 124, 1, 0, 0, 0, 35, 0, 2), 0),
        256: (0, (66, 124, 1, 0, 0, 0, 35, 0, 2), 0),
    },
    (512, 8192): {  # pooler slice
        80: (0, (66, 318, 8, 2, 0, 1, 35, 0, 4), 1310736),
        88: (1, (66, 16, 8, 2, 0, 5, 35, 0, 2), 1441808),
        112: (1, (66, 15, 8, 2, 0, 0, 35, 0, 2), 1835024),
        120: (0, (66, 15, 8, 2, 0, 0, 35, 0, 2), 1966096),
        128: (0, (66, 15, 8, 2, 0, 2, 35, 0, 5), 2097168),
    },
    (1024, 4096): {  # unpooler slice
        40: (0, (66, 15, 8, 2, 0, 0, 35, 0, 2), 1310736),
        48: (0, (66, 15, 8, 2, 0, 0, 35, 0, 2), 1572880),
        56: (1, (66, 15, 8, 2, 0, 2, 35, 0, 2), 1835024),
        64: (1, (66, 15, 8, 2, 0, 2, 35, 0, 2), 2097168),
        72: (1, (66, 317, 4, 2, 0, 1, 35, 0, 2), 1179664),
        80: (0, (66, 319, 8, 2, 0, 3, 35, 0, 2), 2621456),
        88: (0, (66, 315, 4, 2, 0, 1, 35, 0, 2), 1441808),
        96: (1, (66, 26, 8, 2, 0, 1, 35, 0, 2), 3145744),
        104: (0, (66, 15, 4, 2, 0, 0, 35, 0, 2), 1703952),
        112: (1, (66, 15, 4, 2, 0, 0, 35, 0, 2), 1835024),
        120: (1, (66, 15, 4, 2, 0, 0, 35, 0, 2), 1966096),
        128: (1, (66, 15, 4, 2, 0, 2, 35, 0, 5), 2097168),
    },
    (4096, 8192): {  # pooler
        136: (0, (66, 317, 1, 0, 0, 1, 35, 0, 5), 0),
        144: (0, (66, 317, 1, 0, 0, 1, 35, 0, 5), 0),
        152: (1, (66, 318, 1, 0, 0, 5, 35, 0, 2), 0),
        168: (1, (66, 319, 1, 0, 0, 3, 35, 0, 5), 0),
        176: (0, (66, 319, 1, 0, 0, 1, 35, 0, 2), 0),
        184: (1, (66, 27, 1, 0, 0, 0, 35, 0, 2), 0),
        192: (1, (66, 26, 1, 0, 0, 1, 35, 0, 2), 0),
        200: (1, (66, 398, 1, 0, 0, 1, 35, 0, 2), 0),
        208: (1, (66, 320, 1, 0, 0, 1, 35, 0, 5), 0),
        216: (1, (66, 398, 1, 0, 0, 1, 35, 0, 2), 0),
        224: (0, (66, 398, 1, 0, 0, 1, 35, 0, 2), 0),
        232: (0, (66, 322, 1, 0, 0, 1, 35, 0, 5), 0),
        240: (1, (66, 322, 1, 0, 0, 1, 35, 0, 2), 0),
        248: (1, (66, 18, 1, 0, 0, 0, 35, 0, 2), 0),
        256: (1, (66, 18, 1, 0, 0, 0, 35, 0, 2), 0),
    },
    (8192, 4096): {  # unpooler
        176: (1, (66, 401, 1, 0, 0, 5, 35, 0, 2), 0),
        192: (0, (66, 33, 1, 0, 0, 5, 35, 0, 2), 0),
        208: (1, (66, 402, 1, 0, 0, 5, 35, 0, 2), 0),
        224: (0, (66, 403, 1, 0, 0, 7, 35, 0, 5), 0),
    },
}
# END TABLE

_WS_BYTES = (max([e[2] for r in _TABLE.values() for e in r.values()] + [1]) + (1 << 20) - 1) >> 20 << 20
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
            layout, cfg, _ = _TABLE[(n, k)][m]
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
