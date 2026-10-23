"""Lazy loaders for the two OKC inline CUDA extensions.

The sources live in _cuda_src_proj.py / _cuda_src_moe.py byte-identical to the
certified kernel-track lanes, and the load_inline names and flags match them, so a
box that already built the kernel-track extensions hits the torch_extensions cache.
First cold build is ~1-3 min of nvcc at server start (only when a switch needs it).
"""

import threading

_lock = threading.Lock()
_PROJ_EXT = None
_MOE_EXT = None

_PROJ_CPP = """
void gemv_bf16_launch(at::Tensor W, at::Tensor X, at::Tensor out, int64_t m, int64_t n,
                      int64_t k, int64_t MT, int64_t R, int64_t U, int64_t WPC);
void gemv_router_launch(at::Tensor Wp, at::Tensor X, at::Tensor out, int64_t m, int64_t k,
                        int64_t MT, int64_t SK, int64_t U, int64_t WPC);
void mma_launch(at::Tensor Wp, at::Tensor X, at::Tensor out, int64_t m, int64_t nout,
                int64_t k, int64_t MR, int64_t SPLIT, int64_t SK, int64_t WPC, int64_t G);
void cmma_launch(at::Tensor Wm, at::Tensor We, at::Tensor X, at::Tensor out, int64_t m,
                 int64_t nout, int64_t k, int64_t MR, int64_t SPLIT, int64_t SK,
                 int64_t WPC, int64_t EB);
void reduce_launch(at::Tensor P, at::Tensor out, int64_t m, int64_t nout, int64_t SK,
                   int64_t FOLD);
"""


def proj_ext():
    global _PROJ_EXT
    if _PROJ_EXT is None:
        with _lock:
            if _PROJ_EXT is None:
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_proj import (
                    _CUDA_SRC,
                )

                _PROJ_EXT = load_inline(
                    name="okc_v5_stream",
                    cpp_sources=[_PROJ_CPP],
                    cuda_sources=[_CUDA_SRC],
                    functions=[
                        "gemv_bf16_launch",
                        "gemv_router_launch",
                        "mma_launch",
                        "cmma_launch",
                        "reduce_launch",
                    ],
                    extra_cuda_cflags=["-O3"],
                    verbose=False,
                )
    return _PROJ_EXT


def moe_ext():
    global _MOE_EXT
    if _MOE_EXT is None:
        with _lock:
            if _MOE_EXT is None:
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_moe import (
                    _CUDA_SRC,
                )

                decl = (
                    "void {}(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "int64_t, int64_t, int64_t);"
                )
                _MOE_EXT = load_inline(
                    name="moe_w4a16_v14x",
                    cpp_sources="\n".join(decl.format(f) for f in ("k1", "k2", "k1m", "k2m"))
                    + "\nvoid k0x(torch::Tensor, torch::Tensor, int64_t);"
                    + "\nvoid k0f(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);"
                    + "\nvoid k3c(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, int64_t, int64_t, double);",
                    cuda_sources=_CUDA_SRC,
                    functions=["k0x", "k0f", "k3c", "k1", "k2", "k1m", "k2m"],
                    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90a"],
                    verbose=False,
                )
    return _MOE_EXT
