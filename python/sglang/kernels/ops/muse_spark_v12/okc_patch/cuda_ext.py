"""Lazy loaders for the OKC inline CUDA extensions, each built once per box (~1-2 min of nvcc at first use).
pg carries hand wgmma PTX, which needs the sm_90a feature target."""

import os
import threading

_lock = threading.Lock()
_PG_EXT = None
_ST_EXT = None
_PROJ_EXT = None

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


def pg_ext():
    global _PG_EXT
    if _PG_EXT is None:
        with _lock:
            if _PG_EXT is None:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_pg import (
                    _CUDA_SRC,
                )

                _PG_EXT = load_inline(
                    name="okc_pg_moe_c350d3",
                    cpp_sources="void tiles(torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, int64_t, int64_t, int64_t, int64_t);\n"
                    "void gather(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "int64_t);\n"
                    "void gg1(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);\n"
                    "void gg2(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);\n"
                    "void set_snake_max_t(int64_t);",
                    cuda_sources=_CUDA_SRC,
                    functions=["tiles", "gather", "gg1", "gg2", "set_snake_max_t"],
                    extra_cuda_cflags=[
                        "-O3",
                        "--use_fast_math",
                        "-gencode=arch=compute_90a,code=sm_90a",
                    ],
                    verbose=False,
                )
    return _PG_EXT


def st_ext():
    """The st small-T MoE pair (okc_patch/st_moe.py): reads pg's packs, hand mma.sync PTX + griddepcontrol (sm_90a)."""
    global _ST_EXT
    if _ST_EXT is None:
        with _lock:
            if _ST_EXT is None:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_st import (
                    _CUDA_SRC,
                )

                _ST_EXT = load_inline(
                    name="okc_st_moe_c2",
                    cpp_sources="void st_moe(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, int64_t);",
                    cuda_sources=_CUDA_SRC,
                    functions=["st_moe"],
                    extra_cuda_cflags=[
                        "-O3",
                        "--use_fast_math",
                        "-gencode=arch=compute_90a,code=sm_90a",
                    ],
                    verbose=False,
                )
    return _ST_EXT


def proj_ext():
    global _PROJ_EXT
    if _PROJ_EXT is None:
        with _lock:
            if _PROJ_EXT is None:
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_proj import (
                    _CUDA_SRC,
                )

                # Flags match the certified kernel-track build; the name differs because the router GEMV and
                # reduce_k<2> changed (a shared name would rebuild over that lane's torch_extensions cache).
                _PROJ_EXT = load_inline(
                    name="okc_v5_stream_det",
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
