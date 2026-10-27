"""Lazy loader for the pg inline CUDA extension (built once per box, ~1-2 min of nvcc at the
first prefill that dispatches to it). The hand wgmma PTX needs the sm_90a feature target."""

import os
import threading

_lock = threading.Lock()
_PG_EXT = None


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
                    name="okc_pg_moe_c350",
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
