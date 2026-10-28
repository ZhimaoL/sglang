"""Multicast all-gather fused with the FFN norm-residual boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.muse_spark_v12.norm_residual import (
    _cluster_launch_config,
    norm_residual_launch_config,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

_VEC = 8


@cache_once
def _supports_clusters() -> bool:
    capability = torch.cuda.get_device_capability()
    if capability[0] == 9:
        from sglang.kernels.ops.muse_spark_v12 import okc_patch

        # OKC: the fused gather + boundary uses sm_90 features only (multimem stores, clusters, DSMEM).
        return okc_patch.SW_AG_NORM_HOPPER
    return capability[0] >= 10


def _launch_config(hidden_size: int, num_tokens: int) -> tuple[int, int, bool]:
    """norm_residual_launch_config, with the Hopper cluster rule of norm_residual.fused_norm_residual (VG4:
    clusters only at M <= 32, capped at OKC_HOPPER_NORM_MAX_CLUSTER)."""
    config = norm_residual_launch_config(hidden_size=hidden_size, num_tokens=num_tokens)
    if torch.cuda.get_device_capability()[0] != 9:
        return config
    from sglang.kernels.ops.muse_spark_v12 import okc_patch

    if okc_patch.SW_AG_NORM_WIDE and 32 < num_tokens <= 128 and hidden_size % 8192 == 0:
        return 1, 1024, True   # VG17: one pack per thread, the whole row's polls and loads in flight at once
    if config[0] == 1:
        return config

    cluster_size = config[0] if num_tokens <= 32 else 1
    while cluster_size > okc_patch.HOPPER_NORM_MAX_CLUSTER:
        cluster_size //= 2
    if cluster_size == config[0]:
        return config
    threads, prefetch = _cluster_launch_config(hidden_size, num_tokens, cluster_size)
    return cluster_size, threads, prefetch


@cache_once
def _jit_module(
    hidden_size: int,
    world_size: int,
    config: tuple[int, int, bool],
    post_norm_is_input_scale: bool,
    has_post_weight: bool,
    has_gate: bool,
) -> Module:
    cluster_size, num_threads, prefetch_weights = config
    args = make_cpp_args(
        hidden_size,
        world_size,
        num_threads,
        cluster_size,
        is_arch_support_pdl(),
        post_norm_is_input_scale,
        prefetch_weights,
        has_post_weight,
        has_gate,
    )
    return load_jit(
        "muse_spark_v12_all_gather_norm_residual",
        *args,
        cuda_files=["muse_spark_v12/all_gather_norm_residual.cuh"],
        cuda_wrappers=[("run", f"MuseSparkAllGatherNormResidual<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def all_gather_norm_residual_supported(
    comm: Optional[CustomAllReduceV2], num_tokens: int, hidden_size: int
) -> bool:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    if (
        not isinstance(comm, CustomAllReduceV2)
        or comm.disabled
        or not comm.has_multicast
        or comm.world_size < 2
        or num_tokens <= 0
        or hidden_size % (comm.world_size * _VEC)
        or num_tokens * hidden_size * 2 > comm.max_push_size
    ):
        return False
    if not _supports_clusters():
        return False
    if torch.cuda.get_device_capability()[0] == 9:
        from sglang.kernels.ops.muse_spark_v12 import okc_patch

        # H200 x8 (session SI vg-micro, us per boundary fused vs separate): cap 16 M 8 / 16 / 32 -0.15 / -0.16 /
        # -0.85, M 64 +0.45, M 128 -0.87; with the row kernel (cap 1) M 16 / 32 / 64 lose (+0.4 / +1.0 / +0.3).
        if not ((num_tokens <= 32 and okc_patch.HOPPER_NORM_MAX_CLUSTER > 1) or num_tokens >= 128
                or (okc_patch.SW_AG_NORM_WIDE and num_tokens <= 128)):
            return False
    try:
        cluster_size, _, _ = _launch_config(hidden_size, num_tokens)
    except RuntimeError:
        return False
    return comm.config.num_push_blocks >= 2 and (
        cluster_size == 1 or num_tokens * cluster_size <= comm.config.num_push_blocks
    )


def all_gather_norm_residual(
    comm: CustomAllReduceV2,
    local: torch.Tensor,
    stream: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    *,
    post_norm_is_input_scale: bool = False,
    post_weight: Optional[torch.Tensor] = None,
    pre_weight: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return BF16 normalized output and FP32 residual for the gathered branch."""
    num_tokens, hidden_size = stream.shape
    config = _launch_config(hidden_size, num_tokens)
    output = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=stream.device
    )
    new_stream = torch.empty_like(stream)

    module = _jit_module(
        hidden_size,
        comm.world_size,
        config,
        post_norm_is_input_scale,
        post_weight is not None,
        alpha is not None,
    )
    module.run(
        comm.obj,
        local.contiguous(),
        stream,
        output,
        new_stream,
        post_weight,
        pre_weight,
        alpha,
        beta,
        post_eps,
        pre_eps,
    )
    return output, new_stream
