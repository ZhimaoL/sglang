"""OKC: the o-proj all-reduce fused with the post-attention norm_residual boundary (okc_ar_norm.cuh).

Bit-identical to all_reduce_2shot_push -> fused_norm_residual (cluster kernel) where it applies: TP world 8, H 8192,
M a multiple of 8 up to 32 (verify blocks of bs <= 4), the cluster path active for M, no router output.
"""
from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

HIDDEN = 8192
MAX_ROWS = 32


@cache_once
def _jit_module(hidden: int, world: int):
    args = make_cpp_args(hidden, world, is_arch_support_pdl())
    return load_jit(
        "okc_ar_norm",
        *args,
        cuda_files=["muse_spark_v12/okc_ar_norm.cuh"],
        cuda_wrappers=[
            ("run", f"OkcArNorm<{args}>::run"),
            ("run_emulated", f"OkcArNorm<{args}>::run_emulated"),
        ],
    )


def ar_norm_emulated(workspaces: torch.Tensor, counter: torch.Tensor, slot_bytes: int, rank: int, grid: int,
                     x: torch.Tensor, stream: torch.Tensor, output: torch.Tensor, new_stream: torch.Tensor,
                     post_weight: Optional[torch.Tensor], pre_weight: Optional[torch.Tensor],
                     alpha: Optional[torch.Tensor], beta: Optional[torch.Tensor], post_eps: float, pre_eps: float,
                     post_norm_is_input_scale: bool, cluster: int, world: int = 8) -> None:
    """One emulated rank on the current stream (tests: every rank's workspace lives on this GPU)."""
    _jit_module(x.shape[1], world).run_emulated(
        workspaces, counter, int(slot_bytes), int(rank), int(grid), x, stream, output, new_stream,
        post_weight, pre_weight, alpha, beta, float(post_eps), float(pre_eps), bool(post_norm_is_input_scale),
        int(cluster))


def attn_boundary_fusable(boundary, o_proj, probe: torch.Tensor, router_dtype) -> bool:
    """Whether this post-attention boundary can run as the fused kernel. ``probe`` has the o-proj output's shape and
    dtype (the attention input: [M, H] bf16); the o proj must all-reduce over the TP group with the custom all-reduce
    (whose algorithms all give the 2-shot push bits) and the norm must take the cluster path at this M."""
    from sglang.kernels.ops.muse_spark_v12 import okc_patch
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import CustomAllReduceV2
    from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
    from sglang.srt.runtime_context import get_exec, get_forward

    m = probe.shape[0]
    if (not okc_patch.SW_AR_NORM or router_dtype is not None or torch.compiler.is_compiling()
            or probe.dim() != 2 or probe.shape[1] != HIDDEN or probe.dtype is not torch.bfloat16
            or m == 0 or m % 8 or m > MAX_ROWS or not boundary.jit_selected()):
        return False
    # the o proj's own all-reduce must be the plain TP one (RowParallelLinear.forward)
    if (not o_proj.reduce_results or o_proj.use_dp_attention_reduce or getattr(o_proj, "use_decode_attn_tp", False)
            or should_skip_mlp_all_reduce() or get_exec().comm.enable_quant_communications
            or get_forward().sp_active):
        return False
    tp = get_tp_group()
    comm = tp.ca_comm
    if tp.world_size != 8 or not isinstance(comm, CustomAllReduceV2) or comm.disabled:
        return False
    if torch.cuda.get_device_capability(probe.device)[0] != 9 or okc_patch.HOPPER_NORM_MAX_CLUSTER < 4:
        return False
    if tp._can_use_flashinfer_allreduce(probe) or tp._resolve_outplace_all_reduce_method(probe) != "ca":
        return False
    return _virtual_cluster(m) in (4, 8, 16)


def _virtual_cluster(m: int) -> int:
    # fused_norm_residual's automatic choice on sm_90 (verify-glue VG4), which the kernel's reduction tree reproduces
    from sglang.kernels.ops.muse_spark_v12 import okc_patch
    from sglang.kernels.ops.muse_spark_v12.norm_residual import _cluster_size_for_shape

    c = _cluster_size_for_shape(HIDDEN, m)
    while c > okc_patch.HOPPER_NORM_MAX_CLUSTER:
        c //= 2
    return c


def fused_attn_boundary(boundary, stream: torch.Tensor, partial: torch.Tensor):
    """(next sublayer input bf16, new stream fp32) = boundary(stream, all_reduce(partial)), in one kernel."""
    from sglang.srt.distributed import get_tp_group

    boundary._ensure_cached_weights(torch.bfloat16)
    m = partial.shape[0]
    output = torch.empty_like(partial)
    new_stream = torch.empty_like(stream)
    gate = boundary._use_residual_gate
    _jit_module(HIDDEN, 8).run(
        get_tp_group().ca_comm.obj, partial.contiguous(), stream, output, new_stream,
        boundary._cc_post_weight if boundary._post_norm.has_weight else None,
        boundary._cc_pre_weight if boundary._pre_norm.has_weight else None,
        boundary._cc_alpha if gate else None, boundary._cc_beta if gate else None,
        float(boundary._post_norm.eps), float(boundary._pre_norm.eps),
        bool(boundary._post_norm_is_input_scale), _virtual_cluster(m))
    return output, new_stream
