# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Megatron-style LayerNorm sequence parallelism (SP, arXiv:2205.05198).

Under pure tensor parallelism the row-parallel ``all_reduce`` is algebraically a
``reduce_scatter`` (g-bar) followed by an ``all_gather`` (g). Splitting it that
way lets the LayerNorm / residual regions run on sequence-sharded activations --
each rank holds 1/tp of the tokens -- which cuts the transient activation memory
of long-context prefill with no extra communication volume (all_reduce and
reduce_scatter+all_gather move the same bytes).

Everything SP lives here so the feature stays decoupled from model code and from
``dp_attention.py``:

  - which models opt in (the Qwen3-dense allowlist) and config validation,
  - the per-forward ``sp_active`` flag (a ForwardFlags bool) read at depth by the
    participant linears and the layer-boundary batch selector,
  - the entry-scatter / exit-gather collectives, and
  - the fused matmul + collective fast-paths for the participant linears.

SP runs for prefill (EXTEND) only and is off by default; with the flag off,
nothing in this module executes.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.runtime_context import (
    get_flags,
    get_forward,
    get_parallel,
)
from sglang.srt.utils.common import ceil_align

logger = logging.getLogger(__name__)

# Architectures whose decoder layers route attention/MLP through
# layer boundaries with the standard participant linears, and for which SP
# has been validated. Other models reject --enable-layernorm-sp at construction.
# The mechanism is generic; extend the allowlist as families are validated.
MUSE_SP_ARCHITECTURES = frozenset(
    {"MuseSpark1xForCausalLM", "MuseSparkForConditionalGeneration"}
)
SP_SUPPORTED_ARCHITECTURES = MUSE_SP_ARCHITECTURES | frozenset({"Qwen3ForCausalLM"})


def is_muse_spark_architecture(architecture: Optional[str]) -> bool:
    return architecture in MUSE_SP_ARCHITECTURES


class _OkcHopperSP:
    """Muse SP prefill on Hopper without --enable-layernorm-sp: NCCL collectives instead of the
    Blackwell CUTLASS ones, and the MoE all-gathers its inputs instead of the A2A dispatcher."""

    active: bool = False
    symm: bool = False
    symm_ok: bool = False   # the startup probe passed (gates every symm-mem path: fused, ext_coll, copy-engine)
    symm_min_tokens: int = 0
    min_tokens: int = 0


_okc_hopper = _OkcHopperSP()


def _okc_hopper_sp_wanted(architecture: Optional[str]) -> bool:
    from sglang.kernels.ops.muse_spark_v12.okc_patch import SW_SP

    parallel = get_parallel()
    return (
        SW_SP
        and is_muse_spark_architecture(architecture)
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
        and parallel.tp_size > 1
        and parallel.attn_dp_size == 1
        and parallel.moe_tp_size == 1
        and not getattr(parallel, "enable_megamoe_for_layernorm_sp", False)
    )


def okc_hopper_sp() -> bool:
    return _okc_hopper.active


def okc_hopper_sp_symm() -> bool:
    return _okc_hopper.symm


def _okc_symm_local_ok() -> bool:
    """A tiny symmetric allocation on this rank alone (no collective), freed again."""
    import torch.distributed._symmetric_memory as symm_mem

    t = symm_mem.empty(64, dtype=torch.uint8, device=torch.device("cuda", torch.cuda.current_device()))
    del t
    return True


def _okc_symm_probe() -> bool:
    """Run both torch symm-mem fused ops once on a tiny problem across the TP group and check the result, then
    agree: the MIN of every rank's success flag over the CPU group, so all ranks take the same path. A node where
    symm-mem setup or IPC fails runs the NCCL participant linears instead of failing at the first long prefill.
    First, agreed before any collective: a local symmetric allocation, so a rank whose allocation fails alone does
    not leave its peers waiting inside the fused ops (the MIN after them cannot help a rank that never returns)."""
    import torch.distributed as dist

    from sglang.kernels.ops.muse_spark_v12.okc_patch.ext_coll import agreed

    tp_group = get_parallel().tp_group
    world = tp_group.world_size
    if agreed(_okc_symm_local_ok) is None:
        return False
    ok = 1
    try:
        group_name = tp_group.device_group.group_name
        device = torch.device("cuda", torch.cuda.current_device())
        # all ones, K = 256: every output is an exact small integer in bf16
        weight = torch.ones((128, 256), dtype=torch.bfloat16, device=device)
        shard = torch.ones((16, 256), dtype=torch.bfloat16, device=device)
        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            shard, [weight.t()], gather_dim=0, group_name=group_name
        )
        reduced = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            shard.repeat(world, 1), weight.t(), "sum", scatter_dim=0, group_name=group_name
        )
        torch.cuda.synchronize()
        gathered = mm_outputs[0]
        if not (
            tuple(gathered.shape) == (16 * world, 128)
            and bool((gathered == 256).all())
            and tuple(reduced.shape) == (16, 128)
            and bool((reduced == 256 * world).all())
        ):
            logger.warning("OKC SP: torch symm-mem fused ops returned wrong values in the startup probe.")
            ok = 0
    except Exception as e:  # any setup / IPC failure means NCCL
        logger.warning("OKC SP: torch symm-mem fused ops failed the startup probe: %r", e)
        ok = 0
    flag = torch.tensor([ok], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=tp_group.cpu_group)
    return bool(flag.item())


def initialize_layernorm_sp(*, model_config) -> None:
    """Materialize ``flags.sp.enabled``; runs once per worker after distributed
    setup, alongside ``initialize_dp_attention``."""
    architectures = model_config.hf_config.architectures
    architecture = architectures[0] if architectures else None
    _okc_hopper.active = (
        not get_parallel().enable_layernorm_sp and _okc_hopper_sp_wanted(architecture)
    )
    if _okc_hopper.active:
        from sglang.kernels.ops.muse_spark_v12.okc_patch import (
            SP_MIN_TOKENS,
            SP_SYMM,
            SP_SYMM_MIN_TOKENS,
            SW_EXT_AG,
            SW_EXT_RS,
            SW_M1,
            SW_M1RS,
            SP_AG_ROWSFIRST,
        )

        _okc_hopper.min_tokens = SP_MIN_TOKENS
        if get_parallel().tp_rank == 0:   # one per node: the okc extension builds start now, in parallel
            from sglang.kernels.ops.muse_spark_v12.okc_patch import cuda_ext

            cuda_ext.prebuild()

        # Every rank evaluates the same switches, so all of them enter the collective probe together. A failed probe
        # turns off every symm-mem path (the fused linears, ext_coll's multimem gather / reduce, the copy-engine ops).
        want = SP_SYMM or SW_EXT_AG or SW_EXT_RS or SW_M1 or SW_M1RS or SP_AG_ROWSFIRST
        _okc_hopper.symm_ok = want and _HAS_TORCH_SYMM_MEM_FUSED and _okc_symm_probe()
        _okc_hopper.symm = SP_SYMM and _okc_hopper.symm_ok
        if want and not _okc_hopper.symm_ok:
            logger.warning("OKC SP: symm-mem fused linears unavailable on this node; using NCCL collectives.")
        _okc_hopper.symm_min_tokens = SP_SYMM_MIN_TOKENS
        if SP_AG_ROWSFIRST and _okc_hopper.symm_ok:
            # rows-first's scores / ids buffer at boot: ext_coll's agreed setup (local allocation agreed before the
            # rendezvous); a failure on any rank leaves it None on all of them, and the MoE section then gathers as usual
            from sglang.kernels.ops.muse_spark_v12.okc_patch import ext_coll

            ext_coll._buffer(get_parallel().tp_group.device_group.group_name,
                             torch.device("cuda", torch.cuda.current_device()))
    get_flags().sp.enabled = bool(
        (get_parallel().enable_layernorm_sp or _okc_hopper.active)
        and architectures
        and architectures[0] in SP_SUPPORTED_ARCHITECTURES
    )


def layernorm_sp_enabled() -> bool:
    return get_flags().sp.enabled


def runs_sp(forward_mode, num_tokens: Optional[int] = None) -> bool:
    """Whether this forward runs SP: an enabled model, prefill only.

    Code outside the CUDA-graph-captured region must use this and not
    ``get_forward().sp_active``: Python writes made inside that region do not
    re-execute on graph replay, so the flag is stale there.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if not (layernorm_sp_enabled() and forward_mode == ForwardMode.EXTEND):
        return False
    # Every Muse call site passes the forward's token count; the OKC Hopper path keeps small extends unsharded.
    return not (
        _okc_hopper.active
        and num_tokens is not None
        and num_tokens <= _okc_hopper.min_tokens
    )


class _SPForwardState:
    """Real (unpadded) token count of the current SP forward.

    An instance attribute, not a ForwardFlags int slot: this is read inside
    torch.compile-traced linear code, where dynamo gives an attribute-source int
    automatic-dynamic, while a dict-slot int recompiles per sequence length.
    """

    num_tokens: int = 0


_sp_state = _SPForwardState()


def set_sp_num_tokens(num_tokens: int) -> None:
    _sp_state.num_tokens = num_tokens


def sp_num_tokens() -> int:
    return _sp_state.num_tokens


# --- entry scatter / exit gather (once per forward, at the boundary) ----------
def sp_entry_scatter(hidden_states: torch.Tensor) -> torch.Tensor:
    """Shard the replicated ``[M, h]`` hidden states along the token dim.

    Pads M up to a multiple of tp_size; the padding rows are dropped by the exit
    gather. The input is replicated across the TP group, so this is a local slice.
    """
    num_tokens = hidden_states.shape[0]
    set_sp_num_tokens(num_tokens)
    tp_group = get_parallel().tp_group
    tp_size = tp_group.world_size
    if tp_size == 1:
        return hidden_states
    padded = ceil_align(num_tokens, tp_size)
    if padded != num_tokens:
        hidden_states = torch.nn.functional.pad(
            hidden_states, (0, 0, 0, padded - num_tokens)
        )
    return hidden_states.tensor_split(tp_size)[tp_group.rank_in_group].contiguous()


_okc_entry_rs_logged: list = []


def okc_entry_reduce_scatter(embed, norm, input_ids: torch.Tensor, forward_mode) -> Optional[torch.Tensor]:
    """OKC_SP_EMB_RS: the SP entry as one reduce-scatter. The vocab-parallel embedding's masked local lookup [M, h]
    (zeros where another rank owns the token's row) is padded like ``sp_entry_scatter`` and reduce-scattered, then
    ``norm`` runs on this rank's [M_pad / tp, h] shard, instead of an all-reduce of [M, h] and the norm on every row.
    Bit-exact: each row's sum has one nonzero term, the norm is per row and padding rows stay zero. None when this
    forward does not run SP or the embedding would not all-reduce (the caller then takes the usual path)."""
    from sglang.srt.layers.layer_boundary import get_attn_tp_context
    from sglang.srt.utils.async_probe import maybe_detect_oob

    num_tokens = input_ids.shape[0]
    tp_group = get_parallel().tp_group
    tp_size = tp_group.world_size
    if (tp_size == 1 or embed.tp_size != tp_size or embed.use_attn_tp_group
            or get_attn_tp_context().input_scattered or not runs_sp(forward_mode, num_tokens=num_tokens)):
        return None
    set_sp_num_tokens(num_tokens)
    if not _okc_entry_rs_logged:
        _okc_entry_rs_logged.append(num_tokens)
        logger.info("OKC SP entry reduce-scatter active (first T=%d).", num_tokens)
    maybe_detect_oob(input_ids, 0, embed.num_embeddings, "VocabParallelEmbedding input id")
    partial = embed._embed_local_shard(input_ids)
    padded = ceil_align(num_tokens, tp_size)
    if padded != num_tokens:
        partial = torch.nn.functional.pad(partial, (0, 0, 0, padded - num_tokens))
    shard = partial.new_empty((padded // tp_size, partial.shape[1]))
    tp_group.reduce_scatter_tensor(shard, partial.contiguous())
    return shard if norm is None else norm(shard)


def sp_exit_gather(hidden_states: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """g: all-gather the per-rank shards back to the full sequence along dim 0,
    then narrow to ``num_tokens`` (dropping the entry-scatter padding)."""
    tp_group = get_parallel().tp_group
    tp_size = tp_group.world_size
    if tp_size == 1:
        return hidden_states[:num_tokens]
    hidden_states = hidden_states.contiguous()
    output = hidden_states.new_empty(
        (hidden_states.shape[0] * tp_size, *hidden_states.shape[1:])
    )
    tp_group.all_gather_into_tensor(output, hidden_states)
    return output[:num_tokens]


def maybe_exit_gather(
    *,
    hidden_states: torch.Tensor,
    hidden_states_before_norm: Optional[torch.Tensor],
    input_ids: Optional[torch.Tensor],
    forward_mode,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Undo the sequence sharding before the LM head, and leave the region.

    No-op unless this forward runs SP. The token count comes from ``input_ids``
    and the predicate from ``runs_sp``, so this stays correct on CUDA graph
    replay, where the writes made inside the captured region do not re-execute.
    """
    if input_ids is None or not runs_sp(forward_mode, input_ids.shape[0]):
        return hidden_states, hidden_states_before_norm
    num_tokens = input_ids.shape[0]
    hidden_states = sp_exit_gather(hidden_states, num_tokens=num_tokens)
    if hidden_states_before_norm is not None:
        hidden_states_before_norm = sp_exit_gather(
            hidden_states_before_norm, num_tokens=num_tokens
        )
    get_forward().set("sp_active", False)
    return hidden_states, hidden_states_before_norm


# --- fused matmul + collective fast-paths for the participant linears ---------
# Fused matmul+reduce-scatter (g-bar) and all-gather+matmul (g) overlap the
# collective with the GEMM. Availability is probed once at import; TP groups are
# registered for symmetric memory lazily by the fused ops on first use (the old
# enable_symm_mem_for_group is a deprecated no-op), so we only import the module
# to register the torch.ops.symm_mem namespace the probe checks. NVLink/NVSwitch.
try:
    import torch.distributed._symmetric_memory  # noqa: F401

    _HAS_TORCH_SYMM_MEM_FUSED = hasattr(
        torch.ops.symm_mem, "fused_matmul_reduce_scatter"
    ) and hasattr(torch.ops.symm_mem, "fused_all_gather_matmul")
except Exception:
    _HAS_TORCH_SYMM_MEM_FUSED = False


def _okc_ce_tp() -> bool:
    from sglang.kernels.ops.muse_spark_v12.okc_patch import SP_CE_TP

    return SP_CE_TP


def _okc_symm_selected(num_tokens: Optional[int]) -> bool:
    return (
        _okc_hopper.symm
        and num_tokens is not None
        and num_tokens >= _okc_hopper.symm_min_tokens
    )


def sp_fused_matmul_eligible(linear, num_tokens: Optional[int] = None) -> bool:
    """Whether the selected fused matmul+collective path applies.

    Muse Spark selects CUTLASS; other models keep the Torch symmetric-memory
    path. Both require an unquantized, bias-free bf16/fp16 linear. The decision
    depends only on static layer properties and is identical across TP ranks.
    """
    if (
        not (_HAS_TORCH_SYMM_MEM_FUSED or getattr(linear, "_sp_use_cutlass", False))
        or linear.bias is not None
        or (_okc_hopper.active and not _okc_symm_selected(num_tokens))
    ):
        return False
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if not isinstance(linear.quant_method, UnquantizedLinearMethod):
        return False
    return linear.weight.dtype in (torch.bfloat16, torch.float16)


def column_parallel_g_matmul(
    linear, input_parallel: torch.Tensor, bias
) -> torch.Tensor:
    """Megatron SP g for a ColumnParallelLinear participant (qkv / gate_up).

    The input is this rank's sequence shard ``[M_pad/tp, K]``; all-gather it back
    to the full sequence, matmul, and narrow to the real token count (recorded at
    the entry scatter). Eligible Muse Spark layers use one CUTLASS GEMM with
    tile-gated all-gather; other models retain their existing implementation.
    """
    num_tokens = sp_num_tokens()
    if sp_fused_matmul_eligible(linear, num_tokens):
        if getattr(linear, "_sp_use_cutlass", False):
            from sglang.kernels.ops.gemm.sp_collective_gemm import all_gather_matmul

            return all_gather_matmul(
                input_parallel.contiguous(),
                linear.weight,
                get_parallel().tp_group.device_group,
            )[:num_tokens]
        if _okc_hopper.active and _okc_ce_tp():
            from sglang.kernels.ops.muse_spark_v12.okc_patch import sp_tp_ce

            out = sp_tp_ce.ag_matmul(get_parallel().tp_group, input_parallel, linear.weight)
            if out is not None:
                return out[:num_tokens]
        group_name = get_parallel().tp_group.device_group.group_name
        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            input_parallel.contiguous(),
            [linear.weight.t()],
            gather_dim=0,
            group_name=group_name,
        )
        return mm_outputs[0][:num_tokens]
    if _okc_hopper.active:
        from sglang.kernels.ops.muse_spark_v12.okc_patch import ext_coll

        tp_group = get_parallel().tp_group
        full = ext_coll.all_gather(input_parallel, tp_group.device_group.group_name, tp_group.world_size)
        if full is not None:   # OKC_SW_EXT_AG: the extend's qkv-in gather on multimem (same bytes)
            return linear.quant_method.apply(linear, full[:num_tokens], bias)
    gathered = sp_exit_gather(input_parallel, num_tokens=num_tokens)
    return linear.quant_method.apply(linear, gathered, bias)


def row_parallel_gbar_matmul(linear, input_: torch.Tensor, bias) -> torch.Tensor:
    """Megatron SP g-bar for a RowParallelLinear participant (o_proj / down).

    Computes ``input_ @ weight.T`` and reduce-scatters (sum) the result across the
    TP group along dim 0, leaving this rank's ``[M_pad/tp, h]`` shard. The token
    dim is padded to a multiple of tp_size (padding rows are zeros, dropped by the
    exit gather). Eligible Muse Spark layers use one CUTLASS GEMM with a
    multicast reduce-scatter epilogue; other models retain their existing path.
    """
    tp_size = linear.tp_group.world_size if linear.tp_group is not None else 1
    x = input_.contiguous()
    num_tokens = x.shape[0]
    padded = ceil_align(num_tokens, tp_size)
    if padded != num_tokens:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded - num_tokens))
    if sp_fused_matmul_eligible(linear, num_tokens):
        if getattr(linear, "_sp_use_cutlass", False):
            from sglang.kernels.ops.gemm.sp_collective_gemm import matmul_reduce_scatter

            return matmul_reduce_scatter(
                x, linear.weight, get_parallel().tp_group.device_group
            )
        if _okc_hopper.active and _okc_ce_tp():
            from sglang.kernels.ops.muse_spark_v12.okc_patch import sp_tp_ce

            out = sp_tp_ce.matmul_rs(get_parallel().tp_group, x, linear.weight)
            if out is not None:
                return out
        group_name = get_parallel().tp_group.device_group.group_name
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x,
            linear.weight.t(),
            "sum",
            scatter_dim=0,
            group_name=group_name,
        )
    if _okc_hopper.active and bias is None and linear.weight.dtype is torch.bfloat16:
        from sglang.kernels.ops.muse_spark_v12.okc_patch import ext_coll
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        tp_group = get_parallel().tp_group
        gname = tp_group.device_group.group_name
        target = (
            ext_coll.rs_target(padded, linear.weight.shape[0], gname, tp_size)
            if isinstance(linear.quant_method, UnquantizedLinearMethod)
            else None
        )
        if target is not None:   # OKC_SW_EXT_RS: the GEMM straight into the symmetric buffer, then ld_reduce
            torch.mm(x, linear.weight.t(), out=target)
            return ext_coll.rs_finish(target, gname, tp_size, tp_group.rank_in_group)
    full = linear.quant_method.apply(linear, x, bias)
    output = full.new_empty((padded // tp_size, *full.shape[1:]))
    get_parallel().tp_group.reduce_scatter_tensor(output, full)
    return output
