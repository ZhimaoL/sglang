"""Torch-native Muse 1.2 / Muse Spark text model for SGLang.

BF16 checkpoints run a plain ``torch.nn.functional.linear`` expert loop.
Serialized ModelOpt NVFP4 expert weights stay packed and are executed by the
public FlashInfer TRT-LLM MoE kernel via the public ModelOpt quant method. Muse
applies its post-expert norm between GEMM2 and the top-k combine, which that
kernel fuses together, so the combine is finished separately by the CuTeDSL op
in ``sglang_meta_common/ops`` (with a torch fallback).
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.muse_spark_v12.all_gather_norm_residual import (
    all_gather_norm_residual_supported,
)
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import (
    FusedMuseSparkRouting,
)
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import (
    FusedPostNormResidualPreNorm,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.gemm_ag import (
    gemm_ag,
    gemm_ag_supported,
    gemm_local,
    large_m_capable,
    reserve_projection_workspace,
)
from sglang.kernels.ops.muse_spark_v12 import okc_patch

if okc_patch.ANY_ATTN:
    from sglang.kernels.ops.muse_spark_v12.okc_patch import attention as okc_attention
if okc_patch.SW_ROUTER:
    from sglang.kernels.ops.muse_spark_v12.okc_patch import router as okc_router
if okc_patch.QK_FOLD:
    from sglang.kernels.ops.muse_spark_v12.okc_patch import ft_norm as okc_ft_norm
_OKC_RSB = okc_patch.SW_RS_ATTN_BLOCKS or okc_patch.RS_ATTN_BLOCKS_FULL > 0
if okc_patch.SW_RS_NORM_FOLD or _OKC_RSB:
    from sglang.kernels.ops.muse_spark_v12.okc_patch import sp_tp_ce as okc_sp_tp_ce
if _OKC_RSB:
    from sglang.kernels.ops.muse_spark_v12.okc_patch import prefill_fmha as okc_prefill_fmha
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        eager_on_graph,
        is_in_breakable_cuda_graph,
    )
from sglang.kernels.ops.muse_spark_v12.router_gemm import (
    MAX_TOKENS as ROUTER_GEMM_MAX_TOKENS,
)
from sglang.srt.configs.muse_spark import (
    MuseSpark1xTextConfig,
    should_skip_weight_loading,
)
from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_reduce
from sglang.srt.environ import envs
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers import muse_spark_expert_placement
from sglang.srt.layers import muse_spark_expert_tp as expert_tp
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.layers.dp_attention import (
    dp_gather_replicate,
    dp_scatter,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor as OSSLogitsProcessor
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.muse_spark_bf16_moe import MuseSpark1xBf16Experts
from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
    QKVOGateParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer
from sglang.srt.utils import add_prefix, is_cpu, is_hip, use_intel_amx_backend
from sglang.srt.utils.common import get_device_capability, is_cuda

_is_cpu = is_cpu()

logger = logging.getLogger(__name__)

# FusedMoE shard ids that expert weight loaders expect per checkpoint projection.
_EXPERT_SHARD_IDS = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


class MuseSpark1xLogitsProcessor(OSSLogitsProcessor):
    """Public logits pipeline with the checkpoint-requested accumulator dtype."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.use_fp32_output = True
        # The verify lm head megakernel gathers capped FP32 logits over TP4.
        self.use_helion_verify_lm_head = (
            expert_tp.helion_verify_enabled()
            and bool(self.final_logit_softcapping)
            and self.do_tensor_parallel_all_gather
            and not self.use_attn_tp_group
            and not self.use_tp_lm_head_all_to_all
            and not self.do_tensor_parallel_all_gather_dp_attn
            and get_parallel().tp_size == 4
        )

    def _get_logits(
        self,
        hidden_states,
        lm_head,
        logits_metadata,
        embedding_bias=None,
        use_logits_buffer=True,
    ):
        if not (
            self.use_helion_verify_lm_head
            and hidden_states.shape[0] in expert_tp.VERIFY_TOKENS
            and logits_metadata.forward_mode.is_target_verify()
            and embedding_bias is None
            and hasattr(lm_head, "weight")
            and lm_head.weight.shape[0] * get_parallel().tp_size == self.vocab_size
        ):
            return super()._get_logits(
                hidden_states,
                lm_head,
                logits_metadata,
                embedding_bias=embedding_bias,
                use_logits_buffer=use_logits_buffer,
            )
        return self._get_logits_helion_verify(
            hidden_states, lm_head, logits_metadata, use_logits_buffer
        )

    def _get_logits_helion_verify(
        self, hidden_states, lm_head, logits_metadata, use_logits_buffer
    ):
        """One megakernel for the lm head, the logit scale and softcap and the
        gather of the vocab shards, written into the logits buffer when it fits."""
        from sglang.kernels.ops.muse_spark_v12.helion_megakernels.integration import (
            require_helion_features,
        )
        from sglang.kernels.ops.muse_spark_v12.helion_megakernels.lm_head_verify_megakernel import (
            LogitScalars,
            get_lm_head_verify_megakernel,
        )

        require_helion_features()

        weight = lm_head.weight
        tokens = hidden_states.shape[0]
        lm_head_mk = get_lm_head_verify_megakernel(
            device=hidden_states.device,
            group=get_tp_group().device_group,
            scalars=LogitScalars(
                scale=1.0 if self.logit_scale is None else float(self.logit_scale),
                cap=float(self.final_logit_softcapping),
            ),
            tokens=tokens,
            vocab_shard=weight.shape[0],
        )
        out = logits_metadata.next_token_logits_buffer if use_logits_buffer else None
        if (
            out is None
            or tuple(out.shape) != (tokens, self.vocab_size)
            or out.dtype != torch.float32
            or not out.is_contiguous()
        ):
            out = hidden_states.new_empty(tokens, self.vocab_size, dtype=torch.float32)
        return lm_head_mk(
            hidden_states=hidden_states.to(weight.dtype), weight=weight, out=out
        )

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        if not hasattr(lm_head, "weight"):
            return super()._compute_lm_head(hidden_states, lm_head, embedding_bias)
        output_dtype = torch.float32 if self.use_fp32_output else lm_head.weight.dtype
        if use_intel_amx_backend(lm_head):
            # ParallelLMHead swaps in PackWeightMethod on CPU+AMX, which repacks
            # .weight into a VNNI blocked layout *without changing its shape*, so
            # a plain mm silently returns garbage instead of failing.
            logits = torch.ops.sgl_kernel.weight_packed_linear(
                hidden_states.to(lm_head.weight.dtype),
                lm_head.weight,
                None,
                True,
            ).to(output_dtype)
        else:
            logits = torch.mm(
                hidden_states.to(lm_head.weight.dtype),
                lm_head.weight.T,
                out_dtype=output_dtype,
            )
        if embedding_bias is not None:
            logits = logits + embedding_bias
        return logits


def _quantization_name(quant_config: Optional[QuantizationConfig]) -> Optional[str]:
    if quant_config is None:
        return None
    get_name = getattr(quant_config, "get_name", None)
    return get_name() if get_name is not None else type(quant_config).__name__


def _uses_nvfp4_experts(
    quant_config: Optional[QuantizationConfig], layer_prefix: str
) -> bool:
    if _quantization_name(quant_config) != "modelopt_fp4":
        return False
    excluded = getattr(quant_config, "exclude_modules", None) or []
    return not is_layer_skipped(add_prefix("mlp.experts", layer_prefix), excluded)


def _uses_gptq_int4_experts(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    """Whether the CPU quant config selects the GPTQ-packed expert path."""
    if not _is_cpu or quant_config is None:
        return False
    quant_name = _quantization_name(quant_config)
    if quant_name == "gptq":
        return True
    if quant_name != "auto-round":
        return False
    return "gptq" in getattr(quant_config, "packing_format", "") or "gptq" in getattr(
        quant_config, "backend", ""
    )


@functools.lru_cache(maxsize=1)
def _split_bf16_router_supported() -> bool:
    return is_cuda() and get_device_capability()[0] == 9


_okc_sp_logged = False
_okc_p1ss_collect = False   # P1-SS: _forward_gathered_experts returns its _OkcGathered instead of running it
# P1-SS + W1b: layer -> a callable run once, right after the next qkv GEMM of that layer (the other child's A stage), so
# b(j)'s gated rows pushes are enqueued after a(j+1)'s qkv record (STRUCTURAL.md §3.1a push-form amendment 3). For the
# last layer there is no next qkv: the registrant must push ungated there, or M_b waits on flags nobody sets.
_OKC_POST_QKV = {}


def _use_split_bf16_router(num_tokens: int) -> bool:
    if num_tokens <= ROUTER_GEMM_MAX_TOKENS or not _split_bf16_router_supported():
        return False
    return True


class MuseSpark1xRouter(nn.Module):
    """Router stored in HF linear layout: ``[experts, hidden]``."""

    def __init__(self, hidden_size: int, num_experts: int, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=dtype), requires_grad=False
        )

        self.register_buffer(
            "e_score_correction_bias", torch.zeros(num_experts, dtype=torch.float32)
        )
        self._split2_hi = None
        self._split2_lo = None
        self._split2_key = None

    @torch.no_grad()
    def _build_split2(self) -> None:
        w = self.weight.data
        key = (w.data_ptr(), w.shape, w._version)
        if self._split2_key == key:
            return
        hi = w.to(torch.bfloat16)
        lo = (w - hi.to(torch.float32)).to(torch.bfloat16)
        self._split2_hi = hi.t().contiguous()
        self._split2_lo = lo.t().contiguous()
        self._split2_key = key

    def forward_split2(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self._build_split2()
        x = hidden_states.to(torch.bfloat16)
        acc = torch.mm(x, self._split2_hi, out_dtype=torch.float32)
        acc += torch.mm(x, self._split2_lo, out_dtype=torch.float32)
        return acc

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)


_OKC_ATTN = {}


def _okc_attn_rs(q, k, v, o_gate, attn_out, layer_id, rows, fold, save_kv_cache, fb=None):
    """OKC rsfold O1, SP layers: attention (per-destination row blocks), output gate and o-proj RS in one call,
    each GEMM + push as soon as its rows exist. One eager break under the breakable prefill graph (fb None: the live
    batch is the forward context's). Returns the RS output, or None when the RS norm fold reads the slots. Calls that
    are not exactly the blocks (padded, ragged) run whole."""
    m = _OKC_ATTN[layer_id]
    a = m.attn
    layernorm_sp.okc_boundary_mark(layer_id)
    with okc_prefill_fmha.defer_row_blocks(rows, attn_out.shape[0] // rows):
        if fb is None:
            from sglang.srt.layers.radix_attention import unified_attention_with_output

            unified_attention_with_output(q, k.view(-1, a.tp_k_head_num, a.qk_head_dim),
                                          v.view(-1, a.tp_v_head_num, a.v_head_dim), attn_out, save_kv_cache, layer_id)
        else:
            attn_out = a(q, k, v, fb, save_kv_cache=save_kv_cache)
    run = okc_prefill_fmha.take_deferred(attn_out)
    x = m._okc_blocks(attn_out, o_gate, run, rows) if run is not None else m._fused_o_norm_gate(attn_out, o_gate)
    with okc_sp_tp_ce.defer_slot_sum(fold):
        out = okc_sp_tp_ce.matmul_rs(get_parallel().tp_group, x, m.o_proj.weight)
    if out is None:   # matmul_rs declined (gates pre-checked): the stock route
        if not isinstance(x, torch.Tensor):
            x = torch.cat([x.block(j) for j in range(x.shape[0] // rows)])
        return m.o_proj(x)[0]
    slots = okc_sp_tp_ce.take_slots(out)
    if slots is None:
        return out
    okc_sp_tp_ce.defer_slots(None, slots)   # the caller binds its placeholder output to them
    return None


if _OKC_RSB:
    _okc_attn_rs_break = eager_on_graph(True)(_okc_attn_rs)


class MuseSpark1xAttention(nn.Module):
    def __init__(self, config: MuseSpark1xTextConfig, layer_idx: int, prefix: str):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.use_o_gate = True
        self.use_o_norm = True
        self.use_qk_norm = True
        self.use_rope = config.layer_rope_theta[layer_idx] > 0
        sliding_window = (
            config.sliding_window - 1
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

        parallel = get_parallel()
        tp_size = parallel.attn_tp_size
        if config.num_attention_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by attention TP")
        if config.num_key_value_heads >= tp_size:
            if config.num_key_value_heads % tp_size:
                raise ValueError(
                    "num_key_value_heads must be divisible by attention TP"
                )
        elif tp_size % config.num_key_value_heads:
            raise ValueError("attention TP must divide replicated KV heads")
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.scale_query_by = config.qk_scale_factor
        if self.use_qk_norm:
            # RoPE is a rotation, so this post-QK-norm scalar commutes into the
            # softmax scale instead of being multiplied onto q.
            self.scaling *= self.scale_query_by

        if self.use_o_gate:
            self.qkv_proj = QKVOGateParallelLinear(
                hidden_size=config.hidden_size,
                head_size=self.head_dim,
                total_num_heads=config.num_attention_heads,
                total_num_kv_heads=config.num_key_value_heads,
                bias=False,
                prefix=add_prefix("qkv_proj", prefix),
                parallel_group="attn_tp",
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=config.hidden_size,
                head_size=self.head_dim,
                total_num_heads=config.num_attention_heads,
                total_num_kv_heads=config.num_key_value_heads,
                bias=False,
                quant_config=None,
                prefix=add_prefix("qkv_proj", prefix),
                parallel_group="attn_tp",
            )
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=None,
            prefix=add_prefix("o_proj", prefix),
            parallel_group="attn_tp",
            use_dp_attention_reduce=True,
        )
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=getattr(
                    config, "context_len", config.max_position_embeddings
                ),
                base=config.layer_rope_theta[layer_idx],
                is_neox_style=False,
                dtype=torch.float32,
            )
            if self.use_rope
            else None
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            prefix=add_prefix("attn", prefix),
            use_irope=self.use_rope,
            sliding_window_size=sliding_window or -1,
        )
        # The FA3 backend's extend call reads this mark (okc_patch/prefill_fmha.py): target layers only.
        self.attn.okc_prefill_fmha = okc_patch.SW_PREFILL_FMHA
        self.norm_eps = config.rms_norm_eps
        # The output norm and the sigmoid gate are one fused kernel. The norm is
        # weightless here, so this module carries no parameters -- it exists to
        # hand the op its eps.
        self._o_norm_module = (
            MuseSpark1xRMSNorm(self.head_dim, eps=self.norm_eps, has_weight=False)
            if self.use_o_norm
            else None
        )
        self._fused_o_norm_gate = FusedRMSNormGate(
            self._o_norm_module, self.use_o_gate, self.head_dim
        )
        self._fused_qk_norm = (
            FusedQKNorm(
                head_dim=self.head_dim,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                eps=self.norm_eps,
            )
            if self.use_qk_norm
            else None
        )
        if okc_patch.QK_FOLD:
            okc_ft_norm.mark(self)

    def _fused_kv_store(
        self, forward_batch: ForwardBatch
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        backend = get_attn_backend()
        pool = get_token_to_kv_pool()
        if not (
            isinstance(backend, TRTLLMHAAttnBackend)
            and backend.data_type in (torch.bfloat16, torch.float8_e4m3fn)
            and pool.dtype == backend.data_type
            and (pool.dtype == torch.bfloat16 or not backend.is_xqa_impl)
            and self.attn.k_scale is None
            and self.attn.v_scale is None
            and forward_batch.dcp_kv_mask is None
            # Prefill excluded: the fused store segfaults at breakable prefill replay.
            and (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_target_verify()
            )
            and not is_cp_active(forward_batch)
        ):
            return None
        cache_loc = backend._get_layer_cache_loc(self.attn, forward_batch)
        k_cache = pool.get_key_buffer(self.layer_idx)
        v_cache = pool.get_value_buffer(self.layer_idx)
        return (
            k_cache.view(k_cache.shape[0], -1),
            v_cache.view(v_cache.shape[0], -1),
            cache_loc,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv, _ = self.qkv_proj(hidden_states)
        if _OKC_POST_QKV:
            hook = _OKC_POST_QKV.pop(self.layer_idx, None)
            if hook is not None:
                hook()
        rows = _OKC_RSB and self._okc_block_rows(qkv.shape[0])
        if rows:   # OKC rsfold O1: attention through the o-proj RS (okc_attn_rs)
            return self.gated_attention(positions, qkv, forward_batch, okc_rows=rows)
        output, _ = self.o_proj(self.gated_attention(positions, qkv, forward_batch))
        return output

    def _okc_block_rows(self, num_tokens: int) -> int:
        """Rows per destination when o_proj takes sp_tp_ce.matmul_rs on a sliding layer (row_parallel_gbar_matmul's
        gates, shape-only), eager or in a breakable-graph capture; else 0."""
        from sglang.srt.layers import layernorm_sp

        tp = self.o_proj.tp_group.world_size if self.o_proj.tp_group is not None else 1
        if (not self._okc_group() or tp <= 1 or num_tokens % tp or not get_forward().sp_active
                or num_tokens // tp < okc_patch.RS_ATTN_BLOCKS_MIN_ROWS or tp != get_parallel().tp_group.world_size
                or (torch.cuda.is_current_stream_capturing() and not is_in_breakable_cuda_graph())
                or getattr(self.o_proj, "_sp_use_cutlass", False) or not layernorm_sp.okc_hopper_sp()
                or not okc_patch.SP_CE_TP or not layernorm_sp.sp_fused_matmul_eligible(self.o_proj, num_tokens)):
            return 0
        return num_tokens // tp

    def _okc_group(self):   # destinations per fmha launch: sliding layers (switch on) / full layers (0 = off)
        sliding = self.attn.sliding_window_size > 0
        return okc_patch.RS_ATTN_BLOCKS_GROUP * okc_patch.SW_RS_ATTN_BLOCKS if sliding else okc_patch.RS_ATTN_BLOCKS_FULL

    def _okc_blocks(self, output, o_gate, run, R):
        # block j: the fmha + gate of j and the next unmade blocks (<= GROUP; matmul_rs asks me+1..W-1, 0..me)
        nb, G = output.shape[0] // R, max(1, self._okc_group())
        made, ready = set(), {}

        def block(j):
            if j not in made:
                n = 1
                while n < G and j + n < nb and j + n not in made:
                    n += 1
                run(j, n)
                r = slice(j * R, (j + n) * R)
                y = self._fused_o_norm_gate(output[r], o_gate[r] if o_gate is not None else None)
                for i in range(n):
                    made.add(j + i)
                    ready[j + i] = y[i * R:(i + 1) * R]
            return ready.pop(j)

        return okc_sp_tp_ce.RowBlocks(nb, R, output.numel() // output.shape[0], output.dtype, output.device, block)

    def _okc_attn_rs_call(self, q, k, v, o_gate, forward_batch, save_kv_cache, rows):
        _OKC_ATTN[self.attn.layer_id] = self
        a = self.attn
        attn_out = q.new_empty((q.shape[0], a.tp_q_head_num * a.v_head_dim), dtype=torch.bfloat16)
        args = (q, k, v, o_gate, attn_out, a.layer_id, rows, okc_sp_tp_ce._defer, save_kv_cache)
        capturing = torch.cuda.is_current_stream_capturing()
        out = _okc_attn_rs_break(*args) if capturing else _okc_attn_rs(*args, forward_batch)
        if out is None:   # the RS norm fold reads the slots: a placeholder stands for them
            out = q.new_empty((q.shape[0] // get_parallel().tp_group.world_size, self.o_proj.weight.shape[0]))
            okc_sp_tp_ce.defer_slots(out)
        return out

    def gated_attention(
        self,
        positions: torch.Tensor,
        qkv: torch.Tensor,
        forward_batch: ForwardBatch,
        okc_rows: int = 0,
    ) -> torch.Tensor:
        """Everything between the QKV and output projections: QK norm (and RoPE)
        with the KV store, attention, then the output norm and gate (OKC okc_rows: and o_proj)."""
        if okc_patch.ANY_ATTN:
            okc_out = okc_attention.try_attention(self, qkv, forward_batch)
            if okc_out is not None:
                return okc_out
        split_sizes = [self.q_size, self.kv_size, self.kv_size]
        if self.use_o_gate:
            q, k, v, o_gate = qkv.split(split_sizes + [self.q_size], dim=-1)
        else:
            q, k, v = qkv.split(split_sizes, dim=-1)
            o_gate = None
        fuse_rope = self._fused_qk_norm is not None and self.rotary_emb is not None
        kv_store = (
            self._fused_kv_store(forward_batch)
            if self._fused_qk_norm is not None
            and (self.rotary_emb is None or fuse_rope)
            else None
        )
        if okc_patch.QK_FOLD and okc_ft_norm.applies(self.attn, forward_batch) is not None:
            # OKC_QK_FOLD: q / k go to the FA3 backend raw; it folds the norm into the FP8 KV write or applies it
            assert kv_store is None
        elif fuse_rope:
            q, k = self._fused_qk_norm(
                qkv, positions, self.rotary_emb.cos_sin_cache, kv_store=kv_store
            )
        elif self._fused_qk_norm is not None:
            q, k = self._fused_qk_norm(qkv, kv_store=kv_store)
        if self.rotary_emb is not None and not fuse_rope:
            q, k = self.rotary_emb(positions, q, k)
        if okc_rows:
            return self._okc_attn_rs_call(q, k, v, o_gate, forward_batch, kv_store is None, okc_rows)
        output = self.attn(q, k, v, forward_batch, save_kv_cache=kv_store is None)
        return self._fused_o_norm_gate(output, o_gate)


class MuseSpark1xMLP(nn.Module):
    def __init__(self, config: MuseSpark1xTextConfig, prefix: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class MuseSpark1xMoE(nn.Module):
    def __init__(
        self,
        config: MuseSpark1xTextConfig,
        layer_idx: int,
        nvfp4_checkpoint: bool,
        prefix: str,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        if config.num_shared_experts:
            raise ValueError("Muse Spark OSS supports routed experts only")
        parallel = get_parallel()
        int4_checkpoint = _uses_gptq_int4_experts(quant_config)
        expert_tp.validate(int4_checkpoint)
        if config.num_local_experts % parallel.moe_ep_size:
            raise ValueError("num_experts must be divisible by expert parallel size")
        self.layer_idx = layer_idx
        self.tp_size = parallel.tp_size
        self.ep_size = parallel.moe_ep_size
        self.ep_rank = parallel.moe_ep_rank
        self.moe_tp_size = parallel.moe_tp_size
        # Reading the rank resolves the MoE-TP process group, which need not
        # exist when expert TP is off.
        self.moe_tp_rank = parallel.moe_tp_rank if self.moe_tp_size != 1 else 0
        self.num_experts = config.num_local_experts
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.route_eps = 1e-15
        self.router_multiplier = config.output_multiplier
        self.router_in_fp32 = True
        self.hidden_size = config.hidden_size
        self.expert_hidden_size = config.moe_hidden_size
        self.intermediate_size = config.moe_intermediate_size
        router_dtype = (
            torch.float32 if self.router_in_fp32 else torch.get_default_dtype()
        )
        self.gate = MuseSpark1xRouter(
            self.hidden_size, self.num_experts, dtype=router_dtype
        )

        if self.expert_hidden_size != self.hidden_size:
            if _is_cpu:
                self.pre_expert_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.expert_hidden_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("pre_expert_proj", prefix),
                )
                self.post_expert_proj = ReplicatedLinear(
                    self.expert_hidden_size,
                    self.hidden_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("post_expert_proj", prefix),
                )
            else:
                self.pre_expert_proj = nn.Linear(
                    self.hidden_size, self.expert_hidden_size, bias=False
                )
                self.post_expert_proj = nn.Linear(
                    self.expert_hidden_size, self.hidden_size, bias=False
                )
                for linear in (self.pre_expert_proj, self.post_expert_proj):
                    # The CuTe-DSL GEMM exports weights via DLPack, which rejects requires_grad.
                    linear.weight.requires_grad_(False)
        else:
            self.pre_expert_proj = None
            self.post_expert_proj = None
        if (
            self.pre_expert_proj is not None
            and self.tp_size in (4, 8)
            and torch.cuda.is_available()
            and large_m_capable()
        ):
            ca_comm = get_tp_group().ca_comm
            reserve_projection_workspace(
                ca_comm, max(self.expert_hidden_size, self.hidden_size)
            )
        self.pre_expert_norm = MuseSpark1xRMSNorm(
            self.expert_hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            residual_id=2 * layer_idx + 2,
        )
        norm_cls = MuseSpark1xRMSNormWithInputScale
        post_expert_norm = norm_cls(
            self.expert_hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
        )
        self.nvfp4_checkpoint = nvfp4_checkpoint
        self.use_megamoe = getattr(parallel, "enable_megamoe_for_layernorm_sp", False)
        if int4_checkpoint:
            from sglang.srt.layers.muse_spark_int4_moe import MuseSpark1xInt4Experts

            self.experts = MuseSpark1xInt4Experts(
                num_experts=self.num_experts,
                expert_hidden_size=self.expert_hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.top_k,
                layer_id=layer_idx,
                quant_config=quant_config,
                prefix=add_prefix("experts", prefix),
            )
        elif nvfp4_checkpoint:
            # Packed NVFP4: the FlashInfer TRT-LLM kernel runs the experts and we
            # finalize in torch so the post-expert norm lands before the top-k sum.
            # Hot-expert replicas only on the pg-only path (the per-forward balancer lives in okc_patch.moe).
            replica_plan = None
            if okc_patch.SW_MOE_PG and okc_patch.MOE_PG_ONLY and self.moe_tp_size == 1:
                replica_plan = muse_spark_expert_placement.replica_plan(
                    layer_idx=layer_idx,
                    num_experts=self.num_experts,
                    ep_size=self.ep_size,
                    ep_rank=self.ep_rank,
                    count=okc_patch.MOE_REPLICAS,
                )
            self.experts = MuseSpark1xFp4Experts(
                num_experts=self.num_experts,
                num_local_experts=self.num_local_experts,
                local_expert_offset=self.local_expert_start,
                hidden_size=self.expert_hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.top_k,
                quant_config=quant_config,
                local_expert_ids=muse_spark_expert_placement.local_expert_ids(
                    layer_idx=layer_idx,
                    num_experts=self.num_experts,
                    ep_size=self.ep_size,
                    ep_rank=self.ep_rank,
                ),
                replica_plan=replica_plan,
            )
            self.experts._okc_layer_idx = layer_idx
            if okc_patch.ADAPT_REPLICAS and self.experts.replica_plan is not None:
                from sglang.kernels.ops.muse_spark_v12.okc_patch import adaptive as okc_adaptive

                okc_adaptive.attach_buffers(self.experts)
        else:
            self.experts = MuseSpark1xBf16Experts(
                num_experts=self.num_experts,
                num_local_experts=self.num_local_experts,
                local_expert_offset=self.local_expert_start,
                expert_hidden_size=self.expert_hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.top_k,
            )
        self.experts.post_expert_norm = post_expert_norm
        self.experts.use_megamoe = self.use_megamoe
        self.sp_dispatcher = None
        self.finalize_ar_workspace = None
        self.unpool_gather_boundary: Optional[FusedPostNormResidualPreNorm] = None
        self.side_stream = None

    def set_sp_dispatcher(self, dispatcher) -> None:
        """Bind the model-scoped FlashInfer A2A workspace to this MoE layer."""
        self.sp_dispatcher = dispatcher

    def set_finalize_ar_workspace(self, workspace) -> None:
        """Bind the model-scoped decode finalize/all-reduce workspace."""
        self.finalize_ar_workspace = workspace

    def set_unpool_gather_boundary(
        self, boundary: FusedPostNormResidualPreNorm
    ) -> None:
        """Bind the FFN boundary that can absorb the unpooler all-gather."""
        self.unpool_gather_boundary = boundary

    def _side_stream(self) -> torch.cuda.Stream:
        if self.side_stream is None:
            self.side_stream = torch.cuda.Stream()
        return self.side_stream

    def _routing_op(self) -> FusedMuseSparkRouting:
        """The fused router: sigmoid, biased top-k, gather, renormalise, scale.

        Built lazily and rebound if ``expert_bias`` is replaced, because
        ``Module._apply`` (``.to()``, ``.cuda()``) swaps the buffer for a new
        tensor and the op holds a direct reference to it.
        """
        op = getattr(self, "_fused_routing_op", None)
        if op is None or op.expert_bias is not self.gate.e_score_correction_bias:
            op = FusedMuseSparkRouting(
                expert_bias=self.gate.e_score_correction_bias,
                route_scale=self.route_scale,
                router_output_multiplier=self.router_multiplier,
                router_score_division_eps=self.route_eps,
            )
            self._fused_routing_op = op
        return op

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_input: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        if (
            forward_batch is not None
            and is_dp_attention_enabled()
            and get_parallel().attn_dp_size > 1
        ):
            return self._forward_dp_attention(hidden_states, forward_batch)
        return self._forward_tp(hidden_states, router_input, forward_batch)

    def _forward_tp(
        self,
        hidden_states: torch.Tensor,
        router_input: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        ar_workspace = self.finalize_ar_workspace
        # All TP ranks must select the same reduction owner.
        if ar_workspace is not None and not (
            forward_batch is not None
            and (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_target_verify()
            )
            and self.moe_tp_size == 1
            and not get_forward().sp_active
            and hidden_states.dtype == torch.bfloat16
            and 0 < hidden_states.shape[0] <= ar_workspace.max_tokens
            and ar_workspace.can_fuse(get_tp_group(), hidden_states.shape[0])
        ):
            if (
                forward_batch is not None
                and (
                    forward_batch.forward_mode.is_decode()
                    or forward_batch.forward_mode.is_target_verify()
                )
                and not ar_workspace.fallback_logged
            ):
                logger.warning(
                    "Finalize AR fusion unavailable for this decode/verify batch "
                    "(SP=%s, dtype=%s, tokens=%s, capacity=%s); "
                    "requires non-SP BF16 decode/verify within capacity and custom "
                    "AR with a compatible push workspace. Using separate finalize and AR.",
                    get_forward().sp_active,
                    hidden_states.dtype,
                    hidden_states.shape[0],
                    ar_workspace.max_tokens,
                )
                ar_workspace.fallback_logged = True
            ar_workspace = None
        output = self._forward_experts(
            hidden_states,
            router_input,
            ar_workspace=ar_workspace,
        )
        m1 = getattr(self.experts, "_okc_m1_reduced", False)   # okc_patch.m1: the receiver already combined the ranks
        if m1:
            self.experts._okc_m1_reduced = False
        return self._reduce_and_unpool(output, already_reduced=ar_workspace is not None or m1)

    def _reduce_and_unpool(
        self, output: torch.Tensor, *, already_reduced: bool = False
    ) -> torch.Tensor:
        # Expert TP already reduced across ranks, before its post-expert norm.
        if (
            self.tp_size > 1
            and self.moe_tp_size == 1
            and not getattr(get_forward(), "sp_active", False)
            and not already_reduced
        ):
            output = tensor_model_parallel_all_reduce(output)
        if self.post_expert_proj is not None:
            if _is_cpu:
                output, _ = self.post_expert_proj(output)
                return output
            comm = get_tp_group().ca_comm if self.tp_size > 1 else None
            if not getattr(get_forward(), "sp_active", False) and gemm_ag_supported(
                comm, output, self.post_expert_proj.weight, okc_patch.GEMM_AG_UNPOOLER_MAX_TOKENS
            ):
                # Returns this rank's [M, H / world] slice; the FFN boundary gathers it.
                if self._defers_unpool_gather(comm, output.shape[0]):
                    return gemm_local(comm, output, self.post_expert_proj.weight)
                return gemm_ag(comm, output, self.post_expert_proj.weight)
            output = bf16_gemm_dispatch(output, self.post_expert_proj.weight, None)
        return output

    def _defers_unpool_gather(self, comm, num_tokens: int) -> bool:
        boundary = self.unpool_gather_boundary
        return (
            boundary is not None
            and boundary.jit_selected()
            and all_gather_norm_residual_supported(
                comm=comm, num_tokens=num_tokens, hidden_size=self.hidden_size
            )
        )

    def _forward_dp_attention(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        # Every rank runs the experts on the same DP-replicated global batch,
        # so the full-TP all-reduce below stays correct.
        tp_group = get_tp_group()
        global_hidden = get_global_dp_buffer(tp_group)
        dp_gather_replicate(global_hidden, hidden_states, forward_batch)
        output = self._forward_experts(global_hidden)
        if self.tp_size > 1 and self.moe_tp_size == 1:
            output = tensor_model_parallel_all_reduce(output)
        if self.post_expert_proj is not None:
            if _is_cpu:
                output, _ = self.post_expert_proj(output)
            else:
                output = self.post_expert_proj(output)
        local_output = get_local_dp_buffer(tp_group)
        dp_scatter(local_output, output, forward_batch)
        return local_output

    def _forward_experts(
        self,
        hidden_states: torch.Tensor,
        router_input: Optional[torch.Tensor] = None,
        *,
        ar_workspace=None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states.new_zeros((0, self.expert_hidden_size))
        use_split2 = (
            self.router_in_fp32
            and _use_split_bf16_router(hidden_states.shape[0])
            and self.gate.weight.dtype == torch.float32
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
        )
        # With the OKC router on, the fp32 backfill waits until its kernel declines.
        if not use_split2 and router_input is None and not okc_patch.SW_ROUTER:
            router_input = (
                hidden_states.float() if self.router_in_fp32 else hidden_states
            )
        sp_active = getattr(get_forward(), "sp_active", False)
        prepare_early = (
            self.nvfp4_checkpoint
            # Measured to help up to 128 tokens (decode c<=128); untested above.
            and hidden_states.shape[0] <= 128
            and not sp_active
            and not getattr(self.experts, "use_marlin", False)
        )
        prepared_input = None
        use_side_stream = not _is_cpu and torch.cuda.is_available()
        main_stream = torch.cuda.current_stream() if use_side_stream else None
        side_stream = self._side_stream() if use_side_stream else None
        if use_side_stream:
            side_stream.wait_stream(main_stream)
        # nn.Linear bypasses the bf16 GEMM dispatcher.
        expert_prenormed = False
        if self.pre_expert_proj is None:
            expert_input = hidden_states
        elif _is_cpu:
            expert_input, _ = self.pre_expert_proj(hidden_states)
        else:
            comm = get_tp_group().ca_comm if self.tp_size > 1 else None
            if (
                self.tp_size in (4, 8)
                and not sp_active
                and large_m_capable(hidden_states.device, hidden_states.shape[0], okc_patch.GEMM_AG_POOLER_MAX_TOKENS)
                and gemm_ag_supported(comm, hidden_states, self.pre_expert_proj.weight, okc_patch.GEMM_AG_POOLER_MAX_TOKENS)
            ):
                pn = self._okc_ag_prenorm(hidden_states.shape[0])
                expert_input = gemm_ag(comm, hidden_states, self.pre_expert_proj.weight,
                                       norm_weight=pn[0] if pn else None, norm_eps=pn[1] if pn else 0.0)
                expert_prenormed = pn is not None
            else:
                expert_input = bf16_gemm_dispatch(
                    hidden_states, self.pre_expert_proj.weight, None
                )
        if prepare_early:
            prepared_input = self.experts.prepare_input(
                expert_input, self.pre_expert_norm
            )
        with (
            torch.cuda.stream(side_stream)
            if use_side_stream
            else contextlib.nullcontext()
        ):
            router_logits = routed_small = None
            # okc router.py covers T <= 256 (its T 9..256 path beats forward_split2 ~2x); larger T declines to it.
            # route_small: reduce + routing in one launch (SP keeps the logits for its dispatcher)
            if okc_patch.SW_ROUTER and okc_patch.SW_ROUTE_FUSED and not sp_active:
                routed_small = okc_router.route_small(
                    self.gate, hidden_states, self._routing_op(), self.top_k,
                    hist=getattr(self.experts, "_okc_adapt_vcnt", None) if self.ep_rank == 0 else None,
                )
            if routed_small is not None:
                topk_scores, topk_ids, packed_topk_ids = routed_small
            elif okc_patch.SW_ROUTER:
                router_logits = okc_router.router_logits_small(self.gate, hidden_states)
            if routed_small is None and router_logits is None:
                if not use_split2 and router_input is None:
                    router_input = (
                        hidden_states.float() if self.router_in_fp32 else hidden_states
                    )
                router_logits = (
                    self.gate.forward_split2(hidden_states)
                    if use_split2
                    else self.gate(router_input)
                )
            if routed_small is None:
                topk_scores, topk_ids, packed_topk_ids = self._routing_op()(
                    None, router_logits, self.top_k, True, want_packed=True
                )
        # The SP shard's pre-expert norm does not read the routing: run it under the router's tail.
        prenormed = okc_patch.SP_PRENORM_EARLY and sp_active and layernorm_sp.okc_hopper_sp() and self._okc_prenorm_shard()
        if prenormed:
            expert_input = self.pre_expert_norm(expert_input)
        # OKC_SP_AG_ROWSFIRST: the rows' gather leaves now, under the router's tail, on its own stream
        rows_first = _okc_rows_first(self, expert_input) if prenormed and okc_patch.SP_AG_ROWSFIRST else None
        if use_side_stream:
            main_stream.wait_stream(side_stream)
        if use_side_stream and not torch.cuda.is_current_stream_capturing():
            cross_stream_tensors = (
                (router_logits, topk_scores, topk_ids)
                if getattr(get_forward(), "sp_active", False)
                else (topk_scores, topk_ids, packed_topk_ids)
            )
            for tensor in cross_stream_tensors:
                if tensor is not None:
                    tensor.record_stream(main_stream)

        # the ring holds whole prefill forwards of >= ADAPT_MIN_T tokens (SP: shard rows x ranks; every rank alike):
        # small SP extends (ext_coll) would otherwise fill it with 1K-token rows and understate the prefill straggler
        if (okc_patch.ADAPT_REPLICAS and getattr(self.experts, "_okc_adapt_ring", None) is not None
                and topk_ids.shape[0] * (self.ep_size if sp_active else 1) >= okc_patch.ADAPT_MIN_T):
            from sglang.kernels.ops.muse_spark_v12.okc_patch import adaptive as okc_adaptive

            okc_adaptive.count_prefill(self.experts, topk_ids, bool(sp_active), self.ep_rank)
        if (capturer := get_global_experts_capturer()) is not None:
            # Capture on main after the router join, in full unpadded token order.
            # SP experts below still consume the local ids.
            capture_topk_ids = topk_ids
            if sp_active:
                capture_topk_ids = layernorm_sp.sp_exit_gather(
                    topk_ids, num_tokens=layernorm_sp.sp_num_tokens()
                )
            capturer.capture(layer_id=self.layer_idx, topk_indices=capture_topk_ids)
        if getattr(get_forward(), "sp_active", False):
            return self._forward_sp_experts(
                expert_input, topk_scores, topk_ids, router_logits, prenormed=prenormed, rows_first=rows_first
            )
        if self.nvfp4_checkpoint:
            output = self.experts(
                expert_input,
                topk_scores,
                topk_ids,
                self.experts.post_expert_norm,
                pre_expert_norm=None if expert_prenormed else self.pre_expert_norm,
                prepared_input=prepared_input,
                packed_topk_ids=packed_topk_ids,
                ar_workspace=ar_workspace,
            )
        else:
            if self.pre_expert_norm is not None:
                expert_input = self.pre_expert_norm(expert_input)
            output = self.experts(
                expert_input,
                topk_scores,
                topk_ids,
                self.experts.post_expert_norm,
                packed_topk_ids=packed_topk_ids,
                ar_workspace=ar_workspace,
            )
        return output

    def _okc_ag_prenorm(self, tokens: int):
        """OKC_SW_AG_PRENORM: (weight, eps) when st would fold the pre-expert norm: the gather emits it instead."""
        if not (okc_patch.SW_AG_PRENORM and okc_patch.SW_ST_NORM_FOLD and self.nvfp4_checkpoint):
            return None
        state = getattr(self.experts, "_okc_moe_state", None)
        if state is None or not state["pg_only"] or self.expert_hidden_size != 4096:
            return None
        from sglang.kernels.ops.muse_spark_v12.okc_patch import moe as okc_moe

        if not okc_moe.dispatch_path(self.experts, tokens, self.pre_expert_norm).startswith("st"):
            return None
        probe = getattr(self, "_okc_ag_prenorm_w", None)
        if probe is None:
            probe = self._okc_ag_prenorm_w = okc_moe._norm_fold(
                torch.empty((1, self.expert_hidden_size), dtype=torch.bfloat16), self.pre_expert_norm) or ()
        return probe or None

    def _okc_prenorm_shard(self) -> bool:
        state = getattr(self.experts, "_okc_moe_state", None)
        return bool(
            okc_patch.SP_PRENORM_SHARD
            and self.pre_expert_norm is not None
            and (not self.nvfp4_checkpoint or (state is not None and state["pg_only"]))
        )

    def _forward_sp_experts(
        self,
        expert_input: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        prenormed: bool = False,
        rows_first=None,
    ) -> torch.Tensor:
        """Return combined source-token shards, before the nonlinear FFN tail."""
        if layernorm_sp.okc_hopper_sp():
            return self._forward_gathered_experts(expert_input, topk_scores, topk_ids, prenormed=prenormed,
                                                  rows_first=rows_first)
        if self.pre_expert_norm is not None:
            expert_input = self.pre_expert_norm(expert_input)
        if self.use_megamoe:
            from sglang.srt.layers.moe.muse_spark_megamoe import forward_megamoe

            return forward_megamoe(
                self.experts,
                expert_input,
                topk_scores,
                topk_ids,
                self.experts.post_expert_norm,
            )
        if self.sp_dispatcher is None:
            raise RuntimeError(
                "Muse Spark SP prefill requires an initialized FlashInfer A2A dispatcher."
            )
        dispatch_output = self.sp_dispatcher.dispatch(
            expert_input, StandardTopKOutput(topk_scores, topk_ids, router_logits)
        )
        combine_input = self.experts.forward_dispatched(
            dispatch_output, self.experts.post_expert_norm
        )
        return self.sp_dispatcher.combine(combine_input)

    def _forward_gathered_experts(
        self,
        expert_input: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        prenormed: bool = False,
        rows_first=None,
    ) -> torch.Tensor:
        """SP prefill without the A2A dispatcher: every rank runs its local experts on the whole
        sequence, exactly the non-SP EP, and reduce-scatters the partial sums back to the shards.
        Shards are [M_pad / tp, ...] in rank order, so gathering along dim 0 restores token order.
        rows_first: (gathered rows, their stream, the shard) from _okc_rows_first; the scores / ids follow by _okc_ids_gather."""
        global _okc_sp_logged
        if not _okc_sp_logged:
            _okc_sp_logged = True
            logger.info(
                "OKC SP prefill active: %d-token shard, gathered experts.",
                expert_input.shape[0],
            )
        tp_group = get_tp_group()
        world = tp_group.world_size

        def gather(shard: torch.Tensor) -> torch.Tensor:
            shard = shard.contiguous()
            full = shard.new_empty((shard.shape[0] * world, *shard.shape[1:]))
            tp_group.all_gather_into_tensor(full, shard)
            return full

        # The pre-expert RMSNorm is per row: on the shard before the gather it costs 1/tp of normalizing every
        # rank's copy of the whole chunk (16K chunk: 13 vs 85 us per layer), with the same values. pg-only and BF16
        # experts apply it as a plain op; the FlashInfer path fuses it into its input quantization, so it keeps it.
        prenorm_shard = self._okc_prenorm_shard()
        if prenorm_shard and not prenormed:
            expert_input = self.pre_expert_norm(expert_input)
        if rows_first is not None:
            from sglang.kernels.ops.muse_spark_v12.okc_patch import moe as okc_moe

            full_input, rows_stream, _ = rows_first   # the shard stays referenced until the join below
            full_scores, full_ids = _okc_ids_gather(tp_group, topk_scores, topk_ids)
            okc_moe.rows_first_prep(self.experts, full_ids)   # pg's routing prep (ids only) under the rows
            torch.cuda.current_stream().wait_stream(rows_stream)
            grouped = (full_input, full_scores, full_ids)
        else:
            grouped = _okc_push_gather(tp_group, expert_input, topk_scores, topk_ids)
        if grouped is None and okc_patch.SP_AG_GROUP:
            grouped = _okc_grouped_gather(tp_group, (expert_input, topk_scores, topk_ids))
        if grouped is not None:
            full_input, full_scores, full_ids = grouped
        else:
            full_input = gather(expert_input)
            full_scores = gather(topk_scores)
            full_ids = gather(topk_ids)
        g = _OkcGathered(self, tp_group, expert_input.shape[0], full_input, full_scores, full_ids, prenorm_shard)
        if _okc_p1ss_collect:   # P1-SS: the staged caller runs g.experts() / g.tail() in its own stage order
            return g
        g.experts()
        return g.tail()

    def forward_staged(self, hidden_states, router_input=None, forward_batch=None):
        """P1-SS: forward() as a generator with two stage ends, after the rows' gather is issued (A) and after the
        experts (M: gg2 with its M1-RS stores); the M1-RS receive (or the reduce-scatter) and the unpooler are R."""
        global _okc_p1ss_collect
        _okc_p1ss_collect = True
        try:
            g = self._forward_experts(hidden_states, router_input)
        finally:
            _okc_p1ss_collect = False
        if not isinstance(g, _OkcGathered):   # not an SP gathered call: nothing to stage
            yield
            yield
            return self._reduce_and_unpool(g)
        yield
        g.experts(defer=True)
        yield
        return self._reduce_and_unpool(g.tail())


class _OkcGathered:
    """One SP gathered-experts call split at its stage ends (P1-SS); experts() then tail() is the old body."""

    def __init__(self, moe, tp_group, shard_rows, full_input, full_scores, full_ids, prenorm_shard):
        self.moe, self.tp_group, self.shard_rows = moe, tp_group, shard_rows
        self.args = (full_input, full_scores, full_ids, prenorm_shard)
        self.output, self.box = None, []
        self.join = None   # P1-SS + W1b: the wait for the gathered rows, run when the experts start

    def experts(self, defer=False):
        if self.join is not None:
            self.join()
        moe = self.moe
        full_input, full_scores, full_ids, prenorm_shard = self.args
        self.args = None
        if moe.nvfp4_checkpoint:
            from sglang.kernels.ops.muse_spark_v12.okc_patch import moe as okc_moe

            with okc_moe.defer_receive(self.box) if defer else contextlib.nullcontext():
                self.output = moe.experts(
                    full_input,
                    full_scores,
                    full_ids,
                    moe.experts.post_expert_norm,
                    pre_expert_norm=None if prenorm_shard else moe.pre_expert_norm,
                )
        else:
            if moe.pre_expert_norm is not None and not prenorm_shard:
                full_input = moe.pre_expert_norm(full_input)
            self.output = moe.experts(
                full_input, full_scores, full_ids, moe.experts.post_expert_norm
            )

    def tail(self):
        moe, tp_group, output = self.moe, self.tp_group, self.output
        if self.box:   # P1-SS: the deferred M1-RS receive (gg2's stores were issued in the M stage)
            from sglang.kernels.ops.muse_spark_v12.okc_patch import p1ss as okc_p1ss

            okc_p1ss.pdl_fence(output.device)
            output = self.box.pop()()
        if getattr(moe.experts, "_okc_m1rs_shard", False):   # okc_patch.m1 rs_receive: this rank's shard
            moe.experts._okc_m1rs_shard = False
            return output
        output = output.contiguous()
        if okc_patch.SW_EXT_RS:   # extends: ld_reduce over the symmetric buffer (okc_patch/ext_coll.py)
            from sglang.kernels.ops.muse_spark_v12.okc_patch import ext_coll

            rs = ext_coll.reduce_scatter(output, tp_group.device_group.group_name, tp_group.world_size,
                                         tp_group.rank_in_group)
            if rs is not None:
                return rs
        shard = output.new_empty((self.shard_rows, *output.shape[1:]))
        tp_group.reduce_scatter_tensor(shard, output)
        return shard


def _okc_push_gather(tp_group, x, scores, ids):
    """OKC_SP_AG_PUSH: the section's three gathers as one push-plane launch (pg_moe.ag_rows) up to
    OKC_SP_AG_PUSH_MAX_TOKENS gathered rows, else None. The predicate is the same on every rank."""
    comm = tp_group.ca_comm
    if not (okc_patch.SP_AG_PUSH and getattr(comm, "has_multicast", False) and not comm.disabled):
        return None
    from sglang.kernels.ops.muse_spark_v12.okc_patch import pg_moe

    s, w = x.shape[0], comm.world_size
    if not (x.dtype == torch.bfloat16 and tuple(x.shape) == (s, 4096) and scores.dtype == torch.float32
            and ids.dtype == torch.int32 and tuple(scores.shape) == tuple(ids.shape) == (s, 8)
            and s * w <= okc_patch.SP_AG_PUSH_MAX_TOKENS and s * 520 * 16 <= comm.max_push_size
            and comm.config.num_push_blocks >= 2):
        return None
    return pg_moe.ag_rows(comm, x.contiguous(), scores.contiguous(), ids.contiguous())


_okc_rf_state: dict = {}


def _okc_rows_first(moe, x):
    """OKC_SP_AG_ROWSFIRST: start the SP MoE section's row gather [S, 4096] -> [world S, 4096] on its own stream now,
    on the pynccl communicator _okc_grouped_gather uses -> (rows, stream, shard), or None (the section gathers as usual).
    The rows land in the boot-time buffer (layernorm_sp's rows_buf): the next call's gather is queued on the stream after
    a wait on the current one, so it runs after this call's consumers; the caller holds the shard until it joins the
    stream, so neither tensor needs record_stream. Only for pg-only fp4 layers, gathered calls of
    SP_AG_ROWSFIRST_MIN..MAX_TOKENS rows, and when the scores / ids can go by the symmetric-memory gather
    (_okc_ids_buffer). The predicate is the same on every rank."""
    tp_group = get_tp_group()
    pyn, world = tp_group.pynccl_comm, tp_group.world_size
    state = getattr(moe.experts, "_okc_moe_state", None)
    buf = layernorm_sp._okc_hopper.rows_buf
    if (not moe.nvfp4_checkpoint or state is None or not state["pg_only"] or world <= 1
            or not okc_patch.SP_AG_ROWSFIRST_MIN_TOKENS <= x.shape[0] * world <= okc_patch.SP_AG_ROWSFIRST_MAX_TOKENS
            or buf is None or buf.shape[0] < x.shape[0] * world or buf.shape[1] != x.shape[1] or buf.dtype != x.dtype
            or buf.device != x.device
            or pyn is None or not (not pyn.disabled or tp_group.is_symmetric_memory_enabled())
            or _okc_ids_buffer(tp_group, x.device) is None):
        return None
    st = _okc_rf_state.get(x.device)
    if st is None:
        st = _okc_rf_state[x.device] = torch.cuda.Stream(device=x.device)
    if not _okc_rf_state.get("logged"):
        _okc_rf_state["logged"] = True
        logger.info("OKC SP rows-first MoE gather active (first T=%d).", x.shape[0] * world)
    x = x.contiguous()
    full = buf[: x.shape[0] * world]
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st), pyn.change_state(enable=True):
        pyn.all_gather(full, x)
    return full, st, x


def _okc_ids_buffer(tp_group, device):
    """The rows-first scores / ids gather's symmetric buffer: ext_coll's multimem-gather buffer (its agreed setup over
    the TP ranks, made once outside capture; a failed or partial setup on any rank gives None on all of them, so the
    section gathers as usual; the symmfail overlay's alloc / mc points cover it), or None."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import ext_coll

    st = ext_coll._buffer(tp_group.device_group.group_name, device)
    return st[0] if st is not None and st[1] >= okc_patch.SP_AG_ROWSFIRST_MAX_TOKENS * 32 else None


def _okc_ids_gather(tp_group, scores, ids):
    """[S, 8] fp32 scores + [S, 8] int32 ids of every rank -> ([world S, 8], [world S, 8]) in rank order: one multimem
    gather of the packed [S, 16] words through _okc_ids_buffer (exact bytes), copied out of the buffer. Buffer reuse:
    a peer's next call into this buffer comes after the section's reduce-scatter, which needs this rank's gg2 and so
    follows these copies (ext_coll's extend gathers take the buffer only below SP_SYMM_MIN_TOKENS rows)."""
    from sglang.kernels.ops.muse_spark_v12.okc_patch import ext_coll

    key = tp_group.device_group.group_name
    buf, w, s = ext_coll._STATE[key][0], tp_group.world_size, ids.shape[0]
    packed = torch.cat([scores.contiguous().view(torch.int32), ids.contiguous()], dim=1).view(torch.bfloat16)
    out = buf[: w * s * 32]
    torch.ops.symm_mem.multimem_all_gather_out(packed.reshape(-1), key, out)
    words = out.view(w * s, 32).view(torch.int32)
    return words[:, :8].contiguous().view(torch.float32), words[:, 8:].contiguous()


def _okc_grouped_gather(tp_group, shards):
    """All-gather each shard along dim 0 inside one NCCL group (one launch), on the pynccl communicator that
    tp_group.all_gather_into_tensor would use; None when that call would take another path (the caller then gathers
    one by one). The predicate depends only on the group's state, so every rank takes the same branch."""
    pyn = tp_group.pynccl_comm
    if pyn is None or not (not pyn.disabled or tp_group.is_symmetric_memory_enabled()):
        return None
    world = tp_group.world_size
    shards = [s.contiguous() for s in shards]
    outs = [s.new_empty((s.shape[0] * world, *s.shape[1:])) for s in shards]
    with pyn.change_state(enable=True):
        pyn.group_start()
        try:
            for out, shard in zip(outs, shards):
                pyn.all_gather(out, shard)
        finally:
            pyn.group_end()
    return outs


class MuseSpark1xDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MuseSpark1xTextConfig,
        layer_idx: int,
        nvfp4_checkpoint: bool,
        prefix: str,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.operator_dtype = config.dtype
        self.self_attn = MuseSpark1xAttention(
            config, layer_idx, add_prefix("self_attn", prefix)
        )
        self.post_attn_norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.post_norm_eps,
            has_weight=False,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 1,
        )
        self.pre_feedforward_layernorm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            residual_id=2 * layer_idx + 2,
        )
        ffn_norm_cls = MuseSpark1xRMSNormWithInputScale
        self.post_feedforward_layernorm = ffn_norm_cls(
            config.hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
        )
        self.is_moe_layer = self._is_moe_layer(config, layer_idx)
        self._router_dtype = torch.float32 if self.is_moe_layer else None
        self.mlp = (
            MuseSpark1xMoE(
                config,
                layer_idx,
                nvfp4_checkpoint,
                add_prefix("mlp", prefix),
                quant_config,
            )
            if self.is_moe_layer
            else MuseSpark1xMLP(config, add_prefix("mlp", prefix))
        )
        residual_cls = MuseSpark1xNormPreservingResidualConnection
        self.post_attention_residual_gate = residual_cls(
            config.hidden_size, temperature=config.residual_gate_temperature
        )
        self.post_feedforward_residual_gate = residual_cls(
            config.hidden_size, temperature=config.residual_gate_temperature
        )
        # Each boundary -- post-norm, residual add, next pre-norm -- is one fused
        # kernel. The FFN boundary's pre-norm belongs to the next layer, so the
        # model binds it in bind_next_pre_attn_norm once every layer exists.
        self._fused_attn_boundary = FusedPostNormResidualPreNorm(
            self.post_attn_norm,
            self.post_attention_residual_gate,
            self.pre_feedforward_layernorm,
        )
        # OKC (OKC_NORM_PREFETCH): at TP > 1 this boundary's predecessor is the o-proj all-reduce, so the norm may load
        # the stream and its weights before griddepcontrol.wait
        self._fused_attn_boundary._okc_prefetch_stream = get_parallel().tp_size > 1
        self._fused_ffn_boundary = None
        self._next_layer = None
        # OKC: the model's final norm, bound on the last layer only (not a submodule: the model owns it).
        object.__setattr__(self, "_okc_final_norm", None)
        object.__setattr__(self, "_okc_tail_used", False)
        # (forward batch, QKV) the previous layer's Helion verify tail computed.
        self._verify_qkv = None
        # The checkpoint keeps the first and last layers' experts in BF16; those
        # layers stay on the stock path.
        self._megakernel_capable = (
            expert_tp.helion_enabled() and self.is_moe_layer and nvfp4_checkpoint
        )

    @staticmethod
    def _is_moe_layer(config: MuseSpark1xTextConfig, layer_idx: int) -> bool:
        return config.mlp_layer_types[layer_idx] == "sparse"

    def bind_next_pre_attn_norm(self, next_pre_attn_norm: MuseSpark1xRMSNorm) -> None:
        self._fused_ffn_boundary = FusedPostNormResidualPreNorm(
            self.post_feedforward_layernorm,
            self.post_feedforward_residual_gate,
            next_pre_attn_norm,
        )
        if (
            self.is_moe_layer
            and self.operator_dtype == torch.bfloat16
            and envs.SGLANG_ENABLE_MUSE_SPARK_AG_NORM_FUSION.get()
        ):
            self.mlp.set_unpool_gather_boundary(self._fused_ffn_boundary)

    def bind_next_layer(self, next_layer: MuseSpark1xDecoderLayer) -> None:
        # Not a submodule: the model owns it. The Helion verify tail computes its QKV.
        object.__setattr__(self, "_next_layer", next_layer)

    def forward(
        self,
        positions: torch.Tensor,
        stream: torch.Tensor,
        attn_input: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns the updated residual stream and the next layer's attention input."""
        if self._megakernel_capable:
            from sglang.kernels.ops.muse_spark_v12.helion_megakernels.integration import (
                try_helion_megakernel_forward,
            )

            result = try_helion_megakernel_forward(
                self, positions, stream, attn_input, forward_batch
            )
            if result is not None:
                return result
        ffn_input, stream, router_input = self._attn_stage(positions, stream, attn_input, forward_batch)
        if self.is_moe_layer:
            ffn_output = self.mlp(ffn_input, router_input, forward_batch)
        else:
            ffn_output = self.mlp(ffn_input)
        return self._ffn_tail(stream, ffn_output)

    def _attn_stage(self, positions, stream, attn_input, forward_batch):
        """forward() up to the FFN: attention and the attention boundary; returns (ffn_input, stream, router_input)."""
        # OKC rsfold: under SP the o-proj reduce-scatter leaves its slots unsummed for the boundary's norm kernel
        rs_fold = (okc_patch.SW_RS_NORM_FOLD and get_forward().sp_active
                   and self._fused_attn_boundary.jit_selected())
        with okc_sp_tp_ce.defer_slot_sum() if rs_fold else contextlib.nullcontext():
            attn_output = self.self_attn(positions, attn_input, forward_batch)
        rs_slots = okc_sp_tp_ce.take_slots(attn_output) if rs_fold else None
        router_dtype = self._router_dtype
        if router_dtype is not None and (
            _use_split_bf16_router(attn_output.shape[0])
            or (okc_patch.SW_ROUTER and attn_output.shape[0] <= okc_router.MAX_TOKENS)
        ):
            router_dtype = None
        if rs_slots is not None:   # jit_selected(): the call below would take forward_jit
            return self._fused_attn_boundary.forward_jit(
                stream, attn_output, branch_dtype=self.operator_dtype, router_dtype=router_dtype,
                branch_slots=rs_slots)
        return self._fused_attn_boundary(
            stream,
            attn_output,
            branch_dtype=self.operator_dtype,
            router_dtype=router_dtype,
        )

    def _ffn_tail(self, stream, ffn_output):
        """forward() after the FFN: the FFN boundary (the last layer: the model's tail); returns forward()'s value."""
        # A TP-sliced branch means the unpooler gather was deferred to the FFN boundary.
        if ffn_output.shape[-1] != stream.shape[-1]:
            next_attn_input, stream = self._fused_ffn_boundary.forward_with_ag(
                stream, ffn_output, get_tp_group().ca_comm
            )
            return stream, next_attn_input
        if self._fused_ffn_boundary is None:
            okc_tail = None
            if self._okc_final_norm is not None and not getattr(get_forward(), "sp_active", False):
                from sglang.kernels.ops.muse_spark_v12.okc_patch import glue as okc_glue

                okc_tail = okc_glue.tail_boundary(self, stream, ffn_output, self._okc_final_norm)
            # set on every call: MuseSpark1xModel reads it right after this layer to skip its own final norm
            object.__setattr__(self, "_okc_tail_used", okc_tail is not None)
            if okc_tail is not None:
                return okc_tail   # (stream.to(bf16), the model's final norm of it)
            return self.post_feedforward_residual_gate(
                stream, self.post_feedforward_layernorm(ffn_output)
            ), None
        next_attn_input, stream, _ = self._fused_ffn_boundary(
            stream, ffn_output, branch_dtype=self.operator_dtype
        )
        return stream, next_attn_input

    def forward_staged(self, positions, stream, attn_input, forward_batch):
        """P1-SS: forward() as a generator of three stages, cut at the MoE's two stage ends (MuseSpark1xMoE.forward_staged):
        A = attention + boundary + MoE front, M = the experts, R = the rest; returns forward()'s value."""
        ffn_input, stream, router_input = self._attn_stage(positions, stream, attn_input, forward_batch)
        if self.is_moe_layer and isinstance(self.mlp, MuseSpark1xMoE):
            ffn_output = yield from self.mlp.forward_staged(ffn_input, router_input, forward_batch)
        else:
            ffn_output = self.mlp(ffn_input, router_input, forward_batch) if self.is_moe_layer else self.mlp(ffn_input)
            yield
            yield
        return self._ffn_tail(stream, ffn_output)

    def self_attn_input_norm(self, stream: torch.Tensor) -> torch.Tensor:
        return self.input_layernorm(stream).to(self.operator_dtype)


# Register the input norm at the decoder layer, as in the HF checkpoint.
def _install_pre_attn_norm(
    config: MuseSpark1xTextConfig, layer: MuseSpark1xDecoderLayer
) -> None:
    layer.input_layernorm = MuseSpark1xRMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        has_weight=True,
        zero_centered_gamma=True,
        residual_id=2 * layer.layer_idx + 1,
    )


class MuseSpark1xModel(nn.Module):
    def __init__(
        self,
        config: MuseSpark1xTextConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.config = config
        self.operator_dtype = config.dtype
        self.residual_dtype = torch.float32
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=add_prefix("embed_tokens", prefix),
            **get_embedding_tp_kwargs(),
        )
        self.embed_norm = MuseSpark1xRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )
        if okc_patch.SW_BOUNDARY_EARLY:
            layernorm_sp.okc_boundary_layer(config.num_hidden_layers - okc_patch.BOUNDARY_EARLY_LAYERS)
        self.layers = nn.ModuleList()
        # Safe to share: each MoE joins its side work before returning.
        # torch.cuda.Stream() raises outright in a CPU-only build, so the CPU
        # path never builds one; _forward_experts skips the overlap there.
        side_stream = None if _is_cpu else torch.cuda.Stream()
        for layer_idx in range(config.num_hidden_layers):
            layer_prefix = add_prefix(f"layers.{layer_idx}", prefix)
            layer = MuseSpark1xDecoderLayer(
                config,
                layer_idx,
                _uses_nvfp4_experts(quant_config, layer_prefix),
                layer_prefix,
                quant_config,
            )
            if layer.is_moe_layer:
                layer.mlp.side_stream = side_stream
            _install_pre_attn_norm(config, layer)
            self.layers.append(layer)
        for layer, next_layer in zip(self.layers[:-1], self.layers[1:]):
            layer.bind_next_pre_attn_norm(next_layer.input_layernorm)
            layer.bind_next_layer(next_layer)
        self._sp_moe_dispatcher = None
        self._finalize_ar_workspace = None
        if is_cuda() and envs.SGLANG_ENABLE_MUSE_SPARK_FINALIZE_AR.get():
            from sglang.kernels.ops.muse_spark_v12.fi_trtllm_moe_finalize_with_rmsnorm_cutedsl import (
                FinalizePushWorkspace,
            )
            from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
                CustomAllReduceV2,
            )

            tp_group = get_tp_group()
            if (
                get_parallel().tp_size not in (4, 8)
                or get_parallel().moe_ep_size != get_parallel().tp_size
                or is_dp_attention_enabled()
                or torch.cuda.get_device_capability()[0] != 10
            ):
                logger.warning(
                    "Finalize AR fusion unavailable: requires Blackwell TP4/EP4 or TP8/EP8 "
                    "without DP attention. Using separate finalize and AR."
                )
            else:
                moe_layers = [layer.mlp for layer in self.layers if layer.is_moe_layer]
                comm = tp_group.ca_comm
                if not moe_layers:
                    logger.warning("Finalize AR fusion unavailable: no MoE layers")
                elif not isinstance(comm, CustomAllReduceV2) or comm.disabled:
                    logger.warning(
                        "Finalize AR fusion unavailable: custom AR v2 is required. "
                        "Using separate finalize and AR."
                    )
                elif (
                    moe_layers[0].expert_hidden_size <= 0
                    or moe_layers[0].expert_hidden_size % 256
                    or comm.max_push_size < moe_layers[0].expert_hidden_size * 2
                    or not 0
                    < comm._push_counter.numel()
                    <= torch.cuda.get_device_properties(
                        comm.device
                    ).multi_processor_count
                ):
                    logger.warning(
                        "Finalize AR fusion unavailable: unsupported hidden size "
                        "or push workspace capacity. Using separate finalize and AR."
                    )
                else:
                    self._finalize_ar_workspace = FinalizePushWorkspace(
                        comm,
                        tp_group.device_group,
                        hidden_dim=moe_layers[0].expert_hidden_size,
                        multicast=comm.world_size > 4 and comm.has_multicast,
                    )
                for moe_layer in moe_layers:
                    moe_layer.set_finalize_ar_workspace(self._finalize_ar_workspace)
        if layernorm_sp.layernorm_sp_enabled() and layernorm_sp.okc_hopper_sp():
            logger.info(
                "OKC SP prefill on Hopper: %s participant linears, all-gathered experts.",
                "symm-mem fused" if layernorm_sp.okc_hopper_sp_symm() else "NCCL",
            )
        elif layernorm_sp.layernorm_sp_enabled():
            # Scope CUTLASS collectives to Muse; leave the generic Qwen SP
            # path unchanged. Decode never calls the SP participant hooks.
            for layer in self.layers:
                linears = [layer.self_attn.qkv_proj, layer.self_attn.o_proj]
                if not layer.is_moe_layer:
                    linears.extend([layer.mlp.gate_up_proj, layer.mlp.down_proj])
                for linear in linears:
                    linear._sp_use_cutlass = True
            moe_layers = [layer.mlp for layer in self.layers if layer.is_moe_layer]
            if moe_layers and not getattr(
                get_parallel(), "enable_megamoe_for_layernorm_sp", False
            ):
                from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
                    FlashinferDispatcher,
                )

                first_moe = moe_layers[0]
                self._sp_moe_dispatcher = FlashinferDispatcher(
                    group=get_tp_group().device_group,
                    router_topk=first_moe.top_k,
                    num_experts=first_moe.num_experts,
                    num_local_experts=first_moe.num_local_experts,
                    hidden_size=first_moe.expert_hidden_size,
                    moe_runner_config=first_moe.experts.moe_runner_config,
                    enable_sp_prefill_a2a=True,
                )
                for moe_layer in moe_layers:
                    moe_layer.set_sp_dispatcher(self._sp_moe_dispatcher)
        self.norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            output_norm_gain_center_type="zero",
            residual_id=-1,
        )
        # Layer indices whose output residual stream feeds the DFLASH draft context.
        self.layers_to_capture: List[int] = []
        if self.layers and okc_patch.SW_TAIL_NORM:
            object.__setattr__(self.layers[-1], "_okc_final_norm", self.norm)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Prefill CUDA graphs capture this decoder directly, below the causal
        # LM's scope; open it here as well (it nests) so they keep the defaults.
        with expert_tp.default_fused_ops_for_single_request(forward_batch):
            return self._forward(input_ids, positions, forward_batch, input_embeds)

    def _forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # okc_patch caches per-forward metadata here; CUDA-graph capture runs warmup forwards on the same batch
        # first, so a cache left from them would keep the captured graph from recomputing it on every replay.
        forward_batch.okc_fs_meta = None
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        if input_embeds is None and self.embed_norm is not None:
            hidden_states = self.embed_norm(hidden_states)
        sp_active = layernorm_sp.runs_sp(
            forward_batch.forward_mode, num_tokens=hidden_states.shape[0]
        )
        get_forward().set("sp_active", sp_active)
        if sp_active:
            hidden_states = layernorm_sp.sp_entry_scatter(hidden_states)
        stream = hidden_states.to(self.residual_dtype)
        attn_input = self.layers[0].self_attn_input_norm(stream)
        aux_hidden_states = []
        for layer_idx, layer in enumerate(self.layers):
            stream, attn_input = layer(positions, stream, attn_input, forward_batch)
            if layer_idx in self.layers_to_capture:
                # Draft context projection reads operator dtype, not residual_dtype.
                aux_hidden_states.append(stream.to(self.operator_dtype))
        if self.layers and getattr(self.layers[-1], "_okc_tail_used", False):
            # The last layer's OKC tail boundary returned (stream.to(bf16), self.norm of it).
            hidden_states = attn_input
        else:
            norm_input = stream.to(self.operator_dtype)
            hidden_states = self.norm(norm_input).to(self.operator_dtype)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


    def _forward_p1ss(self, input_ids, positions, kids, ctxs, input_embeds):
        """P1-SS (docs/serving/review-1031/STRUCTURAL.md §3.1c): the two children of a two-chunk pass (okc_patch/p1ss.py)
        through the decoder layers on ONE stream in _p1ss_order, the same order on every rank. Every stage runs under
        its child's attention backend (ctxs), SP verdict and token count. Returns [(final hidden (the SP shard when the
        child runs SP), DFlash taps, tokens, SP)] per child."""
        from sglang.srt.model_executor.forward_context import forward_context

        last = len(self.layers) - 1
        st = []
        for fb, ctx in zip(kids, ctxs):
            s0, s1 = fb.tbo_parent_token_range
            fb.okc_fs_meta = None
            if input_embeds is not None:
                x = input_embeds[s0:s1]
            else:
                x = self.embed_tokens(input_ids[s0:s1])
                if self.embed_norm is not None:
                    x = self.embed_norm(x)
            sp = layernorm_sp.runs_sp(fb.forward_mode, num_tokens=s1 - s0)
            get_forward().set("sp_active", sp)
            if sp:
                x = layernorm_sp.sp_entry_scatter(x)
            stream = x.to(self.residual_dtype)
            st.append(dict(pos=positions[s0:s1], stream=stream, attn_input=self.layers[0].self_attn_input_norm(stream),
                           fb=fb, ctx=ctx, n=s1 - s0, sp=sp, taps=[], g=None, h=None))
        for c, kind, j in _p1ss_order(len(self.layers)):
            s = st[c]
            get_forward().set("sp_active", s["sp"])
            layernorm_sp.set_sp_num_tokens(s["n"])
            with forward_context(s["ctx"]):
                if kind == "A":
                    s["g"] = self.layers[j].forward_staged(s["pos"], s["stream"], s["attn_input"], s["fb"])
                    next(s["g"])
                    continue
                if kind == "M":
                    next(s["g"])
                    continue
                try:
                    next(s["g"])
                except StopIteration as e:
                    s["stream"], s["attn_input"] = e.value
                else:
                    raise RuntimeError("P1-SS: a staged layer yielded more than twice")
                if j in self.layers_to_capture:
                    s["taps"].append(s["stream"].to(self.operator_dtype))
                if j == last:   # _forward's tail for this child
                    s["h"] = (s["attn_input"] if getattr(self.layers[-1], "_okc_tail_used", False) else
                              self.norm(s["stream"].to(self.operator_dtype)).to(self.operator_dtype))
        return [(s["h"], s["taps"], s["n"], s["sp"]) for s in st]


def _p1ss_order(num_layers):
    """P1-SS's static stage order, [(child, stage, layer)]: A_a(j) | M_b(j-1) | M_a(j) | R_b(j-1) | A_b(j) | R_a(j). Child
    b's attention at layer j follows child a's (it reads a's K / V); a's rows gather (issued in A) flies under b's M,
    b's under a's R and next A; each R comes after one long stage of the other child (STRUCTURAL.md §3.1)."""
    order = []
    for j in range(num_layers):
        order.append((0, "A", j))
        if j:
            order.append((1, "M", j - 1))
        order.append((0, "M", j))
        if j:
            order.append((1, "R", j - 1))
        order += [(1, "A", j), (0, "R", j)]
    return order + [(1, "M", num_layers - 1), (1, "R", num_layers - 1)]


class MuseSparkModel(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.language_model = MuseSpark1xModel(
            config, quant_config, add_prefix("language_model", prefix)
        )

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return self.language_model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )

    @property
    def layers(self):
        return self.language_model.layers


class MuseSpark1xForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: MuseSpark1xTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if getattr(config, "inference_moe_w13_interleaved", False):
            raise ValueError("Interleaved expert checkpoints are not supported")
        quant_name = _quantization_name(quant_config)
        int4_checkpoint = _uses_gptq_int4_experts(quant_config)
        if quant_name not in (None, "modelopt_fp4") and not int4_checkpoint:
            raise ValueError(
                "Only BF16, serialized ModelOpt NVFP4 and AutoGPTQ int4 are "
                f"supported, got {quant_name}"
            )
        self.config = config
        self.quant_config = quant_config
        self.nvfp4_checkpoint = quant_name == "modelopt_fp4"
        self.int4_checkpoint = int4_checkpoint
        self.model = MuseSparkModel(config, quant_config, add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.language_model.embed_tokens)
        self.logits_processor = MuseSpark1xLogitsProcessor(
            config, logit_scale=config.output_multiplier
        )
        self.capture_aux_hidden_states = False

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        with expert_tp.default_fused_ops_for_single_request(forward_batch):
            return self._forward(input_ids, positions, forward_batch, input_embeds)

    def _forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
    ):
        if (okc_patch.SW_P1SS and forward_batch.forward_mode == ForwardMode.EXTEND
                and positions.shape[0] > okc_patch.P1SS_CUT and not torch.cuda.is_current_stream_capturing()):
            return self._okc_p1ss_forward(input_ids, positions, forward_batch, input_embeds)
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if aux_hidden_states is not None and layernorm_sp.runs_sp(
            forward_batch.forward_mode,
            num_tokens=(
                input_ids.shape[0] if input_ids is not None else input_embeds.shape[0]
            ),
        ):
            # DFLASH consumes every prompt token in replicated token order.
            # Gather the taps before logits pruning applies full-token indices;
            # FULL capture also needs the entry-scatter padding removed.
            num_tokens = (
                input_ids.shape[0] if input_ids is not None else input_embeds.shape[0]
            )
            # Pack while sequence-sharded so the logits processor does not
            # duplicate the full replicated draft context with torch.cat.
            # Release the tap list before allocating the gathered output.
            aux_hidden_states = pack_aux_hidden_states(aux_hidden_states)
            # OKC: the draft's fc + hidden_norm (row-wise) on this shard, so the gather moves [N, hidden] rather than
            # [N, taps * hidden] and no rank projects the whole chunk (OKC_SW_DFLASH_FC_SP; set by the DFlash worker).
            okc_project = getattr(self, "_okc_aux_project", None)
            if okc_project is not None:
                projected = okc_project(aux_hidden_states)
                if projected is not None:
                    aux_hidden_states = projected
            aux_hidden_states = layernorm_sp.sp_exit_gather(
                aux_hidden_states, num_tokens=num_tokens
            )
        # Omni prefill calls the language model with materialized embeddings and
        # ``input_ids=None``. The generic logits processor cannot infer the full
        # token count in that form, so close the SP region here using the
        # replicated embedding length before it indexes last-token positions.
        if (
            input_ids is None
            and input_embeds is not None
            and layernorm_sp.runs_sp(
                forward_batch.forward_mode, num_tokens=input_embeds.shape[0]
            )
        ):
            hidden_states = layernorm_sp.sp_exit_gather(
                hidden_states, num_tokens=input_embeds.shape[0]
            )
            get_forward().set("sp_active", False)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def _okc_p1ss_forward(self, input_ids, positions, forward_batch, input_embeds):
        """P1-SS: a pass above P1SS_CUT tokens (only the scheduler's two-chunk passes are) as two children
        (okc_patch/p1ss.py, MuseSpark1xModel._forward_p1ss); then per child what _forward does with a pass's output
        (DFlash taps packed and projected, SP shards gathered), and the logits over the pass in token order (the
        children are contiguous token ranges)."""
        from sglang.kernels.ops.muse_spark_v12.okc_patch import p1ss as okc_p1ss

        kids, ctxs = okc_p1ss.split(forward_batch, okc_patch.P1SS_CUT)
        _OKC_POST_QKV.clear()   # a hook left by the previous pass's last layer never fires in this one
        logger.info("OKC P1-SS pass: %d + %d tokens", okc_patch.P1SS_CUT, positions.shape[0] - okc_patch.P1SS_CUT)
        outs = getattr(self.model, "language_model", self.model)._forward_p1ss(
            input_ids, positions, kids, ctxs, input_embeds)
        packed = None
        if outs[0][1] and any(o[3] for o in outs):   # an SP pass hands over its taps packed (and projected)
            packed = [pack_aux_hidden_states(o[1]) for o in outs]
            okc_project = getattr(self, "_okc_aux_project", None)
            projected = [okc_project(t) for t in packed] if okc_project is not None else [None]
            if all(t is not None for t in projected):   # one format for the whole pass
                packed = projected
        hid, aux = [], []
        for i, (h, taps, n, sp) in enumerate(outs):
            get_forward().set("sp_active", sp)
            if packed is not None:
                aux.append(layernorm_sp.sp_exit_gather(packed[i], num_tokens=n) if sp else packed[i])
            elif taps:
                aux.append(taps)
            hid.append(layernorm_sp.sp_exit_gather(h, num_tokens=n) if sp else h)
        get_forward().set("sp_active", False)
        aux = (torch.cat(aux) if packed is not None else [torch.cat(t) for t in zip(*aux)]) if aux else None
        # input_ids None: the rows are gathered already (the processor reads input_ids only to gather them)
        return self.logits_processor(None, torch.cat(hid), self.lm_head, forward_batch, aux)

    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
        """Capture the residual stream after each listed decoder layer.

        DFLASH drafts consume the target hidden state produced by layer
        ``layer_ids[k]`` (HF ``hidden_states[layer_ids[k] + 1]``).
        """
        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )
        num_layers = len(self.model.language_model.layers)
        bad = [i for i in layer_ids if not 0 <= i < num_layers]
        if bad:
            raise ValueError(
                f"DFLASH target layer ids {bad} are out of range for a "
                f"{num_layers}-layer Muse Spark target."
            )
        self.capture_aux_hidden_states = True
        self.model.language_model.layers_to_capture = list(layer_ids)

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def get_layers(self):
        return self.model.language_model.layers

    def get_attention_sliding_window_size(self) -> Optional[int]:
        if "sliding_attention" in self.config.layer_types:
            return self.config.sliding_window - 1
        return None

    def _copy_expert_stack(
        self, param: torch.Tensor, loaded_weight: torch.Tensor, name: str
    ) -> None:
        if param.shape == loaded_weight.shape:
            param.data.copy_(loaded_weight)
            return
        parallel = get_parallel()
        if loaded_weight.shape[0] == self.config.num_local_experts:
            experts = self.get_submodule(name.rsplit(".", 1)[0])
            if (
                isinstance(experts, MuseSpark1xFp4Experts)
                and experts.local_expert_ids is not None
            ):
                index = torch.tensor(
                    experts.local_expert_ids, dtype=torch.long, device=loaded_weight.device
                )
                # Byte view: older torch builds lack CPU index_select for the fp8 scale tensors.
                rows = loaded_weight.view(torch.uint8) if loaded_weight.element_size() == 1 else loaded_weight
                loaded_weight = rows.index_select(0, index).view(loaded_weight.dtype)
            else:
                local = loaded_weight.shape[0] // parallel.moe_ep_size
                start = parallel.moe_ep_rank * local
                loaded_weight = loaded_weight.narrow(0, start, local)
        if expert_tp.copy_shard(param, loaded_weight, name):
            return
        if param.shape != loaded_weight.shape:
            raise RuntimeError(
                f"Expert weight shape mismatch for {name}: "
                f"parameter={tuple(param.shape)}, checkpoint={tuple(loaded_weight.shape)}"
            )
        param.data.copy_(loaded_weight)

    @staticmethod
    def _load_int4_expert_tensor(
        param: torch.Tensor,
        param_name: str,
        loaded_weight: torch.Tensor,
        projection: str,
        suffix: str,
        expert_id: int,
    ) -> None:
        loader = getattr(param, "weight_loader", None)
        if loader is None:
            raise ValueError(f"Parameter '{param_name}' has no expert weight loader")

        if projection == "w13" and suffix != "g_idx":
            gate, up = loaded_weight.chunk(2, dim=-1)
            loader(param, gate, param_name, shard_id="w1", expert_id=expert_id)
            loader(param, up, param_name, shard_id="w3", expert_id=expert_id)
        else:
            loader(
                param,
                loaded_weight,
                param_name,
                shard_id="w1" if projection == "w13" else "w2",
                expert_id=expert_id,
            )

    def _load_packed_int4_expert(
        self,
        params: dict[str, torch.Tensor],
        name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        """Load fused per-expert or native stacked GPTQ tensors."""
        import re

        suffix_pattern = r"(qweight|qzeros|scales|g_idx)"
        per_expert = re.match(
            rf"^(.*\.experts)\.(\d+)\."
            rf"(w13_weight|w2_weight|gate_up_proj)\.{suffix_pattern}$",
            name,
        )
        if per_expert is not None:
            prefix, expert_raw, checkpoint_projection, suffix = per_expert.groups()
            projection = (
                "w13"
                if checkpoint_projection in ("w13_weight", "gate_up_proj")
                else "w2"
            )
            param_name = f"{prefix}.{projection}_{suffix}"
            param = params.get(param_name)
            if param is None:
                raise ValueError(
                    f"Checkpoint weight '{name}' has no parameter '{param_name}'"
                )
            self._load_int4_expert_tensor(
                param,
                param_name,
                loaded_weight,
                projection,
                suffix,
                int(expert_raw),
            )
            return True

        stacked = re.match(
            rf"^(.*\.experts)\.(gate_up_proj|down_proj)\.{suffix_pattern}$",
            name,
        )
        if stacked is not None:
            prefix, checkpoint_projection, suffix = stacked.groups()
            projection = "w13" if checkpoint_projection == "gate_up_proj" else "w2"
        else:
            stacked = re.match(
                rf"^(.*\.experts)\.(w13|w2)_{suffix_pattern}$",
                name,
            )
            if stacked is None:
                return False
            prefix, projection, suffix = stacked.groups()
        param_name = f"{prefix}.{projection}_{suffix}"
        param = params.get(param_name)
        if param is None:
            raise ValueError(
                f"Checkpoint weight '{name}' has no parameter '{param_name}'"
            )
        if loaded_weight.shape[0] != self.config.num_local_experts:
            raise RuntimeError(
                f"Expert count mismatch for {name}: "
                f"checkpoint={loaded_weight.shape[0]}, "
                f"config={self.config.num_local_experts}"
            )
        for expert_id, expert_weight in enumerate(loaded_weight):
            self._load_int4_expert_tensor(
                param,
                param_name,
                expert_weight,
                projection,
                suffix,
                expert_id,
            )
        return True

    def _load_expert_projection(
        self,
        params: dict[str, torch.Tensor],
        name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        import re

        match = re.match(
            r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(.+)$", name
        )
        if match is None:
            return False
        prefix, expert_raw, projection, suffix = match.groups()
        expert_id = int(expert_raw)
        if expert_id >= self.config.num_local_experts:
            return False
        is_down = projection == "down_proj"
        if self.int4_checkpoint:
            param_name = f"{prefix}.{'w2' if is_down else 'w13'}_{suffix}"
        elif suffix == "weight":
            param_name = f"{prefix}.{'down_proj' if is_down else 'gate_up_proj'}"
        else:
            return False
        param = params.get(param_name)
        if param is None:
            return False
        param.weight_loader(
            param,
            loaded_weight,
            param_name,
            shard_id=_EXPERT_SHARD_IDS[projection],
            expert_id=expert_id,
        )
        return True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        params.update(
            {name: value for name, value in self.named_buffers() if name not in params}
        )
        stacked = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        stacked.append((".qkv_proj", ".gate_proj", "g"))

        skipped = 0
        for raw_name, loaded_weight in weights:
            name = raw_name
            if name.endswith((".self_attn.q_proj.weight", ".self_attn.k_proj.weight")):
                # For below, using padding logic will have this "original_",
                # and currently it only exists under CPU path. Thus no impacts on others paths.
                heads = (
                    getattr(
                        self.config,
                        "original_num_attention_heads",
                        self.config.num_attention_heads,
                    )
                    if name.endswith(".q_proj.weight")
                    else getattr(
                        self.config,
                        "original_total_num_kv_heads",
                        self.config.num_key_value_heads,
                    )
                )
                shape = loaded_weight.shape
                loaded_weight = (
                    loaded_weight.reshape(heads, 2, shape[0] // heads // 2, *shape[1:])
                    .transpose(1, 2)
                    .reshape(shape)
                    .contiguous()
                )
            if should_skip_weight_loading(name, self.config):
                skipped += 1
                continue
            if name.endswith(
                (".self_attn.q_scale", ".self_attn.k_scale", ".self_attn.v_scale")
            ):
                skipped += 1
                continue
            # Expert activation scales are only meaningful when the experts stay
            # packed: the FP4 quant method turns them into w13/w2_input_scale_quant
            # and g1_scale_c. A BF16 expert loop has no use for them.
            if name.endswith((".gate_up_proj_input_scale", ".down_proj_input_scale")):
                if not self.nvfp4_checkpoint:
                    skipped += 1
                    continue
            # Packed NVFP4 experts reuse the BF16 parameter names without an
            # expert weight loader.
            if not self.nvfp4_checkpoint and self._load_expert_projection(
                params, name, loaded_weight
            ):
                continue
            if self.int4_checkpoint and self._load_packed_int4_expert(
                params, name, loaded_weight
            ):
                continue
            handled = False
            original_name = name
            for param_pattern, weight_pattern, shard_id in stacked:
                if weight_pattern not in name:
                    continue
                candidate = name.replace(weight_pattern, param_pattern)
                param = params.get(candidate)
                if param is None:
                    continue
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight, shard_id)
                handled = True
                break
            if handled:
                continue
            name = original_name
            param = params.get(name)
            if param is None:
                raise ValueError(
                    f"Checkpoint weight '{raw_name}' is outside the Muse Spark text model"
                )
            if ".experts." in name and ".post_expert_norm." not in name:
                self._copy_expert_stack(param, loaded_weight, name)
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
        if skipped:
            logger.info("Skipped %d non-text or unused checkpoint tensors", skipped)
        self.post_load_weights()

    @torch.no_grad()
    def post_load_weights(self) -> None:
        for layer in self.model.language_model.layers:
            for residual in (
                layer.post_attention_residual_gate,
                layer.post_feedforward_residual_gate,
            ):
                if isinstance(residual, MuseSpark1xNormPreservingResidualConnection):
                    residual.precompute_coefficients()
        for module in self.modules():
            if isinstance(module, MuseSpark1xRMSNorm):
                module.precompute_effective_weight()
        # Must follow precompute_effective_weight: the finalize op binds the weight.
        for module in self.modules():
            if (
                isinstance(module, MuseSpark1xMoE)
                and not module.nvfp4_checkpoint
                and not self.int4_checkpoint
            ):
                module.experts.prepare_trtllm_weights(module.experts.post_expert_norm)
        # The OKC small-T state (smallm extension, router hi/lo) is built here: before KV-pool
        # sizing and outside any CUDA-graph capture; the forwards only look it up.
        if okc_patch.SW_SMALLM:
            from sglang.kernels.ops.muse_spark_v12.okc_patch import smallm as okc_smallm

            okc_smallm.prepare()
        if okc_patch.SW_LT:
            from sglang.kernels.ops.muse_spark_v12.okc_patch import lt as okc_lt

            for module in self.modules():
                if isinstance(module, MuseSpark1xAttention):
                    okc_lt.prepare(module.qkv_proj.weight)
                elif isinstance(module, MuseSpark1xMoE) and module.pre_expert_proj is not None:
                    okc_lt.prepare(module.pre_expert_proj.weight)
                    okc_lt.prepare(module.post_expert_proj.weight)
        if okc_patch.SW_ROUTER:
            for module in self.modules():
                if isinstance(module, MuseSpark1xMoE):
                    okc_router.prepare_gate(module.gate)


# Preserve the CUDA implementation and its FlashInfer/JIT optimizations.
# The independently validated MI355 implementation keeps the fused boundaries,
# Quark loader, and kernel rounding points that were measured together.
if is_hip():
    from sglang.srt.rocm_muse_spark.model import MuseSpark1xForCausalLM


EntryClass = [MuseSpark1xForCausalLM]

__all__ = [
    "MuseSpark1xForCausalLM",
    "MuseSpark1xTextConfig",
]
