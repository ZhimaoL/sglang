"""OKC_SW_AR_NORM: o-proj all-reduce + post-attention norm_residual as one kernel (okc_ar_norm.cuh, bit-identical)."""
import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args
from sglang.kernels.ops.muse_spark_v12 import okc_patch
from sglang.kernels.ops.muse_spark_v12.norm_residual import _cluster_size_for_shape

H = 8192


@cache_once
def _jit_module(hidden: int, world: int):
    args = make_cpp_args(hidden, world, is_arch_support_pdl())
    return load_jit("okc_ar_norm", *args, cuda_files=["muse_spark_v12/okc_ar_norm.cuh"],
                    cuda_wrappers=[("run", f"OkcArNorm<{args}>::run")])


def cluster(m: int) -> int:
    c = _cluster_size_for_shape(H, m)
    while c > okc_patch.HOPPER_NORM_MAX_CLUSTER:
        c //= 2
    return c


class _Fused:
    def __init__(self, layer):
        self.layer = layer

    def attn(self, positions, h, forward_batch):   # MuseSpark1xAttention.forward without the o proj's all-reduce
        a = self.layer.self_attn
        qkv, _ = a.qkv_proj(h)
        return a.o_proj(a.gated_attention(positions, qkv, forward_batch), skip_all_reduce=True)[0]

    def __call__(self, stream, partial, **_):   # as the boundary: (next input, new stream, router input)
        from sglang.srt.distributed import get_tp_group

        b = self.layer._fused_attn_boundary
        b._ensure_cached_weights(torch.bfloat16)   # the weights as its JIT path passes them
        g = b._use_residual_gate
        out, ns = torch.empty_like(partial), torch.empty_like(stream)
        _jit_module(H, 8).run(get_tp_group().ca_comm.obj, partial, stream, out, ns,
                              b._cc_post_weight if b._post_norm.has_weight else None,
                              b._cc_pre_weight if b._pre_norm.has_weight else None,
                              b._cc_alpha if g else None, b._cc_beta if g else None,
                              float(b._post_norm.eps), float(b._pre_norm.eps), bool(b._post_norm_is_input_scale),
                              cluster(partial.shape[0]))
        return out, ns, None


def fused(layer, h):
    """_Fused when the layer's o proj all-reduces with the TP8 custom AR, M % 8 == 0, M <= 32, no router output and
    the boundary takes the JIT cluster norm; else None."""
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import CustomAllReduceV2
    from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
    from sglang.srt.models import muse_spark as ms
    from sglang.srt.runtime_context import get_exec, get_forward

    m, o, tp = h.shape[0], layer.self_attn.o_proj, get_tp_group()
    rd = layer._router_dtype
    if rd is not None and (ms._use_split_bf16_router(m) or (okc_patch.SW_ROUTER and m <= ms.okc_router.MAX_TOKENS)):
        rd = None
    ok = (rd is None and layer.operator_dtype == torch.bfloat16 and h.dtype == torch.bfloat16 and h.dim() == 2
          and h.shape[1] == H and 0 < m <= 32 and m % 8 == 0 and not torch.compiler.is_compiling()
          and tp.world_size == 8 and isinstance(tp.ca_comm, CustomAllReduceV2) and not tp.ca_comm.disabled
          and o.reduce_results and not o.use_dp_attention_reduce and not getattr(o, "use_decode_attn_tp", False)
          and not should_skip_mlp_all_reduce() and not get_exec().comm.enable_quant_communications
          and not get_forward().sp_active and layer._fused_attn_boundary.jit_selected()
          and torch.cuda.get_device_capability(h.device)[0] == 9 and cluster(m) in (4, 8, 16)
          and not tp._can_use_flashinfer_allreduce(h) and tp._resolve_outplace_all_reduce_method(h) == "ca")
    return _Fused(layer) if ok else None
