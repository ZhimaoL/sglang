"""OKC verify-step glue (serving/kernels/verify-glue): small fused replacements for torch chains, each bit-identical
to the chain it replaces."""

import logging
from typing import Optional, Tuple

import torch

from sglang.kernels.ops.muse_spark_v12 import okc_patch as _sw

logger = logging.getLogger(__name__)
_logged_tail = False


def tail_boundary(layer, stream: torch.Tensor, ffn_output: torch.Tensor, final_norm) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The last decoder layer's FFN boundary (post-FFN norm with input scale, gated residual) plus the model's final
    norm of stream.to(bf16), one launch (OKC_SW_TAIL_NORM): 9 torch kernels, ~34 us per bs1 verify step.
    -> (stream.to(bf16), final hidden) or None (the caller runs the torch chain)."""
    from sglang.srt.layers.muse_spark_layers import (
        MuseSpark1xNormPreservingResidualConnection,
        MuseSpark1xRMSNorm,
        MuseSpark1xRMSNormWithInputScale,
    )

    if not _sw.SW_TAIL_NORM or torch.compiler.is_compiling():
        return None
    post, gate = layer.post_feedforward_layernorm, layer.post_feedforward_residual_gate
    if type(post) is not MuseSpark1xRMSNormWithInputScale or type(final_norm) is not MuseSpark1xRMSNorm:
        return None
    if type(gate) is not MuseSpark1xNormPreservingResidualConnection or gate._alpha is None or gate._beta is None:
        return None
    m = ffn_output.shape[0] if ffn_output.dim() == 2 else 0
    if m == 0 or ffn_output.shape[1] != 8192 or ffn_output.dtype is not torch.bfloat16 or not ffn_output.is_contiguous():
        return None
    if stream.shape != ffn_output.shape or stream.dtype is not torch.float32 or not stream.is_contiguous():
        return None
    w_post, w_final = post._get_effective_weight(), final_norm._get_effective_weight()
    for w in (w_post, w_final):
        if w is None or w.dtype is not torch.bfloat16 or not w.is_contiguous() or w.numel() != 8192 or not w.is_cuda:
            return None
    alpha, beta = gate._alpha, gate._beta
    for c in (alpha, beta):
        if c.dtype is not torch.float32 or not c.is_cuda or not c.is_contiguous() or c.numel() not in (1, 8192):
            return None
    from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import glue_ext

    out = torch.empty_like(ffn_output)
    stream_bf16 = torch.empty_like(ffn_output)
    glue_ext().tail_boundary(ffn_output, stream, w_post, float(post.eps), alpha, beta, w_final,
                             float(final_norm.eps), out, stream_bf16)
    global _logged_tail
    if not _logged_tail:
        _logged_tail = True
        logger.info("OKC tail boundary active (first M=%d).", m)
    return stream_bf16, out
