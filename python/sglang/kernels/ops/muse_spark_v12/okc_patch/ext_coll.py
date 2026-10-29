"""Own collectives for the follow-up extend's SP calls (serving/kernels/extend E44, X6).

The extend's per-layer qkv-in all-gather ([T, 8192] bf16 at T 512..4095: 8-64 MiB) runs on NCCL's ring, which tops out
at ~170-240 GB/s at those sizes. torch's symmetric-memory multimem all-gather (multimem.st of each rank's shard into
every rank's output through the NVSwitch, then a barrier) is 1.29x at 16 MiB (S10B step M) and moves the same bytes
(exact). Only this call site (layernorm_sp.column_parallel_g_matmul's plain gather, i.e. SP forwards below
OKC_SP_SYMM_MIN_TOKENS) takes it. The first turn's chunks take the fused paths, and verify sizes are not SP.

Buffer reuse: the gather lands in one symmetric buffer sized for the largest call. A rank's next gather into a peer's
buffer starts only after its o-proj reduce-scatter of the same layer completes. That completion needs every peer's
contribution, which each peer sends after its qkv GEMM has read the buffer, so no peer is still reading when the next
layer's stores arrive.
"""

import torch

from sglang.kernels.ops.muse_spark_v12 import okc_patch as _sw

_STATE = {}   # group name -> (symmetric bf16 buffer, elements) or None (no multicast: NCCL)


def _buffer(group_name: str, device):
    st = _STATE.get(group_name, False)
    if st is not False:
        return st
    if torch.cuda.is_current_stream_capturing():
        return None   # never allocate / rendezvous inside a capture (the eager warm-up makes it first)
    st = None
    try:
        import torch.distributed._symmetric_memory as symm_mem

        n = _sw.EXT_AG_MAX_BYTES // 2
        buf = symm_mem.empty(n, dtype=torch.bfloat16, device=device)
        hdl = symm_mem.rendezvous(buf, group_name)
        if int(getattr(hdl, "multicast_ptr", 0) or 0) != 0:
            st = (buf, n)
    except Exception:
        st = None
    _STATE[group_name] = st
    return st


def all_gather(shard: torch.Tensor, group_name: str, world: int):
    """[S, K] bf16 shard of every rank -> [world S, K] in rank order (a view of the symmetric buffer), or None (the
    caller's NCCL gather): OKC_SW_EXT_AG off, other dtype / layout, larger than OKC_EXT_AG_MAX_BYTES, or no multicast."""
    if not _sw.SW_EXT_AG or shard.dtype is not torch.bfloat16 or shard.dim() != 2 or world <= 1:
        return None
    n = world * shard.numel()
    if n * 2 > _sw.EXT_AG_MAX_BYTES:
        return None
    st = _buffer(group_name, shard.device)
    if st is None:
        return None
    out = st[0][:n]
    torch.ops.symm_mem.multimem_all_gather_out(shard.contiguous(), group_name, out)
    return out.view(world * shard.shape[0], shard.shape[1])
