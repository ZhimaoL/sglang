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

import logging

import torch

from sglang.kernels.ops.muse_spark_v12 import okc_patch as _sw

logger = logging.getLogger(__name__)
_STATE = {}   # group name -> (symmetric bf16 buffer, elements) or None (no multicast: NCCL)


def agree(ok) -> bool:
    """True only if every TP rank passes ok (MIN over the CPU group): a symmetric-memory path runs on all ranks or on
    none, since a rank that failed alone would leave its peers waiting inside the next collective."""
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_parallel

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=get_parallel().tp_group.cpu_group)
    return bool(flag.item())


def agreed(step):
    """step() on every TP rank: its result if no rank raised or got None, else None on all of them. Setup runs as
    agreed steps (allocate, rendezvous, first use), so no rank enters a collective that a peer has given up on."""
    try:
        r = step()
    except Exception as e:  # setup / IPC / multicast failure: the NCCL path
        logger.warning("OKC symm-mem setup failed: %r", e)
        r = None
    return r if agree(r is not None) else None


def _symm_ok() -> bool:
    """The startup probe's verdict (layernorm_sp._okc_symm_probe, agreed over the TP ranks)."""
    from sglang.srt.layers.layernorm_sp import _okc_hopper

    return _okc_hopper.symm_ok


def _mc_handle(buf, group_name: str):
    import torch.distributed._symmetric_memory as symm_mem

    hdl = symm_mem.rendezvous(buf, group_name)
    return hdl if int(getattr(hdl, "multicast_ptr", 0) or 0) else None


def _mc_gather_ok(hdl, buf, group_name: str):
    """One small multimem gather of the rank ids through the buffer, checked: True, or None."""
    w = hdl.world_size
    src = torch.full((8,), hdl.rank, dtype=torch.bfloat16, device=buf.device)
    torch.ops.symm_mem.multimem_all_gather_out(src, group_name, buf[: 8 * w])
    want = torch.arange(w, dtype=torch.bfloat16, device=buf.device).repeat_interleave(8)
    return True if torch.equal(buf[: 8 * w], want) else None


def _buffer(group_name: str, device):
    st = _STATE.get(group_name, False)
    if st is not False:
        return st
    if torch.cuda.is_current_stream_capturing():
        return None   # never allocate / rendezvous inside a capture (the eager warm-up makes it first)
    st = None
    if _symm_ok():
        import torch.distributed._symmetric_memory as symm_mem

        n = _sw.EXT_AG_MAX_BYTES // 2
        buf = agreed(lambda: symm_mem.empty(n, dtype=torch.bfloat16, device=device))
        hdl = buf is not None and agreed(lambda: _mc_handle(buf, group_name))
        if hdl and agreed(lambda: _mc_gather_ok(hdl, buf, group_name)):
            st = (buf, n)
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
    # torch checks out.dim() == input.dim() and out.size(0) == world x input.size(0): both flat (S15 step M: a 2-D
    # shard into the 1-D buffer view raised "input/output dimension mismatch")
    torch.ops.symm_mem.multimem_all_gather_out(shard.contiguous().view(-1), group_name, out)
    return out.view(world * shard.shape[0], shard.shape[1])
