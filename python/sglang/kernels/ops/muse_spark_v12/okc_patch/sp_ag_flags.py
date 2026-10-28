"""OKC: sp_tp_ce's all-gather + matmul on readiness flags instead of barriers (OKC_SP_CE_AG_FLAGS).

After each push into rank dest's gather slot the push stream writes 1 into dest's flag[me] (cuStreamWriteValue32,
fenced after the copy); the consumer waits for flag[src] == 1 (cuStreamWaitValue32) before src's GEMM, then resets
it. Two uses of a flag are separated by all-rank collectives (o_proj RS, MoE), as for the gather slots, so 1 / 0
suffices (captured memops keep constant values). All pushes are enqueued before the first wait: a wait blocks its
hardware queue, which the push stream may share.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_flags = None
_failed = False


class Flags:
    """ptrs[p]: rank p's int32 flag array [W] as mapped in this process; flag[src] of rank r is set by rank src."""

    def __init__(self, ptrs, me):
        from cuda.bindings import driver as cu

        self.cu = cu
        self.sig = [cu.CUdeviceptr(p + 4 * me) for p in ptrs]          # flag[me] of every rank
        self.slot = [cu.CUdeviceptr(ptrs[me] + 4 * s) for s in range(len(ptrs))]   # my flag[src]
        self.write_flags = cu.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT
        self.wait_flags = int(cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ)
        dev = _ok(cu, cu.cuCtxGetDevice())
        if _ok(cu, cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, dev)):
            self.wait_flags |= int(cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_FLUSH)
        self._streams = {}

    def _cs(self, stream):
        cs = self._streams.get(stream.cuda_stream)
        if cs is None:
            cs = self._streams[stream.cuda_stream] = self.cu.CUstream(stream.cuda_stream)
        return cs

    def signal(self, stream, dest):
        _ok(self.cu, self.cu.cuStreamWriteValue32(self._cs(stream), self.sig[dest], 1, self.write_flags))

    def wait(self, stream, src):
        cs = self._cs(stream)
        _ok(self.cu, self.cu.cuStreamWaitValue32(cs, self.slot[src], 1, self.wait_flags))
        _ok(self.cu, self.cu.cuStreamWriteValue32(cs, self.slot[src], 0, self.write_flags))


def _ok(cu, res):
    if res[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {res[0]}")
    return res[1] if len(res) > 1 else None


def flags(tp_group, device):
    """Symmetric flag arrays, allocated once outside capture like sp_tp_ce._buffers; None if unusable."""
    global _flags, _failed
    if _flags is not None or _failed:
        return _flags
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        from torch.distributed import _symmetric_memory as symm_mem

        W = tp_group.world_size
        buf = symm_mem.empty(W, dtype=torch.int32, device=device)
        buf.zero_()
        torch.cuda.synchronize(device)      # zeroed before any peer returns from the rendezvous
        hdl = symm_mem.rendezvous(buf, tp_group.device_group.group_name)
        ptrs = [hdl.get_buffer(p, (W,), torch.int32, 0).data_ptr() for p in range(W)]
        _flags = Flags(ptrs, hdl.rank)
        _flags.keep = (buf, hdl)
        logger.info("OKC SP copy-engine all-gather on readiness flags (no barriers).")
    except Exception as e:
        logger.warning("OKC SP copy-engine all-gather flags unavailable, keeping barriers: %r", e)
        _failed = True
        _flags = None
    return _flags


def push(st, fl, x_shard):
    """x_shard [S, K] into slot me of every peer's gather buffer, each followed by its flag (stream st.streams[0])."""
    S, K = x_shard.shape
    W, me = st.world, st.rank
    X = st.streams[0]
    X.wait_stream(torch.cuda.current_stream())
    src = x_shard.view(-1)
    with torch.cuda.stream(X):
        for s in range(1, W):
            dest = (me + s) % W
            st.g[dest][me * S * K:(me + 1) * S * K].copy_(src)
            fl.signal(X, dest)


def gemms(st, fl, x_shard, weight):
    """[W*S, N]: the own block, then sources me-1, me-2, ... (arrival order), each once its flag is up."""
    S, K = x_shard.shape
    W, me = st.world, st.rank
    wt = weight.t()
    cur = torch.cuda.current_stream()
    out = torch.empty((W * S, weight.shape[0]), dtype=x_shard.dtype, device=x_shard.device)
    torch.mm(x_shard, wt, out=out[me * S:(me + 1) * S])
    for s in range(1, W):
        src = (me - s) % W
        fl.wait(cur, src)
        torch.mm(st.g[me][src * S * K:(src + 1) * S * K].view(S, K), wt, out=out[src * S:(src + 1) * S])
    cur.wait_stream(st.streams[0])     # the pushes read x_shard
    return out


def ag_matmul(st, fl, x_shard, weight):
    x_shard = x_shard.contiguous()
    push(st, fl, x_shard)
    return gemms(st, fl, x_shard, weight)
