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


# ---- OKC_SW_EXT_RS: the extend's reduce-scatters (o-proj [T, 8192], MoE out [T, 4096]) as multimem.ld_reduce ----------
# Each rank's [T, N] partials sit in one symmetric buffer (the o-proj GEMM writes there directly; the MoE output is
# copied in), one symmetric-memory barrier, then every rank multimem.ld_reduce's its own T / world rows through the
# multicast address with fp32 accumulation in the switch (.acc::f32: equal to the fp32 sum rounded once; NCCL's ring
# sums in bf16 per hop, so ~half the elements differ from NCCL by an ulp: a reduction-order numerics change, default
# off until a consistency read). Buffer reuse: with OKC_EXT_RS_FENCE (default 1) a second barrier after the ld_reduce
# keeps every rank from writing its buffer again before all peers finished reading it, so the call is safe whatever
# runs around it (S15 step M: two back-to-back calls with nothing between raced at 8 MiB once). With it off, the
# server's order still separates consecutive reduce-scatters by an all-gather (the MoE section's, the next layer's
# qkv-in), which completes only after every peer's previous ld_reduce.
_RS_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void __launch_bounds__(512) okc_rsld(unsigned long long mc, uint4* __restrict__ out, long long nv,
                                                long long off) {
    const long long stride = (long long)gridDim.x * 512 * 4;
    for (long long b = (long long)blockIdx.x * 512 * 4; b < nv; b += stride) {
        uint4 r[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const long long i = b + j * 512 + threadIdx.x;
            if (i >= nv) break;
            const unsigned long long a = mc + (unsigned long long)(off + i) * 16;
            asm volatile("multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
                         : "=r"(r[j].x), "=r"(r[j].y), "=r"(r[j].z), "=r"(r[j].w) : "l"(a) : "memory");
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const long long i = b + j * 512 + threadIdx.x;
            if (i < nv) out[i] = r[j];
        }
    }
}
void okc_rsld_run(int64_t mc, torch::Tensor out, int64_t row0_elems) {
    TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.numel() % 8 == 0 && row0_elems % 8 == 0,
                "okc_rsld: contiguous bf16, 16-byte vectors");
    const long long nv = out.numel() / 8, off = row0_elems / 8;
    const int grid = (int)std::min<long long>((nv + 2047) / 2048, 132 * 8);
    if (nv == 0) return;
    okc_rsld<<<grid, 512, 0, at::cuda::getCurrentCUDAStream()>>>((unsigned long long)mc,
        reinterpret_cast<uint4*>(out.data_ptr()), nv, off);
}
"""
_RS_EXT = None


def _rs_ext():
    global _RS_EXT
    if _RS_EXT is None:
        from sglang.kernels.ops.muse_spark_v12.okc_patch.cuda_ext import _load_inline

        _RS_EXT = _load_inline(name="okc_ext_rsld_v1", cpp_sources="void okc_rsld_run(int64_t, torch::Tensor, int64_t);",
                              cuda_sources=_RS_SRC, functions=["okc_rsld_run"], extra_cuda_cflags=["-O3"],
                              verbose=False)
    return _RS_EXT


_RS_STATE = {}   # group name -> (symmetric bf16 buffer, elements, handle, multicast address) or None


def _rs_buffer(group_name: str, device):
    st = _RS_STATE.get(group_name, False)
    if st is not False:
        return st
    if torch.cuda.is_current_stream_capturing():
        return None
    st = None
    if _symm_ok():
        import torch.distributed._symmetric_memory as symm_mem

        n = _sw.EXT_AG_MAX_BYTES // 2
        buf = agreed(lambda: symm_mem.empty(n, dtype=torch.bfloat16, device=device))
        hdl = buf is not None and agreed(lambda: _mc_handle(buf, group_name))
        if hdl and agreed(_rs_ext):
            st = (buf, n, hdl, int(hdl.multicast_ptr))
    _RS_STATE[group_name] = st
    return st


def rs_target(rows: int, cols: int, group_name: str, world: int):
    """[rows, cols] bf16 view of this rank's symmetric RS buffer for the partials (a producer writes them there), or
    None: OKC_SW_EXT_RS off, rows not divisible by world, larger than the buffer, or no multicast."""
    if not _sw.SW_EXT_RS or world <= 1 or rows % world or (rows // world * cols) % 8 or rows >= _sw.SP_SYMM_MIN_TOKENS:
        return None   # the first turn's chunks (>= OKC_SP_SYMM_MIN_TOKENS rows) stay on their paths
    n = rows * cols
    if n * 2 > _sw.EXT_AG_MAX_BYTES:
        return None
    st = _rs_buffer(group_name, torch.device("cuda", torch.cuda.current_device()))
    if st is None:
        return None
    return st[0][:n].view(rows, cols)


def rs_finish(target: torch.Tensor, group_name: str, world: int, rank: int) -> torch.Tensor:
    """After the partials are in rs_target's view: barrier, then this rank's rows of the sum (fp32 in the switch) as a
    new [rows / world, cols] tensor."""
    buf, _, hdl, mc = _RS_STATE[group_name]
    rows, cols = target.shape
    m = rows // world * cols
    out = torch.empty((rows // world, cols), dtype=torch.bfloat16, device=target.device)
    hdl.barrier(channel=0)
    _rs_ext().okc_rsld_run(mc, out, rank * m)
    if _sw.EXT_RS_FENCE:
        hdl.barrier(channel=1)   # every peer done reading this rank's buffer before its next write
    return out


def reduce_scatter(full: torch.Tensor, group_name: str, world: int, rank: int):
    """[rows, cols] bf16 partials -> this rank's [rows / world, cols] rows of the sum via the RS buffer (one copy in),
    or None (the caller's NCCL reduce-scatter)."""
    if full.dtype is not torch.bfloat16 or full.dim() != 2:
        return None
    t = rs_target(full.shape[0], full.shape[1], group_name, world)
    if t is None:
        return None
    t.copy_(full)
    return rs_finish(t, group_name, world, rank)
