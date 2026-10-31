"""pg_route_f (OKC_SW_PGR): pg_route's small front end (_cuda_src_pg.py) on a short dependency chain, for verify calls of
<= 2048 routes on pg's 64-token-block build (cuda_ext.pgr_ext: -DPGN_XNT=64 -DPGN_WSET=24, pg_narrow64_ext's geometry).
Same outputs as pg_route (local_ids, sorted_ids, block_experts, live; the replica balancer's decision is rep_owner's with
passes 0): the served kernel spends ~12K cycles of one CTA in nine barrier phases (two id loads, a smem-atomic count,
a serial owner loop over load atomics, two per-warp count passes, a block / live pass behind two more barriers). Here
one pass counts each warp's contiguous run per GLOBAL expert and ranks its routes in it; one thread per expert turns
the 16 warp rows into bases and totals (and the static map / slot counts / balancer loads); one thread runs the owner
loop; every warp then derives the slot layout in registers and writes its part. Each slot holds one expert (rep_table /
the home map), so an expert's routes in route order are its slot's.
"""

_CUDA_SRC_PGR = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace pgr {
#define DINL __device__ __forceinline__
DINL void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DINL void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

constexpr int XNT = PGN_XNT, NW = 16, NT = NW * 32, MAXR = 2048, MAXG = MAXR / NT, NE = 64;
// pg_route's width codes and live classes (XNT <= 128: 16-token steps)
constexpr bool N16 = XNT <= 128;
constexpr int NCLS = N16 ? 6 : 4, FULL = N16 ? XNT / 16 - 1 : 3;
DINL int route_code(int v, int wmask) {
    int code = FULL;
    if (N16) {
        if ((wmask & 1) && v <= 64) code = 3;
        if ((wmask & 8) && v <= 32) code = 1;
        if ((wmask & 16) && v <= 16) code = 0;
    } else {
        if ((wmask & 4) && v <= 192) code = 2;
        if ((wmask & 2) && v <= 128) code = 1;
        if ((wmask & 1) && v <= 64) code = 0;
    }
    return code;
}
DINL int route_cls(int i) { return N16 ? (i < 4 ? 15 - 4 * i : 1 - (i - 4)) : FULL - i; }
DINL int nclass(int cls, int n, int cd) { return n == 0 ? 0 : (cls == FULL ? n - 1 : 0) + (cd == cls ? 1 : 0); }

__global__ void __launch_bounds__(NT) pg_route_f(const int* __restrict__ topk_ids, const int* __restrict__ emap,
                                                 int n_global, int numel, int wmask, int nl,
                                                 int* __restrict__ local_ids, int* __restrict__ sorted_ids,
                                                 int* __restrict__ block_experts, int* __restrict__ live,
                                                 const int* __restrict__ rtab, int n_rep, int me) {
    __shared__ __align__(16) int wcg[NW][256];   // per (warp, expert): count -> base (routes of earlier warps)
    __shared__ int s_cnt[256], s_map[256], s_tab[256], s_ord[256], s_scnt[NE], s_load[8];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const unsigned lt = (1u << lane) - 1u;
    reinterpret_cast<int4*>(wcg[w])[lane] = make_int4(0, 0, 0, 0);   // each warp clears its own row
    reinterpret_cast<int4*>(wcg[w])[32 + lane] = make_int4(0, 0, 0, 0);
    __syncwarp();
    if (tid < NE) s_scnt[tid] = 0;
    if (tid < 8) s_load[tid] = 0;
    pdl_wait();
    pdl_trigger();
    // warp w: routes r0 .. r0 + 32 ng - 1, key = global expert (-1: none)
    const int ng = (numel + NT - 1) / NT, r0 = w * ng * 32, gmax = rtab ? 256 : n_global;
    int gk[MAXG], rk[MAXG];
#pragma unroll
    for (int k = 0; k < MAXG; k++) {
        const int r = r0 + k * 32 + lane;
        gk[k] = k < ng && r < numel ? topk_ids[r] : -1;
    }
    int v = 0;
    if (tid < 256) v = rtab ? rtab[tid] : tid < n_global ? (emap ? emap[tid] : tid) : -1;
    if (rtab && tid < n_rep) s_ord[tid] = rtab[256 + tid];
#pragma unroll
    for (int k = 0; k < MAXG; k++) {   // in-warp rank of each route among its expert's (route order: k, lane)
        if (k < ng) {
            const int g = (unsigned)gk[k] < (unsigned)gmax ? gk[k] : -1;
            gk[k] = g;
            const unsigned peers = __match_any_sync(0xFFFFFFFFu, g);
            const int prev = k == 0 || g < 0 ? 0 : wcg[w][g];
            rk[k] = prev + __popc(peers & lt);
            __syncwarp();
            if (g >= 0 && (peers & lt) == 0) wcg[w][g] = prev + __popc(peers);
            __syncwarp();
        }
    }
    if (tid < 256) s_tab[tid] = v;
    __syncthreads();
    if (tid < 256) {   // expert tid: warp bases, total, its static slot, the balancer's loads
        int run = 0;
#pragma unroll
        for (int x = 0; x < NW; x++) {
            const int c = wcg[x][tid];
            wcg[x][tid] = run;
            run += c;
        }
        s_cnt[tid] = run;
        if (!rtab || !((v >> 11) & 1)) {
            const int sl = rtab ? (((v >> 8) & 7) == me && (v & 0xFF) != 0xFF ? (v & 0xFF) : -1) : v;
            const int e = (unsigned)sl < (unsigned)nl ? sl : -1;
            s_map[tid] = e;
            if (e >= 0) s_scnt[e] = run;
            if (rtab && run > 0) atomicAdd(&s_load[(v >> 8) & 7], 1);
        }
    }
    __syncthreads();
    if (rtab) {   // rep_owner's loop (passes 0): each active replicated expert in order to the less loaded holder
        if (tid == 0) {
            unsigned long long L0 = 0, L1 = 0;
#pragma unroll
            for (int r = 0; r < 4; r++) {
                L0 |= (unsigned long long)s_load[r] << (16 * r);
                L1 |= (unsigned long long)s_load[4 + r] << (16 * r);
            }
            for (int j0 = 0; j0 < n_rep; j0 += 8) {
                int eb[8], cb[8], vb[8];
#pragma unroll
                for (int u = 0; u < 8; u++) eb[u] = j0 + u < n_rep ? s_ord[j0 + u] : 0;
#pragma unroll
                for (int u = 0; u < 8; u++) {
                    cb[u] = j0 + u < n_rep ? s_cnt[eb[u]] : 0;
                    vb[u] = s_tab[eb[u]];
                }
#pragma unroll
                for (int u = 0; u < 8; u++) {
                    const int e = eb[u], c = cb[u];
                    if (!c) continue;
                    const int ve = vb[u], h = (ve >> 8) & 7, r = (ve >> 12) & 7;
                    const unsigned lh = (unsigned)(((h & 4) ? L1 : L0) >> (16 * (h & 3))) & 0xFFFFu;
                    const unsigned lr = (unsigned)(((r & 4) ? L1 : L0) >> (16 * (r & 3))) & 0xFFFFu;
                    const int pick = lr < lh ? r : h;
                    const unsigned long long add = 1ull << (16 * (pick & 3));
                    if (pick & 4) L1 += add; else L0 += add;
                    const int sl = pick == me && (ve & 0xFF) != 0xFF && (ve & 0xFF) < nl ? (ve & 0xFF) : -1;
                    s_map[e] = sl;
                    if (sl >= 0) s_scnt[sl] = c;
                }
            }
        }
        __syncthreads();
    }
    // slot layout in every warp's registers (lane l: slots l and 32 + l): blocks, first block, last block's width code
    int sc[2], nbx[2], b0[2], code[2], nb = 0;
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
        sc[hf] = s_scnt[32 * hf + lane];
        nbx[hf] = (sc[hf] + XNT - 1) / XNT;
        int bs = nbx[hf];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int t = __shfl_up_sync(0xFFFFFFFFu, bs, o);
            if (lane >= o) bs += t;
        }
        b0[hf] = nb + bs - nbx[hf];
        nb += __shfl_sync(0xFFFFFFFFu, bs, 31);
        code[hf] = route_code(nbx[hf] > 0 ? sc[hf] - (nbx[hf] - 1) * XNT : 0, wmask);
    }
#pragma unroll
    for (int k = 0; k < MAXG; k++) {   // place: slot e's block start + earlier warps' routes + the in-warp rank
        if (k < ng) {
            const int g = gk[k], e = g >= 0 ? s_map[g] : -1, r = r0 + k * 32 + lane;
            const int s0 = __shfl_sync(0xFFFFFFFFu, b0[0], e & 31), s1 = __shfl_sync(0xFFFFFFFFu, b0[1], e & 31);
            if (r < numel) local_ids[r] = e;
            if (e >= 0) sorted_ids[(e < 32 ? s0 : s1) * XNT + wcg[w][g] + rk[k]] = r;
        }
    }
    for (int x = w; x < NE; x += NW) {   // sentinels: slot x's last block past its routes
        const int hf = x >> 5;
        const int c = __shfl_sync(0xFFFFFFFFu, sc[hf], x & 31), n = __shfl_sync(0xFFFFFFFFu, nbx[hf], x & 31);
        const int b = __shfl_sync(0xFFFFFFFFu, b0[hf], x & 31);
        if (n > 0)
            for (int j = c - (n - 1) * XNT + lane; j < XNT; j += 32) sorted_ids[(b + n - 1) * XNT + j] = numel;
    }
    if (w < 2 * NCLS) {   // live: classes widest first, halves, slots ascending (pg_route's order)
        int before = 0;
#pragma unroll
        for (int j = 0; j < 2 * NCLS; j++)
            if (j < w) before += nclass(route_cls(j >> 1), nbx[j & 1], code[j & 1]);
        const int hf = w & 1, cls = route_cls(w >> 1), n = nbx[hf], kc = nclass(cls, n, code[hf]);
        int inc = kc;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int t = __shfl_up_sync(0xFFFFFFFFu, inc, o);
            if (lane >= o) inc += t;
        }
        int pos = (int)__reduce_add_sync(0xFFFFFFFFu, (unsigned)before) + inc - kc;
        if (cls == FULL) for (int j = 0; j < n - 1; j++) live[1 + pos++] = (b0[hf] + j) | (FULL << 20);
        if (n > 0 && code[hf] == cls) live[1 + pos] = (b0[hf] + n - 1) | (code[hf] << 20);
    } else if (w == 2 * NCLS) {
#pragma unroll
        for (int hf = 0; hf < 2; hf++)
            for (int j = 0; j < nbx[hf]; j++) block_experts[b0[hf] + j] = 32 * hf + lane;
        if (lane == 0) live[0] = nb;
    }
}
}  // namespace pgr

// pg's route (_cuda_src_pg.py) on pg_route_f: same arguments and outputs; numel <= 2048, verify balancer only (me < 8)
void route(torch::Tensor topk_ids, torch::Tensor emap, torch::Tensor rtab, torch::Tensor local_ids,
           torch::Tensor sorted_ids, torch::Tensor block_experts, torch::Tensor live, int64_t numel, int64_t n_global,
           int64_t nl, int64_t pdl, int64_t me) {
    TORCH_CHECK(numel <= pgr::MAXR && topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous() && nl >= 1 &&
                nl <= pgr::NE && n_global <= 256, "pg route_f: int32 ids, at most 2048 routes, 1..64 local slots");
    TORCH_CHECK(sorted_ids.numel() >= (nl + (numel + pgr::XNT - 1) / pgr::XNT) * pgr::XNT &&
                live.numel() >= nl + (numel + pgr::XNT - 1) / pgr::XNT + 1, "pg route_f: output sizes");
    const int* em = emap.numel() > 0 ? emap.data_ptr<int>() : nullptr;
    const int* rt = rtab.numel() > 0 ? rtab.data_ptr<int>() : nullptr;
    TORCH_CHECK(!rt || (rtab.scalar_type() == at::kInt && rtab.numel() >= 256 && rtab.numel() <= 512 && me >= 0 &&
                        me < 8), "pg route_f: replica table (verify balancer)");
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(pgr::NT);
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, pgr::pg_route_f, topk_ids.data_ptr<int>(), em, (int)n_global, (int)numel,
                                      (int)PGN_WSET, (int)nl, local_ids.data_ptr<int>(), sorted_ids.data_ptr<int>(),
                                      block_experts.data_ptr<int>(), live.data_ptr<int>(), rt,
                                      rt ? (int)rtab.numel() - 256 : 0, (int)me));
}
"""
