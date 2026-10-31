"""pg_route_f (OKC_SW_PGR): pg_route's small front end (_cuda_src_pg.py) on a shorter, leaner chain, for verify calls of
<= 2048 routes on pg's 64-token-block build (cuda_ext.pgr_ext: -DPGN_XNT=64 -DPGN_WSET=24, pg_narrow64_ext's geometry).
Same outputs as pg_route (local_ids, sorted_ids, block_experts, live; the replica decision is rep_owner's with passes 0).
The served kernel is one CTA of 32 warps that repeats work in every warp (two match.any passes, scattered sorted_ids
stores, a sentinel pass over every block); here, 16 warps:
  - one route per lane per round; a lane's rank among its expert's routes in the round comes from a per-warp bit row
    (one shared OR per route); rounds chain through the warp's count row
  - one thread per expert turns the 16 warp rows into bases and totals while one warp runs the owner loop (branch-free,
    from the active set) and then lays out the slots' blocks for every warp
  - sorted_ids is staged in shared memory, prefilled with the sentinel and written out coalesced; one warp per live run
Keys are GLOBAL experts: each slot holds one expert (rep_table / the home map), so an expert's routes in route order are
its slot's.
"""

_CUDA_SRC_PGR = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace pgr {
#define DINL __device__ __forceinline__
DINL void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DINL void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

constexpr int XNT = PGN_XNT, NW = 16, NT = NW * 32, MAXR = 2048, MAXG = MAXR / NT, NE = 64;
constexpr int MAXS = (NE + MAXR / XNT) * XNT;   // sorted_ids entries at most
// shared memory (ints): per-warp count rows (-> bases) and bit rows, sorted_ids, per-expert / per-slot tables
constexpr int SM_BITS = NW * 256, SM_SORT = 2 * NW * 256, SM_MAP = SM_SORT + MAXS, SM_TAB = SM_MAP + 256,
              SM_ORD = SM_TAB + 256, SM_SCNT = SM_ORD + 256, SM_DROP = SM_SCNT + NE, SM_LAY = SM_DROP + NE,
              SM_ACT = SM_LAY + NE, SM_HM = SM_ACT + 8, SM_NB = SM_HM + 64, SM_INTS = SM_NB + 4;
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
// blocks of class cls in a slot of n blocks whose last block has width code cd (its full blocks are class FULL)
DINL int nclass(int cls, int n, int cd) { return n == 0 ? 0 : (cls == FULL ? n - 1 : 0) + (cd == cls ? 1 : 0); }
DINL int scan32(int x, unsigned lt) {   // exclusive prefix of 0 <= x < 64 over the warp's lanes: one ballot per bit
    int s = 0;
#pragma unroll
    for (int b = 0; b < 6; b++) s += __popc(__ballot_sync(0xFFFFFFFFu, (x >> b) & 1) & lt) << b;
    return s;
}

__global__ void __launch_bounds__(NT) pg_route_f(const int* __restrict__ topk_ids, const int* __restrict__ emap,
                                                 int n_global, int numel, int wmask, int nl,
                                                 int* __restrict__ local_ids, int* __restrict__ sorted_ids,
                                                 int* __restrict__ block_experts, int* __restrict__ live,
                                                 const int* __restrict__ rtab, int n_rep, int me) {
    extern __shared__ __align__(16) int sm[];
    int(*wcg)[256] = reinterpret_cast<int(*)[256]>(sm);   // per (warp, expert): routes -> base (earlier warps')
    unsigned(*bits)[256] = reinterpret_cast<unsigned(*)[256]>(sm + SM_BITS);   // this round's lanes per expert
    int *s_sorted = sm + SM_SORT, *s_map = sm + SM_MAP, *s_tab = sm + SM_TAB, *s_ord = sm + SM_ORD;
    int *s_scnt = sm + SM_SCNT, *s_drop = sm + SM_DROP, *s_lay = sm + SM_LAY, *s_nb = sm + SM_NB;
    unsigned *s_act = reinterpret_cast<unsigned*>(sm + SM_ACT), *s_hm = reinterpret_cast<unsigned*>(sm + SM_HM);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const unsigned lt = (1u << lane) - 1u;
    const int ng = (numel + NT - 1) / NT, r0 = w * ng * 32, gmax = rtab ? 256 : n_global;
    const int4 z = make_int4(0, 0, 0, 0), f = make_int4(numel, numel, numel, numel);
    reinterpret_cast<int4*>(wcg[w])[lane] = z;   // each warp's own rows: no barrier before the counting
    reinterpret_cast<int4*>(wcg[w])[32 + lane] = z;
    reinterpret_cast<int4*>(bits[w])[lane] = z;
    reinterpret_cast<int4*>(bits[w])[32 + lane] = z;
    __syncwarp();
    for (int i = tid; i < (nl + (numel + XNT - 1) / XNT) * XNT / 4; i += NT)   // every slot past a block's routes
        reinterpret_cast<int4*>(s_sorted)[i] = f;
    if (tid < NE) { s_scnt[tid] = 0; s_drop[tid] = 0; }
    if (tid < 8) s_act[tid] = 0u;
    pdl_wait();
    pdl_trigger();
    int gk[MAXG], rk[MAXG];
#pragma unroll
    for (int k = 0; k < MAXG; k++) {
        const int r = r0 + k * 32 + lane;
        gk[k] = k < ng && r < numel ? topk_ids[r] : -1;
    }
    int v = 0, oe = 0;
    if (tid < 256) v = rtab ? rtab[tid] : tid < n_global ? (emap ? emap[tid] : tid) : -1;
    if (rtab && tid < n_rep) s_ord[tid] = rtab[256 + tid];
    if (rtab && w == NW - 1 && lane < n_rep) oe = rtab[256 + lane];   // the owner warp's first 32, ahead
    if (tid < 256) {
        s_tab[tid] = v;
        if (rtab) {   // warp w < 8 = experts 32 w + lane: the non-replicated ones per home rank
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const unsigned m = __ballot_sync(0xFFFFFFFFu, !((v >> 11) & 1) && ((v >> 8) & 7) == r);
                if (lane == r) s_hm[8 * r + w] = m;
            }
        }
    }
    __syncthreads();   // the clears above, for every warp (the loads are in flight)
#pragma unroll
    for (int k = 0; k < MAXG; k++) {   // round k: rank among the expert's routes of this warp (route order k, lane)
        if (k < ng) {
            const int g = (unsigned)gk[k] < (unsigned)gmax ? gk[k] : -1;
            gk[k] = g;
            if (g >= 0) atomicOr(&bits[w][g], 1u << lane);
            __syncwarp();
            const unsigned m = g >= 0 ? bits[w][g] : 0u;
            const int c0 = g >= 0 ? wcg[w][g] : 0;
            rk[k] = c0 + __popc(m & lt);
            __syncwarp();
            if (g >= 0 && (m & lt) == 0) { wcg[w][g] = c0 + __popc(m); bits[w][g] = 0u; }
            __syncwarp();
        }
    }
    if (rtab) {   // the active experts (bit per expert): this warp's words into the CTA's
        unsigned mine = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned m = __ballot_sync(0xFFFFFFFFu, wcg[w][32 * i + lane] > 0);
            if (lane == i) mine = m;
        }
        if (lane < 8 && mine) atomicOr(&s_act[lane], mine);
    }
    __syncthreads();
    if (tid < 256) {   // expert tid: warp bases, total, its slot count (a replicated one's is dropped by s_drop)
        int c[NW];
#pragma unroll
        for (int x = 0; x < NW; x++) c[x] = wcg[x][tid];
        int run = 0;
#pragma unroll
        for (int x = 0; x < NW; x++) {
            wcg[x][tid] = run;
            run += c[x];
        }
        const bool rep = rtab && ((v >> 11) & 1);
        const int sl = rtab ? ((v & 0xFF) != 0xFF ? (v & 0xFF) : -1) : v;
        const int e = (unsigned)sl < (unsigned)nl && (!rtab || rep || ((v >> 8) & 7) == me) ? sl : -1;
        if (!rep) s_map[tid] = e;
        if (e >= 0) s_scnt[e] = run;
        asm volatile("bar.arrive 1, 288;" ::: "memory");   // to the layout warp
    } else if (w == NW - 1) {
        if (rtab) {   // rep_owner's decision (passes 0): load = active non-replicated experts per home rank; each
            // active replicated expert in order to the less loaded holder (tie: home)
            const unsigned a = s_act[lane & 7];   // lane l: word l & 7 of the active set
            const int q = lane >> 2, i0 = 2 * (lane & 3);   // lane: rank q, words i0, i0 + 1
            int ld = __popc(__shfl_sync(0xFFFFFFFFu, a, i0) & s_hm[8 * q + i0]) +
                     __popc(__shfl_sync(0xFFFFFFFFu, a, i0 + 1) & s_hm[8 * q + i0 + 1]);
            ld += __shfl_xor_sync(0xFFFFFFFFu, ld, 1);
            ld += __shfl_xor_sync(0xFFFFFFFFu, ld, 2);
            unsigned long long L0 = 0, L1 = 0;   // 16 bits per rank (as rep_owner)
#pragma unroll
            for (int r = 0; r < 4; r++) {
                L0 |= (unsigned long long)__shfl_sync(0xFFFFFFFFu, ld, 4 * r) << (16 * r);
                L1 |= (unsigned long long)__shfl_sync(0xFFFFFFFFu, ld, 16 + 4 * r) << (16 * r);
            }
            for (int j0 = 0; j0 < n_rep; j0 += 32) {
                const int j = j0 + lane, e = j0 == 0 ? oe & 255 : j < n_rep ? s_ord[j] : 0, ve = s_tab[e];
                const int act = (__shfl_sync(0xFFFFFFFFu, a, (e >> 5) & 7) >> (e & 31)) & 1;
                const int info = (j < n_rep ? act : 0) | (((ve >> 8) & 7) << 1) | (((ve >> 12) & 7) << 4);
                unsigned pr = 0;   // bit u: expert j0 + u went to its replica rank
                const int nu = min(32, n_rep - j0);
#pragma unroll 8
                for (int u = 0; u < nu; u++) {   // the same steps in every lane, branch-free
                    const int fu = __shfl_sync(0xFFFFFFFFu, info, u), h = (fu >> 1) & 7, r = (fu >> 4) & 7;
                    const unsigned lh = (unsigned)(((h & 4) ? L1 : L0) >> (16 * (h & 3))) & 0xFFFFu;
                    const unsigned lr = (unsigned)(((r & 4) ? L1 : L0) >> (16 * (r & 3))) & 0xFFFFu;
                    const bool p = lr < lh;
                    const int pick = p ? r : h;
                    const unsigned long long add = (unsigned long long)(fu & 1) << (16 * (pick & 3));
                    L0 += (pick & 4) ? 0ull : add;
                    L1 += (pick & 4) ? add : 0ull;
                    pr |= (unsigned)(p && (fu & 1)) << u;
                }
                if (info & 1) {
                    const int pick = (pr >> lane) & 1 ? (info >> 4) & 7 : (info >> 1) & 7, sl = ve & 0xFF;
                    const bool held = sl != 0xFF && sl < nl;
                    s_map[e] = held && pick == me ? sl : -1;
                    if (held && pick != me) s_drop[sl] = 1;
                }
            }
        }
        asm volatile("bar.sync 1, 288;" ::: "memory");   // the slot counts are in
        // slot layout (lane l: slots l and 32 + l): first block | blocks << 10 | last block's width code << 20
        int nb = 0;
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            const int x = 32 * hf + lane, c = s_drop[x] ? 0 : s_scnt[x], n = (c + XNT - 1) / XNT;
            s_lay[x] = (nb + scan32(n, lt)) | (n << 10) | (route_code(n > 0 ? c - (n - 1) * XNT : 0, wmask) << 20);
            nb += (int)__reduce_add_sync(0xFFFFFFFFu, (unsigned)n);
        }
        if (lane == 0) *s_nb = nb;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < MAXG; k++) {   // place: slot e's block start + earlier warps' routes + the in-warp rank
        if (k < ng) {
            const int g = gk[k], e = g >= 0 ? s_map[g] : -1, r = r0 + k * 32 + lane;
            if (r < numel) local_ids[r] = e;
            if (e >= 0) s_sorted[(s_lay[e] & 0x3FF) * XNT + wcg[w][g] + rk[k]] = r;
        }
    }
    if (w < 2 * NCLS) {   // live run w = class route_cls(w / 2), half w % 2: runs back to back, slots ascending
        const int y0 = s_lay[lane], y1 = s_lay[32 + lane];
        const int n0 = (y0 >> 10) & 0x3FF, n1 = (y1 >> 10) & 0x3FF, d0 = y0 >> 20, d1 = y1 >> 20;
        int before = 0;
#pragma unroll
        for (int j = 0; j < 2 * NCLS; j++)
            if (j < w) before += nclass(route_cls(j >> 1), j & 1 ? n1 : n0, j & 1 ? d1 : d0);
        const int y = w & 1 ? y1 : y0, cls = route_cls(w >> 1), n = (y >> 10) & 0x3FF, cd = y >> 20, b = y & 0x3FF;
        int pos = (int)__reduce_add_sync(0xFFFFFFFFu, (unsigned)before) + scan32(nclass(cls, n, cd), lt);
        if (cls == FULL) for (int j = 0; j < n - 1; j++) live[1 + pos++] = (b + j) | (FULL << 20);
        if (n > 0 && cd == cls) live[1 + pos] = (b + n - 1) | (cd << 20);
    } else if (w == 2 * NCLS) {
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            const int y = s_lay[32 * hf + lane], n = (y >> 10) & 0x3FF;
            for (int j = 0; j < n; j++) block_experts[(y & 0x3FF) + j] = 32 * hf + lane;
        }
        if (lane == 0) live[0] = *s_nb;
    }
    __syncthreads();
    const int nb = *s_nb;
    for (int i = tid; i < nb * XNT / 4; i += NT)
        reinterpret_cast<int4*>(sorted_ids)[i] = reinterpret_cast<const int4*>(s_sorted)[i];
}
}  // namespace pgr

// pg's route (_cuda_src_pg.py) on pg_route_f: same arguments and outputs; numel <= 2048, verify balancer only (me < 8)
void route(torch::Tensor topk_ids, torch::Tensor emap, torch::Tensor rtab, torch::Tensor local_ids,
           torch::Tensor sorted_ids, torch::Tensor block_experts, torch::Tensor live, int64_t numel, int64_t n_global,
           int64_t nl, int64_t pdl, int64_t me) {
    TORCH_CHECK(numel <= pgr::MAXR && topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous() && nl >= 1 &&
                nl <= pgr::NE && n_global <= 256, "pg route_f: int32 ids, at most 2048 routes, 1..64 local slots");
    TORCH_CHECK(sorted_ids.numel() >= (nl + (numel + pgr::XNT - 1) / pgr::XNT) * pgr::XNT &&
                live.numel() >= nl + (numel + pgr::XNT - 1) / pgr::XNT + 1 &&
                reinterpret_cast<uintptr_t>(sorted_ids.data_ptr<int>()) % 16 == 0, "pg route_f: output sizes");
    const int* em = emap.numel() > 0 ? emap.data_ptr<int>() : nullptr;
    const int* rt = rtab.numel() > 0 ? rtab.data_ptr<int>() : nullptr;
    TORCH_CHECK(!rt || (rtab.scalar_type() == at::kInt && rtab.numel() >= 256 && rtab.numel() <= 512 && me >= 0 &&
                        me < 8), "pg route_f: replica table (verify balancer)");
    C10_CUDA_CHECK(cudaFuncSetAttribute(pgr::pg_route_f, cudaFuncAttributeMaxDynamicSharedMemorySize, pgr::SM_INTS * 4));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(pgr::NT);
    cfg.dynamicSmemBytes = pgr::SM_INTS * 4;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = la;
    cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, pgr::pg_route_f, topk_ids.data_ptr<int>(), em, (int)n_global, (int)numel,
                                      (int)PGN_WSET, (int)nl, local_ids.data_ptr<int>(), sorted_ids.data_ptr<int>(),
                                      block_experts.data_ptr<int>(), live.data_ptr<int>(), rt,
                                      rt ? (int)rtab.numel() - 256 : 0, (int)me));
}
"""
