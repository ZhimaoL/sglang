"""CUDA source for the pg grouped-GEMM MoE pair (prefill-gemm graft, BF16 end to end).

Kernels pg_tiles / pg_gather / pg_gemm (namespace pgk) copied from the pg_kernel module of the prefill
bank kernels/muse_prefill_layer/plt_c3.py (sha256 2d1fad0d0b7c42f1). Edits: five stale comments that still said
fp16 / 4608 B (the code is bf16 with WSB = 5120 B blocks) and an unused cuda_fp16.h include dropped.
Not carried over: the pgkw P2 kernel, the pg_n4 64-token probe kernel and the probe / split
configurations (none is on the bank's default cfg 350 path).
Entry points: tiles (live block list), gather (bf16 x rows -> bf16 swizzled token images), gg1
(gate+up GEMM, silu*mul fused, bf16 intermediate), gg2 (down GEMM, scatter by sorted_ids, bf16 rows).
Weights stay packed NVFP4 (pack_wsf in pg_moe.py: e2m1 codes reordered, e4m3 block scales stored
exactly as bf16) and are decoded in registers.

Narrow build (cuda_ext.pg_narrow_ext, -DPGN_WSET=17 -DPGN_NST=7: the #ifdef PGN_WSET blocks) for verify steps and
follow-up extends; -DPGN_XNT=64 -DPGN_WSET=24 (cuda_ext.pg_narrow64_ext) is the same build on 64-token blocks with widths
16 / 32 / 64 for verify T up to OKC_PG64_MAX_T (bs 17-32: most active experts hold <= 64 tokens and the 17-32-token
ones get n32 items instead of n64). At mid T most local experts get few tokens (rank 0 at T 1024: p10 / p50 / p90 = 2 / 16 / 96) and a
pg item costs about the same whatever share of its token columns is real, so it runs 128-token blocks (XNT 128) with
item widths 16 / 64 / 128 (PGN_WSET 17; ptxas serializes the wgmmas beyond three width loops, C7512) and PGN_NST = 7
ring slots (the 16 KB X stage). Same weights and accumulation per output element: real tokens' outputs are
bit-identical to the shipped build; only padded columns are skipped. Width codes become nv / 16 - 1 (wmask bits 0 = 64,
1 = 128, 2 = 192, 3 = 32, 4 = 16); items of <= 64 tokens stage two k-tiles per ring slot.
"""

from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_rep import _CUDA_SRC_REP


def _wgmma_asm(n):
    """m64nNk16 f32 += bf16(A regs, '+r') x bf16(B desc) with a runtime scale-d predicate."""
    nacc = n // 2
    dlist = ",".join(f"%{i}" for i in range(nacc))
    alist = ",".join(f"%{nacc + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
    return (
        f"DINL void wgmma_n{n}(float* d, unsigned* a, unsigned long long db, int accum) {{\n"
        f"    asm volatile(\n"
        f"        \"{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nacc + 5}, 0;\\n\"\n"
        f"        \"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16\\n\"\n"
        f"        \"{{{dlist}}},\"\n"
        f"        \"{{{alist}}}, %{nacc + 4}, p, 1, 1, 0;\\n}}\\n\"\n"
        f"        : {outs},\n"
        f"          \"+r\"(a[0]), \"+r\"(a[1]), \"+r\"(a[2]), \"+r\"(a[3])\n"
        f"        : \"l\"(db), \"r\"(accum));\n"
        f"}}\n")


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace pgk {
#define DINL __device__ __forceinline__

DINL unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

// Programmatic dependent launch (no-ops unless the launch carries the programmatic-serialization attribute):
// every kernel waits before its first read of a predecessor's output and before its first global write, and
// lets its dependent launch right after (the dependent waits too, so an early trigger only overlaps prologues).
DINL void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DINL void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

DINL void mbar_init(unsigned bar, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
DINL void mbar_arrive(unsigned bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(bar) : "memory");
}
DINL void mbar_expect_tx(unsigned bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
DINL void mbar_wait(unsigned bar, unsigned phase) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}\n" :: "r"(bar), "r"(phase) : "memory");
}
DINL void bulk_g2s(unsigned sdst, const void* gsrc, unsigned bytes, unsigned bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sdst), "l"(gsrc), "r"(bytes), "r"(bar) : "memory");
}
DINL void bulk_s2g(void* gdst, unsigned ssrc, unsigned bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(gdst), "r"(ssrc), "r"(bytes) : "memory");
}
DINL void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
DINL void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
DINL void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
DINL void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
DINL void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
template <int N> DINL void reg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N)); }
template <int N> DINL void reg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N)); }

DINL void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DINL void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> DINL void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N)); }

DINL void fence_u(unsigned& x) { asm volatile("" : "+r"(x)); }
DINL void fence_bank(unsigned* b) {
#pragma unroll
    for (int i = 0; i < 16; i++) fence_u(b[i]);
}
DINL void fence_acc(float* d, int n) {
#pragma unroll
    for (int i = 0; i < n; i++) asm volatile("" : "+f"(d[i]));
}

DINL __nv_bfloat162 u32b2(unsigned v) { return *reinterpret_cast<__nv_bfloat162*>(&v); }
DINL unsigned b2u32(__nv_bfloat162 v) { return *reinterpret_cast<unsigned*>(&v); }

// bf16 weight word (pack_wsf, DQ3 placement _B3): codes c0..c7 (A-fragment order: c0 / c4 -> register 0 lo / hi,
// c1 / c5 -> 1, c2 / c6 -> 2, c3 / c7 -> 3) with magnitude bits m and sign bit s at:
//   c0 m 6-8      s 15 | c4 m 22-24    s 31 -> o0 = t & 0x81C081C0                              (in place)
//   c1 m 28-30    s 5  | c5 m 12-14    s 21 -> o1 = rotl(t, 10) & 0x81C081C0
//   c2 m 25-27    s 2  | c6 m 9-11     s 18 -> o2 = rotl(t, 13) & 0x81C081C0
//   c3 m 4, 0, 1  s 3  | c7 m 20,16,17 s 19 -> o3 = ((t & 0x001B001B) * 0x1084) & 0x81C081C0  (copies << 2, 7, 12)
// = e2m1 x 2^-126 as bf16 (magnitude to bf16 bits 6-8, sign to 15; e = 0, m = 1 is the subnormal 2^-127): 8 ops
// (5 LOP3 + 2 SHF + 1 IMAD; the kernel lane's rd_pack_codes_k dq=3 bit layout; the bank's v7 layout took 11).
DINL void dq3(unsigned t, __nv_bfloat162 d[4]) {
    d[0] = u32b2(t & 0x81C081C0u);
    d[1] = u32b2(__funnelshift_l(t, t, 10) & 0x81C081C0u);
    d[2] = u32b2(__funnelshift_l(t, t, 13) & 0x81C081C0u);
    d[3] = u32b2(((t & 0x001B001Bu) * 0x1084u) & 0x81C081C0u);
}

#ifdef PGN_WSET
DINL void stsm_x2_trans(unsigned addr, unsigned r0, unsigned r1) {
    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};"
                 :: "r"(addr), "r"(r0), "r"(r1) : "memory");
}
#endif  // PGN_WSET
DINL void stsm_x4_trans(unsigned addr, unsigned r0, unsigned r1, unsigned r2, unsigned r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

@@WGMMA@@

#ifdef PGN_WSET
@@WGMMA_N@@
// ------------------------------------------------------------------ geometry
#ifndef PGN_XNT
#define PGN_XNT 128
#endif
#if PGN_XNT == 64 && !defined(PG_M1RS)
#define PG_M1_STORE 1                    // M1 (OKC_SW_M1) runs on the 64-block build only (moe._moe_pg_small: bm == 64)
#endif
constexpr int XNT = PGN_XNT;             // tokens per block (narrow build: the widest item)
#else   // PGN_WSET
// ------------------------------------------------------------------ geometry
constexpr int XNT = 256;                 // tokens per tile (wgmma n)
#endif  // PGN_WSET
constexpr int BK  = 64;                  // k per stage
constexpr int NKT = 64;                  // k-tiles (K = 4096 for both GEMMs)
constexpr int XB  = XNT * BK * 2;        // 32768: one k-tile token image [256 t][128 B], 128B-swizzled K-major
constexpr int WSB = 5120;                // merged W(4096) + bf16 scale pairs (1024) per (wt, kt)
#ifdef PGN_WSET
constexpr int SB  = (16384 + 2 * WSB + 1023) / 1024 * 1024;   // PAIR stage: X 2 x 8 KB + W|sf 2 x WSB
#else   // PGN_WSET
constexpr int SB  = 37888;               // stage stride: X 32768 + W|sf 5120 (a whole number of 1024 B atoms)
#endif  // PGN_WSET
constexpr int PREGS = 40, CREGS = 232;   // setmaxnreg targets
static_assert(128 * PREGS + 256 * CREGS <= 384 * 168, "setmaxnreg budget exceeds the launch allocation");
constexpr int EPI1 = XB / 2;             // gg1 epilogue per WG: its half of the inter tile [256 t][64 B] (SW64)
constexpr int EPI2_PITCH = 144;          // gg2 epilogue row pitch (64 bf16 + 16 B pad: conflict-free stmatrix)
constexpr int EPI2 = 64 * EPI2_PITCH;    // gg2 epilogue buffer per consumer WG: 64 tokens per pass
constexpr int BAR_BYTES = 256;              // full/empty[NST] mbarriers + 2 item-metadata slots
constexpr int RIDB = 2 * XNT * 4;           // gg2: route ids of the 2 most recent items (TMA'd with the item)
template <int GATED, int MODE> constexpr int smem_bytes(int nst) {
    return nst * SB + BAR_BYTES + (MODE == 4 ? 0 : 2 * (GATED ? EPI1 : EPI2)) + (GATED ? 0 : RIDB);
}

// ------------------------------------------------------------------ live tile list
// live[0] = number of live blocks (local expert < nle, inside n_padded); live[1 + i] = pid_m | (wcode << 20).
// wcode = n/64 - 1 for the block's wgmma width n = the smallest allowed width (wmask bit0: 64, bit1: 128,
// bit2: 192; 256 always) covering its valid-token prefix (moe_align pads each expert's range at its end).
// Order: widest class first, pid_m ascending inside a class (round-robin CTAs then end on cheap narrow items).
__global__ void __launch_bounds__(1024) pg_tiles(const int* __restrict__ block_experts,
                                                 const int* __restrict__ n_padded,
                                                 const int* __restrict__ sorted_ids, int numel,
                                                 int num_pid_m, int bm, int wmask, int nle,
                                                 int* __restrict__ live) {
    __shared__ int wsum[32];
    __shared__ int total_s;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    pdl_wait();
    pdl_trigger();
    const int npm = min(num_pid_m, *n_padded / bm);
    if (tid == 0) total_s = 0;
    __syncthreads();
#ifdef PGN_WSET
    const int CLS[6] = {15, 11, 7, 3, 1, 0};
    const int CLM[6] = {0, 4, 2, 1, 8, 16};
    for (int ci = 0; ci < 6; ci++) {
        const int cls = CLS[ci];
        if (cls != XNT / 16 - 1 && !(wmask & CLM[ci])) continue;   // the full-block class always runs
#else   // PGN_WSET
    for (int cls = 3; cls >= 0; cls--) {
#endif  // PGN_WSET
        for (int base = 0; base < npm; base += 1024) {
            const int pm = base + tid;
            const bool live_blk = pm < npm && (unsigned)block_experts[pm] < (unsigned)nle;
#ifdef PGN_WSET
            int code = XNT / 16 - 1;             // a full block
            if (live_blk && bm == XNT) {
                const int* sid = sorted_ids + (size_t)pm * XNT;
                if ((wmask & 1) && sid[64] >= numel) code = 3;
                if ((wmask & 8) && sid[32] >= numel) code = 1;
                if ((wmask & 16) && sid[16] >= numel) code = 0;
#else   // PGN_WSET
            int code = 3;
            if (live_blk && bm == 256) {
                const int* sid = sorted_ids + (size_t)pm * 256;
                if ((wmask & 4) && sid[192] >= numel) code = 2;
                if ((wmask & 2) && sid[128] >= numel) code = 1;
                if ((wmask & 1) && sid[64] >= numel) code = 0;
#endif  // PGN_WSET
            }
            const bool ok = live_blk && code == cls;
            const unsigned bal = __ballot_sync(0xFFFFFFFFu, ok);
            if (lane == 0) wsum[w] = __popc(bal);
            __syncthreads();
            int off = total_s;
            for (int j = 0; j < w; j++) off += wsum[j];
            if (ok) live[1 + off + __popc(bal & ((1u << lane) - 1u))] = pm | (code << 20);
            __syncthreads();
            if (tid == 0) { int t = 0; for (int j = 0; j < 32; j++) t += wsum[j]; total_s += t; }
            __syncthreads();
        }
    }
    if (tid == 0) live[0] = total_s;
}

// the replica balancer (rep_owner, _cuda_src_rep.py, shared with st): pg_route and pg_rep_assign
@@REP@@
// Counts-in form of the replica balancer for extend-sized calls (rep_align_scan below; rep_owner is the verify / st form): cnt [256] this
// call's routes per global expert, tabs [256] / ord [n_rep] the table (all shared). An active expert weighs (wmode)
// 0: 1, 1: ceil(c / 8), 2: its pg cost in 1/8 SM-us (bm-token blocks of 64 + bm, a tail of 64 + its width class: bm 128
// narrow 16 / 64 / 128, else 64 / 128 / 256); sort: replicated experts by count descending (ties: id ascending, as
// pg_rep_cut), else the table order; each active one goes to the holder with the lower load (tie: home). Leaves
// owner [256] (synced); load [8] / sord [256] shared scratch; loads are 32-bit.
__device__ __forceinline__ int rep_cost(int c, int wmode, int bm) {
    if (c <= 0) return 0;
    if (wmode == 0) return 1;
    if (wmode == 1) return (c + 7) >> 3;
    const int full = c / bm, tail = c - full * bm;
    const int tw = tail <= 0 ? 0 : bm == 128 ? (tail <= 16 ? 16 : tail <= 64 ? 64 : 128)
                                             : (tail <= 64 ? 64 : tail <= 128 ? 128 : 256);
    return full * (64 + bm) + (tail > 0 ? 64 + tw : 0);
}

__device__ __forceinline__ void rep_decide(const int* cnt, const int* tabs, const int* ord, int n_rep, int wmode,
                                           int bm, int sort, int* owner, int* load, int* sord) {
    const int t = threadIdx.x;
    if (t < 256) owner[t] = (tabs[t] >> 8) & 7;
    if (t < 8) load[t] = 0;
    __syncthreads();
    if (t < 256 && !((tabs[t] >> 11) & 1)) {
        const int w = rep_cost(cnt[t], wmode, bm);
        if (w) atomicAdd(&load[(tabs[t] >> 8) & 7], w);
    }
    __syncthreads();
    if (t == 0) {
        for (int j = 0; j < n_rep; j++) {
            const int e = ord[j];
            int k = j;
            if (sort)
                while (k > 0 && (cnt[sord[k - 1]] < cnt[e] || (cnt[sord[k - 1]] == cnt[e] && sord[k - 1] > e))) {
                    sord[k] = sord[k - 1];
                    k--;
                }
            sord[k] = e;
        }
        for (int j = 0; j < n_rep; j++) {
            const int e = sord[j], c = cnt[e];
            if (!c) continue;
            const int v = tabs[e], h = (v >> 8) & 7, r = (v >> 12) & 7;
            const int pick = load[r] < load[h] ? r : h;
            owner[e] = pick;
            load[pick] += rep_cost(c, wmode, bm);
        }
    }
    __syncthreads();
}

// ------------------------------------------------------------------ routing front end (verify / decode / extends)
// One CTA in place of topk_ids -> local ids (expert map), moe_align_block_size and pg_tiles, for numel <= RT_MAX
// routes (up to 3072-token extends): local slot e < nl <= RT_E (32 home + replicas) gets ceil(count_e / XNT)
// consecutive blocks, slots ascending, none for non-local routes; a block's sorted_ids hold its routes in route order,
// then the numel sentinel. block_experts / live are what pg_tiles writes for the same counts (widest class first,
// blocks ascending), so gather / gg1 / gg2 run unchanged and every route's row is the moe_align path's. local_ids[r] =
// e or -1. Warp w takes a contiguous run of whole 32-route groups and counts it into its own row of per-slot counts
// (__match_any_sync leaders, no atomics), one warp scan per slot (warp w: slots w, w + 32) turns the 32 rows into
// bases, one warp builds the block / live lists over two 32-slot halves, and every warp places its routes from its row.
constexpr int RT_MAX = 24576, RT_G = RT_MAX / 1024, RT_E = 64, RT_NB = RT_E + RT_MAX / XNT;
// width code of a block with v routes (pg_tiles' rule) and the live classes, widest first; the narrow build
// (_cuda_src_pgn, the XNT 128 one) codes widths in 16-token steps
constexpr bool RT_N16 = XNT <= 128;
constexpr int RT_NCLS = RT_N16 ? 6 : 4, RT_FULL = RT_N16 ? XNT / 16 - 1 : 3;
DINL int route_code(int v, int wmask) {
    int code = RT_FULL;
    if (RT_N16) {
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
DINL int route_cls(int i) { return RT_N16 ? (i < 4 ? 15 - 4 * i : 1 - (i - 4)) : RT_FULL - i; }
// XB: the extend balancer (me = rank + 8), a separate instance so the verify one keeps its code
template <bool XB>
__global__ void __launch_bounds__(1024) pg_route(const int* __restrict__ topk_ids, const int* __restrict__ emap,
                                                 int n_global, int numel, int wmask, int nl,
                                                 int* __restrict__ local_ids, int* __restrict__ sorted_ids,
                                                 int* __restrict__ block_experts, int* __restrict__ live,
                                                 const int* __restrict__ rtab, int n_rep, int me) {
    __shared__ int wc[32][RT_E + 1];   // per (warp, slot): count -> base (row pad: the scan reads columns)
    __shared__ int r_tabs[256], r_cnt[256], r_own[256], r_ord[256], r_load[8], r_map[256];
    __shared__ int cnt[RT_E], bstart[RT_E], bexp[RT_NB], ltot[2 * RT_NCLS], nblk;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const unsigned lt = (1u << lane) - 1u;
    wc[w][lane] = 0;
    wc[w][32 + lane] = 0;
    __syncwarp();
    pdl_wait();
    pdl_trigger();
    if (rtab != nullptr) {   // the replica balancer here: this call's map
        if (XB) {            // the extend balancer (rep_align_scan's decision: pg cost weights, count order)
            if (tid < 256) { r_tabs[tid] = rtab[tid]; r_cnt[tid] = 0; }
            if (tid < n_rep) r_ord[tid] = rtab[256 + tid];
            __syncthreads();
            for (int i = tid; i < numel; i += 1024) {
                const int g = topk_ids[i];
                if ((unsigned)g < 256u) atomicAdd(&r_cnt[g], 1);
            }
            __syncthreads();
            rep_decide(r_cnt, r_tabs, r_ord, n_rep, 2, XNT, 1, r_own, r_load, r_map);   // r_map: its sort scratch
        } else {             // rep_owner, pg_rep_assign's decision, passes 0
            rep_owner<2>(topk_ids, numel, rtab, n_rep, 0, r_tabs, r_cnt, r_own, r_ord, r_load);
        }
        if (tid < 256) {
            const int v = r_tabs[tid];
            r_map[tid] = (r_own[tid] == (XB ? me - 8 : me) && (v & 0xFF) != 0xFF) ? (v & 0xFF) : -1;
        }
        __syncthreads();
    }
    const int ng = (numel + 1023) / 1024;   // 32-route groups per warp
    const int r0 = w * ng * 32;
    int er[RT_G];
#pragma unroll
    for (int k = 0; k < RT_G; k++) {
        const int r = r0 + k * 32 + lane;
        int e = -1;
        if (k < ng && r < numel) {
            const int g = topk_ids[r];
            if (rtab != nullptr) e = (unsigned)g < 256u ? r_map[g] : -1;
            else if ((unsigned)g < (unsigned)n_global) e = emap ? emap[g] : g;
            if ((unsigned)e >= (unsigned)nl) e = -1;
            local_ids[r] = e;
        }
        er[k] = e;
    }
#pragma unroll
    for (int k = 0; k < RT_G; k++) {
        if (k < ng) {
            const unsigned peers = __match_any_sync(0xFFFFFFFFu, er[k]);
            if (er[k] >= 0 && (peers & lt) == 0) wc[w][er[k]] += __popc(peers);
            __syncwarp();
        }
    }
    __syncthreads();
#pragma unroll
    for (int x = w; x < RT_E; x += 32) {   // warp w = slots w, w + 32: exclusive scan of the 32 warp counts
        const int c = wc[lane][x];
        int inc = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xFFFFFFFFu, inc, o);
            if (lane >= o) inc += v;
        }
        wc[lane][x] = inc - c;
        if (lane == 31) cnt[x] = inc;
    }
    __syncthreads();
    // lane l = slots l (half 0) and 32 + l (half 1). Every list warp derives the blocks from cnt; warp 0 writes the
    // block lists, warp w < 2 RT_NCLS counts class route_cls(w / 2)'s live blocks of half w % 2, and after the barrier
    // each one writes its own run of the live list (the old one-warp order: classes widest first, halves, slots).
    int nbx[2], code[2], b0[2], carry = 0, pos = 0;
    if (w < 2 * RT_NCLS) {
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            const int x = 32 * hf + lane;
            const int c = cnt[x];
            nbx[hf] = (c + XNT - 1) / XNT;
            int bs = nbx[hf];
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xFFFFFFFFu, bs, o);
                if (lane >= o) bs += v;
            }
            b0[hf] = carry + bs - nbx[hf];
            carry += __shfl_sync(0xFFFFFFFFu, bs, 31);
            if (w == 0) {
                bstart[x] = b0[hf];
                for (int j = 0; j < nbx[hf]; j++) { bexp[b0[hf] + j] = x; block_experts[b0[hf] + j] = x; }
            }
            // the slot's last block holds v = c - XNT (nbx - 1) routes, the others XNT (the full class)
            const int vl = nbx[hf] > 0 ? c - (nbx[hf] - 1) * XNT : 0;
            code[hf] = route_code(vl, wmask);
        }
        const int cls = route_cls(w >> 1), n = nbx[w & 1], cd = code[w & 1];
        // blocks of class cls a slot contributes: its full blocks (RT_FULL) and / or its last block
        const int k_c = n == 0 ? 0 : (cls == RT_FULL ? n - 1 : 0) + (cd == cls ? 1 : 0);
        int inc = k_c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xFFFFFFFFu, inc, o);
            if (lane >= o) inc += v;
        }
        pos = inc - k_c;
        if (lane == 31) ltot[w] = inc;
        if (w == 0 && lane == 0) { live[0] = carry; nblk = carry; }
    }
    __syncthreads();
    if (w < 2 * RT_NCLS) {
        for (int j = 0; j < w; j++) pos += ltot[j];
        const int hf = w & 1, cls = route_cls(w >> 1), n = nbx[hf];
        if (cls == RT_FULL) for (int j = 0; j < n - 1; j++) live[1 + pos++] = (b0[hf] + j) | (RT_FULL << 20);
        if (n > 0 && code[hf] == cls) live[1 + pos] = (b0[hf] + n - 1) | (code[hf] << 20);
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < RT_G; k++) {
        if (k < ng) {
            const int e = er[k];
            const unsigned peers = __match_any_sync(0xFFFFFFFFu, e);
            if (e >= 0) sorted_ids[bstart[e] * XNT + wc[w][e] + __popc(peers & lt)] = r0 + k * 32 + lane;
            __syncwarp();
            if (e >= 0 && (peers & lt) == 0) wc[w][e] += __popc(peers);
            __syncwarp();
        }
    }
    const int nb = nblk;
    for (int i = tid; i < nb * XNT; i += 1024) {          // sentinel slots (disjoint from the route slots)
        const int x = bexp[i / XNT];
        if (i - bstart[x] * XNT >= cnt[x]) sorted_ids[i] = numel;
    }
}

// effective wgmma width of a block: the smallest COMPILED width (WSET bit0 64, bit1 128, bit2 192; 256 always)
// covering the width code from pg_tiles. Producer and consumers must agree (the producer stages nv columns).
template <int WSET> DINL int eff_nv(int wcode) {
#ifdef PGN_WSET
    const int need = 16 * (wcode + 1);
    if ((WSET & 16) && need <= 16) return 16;
    if ((WSET & 8) && need <= 32) return 32;
    if ((WSET & 1) && need <= 64) return 64;
    if ((WSET & 2) && need <= 128) return 128;
    return XNT;
#else   // PGN_WSET
    if ((WSET & 1) && wcode == 0) return 64;
    if ((WSET & 2) && wcode <= 1) return 128;
    if ((WSET & 4) && wcode <= 2) return 192;
    return 256;
#endif  // PGN_WSET
}

// item of CTA b in round j: round-robin, or snake (odd rounds reversed) over the widest-first item list, which
// balances CTAs that get few items of uneven cost (turn 1 / turn 4: +4% at turn 1) without changing the concurrent
// item window; with ~50 mostly wide items per CTA (first turn) round-robin is as good or better (gg1 0.97-1.00x).
DINL int sched_item(int j, int b, int G, int snake) {
    return j * G + ((snake && (j & 1)) ? G - 1 - b : b);
}

template <int NV> DINL void wgmma_nv(float* d, unsigned* a, unsigned long long db, int accum) {
    if constexpr (NV == 256) wgmma_n256(d, a, db, accum);
    else if constexpr (NV == 192) wgmma_n192(d, a, db, accum);
    else if constexpr (NV == 128) wgmma_n128(d, a, db, accum);
#ifdef PGN_WSET
    else if constexpr (NV == 64) wgmma_n64(d, a, db, accum);
    else if constexpr (NV == 32) wgmma_n32(d, a, db, accum);
    else wgmma_n16(d, a, db, accum);
#else   // PGN_WSET
    else wgmma_n64(d, a, db, accum);
#endif  // PGN_WSET
}

// ------------------------------------------------------------------ consumer pieces (force-inlined)
// Consumer cursor + per-thread constants. Passed by reference into force-inlined templates (the wg_kernel iter_body
// pattern): a generic lambda per width let nvcc out-line the k-step, which put the accumulators on the stack
// (912 B frame, C7512 serialization).
struct Ctx {
    char* smem;
    unsigned sbase, mb, epi;
    unsigned long long dhi;
    int ctid, wtid, lw, lane, r0l, r1l, cw, sfo;
    int bar, hf, woff;          // named barrier, row half of the 128-row w-tile (output offsets), W|sf slot offset
    int sl, cph, rsl;
};

// LAG = wgmma groups kept in flight (wait<LAG>; 1 - a 4-bank LAG 2 loop serializes); stage kt-LAG is released
// after the wait, the item's last LAG stages after its final wait<0> (release_rest).
// PAIR = 1 (64-token items): k-tiles 2s and 2s+1 share ring slot s (X at SUB * 8 KB, W|sf at 16 KB + SUB * WSB):
// the k-tile with SUB 0 waits for the slot and releases slot s-1 (its last group retired by this wait<1>), SUB 1
// advances. Twice the k-tiles in flight for the same ring: 64-token items are bound by ring depth / TMA latency.
template <int GATED, int NV, int NST, int MODE, int LAG, int PAIR = 0, int SUB = 0, int N4 = 0>
DINL void kstep(Ctx& c, int kt, float* acc, unsigned (&ab)[16]) {
    constexpr unsigned XOFF = PAIR ? SUB * 8192 : 0;
    constexpr unsigned WOFF = PAIR ? 16384 + SUB * WSB : XB;
    if (!PAIR || SUB == 0) mbar_wait(c.mb + c.sl * 8, c.cph);
    if (MODE == 0) {
        const char* stp = c.smem + c.sl * SB + (N4 ? c.woff : 0);
        const uint4 w = *reinterpret_cast<const uint4*>(stp + WOFF + c.ctid * 16);
        const uint4 sfp = *reinterpret_cast<const uint4*>(stp + WOFF + 4096 + c.sfo);   // bf16 (r0, r1) x 4
        const unsigned* sfq = reinterpret_cast<const unsigned*>(&sfp);
        const unsigned* wq = reinterpret_cast<const unsigned*>(&w);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            __nv_bfloat162 d[4];
            dq3(wq[q], d);
            const __nv_bfloat162 sf2 = u32b2(sfq[q]);
            const __nv_bfloat162 s0 = __bfloat162bfloat162(__low2bfloat16(sf2));
            const __nv_bfloat162 s1 = __bfloat162bfloat162(__high2bfloat16(sf2));
            ab[4 * q + 0] = b2u32(__hmul2(d[0], s0));
            ab[4 * q + 1] = b2u32(__hmul2(d[1], s1));
            ab[4 * q + 2] = b2u32(__hmul2(d[2], s0));
            ab[4 * q + 3] = b2u32(__hmul2(d[3], s1));
        }
    }
    if (MODE != 3) {
        const unsigned xaddr = c.sbase + c.sl * SB + XOFF;
        constexpr unsigned HS = PAIR ? 4096 : XB / 2;     // gg2: stride of the two SW64 half-tiles in the slot
        // the descriptor's start-address field is (addr >> 4) < 2^14 for every smem address (16 B aligned), below
        // the LBO bits of dhi's low word: base + compile-time k16 offset needs no mask and no carry handling
        const unsigned dlo0 = (unsigned)c.dhi + (xaddr >> 4);
        const unsigned dhi1 = (unsigned)(c.dhi >> 32);
        wgmma_fence();
#pragma unroll
        for (int q = 0; q < 4; q++) {
            // gg1: k16 step q = +32 B inside the 128 B SW128 token row; gg2: steps 0-1 / 2-3 live in the two
            // 64 B SW64 half-tiles (features 0-31 / 32-63 of the k-tile, one per gg1 warpgroup)
            const unsigned qo = GATED ? q * 2 : ((q >> 1) * HS + (q & 1) * 32) >> 4;
            const unsigned long long db = ((unsigned long long)dhi1 << 32) | (unsigned long long)(dlo0 + qo);
            wgmma_nv<NV>(acc, ab + 4 * q, db, (kt > 0 || q > 0) ? 1 : 0);
        }
        wgmma_commit();
        wgmma_wait<LAG>();
    } else {
        fence_bank(ab);
    }
    if (PAIR ? (SUB == 0 && kt >= 2) : (kt >= LAG)) {
        mbar_arrive(c.mb + NST * 8 + c.rsl * 8);
        if (++c.rsl == NST) c.rsl = 0;
    }
    if (!PAIR || SUB == 1) {
        if (++c.sl == NST) { c.sl = 0; c.cph ^= 1; }
    }
}

template <int NST>
DINL void release_rest(Ctx& c, int n) {
    for (int i = 0; i < n; i++) {
        mbar_arrive(c.mb + NST * 8 + c.rsl * 8);
        if (++c.rsl == NST) c.rsl = 0;
    }
}

DINL float okc_silu_mul(float a, float b, float ngl, float gsq) {   // (a b gsq) sigmoid(-a ngl / log2 e)
    float e;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(fminf(a * ngl, 126.0f)));
    const float d = 1.0f + e;
    float r = __int_as_float(0x7EF311C7 - __float_as_int(d));
    r = r * fmaf(-d, r, 2.0f);
    r = r * fmaf(-d, r, 2.0f);
    r = r * fmaf(-d, r, 2.0f);
    return a * b * gsq * r;
}
// One item (block pm x w-tile wt) of width NV: mainloop + this warpgroup's epilogue. Same consumer structure as
// wg12 (ptxas keeps the two A banks in distinct registers): item-scoped banks/accumulators, compile-time bank
// selection, unconditional wait<1>, every consumer thread releases the stage of k-tile kt-1 (and the item's
// last stage after its final wait<0>).
// acc / bA / bB are ONE set shared by all width specializations (each width uses a prefix of acc): with
// per-width arrays ptxas kept every width's accumulators apart and ran out of the 232 registers (C7512, spills).
// gsv = the expert's global scale (gg1), rids = the item's 256 route ids in smem (gg2): both staged by the
// producer with the item's first stage, so no consumer ever waits on a global load.
#ifdef PGN_WSET
template <int GATED, int NST, int MODE, int NV, int PAIR64, int ACC = XNT / 2, int N4 = 0>
DINL void item_body(Ctx& c, float (&acc)[ACC], unsigned (&bA)[16], unsigned (&bB)[16], int pm, int wt, float gsv,
                    const int* rids, void* __restrict__ out, int NWT, int numel, __nv_bfloat16* m1) {
    constexpr int PR = (PAIR64 && NV <= 64) ? 1 : 0;
#else   // PGN_WSET
template <int GATED, int NST, int MODE, int NV, int PAIR64, int ACC = 128, int N4 = 0>
DINL void item_body(Ctx& c, float (&acc)[ACC], unsigned (&bA)[16], unsigned (&bB)[16], int pm, int wt, float gsv,
                    const int* rids, void* __restrict__ out, int NWT, int numel, __nv_bfloat16* m1) {
    constexpr int PR = (PAIR64 && NV == 64) ? 1 : 0;
#endif  // PGN_WSET
    for (int kt = 0; kt < NKT; kt += 2) {
        kstep<GATED, NV, NST, MODE, 1, PR, 0, N4>(c, kt, acc, bA);
        kstep<GATED, NV, NST, MODE, 1, PR, 1, N4>(c, kt + 1, acc, bB);
    }
    wgmma_wait<0>();
    release_rest<NST>(c, 1);
    fence_bank(bA); fence_bank(bB);
    fence_acc(acc, NV / 2);
    if (MODE == 4) {                                     // probe: no epilogue (acc kept live, never stored)
        if (numel == -12345) {
            float* o = reinterpret_cast<float*>(out) + (size_t)c.ctid * 128;
#pragma unroll
            for (int i = 0; i < NV / 2; i++) o[i] = acc[i];
        }
        return;
    }
    const int lw = c.lw, lane = c.lane, wtid = c.wtid, cw = c.hf, bar = c.bar;
    const unsigned epi = c.epi;
    if (GATED) {
        const float gsc = gsv * 256.0f;                  // 2^8 decode compensation x global scale
        const float hgs = 0.5f * gsc, gus = gsc * gsc * 0.5f;
        const float ngl = -1.4426950408889634f * gsc, gsq = gsc * gsc;
        if (wtid == 0) bulk_wait_read0();                // previous item's bulk store finished reading epi
        named_bar_sync(bar, 128);
        // this warpgroup's 32 features = one 64 B row per token (SW64 half-tile, independent of the other WG):
        // stmatrix.trans matrix m of an x4 = i-block 4s+m (tokens 8i..8i+7); lane 8m+j -> token t = 8i+j gets this
        // warp's 8 features = chunk lw of the row, stored at chunk lw ^ ((t >> 1) & 3) (64B swizzle; t & 7 = lane & 7)
        const unsigned lanebase = epi + ((lane >> 3) * 8 + (lane & 7)) * 64 + ((lw ^ (((lane & 7) >> 1) & 3)) << 4);
#ifdef PGN_WSET
        if constexpr (NV == 16) {
            unsigned r[2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                r[i] = b2u32(__floats2bfloat162_rn(okc_silu_mul(acc[4 * i], acc[4 * i + 2], ngl, gsq),
                                                    okc_silu_mul(acc[4 * i + 1], acc[4 * i + 3], ngl, gsq)));
            }
            stsm_x2_trans(lanebase, r[0], r[1]);
        }
#endif  // PGN_WSET
#pragma unroll
        for (int s = 0; s < NV / 32; s++) {
            unsigned r[4];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const int i = 4 * s + m;
                // silu(g) u 2^-6 = (g u 2^-7)(1 + tanh(g / 2)), g / u = acc * gsc
#ifdef PGN_WSET
                r[m] = b2u32(__floats2bfloat162_rn(okc_silu_mul(acc[4 * i], acc[4 * i + 2], ngl, gsq),
                                                    okc_silu_mul(acc[4 * i + 1], acc[4 * i + 3], ngl, gsq)));
#else
                float t0, t1;
                asm("tanh.approx.f32 %0, %1;" : "=f"(t0) : "f"(acc[4 * i] * hgs));
                asm("tanh.approx.f32 %0, %1;" : "=f"(t1) : "f"(acc[4 * i + 1] * hgs));
                const float p0 = acc[4 * i] * acc[4 * i + 2] * gus, p1 = acc[4 * i + 1] * acc[4 * i + 3] * gus;
                r[m] = b2u32(__floats2bfloat162_rn(fmaf(p0, t0, p0), fmaf(p1, t1, p1)));
#endif
            }
            stsm_x4_trans(lanebase + s * 32 * 64, r[0], r[1], r[2], r[3]);
        }
        fence_proxy_async();
        named_bar_sync(bar, 128);
        if (wtid == 0) {                                 // the half-tile's first NV token rows are contiguous
            bulk_s2g(reinterpret_cast<char*>(out) + ((size_t)pm * NWT + wt) * XB + cw * (XB / 2), epi, NV * 64);
            bulk_commit();
        }
    } else {
        const float sc2 = 256.0f;                        // 2^8 decode comp (bf16 intermediate: no headroom)
        __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(out);
#ifdef PG_M1RS   // rid * rs_mg >> 40 == rid / rs_rows exactly (rid, rs_rows < 2^17)
        const unsigned rs_rows = (unsigned)numel >> 3;
        const unsigned long long rs_mg = ((1ull << 40) + rs_rows - 1) / (rs_rows ? rs_rows : 1u);
#endif
        const char* epp = c.smem + (epi - c.sbase);
#ifdef PGN_WSET
        constexpr int NPS = (NV + 63) / 64, NSG = NV >= 64 ? 4 : NV / 16;   // passes, x4 groups per pass
#pragma unroll
        for (int p = 0; p < NPS; p++) {                  // 64-token passes (one partial pass below 64)
            named_bar_sync(bar, 128);                 // previous pass's readers are done
#pragma unroll
            for (int s = 0; s < NSG; s++) {              // 4 x4-groups = 16 matrices (8 i-blocks x 2 row halves)
#else   // PGN_WSET
#pragma unroll
        for (int p = 0; p < NV / 64; p++) {              // 64-token passes
            named_bar_sync(bar, 128);                 // previous pass's readers are done
#pragma unroll
            for (int s = 0; s < 4; s++) {                // 4 x4-groups = 16 matrices (8 i-blocks x 2 row halves)
#endif  // PGN_WSET
                unsigned r[4];
#pragma unroll
                for (int m = 0; m < 4; m++) {
                    const int mm = 4 * s + m;            // matrix: i-block 8p + (mm >> 1), row half mm & 1
                    const int i = 8 * p + (mm >> 1), h = mm & 1;
                    __nv_bfloat162 v = __floats2bfloat162_rn(acc[4 * i + 2 * h] * sc2, acc[4 * i + 2 * h + 1] * sc2);
                    r[m] = *reinterpret_cast<unsigned*>(&v);
                }
                // lane 8m+j: row j of matrix (4s+m) -> token 8*il + j of the pass, feature cols lw*16 + 8h
                const int mm = 4 * s + (lane >> 3);
                const int il = mm >> 1, h = mm & 1;
                const unsigned a = epi + (8 * il + (lane & 7)) * EPI2_PITCH + (lw * 16 + 8 * h) * 2;
                stsm_x4_trans(a, r[0], r[1], r[2], r[3]);
            }
            named_bar_sync(bar, 128);
#pragma unroll
#ifdef PGN_WSET
            for (int cc = 0; cc < NSG; cc++) {
#else   // PGN_WSET
            for (int cc = 0; cc < 4; cc++) {
#endif  // PGN_WSET
                const int idx = cc * 128 + wtid;
                const int row = idx >> 3, ch = idx & 7;
                const int rid = rids[64 * p + row];
                if (rid < numel) {
                    const uint4 v = *reinterpret_cast<const uint4*>(epp + row * EPI2_PITCH + ch * 16);
                    const size_t off = (size_t)rid * (NWT * 128) + wt * 128 + cw * 64 + ch * 8;
#ifdef PG_M1_STORE   // M1 (the 64-block build; never with M1-RS, whose m1 is the home table): the row goes into
                     // every rank's slot buffer
                    if (m1 != nullptr)
                        asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1, %2, %3, %4};"
                                     :: "l"(m1 + off), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
                    else
#endif
#ifdef PG_M1RS   // M1-RS (its own builds): the row goes to its token's home rank (SP shard rid / (numel / 8))
                    if (m1 != nullptr) {
                        const int home = (int)(((unsigned long long)rid * rs_mg) >> 40);
#ifdef PG_M1RS_CHECK   // debug builds: a home outside the 8-entry table traps instead of storing through garbage
                        if ((unsigned)home >= 8u) __trap();
#endif
                        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(
                            __ldg(reinterpret_cast<const unsigned long long*>(m1) + home));
                        *reinterpret_cast<uint4*>(dst + (off - (size_t)home * rs_rows * (NWT * 128))) = v;
                    } else
#endif
                        *reinterpret_cast<uint4*>(yp + off) = v;
                }
            }
        }
    }
}

// ------------------------------------------------------------------ token gather
// bf16 x rows -> bf16 k-tile token images in the 128B-swizzled K-major layout the GEMMs' B descriptors read:
// image (pm, kt) at xg + (pm*NKT + kt)*XB, token t's 64 k-values = row t (128 B), 16 B chunk c stored at chunk
// c ^ (t & 7). Live blocks only, tokens t < the block's wgmma width (padded routes zero-filled); everything else
// is never read. One thread per (token, chunk), KTB k-tiles each: coalesced 16 B loads and stores, no transpose.
template <int WSET, int KTB>
__global__ void __launch_bounds__(256) pg_gather(const __nv_bfloat16* __restrict__ x, char* __restrict__ xg,
                                                 const int* __restrict__ sorted_ids, const int* __restrict__ live,
                                                 int numel) {
    pdl_wait();
    pdl_trigger();
    const int li = blockIdx.y;
    if (li >= live[0]) return;
    const int ent = live[1 + li];
    const int pm = ent & 0xFFFFF;
    const int nv = eff_nv<WSET>(ent >> 20);
    const int kt0 = blockIdx.x * KTB;
    const int cc = threadIdx.x & 7;
    for (int t = threadIdx.x >> 3; t < nv; t += 32) {
        const int rid = sorted_ids[pm * XNT + t];
        const bool ok = rid < numel;                    // topk == 8 (host-checked): token row = rid >> 3
        const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)(ok ? (rid >> 3) : 0) * (NKT * 64) + kt0 * 64 + cc * 8);
        char* dst = xg + ((size_t)pm * NKT + kt0) * XB + t * 128 + ((cc ^ (t & 7)) << 4);
        uint4 v[KTB];
#pragma unroll
        for (int k = 0; k < KTB; k++) v[k] = ok ? src[k * 8] : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int k = 0; k < KTB; k++) *reinterpret_cast<uint4*>(dst + (size_t)k * XB) = v[k];   // bf16 images
    }
}

// pg_gather with the pre-expert RMSNorm applied on the way (small-T path, where it replaces the F.rms_norm launch).
// torch's vectorized rms_norm (block 32 x 4) sums x^2 per lane l of 128 over bf16x4 vectors l + 128 j (j = 0..7, by
// fma), reduces each warp with shfl_down 16, 8, 4, 2, 1 and the warps as (0 + 2) + (1 + 3), divides by 4096 and takes
// rsqrtf(. + eps); out = bf16(gamma * (rstd * x)). Here one warp per token holds 4 of those virtual lanes per lane
// (lane p: virtual warp p >> 3, virtual lanes (p & 7) + 8 q, q = 0..3), so the 16 and 8 steps are in-register adds,
// the 4 / 2 / 1 steps shuffles inside each 8-lane group and the warp combine 4 shuffles: the same fp32 operations in
// the same order, so the images hold exactly the bf16 values F.rms_norm returns. The CTA's 512 columns are vector
// j = blockIdx.x of every virtual lane, so each lane stores its own 4 x 4 columns of the token's image row.
template <int WSET>
__global__ void __launch_bounds__(256) pg_gather_norm(const __nv_bfloat16* __restrict__ x,
                                                      const __nv_bfloat16* __restrict__ gamma, float eps,
                                                      char* __restrict__ xg, const int* __restrict__ sorted_ids,
                                                      const int* __restrict__ live, int numel) {
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int vy = lane >> 3, vi = lane & 7;
    const int kb = blockIdx.x;
    uint2 g[4];
#pragma unroll
    for (int q = 0; q < 4; q++) g[q] = reinterpret_cast<const uint2*>(gamma)[128 * kb + 32 * vy + vi + 8 * q];
    pdl_wait();
    pdl_trigger();
    const int li = blockIdx.y;
    const int nlive = live[0];
    const int ent = live[1 + li];            // in bounds for every li < grid (live has grid + 1 entries)
    if (li >= nlive) return;
    const int pm = ent & 0xFFFFF;
    const int nv = eff_nv<WSET>(ent >> 20);
    const int* sid = sorted_ids + pm * XNT;
    char* img = xg + (size_t)pm * NKT * XB;
    for (int t = w + 8 * blockIdx.z; t < nv; t += 8 * gridDim.z) {   // gridDim.z CTAs share a block's rows
        const int rid = sid[t];
        uint2 o[4] = {make_uint2(0u, 0u), make_uint2(0u, 0u), make_uint2(0u, 0u), make_uint2(0u, 0u)};
        if (rid < numel) {
            const uint2* row = reinterpret_cast<const uint2*>(x + (size_t)(rid >> 3) * (NKT * 64));
            uint2 v[4][8], vk[4];
#pragma unroll
            for (int q = 0; q < 4; q++)
#pragma unroll
                for (int j = 0; j < 8; j++) v[q][j] = row[128 * j + 32 * vy + vi + 8 * q];
#pragma unroll
            for (int q = 0; q < 4; q++) {           // vector j = kb (a runtime index: select, no local memory)
                vk[q] = v[q][0];
#pragma unroll
                for (int j = 1; j < 8; j++) if (kb == j) vk[q] = v[q][j];
            }
            float sq[4];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                float a = 0.f;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float2 lo = __bfloat1622float2(u32b2(v[q][j].x));
                    const float2 hi = __bfloat1622float2(u32b2(v[q][j].y));
                    a = __fmaf_rn(lo.x, lo.x, a);
                    a = __fmaf_rn(lo.y, lo.y, a);
                    a = __fmaf_rn(hi.x, hi.x, a);
                    a = __fmaf_rn(hi.y, hi.y, a);
                }
                sq[q] = a;
            }
            float c = __fadd_rn(__fadd_rn(sq[0], sq[2]), __fadd_rn(sq[1], sq[3]));   // steps 16, 8
            c = __fadd_rn(c, __shfl_down_sync(0xFFFFFFFFu, c, 4));
            c = __fadd_rn(c, __shfl_down_sync(0xFFFFFFFFu, c, 2));
            c = __fadd_rn(c, __shfl_down_sync(0xFFFFFFFFu, c, 1));
            const float w0 = __shfl_sync(0xFFFFFFFFu, c, 0), w1 = __shfl_sync(0xFFFFFFFFu, c, 8);
            const float w2 = __shfl_sync(0xFFFFFFFFu, c, 16), w3 = __shfl_sync(0xFFFFFFFFu, c, 24);
            const float m2 = __fdiv_rn(__fadd_rn(__fadd_rn(w0, w2), __fadd_rn(w1, w3)), 4096.f);
            const float rs = rsqrtf(__fadd_rn(m2, eps));
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float2 xl = __bfloat1622float2(u32b2(vk[q].x)), xh = __bfloat1622float2(u32b2(vk[q].y));
                const float2 gl = __bfloat1622float2(u32b2(g[q].x)), gh = __bfloat1622float2(u32b2(g[q].y));
                o[q].x = b2u32(__floats2bfloat162_rn(__fmul_rn(gl.x, __fmul_rn(rs, xl.x)), __fmul_rn(gl.y, __fmul_rn(rs, xl.y))));
                o[q].y = b2u32(__floats2bfloat162_rn(__fmul_rn(gh.x, __fmul_rn(rs, xh.x)), __fmul_rn(gh.y, __fmul_rn(rs, xh.y))));
            }
        }
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int vl = 32 * vy + vi + 8 * q;            // this CTA's columns 4 vl .. 4 vl + 3
            const int kt = kb * 8 + (vl >> 4), ch = (vl & 15) >> 1;
            *reinterpret_cast<uint2*>(img + (size_t)kt * XB + t * 128 + ((ch ^ (t & 7)) << 4) + ((vl & 1) << 3)) = o[q];
        }
    }
}

// number of live entries before the first 64-wide one (pg_tiles stores the classes widest first)
DINL int n_wide_entries(const int* __restrict__ live) {
    int lo = 0, hi = live[0];
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (((live[1 + mid] >> 20) & 0xF) == 0) hi = mid; else lo = mid + 1;
    }
    return lo;
}

// ------------------------------------------------------------------ the GEMM
// Static round-robin items (block, w-tile), w-tile fastest; pg_tiles orders the blocks widest-class first, so every
// CTA ends on its cheap narrow items (balanced tail, measured faster than a dynamic atomic scheduler).
// WSET: width specializations compiled in (bit0 64, bit1 128, bit2 192; 256 always); a block whose width is not
// compiled in runs the next wider one. Default 3 = v16f4's rule (64 / 128 / 256). n192 together with 64 and 128
// compiles to a serialized pipeline (32 HGMMA / 32 ARRIVE / 32 DEPBAR) - keep it off.
// PAIR64: 64-token items stage two k-tiles per ring slot (see kstep).
template <int GATED, int NST, int MODE, int WSET = 3, int PAIR64 = 1, int SPL = 0>
__global__ void __launch_bounds__(384, 1) pg_gemm(
    const char* __restrict__ xg,         // token images [num_pid_m, NKT, 256 t, 128 B] bf16, 128B-swizzled
    const char* __restrict__ wsf,        // [E, NWT, NKT, WSB] merged W|sf
    const float* __restrict__ gs,        // [E] global scales (GATED)
    void* __restrict__ out,              // GATED: inter images (same layout as xg, NWT k-tiles); else y bf16 [numel, NWT*128]
    const int* __restrict__ sorted_ids,
    const int* __restrict__ block_experts,
    const int* __restrict__ live,
    int NWT, int numel, int snake, __nv_bfloat16* m1, unsigned* m1_ep)
{
    extern __shared__ __align__(1024) char smem[];
    const int tid = threadIdx.x;
    const unsigned sbase = smem_u32(smem);
    if (sbase & 1023) __trap();                         // the 128B swizzle atoms need 1024 B aligned stages
    const unsigned mb = sbase + NST * SB;               // full[NST] | empty[NST] | meta[2][4]
    int* meta = reinterpret_cast<int*>(smem + NST * SB + 2 * NST * 8);
    const unsigned ebase = mb + BAR_BYTES;              // 2 consumer epilogue buffers
    const unsigned rbase = ebase + (MODE == 4 ? 0 : 2 * (GATED ? EPI1 : EPI2));   // gg2 route-id slots
    if (tid < NST) {
        mbar_init(mb + tid * 8, 1);                     // full: producer expect_tx arrival + tx bytes
        mbar_init(mb + NST * 8 + tid * 8, 256);         // empty: every consumer thread arrives
    }
    pdl_wait();                                         // xg / live / block_experts come from the predecessors
    pdl_trigger();
    if (SPL && tid == 0) meta[8] = n_wide_entries(live);   // SPL: the 64-wide blocks are pg_n4's
    __syncthreads();                                    // the last CTA-wide barrier
    const int n_items = (SPL ? meta[8] : live[0]) * NWT;
    const int wg = tid >> 7;

    if (wg == 0) {
        // ------------------------------------------------ producer warpgroup
        reg_dec<PREGS>();
        if (tid == 0) {
            // one thread issues every bulk copy (issuing a stage's copies from 9 lanes in one instruction was
            // measured slower: the TMA unit's per-request cost dominates, not the issuing thread)
            int sl = 0, rnd = 0;
            int item = sched_item(0, blockIdx.x, gridDim.x, snake);
            // item metadata is fetched one item ahead (live entry, expert, global scale): the loads overlap the
            // current item's 64 stages instead of stalling the ring at the item boundary
            int ent_n = item < n_items ? live[1 + item / NWT] : 0;
            int e_n = item < n_items ? block_experts[ent_n & 0xFFFFF] : 0;
            float g_n = (GATED && item < n_items) ? gs[e_n] : 0.f;
            for (int j = 0; item < n_items; j++) {
                const int wt = item % NWT;
                const int ent = ent_n, e = e_n;
                const float gsv = g_n;
                const int pm = ent & 0xFFFFF, wcode = ent >> 20;
                const int nv = eff_nv<WSET>(wcode);
                const int nxt = sched_item(j + 1, blockIdx.x, gridDim.x, snake);
                if (nxt < n_items) ent_n = live[1 + nxt / NWT];
                int* mt = meta + (j & 1) * 4;           // published before this item's first stage arrives
                mt[0] = pm; mt[1] = wt; mt[2] = wcode; mt[3] = __float_as_int(gsv);
                const char* xs = xg + (size_t)pm * NKT * XB;
                const char* ws = wsf + ((size_t)e * NWT + wt) * (size_t)NKT * WSB;
#ifdef PGN_WSET
                const bool pr = PAIR64 && nv <= 64;     // two k-tiles per slot (see kstep)
#else   // PGN_WSET
                const bool pr = PAIR64 && nv == 64;     // two k-tiles per slot (see kstep)
#endif  // PGN_WSET
                const int nslot = pr ? NKT / 2 : NKT;
                for (int ks = 0; ks < nslot; ks++) {
                    if (rnd > 0) mbar_wait(mb + NST * 8 + sl * 8, (rnd - 1) & 1);
                    const unsigned st = sbase + sl * SB;
                    const bool rid_tma = !GATED && MODE != 4 && ks == 0;
                    const unsigned ridb = rid_tma ? XNT * 4 : 0;
                    if (rid_tma)
                        bulk_g2s(rbase + (j & 1) * (XNT * 4), sorted_ids + (size_t)pm * XNT, XNT * 4, mb + sl * 8);
                    if (!pr) {
                        const int kt = ks;
                        mbar_expect_tx(mb + sl * 8, nv * 128 + WSB + ridb);
                        // the block's first nv tokens are contiguous at every width: gg1 one copy (128 B SW128
                        // rows), gg2 one copy per 64 B SW64 half-tile
                        if (GATED) {
                            bulk_g2s(st, xs + (size_t)kt * XB, nv * 128, mb + sl * 8);
                        } else {
                            bulk_g2s(st, xs + (size_t)kt * XB, nv * 64, mb + sl * 8);
                            bulk_g2s(st + XB / 2, xs + (size_t)kt * XB + XB / 2, nv * 64, mb + sl * 8);
                        }
                        bulk_g2s(st + XB, ws + (size_t)kt * WSB, WSB, mb + sl * 8);
                    } else {
                        const int kt = 2 * ks;
#ifdef PGN_WSET
                        mbar_expect_tx(mb + sl * 8, 2 * nv * 128 + 2 * WSB + ridb);
#else   // PGN_WSET
                        mbar_expect_tx(mb + sl * 8, 2 * 8192 + 2 * WSB + ridb);
#endif  // PGN_WSET
#pragma unroll
                        for (int sub = 0; sub < 2; sub++) {
                            const char* xk = xs + (size_t)(kt + sub) * XB;
                            if (GATED) {
#ifdef PGN_WSET
                                bulk_g2s(st + sub * 8192, xk, nv * 128, mb + sl * 8);
                            } else {
                                bulk_g2s(st + sub * 8192, xk, nv * 64, mb + sl * 8);
                                bulk_g2s(st + sub * 8192 + 4096, xk + XB / 2, nv * 64, mb + sl * 8);
#else   // PGN_WSET
                                bulk_g2s(st + sub * 8192, xk, 8192, mb + sl * 8);
                            } else {
                                bulk_g2s(st + sub * 8192, xk, 4096, mb + sl * 8);
                                bulk_g2s(st + sub * 8192 + 4096, xk + XB / 2, 4096, mb + sl * 8);
#endif  // PGN_WSET
                            }
                        }
                        bulk_g2s(st + 16384, ws + (size_t)kt * WSB, 2 * WSB, mb + sl * 8);   // W|sf kt, kt+1
                    }
                    if (++sl == NST) { sl = 0; rnd++; }
                    if (ks == 8 && nxt < n_items) e_n = block_experts[ent_n & 0xFFFFF];
                    if (GATED && ks == 16 && nxt < n_items) g_n = gs[e_n];
                }
                item = nxt;
            }
        }
        return;
    }

    // ---------------------------------------------------- consumer warpgroups
    reg_inc<CREGS>();
    Ctx c;
    c.smem = smem;
    c.sbase = sbase;
    c.mb = mb;
    c.cw = wg - 1;                          // consumer warpgroup 0/1 = weight rows 64*cw .. +63 of the tile
    c.bar = 1 + c.cw; c.hf = c.cw; c.woff = 0;
    c.ctid = tid - 128;                     // 0..255: index into the packed W stage
    c.wtid = tid & 127;
    c.lw = (tid >> 5) & 3;
    c.lane = tid & 31;
    c.r0l = c.cw * 64 + c.lw * 16 + (c.lane >> 2);
    c.r1l = c.r0l + 8;
    c.sfo = 16 * (c.cw * 32 + c.lw * 8 + (c.lane >> 2));   // this thread's (r0, r1) bf16 scale pairs, 16 B per k-tile
    // B descriptor, K-major, LBO unused (1): gg1 = SW128 token rows (SBO 1024 B per 8-row atom, mode 1);
    // gg2 = SW64 half-tiles (SBO 512 B, mode 2)
    c.dhi = GATED ? ((1ull << 16) | ((unsigned long long)(1024 >> 4) << 32) | (1ull << 62))
                  : ((1ull << 16) | ((unsigned long long)(512 >> 4) << 32) | (2ull << 62));
    c.epi = ebase + c.cw * (GATED ? EPI1 : EPI2);
    c.sl = 0; c.cph = 0;                    // consume cursor
    c.rsl = 0;                              // release cursor = oldest consumed, unreleased stage
    const int wtid = c.wtid;
    const int* rid_s = reinterpret_cast<const int*>(smem + (rbase - sbase));

    for (int j = 0, item = sched_item(0, blockIdx.x, gridDim.x, snake); item < n_items;
         j++, item = sched_item(j, blockIdx.x, gridDim.x, snake)) {
        mbar_wait(c.mb + c.sl * 8, c.cph);  // the item's first stage (and its metadata) has landed
        const volatile int* mt = meta + (j & 1) * 4;
        const int pm = mt[0], wt = mt[1], wcode = mt[2];
        const float gsv = __int_as_float(mt[3]);
        const int* rids = rid_s + (j & 1) * XNT;
#ifdef PGN_WSET
        float acc[XNT / 2];
#else   // PGN_WSET
        float acc[128];
#endif  // PGN_WSET
        unsigned bA[16], bB[16];
        if (MODE != 0) {
#pragma unroll
            for (int i = 0; i < 16; i++) { bA[i] = 0x3F803F80u; bB[i] = 0x3F803F80u; }
        }
        const int nv = eff_nv<WSET>(wcode);
#ifdef PGN_WSET
        if ((WSET & 16) && nv == 16)
            item_body<GATED, NST, MODE, 16, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel, m1);
        else if ((WSET & 8) && nv == 32)
            item_body<GATED, NST, MODE, 32, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel, m1);
        else if ((WSET & 1) && nv == 64)
            item_body<GATED, NST, MODE, 64, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel, m1);
        else
            item_body<GATED, NST, MODE, XNT, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel, m1);
#else   // PGN_WSET
        if ((WSET & 1) && nv == 64)
            item_body<GATED, NST, MODE, 64, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel, m1);
        else if ((WSET & 2) && nv == 128)
            item_body<GATED, NST, MODE, 128, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel, m1);
        else if ((WSET & 4) && nv == 192)
            item_body<GATED, NST, MODE, 192, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel, m1);
        else
            item_body<GATED, NST, MODE, 256, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel, m1);
#endif  // PGN_WSET
    }
    if (GATED && wtid == 0) bulk_wait0();               // stores complete before the CTA retires its smem
#if defined(PG_M1_STORE) || defined(PG_M1RS)
    if (!GATED && m1_ep != nullptr) {                   // M1 / M1-RS: one release per CTA on every rank's epoch word
        named_bar_sync(3, 256);
        if (c.ctid == 0)
            asm volatile("fence.acq_rel.sys;\n\tmultimem.red.release.sys.global.add.u32 [%0], %1;"
                         :: "l"(m1_ep), "r"(1u) : "memory");
    }
#endif
}


// ------------------------------------------------------------------ large-T routing front end
// Local-slot align for large calls (the prefill bank's skr::align_count / align_scan / align_scatter, plt_c3 /
// plt_c4b): count the routes into nl <= AL_MAX slot counters -> one-CTA scan (ranges padded to BM, block -> slot map,
// sentinels = numel) -> CTA-aggregated scatter: moe_align_block_size's layout over nl slots without the 256-expert
// sort (route order inside a slot's range is free). ids: global ids mapped by emap, or local slot ids (no emap); ids
// outside [0, n_global) and slots outside [0, nl) are remote; local_ids[r] = the slot or -1.
constexpr int AL_MAX = 64;

template <typename IT>
__global__ void __launch_bounds__(256) align_count(const IT* __restrict__ ids, const int* __restrict__ emap,
                                                   int n_global, int nl, int* __restrict__ lids, int numel,
                                                   int* __restrict__ cnt) {
    __shared__ int h[AL_MAX];
    if (threadIdx.x < AL_MAX) h[threadIdx.x] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += gridDim.x * blockDim.x) {
        const long long g = (long long)ids[i];
        int e = -1;
        if (g >= 0 && g < n_global) e = emap ? emap[g] : (int)g;
        if ((unsigned)e >= (unsigned)nl) e = -1;
        lids[i] = e;
        if (e >= 0) atomicAdd(&h[e], 1);
    }
    __syncthreads();
    if (threadIdx.x < nl && h[threadIdx.x]) atomicAdd(&cnt[threadIdx.x], h[threadIdx.x]);
}

__global__ void __launch_bounds__(256) align_scan(const int* __restrict__ cnt, int nl, int* __restrict__ off,
                                                  int* __restrict__ cur, int* __restrict__ sorted_ids,
                                                  int* __restrict__ block_experts, int* __restrict__ n_padded,
                                                  int numel, int nblocks, int BM) {
    __shared__ int so[AL_MAX + 1], sc[AL_MAX];
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < nl; e++) {
            const int c = cnt[e];
            sc[e] = c;
            so[e] = acc;
            acc += (c + BM - 1) / BM * BM;
        }
        so[nl] = acc;
        *n_padded = acc;
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        const int pos = b * BM;
        int ex = -1;
        for (int e = 0; e < nl; e++)
            if (pos >= so[e] && pos < so[e + 1]) ex = e;
        block_experts[b] = ex;
    }
    for (int e = 0; e < nl; e++)
        for (int p = so[e] + sc[e] + threadIdx.x; p < so[e + 1]; p += blockDim.x) sorted_ids[p] = numel;
    if (threadIdx.x < nl) {
        off[threadIdx.x] = so[threadIdx.x];
        cur[threadIdx.x] = 0;
    }
}

__global__ void __launch_bounds__(256) align_scatter(const int* __restrict__ lids, int numel, int nl,
                                                     const int* __restrict__ off, int* __restrict__ cur,
                                                     int* __restrict__ sorted_ids, int chunk) {
    __shared__ int h[AL_MAX], base[AL_MAX];
    if (threadIdx.x < AL_MAX) h[threadIdx.x] = 0;
    __syncthreads();
    const int lo = blockIdx.x * chunk, hi = min(numel, lo + chunk);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const int e = lids[i];
        if (e >= 0) atomicAdd(&h[e], 1);
    }
    __syncthreads();
    if (threadIdx.x < nl) {
        base[threadIdx.x] = h[threadIdx.x] ? off[threadIdx.x] + atomicAdd(&cur[threadIdx.x], h[threadIdx.x]) : 0;
        h[threadIdx.x] = 0;
    }
    __syncthreads();
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const int e = lids[i];
        if (e >= 0) sorted_ids[base[e] + atomicAdd(&h[e], 1)] = i;
    }
}

// ------------------------------------------------------------------ replica balancer (small-T forwards)
// rep_owner (pasted above the route front end); out[i] = route i's slot here or -1; emap_out[e] = the same per expert.
template <typename IdT>
__global__ void __launch_bounds__(256) pg_rep_assign(const IdT* __restrict__ ids, int n_routes,
                                                     const int* __restrict__ tab, int n_rep, int me, int passes,
                                                     int* __restrict__ out, int* __restrict__ emap_out) {
    __shared__ int tabs[256], ord[256], cnt[256], owner[256], load[8];
    rep_owner<4>(ids, n_routes, tab, n_rep, passes, tabs, cnt, owner, ord, load);
    const int t = threadIdx.x;
    const int v = tabs[t];
    if (emap_out != nullptr) emap_out[t] = (owner[t] == me && (v & 0xFF) != 0xFF) ? (v & 0xFF) : -1;
    if (out == nullptr) return;
    for (int i = t; i < n_routes; i += 256) {
        const int e = (int)ids[i];
        int s = -1;
        if ((unsigned)e < 256u && owner[e] == me && (tabs[e] & 0xFF) != 0xFF) s = tabs[e] & 0xFF;
        out[i] = s;
    }
}

// ------------------------------------------------------------------ replica balancer (prefill chunks)
// At a long prefill every expert is active and pg time follows routed tokens, so a replicated expert's routes are split
// between its two holders. Same decision on every rank (integers only): c[e] = routes of e; load[r] = routes of the
// non-replicated experts homed on r; the replicated experts by count (descending, ties: lower id) in turn: x =
// clamp((load[home] + c[e] - load[replica]) / 2, 0, c[e]); the replica takes e's routes of the tokens before cut[e],
// the end of the first sb-token block where e's cumulative count reaches x (0 if x is 0), the home the rest.
// pg_rep_count (one CTA per sb-token block): cnt (global atomics, zeroed by the caller) and bc[j][b] per replicated
// expert and block; pg_rep_cut (one CTA): cut[256]; pg_rep_split: route i (token i / k) -> its slot here or -1.
__global__ void __launch_bounds__(256) pg_rep_count(const int* __restrict__ ids, int n_routes, int topk,
                                                    const int* __restrict__ tab, int n_rep, int sb, int nb,
                                                    int* __restrict__ cnt, int* __restrict__ bc) {
    __shared__ int c_s[256], ridx[256], b_s[64];
    const int t = threadIdx.x, b = blockIdx.x;
    c_s[t] = 0;
    ridx[t] = -1;
    if (t < 64) b_s[t] = 0;
    __syncthreads();
    if (t < n_rep) ridx[tab[256 + t]] = t;
    __syncthreads();
    const int i1 = min(n_routes, (b + 1) * sb * topk);
    for (int i = b * sb * topk + t; i < i1; i += 256) {
        const int e = ids[i];
        if ((unsigned)e >= 256u) continue;
        atomicAdd(&c_s[e], 1);
        if (ridx[e] >= 0) atomicAdd(&b_s[ridx[e]], 1);
    }
    __syncthreads();
    if (c_s[t]) atomicAdd(&cnt[t], c_s[t]);
    if (t < n_rep) bc[t * nb + b] = b_s[t];
}

__global__ void __launch_bounds__(1024) pg_rep_cut(const int* __restrict__ tab, int n_rep, int sb, int nb,
                                                   const int* __restrict__ cnt_g, const int* __restrict__ bc_g,
                                                   int* __restrict__ cut) {
    extern __shared__ int bc[];                 // [n_rep][nb] route counts per sb-token block, then cumulative
    __shared__ int cnt[256], ridx[256], load[8];
    const int t = threadIdx.x;
    if (t < 256) { cnt[t] = cnt_g[t]; ridx[t] = -1; }
    if (t < 8) load[t] = 0;
    for (int i = t; i < n_rep * nb; i += 1024) bc[i] = bc_g[i];
    __syncthreads();
    if (t < n_rep) ridx[tab[256 + t]] = t;
    __syncthreads();
    if (t < 256 && ridx[t] < 0) atomicAdd(&load[(tab[t] >> 8) & 7], cnt[t]);
    // inclusive scan of each replicated expert's block counts: one warp per row (lane = a run of consecutive blocks)
    for (int j = t >> 5; j < n_rep; j += 32) {
        const int lane = t & 31, per = (nb + 31) / 32, b0 = lane * per, b1 = min(nb, b0 + per);
        int* row = bc + j * nb;
        int acc = 0;
        for (int b = b0; b < b1; b++) acc += row[b];
        int incl = acc;
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o) incl += v;
        }
        int run = incl - acc;
        for (int b = b0; b < b1; b++) { run += row[b]; row[b] = run; }
    }
    __syncthreads();
    if (t == 0) {
        int ord[64];
        for (int j = 0; j < n_rep; j++) {       // insertion sort: count descending, expert id ascending
            const int e = tab[256 + j];
            int k = j;
            while (k > 0 && (cnt[ord[k - 1]] < cnt[e] || (cnt[ord[k - 1]] == cnt[e] && ord[k - 1] > e))) {
                ord[k] = ord[k - 1];
                k--;
            }
            ord[k] = e;
        }
        for (int j = 0; j < n_rep; j++) {
            const int e = ord[j], v = tab[e], h = (v >> 8) & 7, r = (v >> 12) & 7, c = cnt[e];
            const int* cum = bc + ridx[e] * nb;
            int x = (load[h] + c - load[r]) / 2;
            x = x < 0 ? 0 : (x > c ? c : x);
            int part = 0, cu = 0;
            if (x > 0) {
                int lo = 0, hi = nb - 1;        // first block with cum >= x (cum[nb - 1] = c >= x)
                while (lo < hi) { const int mid = (lo + hi) >> 1; if (cum[mid] >= x) hi = mid; else lo = mid + 1; }
                part = cum[lo];
                cu = (lo + 1) * sb;
            }
            load[r] += part;
            load[h] += c - part;
            cnt[e] = -1 - cu;                   // stash the cut (cnt is no longer needed for e)
        }
    }
    __syncthreads();
    if (t < 256) cut[t] = ridx[t] >= 0 ? -1 - cnt[t] : 0;
}

__global__ void __launch_bounds__(256) pg_rep_split(const int* __restrict__ ids, int n_routes, int topk,
                                                    const int* __restrict__ tab, const int* __restrict__ cut, int me,
                                                    int* __restrict__ out) {
    __shared__ int tabs[256], cuts[256];
    tabs[threadIdx.x] = tab[threadIdx.x];
    cuts[threadIdx.x] = cut[threadIdx.x];
    __syncthreads();
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n_routes; i += gridDim.x * 256) {
        const int e = ids[i];
        int s = -1;
        if ((unsigned)e < 256u) {
            const int v = tabs[e];
            const int owner = (i / topk) < cuts[e] ? (v >> 12) & 7 : (v >> 8) & 7;
            if (owner == me && (v & 0xFF) != 0xFF) s = v & 0xFF;
        }
        out[i] = s;
    }
}


// ------------------------------------------------------------------ SP row gather on the push plane (OKC_SP_AG_PUSH)
// The SP MoE section's three all-gathers in one launch on CustomAllReduceV2's push plane, with its protocol (+0.0
// halves mark "not arrived", one phase counter per block): every rank's [S, 4096] bf16 rows, [S, 8] fp32 scores and
// [S, 8] int32 ids -> [world S, ...] in rank order. A slot row is the row's 512 16 B vectors, then 8 vectors holding the
// 16 words two per vector as 11 / 11 / 10-bit pieces with 0x8000 in every half. A +0.0 in the rows arrives as -0.0.
constexpr int AGR_V = NKT * 8 + 8;
DINL bool agr_ready(uint4 r) {
    return (r.x & 0xffffu) && (r.x >> 16) && (r.y & 0xffffu) && (r.y >> 16) && (r.z & 0xffffu) && (r.z >> 16) &&
           (r.w & 0xffffu) && (r.w >> 16);
}
__global__ void __launch_bounds__(512) pg_ag_rows(const uint4* __restrict__ x, const unsigned* __restrict__ sc,
                                                  const unsigned* __restrict__ ids, uint4* __restrict__ out,
                                                  unsigned* __restrict__ osc, unsigned* __restrict__ oids,
                                                  unsigned* counter, int nctr, char* ws, char* mc, unsigned half,
                                                  int rank, int world, int S) {
    const int bx = blockIdx.x, tx = threadIdx.x;
    pdl_wait();
    const unsigned phase = counter[bx] & 1;
    ws += half * phase;
    mc += half * phase;
    if (bx == gridDim.x - 1) {   // the last block flips its own counter and every one past the grid
        pdl_trigger();
        __syncthreads();
        for (int i = bx + tx; i < nctr; i += 512) counter[i] = phase ^ 1;
        return;
    }
    const int stride = (gridDim.x - 1) * 512;
    for (int v = bx * 512 + tx; v < S * AGR_V; v += stride) {
        const int row = v / AGR_V, c = v % AGR_V;
        uint4 r;
        if (c < NKT * 8) {
            r = x[row * NKT * 8 + c];
            unsigned* w = reinterpret_cast<unsigned*>(&r);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if (!(w[j] & 0xffffu)) w[j] |= 0x8000u;
                if (!(w[j] >> 16)) w[j] |= 0x80000000u;
            }
        } else {
            const int k = (c - NKT * 8) * 2;
            const unsigned b0 = k < 8 ? sc[row * 8 + k] : ids[row * 8 + k - 8];
            const unsigned b1 = k < 8 ? sc[row * 8 + k + 1] : ids[row * 8 + k - 7];
            r = make_uint4((b0 & 0x7ffu) | (b0 >> 11 & 0x7ffu) << 16 | 0x80008000u,
                           b0 >> 22 | (b1 & 0x7ffu) << 16 | 0x80008000u,
                           (b1 >> 11 & 0x7ffu) | b1 >> 22 << 16 | 0x80008000u, 0x80008000u);
        }
        asm volatile("multimem.st.weak.global.v4.f32 [%4], {%0, %1, %2, %3};" ::"r"(r.x), "r"(r.y), "r"(r.z),
                     "r"(r.w), "l"(mc + ((size_t)(rank * S + row) * AGR_V + c) * 16) : "memory");
    }
    pdl_trigger();
    for (int v = bx * 512 + tx; v < world * S * AGR_V; v += stride) {
        uint4* src = reinterpret_cast<uint4*>(ws) + v;
        uint4 r;
        do {
            asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z),
                         "=r"(r.w) : "l"(src));
        } while (!agr_ready(r));
        const int g = v / AGR_V, c = v % AGR_V;
        if (c < NKT * 8) {
            out[g * NKT * 8 + c] = r;
        } else {
            const int k = (c - NKT * 8) * 2;
            unsigned* o = k < 8 ? osc + g * 8 + k : oids + g * 8 + k - 8;
            o[0] = (r.x & 0x7ffu) | (r.x >> 16 & 0x7ffu) << 11 | (r.y & 0x3ffu) << 22;
            o[1] = (r.y >> 16 & 0x7ffu) | (r.z & 0x7ffu) << 11 | (r.z >> 16 & 0x3ffu) << 22;
        }
        *src = make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();
    if (tx == 0) counter[bx] = phase ^ 1;
}
// Extend-sized calls with replicas (above OKC_MOE_REP_MAX_TOKENS, below the split): align_local's three kernels with
// the balancer inside the scan, from this call's per-expert counts: count the routes per global expert -> one CTA:
// rep_decide (pg cost weights, count order) -> this call's [256] global -> slot map, the slot counts and the scan ->
// scatter through the map. align_local(ids, map)'s layout; every rank decides alike from the same counts.
template <typename IT>
__global__ void __launch_bounds__(256) rep_align_count(const IT* __restrict__ ids, int numel, int* __restrict__ gcnt) {
    __shared__ int h[256];
    h[threadIdx.x] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += gridDim.x * blockDim.x) {
        const long long g = (long long)ids[i];
        if (g >= 0 && g < 256) atomicAdd(&h[g], 1);
    }
    __syncthreads();
    if (h[threadIdx.x]) atomicAdd(&gcnt[threadIdx.x], h[threadIdx.x]);
}

__global__ void __launch_bounds__(256) rep_align_scan(int* __restrict__ gcnt, const int* __restrict__ tab,
                                                      int n_rep, int me, int nl, int* __restrict__ emap,
                                                      int* __restrict__ off, int* __restrict__ cur,
                                                      int* __restrict__ sorted_ids, int* __restrict__ block_experts,
                                                      int* __restrict__ n_padded, int numel, int nblocks, int BM) {
    __shared__ int cnt[256], tabs[256], ord[256], owner[256], sord[256], load[8], so[AL_MAX + 1], sc[AL_MAX];
    const int t = threadIdx.x;
    const int v = tab[t];
    tabs[t] = v;
    cnt[t] = gcnt[t];
    gcnt[t] = 0;                               // the layer's counter, zero for its next call
    if (t < n_rep) ord[t] = tab[256 + t];
    if (t < AL_MAX) sc[t] = 0;
    __syncthreads();
    rep_decide(cnt, tabs, ord, n_rep, 2, BM, 1, owner, load, sord);
    const int slot = (owner[t] == me && (v & 0xFF) != 0xFF && (v & 0xFF) < nl) ? (v & 0xFF) : -1;
    emap[t] = slot;
    if (slot >= 0) sc[slot] = cnt[t];          // a slot holds one expert
    __syncthreads();
    if (t == 0) {
        int acc = 0;
        for (int e = 0; e < nl; e++) {
            so[e] = acc;
            acc += (sc[e] + BM - 1) / BM * BM;
        }
        so[nl] = acc;
        *n_padded = acc;
    }
    __syncthreads();
    for (int b = t; b < nblocks; b += blockDim.x) {
        const int pos = b * BM;
        int ex = -1;
        for (int e = 0; e < nl; e++)
            if (pos >= so[e] && pos < so[e + 1]) ex = e;
        block_experts[b] = ex;
    }
    for (int e = 0; e < nl; e++)
        for (int p = so[e] + sc[e] + t; p < so[e + 1]; p += blockDim.x) sorted_ids[p] = numel;
    if (t < nl) {
        off[t] = so[t];
        cur[t] = 0;
    }
}

template <typename IT>
__global__ void __launch_bounds__(256) rep_align_scatter(const IT* __restrict__ ids, const int* __restrict__ emap,
                                                         int numel, int nl, const int* __restrict__ off,
                                                         int* __restrict__ cur, int* __restrict__ lids,
                                                         int* __restrict__ sorted_ids, int chunk) {
    __shared__ int em[256], h[AL_MAX], base[AL_MAX];
    em[threadIdx.x] = emap[threadIdx.x];
    if (threadIdx.x < AL_MAX) h[threadIdx.x] = 0;
    __syncthreads();
    const int lo = blockIdx.x * chunk, hi = min(numel, lo + chunk);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const long long g = (long long)ids[i];
        const int e = (g >= 0 && g < 256) ? em[g] : -1;
        lids[i] = e;
        if (e >= 0) atomicAdd(&h[e], 1);
    }
    __syncthreads();
    if (threadIdx.x < nl) {
        base[threadIdx.x] = h[threadIdx.x] ? off[threadIdx.x] + atomicAdd(&cur[threadIdx.x], h[threadIdx.x]) : 0;
        h[threadIdx.x] = 0;
    }
    __syncthreads();
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const int e = lids[i];
        if (e >= 0) sorted_ids[base[e] + atomicAdd(&h[e], 1)] = i;
    }
}

}  // namespace pgk

// ------------------------------------------------------------------ host
static int pg_num_sms() {
    static int n = 0;
    if (!n) n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    return n;
}

// pdl = 1: launch with programmatic stream serialization (the kernel may start while its predecessor drains; every
// pg kernel waits on griddepcontrol before touching predecessor data). Captured graphs keep the programmatic edge.
template <typename... KArgs, typename... Args>
static void pg_launch(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, bool pdl, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...));
}

void tiles(torch::Tensor block_experts, torch::Tensor n_padded, torch::Tensor sorted_ids, torch::Tensor live,
           int64_t num_pid_m, int64_t bm, int64_t numel, int64_t wmask, int64_t nle) {
    pgk::pg_tiles<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
        block_experts.data_ptr<int>(), n_padded.data_ptr<int>(), sorted_ids.data_ptr<int>(), (int)numel,
#ifdef PGN_WSET
        (int)num_pid_m, (int)bm, (int)PGN_WSET, (int)nle, live.data_ptr<int>());
#else   // PGN_WSET
        (int)num_pid_m, (int)bm, (int)wmask, (int)nle, live.data_ptr<int>());
#endif  // PGN_WSET
}

// out: [numel] per-route slots, emap: [256] per-expert map; either may be empty (not written); passes: st weights
void rep_assign(torch::Tensor ids, torch::Tensor tab, torch::Tensor out, torch::Tensor emap, int64_t me, int64_t passes) {
    const int n_rep = (int)tab.size(0) - 256;
    TORCH_CHECK(tab.scalar_type() == at::kInt && n_rep >= 0 && n_rep <= 256 && ids.is_contiguous() && me >= 0 &&
                me < 8 && (out.numel() == 0 || (out.scalar_type() == at::kInt && out.is_contiguous() &&
                                                out.numel() == ids.numel())) &&
                (emap.numel() == 0 || (emap.scalar_type() == at::kInt && emap.numel() == 256)),
                "pg rep_assign: int32 table [256 + n_rep], int32 out like ids, int32 map [256], rank < 8");
    int* o = out.numel() > 0 ? out.data_ptr<int>() : nullptr;
    int* m = emap.numel() > 0 ? emap.data_ptr<int>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (ids.scalar_type() == at::kLong)
        pgk::pg_rep_assign<int64_t><<<1, 256, 0, stream>>>(ids.data_ptr<int64_t>(), (int)ids.numel(),
                                                             tab.data_ptr<int>(), n_rep, (int)me, (int)passes, o, m);
    else
        pgk::pg_rep_assign<int><<<1, 256, 0, stream>>>(ids.data_ptr<int>(), (int)ids.numel(), tab.data_ptr<int>(),
                                                       n_rep, (int)me, (int)passes, o, m);
}

void rep_split(torch::Tensor ids, torch::Tensor tab, torch::Tensor cnt, torch::Tensor bc, torch::Tensor cut,
               torch::Tensor out, int64_t me, int64_t sb) {
    const int n_rep = (int)tab.size(0) - 256;
    const int topk = (int)ids.size(1), n_routes = (int)ids.numel();
    const int nb = (int)((ids.size(0) + sb - 1) / sb);
    const int smem = n_rep * nb * 4;
    TORCH_CHECK(tab.scalar_type() == at::kInt && out.scalar_type() == at::kInt && ids.scalar_type() == at::kInt &&
                cut.scalar_type() == at::kInt && cut.numel() == 256 && cnt.scalar_type() == at::kInt &&
                cnt.numel() == 256 && bc.scalar_type() == at::kInt && bc.numel() >= n_rep * nb && n_rep >= 0 &&
                n_rep <= 64 && ids.is_contiguous() && out.is_contiguous() && out.numel() == ids.numel() && me >= 0 &&
                me < 8 && nb >= 1 && smem <= 48 * 1024,
                "pg rep_split: int32 ids [T, k], table [256 + n_rep <= 64], cnt / cut [256] (cnt zeroed), "
                "bc [n_rep x token blocks <= 12288], out like ids, rank < 8");
    auto stream = at::cuda::getCurrentCUDAStream();
    pgk::pg_rep_count<<<nb, 256, 0, stream>>>(ids.data_ptr<int>(), n_routes, topk, tab.data_ptr<int>(), n_rep,
                                              (int)sb, nb, cnt.data_ptr<int>(), bc.data_ptr<int>());
    pgk::pg_rep_cut<<<1, 1024, smem, stream>>>(tab.data_ptr<int>(), n_rep, (int)sb, nb, cnt.data_ptr<int>(),
                                               bc.data_ptr<int>(), cut.data_ptr<int>());
    const int grid = (n_routes + 255) / 256 < 4 * pg_num_sms() ? (n_routes + 255) / 256 : 4 * pg_num_sms();
    pgk::pg_rep_split<<<grid, 256, 0, stream>>>(ids.data_ptr<int>(), n_routes, topk, tab.data_ptr<int>(),
                                                cut.data_ptr<int>(), (int)me, out.data_ptr<int>());
}

void gather(torch::Tensor x, torch::Tensor xg, torch::Tensor sorted_ids, torch::Tensor live, int64_t numel,
            int64_t pdl) {
    TORCH_CHECK(x.size(1) == pgk::NKT * 64 && numel == x.size(0) * 8, "pg gather: K 4096, topk 8");
    const int npm = live.size(0) - 1;
#ifdef PGN_WSET
    pg_launch(pgk::pg_gather<PGN_WSET, 8>, dim3(pgk::NKT / 8, npm), dim3(256), 0, pdl != 0,
#else   // PGN_WSET
    pg_launch(pgk::pg_gather<3, 8>, dim3(pgk::NKT / 8, npm), dim3(256), 0, pdl != 0,
#endif  // PGN_WSET
              reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<char*>(xg.data_ptr()),
              sorted_ids.data_ptr<int>(), live.data_ptr<int>(), (int)numel);
}

// small-T front end: topk_ids [T, 8] int32 (global ids), emap [n_global] int32 global -> local (-1 = remote) or
// an empty tensor (ids already local); outputs local_ids [T * 8], sorted_ids [nb_max * 256], block_experts
// [nb_max], live [nb_max + 1] with nb_max = 32 + ceil(numel / 256). rtab non-empty (a replica table): the balancer
// runs in the kernel (rep_owner, as pg_rep_assign) and emap is not read; me = this rank, or this rank + 8 for the
// extend balancer (rep_align_scan's decision, as align with a table).
void route(torch::Tensor topk_ids, torch::Tensor emap, torch::Tensor rtab, torch::Tensor local_ids,
           torch::Tensor sorted_ids, torch::Tensor block_experts, torch::Tensor live, int64_t numel, int64_t n_global,
           int64_t nl, int64_t pdl, int64_t me) {
    TORCH_CHECK(numel <= pgk::RT_MAX && topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous() &&
                nl >= 1 && nl <= pgk::RT_E, "pg route: int32 ids, at most 24576 routes, 1..64 local slots");
    TORCH_CHECK(sorted_ids.numel() >= (nl + (numel + pgk::XNT - 1) / pgk::XNT) * pgk::XNT &&
                live.numel() >= nl + (numel + pgk::XNT - 1) / pgk::XNT + 1, "pg route: output sizes");
    const int* em = emap.numel() > 0 ? emap.data_ptr<int>() : nullptr;
    const int* rt = rtab.numel() > 0 ? rtab.data_ptr<int>() : nullptr;
    TORCH_CHECK(!rt || (rtab.scalar_type() == at::kInt && rtab.numel() >= 256 && rtab.numel() <= 512 && me >= 0 &&
                        me < 16), "pg route: replica table");
    pg_launch(rt && me >= 8 ? pgk::pg_route<true> : pgk::pg_route<false>, dim3(1), dim3(1024), 0, pdl != 0,
              topk_ids.data_ptr<int>(), em, (int)n_global,
#ifdef PGN_WSET
              (int)numel, (int)PGN_WSET, (int)nl, local_ids.data_ptr<int>(), sorted_ids.data_ptr<int>(),
#else   // PGN_WSET
              (int)numel, 3 /* = _WMASK */, (int)nl, local_ids.data_ptr<int>(), sorted_ids.data_ptr<int>(),
#endif  // PGN_WSET
              block_experts.data_ptr<int>(), live.data_ptr<int>(), rt, rt ? (int)rtab.numel() - 256 : 0, (int)me);
}

// SP row gather (pg_ag_rows): x [S, 4096] bf16, sc [S, 8] fp32, ids [S, 8] int32 -> out [world S, 4096], osc, oids;
// counter / ws / mc / half: CustomAllReduceV2's push counter, this rank's push workspace base, its multicast base and
// world x slot bytes
void ag_rows(torch::Tensor x, torch::Tensor sc, torch::Tensor ids, torch::Tensor out, torch::Tensor osc,
             torch::Tensor oids, torch::Tensor counter, int64_t ws, int64_t mc, int64_t half, int64_t rank,
             int64_t world, int64_t pdl) {
    const int S = (int)x.size(0);
    TORCH_CHECK(x.size(1) == pgk::NKT * 64 && sc.size(0) == S && ids.size(0) == S && out.size(0) == S * world &&
                (int64_t)S * pgk::AGR_V * 16 * world <= half && counter.numel() >= 2, "pg ag_rows: shapes / capacity");
    const int nctr = (int)counter.numel();
    const int nb = std::min((int)((world * S * pgk::AGR_V + 511) / 512), nctr - 1);
    pg_launch(pgk::pg_ag_rows, dim3(nb + 1), dim3(512), 0, pdl != 0, reinterpret_cast<const uint4*>(x.data_ptr()),
              reinterpret_cast<const unsigned*>(sc.data_ptr()), reinterpret_cast<const unsigned*>(ids.data_ptr()),
              reinterpret_cast<uint4*>(out.data_ptr()), reinterpret_cast<unsigned*>(osc.data_ptr()),
              reinterpret_cast<unsigned*>(oids.data_ptr()), reinterpret_cast<unsigned*>(counter.data_ptr()), nctr,
              reinterpret_cast<char*>(ws), reinterpret_cast<char*>(mc), (unsigned)half, (int)rank, (int)world, S);
}

// Large-T front end: ids [numel] int32 / int64 global ids (emap [n_global] int32, global -> local slot) or local slot ids
// (emap empty); writes local_ids and moe_align_block_size's outputs over nl slots; cnt [nl] must be zero on entry.
// tab non-empty: the extend balancer (rep_align_*) on global ids: cnt = the layer's [256] counter (zero on entry and
// exit), emap = this call's [256] map (written), me = this rank.
void align(torch::Tensor ids, torch::Tensor emap, torch::Tensor tab, torch::Tensor local_ids, torch::Tensor cnt,
           torch::Tensor off, torch::Tensor cur, torch::Tensor sorted_ids, torch::Tensor block_experts,
           torch::Tensor n_padded, int64_t numel, int64_t BM, int64_t nl, int64_t n_global, int64_t me) {
    TORCH_CHECK(nl >= 1 && nl <= pgk::AL_MAX && cnt.numel() >= nl, "pg align: 1..64 local slots");
    TORCH_CHECK(ids.is_contiguous() && local_ids.numel() >= numel && local_ids.scalar_type() == at::kInt,
                "pg align: ids / local_ids");
    TORCH_CHECK(emap.numel() == 0 || (emap.scalar_type() == at::kInt && emap.numel() >= n_global), "pg align: emap");
    auto st = at::cuda::getCurrentCUDAStream();
    const int* em = emap.numel() > 0 ? emap.data_ptr<int>() : nullptr;
    const int nblocks = (int)block_experts.numel();
    const int chunk = 4096;
    const int nscat = (int)((numel + chunk - 1) / chunk);
    if (tab.numel() > 0) {
        const int n_rep = (int)tab.size(0) - 256;
        TORCH_CHECK(n_rep >= 0 && cnt.numel() == 256 && emap.numel() == 256 && me >= 0 && me < 8, "pg align: rep");
        const bool i64 = ids.scalar_type() == at::kLong;
        if (i64)
            pgk::rep_align_count<long long><<<nscat, 256, 0, st>>>(
                reinterpret_cast<const long long*>(ids.data_ptr()), (int)numel, cnt.data_ptr<int>());
        else
            pgk::rep_align_count<int><<<nscat, 256, 0, st>>>(ids.data_ptr<int>(), (int)numel, cnt.data_ptr<int>());
        pgk::rep_align_scan<<<1, 256, 0, st>>>(cnt.data_ptr<int>(), tab.data_ptr<int>(), n_rep, (int)me, (int)nl,
                                              emap.data_ptr<int>(), off.data_ptr<int>(), cur.data_ptr<int>(),
                                              sorted_ids.data_ptr<int>(), block_experts.data_ptr<int>(),
                                              n_padded.data_ptr<int>(), (int)numel, nblocks, (int)BM);
        if (i64)
            pgk::rep_align_scatter<long long><<<nscat, 256, 0, st>>>(
                reinterpret_cast<const long long*>(ids.data_ptr()), emap.data_ptr<int>(), (int)numel, (int)nl,
                off.data_ptr<int>(), cur.data_ptr<int>(), local_ids.data_ptr<int>(), sorted_ids.data_ptr<int>(), chunk);
        else
            pgk::rep_align_scatter<int><<<nscat, 256, 0, st>>>(ids.data_ptr<int>(), emap.data_ptr<int>(), (int)numel,
                (int)nl, off.data_ptr<int>(), cur.data_ptr<int>(), local_ids.data_ptr<int>(),
                sorted_ids.data_ptr<int>(), chunk);
        return;
    }
    if (ids.scalar_type() == at::kLong)
        pgk::align_count<long long><<<nscat, 256, 0, st>>>(reinterpret_cast<const long long*>(ids.data_ptr()), em,
            (int)n_global, (int)nl, local_ids.data_ptr<int>(), (int)numel, cnt.data_ptr<int>());
    else {
        TORCH_CHECK(ids.scalar_type() == at::kInt, "pg align: ids must be int32 or int64");
        pgk::align_count<int><<<nscat, 256, 0, st>>>(ids.data_ptr<int>(), em, (int)n_global, (int)nl,
            local_ids.data_ptr<int>(), (int)numel, cnt.data_ptr<int>());
    }
    pgk::align_scan<<<1, 256, 0, st>>>(cnt.data_ptr<int>(), (int)nl, off.data_ptr<int>(), cur.data_ptr<int>(),
                                      sorted_ids.data_ptr<int>(), block_experts.data_ptr<int>(),
                                      n_padded.data_ptr<int>(), (int)numel, nblocks, (int)BM);
    pgk::align_scatter<<<nscat, 256, 0, st>>>(local_ids.data_ptr<int>(), (int)numel, (int)nl, off.data_ptr<int>(),
                                             cur.data_ptr<int>(), sorted_ids.data_ptr<int>(), chunk);
}

void gather_norm(torch::Tensor x, torch::Tensor gamma, double eps, torch::Tensor xg, torch::Tensor sorted_ids,
                 torch::Tensor live, int64_t numel, int64_t pdl, int64_t split) {
    TORCH_CHECK(x.size(1) == pgk::NKT * 64 && numel == x.size(0) * 8 && x.is_contiguous(), "pg gather: K 4096, topk 8");
    TORCH_CHECK(gamma.numel() == pgk::NKT * 64 && gamma.scalar_type() == at::kBFloat16 && gamma.is_contiguous(),
                "pg gather_norm: bf16 [4096] weight");
    const int npm = live.size(0) - 1;
    const dim3 grid(pgk::NKT / 8, npm, split > 1 ? (unsigned)split : 1u);   // split: CTAs per block's token rows
#ifdef PGN_WSET
    pg_launch(pgk::pg_gather_norm<PGN_WSET>, grid, dim3(256), 0, pdl != 0,
#else   // PGN_WSET
    pg_launch(pgk::pg_gather_norm<3>, grid, dim3(256), 0, pdl != 0,
#endif  // PGN_WSET
              reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()), (float)eps,
              reinterpret_cast<char*>(xg.data_ptr()), sorted_ids.data_ptr<int>(), live.data_ptr<int>(), (int)numel);
}

template <int GATED, int NST, int MODE, int WSET, int PAIR64 = 1, int SPL = 0>
static void launch(const torch::Tensor& xg, const torch::Tensor& wsf, const float* gs, void* out,
                   const int* sorted_ids, const torch::Tensor& block_experts, const torch::Tensor& live,
                   int numel, int grid, int snake, bool pdl, int64_t m1 = 0, int64_t m1_ep = 0) {
    auto fn = pgk::pg_gemm<GATED, NST, MODE, WSET, PAIR64, SPL>;
    constexpr int smem = pgk::smem_bytes<GATED, MODE>(NST);
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    const int NWT = wsf.size(1);
    TORCH_CHECK(wsf.size(2) == pgk::NKT && wsf.size(3) == pgk::WSB, "pg: wsf layout");
    pg_launch(fn, dim3(grid > 0 ? grid : pg_num_sms()), dim3(384), smem, pdl,
              reinterpret_cast<const char*>(xg.data_ptr()), reinterpret_cast<const char*>(wsf.data_ptr()),
              gs, out, sorted_ids, block_experts.data_ptr<int>(), live.data_ptr<int>(), NWT, numel, snake,
              reinterpret_cast<__nv_bfloat16*>(m1), reinterpret_cast<unsigned*>(m1_ep));
}

// snake item order for small calls (few items per CTA): tokens = numel / 8 <= pgk_snake_max_t
static int pgk_snake_max_t = 4096;
void set_snake_max_t(int64_t t) { pgk_snake_max_t = (int)t; }

// cfg 350 = the bank's shipped configuration (widths 64 / 128 / 256, 5 stages), the only one built here.
void gg1(torch::Tensor xg, torch::Tensor wsf, torch::Tensor gs, torch::Tensor inter,
         torch::Tensor block_experts, torch::Tensor live, int64_t numel, int64_t cfg, int64_t grid, int64_t pdl) {
    TORCH_CHECK(cfg == 350, "pg: only cfg 350 is built");
    const int snake = (numel / 8 <= pgk_snake_max_t) ? 1 : 0;
#ifdef PGN_WSET
    launch<1, PGN_NST, 0, PGN_WSET>(xg, wsf, gs.data_ptr<float>(), inter.data_ptr(), nullptr, block_experts, live, 0, grid,
#else   // PGN_WSET
    launch<1, 5, 0, 3>(xg, wsf, gs.data_ptr<float>(), inter.data_ptr(), nullptr, block_experts, live, 0, grid,
#endif  // PGN_WSET
                       snake, pdl != 0);
}

void gg2(torch::Tensor xg, torch::Tensor wsf, torch::Tensor y, torch::Tensor sorted_ids,
         torch::Tensor block_experts, torch::Tensor live, int64_t numel, int64_t cfg, int64_t grid, int64_t pdl,
         int64_t m1, int64_t m1_ep) {
    TORCH_CHECK(cfg == 350, "pg: only cfg 350 is built");
    const int snake = (numel / 8 <= pgk_snake_max_t) ? 1 : 0;
#ifdef PGN_WSET
    launch<0, PGN_NST, 0, PGN_WSET>(xg, wsf, nullptr, y.data_ptr(), sorted_ids.data_ptr<int>(), block_experts, live,
#else   // PGN_WSET
    launch<0, 5, 0, 3>(xg, wsf, nullptr, y.data_ptr(), sorted_ids.data_ptr<int>(), block_experts, live,
#endif  // PGN_WSET
                       (int)numel, grid, snake, pdl != 0, m1, m1_ep);
}

// M1 receiver (OKC_SW_M1): every rank's gg2 multicast its route rows into this rank's slot buffer [T * 8, 4096] and
// released once per CTA; spin (ld.acquire.sys) until `want` releases, then per token the 8-slot fp32 sum of
// post-expert-normed (rms_norm(w * y) / rms_norm(y) * w), score-weighted rows: the combined MoE output, the same on
// every rank (no all-reduce). The last CTA resets this rank's epoch word. Rows stay packed bf16 (2 CTAs / SM).
__device__ __forceinline__ float2 m1_bf2(unsigned u) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u)); }
__global__ void __launch_bounds__(512, 2) m1_recv_kernel(const __nv_bfloat16* __restrict__ y, const float* __restrict__ s,
                                                         const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                                                         float eps, int wfirst, unsigned* __restrict__ epoch, unsigned want) {
    const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    __shared__ float red[16][8];
    __shared__ float tot[8];
    if (tid == 0) {
        unsigned have;
        do {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(have) : "l"(epoch) : "memory");
        } while (have < want);
    }
    __syncthreads();
    const uint4 wr = *reinterpret_cast<const uint4*>(w + tid * 8);
    const unsigned wu[4] = {wr.x, wr.y, wr.z, wr.w};
    uint4 xr[8];
#pragma unroll
    for (int k = 0; k < 8; k++) xr[k] = __ldcg(reinterpret_cast<const uint4*>(y + ((size_t)tok * 8 + k) * 4096 + tid * 8));
    float ss[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const unsigned xu[4] = {xr[k].x, xr[k].y, xr[k].z, xr[k].w};
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 f = m1_bf2(xu[i]);
            if (wfirst) { const float2 g = m1_bf2(wu[i]); f.x *= g.x; f.y *= g.y; }
            a += f.x * f.x + f.y * f.y;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        ss[k] = a;
    }
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < 8; k++) red[wp][k] = ss[k];
    }
    __syncthreads();
    if (tid < 8) {
        float t = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) t += red[i][tid];
        tot[tid] = t;
    }
    __syncthreads();
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const float rk = rsqrtf(tot[k] * (1.0f / 4096) + eps) * s[tok * 8 + k];
        const unsigned xu[4] = {xr[k].x, xr[k].y, xr[k].z, xr[k].w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 f = m1_bf2(xu[i]), g = m1_bf2(wu[i]);
            acc[2 * i] += (f.x * g.x) * rk;
            acc[2 * i + 1] += (f.y * g.y) * rk;
        }
    }
    uint4 o;
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; i++) ob[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    *reinterpret_cast<uint4*>(out + (size_t)tok * 4096 + tid * 8) = o;
    if (tid == 0 && atomicAdd(epoch + 1, 1u) == gridDim.x - 1) { epoch[0] = 0; epoch[1] = 0; }
}

void m1_recv(torch::Tensor y, torch::Tensor s, torch::Tensor w, torch::Tensor out, double eps, int64_t wfirst,
             int64_t epoch, int64_t want) {
    pg_launch(m1_recv_kernel, dim3((unsigned)out.size(0)), dim3(512), 0, true,
              reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), (const float*)s.data_ptr<float>(),
              reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
              (float)eps, (int)wfirst, reinterpret_cast<unsigned*>(epoch), (unsigned)want);
}
"""

_CUDA_SRC = _CUDA_SRC.replace("@@WGMMA@@", "".join(_wgmma_asm(n) for n in (256, 192, 128, 64))).replace(
    "@@WGMMA_N@@", _wgmma_asm(32) + _wgmma_asm(16)).replace("@@REP@@", _CUDA_SRC_REP)
