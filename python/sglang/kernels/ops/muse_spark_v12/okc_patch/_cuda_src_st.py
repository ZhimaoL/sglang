"""CUDA source of the st small-T MoE pair (okc_patch/st_moe.py; developed in muse-opt serving/kernels/smallt-moe).

st_gemm<1>: gate/up over the call's active local experts, silu*mul into a bf16 intermediate; st_gemm<2>: down into
the per-route bf16 rows. Both read pg_moe.pack_wsf's blocks (no other weight copy) and decode them with pg's dq3 and
the pack's bf16 block scales into mma.sync m16n8k16 A fragments (one warp's 16 rows of a pg block are exactly that
fragment). Local slots: up to ELOC (32 home + hot-expert replicas), the count from wsf13.
"""

from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_rep import _CUDA_SRC_REP

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace stk {
#define DINL __device__ __forceinline__

constexpr int H = 4096;
constexpr int NKT = 64;           // k-tiles of 64 (K = 4096 in both GEMMs)
// pg block per (expert, w-tile, k-tile): 4096 B codes + the 64 row pairs' bf16 scale pairs (SFB bytes each)
constexpr int SFB = 16;
constexpr int WSB = 4096 + 64 * SFB;
constexpr int NW = 16;            // warps per CTA; warp w owns k-tiles KPW*w .. KPW*w + KPW - 1 of every slab
constexpr int KPW = NKT / NW;     // 4
constexpr int NTHR = NW * 32;
constexpr int ELOC = 64;          // local slots at most (32 home + replicas); the call's count nl comes from wsf13
constexpr int NUMEL_MAX = 1024;   // T <= 128, top-8 (RPT routes per table thread)
constexpr int RPT = NUMEL_MAX / NTHR;
constexpr int VAMAX = ELOC + NUMEL_MAX / 8;   // passes (<= 8 tokens each): <= nl + numel / 8
constexpr int R = 4;              // partial ring slots

DINL unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
DINL void mbar_init(unsigned bar, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
DINL void mbar_arrive(unsigned bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(bar) : "memory");
}
DINL void mbar_wait(unsigned bar, unsigned phase) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}\n" :: "r"(bar), "r"(phase) : "memory");
}
DINL void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
DINL void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

DINL uint4 ldw(const char* p) {          // weight codes: streamed once, kept out of L1
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
typedef uint4 sf_t;
DINL sf_t lds8(const char* p) {          // bf16 scale pairs (the 4 lanes of a row pair read the same 16 B)
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
DINL unsigned ldb32(const __nv_bfloat16* p) {   // B fragment word of an x row (written before this grid started)
    return *reinterpret_cast<const unsigned*>(p);
}
DINL unsigned ldb32cg(const __nv_bfloat16* p) { // B fragment word of the intermediate (under PDL K1 may still run)
    unsigned v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

DINL __nv_bfloat162 u32b2(unsigned v) { return *reinterpret_cast<__nv_bfloat162*>(&v); }
DINL unsigned b2u32(__nv_bfloat162 v) { return *reinterpret_cast<unsigned*>(&v); }

// pg's dq3 (pg_moe.pack_wsf DQ3 bit placement): one u32 of codes -> the 4 A registers as e2m1 x 2^-126 in bf16
DINL void dq3(unsigned t, __nv_bfloat162 d[4]) {
    d[0] = u32b2(t & 0x81C081C0u);
    d[1] = u32b2(__funnelshift_l(t, t, 10) & 0x81C081C0u);
    d[2] = u32b2(__funnelshift_l(t, t, 13) & 0x81C081C0u);
    d[3] = u32b2(((t & 0x001B001Bu) * 0x1084u) & 0x81C081C0u);
}

DINL void mma_bf16(float c[4], const unsigned a[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// one k-tile of one warp's slab piece: 16 B codes (4 k16 steps) + the row pair's 4 scale pairs, x 4 HMMA
DINL void ktile(float acc[4], const uint4 w, const sf_t sfp, const unsigned (&b)[4][2]) {
    const unsigned wq[4] = {w.x, w.y, w.z, w.w};
    const unsigned sfw[4] = {sfp.x, sfp.y, sfp.z, sfp.w};
#pragma unroll
    for (int q = 0; q < 4; q++) {
        __nv_bfloat162 d[4];
        dq3(wq[q], d);
        const __nv_bfloat162 sf2 = u32b2(sfw[q]);   // bf16 scale pair (row r0, row r0 + 8) of k16 step q
        const __nv_bfloat162 s0 = __bfloat162bfloat162(__low2bfloat16(sf2));
        const __nv_bfloat162 s1 = __bfloat162bfloat162(__high2bfloat16(sf2));
        unsigned a[4];
        a[0] = b2u32(__hmul2(d[0], s0));
        a[1] = b2u32(__hmul2(d[1], s1));
        a[2] = b2u32(__hmul2(d[2], s0));
        a[3] = b2u32(__hmul2(d[3], s1));
        mma_bf16(acc, a, b[q][0], b[q][1]);
    }
}

struct Tab {
    int nva;
    int va_e[VAMAX];
    int va_n[VAMAX];
    unsigned short va_rt[VAMAX][8];   // route index (token * 8 + k) of the pass's tokens, route order
    float gs[ELOC];
};

// Replica balancer folded in (rtab != null): every CTA of both kernels runs pg_rep_assign's rep_owner
// (_cuda_src_rep.py) on the call's global ids instead of that launch; build_tab maps id e to tabs[e]'s slot when
// owner[e] == me.
@@REP@@

// Routing table of the call from local_ids (-1 = not local), or from global ids through emap, identical in every CTA
// of both kernels: an expert's routes in route order, split in passes of 8 (ascending expert, then pass). No atomics.
// lids != null: the call's per-route slots (what pg_rep_assign returns) for the finalize.
DINL void build_tab(const int* __restrict__ ids, int numel, int nl, const float* __restrict__ gs, Tab& tb,
                    unsigned short (*wcnt)[ELOC], int* vbase, const int* rtabs, const int* rown, int me,
                    int* __restrict__ lids) {
    // route r = tid + j * NTHR (j < RPT) sits in 32-route group (w + j * NW): per-group, per-expert counts
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    for (int i = tid; i < (NUMEL_MAX / 32) * ELOC; i += NTHR) (&wcnt[0][0])[i] = 0;
    int id[RPT], rank[RPT];
#pragma unroll
    for (int j = 0; j < RPT; j++) {
        const int r = tid + j * NTHR;
        id[j] = r < numel ? ids[r] : -1;
        if (rtabs != nullptr) {
            const int e = id[j];
            id[j] = (unsigned)e < 256u && rown[e] == me && (rtabs[e] & 0xFF) != 0xFF ? rtabs[e] & 0xFF : -1;
        }
        if ((unsigned)id[j] >= (unsigned)nl) id[j] = -1;
        if (lids != nullptr && r < numel) lids[r] = id[j];
    }
    if (gs != nullptr && tid >= NTHR - ELOC && tid - (NTHR - ELOC) < nl)
        tb.gs[tid - (NTHR - ELOC)] = gs[tid - (NTHR - ELOC)];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < RPT; j++) {
        const unsigned m = __match_any_sync(0xFFFFFFFFu, id[j]);
        rank[j] = __popc(m & ((1u << lane) - 1u));
        if (id[j] >= 0 && rank[j] == 0) wcnt[w + j * NW][id[j]] = __popc(m);
    }
    __syncthreads();
    if (w == 0) {                                        // warp 0: per-expert totals, pass counts, scan
        int carry = 0;                                   // lane l: experts l, then 32 + l
#pragma unroll
        for (int hf = 0; hf < ELOC / 32; hf++) {
            const int e = 32 * hf + lane;
            int s = 0;
#pragma unroll
            for (int ww = 0; ww < NUMEL_MAX / 32; ww++) { const int c = wcnt[ww][e]; wcnt[ww][e] = s; s += c; }
            const int v = (s + 7) >> 3;
            int inc = v;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xFFFFFFFFu, inc, o);
                if (lane >= o) inc += y;
            }
            const int base = carry + inc - v;
            vbase[e] = base;
            for (int p = 0; p < v; p++) { tb.va_e[base + p] = e; tb.va_n[base + p] = min(8, s - 8 * p); }
            carry += __shfl_sync(0xFFFFFFFFu, inc, 31);
        }
        if (lane == 0) tb.nva = carry;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < RPT; j++) {
        if (id[j] >= 0) {
            const int k = wcnt[w + j * NW][id[j]] + rank[j];
            tb.va_rt[vbase[id[j]] + (k >> 3)][k & 7] = tid + j * NTHR;
        }
    }
    __syncthreads();
}

// KIND 1: gate/up (W13, 64 w-tiles; B = x rows of the tokens) -> h[va * 8 + n][f] = silu(gate) * up (pg gg1 math)
// KIND 2: down (W2, 32 w-tiles; B = h rows) -> y[route][row] = acc * 2^8 (pg gg2 math)
// Work: slabs = 16 weight rows (one warp's piece of a pg block) x full K of the passes, an equal contiguous range per
// CTA; inside the CTA the 16 warps split each slab's K; their partials meet in a 4-slot smem ring and one warp per slab
// (slab % 16) sums them in warp order (deterministic) one slab later and runs the epilogue.
// Pre-expert RMSNorm fold (KIND 1, gamma != null): xin holds the un-normed rows. At each pass switch one thread bulk-
// copies the pass's token rows into shared memory (the bytes the B loads read anyway, once per CTA); warps 0..7 take
// each token's rstd there in torch's vectorized rms_norm order (pg_gather_norm's: per virtual lane l of 128 an fma
// chain over bf16x4 vectors l + 128 j, shfl_down 16 .. 1, warps (0 + 2) + (1 + 3), / 4096, rsqrtf(. + eps)) and every
// B word becomes bf16(gamma * (rstd * x)): exactly the values F.rms_norm returns, so the products are the unfused ones.
DINL float rms_rstd(const __nv_bfloat16* xrow, float eps, int lane) {   // xrow: a staged row in shared memory
    const uint2* row = reinterpret_cast<const uint2*>(xrow);
    const int vy = lane >> 3, vi = lane & 7;
    float sq[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint2 v = row[128 * j + 32 * vy + vi + 8 * q];
            const float2 lo = __bfloat1622float2(u32b2(v.x));
            const float2 hi = __bfloat1622float2(u32b2(v.y));
            sq[q] = __fmaf_rn(lo.x, lo.x, sq[q]);
            sq[q] = __fmaf_rn(lo.y, lo.y, sq[q]);
            sq[q] = __fmaf_rn(hi.x, hi.x, sq[q]);
            sq[q] = __fmaf_rn(hi.y, hi.y, sq[q]);
        }
    float c = __fadd_rn(__fadd_rn(sq[0], sq[2]), __fadd_rn(sq[1], sq[3]));   // steps 16, 8
    c = __fadd_rn(c, __shfl_down_sync(0xFFFFFFFFu, c, 4));
    c = __fadd_rn(c, __shfl_down_sync(0xFFFFFFFFu, c, 2));
    c = __fadd_rn(c, __shfl_down_sync(0xFFFFFFFFu, c, 1));
    const float w0 = __shfl_sync(0xFFFFFFFFu, c, 0), w1 = __shfl_sync(0xFFFFFFFFu, c, 8);
    const float w2 = __shfl_sync(0xFFFFFFFFu, c, 16), w3 = __shfl_sync(0xFFFFFFFFu, c, 24);
    const float m2 = __fdiv_rn(__fadd_rn(__fadd_rn(w0, w2), __fadd_rn(w1, w3)), 4096.f);
    return rsqrtf(__fadd_rn(m2, eps));
}

// norm fold staging (dynamic smem): two buffers of a pass's <= 8 token rows (one bulk copy each, row pitch padded by
// 16 B: conflict-free B reads), the next pass's rows prefetched into the idle buffer two slabs ahead; gamma once.
constexpr int XROW = H + 8;
constexpr int XBUF = 8 * XROW;                    // bf16 elements per buffer
constexpr int XSTAGE_BYTES = (2 * XBUF + H) * 2;
DINL void bulk_row(unsigned sdst, const void* gsrc, unsigned bytes, unsigned bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sdst), "l"(gsrc), "r"(bytes), "r"(bar) : "memory");
}
DINL void mbar_expect_tx(unsigned bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}

DINL unsigned rms_word(unsigned xw, unsigned gw, float rs) {
    const float2 x2 = __bfloat1622float2(u32b2(xw)), g2 = __bfloat1622float2(u32b2(gw));
    return b2u32(__floats2bfloat162_rn(__fmul_rn(g2.x, __fmul_rn(rs, x2.x)), __fmul_rn(g2.y, __fmul_rn(rs, x2.y))));
}

template <int KIND>
__global__ void __launch_bounds__(NTHR, 1) st_gemm(const __nv_bfloat16* __restrict__ xin, const char* __restrict__ wsf,
                                                   const float* __restrict__ gs, __nv_bfloat16* __restrict__ out,
                                                   const int* __restrict__ ids, int numel, int nl, int pdl,
                                                   const __nv_bfloat16* __restrict__ gamma, float eps,
                                                   const int* __restrict__ rtab, int n_rep, int me, int passes,
                                                   int* __restrict__ lids) {
    constexpr int NWT = KIND == 1 ? 64 : 32;
    constexpr int LSPE = KIND == 1 ? 9 : 8;            // log2 slabs per pass (NWT x 8 slabs)
    __shared__ Tab tb;
    __shared__ __align__(16) float4 part[R][NW][32];   // per-warp slab partials
    __shared__ __align__(16) unsigned short scr[NW][128];
    __shared__ __align__(8) unsigned long long bars[2 * R];   // full[R] | empty[R]
    __shared__ unsigned short s_wcnt[NUMEL_MAX / 32][ELOC];
    __shared__ int s_vbase[ELOC];
    __shared__ float s_rs[2][8];                       // norm fold: rstd of a pass's tokens, per staging buffer
    __shared__ __align__(8) unsigned long long s_xbar[2];   // norm fold: one staging barrier per buffer
    extern __shared__ __align__(16) __nv_bfloat16 s_x[];   // norm fold: [2][8][XROW] staged rows, gamma [H]
    __shared__ int s_rtab[256], s_rcnt[256], s_rown[256], s_rord[256], s_rload[8];   // replica balancer fold

    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    if (KIND == 1 && pdl) pdl_trigger();               // KIND 2 CTAs only fit on SMs this kernel has left
    if (tid < R) {
        mbar_init(smem_u32(&bars[tid]), NTHR);
        mbar_init(smem_u32(&bars[R + tid]), 32);
    }
    if (tid == R || tid == R + 1) mbar_init(smem_u32(&s_xbar[tid - R]), 1);
    if (rtab != nullptr) rep_owner<RPT>(ids, numel, rtab, n_rep, passes, s_rtab, s_rcnt, s_rown, s_rord, s_rload);
    build_tab(ids, numel, nl, KIND == 1 ? gs : nullptr, tb, s_wcnt, s_vbase, rtab != nullptr ? s_rtab : nullptr,
              s_rown, me, KIND == 1 && blockIdx.x == 0 ? lids : nullptr);   // ends with __syncthreads

    const int U = tb.nva << LSPE;
    const int s0 = (int)(((long long)U * blockIdx.x) / gridDim.x);
    const int nslab = (int)(((long long)U * (blockIdx.x + 1)) / gridDim.x) - s0;
    if (nslab <= 0) {
        if (KIND == 2 && pdl) pdl_wait();              // completion still implies K1's
        return;
    }
    const unsigned fullb = smem_u32(&bars[0]), emptyb = smem_u32(&bars[R]);

    // weight piece of slab jl, k-tile i: block (e, wt, KPW*w + i), warp-slab wi: codes at + wi*512 + lane*16,
    // scales at + 4096 + (wi*8 + g)*SFB
    auto slab_ptr = [&](int jl) -> const char* {
        const int s = s0 + min(jl, nslab - 1);
        const int va = s >> LSPE, loc = s & ((1 << LSPE) - 1);
        const int e = tb.va_e[va], wt = loc >> 3, wi = loc & 7;
        return wsf + ((size_t)(e * NWT + wt) * NKT + KPW * w) * WSB + wi * 512;
    };
    const int lofs = lane * 16, sofs = 4096 + g * SFB;
    auto sofs_of = [&](int jl) -> int { return sofs - ((s0 + min(jl, nslab - 1)) & 7) * (512 - 8 * SFB); };

    // one slab (4 k-tiles x the warp's piece) in flight per warp: the loop is decode-bound, and a second slab in
    // registers measured 2.5-3 % slower (serving/kernels/smallt-moe LOG E10)
    uint4 wr[KPW];
    sf_t sr[KPW];
    {
        const char* p = slab_ptr(0);
        const int so = sofs_of(0);
#pragma unroll
        for (int i = 0; i < KPW; i++) {
            wr[i] = ldw(p + (size_t)i * WSB + lofs);
            sr[i] = lds8(p + (size_t)i * WSB + so);
        }
    }
    if (KIND == 2 && pdl) pdl_wait();                  // the intermediate is complete from here on

    unsigned b[KPW][4][2];
    int cva = -1;
    // norm fold: buffer of the next switch, the pass whose rows are in flight to it (-1: none), barrier phases.
    // A buffer is refilled only for the switch after next, issued after this switch's barriers: its readers are done.
    unsigned xph0 = 0, xph1 = 0;
    int nbuf = 0, pva = -1;
    auto stage = [&](int va, int bi, bool with_gamma) {   // thread 0
        const int nv = tb.va_n[va];
        const unsigned xb = smem_u32(&s_xbar[bi]), xs = smem_u32(s_x + bi * XBUF);
        mbar_expect_tx(xb, (nv + (with_gamma ? 1 : 0)) * H * 2);
        if (with_gamma) bulk_row(smem_u32(s_x + 2 * XBUF), gamma, H * 2, xb);
        for (int n = 0; n < nv; n++) bulk_row(xs + n * XROW * 2, xin + (size_t)(tb.va_rt[va][n] >> 3) * H, H * 2, xb);
    };
    if (KIND == 1 && gamma != nullptr) {
        const int va0 = s0 >> LSPE;
        if (tid == 0) stage(va0, 0, true);
        pva = va0;
    }
    auto load_b = [&](int va) {
        const bool live = g < tb.va_n[va];             // padded token columns stay zero (their outputs are dropped)
        const __nv_bfloat16* row = KIND == 1 ? xin + (size_t)(tb.va_rt[va][live ? g : 0] >> 3) * H
                                             : xin + (size_t)(va * 8 + g) * H;
        const __nv_bfloat16* grow = gamma;
        float rs = 0.f;
        if (KIND == 1 && gamma != nullptr) {           // every warp reaches this switch at the same slab
            const int nv = tb.va_n[va], bi = nbuf;
            if (pva != va && tid == 0) stage(va, bi, false);   // not prefetched (a pass shorter than 2 slabs)
            pva = -1;
            nbuf ^= 1;
            mbar_wait(smem_u32(&s_xbar[bi]), bi ? xph1 : xph0);
            if (bi) xph1 ^= 1; else xph0 ^= 1;
            const __nv_bfloat16* xb = s_x + bi * XBUF;
            if (w < nv) {
                const float r = rms_rstd(xb + w * XROW, eps, lane);
                if (lane == 0) s_rs[bi][w] = r;
            }
            __syncthreads();
            if (live) rs = s_rs[bi][g];
            row = xb + g * XROW;                       // B words and gamma from shared memory
            grow = s_x + 2 * XBUF;
        }
        row += KPW * w * 64 + 2 * t;
        grow += KPW * w * 64 + 2 * t;
#pragma unroll
        for (int i = 0; i < KPW; i++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                if (KIND == 1) {
                    b[i][q][0] = live ? ldb32(row + i * 64 + 16 * q) : 0u;
                    b[i][q][1] = live ? ldb32(row + i * 64 + 16 * q + 8) : 0u;
                    if (gamma != nullptr && live) {
                        b[i][q][0] = rms_word(b[i][q][0], ldb32(grow + i * 64 + 16 * q), rs);
                        b[i][q][1] = rms_word(b[i][q][1], ldb32(grow + i * 64 + 16 * q + 8), rs);
                    }
                } else {
                    b[i][q][0] = live ? ldb32cg(row + i * 64 + 16 * q) : 0u;
                    b[i][q][1] = live ? ldb32cg(row + i * 64 + 16 * q + 8) : 0u;
                }
            }
    };

    auto reduce = [&](int jr) {
        const int slot = jr % R;
        mbar_wait(fullb + slot * 8, (jr / R) & 1);
        float4 v = part[slot][0][lane];
#pragma unroll
        for (int ww = 1; ww < NW; ww++) {
            const float4 p = part[slot][ww][lane];
            v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
        }
        mbar_arrive(emptyb + slot * 8);
        const int s = s0 + jr;
        const int va = s >> LSPE, loc = s & ((1 << LSPE) - 1);
        const int wt = loc >> 3, wi = loc & 7;
        if (KIND == 1) {
            // pg gg1 epilogue: g / u = acc * gs * 2^8, silu(g) u = (g u / 2)(1 + tanh(g / 2))
            const float gsc = tb.gs[tb.va_e[va]] * 256.0f;
            const float hgs = 0.5f * gsc, gus = gsc * gsc * 0.5f;
            float t0, t1;
            asm("tanh.approx.f32 %0, %1;" : "=f"(t0) : "f"(v.x * hgs));
            asm("tanh.approx.f32 %0, %1;" : "=f"(t1) : "f"(v.y * hgs));
            const float p0 = v.x * v.z * gus, p1 = v.y * v.w * gus;
            scr[w][(2 * t) * 8 + g] = __bfloat16_as_ushort(__float2bfloat16_rn(fmaf(p0, t0, p0)));
            scr[w][(2 * t + 1) * 8 + g] = __bfloat16_as_ushort(__float2bfloat16_rn(fmaf(p1, t1, p1)));
            __syncwarp();
            if (lane < tb.va_n[va])                    // padded rows are never read (gg2 zeroes those columns)
                *reinterpret_cast<uint4*>(out + (size_t)(va * 8 + lane) * H + wt * 64 + wi * 8) =
                    *reinterpret_cast<const uint4*>(&scr[w][lane * 8]);
        } else {
            const __nv_bfloat162 r0 = __floats2bfloat162_rn(v.x * 256.0f, v.y * 256.0f);   // row g, n 2t / 2t+1
            const __nv_bfloat162 r1 = __floats2bfloat162_rn(v.z * 256.0f, v.w * 256.0f);   // row g + 8
            scr[w][(2 * t) * 16 + g] = __bfloat16_as_ushort(r0.x);
            scr[w][(2 * t + 1) * 16 + g] = __bfloat16_as_ushort(r0.y);
            scr[w][(2 * t) * 16 + g + 8] = __bfloat16_as_ushort(r1.x);
            scr[w][(2 * t + 1) * 16 + g + 8] = __bfloat16_as_ushort(r1.y);
            __syncwarp();
            const int n = lane >> 1, hf = lane & 1;
            if (lane < 16 && n < tb.va_n[va])
                *reinterpret_cast<uint4*>(out + (size_t)tb.va_rt[va][n] * H + wt * 128 + wi * 16 + hf * 8) =
                    *reinterpret_cast<const uint4*>(&scr[w][n * 16 + hf * 8]);
        }
        __syncwarp();
    };

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int jl = 0; jl < nslab; jl++) {
        const int va = (s0 + jl) >> LSPE;
        if (va != cva) { cva = va; load_b(va); }
        if (KIND == 1 && gamma != nullptr && pva < 0) {   // prefetch the next pass's rows two slabs ahead
            const int va2 = (s0 + min(jl + 2, nslab - 1)) >> LSPE;
            if (va2 != cva) {
                if (tid == 0) stage(va2, nbuf, false);
                pva = va2;
            }
        }
        const char* p = slab_ptr(jl + 1);
        const int so = sofs_of(jl + 1);
#pragma unroll
        for (int i = 0; i < KPW; i++) {
            ktile(acc, wr[i], sr[i], b[i]);
            wr[i] = ldw(p + (size_t)i * WSB + lofs);     // unconditional refill (the last slab re-reads itself)
            sr[i] = lds8(p + (size_t)i * WSB + so);
        }
        const int slot = jl % R;
        if (jl >= R) mbar_wait(emptyb + slot * 8, ((jl / R) - 1) & 1);
        part[slot][w][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        mbar_arrive(fullb + slot * 8);
        acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        if (jl >= 1 && (jl - 1) % NW == w) reduce(jl - 1);
    }
    if ((nslab - 1) % NW == w) reduce(nslab - 1);
}

}  // namespace stk

static int st_num_sms() {
    static int n = 0;
    if (!n) n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    return n;
}

// y [numel, 4096] bf16 (local routes written), h scratch [>= (nl + numel / 8) * 8, 4096] bf16, nl = wsf13.size(0)
// gamma: the pre-expert RMSNorm weight to fold (bf16 [4096], x un-normed) or an empty tensor (x already normed)
// tab: pg_rep_assign's int32 table [256 + n_rep] (ids = global top-k ids; lids [numel] gets the per-route slots) or
// an empty tensor (ids = local slots)
void st_moe(torch::Tensor x, torch::Tensor ids, torch::Tensor wsf13, torch::Tensor gs13, torch::Tensor wsf2,
            torch::Tensor h, torch::Tensor y, int64_t pdl, torch::Tensor gamma, double eps, torch::Tensor tab,
            int64_t me, int64_t passes, torch::Tensor lids) {
    const int numel = (int)ids.numel();
    TORCH_CHECK(numel <= stk::NUMEL_MAX && numel == x.size(0) * 8 && x.size(1) == stk::H, "st: T <= 128, top-8, H 4096");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.is_contiguous() && x.is_contiguous(), "st: int32 ids, contiguous x");
    const int nl = (int)wsf13.size(0);
    TORCH_CHECK(nl >= 1 && nl <= stk::ELOC && wsf13.size(1) == 64 && wsf13.size(2) == stk::NKT &&
                wsf13.size(3) == stk::WSB, "st: wsf13 layout (local slots <= 64, block bytes = this build's WSB)");
    TORCH_CHECK(wsf2.size(0) == nl && wsf2.size(1) == 32 && wsf2.size(2) == stk::NKT && wsf2.size(3) == stk::WSB,
                "st: wsf2 layout");
    TORCH_CHECK(h.size(0) >= (nl + numel / 8) * 8 && gs13.numel() == nl, "st: h rows / gs13");
    const bool fold = gamma.numel() > 0;
    TORCH_CHECK(!fold || (gamma.numel() == stk::H && gamma.scalar_type() == at::kBFloat16 && gamma.is_contiguous()),
                "st: norm weight bf16 [4096]");
    const auto* gp = fold ? reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()) : nullptr;
    const bool rep = tab.numel() > 0;
    const int n_rep = rep ? (int)tab.numel() - 256 : 0;
    TORCH_CHECK(!rep || (tab.scalar_type() == at::kInt && tab.is_contiguous() && n_rep >= 0 && n_rep <= 256 &&
                         me >= 0 && me < 8 && lids.scalar_type() == at::kInt && lids.is_contiguous() &&
                         lids.numel() == numel),
                "st: replica table int32 [256 + n_rep], rank < 8, int32 lids like ids");
    const int* rp = rep ? tab.data_ptr<int>() : nullptr;
    int* lp = rep ? lids.data_ptr<int>() : nullptr;
    const int G = st_num_sms();
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto* hp = reinterpret_cast<__nv_bfloat16*>(h.data_ptr());
    auto* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
    const int* ip = ids.data_ptr<int>();
    static bool attr = false;
    if (!attr) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(stk::st_gemm<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            stk::XSTAGE_BYTES));
        attr = true;
    }
    stk::st_gemm<1><<<G, stk::NTHR, fold ? stk::XSTAGE_BYTES : 0, st>>>(xp, reinterpret_cast<const char*>(wsf13.data_ptr()),
                                            gs13.data_ptr<float>(), hp, ip, numel, nl, (int)pdl, gp, (float)eps,
                                            rp, n_rep, (int)me, (int)passes, lp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (pdl) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(G);
        cfg.blockDim = dim3(stk::NTHR);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = st;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at;
        cfg.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, stk::st_gemm<2>, (const __nv_bfloat16*)hp,
                                          reinterpret_cast<const char*>(wsf2.data_ptr()), (const float*)nullptr, yp, ip,
                                          numel, nl, 1, (const __nv_bfloat16*)nullptr, 0.f, rp, n_rep, (int)me,
                                          (int)passes, (int*)nullptr));
    } else {
        stk::st_gemm<2><<<G, stk::NTHR, 0, st>>>(hp, reinterpret_cast<const char*>(wsf2.data_ptr()), nullptr, yp, ip,
                                                numel, nl, 0, nullptr, 0.f, rp, n_rep, (int)me, (int)passes, nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
"""

_CUDA_SRC = _CUDA_SRC.replace("@@REP@@", _CUDA_SRC_REP)
