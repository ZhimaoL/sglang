"""muse_moe_w4a16 v14 CUDA source, byte-identical to the certified kernel-track lane
(same load_inline name and flags -> torch_extensions build-cache hit)."""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DINL __device__ __forceinline__
#ifndef DQ_ON
#define DQ_ON 1                // 0 = stream-only bandwidth probe (diagnostics)
#endif
#ifndef MMA_ON
#define MMA_ON 1               // 0 = skip tensor-core MMAs (diagnostics)
#endif
#ifndef BSTAGE_ON
#define BSTAGE_ON 1            // 0 = skip B slab staging + reads (diagnostics)
#endif
constexpr int HD = 4096;       // d_hidden = d_expert = 4096 (guarded host-side)
constexpr int TC = 8;          // token chunk per pass

DINL void cpa16(void* smem, const void* gmem) {   // 16B L2->smem async copy
    unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sm), "l"(gmem));
}
DINL void cpa_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> DINL void cpa_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

DINL __half2 u32h2(unsigned int v) { return *reinterpret_cast<__half2*>(&v); }

// 8 e2m1 values (pre-permuted nibbles k0,k2,k4,k6,k1,k3,k5,k7) -> 4 half2
// (k0,k1),(k2,k3),(k4,k5),(k6,k7), each value * 2^-14 exactly.
DINL void dq8(unsigned int t, __half2 d[4]) {
    // multiplies (= shifts mod 2^32) so ptxas can issue them on the FMA pipe
    // (IMAD) instead of the saturated INT/SHF pipe.
    unsigned int o0 = ((t * 512u)  & 0x0E000E00u) | ((t * 4096u) & 0x80008000u);
    unsigned int o1 = ((t * 32u)   & 0x0E000E00u) | ((t * 256u)  & 0x80008000u);
    unsigned int o2 = ((t * 2u)    & 0x0E000E00u) | ((t * 16u)   & 0x80008000u);
    unsigned int o3 = ((t >> 3)    & 0x0E000E00u) | ( t          & 0x80008000u);
    d[0] = u32h2(o0); d[1] = u32h2(o1); d[2] = u32h2(o2); d[3] = u32h2(o3);
}

DINL __half2 sfh2(unsigned short v) {   // 2 x e4m3 -> 2 x fp16 (exact)
    unsigned int r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return u32h2(r);
}

// --------------------------- k0x: x bf16 -> fp16 (one-time, tiny)
__global__ void k0x_kernel(const __nv_bfloat16* __restrict__ x, __half* __restrict__ xh, int n8) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n8) {
        const __nv_bfloat162* s = reinterpret_cast<const __nv_bfloat162*>(x) + (long long)i * 4;
        __half2* d = reinterpret_cast<__half2*>(xh) + (long long)i * 4;
#pragma unroll
        for (int k = 0; k < 4; k++) d[k] = __float22half2_rn(__bfloat1622float2(s[k]));
    }
}

// dot of 4 consecutive weight rows (2048B each, sf rows 256B) against CNT tokens
// staged in smem at half-offsets xoff[t]; weights stream L2->smem via a
// per-thread cp.async double buffer (wsb: uint4[2 stages][4 rows][256 threads],
// each thread touches only its own slot). af[r][t] = full-warp sums.
template <int CNT>
DINL void dot4(const unsigned char* __restrict__ w0, const unsigned char* __restrict__ s0,
               const __half* __restrict__ xg, const int* xoff, int lane, int tid, uint4* wsb,
               float af[4][TC]) {
    __half2 acc[4][CNT];
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int t = 0; t < CNT; t++) { acc[r][t] = u32h2(0u); af[r][t] = 0.f; }
#pragma unroll
    for (int p = 0; p < 2; p++) {
#pragma unroll
        for (int r = 0; r < 4; r++)
            cpa16(wsb + (p * 4 + r) * 256 + tid, w0 + r * 2048 + (lane + 32 * p) * 16);
        cpa_commit();
    }
    unsigned short sv[4];
#pragma unroll
    for (int r = 0; r < 4; r++)
        sv[r] = *reinterpret_cast<const unsigned short*>(s0 + r * 256 + lane * 2);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int st = j % 3;
        if (j < 2) {
            const int nst = (j + 2) % 3;
            const int ncol = lane + 32 * (j + 2);
#pragma unroll
            for (int r = 0; r < 4; r++)
                cpa16(wsb + (nst * 4 + r) * 256 + tid, w0 + r * 2048 + ncol * 16);
            cpa_commit();
        }
        __half2 sl[4], sh[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            const __half2 sc2 = sfh2(sv[r]);
            sl[r] = __half2half2(__low2half(sc2));
            sh[r] = __half2half2(__high2half(sc2));
        }
        if (j < 3) {
            const int ncol = lane + 32 * (j + 1);
#pragma unroll
            for (int r = 0; r < 4; r++)
                sv[r] = *reinterpret_cast<const unsigned short*>(s0 + r * 256 + ncol * 2);
        }
        if (j < 2) cpa_wait<2>(); else if (j == 2) cpa_wait<1>(); else cpa_wait<0>();
        __syncwarp();
        uint4 wv[4];
#pragma unroll
        for (int r = 0; r < 4; r++) wv[r] = wsb[(st * 4 + r) * 256 + tid];
        const int col = lane + 32 * j;
#if !DQ_ON
        af[0][0] += (float)((wv[0].x & 1u) + (wv[1].y & 1u) + (wv[2].z & 1u) + (wv[3].w & 1u))
                    + __low2float(sl[0]) + __low2float(sh[3]);
        if (false)
#endif
#pragma unroll
        for (int p = 0; p < 4; p++) {
            __half2 ds[4][4];
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const unsigned int q = (&wv[r].x)[p];
                __half2 d[4]; dq8(q, d);
                const __half2 sc = (p < 2) ? sl[r] : sh[r];
#pragma unroll
                for (int k = 0; k < 4; k++) ds[r][k] = __hmul2(d[k], sc);
            }
            const int u = col * 4 + p;
#pragma unroll
            for (int t = 0; t < CNT; t++) {
                const uint4 xv = *reinterpret_cast<const uint4*>(xg + xoff[t] + u * 8);
                const __half2* x2 = reinterpret_cast<const __half2*>(&xv);
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    acc[r][t] = __hfma2(ds[r][0], x2[0], acc[r][t]);
                    acc[r][t] = __hfma2(ds[r][1], x2[1], acc[r][t]);
                    acc[r][t] = __hfma2(ds[r][2], x2[2], acc[r][t]);
                    acc[r][t] = __hfma2(ds[r][3], x2[3], acc[r][t]);
                }
            }
        }
        if (j & 1) {   // fold fp16 partials to fp32 every 2 cols (64 values/row)
#pragma unroll
            for (int r = 0; r < 4; r++)
#pragma unroll
                for (int t = 0; t < CNT; t++) {
                    af[r][t] += __low2float(acc[r][t]) + __high2float(acc[r][t]);
                    acc[r][t] = u32h2(0u);
                }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int t = 0; t < CNT; t++) {
            float v = af[r][t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            af[r][t] = v;
        }
}

#define DOT4_SWITCH(cn, w0, s0, xs, xoff, lane, tid, wsb, af)            \
    switch (cn) {                                                        \
        case 1: dot4<1>(w0, s0, xs, xoff, lane, tid, wsb, af); break;    \
        case 2: dot4<2>(w0, s0, xs, xoff, lane, tid, wsb, af); break;    \
        case 3: dot4<3>(w0, s0, xs, xoff, lane, tid, wsb, af); break;    \
        default: dot4<4>(w0, s0, xs, xoff, lane, tid, wsb, af); break;   \
    }

// ------------------------- K1: grouped gemv1 (gate/up interleaved) + silu*mul
// Persistent CTAs: grid = NCTA, items = (active expert, 32-row tile), contiguous
// static partition so each CTA sees runs of the same expert (x restaged on change).
__global__ void __launch_bounds__(256, 2) k1_kernel(
    const unsigned char* __restrict__ w, const unsigned char* __restrict__ sf,
    const float* __restrict__ gs, const int* __restrict__ counts,
    const int* __restrict__ act, const int* __restrict__ actn,
    const int* __restrict__ tok, const __half* __restrict__ xh,
    __half* __restrict__ inter, int MP, int ls, int M) {
    extern __shared__ char smem[];
    uint4* ws = reinterpret_cast<uint4*>(smem);                         // 3 stages = 48KB
    __half (*sint)[TC] = reinterpret_cast<__half (*)[TC]>(smem + 49152);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int n_items = actn[0] * 256;
    const int cmax = (M <= TC) ? TC : 4;
    for (int it = blockIdx.x; it < n_items; it += gridDim.x) {
        const int ei = it >> 8, tile = it & 255;
        const int e = act[ei];
        const int cnt = counts[e];
        if (cnt > cmax) continue;               // heavy experts go to the MMA path
        const long long eg = ls + e;
        const unsigned char* w0 = w + eg * (8192LL * 2048) + (long long)(tile * 32 + wid * 4) * 2048;
        const unsigned char* s0 = sf + eg * (8192LL * 256) + (long long)(tile * 32 + wid * 4) * 256;
        const float gsv = gs[eg] * 16384.f;
        const int nb = tile * 16;
        for (int tc0 = 0; tc0 < cnt; tc0 += 4) {
            const int cn = min(cnt - tc0, 4);
            const int slot0 = e * MP + tc0;
            int xoff[4];
#pragma unroll
            for (int t = 0; t < 4; t++)
                xoff[t] = (t < cn) ? tok[slot0 + t] * HD : 0;
            float af[4][TC];
            DOT4_SWITCH(cn, w0, s0, xh, xoff, lane, tid, ws, af);
            if (lane < cn) {
#pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    float g = af[2 * pr][0], u = af[2 * pr + 1][0];
#pragma unroll
                    for (int t = 1; t < 4; t++)
                        if (lane == t) { g = af[2 * pr][t]; u = af[2 * pr + 1][t]; }
                    g *= gsv; u *= gsv;
                    sint[wid * 2 + pr][lane] = __float2half(g / (1.f + __expf(-g)) * u);
                }
            }
            __syncthreads();
            for (int i = tid; i < 16 * cn; i += 256)
                inter[(long long)(slot0 + (i >> 4)) * HD + nb + (i & 15)] = sint[i & 15][i >> 4];
            __syncthreads();
        }
    }
}

// ------------------- K2: grouped gemv2 * (1+w) -> z fp16 + ssq partials
__global__ void __launch_bounds__(256, 2) k2_kernel(
    const unsigned char* __restrict__ w, const unsigned char* __restrict__ sf,
    const float* __restrict__ pw, const int* __restrict__ counts,
    const int* __restrict__ act, const int* __restrict__ actn,
    const __half* __restrict__ inter, __half* __restrict__ z,
    float* __restrict__ ssp, int MP, int ls, int M) {
    extern __shared__ char smem[];
    uint4* ws = reinterpret_cast<uint4*>(smem);                         // 3 stages = 48KB
    __half (*sz)[TC] = reinterpret_cast<__half (*)[TC]>(smem + 49152);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int n_items = actn[0] * 128;
    const int cmax = (M <= TC) ? TC : 4;
    for (int it = blockIdx.x; it < n_items; it += gridDim.x) {
        const int ei = it >> 7, tile = it & 127;
        const int e = act[ei];
        const int cnt = counts[e];
        if (cnt > cmax) continue;               // heavy experts go to the MMA path
        const long long eg = ls + e;
        const unsigned char* w0 = w + eg * (4096LL * 2048) + (long long)(tile * 32 + wid * 4) * 2048;
        const unsigned char* s0 = sf + eg * (4096LL * 256) + (long long)(tile * 32 + wid * 4) * 256;
        const int nb = tile * 32;
        for (int tc0 = 0; tc0 < cnt; tc0 += 4) {
            const int cn = min(cnt - tc0, 4);
            const int slot0 = e * MP + tc0;
            int xoff[4];
#pragma unroll
            for (int t = 0; t < 4; t++) xoff[t] = (slot0 + t) * HD;
            float af[4][TC];
            DOT4_SWITCH(cn, w0, s0, inter, xoff, lane, tid, ws, af);
            if (lane < cn) {
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    float v = af[r][0];
#pragma unroll
                    for (int t = 1; t < 4; t++)
                        if (lane == t) v = af[r][t];
                    sz[wid * 4 + r][lane] = __float2half(v * pw[nb + wid * 4 + r]);
                }
            }
            __syncthreads();
            if (tid < cn) {
                float s = 0.f;
#pragma unroll
                for (int m = 0; m < 32; m++) { const float v = __half2float(sz[m][tid]); s += v * v; }
                ssp[(long long)(slot0 + tid) * 128 + tile] = s;
            }
            for (int i = tid; i < 32 * cn; i += 256)
                z[(long long)(slot0 + (i >> 5)) * HD + nb + (i & 31)] = sz[i & 31][i >> 5];
            __syncthreads();
        }
    }
}

// ===================== MMA path (M > 32: verify16/32) ========================
// Warp computes m16 (weight rows) x n-tiles of 8 tokens x k4096 with
// mma.m16n8k16. A streams L2->smem via a 4-deep per-thread cp.async ring from
// the offline-packed fragment layout (u32 per (lane, k16) = 8 e2m1 values,
// decoded by the same dq8; sf pairs (row, row+8) per k16 via broadcast u16
// loads). B (x or inter rows) is staged per CTA in a double-buffered slab per
// 64-k group: uint2 per (token, k16, kquad) = the two k-pair half2 b-frags.
constexpr int NTMAX = 7;
constexpr int TCM = NTMAX * 8;   // 56 tokens per chunk
constexpr int KGM = 64;          // 4096 / 64

DINL void mma16816(float c[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                   unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

DINL unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned*>(&h); }

constexpr int TSTR = 20;                       // slab token stride (uint2): 16 units + pad=4
constexpr int GKG = 4;                         // k16-groups staged per sync round
constexpr int SLABU = GKG * TCM * TSTR;        // uint2 units per stage (4480)

DINL void cpa4(void* smem, const void* gmem) {
    unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sm), "l"(gmem));
}

// async-stage one 4-kg group of B rows into slab (f16 source, token rows via tok
// or slot-contiguous). Zero-fill for padded tokens happens via a guarded branch.
template <bool TOKIDX>
DINL void stage_bg(uint2* slab, const __half* __restrict__ src, const int* __restrict__ tok,
                   int slot0, int ntok, int cnt_c, int kg0, int tid) {
    for (int i = tid; i < GKG * ntok * 16; i += 512) {
        const int u = i & 15, t = (i >> 4) % ntok, g = i / (16 * ntok);
        const int b = u >> 2, kp = u & 3;
        uint2* dst = slab + ((g * TCM + t) * TSTR + u);
        if (t < cnt_c) {
            const int k0 = ((kg0 + g) * 4 + b) * 16 + kp * 2;
            const long long row = TOKIDX ? (long long)tok[slot0 + t] * HD
                                         : (long long)(slot0 + t) * HD;
            cpa4(&dst->x, src + row + k0);
            cpa4(&dst->y, src + row + k0 + 8);
        } else {
            *dst = make_uint2(0u, 0u);
        }
    }
}

// each warp stages its own 16-row group's A (wm0 is per-warp already)
DINL void stage_ag(uint4* wsa, const unsigned char* wm0, int kg0, int lane, int tid) {
#pragma unroll
    for (int g = 0; g < GKG; g++)
        cpa16(wsa + g * 512 + tid, wm0 + (((kg0 + g) * 32 + lane) << 4));
}

// one (expert, 128-row tile, <=56-token chunk): k-loop + epilogue, all
// compile-time indexed so acc stays in registers. KIND 1 = gemm1+silu (writes
// sint[64][TCM]), 2 = gemm2*(1+pw) (writes sz[128][TCM]).
// A and B both stream via cp.async, double-buffered per 4-kg group with ONE
// commit each; sf prefetched one group ahead as 4x LDG.64 (sfm2 layout).
template <int NT, int KIND>
DINL void mma_item(const unsigned char* wm0, const unsigned char* sf0,
                   const __half* __restrict__ bsrc, const int* tok, int slot0, int cnt_c,
                   int lane, int wid, int tid, uint4* wsa, uint2* bslab,
                   float gsv, float pw0, float pw1, __half (*sout)[TCM]) {
    constexpr bool TOKIDX = (KIND == 1);
    constexpr int NG = KGM / GKG;              // 16 groups
    const int ntok = NT * 8;
    const int gr = lane >> 2;
    float acc[NT][4];
#pragma unroll
    for (int q = 0; q < NT; q++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[q][j] = 0.f;
    // prologue: group 0 in flight (A then B, one commit each)
    stage_ag(wsa, wm0, 0, lane, tid);
    cpa_commit();
    if (BSTAGE_ON) stage_bg<TOKIDX>(bslab, bsrc, tok, slot0, ntok, cnt_c, 0, tid);
    cpa_commit();
    uint2 sv64[GKG];
#pragma unroll
    for (int g = 0; g < GKG; g++)
        sv64[g] = *reinterpret_cast<const uint2*>(sf0 + g * 64 + gr * 8);
#pragma unroll 1
    for (int gi = 0; gi < NG; gi++) {
        const int st = gi & 1, nst = st ^ 1;
        if (gi + 1 < NG) {
            stage_ag(wsa + nst * (GKG * 512), wm0, (gi + 1) * GKG, lane, tid);
            cpa_commit();
            if (BSTAGE_ON)
                stage_bg<TOKIDX>(bslab + nst * SLABU, bsrc, tok, slot0, ntok, cnt_c,
                                 (gi + 1) * GKG, tid);
            cpa_commit();
        }
        uint2 svn[GKG];
        if (gi + 1 < NG) {
#pragma unroll
            for (int g = 0; g < GKG; g++)
                svn[g] = *reinterpret_cast<const uint2*>(
                    sf0 + ((gi + 1) * GKG + g) * 64 + gr * 8);
        }
        if (gi + 1 < NG) cpa_wait<2>(); else cpa_wait<0>();
        __syncthreads();
        const uint4* wcur = wsa + st * (GKG * 512);
        const uint2* bcur = bslab + st * SLABU;
#pragma unroll
        for (int gk = 0; gk < GKG; gk++) {
            const uint4 av = wcur[gk * 512 + tid];
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const unsigned short sv = (unsigned short)(
                    (b < 2 ? sv64[gk].x >> (16 * b) : sv64[gk].y >> (16 * (b - 2))) & 0xFFFFu);
                const __half2 s2 = sfh2(sv);
                const __half2 sl = __half2half2(__low2half(s2));
                const __half2 sh = __half2half2(__high2half(s2));
                unsigned a0, a1, a2, a3;
#if DQ_ON
                __half2 d[4]; dq8((&av.x)[b], d);
                a0 = h2u(__hmul2(d[0], sl));
                a2 = h2u(__hmul2(d[1], sl));
                a1 = h2u(__hmul2(d[2], sh));
                a3 = h2u(__hmul2(d[3], sh));
#else
                a0 = a1 = a2 = a3 = (&av.x)[b] & 0x3c003c3cu;
                acc[0][0] += __low2float(sl) + __low2float(sh);
#endif
#pragma unroll
                for (int q = 0; q < NT; q++) {
#if BSTAGE_ON
                    const uint2 bb = bcur[(gk * TCM + q * 8 + gr) * TSTR + b * 4 + (lane & 3)];
#else
                    const uint2 bb = make_uint2(0x3c003c00u, 0x3c003c00u);
#endif
#if MMA_ON
                    mma16816(acc[q], a0, a1, a2, a3, bb.x, bb.y);
#else
                    acc[q][0] += (float)(bb.x & 1u) + (float)(a0 & 1u) + (float)(a3 & 1u);
#endif
                }
            }
        }
#pragma unroll
        for (int g = 0; g < GKG; g++) sv64[g] = svn[g];
        __syncthreads();
    }
    // epilogue: compile-time q/j loops; writes masked later by cnt_c bounds
#pragma unroll
    for (int q = 0; q < NT; q++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int t = q * 8 + (lane & 3) * 2 + (j & 1);
            if (KIND == 1) {
                const float vv = acc[q][j] * gsv;
                const float pv = __shfl_xor_sync(0xffffffffu, vv, 4);
                if (!(gr & 1)) {
                    const int nl = wid * 8 + (gr >> 1) + ((j >> 1) ? 4 : 0);
                    sout[nl][t] = __float2half(vv / (1.f + __expf(-vv)) * pv);
                }
            } else {
                const int rr = wid * 16 + gr + ((j >> 1) ? 8 : 0);
                sout[rr][t] = __float2half(acc[q][j] * ((j >> 1) ? pw1 : pw0));
            }
        }
    }
}

#define MMA_SWITCH(nt, K, wm0, sf0, bsrc, tok, slot0, cn, lane, wid, tid, wsa, bslab, gsv, p0, p1, so) \
    switch (nt) {                                                                       \
        case 1: mma_item<1, K>(wm0, sf0, bsrc, tok, slot0, cn, lane, wid, tid, wsa, bslab, gsv, p0, p1, so); break; \
        case 2: mma_item<2, K>(wm0, sf0, bsrc, tok, slot0, cn, lane, wid, tid, wsa, bslab, gsv, p0, p1, so); break; \
        case 3: case 4: mma_item<4, K>(wm0, sf0, bsrc, tok, slot0, cn, lane, wid, tid, wsa, bslab, gsv, p0, p1, so); break; \
        default: mma_item<7, K>(wm0, sf0, bsrc, tok, slot0, cn, lane, wid, tid, wsa, bslab, gsv, p0, p1, so); break; \
    }

constexpr int SMEM1 = 49152 + 16 * TC * 2;
constexpr int SMEM2 = 49152 + 32 * TC * 2;
constexpr int SMEMB = 2 * SLABU * 8;           // 71680
constexpr int SMEMA = 2 * GKG * 512 * 16;      // 65536
// epilogue staging aliases the B slab (consumed by the final group sync)
constexpr int SMEM1M = SMEMA + SMEMB;
constexpr int SMEM2M = SMEMA + SMEMB;


// ------------------ K1m: gemm1 (MMA) + silu*mul, 128-row tiles, 64/expert
__global__ void __launch_bounds__(512, 1) k1m_kernel(
    const unsigned char* __restrict__ wm, const unsigned char* __restrict__ sfm,
    const float* __restrict__ gs, const int* __restrict__ counts,
    const int* __restrict__ act, const int* __restrict__ actn,
    const int* __restrict__ tok, const __half* __restrict__ xh,
    __half* __restrict__ inter, int MP, int ls, int cmin) {
    extern __shared__ char smem[];
    uint4* wsa = reinterpret_cast<uint4*>(smem);                      // SMEMA
    uint2* bslab = reinterpret_cast<uint2*>(smem + SMEMA);            // 2 x SLABU x 8
    __half (*sint)[TCM] = reinterpret_cast<__half (*)[TCM]>(smem + SMEMA);  // aliases slab
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;     // wid: 16-row group
    const int n_items = actn[0] * 32;                                 // 256-row tiles
    for (int it = blockIdx.x; it < n_items; it += gridDim.x) {
        const int ei = it >> 5, tile = it & 31;
        const int e = act[ei];
        const int cnt = counts[e];
        if (cnt < cmin) continue;               // light experts stay on the GEMV path
        const long long eg = ls + e;
        const unsigned char* wm0 = wm + (eg * 512 + tile * 16 + wid) * (64LL * 32 * 16);
        const unsigned char* sf0 = sfm + (eg * 512 + tile * 16 + wid) * (256LL * 16);
        const float gsv = gs[eg] * 16384.f;
        for (int tc0 = 0; tc0 < cnt; tc0 += TCM) {
            const int cn = min(cnt - tc0, TCM);
            const int slot0 = e * MP + tc0;
            const int nt = (cn + 7) >> 3;
            MMA_SWITCH(nt, 1, wm0, sf0, xh, tok, slot0, cn, lane, wid, tid, wsa, bslab,
                       gsv, 0.f, 0.f, sint);
            __syncthreads();
            for (int i = tid; i < 128 * cn; i += 512) {
                const int t = i >> 7, nl = i & 127;
                inter[(long long)(slot0 + t) * HD + tile * 128 + nl] = sint[nl][t];
            }
            __syncthreads();
        }
    }
}

// ------------------ K2m: gemm2 (MMA) * (1+w) -> z + ssq partials (4 segs/tile)
__global__ void __launch_bounds__(512, 1) k2m_kernel(
    const unsigned char* __restrict__ wm, const unsigned char* __restrict__ sfm,
    const float* __restrict__ pw, const int* __restrict__ counts,
    const int* __restrict__ act, const int* __restrict__ actn,
    const __half* __restrict__ inter, __half* __restrict__ z,
    float* __restrict__ ssp, int MP, int ls, int cmin) {
    extern __shared__ char smem[];
    uint4* wsa = reinterpret_cast<uint4*>(smem);
    uint2* bslab = reinterpret_cast<uint2*>(smem + SMEMA);
    __half (*sz)[TCM] = reinterpret_cast<__half (*)[TCM]>(smem + SMEMA);    // aliases slab
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int gr = lane >> 2;
    const int n_items = actn[0] * 16;                                 // 256-row tiles
    for (int it = blockIdx.x; it < n_items; it += gridDim.x) {
        const int ei = it >> 4, tile = it & 15;
        const int e = act[ei];
        const int cnt = counts[e];
        if (cnt < cmin) continue;               // light experts stay on the GEMV path
        const long long eg = ls + e;
        const unsigned char* wm0 = wm + (eg * 256 + tile * 16 + wid) * (64LL * 32 * 16);
        const unsigned char* sf0 = sfm + (eg * 256 + tile * 16 + wid) * (256LL * 16);
        const int rb = tile * 256 + wid * 16;
        const float pw0 = pw[rb + gr], pw1 = pw[rb + gr + 8];
        for (int tc0 = 0; tc0 < cnt; tc0 += TCM) {
            const int cn = min(cnt - tc0, TCM);
            const int slot0 = e * MP + tc0;
            const int nt = (cn + 7) >> 3;
            MMA_SWITCH(nt, 2, wm0, sf0, inter, (const int*)nullptr, slot0, cn, lane, wid,
                       tid, wsa, bslab, 0.f, pw0, pw1, sz);
            __syncthreads();
            if (tid < cn * 8) {
                const int t = tid >> 3, seg = tid & 7;
                float s = 0.f;
#pragma unroll 4
                for (int m = 0; m < 32; m++) {
                    const float zz = __half2float(sz[seg * 32 + m][t]);
                    s += zz * zz;
                }
                ssp[(long long)(slot0 + t) * 128 + tile * 8 + seg] = s;
            }
            for (int i = tid; i < 256 * cn; i += 512) {
                const int t = i >> 8, m = i & 255;
                z[(long long)(slot0 + t) * HD + tile * 256 + m] = sz[m][t];
            }
            __syncthreads();
        }
    }
}

// ---------------- K0f: fused routing (CTA0) + x bf16->fp16 convert (CTAs 1..)
// Replaces the Triton _k0za launch + k0x launch with ONE kernel for the M<=8 path.
__global__ void k0f_kernel(const int* __restrict__ ids, int* __restrict__ slots,
                           int* __restrict__ tok, int* __restrict__ counts,
                           int* __restrict__ act, int* __restrict__ actn,
                           const __nv_bfloat16* __restrict__ x, __half* __restrict__ xh,
                           int n_routes, int MP, int TK, int ls, int lc, int n8) {
    if (blockIdx.x == 0) {
        for (int c = threadIdx.x; c < lc; c += blockDim.x) counts[c] = 0;
        if (threadIdx.x == 0) actn[0] = 0;
        __syncthreads();
        for (int r = threadIdx.x; r < n_routes; r += blockDim.x) {
            const int e = ids[r] - ls;
            int slot = -1;
            if (e >= 0 && e < lc) {
                const int pos = atomicAdd(counts + e, 1);
                if (pos == 0) act[atomicAdd(actn, 1)] = e;
                slot = e * MP + pos;
                tok[slot] = r / TK;
            }
            slots[r] = slot;
        }
    } else {
        const int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
        if (i < n8) {
            const __nv_bfloat162* s = reinterpret_cast<const __nv_bfloat162*>(x) + (long long)i * 4;
            __half2* d = reinterpret_cast<__half2*>(xh) + (long long)i * 4;
#pragma unroll
            for (int k = 0; k < 4; k++) d[k] = __float22half2_rn(__bfloat1622float2(s[k]));
        }
    }
}

// ---------------- K3c: CUDA finalize (TK<=8, H=4096): out[t] = sum_j tw*z*rsqrt
__global__ void k3c_kernel(const int* __restrict__ slots, const float* __restrict__ tw,
                           const float* __restrict__ ssp, const __half* __restrict__ z,
                           __nv_bfloat16* __restrict__ out, int TK, float inv_h) {
    const int t = blockIdx.x, cb = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    __shared__ float s_coef[8];
    __shared__ int s_slot[8];
    if (wid < TK) {
        const int sl = slots[t * TK + wid];
        float s = 0.f;
        if (sl >= 0) {
            const float4 v = reinterpret_cast<const float4*>(ssp + (long long)sl * 128)[lane];
            s = v.x + v.y + v.z + v.w;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) {
            s_slot[wid] = sl;
            s_coef[wid] = (sl >= 0) ? tw[t * TK + wid] * rsqrtf(s * inv_h + 1e-8f) : 0.f;
        }
    }
    __syncthreads();
    const int c0 = cb * 1024 + tid * 4;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int j = 0; j < TK; j++) {
        const int sl = s_slot[j];
        if (sl >= 0) {
            const float cf = s_coef[j];
            const __half2* zp = reinterpret_cast<const __half2*>(z + (long long)sl * 4096 + c0);
            const float2 va = __half22float2(zp[0]);
            const float2 vb = __half22float2(zp[1]);
            a0 += cf * va.x; a1 += cf * va.y; a2 += cf * vb.x; a3 += cf * vb.y;
        }
    }
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(out + (long long)t * 4096 + c0);
    op[0] = __floats2bfloat162_rn(a0, a1);
    op[1] = __floats2bfloat162_rn(a2, a3);
}

constexpr int NCTA = 264;      // 2 CTAs x 132 SMs, persistent

void k0x(torch::Tensor x, torch::Tensor xh, int64_t n8) {
    k0x_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<__half*>(xh.data_ptr<at::Half>()), (int)n8);
}

void k0f(torch::Tensor ids, torch::Tensor slots, torch::Tensor tok, torch::Tensor counts,
         torch::Tensor act, torch::Tensor actn, torch::Tensor x, torch::Tensor xh,
         int64_t n_routes, int64_t MP, int64_t TK, int64_t ls, int64_t lc, int64_t n8) {
    const unsigned nb = 1u + (unsigned)((n8 + 255) / 256);
    k0f_kernel<<<nb, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int>(), slots.data_ptr<int>(), tok.data_ptr<int>(),
        counts.data_ptr<int>(), act.data_ptr<int>(), actn.data_ptr<int>(),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<__half*>(xh.data_ptr<at::Half>()),
        (int)n_routes, (int)MP, (int)TK, (int)ls, (int)lc, (int)n8);
}

void k3c(torch::Tensor slots, torch::Tensor tw, torch::Tensor ssp, torch::Tensor z,
         torch::Tensor out, int64_t M, int64_t TK, double inv_h) {
    dim3 g((unsigned)M, 4);
    k3c_kernel<<<g, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        slots.data_ptr<int>(), tw.data_ptr<float>(), ssp.data_ptr<float>(),
        reinterpret_cast<const __half*>(z.data_ptr<at::Half>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        (int)TK, (float)inv_h);
}

void k1(torch::Tensor w, torch::Tensor sf, torch::Tensor gs, torch::Tensor counts,
        torch::Tensor act, torch::Tensor actn, torch::Tensor tok, torch::Tensor xh,
        torch::Tensor inter, int64_t MP, int64_t ls, int64_t M) {
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(k1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM1);
        init = true;
    }
    k1_kernel<<<NCTA, 256, SMEM1, at::cuda::getCurrentCUDAStream()>>>(
        w.data_ptr<unsigned char>(), sf.data_ptr<unsigned char>(), gs.data_ptr<float>(),
        counts.data_ptr<int>(), act.data_ptr<int>(), actn.data_ptr<int>(), tok.data_ptr<int>(),
        reinterpret_cast<const __half*>(xh.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(inter.data_ptr<at::Half>()), (int)MP, (int)ls, (int)M);
}

void k2(torch::Tensor w, torch::Tensor sf, torch::Tensor pw, torch::Tensor counts,
        torch::Tensor act, torch::Tensor actn, torch::Tensor inter, torch::Tensor z,
        torch::Tensor ssp, int64_t MP, int64_t ls, int64_t M) {
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(k2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM2);
        init = true;
    }
    k2_kernel<<<NCTA, 256, SMEM2, at::cuda::getCurrentCUDAStream()>>>(
        w.data_ptr<unsigned char>(), sf.data_ptr<unsigned char>(), pw.data_ptr<float>(),
        counts.data_ptr<int>(), act.data_ptr<int>(), actn.data_ptr<int>(),
        reinterpret_cast<const __half*>(inter.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(z.data_ptr<at::Half>()), ssp.data_ptr<float>(),
        (int)MP, (int)ls, (int)M);
}

void k1m(torch::Tensor wm, torch::Tensor sfm, torch::Tensor gs, torch::Tensor counts,
         torch::Tensor act, torch::Tensor actn, torch::Tensor tok, torch::Tensor x,
         torch::Tensor inter, int64_t MP, int64_t ls, int64_t cmin) {
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(k1m_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM1M);
        init = true;
    }
    k1m_kernel<<<132, 512, SMEM1M, at::cuda::getCurrentCUDAStream()>>>(
        wm.data_ptr<unsigned char>(), sfm.data_ptr<unsigned char>(), gs.data_ptr<float>(),
        counts.data_ptr<int>(), act.data_ptr<int>(), actn.data_ptr<int>(), tok.data_ptr<int>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(inter.data_ptr<at::Half>()), (int)MP, (int)ls, (int)cmin);
}

void k2m(torch::Tensor wm, torch::Tensor sfm, torch::Tensor pw, torch::Tensor counts,
         torch::Tensor act, torch::Tensor actn, torch::Tensor inter, torch::Tensor z,
         torch::Tensor ssp, int64_t MP, int64_t ls, int64_t cmin) {
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(k2m_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM2M);
        init = true;
    }
    k2m_kernel<<<132, 512, SMEM2M, at::cuda::getCurrentCUDAStream()>>>(
        wm.data_ptr<unsigned char>(), sfm.data_ptr<unsigned char>(), pw.data_ptr<float>(),
        counts.data_ptr<int>(), act.data_ptr<int>(), actn.data_ptr<int>(),
        reinterpret_cast<const __half*>(inter.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(z.data_ptr<at::Half>()), ssp.data_ptr<float>(),
        (int)MP, (int)ls, (int)cmin);
}
"""
