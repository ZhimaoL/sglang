"""CUDA source for the warp-specialized FP8 prefill FMHA (ws_attn graft).

Byte-identical kernel code to the certified kernels/muse_prefill_layer/ws_attn.py
(v11 lane: 1.18-1.21x FA3 at 16K-query full-attention, courts reps 4/6, 32 checks
OK). The entry points are gather (paged fp8 cache -> per-(kv head, 128-key) 16KB
stage images, V pi-permuted) and fmha (nst picks the kernel; 35 = ws_fmha2
NST4/EXPM5 = the certified configuration).
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DINL __device__ __forceinline__

// ---------------------------------------------------------------- wgmma plumbing
// (discipline copied from the court-certified wg_kernel.py: "memory" clobbers on
// fence/commit are compiler barriers; the wait carries none - the clobber there
// was measured to serialize independent work behind the MMA.)
DINL void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DINL void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> DINL void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N)); }

// ---------------------------------------------------------------- mbarriers + TMA
DINL void mbar_init(unsigned bar, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
DINL void mbar_arrive(unsigned bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(bar) : "memory");
}
DINL void mbar_expect_tx(unsigned bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(bar), "r"(bytes) : "memory");
}
DINL void bulk_g2s(void* smem_dst, const void* gsrc, unsigned bytes, unsigned bar) {
    unsigned sd = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sd), "l"(gsrc), "r"(bytes), "r"(bar) : "memory");
}
DINL void mbar_wait(unsigned bar, unsigned phase) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}\n" :: "r"(bar), "r"(phase) : "memory");
}
// Generic-proxy smem writes (plain STS) must be fenced before wgmma (async
// proxy) reads them - needed once for the constant ones-tile.
DINL void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// Named barriers (ids 1/2, the 256 consumer threads) enforce the FA3 pingpong:
// the two consumer WGs ALTERNATE their softmax sections, so one WG's MUFU burst
// runs under the other's wgmma groups instead of colliding with its twin
// (lockstep WGs measured 0.72x of FA3: every pipe serialized, ~4.6k cy/iter-pair).
DINL void nbar_sync(int id)   { asm volatile("bar.sync %0, 256;"   :: "r"(id) : "memory"); }
DINL void nbar_arrive(int id) { asm volatile("bar.arrive %0, 256;" :: "r"(id) : "memory"); }

// Register-lifetime fences (wg_kernel hard-won fact: wgmma reads register
// operands ASYNCHRONOUSLY until the group completes; "+r"/"+f" plus these empty
// asms keep ptxas from reusing them inside the async window).
DINL void fence_u(unsigned& x) { asm volatile("" : "+r"(x)); }
DINL void fence_f(float& x)    { asm volatile("" : "+f"(x)); }
template <int N> DINL void fence_ubank(unsigned (&b)[N]) {
#pragma unroll
    for (int i = 0; i < N; i++) fence_u(b[i]);
}
template <int N> DINL void fence_fbank(float (&b)[N]) {
#pragma unroll
    for (int i = 0; i < N; i++) fence_f(b[i]);
}

// B-matrix descriptor (no swizzle; canonical core-matrix form, v10-locked).
DINL unsigned long long make_desc(unsigned saddr, unsigned lbo, unsigned sbo) {
    return (unsigned long long)((saddr >> 4) & 0x3FFF)
         | ((unsigned long long)((lbo >> 4) & 0x3FFF) << 16)
         | ((unsigned long long)((sbo >> 4) & 0x3FFF) << 32);
}

// QK: wgmma m64n128k32, D = f16 (32 b32 regs), A = e4m3 regs (Q), B = desc (K).
// A-fragment regs are "+r": the hardware reads them ASYNCHRONOUSLY until the
// group completes (wg_kernel hard-won fact; plain "r" lets ptxas reuse them
// inside the async window -> nondeterministic corruption sanitizers can't see).
template <int ACC>
DINL void wgmma_qk(unsigned* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
          "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
        : "l"(db), "n"(ACC));
}

// PV: wgmma m64n64k32, D = f32 (32 f regs), A = e4m3 regs (P), B = desc (V).
DINL void wgmma_pv(float* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
        : "l"(db), "n"(1));
}

// ---------------------------------------------------------------- f16 helpers
DINL __half2 u32h2(unsigned v) { return *reinterpret_cast<__half2*>(&v); }
DINL unsigned h2u32(__half2 v) { return *reinterpret_cast<unsigned*>(&v); }
// Pure ALU asms are NOT volatile: volatile pins them in strict program order,
// which serialized the whole softmax burst (no interleave of the two tiles'
// independent chains, measured ~2x cost before this).
DINL unsigned ex2_f16x2(unsigned x) {
    unsigned r;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x));
    return r;
}
DINL unsigned short cvt_e4m3x2(unsigned h2) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(h2));
    return r;
}

// pi: A-fragment k-lane -> key offset within a 32-key group (see module doc).
DINL int pi_lane(int lam) {
    return (lam & 1) | (((lam >> 2) & 3) << 1) | (((lam >> 1) & 1) << 3) | (lam & 16);
}

constexpr int SB = 16384;        // stage bytes: K 8KB + V 8KB
constexpr unsigned NEG = 0xFC00; // f16 -inf
#define M_INIT (-30000.0f)

// Per-tile softmax, split so only the p-producing half gates the PV wgmma.
// part1: mask, rowmax (b32-level trees + half2 shuffles), m/alpha update, acc
// rescale, x = HFMA2, p = exp2, e4m3 pack. part2 (runs AFTER the PV commit,
// off the tensor-core critical path): f16 rowsum tree + f32 l carry.
// s: 32 f16x2 regs (r0 even, r1 odd); out s = p (scaled x2^7), pk = A-frags.
//
// EXPM 0: p = ex2.approx.f16x2 - which ptxas legalizes to 2x MUFU.EX2.F16 +
//   PRMT (seen in SASS), i.e. ONE MUFU op per score: the 4.6ms-at-H MUFU wall
//   that binds FA3 itself. EXPM 1: Schraudolph-in-f16x2 - exp2 via the float
//   bit pattern, entirely on the FMA/ALU pipes, zero MUFU:
//     x' = 1024*x + 57344 maps x in [-24, 8) onto the f16 binade [2^15, 2^16)
//     whose mantissa m10 then equals 32*(x+24) (5 bits int, 5 bits fraction);
//     p_bits = (m10 << 5) - 2048 re-biases so p = 2^x * (1+f)/2^f, the classic
//     linear-between-powers approximation (ratio within +-4.4% after the -59
//     optimal-bias fold; the e4m3 quantization right after is itself +-3%).
//   Clamp at x >= -21 keeps the packed subtract borrow-free; masked keys
//   (s = -inf) land on the clamp -> p ~ 2^-14, which e4m3 rounds to exact 0.
template <int EXPM>
DINL void softmax_p1(unsigned (&s)[32], unsigned (&pk)[16], float (&m)[2], float (&al)[2],
                     float (&acc)[32], float qk2f, __half2 qk2h2,
                     int diag, int j0, int lim0, int t) {
    if (diag) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int k0 = j0 + 2 * t + 8 * j;
            if (k0 > lim0)     s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
            if (k0 + 1 > lim0) s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
            if (k0 > lim0 + 1)     s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
            if (k0 + 1 > lim0 + 1) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
        }
    }
    // row maxima (raw units; qk2f > 0 so scaling commutes with max); log-depth
    // trees, quad reduce as half2 (one shfl chain serves both rows)
    __half2 t0v[8], t1v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        t0v[j] = __hmax2(u32h2(s[4 * j]), u32h2(s[4 * j + 2]));
        t1v[j] = __hmax2(u32h2(s[4 * j + 1]), u32h2(s[4 * j + 3]));
    }
#pragma unroll
    for (int st = 4; st > 0; st >>= 1)
#pragma unroll
        for (int j = 0; j < 8; j++) if (j < st) {
            t0v[j] = __hmax2(t0v[j], t0v[j + st]);
            t1v[j] = __hmax2(t1v[j], t1v[j + st]);
        }
    // pack (rowmax0, rowmax1) into ONE half2 so the quad shuffle is one chain
    __half2 hm = __halves2half2(__hmax(__low2half(t0v[0]), __high2half(t0v[0])),
                                __hmax(__low2half(t1v[0]), __high2half(t1v[0])));
    hm = __hmax2(hm, u32h2(__shfl_xor_sync(0xFFFFFFFFu, h2u32(hm), 1)));
    hm = __hmax2(hm, u32h2(__shfl_xor_sync(0xFFFFFFFFu, h2u32(hm), 2)));
    float mn0 = fmaxf(m[0], __half2float(__low2half(hm)) * qk2f);
    float mn1 = fmaxf(m[1], __half2float(__high2half(hm)) * qk2f);
    float a0 = exp2f(m[0] - mn0), a1 = exp2f(m[1] - mn1);
    m[0] = mn0; m[1] = mn1;
    al[0] = a0; al[1] = a1;
    // x = s*qk2 - m + 7  (the +7 scales p by 128 into the e4m3 range; it rides
    // both acc and l identically, so it cancels in o = acc/l)
    __half2 nm0 = __float2half2_rn(7.0f - mn0);
    __half2 nm1 = __float2half2_rn(7.0f - mn1);
    if (EXPM == 0) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            s[2 * j]     = ex2_f16x2(h2u32(__hfma2(u32h2(s[2 * j]),     qk2h2, nm0)));
            s[2 * j + 1] = ex2_f16x2(h2u32(__hfma2(u32h2(s[2 * j + 1]), qk2h2, nm1)));
        }
    } else {
        // The bit mapping itself emits 2^(y+7) (the -2048 bias bakes in the
        // x128 p-scale), so the trick's addend must NOT carry the +7 again.
        // EXPM 2 subtracts the sawtooth residual (1+f-2^f, 8-bucket LUT via
        // PRMT): ratio error +-4.4% -> ~+-1%. EXPM 4: hybrid - even regs take
        // the (otherwise idle) MUFU ex2 path, odd regs the ALU bit trick:
        // fewer issue slots, smoother pipe mix, and exact exp2 on half the p.
        const __half2 nm0e  = __float2half2_rn(-mn0);
        const __half2 nm1e  = __float2half2_rn(-mn1);
        const __half2 hsc   = __float2half2_rn(1024.0f);
        const __half2 hmag  = __float2half2_rn(57344.0f);
        const __half2 hclmp = __float2half2_rn(35840.0f);
#pragma unroll
        for (int j = 0; j < 32; j++) {
            if (EXPM == 4 && (j & 2) == 0) {
                s[j] = ex2_f16x2(h2u32(__hfma2(u32h2(s[j]), qk2h2, (j & 1) ? nm1 : nm0)));
                continue;
            }
            __half2 y  = __hfma2(u32h2(s[j]), qk2h2, (j & 1) ? nm1e : nm0e);
            __half2 xb = __hmax2(__hfma2(y, hsc, hmag), hclmp);
            unsigned u = h2u32(xb);
            if (EXPM == 2) {
                unsigned sh = (u & 0x03FF03FFu) << 5;
                unsigned c1 = (u >> 2) & 0x00070007u;
                unsigned cr = __byte_perm(0x55483213u, 0x173A4F58u, c1 | (c1 >> 8)) & 0x00FF00FFu;
                s[j] = sh - 0x08000800u - cr;
            } else {
                // (m10 << 5) - bias in ONE IMAD; -59 centers the sawtooth
                s[j] = (u & 0x03FF03FFu) * 32u - 0x083B083Bu;
            }
        }
    }
    // pack e4m3 A-fragments: a_lane bytes (4t..4t+3) = cols {2t,2t+1,2t+8,2t+9}
    // of k32 slice js (pi-permuted V rows complete the contract)
#pragma unroll
    for (int js = 0; js < 4; js++) {
        pk[4 * js]     = (unsigned)cvt_e4m3x2(s[8 * js])     | ((unsigned)cvt_e4m3x2(s[8 * js + 2]) << 16);
        pk[4 * js + 1] = (unsigned)cvt_e4m3x2(s[8 * js + 1]) | ((unsigned)cvt_e4m3x2(s[8 * js + 3]) << 16);
        pk[4 * js + 2] = (unsigned)cvt_e4m3x2(s[8 * js + 4]) | ((unsigned)cvt_e4m3x2(s[8 * js + 6]) << 16);
        pk[4 * js + 3] = (unsigned)cvt_e4m3x2(s[8 * js + 5]) | ((unsigned)cvt_e4m3x2(s[8 * js + 7]) << 16);
    }
    // rescale the PV accumulator (prev PV completion guaranteed by caller's
    // wait); skipped by warp vote when no row's max moved (steady state)
    if (!__all_sync(0xFFFFFFFFu, (a0 == 1.0f) & (a1 == 1.0f))) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[4 * j]     *= a0; acc[4 * j + 1] *= a0;
            acc[4 * j + 2] *= a1; acc[4 * j + 3] *= a1;
        }
    }
}

// Split softmax for the MJ1 pipeline (EXPM 3): d3a = mask + rowmax + m/alpha
// (touches neither acc nor pk, so it runs while the PREVIOUS PV group is still
// in flight); d3b = vote rescale + Schraudolph exp (f16, IMAD-fused shift-sub)
// + e4m3 cvt pack (runs after the PV wait). l comes from the ones-column n8
// wgmma, so there is no rowsum tree at all.
DINL void softmax_d3a(unsigned (&s)[32], float (&m)[2], float (&al)[2], float qk2f,
                      int diag, int j0, int lim0, int t) {
    if (diag) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int k0 = j0 + 2 * t + 8 * j;
            if (k0 > lim0)     s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
            if (k0 + 1 > lim0) s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
            if (k0 > lim0 + 1)     s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
            if (k0 + 1 > lim0 + 1) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
        }
    }
    __half2 t0v[8], t1v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        t0v[j] = __hmax2(u32h2(s[4 * j]), u32h2(s[4 * j + 2]));
        t1v[j] = __hmax2(u32h2(s[4 * j + 1]), u32h2(s[4 * j + 3]));
    }
#pragma unroll
    for (int st = 4; st > 0; st >>= 1)
#pragma unroll
        for (int j = 0; j < 8; j++) if (j < st) {
            t0v[j] = __hmax2(t0v[j], t0v[j + st]);
            t1v[j] = __hmax2(t1v[j], t1v[j + st]);
        }
    __half2 hm = __halves2half2(__hmax(__low2half(t0v[0]), __high2half(t0v[0])),
                                __hmax(__low2half(t1v[0]), __high2half(t1v[0])));
    hm = __hmax2(hm, u32h2(__shfl_xor_sync(0xFFFFFFFFu, h2u32(hm), 1)));
    hm = __hmax2(hm, u32h2(__shfl_xor_sync(0xFFFFFFFFu, h2u32(hm), 2)));
    float mn0 = fmaxf(m[0], __half2float(__low2half(hm)) * qk2f);
    float mn1 = fmaxf(m[1], __half2float(__high2half(hm)) * qk2f);
    al[0] = exp2f(m[0] - mn0);
    al[1] = exp2f(m[1] - mn1);
    m[0] = mn0; m[1] = mn1;
}

DINL void softmax_d3b(unsigned (&s)[32], unsigned (&pk)[16], const float (&m)[2],
                      const float (&al)[2], float (&acc)[32], float (&accl)[4],
                      __half2 qk2h2) {
    const float a0 = al[0], a1 = al[1];
    if (!__all_sync(0xFFFFFFFFu, (a0 == 1.0f) & (a1 == 1.0f))) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[4 * j]     *= a0; acc[4 * j + 1] *= a0;
            acc[4 * j + 2] *= a1; acc[4 * j + 3] *= a1;
        }
        accl[0] *= a0; accl[1] *= a0; accl[2] *= a1; accl[3] *= a1;
    }
    const __half2 nm0e  = __float2half2_rn(-m[0]);
    const __half2 nm1e  = __float2half2_rn(-m[1]);
    const __half2 hsc   = __float2half2_rn(1024.0f);
    const __half2 hmag  = __float2half2_rn(57344.0f);
    const __half2 hclmp = __float2half2_rn(35840.0f);
#pragma unroll
    for (int j = 0; j < 32; j++) {
        __half2 y  = __hfma2(u32h2(s[j]), qk2h2, (j & 1) ? nm1e : nm0e);
        __half2 xb = __hmax2(__hfma2(y, hsc, hmag), hclmp);
        // (m10 << 5) - bias via one IMAD: packed halves can't carry (m10<=1023)
        s[j] = (h2u32(xb) & 0x03FF03FFu) * 32u - 0x083B083Bu;
    }
#pragma unroll
    for (int js = 0; js < 4; js++) {
        pk[4 * js]     = (unsigned)cvt_e4m3x2(s[8 * js])     | ((unsigned)cvt_e4m3x2(s[8 * js + 2]) << 16);
        pk[4 * js + 1] = (unsigned)cvt_e4m3x2(s[8 * js + 1]) | ((unsigned)cvt_e4m3x2(s[8 * js + 3]) << 16);
        pk[4 * js + 2] = (unsigned)cvt_e4m3x2(s[8 * js + 4]) | ((unsigned)cvt_e4m3x2(s[8 * js + 6]) << 16);
        pk[4 * js + 3] = (unsigned)cvt_e4m3x2(s[8 * js + 5]) | ((unsigned)cvt_e4m3x2(s[8 * js + 7]) << 16);
    }
}

// Lazy-max softmax (EXPM 5, the FA3 fp8 recipe): the exp loop uses the RUNNING
// max immediately - no rowmax reduce, no shuffles, no alpha on the common path.
// A warp-local conservative vote (per-thread partial maxima vs m + 0.75) drops
// to a slow path only when some row's new max approaches the e4m3 headroom
// (p <= 2^7.75 = 215 << 448); the slow path does the full reduce, bumps m with
// +0.5 headroom and rescales acc/l. Fires O(log NT) times per stream.
// The trick's magic is shifted (50176 = 57344 - 7*1024) so ONE addend pair
// nmc = (7 - m) serves both the MUFU and the bit-trick halves.
template <int FORCE = 0>
DINL void softmax_lz(unsigned (&s)[32], unsigned (&pk)[16], float (&m)[2], float (&al)[2],
                     __half2 (&nmc)[2], float (&acc)[32], float qk2f, __half2 qk2h2,
                     int diag, int j0, int lim0, int t) {
    if (diag) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int k0 = j0 + 2 * t + 8 * j;
            if (k0 > lim0)     s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
            if (k0 + 1 > lim0) s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
            if (k0 > lim0 + 1)     s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
            if (k0 + 1 > lim0 + 1) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
        }
    }
    __half2 t0v[8], t1v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        t0v[j] = __hmax2(u32h2(s[4 * j]), u32h2(s[4 * j + 2]));
        t1v[j] = __hmax2(u32h2(s[4 * j + 1]), u32h2(s[4 * j + 3]));
    }
#pragma unroll
    for (int st = 4; st > 0; st >>= 1)
#pragma unroll
        for (int j = 0; j < 8; j++) if (j < st) {
            t0v[j] = __hmax2(t0v[j], t0v[j + st]);
            t1v[j] = __hmax2(t1v[j], t1v[j + st]);
        }
    const float f0 = fmaxf(__half2float(__low2half(t0v[0])), __half2float(__high2half(t0v[0])));
    const float f1 = fmaxf(__half2float(__low2half(t1v[0])), __half2float(__high2half(t1v[0])));
    al[0] = 1.0f; al[1] = 1.0f;
    const bool ok = (f0 * qk2f <= m[0] + 0.75f) & (f1 * qk2f <= m[1] + 0.75f);
    if (FORCE || !__all_sync(0xFFFFFFFFu, ok)) {        // rare: new max near headroom
        float r0 = f0, r1 = f1;
        r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 1));
        r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 2));
        r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 1));
        r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 2));
        // zero headroom: a +0.5 bump measured 2x the output error (p landing a
        // half-binade down degrades the e4m3/f16 interplay); laziness comes from
        // the 0.75 vote margin alone.
        const float mn0 = fmaxf(m[0], r0 * qk2f);
        const float mn1 = fmaxf(m[1], r1 * qk2f);
        al[0] = exp2f(m[0] - mn0);
        al[1] = exp2f(m[1] - mn1);
        m[0] = mn0; m[1] = mn1;
        nmc[0] = __float2half2_rn(7.0f - mn0);
        nmc[1] = __float2half2_rn(7.0f - mn1);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[4 * j]     *= al[0]; acc[4 * j + 1] *= al[0];
            acc[4 * j + 2] *= al[1]; acc[4 * j + 3] *= al[1];
        }
    }
    const __half2 hsc   = __float2half2_rn(1024.0f);
    const __half2 hmag  = __float2half2_rn(50176.0f);   // 57344 - 7*1024
    const __half2 hclmp = __float2half2_rn(35840.0f);
#pragma unroll
    for (int j = 0; j < 32; j++) {
        if ((j & 2) == 0) {    // MUFU half
            s[j] = ex2_f16x2(h2u32(__hfma2(u32h2(s[j]), qk2h2, nmc[j & 1])));
        } else {               // ALU bit-trick half
            __half2 y  = __hfma2(u32h2(s[j]), qk2h2, nmc[j & 1]);
            __half2 xb = __hmax2(__hfma2(y, hsc, hmag), hclmp);
            s[j] = (h2u32(xb) & 0x03FF03FFu) * 32u - 0x083B083Bu;
        }
    }
#pragma unroll
    for (int js = 0; js < 4; js++) {
        pk[4 * js]     = (unsigned)cvt_e4m3x2(s[8 * js])     | ((unsigned)cvt_e4m3x2(s[8 * js + 2]) << 16);
        pk[4 * js + 1] = (unsigned)cvt_e4m3x2(s[8 * js + 1]) | ((unsigned)cvt_e4m3x2(s[8 * js + 3]) << 16);
        pk[4 * js + 2] = (unsigned)cvt_e4m3x2(s[8 * js + 4]) | ((unsigned)cvt_e4m3x2(s[8 * js + 6]) << 16);
        pk[4 * js + 3] = (unsigned)cvt_e4m3x2(s[8 * js + 5]) | ((unsigned)cvt_e4m3x2(s[8 * js + 7]) << 16);
    }
}

// l-sum via ones-column: wgmma m64n8k32 against a constant ones B-tile; col 0 of
// the f32 D fragment accumulates sum(p-hat) per row (held by the t==0 lanes).
DINL void wgmma_l(float* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3\n"
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
        : "l"(db), "n"(1));
}

DINL void softmax_p2(unsigned (&s)[32], float (&l)[2], const float (&al)[2]) {
    __half2 t0v[8], t1v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        t0v[j] = __hadd2(u32h2(s[4 * j]), u32h2(s[4 * j + 2]));
        t1v[j] = __hadd2(u32h2(s[4 * j + 1]), u32h2(s[4 * j + 3]));
    }
#pragma unroll
    for (int st = 4; st > 0; st >>= 1)
#pragma unroll
        for (int j = 0; j < 8; j++) if (j < st) {
            t0v[j] = __hadd2(t0v[j], t0v[j + st]);
            t1v[j] = __hadd2(t1v[j], t1v[j + st]);
        }
    __half2 hs = __halves2half2(__hadd(__low2half(t0v[0]), __high2half(t0v[0])),
                                __hadd(__low2half(t1v[0]), __high2half(t1v[0])));
    hs = __hadd2(hs, u32h2(__shfl_xor_sync(0xFFFFFFFFu, h2u32(hs), 1)));
    hs = __hadd2(hs, u32h2(__shfl_xor_sync(0xFFFFFFFFu, h2u32(hs), 2)));
    l[0] = l[0] * al[0] + __half2float(__low2half(hs));
    l[1] = l[1] * al[1] + __half2float(__high2half(hs));
}

// ---------------------------------------------------------------- main kernel
// Grid (cdiv(L,32), B*HKV); 288 threads: consumers tid 0..255, producer 256..287.
// Heavy (late-q) blocks launch first. NT tiles of 128 keys per CTA.
// MODE: 0 = full; 1 = no-softmax probe (constant P, full MMA + pipeline);
//       3 = no-MMA probe (softmax + pipeline); 4 = pipeline-only probe.
// ORD: 1 = pingpong-order the softmax_p1 sections across the two consumer WGs.
// EXPM: exp2 implementation (see softmax_p1).
template <int NST, int MODE = 0, int ORD = 0, int EXPM = 1>
__global__ void __launch_bounds__(288, 1) ws_fmha(
    const unsigned char* __restrict__ q,     // fp8 [B*L, 16, 64]
    const unsigned char* __restrict__ kvt,   // [B, 2, cap/128, 16KB] stage images
    __nv_bfloat16* __restrict__ out,         // [B*L, 16, 64]
    const int* __restrict__ lens,            // [B] = ctx + L
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    float smqk_log2e, int B, int L, int capblks)
{
    const int tid = threadIdx.x;
    const int b = blockIdx.y >> 1, kvh = blockIdx.y & 1;
    const int qs = (gridDim.x - 1 - blockIdx.x) * 32;
    const int len = lens[b];
    const int ctx = len - L;
    int nt = min(ctx + qs + 32, len);
    const int NT = min(capblks, (nt + 127) >> 7);

    extern __shared__ __align__(1024) char smem[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned mb = sbase + NST * SB;
    if (tid < NST) {
        mbar_init(mb + tid * 8, 1);                 // full[s]: expect_tx arrival
        mbar_init(mb + NST * 8 + tid * 8, 256);     // empty[s]: both consumer WGs
    }
    __syncthreads();

    if (tid >= 256) {                                // ---------- producer warp
        if (tid == 256) {
            const char* src = reinterpret_cast<const char*>(kvt)
                            + ((size_t)(b * 2 + kvh) * capblks) * SB;
            int psl = 0, pround = 0;
            for (int kt = 0; kt < NT; kt++) {
                if (pround > 0) mbar_wait(mb + NST * 8 + psl * 8, (pround - 1) & 1);
                mbar_expect_tx(mb + psl * 8, SB);
                bulk_g2s(smem + psl * SB, src + ((size_t)kt) * SB, SB, mb + psl * 8);
                psl++; if (psl == NST) { psl = 0; pround++; }
            }
        }
        return;
    }

    // ---------- consumer warpgroups
    const int wg = tid >> 7;            // 0/1: owns m64 tiles {2wg, 2wg+1}
    const int wiw = (tid >> 5) & 3;     // warp in wg -> tokens 2wiw, 2wiw+1 of a tile
    const int lane = tid & 31;
    const int t = lane & 3;             // quad col group
    const int h = lane >> 2;            // q-head within the kv group (0..7)

    const float qk2f = k_scale[0] * smqk_log2e;
    const __half2 qk2h2 = __float2half2_rn(qk2f);
    const float vs = v_scale[0];

    // Q A-fragments, 2 tiles x 2 k32 slices x {a0..a3}; rows past L load zero.
    unsigned qf[2][8];
#pragma unroll
    for (int tt = 0; tt < 2; tt++) {
        const int tok0 = qs + 8 * (2 * wg + tt) + 2 * wiw;
#pragma unroll
        for (int sl2 = 0; sl2 < 2; sl2++) {
#pragma unroll
            for (int rr = 0; rr < 2; rr++) {   // a0/a1 rows (tok0, tok0+1)
                const int tok = tok0 + rr;
                const unsigned char* qp = q + ((size_t)(b * L + tok) * 16 + kvh * 8 + h) * 64
                                        + sl2 * 32 + 4 * t;
                qf[tt][4 * sl2 + rr]     = (tok < L) ? *reinterpret_cast<const unsigned*>(qp) : 0u;
                qf[tt][4 * sl2 + rr + 2] = (tok < L) ? *reinterpret_cast<const unsigned*>(qp + 16) : 0u;
            }
        }
    }

    float m0[2] = {M_INIT, M_INIT}, l0[2] = {0.f, 0.f}, al0[2];
    float m1[2] = {M_INIT, M_INIT}, l1[2] = {0.f, 0.f}, al1[2];
    float acc0[32], acc1[32];
#pragma unroll
    for (int i = 0; i < 32; i++) { acc0[i] = 0.f; acc1[i] = 0.f; }
    unsigned s0[32] = {}, s1[32] = {}, p0[16] = {}, p1[16] = {};

    const int lim0 = ctx + qs + 8 * (2 * wg)     + 2 * wiw;  // r0 causal limit, tile0
    const int lim1 = ctx + qs + 8 * (2 * wg + 1) + 2 * wiw;  // tile1
    const int dg0 = ctx + qs + 8 * (2 * wg);                 // tile-level diag tests
    const int dg1 = ctx + qs + 8 * (2 * wg + 1);

    int sl = 0, cph = 0, prev = -1;
    if (wg == 1) nbar_arrive(1);        // bootstrap: WG0 owns the first token
    for (int kt = 0; kt < NT; kt++) {
        mbar_wait(mb + sl * 8, cph);
        const unsigned st = sbase + sl * SB;
        const int j0 = kt << 7;
        const int diag0 = (j0 + 127 > dg0), diag1 = (j0 + 127 > dg1);
        // All descriptors precomputed BEFORE the fence: a non-wgmma def of a
        // wgmma input inside the fence..commit window makes ptxas serialize the
        // whole pipeline stage (C7513 - measured 0.72x of FA3 before this).
        const unsigned long long dK0 = make_desc(st, 2048, 128);
        const unsigned long long dK1 = make_desc(st + 4096, 2048, 128);
        unsigned long long dV[4];
#pragma unroll
        for (int js = 0; js < 4; js++) dV[js] = make_desc(st + 8192 + js * 2048, 1024, 128);

        // One wgmma window per iteration: the two QK groups land behind the two
        // PV groups of iter kt-1, so the single wait<0> packs all tensor-core
        // work into one stall that the OTHER warpgroup's softmax burst covers
        // (ORD=1 anti-phases the bursts). Both softmaxes then run back-to-back
        // with no asm barrier between them, so ptxas interleaves the two tiles'
        // independent reduce/MUFU chains (2x ILP on the latency-bound heads).
        if (MODE == 0 || MODE == 1) {
            wgmma_fence();
            wgmma_qk<0>(s0, qf[0],     dK0);
            wgmma_qk<1>(s0, qf[0] + 4, dK1);
            wgmma_commit();
            wgmma_qk<0>(s1, qf[1],     dK0);
            wgmma_qk<1>(s1, qf[1] + 4, dK1);
            wgmma_commit();
            wgmma_wait<0>();                   // s0+s1 ready; prev PVs done
        }
        if (prev >= 0) mbar_arrive(mb + NST * 8 + prev * 8);

        if (MODE == 0 || MODE == 3) {
            if (ORD) nbar_sync(1 + wg);
            softmax_p1<EXPM>(s0, p0, m0, al0, acc0, qk2f, qk2h2, diag0, j0, lim0, t);
            softmax_p1<EXPM>(s1, p1, m1, al1, acc1, qk2f, qk2h2, diag1, j0, lim1, t);
            if (ORD) nbar_arrive(2 - wg);
        }
        if (MODE == 0 || MODE == 1) {
            wgmma_fence();
#pragma unroll
            for (int js = 0; js < 4; js++)
                wgmma_pv(acc0, p0 + 4 * js, dV[js]);
            wgmma_commit();
            wgmma_fence();
#pragma unroll
            for (int js = 0; js < 4; js++)
                wgmma_pv(acc1, p1 + 4 * js, dV[js]);
            wgmma_commit();
        }
        if (MODE == 0 || MODE == 3) {          // rowsums run under the PV groups
            softmax_p2(s0, l0, al0);
            softmax_p2(s1, l1, al1);
        }

        prev = sl;
        sl++; if (sl == NST) { sl = 0; cph ^= 1; }
    }
    wgmma_wait<0>();
    fence_ubank(p0); fence_ubank(p1);
    fence_fbank(acc0); fence_fbank(acc1);

    // epilogue: o = acc * (v_scale / l); bf16 pair stores
#pragma unroll
    for (int tt = 0; tt < 2; tt++) {
        const float* acc = tt ? acc1 : acc0;
        const float* lv = tt ? l1 : l0;
        const int tok0 = qs + 8 * (2 * wg + tt) + 2 * wiw;
        const float inv0 = vs / fmaxf(lv[0], 1e-37f);
        const float inv1 = vs / fmaxf(lv[1], 1e-37f);
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            if (tok >= L) continue;
            const float inv = rr ? inv1 : inv0;
            __nv_bfloat16* op = out + ((size_t)(b * L + tok) * 16 + kvh * 8 + h) * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                __nv_bfloat162 o2 = __floats2bfloat162_rn(acc[4 * j + 2 * rr] * inv,
                                                          acc[4 * j + 2 * rr + 1] * inv);
                *reinterpret_cast<unsigned*>(op + 8 * j) = *reinterpret_cast<unsigned*>(&o2);
            }
        }
    }
}

// ------------------------------------------------- MJ=1 kernel with QK-ahead
// Each consumer WG owns ONE m64 tile (8 tokens; CTA = 16) and software-pipelines
// the key loop: QK(kt+1) is committed into the spare S bank BEFORE softmax(kt),
// so exactly one wgmma group is always in flight per WG during its softmax -
// the WG hides its own MMA instead of relying on cross-WG phase luck.
// Grid (cdiv(L,16), B*HKV).
template <int NST, int EXPM = 1>
__global__ void __launch_bounds__(288, 1) ws_fmha1(
    const unsigned char* __restrict__ q, const unsigned char* __restrict__ kvt,
    __nv_bfloat16* __restrict__ out, const int* __restrict__ lens,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    float smqk_log2e, int B, int L, int capblks)
{
    const int tid = threadIdx.x;
    const int b = blockIdx.y >> 1, kvh = blockIdx.y & 1;
    const int qs = (gridDim.x - 1 - blockIdx.x) * 16;
    const int len = lens[b];
    const int ctx = len - L;
    const int NT = min(capblks, (min(ctx + qs + 16, len) + 127) >> 7);

    extern __shared__ __align__(1024) char smem[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned mb = sbase + NST * SB;
    if (tid < NST) {
        mbar_init(mb + tid * 8, 1);
        mbar_init(mb + NST * 8 + tid * 8, 256);
    }
    if (EXPM == 3 && tid < 64)  // constant ones B-tile for the l-sum n8 wgmma
        *reinterpret_cast<unsigned*>(smem + NST * SB + 256 + tid * 4) =
            ((tid & 31) < 4) ? 0x38383838u : 0u;
    __syncthreads();
    if (EXPM == 3 && tid < 256) fence_proxy_async();

    if (tid >= 256) {
        if (tid == 256) {
            const char* src = reinterpret_cast<const char*>(kvt)
                            + ((size_t)(b * 2 + kvh) * capblks) * SB;
            int psl = 0, pround = 0;
            for (int kt = 0; kt < NT; kt++) {
                if (pround > 0) mbar_wait(mb + NST * 8 + psl * 8, (pround - 1) & 1);
                mbar_expect_tx(mb + psl * 8, SB);
                bulk_g2s(smem + psl * SB, src + ((size_t)kt) * SB, SB, mb + psl * 8);
                psl++; if (psl == NST) { psl = 0; pround++; }
            }
        }
        return;
    }

    const int wg = tid >> 7;
    const int wiw = (tid >> 5) & 3;
    const int lane = tid & 31;
    const int t = lane & 3;
    const int h = lane >> 2;

    const float qk2f = k_scale[0] * smqk_log2e;
    const __half2 qk2h2 = __float2half2_rn(qk2f);
    const float vs = v_scale[0];

    unsigned qf[8];
    const int tok0 = qs + 8 * wg + 2 * wiw;
#pragma unroll
    for (int sl2 = 0; sl2 < 2; sl2++)
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            const unsigned char* qp = q + ((size_t)(b * L + tok) * 16 + kvh * 8 + h) * 64
                                    + sl2 * 32 + 4 * t;
            qf[4 * sl2 + rr]     = (tok < L) ? *reinterpret_cast<const unsigned*>(qp) : 0u;
            qf[4 * sl2 + rr + 2] = (tok < L) ? *reinterpret_cast<const unsigned*>(qp + 16) : 0u;
        }

    float m[2] = {M_INIT, M_INIT}, l[2] = {0.f, 0.f}, al[2];
    float acc[32], accl[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;
    unsigned sA[32] = {}, sB[32] = {}, pk[16] = {};
    const int lim = ctx + qs + 8 * wg + 2 * wiw;
    const int dgt = ctx + qs + 8 * wg;
    const unsigned long long dOnes = make_desc(sbase + NST * SB + 256, 128, 128);

    // prologue: QK(0) into sA
    mbar_wait(mb + 0, 0);
    {
        const unsigned st0 = sbase;
        const unsigned long long dK0 = make_desc(st0, 2048, 128);
        const unsigned long long dK1 = make_desc(st0 + 4096, 2048, 128);
        wgmma_fence();
        wgmma_qk<0>(sA, qf,     dK0);
        wgmma_qk<1>(sA, qf + 4, dK1);
        wgmma_commit();
    }

#define ITER1(KT, SC, SN)                                                              \
    if ((KT) < NT) {                                                                   \
        const int slc = (KT) % NST;                                                    \
        const unsigned stc = sbase + slc * SB;                                         \
        unsigned long long dV[4];                                                      \
        _Pragma("unroll")                                                              \
        for (int js = 0; js < 4; js++) dV[js] = make_desc(stc + 8192 + js * 2048, 1024, 128); \
        const int j0 = (KT) << 7;                                                      \
        const int dg = (j0 + 127 > dgt);                                               \
        const int pf = (KT) + 1 < NT;                                                  \
        if (pf) {                                                                      \
            const int sln = ((KT) + 1) % NST;                                          \
            mbar_wait(mb + sln * 8, (((KT) + 1) / NST) & 1);                           \
            const unsigned stn = sbase + sln * SB;                                     \
            const unsigned long long dK0 = make_desc(stn, 2048, 128);                  \
            const unsigned long long dK1 = make_desc(stn + 4096, 2048, 128);           \
            wgmma_fence();                                                             \
            wgmma_qk<0>(SN, qf,     dK0);                                              \
            wgmma_qk<1>(SN, qf + 4, dK1);                                              \
            wgmma_commit();                                                            \
        }                                                                              \
        if (EXPM == 3) {                                                               \
            /* deferred ladder: QK(KT) done -> d3a runs while PV(KT-1) and      */     \
            /* QK(KT+1) execute; PV(KT-1) done only before d3b's acc touch      */     \
            if (pf) { if ((KT) > 0) wgmma_wait<2>(); else wgmma_wait<1>(); }           \
            else    { if ((KT) > 0) wgmma_wait<1>(); else wgmma_wait<0>(); }           \
            softmax_d3a(SC, m, al, qk2f, dg, j0, lim, t);                              \
            if (pf) wgmma_wait<1>(); else wgmma_wait<0>();                             \
            if ((KT) > 0) mbar_arrive(mb + NST * 8 + (((KT) - 1) % NST) * 8);          \
            softmax_d3b(SC, pk, m, al, acc, accl, qk2h2);                              \
            wgmma_fence();                                                             \
            _Pragma("unroll")                                                          \
            for (int js = 0; js < 4; js++) {                                           \
                wgmma_pv(acc, pk + 4 * js, dV[js]);                                    \
                wgmma_l(accl, pk + 4 * js, dOnes);                                     \
            }                                                                          \
            wgmma_commit();                                                            \
        } else {                                                                       \
            if (pf) wgmma_wait<1>(); else wgmma_wait<0>();                             \
            if ((KT) > 0) mbar_arrive(mb + NST * 8 + (((KT) - 1) % NST) * 8);          \
            softmax_p1<EXPM>(SC, pk, m, al, acc, qk2f, qk2h2, dg, j0, lim, t);         \
            wgmma_fence();                                                             \
            _Pragma("unroll")                                                          \
            for (int js = 0; js < 4; js++) wgmma_pv(acc, pk + 4 * js, dV[js]);         \
            wgmma_commit();                                                            \
            softmax_p2(SC, l, al);                                                     \
        }                                                                              \
    }

    for (int kt = 0; kt < NT; kt += 2) {
        ITER1(kt, sA, sB)
        ITER1(kt + 1, sB, sA)
    }
#undef ITER1
    wgmma_wait<0>();
    fence_ubank(pk);
    fence_fbank(acc);
    if (EXPM == 3) {   // l lives in the t==0 lanes' col-0 accumulators
        fence_fbank(accl);
        l[0] = __shfl_sync(0xFFFFFFFFu, accl[0], lane & ~3);
        l[1] = __shfl_sync(0xFFFFFFFFu, accl[2], lane & ~3);
    }
    const float inv0 = vs / fmaxf(l[0], 1e-37f);
    const float inv1 = vs / fmaxf(l[1], 1e-37f);
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const int tok = tok0 + rr;
        if (tok >= L) continue;
        const float inv = rr ? inv1 : inv0;
        __nv_bfloat16* op = out + ((size_t)(b * L + tok) * 16 + kvh * 8 + h) * 64 + 2 * t;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            __nv_bfloat162 o2 = __floats2bfloat162_rn(acc[4 * j + 2 * rr] * inv,
                                                      acc[4 * j + 2 * rr + 1] * inv);
            *reinterpret_cast<unsigned*>(op + 8 * j) = *reinterpret_cast<unsigned*>(&o2);
        }
    }
}

// ------------------------------------------- 2-CTA/SM kernel (occupancy build)
// ncu on the 1-CTA builds: 2.25 active warps/scheduler, 4.24 warp-cycles per
// instruction, issue 53% - latency-starved, nothing saturated. This variant
// fits TWO CTAs per SM (256 threads, no producer warp - tid 0 issues the TMA
// prefetch - no S double-buffer, NST=6 -> 96KB smem, <=128 regs), so stalls in
// one CTA are covered by the other. Grid (cdiv(L,16), B*HKV).
template <int NST, int EXPM = 1>
__global__ void __launch_bounds__(256, 2) ws_fmha2(
    const unsigned char* __restrict__ q, const unsigned char* __restrict__ kvt,
    __nv_bfloat16* __restrict__ out, const int* __restrict__ lens,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    float smqk_log2e, int B, int L, int capblks)
{
    constexpr int PF = NST - 2;
    const int tid = threadIdx.x;
    const int b = blockIdx.y >> 1, kvh = blockIdx.y & 1;
    const int qs = (gridDim.x - 1 - blockIdx.x) * 16;
    const int len = lens[b];
    const int ctx = len - L;
    const int NT = min(capblks, (min(ctx + qs + 16, len) + 127) >> 7);
    const char* src = reinterpret_cast<const char*>(kvt) + ((size_t)(b * 2 + kvh) * capblks) * SB;

    extern __shared__ __align__(1024) char smem[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned mb = sbase + NST * SB;
    if (tid < NST) {
        mbar_init(mb + tid * 8, 1);
        mbar_init(mb + NST * 8 + tid * 8, 256);
    }
    __syncthreads();
    if (tid == 0) {           // prologue stages
        const int n0 = min(PF, NT);
        for (int kt = 0; kt < n0; kt++) {
            mbar_expect_tx(mb + kt * 8, SB);
            bulk_g2s(smem + kt * SB, src + (size_t)kt * SB, SB, mb + kt * 8);
        }
    }

    const int wg = tid >> 7;
    const int wiw = (tid >> 5) & 3;
    const int lane = tid & 31;
    const int t = lane & 3;
    const int h = lane >> 2;

    const float qk2f = k_scale[0] * smqk_log2e;
    const __half2 qk2h2 = __float2half2_rn(qk2f);
    const float vs = v_scale[0];

    unsigned qf[8];
    const int tok0 = qs + 8 * wg + 2 * wiw;
#pragma unroll
    for (int sl2 = 0; sl2 < 2; sl2++)
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            const unsigned char* qp = q + ((size_t)(b * L + tok) * 16 + kvh * 8 + h) * 64
                                    + sl2 * 32 + 4 * t;
            qf[4 * sl2 + rr]     = (tok < L) ? *reinterpret_cast<const unsigned*>(qp) : 0u;
            qf[4 * sl2 + rr + 2] = (tok < L) ? *reinterpret_cast<const unsigned*>(qp + 16) : 0u;
        }

    float m[2] = {M_INIT, M_INIT}, l[2] = {0.f, 0.f}, al[2];
    __half2 nmc[2] = {__float2half2_rn(7.0f - M_INIT), __float2half2_rn(7.0f - M_INIT)};
    float acc[32];
#pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;
    unsigned s[32] = {}, pk[16] = {};
    const int lim = ctx + qs + 8 * wg + 2 * wiw;
    const int dgt = ctx + qs + 8 * wg;

    int sl = 0, cph = 0, prev = -1;
    for (int kt = 0; kt < NT; kt++) {
        if (tid == 0 && kt + PF < NT) {     // consumer-elected TMA prefetch
            const int ps = (kt + PF) % NST;
            const int pr = (kt + PF) / NST;
            if (pr > 0) mbar_wait(mb + NST * 8 + ps * 8, (pr - 1) & 1);
            mbar_expect_tx(mb + ps * 8, SB);
            bulk_g2s(smem + ps * SB, src + (size_t)(kt + PF) * SB, SB, mb + ps * 8);
        }
        mbar_wait(mb + sl * 8, cph);
        const unsigned st = sbase + sl * SB;
        const unsigned long long dK0 = make_desc(st, 2048, 128);
        const unsigned long long dK1 = make_desc(st + 4096, 2048, 128);
        unsigned long long dV[4];
#pragma unroll
        for (int js = 0; js < 4; js++) dV[js] = make_desc(st + 8192 + js * 2048, 1024, 128);
        const int j0 = kt << 7;
        const int dg = (j0 + 127 > dgt);

        wgmma_fence();
        wgmma_qk<0>(s, qf,     dK0);
        wgmma_qk<1>(s, qf + 4, dK1);
        wgmma_commit();
        wgmma_wait<0>();
        if (prev >= 0) mbar_arrive(mb + NST * 8 + prev * 8);

        if (EXPM == 5)
            softmax_lz<0>(s, pk, m, al, nmc, acc, qk2f, qk2h2, dg, j0, lim, t);
        else if (EXPM == 6)
            softmax_lz<1>(s, pk, m, al, nmc, acc, qk2f, qk2h2, dg, j0, lim, t);
        else
            softmax_p1<EXPM>(s, pk, m, al, acc, qk2f, qk2h2, dg, j0, lim, t);
        wgmma_fence();
#pragma unroll
        for (int js = 0; js < 4; js++) wgmma_pv(acc, pk + 4 * js, dV[js]);
        wgmma_commit();
        softmax_p2(s, l, al);

        prev = sl;
        sl++; if (sl == NST) { sl = 0; cph ^= 1; }
    }
    wgmma_wait<0>();
    fence_ubank(pk);
    fence_fbank(acc);

    const float inv0 = vs / fmaxf(l[0], 1e-37f);
    const float inv1 = vs / fmaxf(l[1], 1e-37f);
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const int tok = tok0 + rr;
        if (tok >= L) continue;
        const float inv = rr ? inv1 : inv0;
        __nv_bfloat16* op = out + ((size_t)(b * L + tok) * 16 + kvh * 8 + h) * 64 + 2 * t;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            __nv_bfloat162 o2 = __floats2bfloat162_rn(acc[4 * j + 2 * rr] * inv,
                                                      acc[4 * j + 2 * rr + 1] * inv);
            *reinterpret_cast<unsigned*>(op + 8 * j) = *reinterpret_cast<unsigned*>(&o2);
        }
    }
}

// ---------------------------------------------------------------- gather
// Paged fp8 cache -> per-(kv head, 128-key block) 16KB stage images, zero-filled
// past len. K chunks are direct 16B copies; V bounces through 68B-pitch smem
// (conflict-free byte transpose) and lands pi-permuted within 32-key groups.
__global__ void __launch_bounds__(256) ws_gather(
    const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc,
    unsigned char* __restrict__ kvt,
    const int* __restrict__ pt, const int* __restrict__ lens,
    int pt_stride, int pagelog, int capblks)
{
    __shared__ __align__(16) unsigned char sv[128 * 68];
    const int b = blockIdx.y >> 1, kvh = blockIdx.y & 1;
    const int blk = blockIdx.x;
    const int len = lens[b];
    const int tid = threadIdx.x;
    unsigned char* tile = kvt + ((size_t)(b * 2 + kvh) * capblks + blk) * SB;

    long rowb[2]; bool val[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int key = c >> 2;
        const int pos = (blk << 7) + key;
        val[i] = pos < len;
        if (val[i]) {
            const int page = pt[b * pt_stride + (pos >> pagelog)];
            rowb[i] = ((long)page << pagelog) + (pos & ((1 << pagelog) - 1));
        } else rowb[i] = 0;
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int key = c >> 2, kc4 = c & 3;
        const long off = rowb[i] * 128 + kvh * 64 + kc4 * 16;
        uint4 kw = val[i] ? *reinterpret_cast<const uint4*>(kc + off) : make_uint4(0, 0, 0, 0);
        uint4 vw = val[i] ? *reinterpret_cast<const uint4*>(vc + off) : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(tile + kc4 * 2048 + key * 16) = kw;
        unsigned* sp = reinterpret_cast<unsigned*>(sv + key * 68 + kc4 * 16);
        sp[0] = vw.x; sp[1] = vw.y; sp[2] = vw.z; sp[3] = vw.w;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int kcv = c >> 6, d = c & 63;
        unsigned w[4];
#pragma unroll
        for (int ww = 0; ww < 4; ww++) {
            unsigned r = 0;
#pragma unroll
            for (int bb = 0; bb < 4; bb++) {
                const int lam = kcv * 16 + ww * 4 + bb;
                const int kk = (lam & ~31) | pi_lane(lam & 31);
                r |= (unsigned)sv[kk * 68 + d] << (8 * bb);
            }
            w[ww] = r;
        }
        uint4* dp = reinterpret_cast<uint4*>(tile + 8192 + kcv * 1024 + d * 16);
        *dp = make_uint4(w[0], w[1], w[2], w[3]);
    }
}

// ---------------------------------------------------------------- launchers
template <int NST, int MODE = 0, int ORD = 0, int EXPM = 1>
static void launch_fmha(dim3 grid, cudaStream_t stream,
                        const unsigned char* qp, const unsigned char* kp, __nv_bfloat16* op,
                        const int* lens, const float* ks, const float* vs,
                        float smqk, int B, int L, int capblks) {
    auto fn = ws_fmha<NST, MODE, ORD, EXPM>;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, NST * SB + 512);
        set = true;
    }
    fn<<<grid, 288, NST * SB + 512, stream>>>(qp, kp, op, lens, ks, vs, smqk, B, L, capblks);
}

template <int NST, int EXPM = 1>
static void launch_fmha1(dim3 grid, cudaStream_t stream,
                         const unsigned char* qp, const unsigned char* kp, __nv_bfloat16* op,
                         const int* lens, const float* ks, const float* vs,
                         float smqk, int B, int L, int capblks) {
    auto fn = ws_fmha1<NST, EXPM>;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, NST * SB + 512);
        set = true;
    }
    fn<<<grid, 288, NST * SB + 512, stream>>>(qp, kp, op, lens, ks, vs, smqk, B, L, capblks);
}

template <int NST, int EXPM = 1>
static void launch_fmha2(dim3 grid, cudaStream_t stream,
                         const unsigned char* qp, const unsigned char* kp, __nv_bfloat16* op,
                         const int* lens, const float* ks, const float* vs,
                         float smqk, int B, int L, int capblks) {
    auto fn = ws_fmha2<NST, EXPM>;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, NST * SB + 512);
        set = true;
    }
    fn<<<grid, 256, NST * SB + 512, stream>>>(qp, kp, op, lens, ks, vs, smqk, B, L, capblks);
}

// nst: 6/8/12 = MJ2 kernel (7 EXPM0, 9 ORD1, 10 EXPM2); 21/22/23/24 = MJ1
// QK-ahead kernel (EXPM 1/2/0/3); 31/32 = 2-CTA/SM kernel (EXPM 1, NST 6/4);
// 101/103/107/104 = probe MODEs at NST 8.
void fmha(torch::Tensor q, torch::Tensor kvt, torch::Tensor out, torch::Tensor lens,
          torch::Tensor k_scale, torch::Tensor v_scale, double smqk_log2e,
          int64_t B, int64_t L, int64_t capblks, int64_t nst) {
    dim3 grid((L + 31) / 32, B * 2);
    dim3 grid1((L + 15) / 16, B * 2);
    auto stream = at::cuda::getCurrentCUDAStream();
    const unsigned char* qp = reinterpret_cast<const unsigned char*>(q.data_ptr());
    const unsigned char* kp = reinterpret_cast<const unsigned char*>(kvt.data_ptr());
    __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const int* lp = lens.data_ptr<int>();
    const float *ks = k_scale.data_ptr<float>(), *vs = v_scale.data_ptr<float>();
    const float sm = (float)smqk_log2e;
    if (nst == 6)       launch_fmha<6>(grid, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 12) launch_fmha<12>(grid, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 7)  launch_fmha<8, 0, 0, 0>(grid, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 9)  launch_fmha<8, 0, 1>(grid, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 10) launch_fmha<8, 0, 1, 2>(grid, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 21) launch_fmha1<8, 1>(grid1, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 22) launch_fmha1<8, 2>(grid1, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 23) launch_fmha1<8, 0>(grid1, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 24) launch_fmha1<8, 3>(grid1, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 31) launch_fmha2<6, 1>(grid1, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 32) launch_fmha2<4, 1>(grid1, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 33) launch_fmha2<4, 4>(grid1, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 34) launch_fmha2<6, 4>(grid1, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 35) launch_fmha2<4, 5>(grid1, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 36) launch_fmha2<6, 5>(grid1, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 37) launch_fmha2<4, 6>(grid1, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 101) launch_fmha<8, 1>(grid, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 103) launch_fmha<8, 3>(grid, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 107) launch_fmha<8, 3, 0, 0>(grid, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else if (nst == 104) launch_fmha<8, 4>(grid, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
    else                launch_fmha<8>(grid, stream, qp, kp, op, lp, ks, vs, sm, (int)B, (int)L, (int)capblks);
}

void gather(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt,
            torch::Tensor page_table, torch::Tensor lens,
            int64_t pt_stride, int64_t pagelog, int64_t capblks, int64_t B) {
    dim3 grid(capblks, B * 2);
    ws_gather<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const unsigned char*>(k_cache.data_ptr()),
        reinterpret_cast<const unsigned char*>(v_cache.data_ptr()),
        reinterpret_cast<unsigned char*>(kvt.data_ptr()),
        page_table.data_ptr<int>(), lens.data_ptr<int>(),
        (int)pt_stride, (int)pagelog, (int)capblks);
}
"""
