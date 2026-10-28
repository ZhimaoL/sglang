"""CUDA source of the OKC prefill FMHA (okc_patch/prefill_fmha.py): the prefill lane's fmha7 path, trimmed
from muse-opt serving/kernels/fmha/wsattn_c4b.py (md5 56ad280c; bank plt_c4b 5db91e12) by mk_src.py:
pgfa (fmha7 + combine) and skw::gather_kp verbatim, plus three host launchers."""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DINL __device__ __forceinline__

#ifndef PG_NOWIN
#define PG_NOWIN 0
#endif
#ifndef PG_DBG
#define PG_DBG 0
#endif

namespace pgfa {

constexpr int TRK = 48;
__device__ unsigned long long* g_trace = nullptr;

DINL unsigned long long clk() { unsigned long long t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }

DINL void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DINL void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> DINL void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N)); }

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
DINL void bulk_g2s_u(unsigned sd, const void* gsrc, unsigned bytes, unsigned bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sd), "l"(gsrc), "r"(bytes), "r"(bar) : "memory");
}
DINL void mbar_wait(unsigned bar, unsigned phase) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}\n" :: "r"(bar), "r"(phase) : "memory");
}
DINL void mbar_wait_sleep(unsigned bar, unsigned phase) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n"
        "@!p bra WAIT_%=;\n}\n" :: "r"(bar), "r"(phase), "r"(0x989680) : "memory");
}

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

DINL unsigned long long make_desc(unsigned saddr, unsigned lbo, unsigned sbo) {
    return (unsigned long long)((saddr >> 4) & 0x3FFF)
         | ((unsigned long long)((lbo >> 4) & 0x3FFF) << 16)
         | ((unsigned long long)((sbo >> 4) & 0x3FFF) << 32);
}
DINL unsigned long long desc_lo(unsigned lo) { return (unsigned long long)lo | (8ull << 32); }

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

template <int ACC>
DINL void wgmma_qk_ss(unsigned* d, unsigned long long da, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        " %32, %33, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
        : "l"(da), "l"(db), "n"(ACC));
}

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

template <int ACC>
DINL void wgmma_l(float* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3\n"
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(ACC));
}

DINL __half2 u32h2(unsigned v) { return *reinterpret_cast<__half2*>(&v); }
DINL unsigned h2u32(__half2 v) { return *reinterpret_cast<unsigned*>(&v); }
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
DINL int pi_lane(int lam) {
    return (lam & 1) | (((lam >> 2) & 3) << 1) | (((lam >> 1) & 1) << 3) | (lam & 16);
}

constexpr int SB = 16384;
constexpr unsigned NEG = 0xFC00;
#define M_INIT (-30000.0f)

DINL void reg_dealloc_24() { asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n" ::: "memory"); }
DINL void reg_alloc_160() { asm volatile("setmaxnreg.inc.sync.aligned.u32 160;\n" ::: "memory"); }
DINL void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
DINL void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }

template <long long OPT> struct Cfg {
    static constexpr bool HOIST = (OPT & 2) != 0;
    static constexpr int EMU = (OPT & 4) ? 1 : 0;
    static constexpr bool DBP = (OPT & 8) != 0;
    static constexpr int RELU = (OPT & 16) ? 1 : 0;
    static constexpr int F16C = (OPT & 32) ? 1 : 0;
    static constexpr int MUC = (OPT >> 8) & 7;
    static constexpr int MU = MUC == 1 ? 3 : MUC == 2 ? 2 : MUC == 3 ? 5 : MUC == 4 ? 1 : MUC == 5 ? 0 :
                              MUC == 6 ? 6 : MUC == 7 ? 8 : 4;
    static constexpr bool PP = (OPT & 2048) == 0;
    static constexpr bool PEEL = (OPT & 4096) != 0;
    static constexpr bool PIN = (OPT & 8192) != 0;
    static constexpr int ABL = (OPT >> 16) & 31;
    static constexpr bool TRACE = ((OPT >> 21) & 1) != 0;
    static constexpr bool PIPE = ((OPT >> 22) & 1) != 0;
    static constexpr bool QSM = ((OPT >> 23) & 1) != 0;
    static constexpr bool DUI = ((OPT >> 24) & 1) != 0;   // dual: second phase = first PWL's bits + 512 per half
    static constexpr bool SPEC = ((OPT >> 25) & 1) != 0;
    static constexpr bool IMADX = ((OPT >> 26) & 1) != 0;
    static constexpr bool PREF = ((OPT >> 27) & 1) != 0;
    static constexpr int NC = ((OPT >> 28) & 1) ? 4 : 3;   // consumer warpgroups (8 tokens x 8 heads each)
    // NC 4: producer warpgroup at 24 registers, consumers at 112 (launch 96: 128*24 + 512*112 <= 640*96 + 128*72);
    // a lone producer warp caps the block at 96 registers, and bulk copies issued by a consumer thread stall
    // that warpgroup ~1000 clk per block
    static constexpr bool INL = false;
    static constexpr bool PW = false;
    static constexpr int THREADS = 128 * (NC + 1);
    // dual-phase emulated exp: registers per 8 (bits 29-30: 0, 2, 3, 4; bit 15 adds 4) computed as
    // PWL(y) + 2^-0.5 PWL(y + 0.5)
    // (the two Schraudolph PWL errors are half a period apart: ratio spread 0.947..1.037 -> 0.975..1.021);
    // with DU > 0 every exp path produces 2^(x + 8) (P <= 2^8.75 < 448) so the classes share one scale
    static constexpr int DUC = (OPT >> 29) & 3;
    static constexpr int DU = (DUC == 1 ? 2 : DUC == 2 ? 3 : DUC == 3 ? 4 : 0) + ((OPT & 32768) ? 4 : 0);
    static constexpr bool DUD = (OPT & 16384) != 0;   // dither: second phase at y + 0.5 + 1/64 (own relu)
    static constexpr float EOFF = DU > 0 ? 8.0f : 7.0f;
    // bit 32: MUFU path shifted by MB = log2(1.0113 / 0.9992) to match the PWL's mean ratio after e4m3
    // bit 33: nmr derived from the rounded nmc (nmr = nmc + 14 - (EOFF - 7)) so both classes share one rounding
    static constexpr float MB = ((OPT >> 32) & 1) ? 0.01737f : 0.0f;
    static constexpr bool NMX = ((OPT >> 33) & 1) != 0;
    static constexpr unsigned PWLC = DU > 0 ? 0xF7B5F7C5u + 0x04000400u : 0xF7B5F7C5u;
};

struct Prod { int nxt; int NT; const char* src; };

// non-blocking mbarrier phase test (1 if the phase with this parity has completed)
DINL unsigned mbar_test(unsigned bar, unsigned phase) {
    unsigned r;
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(r) : "r"(bar), "r"(phase) : "memory");
    return r;
}

// fmha4's softmax_a (mask, lazy max, hybrid exp in place)
template <int WIN, long long OPT>
DINL int softmax7(unsigned (&s)[32], float (&m)[2], float (&al)[2], __half2 (&nmc)[2], __half2 (&nmr)[2],
                  __half2 (&thr)[2], float qk2f, __half2 qk2h2,
                  int diag, int j0, int lim0, int t, int lowm = 0, int lo0 = 0, unsigned k32 = 32u) {
    using C = Cfg<OPT>;
    if (WIN && lowm) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int k0 = j0 + 2 * t + 8 * j;
            if (k0 < lo0)         s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
            if (k0 + 1 < lo0)     s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
            if (k0 < lo0 + 1)     s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
            if (k0 + 1 < lo0 + 1) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
        }
    }
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
    al[0] = 1.0f; al[1] = 1.0f;
    int resc = 0;
    if (!(C::ABL & 2)) {
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
        bool ok;
        if (C::F16C) {
            ok = __hble2(t0v[0], thr[0]) & __hble2(t1v[0], thr[1]);
        } else {
            const float f0 = fmaxf(__half2float(__low2half(t0v[0])), __half2float(__high2half(t0v[0])));
            const float f1 = fmaxf(__half2float(__low2half(t1v[0])), __half2float(__high2half(t1v[0])));
            ok = (f0 * qk2f <= m[0] + 0.75f) & (f1 * qk2f <= m[1] + 0.75f);
        }
        resc = !__all_sync(0xFFFFFFFFu, ok);
        if (resc) {
            const float f0 = fmaxf(__half2float(__low2half(t0v[0])), __half2float(__high2half(t0v[0])));
            const float f1 = fmaxf(__half2float(__low2half(t1v[0])), __half2float(__high2half(t1v[0])));
            float r0 = f0, r1 = f1;
            r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 1));
            r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 2));
            r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 1));
            r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 2));
            const float mn0 = fmaxf(m[0], r0 * qk2f);
            const float mn1 = fmaxf(m[1], r1 * qk2f);
            al[0] = exp2f(m[0] - mn0);
            al[1] = exp2f(m[1] - mn1);
            m[0] = mn0; m[1] = mn1;
            nmc[0] = __float2half2_rn(C::EOFF + C::MB - mn0);
            nmc[1] = __float2half2_rn(C::EOFF + C::MB - mn1);
            if (C::RELU) {
                if (C::NMX) {
                    const __half2 d = __float2half2_rn(21.0f - C::EOFF - C::MB);
                    nmr[0] = __hadd2(nmc[0], d);
                    nmr[1] = __hadd2(nmc[1], d);
                } else {
                    nmr[0] = __float2half2_rn(21.0f - mn0);
                    nmr[1] = __float2half2_rn(21.0f - mn1);
                }
            }
            if (C::F16C) {
                thr[0] = __float2half2_rn((mn0 + 0.75f) / qk2f);
                thr[1] = __float2half2_rn((mn1 + 0.75f) / qk2f);
            }
        }
    }
    if (!(C::ABL & 1)) {
        const __half2 hsc   = __float2half2_rn(1024.0f);
        const __half2 hmag  = __float2half2_rn(50176.0f);
        const __half2 hclmp = __float2half2_rn(35840.0f);
        const __half2 hclmp2 = __float2half2_rn(36352.0f);
        const __half2 hcd = __float2half2_rn(C::DUD ? 0.6994198f : 0.70710678f);   // 2^-(0.5 + 1/64) / 2^-0.5
        const __half2 hdd = __float2half2_rn(0.515625f);
        constexpr int MU = C::MU;
        constexpr int DU = C::DU;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const int rk = (((j >> 1) & 7) * 2 + (j & 1)) % 16;
            const bool mufu = (MU == 4 && DU == 0) ? ((j & 2) == 0) : (rk < 2 * MU);
            const bool dual = DU > 0 && !mufu && rk < 2 * (MU + DU);
            if (mufu) {
                s[j] = ex2_f16x2(h2u32(__hfma2(u32h2(s[j]), qk2h2, nmc[j & 1])));
            } else if (dual) {
                const __half2 yp = __hfma2_relu(u32h2(s[j]), qk2h2, nmr[j & 1]);
                const unsigned e1 = h2u32(__hfma2(yp, hsc, hclmp)) * 32u + 0xF7B5F7C5u;
                unsigned e2;
                if (C::DUI && !C::DUD) {
                    e2 = e1 + 0x02000200u;
                } else if (C::DUD) {
                    const __half2 yp2 = __hadd2(yp, hdd);
                    e2 = h2u32(__hfma2(yp2, hsc, hclmp)) * (C::IMADX ? k32 : 32u) + 0xF7B5F7C5u;
                } else {
                    e2 = h2u32(__hfma2(yp, hsc, hclmp2)) * (C::IMADX ? k32 : 32u) + 0xF7B5F7C5u;
                }
                s[j] = h2u32(__hfma2(u32h2(e2), hcd, u32h2(e1)));
            } else if (C::RELU) {
                const __half2 yp = __hfma2_relu(u32h2(s[j]), qk2h2, nmr[j & 1]);
                if (C::IMADX) s[j] = h2u32(__hfma2(yp, hsc, hclmp)) * k32 + C::PWLC;
                else s[j] = h2u32(__hfma2(yp, hsc, hclmp)) * 32u + C::PWLC;
            } else {
                __half2 y  = __hfma2(u32h2(s[j]), qk2h2, nmc[j & 1]);
                __half2 xb = __hmax2(__hfma2(y, hsc, hmag), hclmp);
                if (C::EMU) s[j] = h2u32(xb) * 32u - 0x084A083Bu;
                else s[j] = (h2u32(xb) & 0x03FF03FFu) * 32u - 0x083B083Bu;
            }
        }
    }
    return resc;
}

// hybrid exp of 32 registers (softmax7's exp loop, out of place)
template <long long OPT>
DINL void exp_regs(const unsigned (&s)[32], unsigned (&e)[32], const __half2 (&nmc)[2], const __half2 (&nmr)[2],
                   __half2 qk2h2) {
    using C = Cfg<OPT>;
    const __half2 hsc   = __float2half2_rn(1024.0f);
    const __half2 hmag  = __float2half2_rn(50176.0f);
    const __half2 hclmp = __float2half2_rn(35840.0f);
    constexpr int MU = C::MU;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        const bool mufu = (MU == 4) ? ((j & 2) == 0) : ((((j >> 1) & 7) * 2 + (j & 1)) % 16 < 2 * MU);
        if (mufu) {
            e[j] = ex2_f16x2(h2u32(__hfma2(u32h2(s[j]), qk2h2, nmc[j & 1])));
        } else if (C::RELU) {
            const __half2 yp = __hfma2_relu(u32h2(s[j]), qk2h2, nmr[j & 1]);
            e[j] = h2u32(__hfma2(yp, hsc, hclmp)) * 32u - 0x084A083Bu;
        } else {
            __half2 y  = __hfma2(u32h2(s[j]), qk2h2, nmc[j & 1]);
            __half2 xb = __hmax2(__hfma2(y, hsc, hmag), hclmp);
            if (C::EMU) e[j] = h2u32(xb) * 32u - 0x084A083Bu;
            else e[j] = (h2u32(xb) & 0x03FF03FFu) * 32u - 0x083B083Bu;
        }
    }
}

// SPEC: the exp runs with the running max before the lazy-max vote (off the max tree's dependency chain) into e;
// a failed vote (some score above m + 0.75) recomputes it from s with the new max
template <int WIN, long long OPT>
DINL int softmax_spec(unsigned (&s)[32], unsigned (&e)[32], float (&m)[2], float (&al)[2], __half2 (&nmc)[2],
                      __half2 (&nmr)[2], __half2 (&thr)[2], float qk2f, __half2 qk2h2,
                      int diag, int j0, int lim0, int t, int lowm = 0, int lo0 = 0) {
    using C = Cfg<OPT>;
    if (WIN && lowm) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int k0 = j0 + 2 * t + 8 * j;
            if (k0 < lo0)         s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
            if (k0 + 1 < lo0)     s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
            if (k0 < lo0 + 1)     s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
            if (k0 + 1 < lo0 + 1) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
        }
    }
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
    exp_regs<OPT>(s, e, nmc, nmr, qk2h2);
    al[0] = 1.0f; al[1] = 1.0f;
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
    bool ok;
    if (C::F16C) {
        ok = __hble2(t0v[0], thr[0]) & __hble2(t1v[0], thr[1]);
    } else {
        const float f0 = fmaxf(__half2float(__low2half(t0v[0])), __half2float(__high2half(t0v[0])));
        const float f1 = fmaxf(__half2float(__low2half(t1v[0])), __half2float(__high2half(t1v[0])));
        ok = (f0 * qk2f <= m[0] + 0.75f) & (f1 * qk2f <= m[1] + 0.75f);
    }
    const int resc = !__all_sync(0xFFFFFFFFu, ok);
    if (resc) {
        const float f0 = fmaxf(__half2float(__low2half(t0v[0])), __half2float(__high2half(t0v[0])));
        const float f1 = fmaxf(__half2float(__low2half(t1v[0])), __half2float(__high2half(t1v[0])));
        float r0 = f0, r1 = f1;
        r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 1));
        r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 2));
        r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 1));
        r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 2));
        const float mn0 = fmaxf(m[0], r0 * qk2f);
        const float mn1 = fmaxf(m[1], r1 * qk2f);
        al[0] = exp2f(m[0] - mn0);
        al[1] = exp2f(m[1] - mn1);
        m[0] = mn0; m[1] = mn1;
        nmc[0] = __float2half2_rn(7.0f - mn0);
        nmc[1] = __float2half2_rn(7.0f - mn1);
        if (C::RELU) {
            nmr[0] = __float2half2_rn(21.0f - mn0);
            nmr[1] = __float2half2_rn(21.0f - mn1);
        }
        if (C::F16C) {
            thr[0] = __float2half2_rn((mn0 + 0.75f) / qk2f);
            thr[1] = __float2half2_rn((mn1 + 0.75f) / qk2f);
        }
        exp_regs<OPT>(s, e, nmc, nmr, qk2h2);
    }
    return resc;
}

template <long long OPT>
DINL void softmax_cvt(const unsigned (&s)[32], unsigned (&pk)[16]) {
    if (Cfg<OPT>::ABL & 8) {
#pragma unroll
        for (int i = 0; i < 16; i++) pk[i] = s[2 * i] ^ s[2 * i + 1];
        return;
    }
#pragma unroll
    for (int js = 0; js < 4; js++) {
        pk[4 * js]     = (unsigned)cvt_e4m3x2(s[8 * js])     | ((unsigned)cvt_e4m3x2(s[8 * js + 2]) << 16);
        pk[4 * js + 1] = (unsigned)cvt_e4m3x2(s[8 * js + 1]) | ((unsigned)cvt_e4m3x2(s[8 * js + 3]) << 16);
        pk[4 * js + 2] = (unsigned)cvt_e4m3x2(s[8 * js + 4]) | ((unsigned)cvt_e4m3x2(s[8 * js + 6]) << 16);
        pk[4 * js + 3] = (unsigned)cvt_e4m3x2(s[8 * js + 5]) | ((unsigned)cvt_e4m3x2(s[8 * js + 7]) << 16);
    }
}

DINL void rescale_acc(float (&acc)[32], const float (&al)[2]) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
        acc[4 * j]     *= al[0]; acc[4 * j + 1] *= al[0];
        acc[4 * j + 2] *= al[1]; acc[4 * j + 3] *= al[1];
    }
}

DINL void issue_qk(unsigned (&s)[32], unsigned* qf, unsigned st) {
    const unsigned long long dK0 = make_desc(st, 2048, 128);
    const unsigned long long dK1 = make_desc(st + 4096, 2048, 128);
    wgmma_fence();
    wgmma_qk<0>(s, qf, dK0);
    wgmma_qk<1>(s, qf + 4, dK1);
    wgmma_commit();
}

template <long long OPT>
DINL void issue_pvl(float (&acc)[32], float (&lblk)[4], unsigned (&pk)[16], unsigned st, unsigned long long dones) {
    fence_fbank(acc);
    fence_ubank(pk);
    wgmma_fence();
#pragma unroll
    for (int js = 0; js < 4; js++) {
        wgmma_pv(acc, pk + 4 * js, make_desc(st + 8192 + js * 2048, 1024, 128));
        if (!(Cfg<OPT>::ABL & 4)) {
            if (js == 0) wgmma_l<0>(lblk, pk, dones);
            else wgmma_l<1>(lblk, pk + 4 * js, dones);
        }
    }
    wgmma_commit();
}

DINL void issue_qk_h(unsigned (&s)[32], unsigned* qf, unsigned dlo) {
    wgmma_fence();
    wgmma_qk<0>(s, qf, desc_lo(dlo));
    wgmma_qk<1>(s, qf + 4, desc_lo(dlo + 256));
    wgmma_commit();
}

template <long long OPT>
DINL void issue_pvl_h(float (&acc)[32], float (&lblk)[4], unsigned (&pk)[16], unsigned dlo,
                      unsigned long long dones) {
    fence_fbank(acc);
    fence_ubank(pk);
    wgmma_fence();
#pragma unroll
    for (int js = 0; js < 4; js++) {
        wgmma_pv(acc, pk + 4 * js, desc_lo(dlo + js * 128));
        if (!(Cfg<OPT>::ABL & 4)) {
            if (js == 0) wgmma_l<0>(lblk, pk, dones);
            else wgmma_l<1>(lblk, pk + 4 * js, dones);
        }
    }
    wgmma_commit();
}

DINL void issue_qk_s(unsigned (&s)[32], unsigned dqa, unsigned dlo) {
    wgmma_fence();
    wgmma_qk_ss<0>(s, desc_lo(dqa), desc_lo(dlo));
    wgmma_qk_ss<1>(s, desc_lo(dqa + 128), desc_lo(dlo + 256));
    wgmma_commit();
}

template <long long OPT>
DINL void issue_qk_x(unsigned (&s)[32], unsigned (&qf)[8], unsigned dqa, unsigned dlo) {
    if (Cfg<OPT>::QSM) issue_qk_s(s, dqa, dlo);
    else issue_qk_h(s, qf, dlo);
}

// PIPE form of issue_pvl_h: the row sums accumulate across blocks in lacc (rescaled with acc)
template <long long OPT>
DINL void issue_pvl_acc(float (&acc)[32], float (&lacc)[4], unsigned (&pk)[16], unsigned dlo,
                        unsigned long long dones) {
    fence_fbank(acc);
    fence_fbank(lacc);
    fence_ubank(pk);
    wgmma_fence();
#pragma unroll
    for (int js = 0; js < 4; js++) {
        wgmma_pv(acc, pk + 4 * js, desc_lo(dlo + js * 128));
        if (!(Cfg<OPT>::ABL & 4) && !(PG_DBG & 16)) wgmma_l<1>(lacc, pk + 4 * js, dones);
    }
    wgmma_commit();
}

#define STAMP(i) do { if (C::TRACE && tr != nullptr && kt < TRK) tr[kt * 8 + (i)] = clk(); } while (0)

// PIPE block kt >= 1 (nothing in flight on entry): sx holds S(kt) (QK issued and retired one block earlier), py
// holds P(kt-1); sy and px are free. Issues QK(kt+1) -> sy (unless LAST) and PV(kt-1) from py, runs the softmax of
// S(kt) into px under them, then retires both at the end of the block: no wgmma is in flight across a loop back
// edge (ptxas serializes every wgmma (C7514/C7515) when one is and its registers are touched later in the loop).
template <int NST, int WIN, long long OPT, bool LAST>
DINL void blockP(int kt, unsigned (&sx)[32], unsigned (&sy)[32], unsigned (&px)[16], unsigned (&py)[16],
                 float (&acc)[32], float (&lacc)[4], float (&m)[2], float (&al)[2], __half2 (&nmc)[2],
                 __half2 (&nmr)[2], __half2 (&thr)[2], unsigned (&qf)[8], unsigned dqa, unsigned mb, int cw,
                 unsigned dqk, unsigned dpv, unsigned long long dones, int kb0, int dgt, int lim, int t, int win,
                 float qk2f, __half2 qk2h2, unsigned long long* tr) {
    using C = Cfg<OPT>;
    constexpr int NC = C::NC;
    constexpr bool PP = C::PP;
    const unsigned sl1 = (unsigned)(kt + 1) & (NST - 1);
    const unsigned slp = (unsigned)(kt - 1) & (NST - 1);
    STAMP(0);
    if (!LAST) mbar_wait(mb + sl1 * 8, ((unsigned)(kt + 1) / NST) & 1);
    STAMP(1);
    if (PP) named_bar_sync(1 + cw, 256);
    STAMP(2);
    if (!LAST) issue_qk_x<OPT>(sy, qf, dqa, dqk + (sl1 << 10));
    issue_pvl_acc<OPT>(acc, lacc, py, dpv + (slp << 10), dones);
    if (PP) named_bar_arrive(1 + (cw + 1) % NC, 256);
    STAMP(3);
    const int j0 = (kb0 + kt) << 7;
    const int resc = softmax7<WIN, OPT>(sx, m, al, nmc, nmr, thr, qk2f, qk2h2, (j0 + 127 > dgt), j0, lim, t,
                                        WIN ? (j0 < dgt + 7 - (win - 1)) : 0, lim - (win - 1));
    STAMP(4);
    if (resc) {
        wgmma_wait<0>();
        fence_fbank(acc);
        fence_fbank(lacc);
        rescale_acc(acc, al);
#pragma unroll
        for (int i = 0; i < 4; i++) lacc[i] *= al[i >> 1];
    }
    STAMP(5);
    softmax_cvt<OPT>(sx, px);
    STAMP(6);
    wgmma_wait<0>();
    fence_ubank(sy);
    fence_ubank(px);
    fence_ubank(py);
    fence_fbank(acc);
    fence_fbank(lacc);
    mbar_arrive(mb + NST * 8 + slp * 8);
    STAMP(7);
}

// one key block kt >= 1 of a consumer warpgroup: QK(kt) + PV(kt-1) (P from pkc); P(kt) into pkn
template <int NST, int WIN, long long OPT, int MASK = 1>
DINL void block7(int kt, unsigned (&pkc)[16], unsigned (&pkn)[16], float (&acc)[32], float (&l)[2], float (&m)[2],
                 float (&al)[2], __half2 (&nmc)[2], __half2 (&nmr)[2], __half2 (&thr)[2], unsigned (&qf)[8],
                 unsigned mb, int cw, unsigned sbase,
                 unsigned dqk, unsigned dpv, unsigned long long dones, int kb0, int dgt, int lim, int t, int win,
                 float qk2f, __half2 qk2h2, unsigned long long* tr, unsigned k32, unsigned& rdy, Prod& pd,
                 bool pthr, unsigned dqa) {
    using C = Cfg<OPT>;
    constexpr int NC = C::NC;
    constexpr bool HOIST = C::HOIST;
    constexpr bool PP = C::PP;
    unsigned s[32];
    float lblk[4];
    const unsigned sl = HOIST ? ((unsigned)kt & (NST - 1)) : (unsigned)(kt % NST);
    const unsigned slp = HOIST ? ((unsigned)(kt - 1) & (NST - 1)) : (unsigned)((kt - 1) % NST);
    STAMP(0);
    if (!C::PREF || !rdy) mbar_wait(mb + sl * 8, HOIST ? (((unsigned)kt / NST) & 1) : ((kt / NST) & 1));
    STAMP(1);
    if (PP) named_bar_sync(1 + cw, 256);
    STAMP(2);
    if (C::ABL & 16) {
#pragma unroll
        for (int j = 0; j < 32; j++) s[j] = qf[j & 7] + (unsigned)kt * 0x00010001u;
        fence_ubank(pkc);
#pragma unroll
        for (int j = 0; j < 4; j++) lblk[j] = __uint_as_float(pkc[j]);
    } else if (HOIST) {
        issue_qk_x<OPT>(s, qf, dqa, dqk + (sl << 10));
        issue_pvl_h<OPT>(acc, lblk, pkc, dpv + (slp << 10), dones);
    } else {
        issue_qk(s, qf, sbase + sl * SB);
        issue_pvl<OPT>(acc, lblk, pkc, sbase + slp * SB, dones);
    }
    if (PP) named_bar_arrive(1 + (cw + 1) % NC, 256);
    // the next stage's full barrier, tested now so the test latency hides under the QK wait
    if (C::PREF) rdy = mbar_test(mb + ((unsigned)(kt + 1) & (NST - 1)) * 8, ((unsigned)(kt + 1) / NST) & 1);
    STAMP(3);
    wgmma_wait<1>();
    fence_ubank(s);
    STAMP(4);
    const int j0 = (kb0 + kt) << 7;
    unsigned e[32];
    int resc;
    if (C::SPEC)
        resc = softmax_spec<WIN && MASK, OPT>(s, e, m, al, nmc, nmr, thr, qk2f, qk2h2,
                                             MASK ? (j0 + 127 > dgt) : 0, j0, lim, t,
                                             (WIN && MASK) ? (j0 < dgt + 7 - (win - 1)) : 0, lim - (win - 1));
    else
        resc = softmax7<WIN && MASK, OPT>(s, m, al, nmc, nmr, thr, qk2f, qk2h2,
                                         MASK ? (j0 + 127 > dgt) : 0, j0, lim, t,
                                         (WIN && MASK) ? (j0 < dgt + 7 - (win - 1)) : 0, lim - (win - 1), k32);
    STAMP(5);
    if (C::DBP) softmax_cvt<OPT>(s, pkn);
    wgmma_wait<0>();
    fence_fbank(acc);
    fence_fbank(lblk);
    fence_ubank(pkc);
    mbar_arrive(mb + NST * 8 + slp * 8);
    if (C::INL && pthr) {
        while (pd.nxt < pd.NT && pd.nxt < kt + NST) {
            const unsigned ps = (unsigned)pd.nxt & (NST - 1), pr = (unsigned)pd.nxt / NST;
            if (!mbar_test(mb + NST * 8 + ps * 8, (pr - 1) & 1)) break;
            mbar_expect_tx(mb + ps * 8, SB);
            bulk_g2s_u(sbase + ps * SB, pd.src + (size_t)pd.nxt * SB, SB, mb + ps * 8);
            pd.nxt++;
        }
    }
    STAMP(6);
    if (OPT & 64) {
        if (resc) {
            rescale_acc(acc, al);
            l[0] = (l[0] + lblk[0]) * al[0];
            l[1] = (l[1] + lblk[2]) * al[1];
        } else {
            l[0] += lblk[0];
            l[1] += lblk[2];
        }
    } else {
        if (resc) rescale_acc(acc, al);
        l[0] = (l[0] + lblk[0]) * al[0];
        l[1] = (l[1] + lblk[2]) * al[1];
    }
    if (!C::DBP) {
        if (C::SPEC) softmax_cvt<OPT>(e, pkn);
        else softmax_cvt<OPT>(s, pkn);
    }
    fence_ubank(pkn);
    STAMP(7);
}

template <int NST, int SPLIT, int WIN, long long OPT>
__global__ void __launch_bounds__(Cfg<OPT>::THREADS, 1) fmha7(
    const unsigned char* __restrict__ q, const unsigned char* __restrict__ kvt,
    __nv_bfloat16* __restrict__ out, __half* __restrict__ po, float2* __restrict__ pml,
    const int* __restrict__ lens,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    float smqk_log2e, int B, int L, int capblks, int win,
    const __nv_bfloat16* __restrict__ gate, int gstride, float ng_eps)
{
    using C = Cfg<OPT>;
    constexpr int NC = C::NC;
    constexpr int QB = 8 * NC;
    constexpr bool HOIST = C::HOIST;
    const int tid = threadIdx.x;
    const int b = blockIdx.y >> 1, kvh = blockIdx.y & 1;
    const int qs = (gridDim.x - 1 - blockIdx.x) * QB;
    const int len = lens[b];
    const int ctx = len - L;
    const int NTF = min(capblks, (min(ctx + qs + QB, len) + 127) >> 7);
    const int KLO = WIN ? (max(0, ctx + qs - (win - 1)) >> 7) : 0;
    int kb0 = KLO, kb1 = NTF;
    if (SPLIT) {
        kb0 = KLO + (int)(((long long)(NTF - KLO) * blockIdx.z) / gridDim.z);
        kb1 = KLO + (int)(((long long)(NTF - KLO) * (blockIdx.z + 1)) / gridDim.z);
    }
    const int NT = kb1 - kb0;
    const char* src = reinterpret_cast<const char*>(kvt) + ((size_t)(b * 2 + kvh) * capblks + kb0) * SB;

    extern __shared__ __align__(1024) char smem[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned mb = sbase + NST * SB;
    if (tid < NST) {
        mbar_init(mb + tid * 8, 1);
        mbar_init(mb + NST * 8 + tid * 8, 128 * NC);
    }
    if (tid >= 64 && tid < 128) {
        reinterpret_cast<unsigned*>(smem + NST * SB + 256)[tid - 64] = 0x38383838u;   // e4m3 1.0
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    const int wg = tid >> 7;
    Prod pd{0, NT, src};
    if (C::INL) {
        if (tid == 0) {
            for (; pd.nxt < min(NT, NST); pd.nxt++) {
                mbar_expect_tx(mb + pd.nxt * 8, SB);
                bulk_g2s(smem + pd.nxt * SB, src + (size_t)pd.nxt * SB, SB, mb + pd.nxt * 8);
            }
        }
    } else if (C::PW ? (tid >= 128 * NC) : (wg == 0)) {
        if (!C::PW) reg_dealloc_24();
        if (tid == (C::PW ? 128 * NC : 0)) {
            for (int kt = 0; kt < NT; kt++) {
                const int ps = kt % NST, pr = kt / NST;
                if (pr > 0) {
                    if (OPT & 1) mbar_wait_sleep(mb + NST * 8 + ps * 8, (pr - 1) & 1);
                    else mbar_wait(mb + NST * 8 + ps * 8, (pr - 1) & 1);
                }
                mbar_expect_tx(mb + ps * 8, SB);
                bulk_g2s(smem + ps * SB, src + (size_t)kt * SB, SB, mb + ps * 8);
            }
        }
        return;
    }
    if (NC == 4) asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\n" ::: "memory");
    else if (!C::INL && !C::PW) reg_alloc_160();
    const int cw = (C::INL || C::PW) ? wg : wg - 1;
    const bool pthr = C::INL && tid == 0;
    const int wiw = (tid >> 5) & 3;
    const int lane = tid & 31;
    const int t = lane & 3;
    const int h = lane >> 2;
    unsigned long long* tr = nullptr;
    if (C::TRACE && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && (tid & 127) == 0)
        tr = g_trace + cw * TRK * 8;

    const float qk2f = k_scale[0] * smqk_log2e;
    const __half2 qk2h2 = __float2half2_rn(qk2f);
    const float vs = v_scale[0];

    unsigned qf[8];
    const int tok0 = qs + 8 * cw + 2 * wiw;
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
    __half2 nmc[2] = {__float2half2_rn(C::EOFF + C::MB - M_INIT), __float2half2_rn(C::EOFF + C::MB - M_INIT)};
    __half2 nmr[2] = {__float2half2_rn(21.0f - M_INIT), __float2half2_rn(21.0f - M_INIT)};
    __half2 thr[2] = {__float2half2_rn((M_INIT + 0.75f) / qk2f), __float2half2_rn((M_INIT + 0.75f) / qk2f)};
    constexpr bool PP = C::PP;
    float acc[32];
#pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;
    unsigned pk[16], pk2[16];
    unsigned long long dones = make_desc(sbase + NST * SB + 256, 128, 128);
    unsigned dqk = ((sbase >> 4) & 0x3FFF) | (128u << 16);
    unsigned dpv = (((sbase + 8192) >> 4) & 0x3FFF) | (64u << 16);
    if (C::PIN) {
        dqk = __shfl_sync(0xFFFFFFFFu, dqk, 0);
        dpv = __shfl_sync(0xFFFFFFFFu, dpv, 0);
        dones = desc_lo(__shfl_sync(0xFFFFFFFFu, (unsigned)dones, 0));
    }
    const unsigned qsm = sbase + NST * SB + 512 + cw * 4096;
    const unsigned dqa = ((qsm >> 4) & 0x3FFF) | (64u << 16);
    if (C::QSM) {
#pragma unroll
        for (int sl2 = 0; sl2 < 2; sl2++)
#pragma unroll
            for (int rr = 0; rr < 2; rr++)
#pragma unroll
                for (int hk = 0; hk < 2; hk++) {
                    const unsigned a = qsm + (2 * sl2 + hk) * 1024 + (wiw * 2 + rr) * 128 + h * 16 + 4 * t;
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(a), "r"(qf[4 * sl2 + rr + 2 * hk]) : "memory");
                }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        named_bar_sync(8 + cw, 128);
    }
    const int lim = ctx + qs + 8 * cw + 2 * wiw;
    const int dgt = ctx + qs + 8 * cw;
    const unsigned k32 = 32u + (unsigned)(__float_as_uint(qk2f) == 0x7FC00001u);
    unsigned rdy = 0;
#define BLOCK7(KT, PKC, PKN) block7<NST, WIN, OPT, 1>(KT, PKC, PKN, acc, l, m, al, nmc, nmr, thr, qf, mb, cw, sbase, \
                                                     dqk, dpv, dones, kb0, dgt, lim, t, win, qk2f, qk2h2, tr, k32, rdy, \
                                                     pd, pthr, dqa)
#define BLOCK7U(KT, PKC, PKN) block7<NST, WIN, OPT, 0>(KT, PKC, PKN, acc, l, m, al, nmc, nmr, thr, qf, mb, cw, sbase, \
                                                     dqk, dpv, dones, kb0, dgt, lim, t, win, qk2f, qk2h2, tr, k32, rdy, \
                                                     pd, pthr, dqa)

    if (C::PIPE && NT > 0) {
        unsigned sA[32], sB[32], pB[16];
        float lacc[4] = {0.f, 0.f, 0.f, 0.f};
        mbar_wait(mb, 0);
        if (NT > 1) mbar_wait(mb + 8, 0);
        issue_qk_x<OPT>(sA, qf, dqa, dqk);
        if (NT > 1) issue_qk_x<OPT>(sB, qf, dqa, dqk + (1u << 10));
        wgmma_wait<0>();
        fence_ubank(sA);
        fence_ubank(sB);
        {
            const int j0 = kb0 << 7;
            softmax7<WIN, OPT>(sA, m, al, nmc, nmr, thr, qk2f, qk2h2, (j0 + 127 > dgt), j0, lim, t,
                               WIN ? (j0 < dgt + 7 - (win - 1)) : 0, lim - (win - 1));
            softmax_cvt<OPT>(sA, pk);
        }
        if (PP && NT > 1 && cw == NC - 1) named_bar_arrive(1, 256);
#define BLOCKP(LST, KT, SX, SY, PX, PY) blockP<NST, WIN, OPT, LST>(KT, SX, SY, PX, PY, acc, lacc, m, al, nmc, nmr, \
                                        thr, qf, dqa, mb, cw, dqk, dpv, dones, kb0, dgt, lim, t, win, qk2f, qk2h2, tr)
        int kt = 1;
        for (; kt + 2 < NT; kt += 2) {
            BLOCKP(false, kt, sB, sA, pB, pk);
            BLOCKP(false, kt + 1, sA, sB, pk, pB);
        }
        // kt == NT - 2 (one full block, then the last) or kt == NT - 1 (the last) or kt == NT (none)
        bool lastB;   // the last P sits in pB
        if (kt + 1 < NT) {
            BLOCKP(false, kt, sB, sA, pB, pk);
            BLOCKP(true, kt + 1, sA, sB, pk, pB);
            lastB = false;
        } else if (kt < NT) {
            BLOCKP(true, kt, sB, sA, pB, pk);
            lastB = true;
        } else {
            lastB = false;
        }
#undef BLOCKP
        if (PP && NT > 1 && cw == 0) named_bar_sync(1, 256);
        const unsigned sll = (unsigned)(NT - 1) & (NST - 1);
        if (lastB) issue_pvl_acc<OPT>(acc, lacc, pB, dpv + (sll << 10), dones);
        else issue_pvl_acc<OPT>(acc, lacc, pk, dpv + (sll << 10), dones);
        wgmma_wait<0>();
        fence_fbank(acc);
        fence_fbank(lacc);
        l[0] = lacc[0];
        l[1] = lacc[2];
    } else if (NT > 0) {
        {
            unsigned s[32];
            mbar_wait(mb, 0);
            if (C::ABL & 16) {
#pragma unroll
                for (int j = 0; j < 32; j++) s[j] = qf[j & 7];
            } else {
                if (HOIST) issue_qk_x<OPT>(s, qf, dqa, dqk);
                else issue_qk(s, qf, sbase);
                wgmma_wait<0>();
            }
            fence_ubank(s);
            const int j0 = kb0 << 7;
            softmax7<WIN, OPT>(s, m, al, nmc, nmr, thr, qk2f, qk2h2, (j0 + 127 > dgt), j0, lim, t,
                               WIN ? (j0 < dgt + 7 - (win - 1)) : 0, lim - (win - 1));
            softmax_cvt<OPT>(s, pk);
        }
        if (PP && NT > 1 && cw == NC - 1) named_bar_arrive(1, 256);
        if (C::DBP) {
            int kt = 1;
            for (; kt + 1 < NT; kt += 2) {
                BLOCK7(kt, pk, pk2);
                BLOCK7(kt + 1, pk2, pk);
            }
            if (kt < NT) {
                BLOCK7(kt, pk, pk2);
#pragma unroll
                for (int i = 0; i < 16; i++) pk[i] = pk2[i];
            }
        } else {
            if (C::PEEL) {
                const int klo = WIN ? max(1, min(NT, ((dgt + 8 - win + 127) >> 7) - kb0)) : 1;
                const int kdg = max(klo, min(NT, ((dgt - 127) >> 7) + 1 - kb0));
                int kt = 1;
                for (; kt < klo; kt++) BLOCK7(kt, pk, pk);
                for (; kt < kdg; kt++) BLOCK7U(kt, pk, pk);
                for (; kt < NT; kt++) BLOCK7(kt, pk, pk);
            } else {
                for (int kt = 1; kt < NT; kt++) BLOCK7(kt, pk, pk);
            }
        }
#undef BLOCK7
#undef BLOCK7U
        if (PP && NT > 1 && cw == 0) named_bar_sync(1, 256);
        float lblk[4];
        const unsigned sll = HOIST ? ((unsigned)(NT - 1) & (NST - 1)) : (unsigned)((NT - 1) % NST);
        if (!(C::ABL & 16)) {
            if (HOIST) issue_pvl_h<OPT>(acc, lblk, pk, dpv + (sll << 10), dones);
            else issue_pvl<OPT>(acc, lblk, pk, sbase + sll * SB, dones);
            wgmma_wait<0>();
        } else {
#pragma unroll
            for (int j = 0; j < 4; j++) lblk[j] = __uint_as_float(pk[j]);
        }
        fence_fbank(acc);
        fence_fbank(lblk);
        l[0] += lblk[0];
        l[1] += lblk[2];
    }

    const float inv0 = vs / fmaxf(l[0], 1e-37f);
    const float inv1 = vs / fmaxf(l[1], 1e-37f);
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const int tok = tok0 + rr;
        if (tok >= L) continue;
        const float inv = rr ? inv1 : inv0;
        if (!SPLIT && gate != nullptr) {
            // fused o_norm_gate (sk_fa7's epilogue): y = bf16(x * rsqrt(mean(x^2) + eps) * sigmoid(g)) over the
            // head's 64 dims, x = the bf16 attention output; the row is spread over the 4 lanes with this h
            const size_t r = (size_t)(b * L + tok) * 16 + kvh * 8 + h;
            float xv[16];
            float ss = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float2 x2 = __bfloat1622float2(__floats2bfloat162_rn(acc[4 * j + 2 * rr] * inv,
                                                                           acc[4 * j + 2 * rr + 1] * inv));
                xv[2 * j] = x2.x;
                xv[2 * j + 1] = x2.y;
                ss = fmaf(x2.x, x2.x, ss);
                ss = fmaf(x2.y, x2.y, ss);
            }
            ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 1);
            ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 2);
            const float rstd = rsqrtf(fmaf(ss, 1.0f / 64.0f, ng_eps));
            const __nv_bfloat16* gp = gate + (size_t)(b * L + tok) * gstride + (kvh * 8 + h) * 64 + 2 * t;
            __nv_bfloat16* op = out + r * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp + 8 * j));
                const float y0 = xv[2 * j] * rstd * (1.0f / (1.0f + __expf(-g2.x)));
                const float y1 = xv[2 * j + 1] * rstd * (1.0f / (1.0f + __expf(-g2.y)));
                __nv_bfloat162 o2 = __floats2bfloat162_rn(y0, y1);
                *reinterpret_cast<unsigned*>(op + 8 * j) = *reinterpret_cast<unsigned*>(&o2);
            }
        } else if (!SPLIT) {
            __nv_bfloat16* op = out + ((size_t)(b * L + tok) * 16 + kvh * 8 + h) * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                __nv_bfloat162 o2 = __floats2bfloat162_rn(acc[4 * j + 2 * rr] * inv,
                                                          acc[4 * j + 2 * rr + 1] * inv);
                *reinterpret_cast<unsigned*>(op + 8 * j) = *reinterpret_cast<unsigned*>(&o2);
            }
        } else {
            const size_t row = ((size_t)blockIdx.z * B * L + (size_t)b * L + tok) * 16 + kvh * 8 + h;
            __half* op = po + row * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                __half2 o2 = __floats2half2_rn(acc[4 * j + 2 * rr] * inv, acc[4 * j + 2 * rr + 1] * inv);
                *reinterpret_cast<__half2*>(op + 8 * j) = o2;
            }
            if (t == 0) pml[row] = make_float2(m[rr], l[rr]);
        }
    }
}

template <int NST, int SPLIT, int WIN, long long OPT>
static void launch_fmha7(dim3 grid, cudaStream_t stream,
                         const unsigned char* qp, const unsigned char* kp, __nv_bfloat16* op,
                         __half* pop, float2* pmp,
                         const int* lens, const float* ks, const float* vs,
                         float smqk, int B, int L, int capblks, int win,
                         const __nv_bfloat16* gate, int gstride, float ng_eps) {
    auto fn = fmha7<NST, SPLIT, WIN, OPT>;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, NST * SB + 512 + 4 * 4096);
        set = true;
    }
    fn<<<grid, Cfg<OPT>::THREADS, NST * SB + 512 + (Cfg<OPT>::QSM ? 4 * 4096 : 0), stream>>>(qp, kp, op, pop, pmp, lens, ks, vs, smqk, B, L, capblks, win,
                                                                                           gate, gstride, ng_eps);
}

template <long long OPT>
static void run_fmha7(dim3 grid, cudaStream_t st, const unsigned char* qp, const unsigned char* kp,
                      __nv_bfloat16* op, __half* pop, float2* pmp, const int* lp, const float* ks,
                      const float* vsp, float smqk, int B, int L, int capblks, int S, int win,
                      const __nv_bfloat16* gate = nullptr, int gstride = 0, float ng_eps = 0.f) {
    if (S <= 1) {
#if !PG_NOWIN
        if (win > 0) launch_fmha7<8, 0, 1, OPT>(grid, st, qp, kp, op, nullptr, nullptr, lp, ks, vsp, smqk, B, L, capblks, win, gate, gstride, ng_eps);
        else
#endif
        launch_fmha7<8, 0, 0, OPT>(grid, st, qp, kp, op, nullptr, nullptr, lp, ks, vsp, smqk, B, L, capblks, 0, gate, gstride, ng_eps);
    } else {
#if !PG_NOWIN
        if (win > 0) launch_fmha7<8, 1, 1, OPT>(grid, st, qp, kp, nullptr, pop, pmp, lp, ks, vsp, smqk, B, L, capblks, win, nullptr, 0, 0.f);
        else
#endif
        launch_fmha7<8, 1, 0, OPT>(grid, st, qp, kp, nullptr, pop, pmp, lp, ks, vsp, smqk, B, L, capblks, 0, nullptr, 0, 0.f);
    }
}

__global__ void __launch_bounds__(256) combine(
    const __half* __restrict__ po, const float2* __restrict__ pml,
    __nv_bfloat16* __restrict__ out, int rows, int S)
{
    const int gid = blockIdx.x * 256 + threadIdx.x;
    const int row = gid >> 3, c = gid & 7;
    if (row >= rows) return;
    float M = M_INIT;
    for (int s = 0; s < S; s++) M = fmaxf(M, pml[(size_t)s * rows + row].x);
    float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float wsum = 0.f;
    for (int s = 0; s < S; s++) {
        const float2 ml = pml[(size_t)s * rows + row];
        const float w = ml.y * exp2f(ml.x - M);
        wsum += w;
        const uint4 v = *reinterpret_cast<const uint4*>(po + ((size_t)s * rows + row) * 64 + c * 8);
        const unsigned vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 f = __half22float2(u32h2(vv[i]));
            a[2 * i] += w * f.x;
            a[2 * i + 1] += w * f.y;
        }
    }
    const float inv = 1.0f / fmaxf(wsum, 1e-37f);
    unsigned o[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        __nv_bfloat162 b2 = __floats2bfloat162_rn(a[2 * i] * inv, a[2 * i + 1] * inv);
        o[i] = *reinterpret_cast<unsigned*>(&b2);
    }
    *reinterpret_cast<uint4*>(out + (size_t)row * 64 + c * 8) = make_uint4(o[0], o[1], o[2], o[3]);
}

// split-KV merge with the fused o_norm_gate (sk_fa7's combine_ng): 8 threads per (token, head) row, 8 dims each
__global__ void __launch_bounds__(256) combine_ng(
    const __half* __restrict__ po, const float2* __restrict__ pml,
    __nv_bfloat16* __restrict__ out, int rows, int S,
    const __nv_bfloat16* __restrict__ gate, int gstride, float ng_eps)
{
    const int gid = blockIdx.x * 256 + threadIdx.x;
    const int row = gid >> 3, c = gid & 7;
    const bool live = row < rows;
    float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float wsum = 0.f;
    if (live) {
        float M = M_INIT;
        for (int s = 0; s < S; s++) M = fmaxf(M, pml[(size_t)s * rows + row].x);
        for (int s = 0; s < S; s++) {
            const float2 ml = pml[(size_t)s * rows + row];
            const float w = ml.y * exp2f(ml.x - M);
            wsum += w;
            const uint4 v = *reinterpret_cast<const uint4*>(po + ((size_t)s * rows + row) * 64 + c * 8);
            const unsigned vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float2 f = __half22float2(u32h2(vv[i]));
                a[2 * i] += w * f.x;
                a[2 * i + 1] += w * f.y;
            }
        }
    }
    const float inv = 1.0f / fmaxf(wsum, 1e-37f);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const float2 x2 = __bfloat1622float2(__floats2bfloat162_rn(a[i] * inv, a[i + 1] * inv));
        a[i] = x2.x;
        a[i + 1] = x2.y;
        ss = fmaf(x2.x, x2.x, ss);
        ss = fmaf(x2.y, x2.y, ss);
    }
    ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 1);
    ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 2);
    ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 4);
    if (!live) return;
    const float rstd = rsqrtf(fmaf(ss, 1.0f / 64.0f, ng_eps));
    const int tok = row >> 4, head = row & 15;
    const uint4 gv = *reinterpret_cast<const uint4*>(gate + (size_t)tok * gstride + head * 64 + c * 8);
    const unsigned gg[4] = {gv.x, gv.y, gv.z, gv.w};
    unsigned o[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gg[i]));
        __nv_bfloat162 b2 = __floats2bfloat162_rn(a[2 * i] * rstd * (1.0f / (1.0f + __expf(-g2.x))),
                                                  a[2 * i + 1] * rstd * (1.0f / (1.0f + __expf(-g2.y))));
        o[i] = *reinterpret_cast<unsigned*>(&b2);
    }
    *reinterpret_cast<uint4*>(out + (size_t)row * 64 + c * 8) = make_uint4(o[0], o[1], o[2], o[3]);
}

__global__ void __launch_bounds__(256) gather_k(
    const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc,
    unsigned char* __restrict__ kvt,
    const int* __restrict__ pt, const int* __restrict__ lens, const int* __restrict__ k0s,
    int pt_stride, int pagelog, int capblks)
{
    __shared__ __align__(16) unsigned char sv[128 * 68];
    const int b = blockIdx.y >> 1, kvh = blockIdx.y & 1;
    const int blk = blockIdx.x;
    const int len = lens[b];
    const int kst = k0s ? k0s[b] : 0;
    const int tid = threadIdx.x;
    unsigned char* tile = kvt + ((size_t)(b * 2 + kvh) * capblks + blk) * SB;

    long rowb[2]; bool val[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int key = c >> 2;
        const int pos = kst + (blk << 7) + key;
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

}  // namespace pgfa

namespace pfg {

constexpr int SB = 16384;
DINL int pi_lane(int lam) {
    return (lam & 1) | (((lam >> 2) & 3) << 1) | (((lam >> 1) & 1) << 3) | (lam & 16);
}

// skw::gather_kp, verbatim
__global__ void __launch_bounds__(256) gather_kp(
    const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc,
    unsigned char* __restrict__ kvt,
    const int* __restrict__ pt, const int* __restrict__ lens, const int* __restrict__ seq,
    int* __restrict__ lens_out, int pt_stride, int pagelog, int capblks, int L, int win)
{
    __shared__ __align__(16) unsigned char sv[128 * 68];
    const int b = blockIdx.y >> 1, kvh = blockIdx.y & 1;
    const int blk = blockIdx.x;
    const int len = lens[b];
    const int kst = seq ? max(seq[b] - (win - 1), 0) : 0;
    const bool old_blk = kst + (blk << 7) + 127 < len - L;
    const int tid = threadIdx.x;
    if (lens_out && blk == 0 && kvh == 0 && tid == 0) lens_out[b] = len - kst;
    unsigned char* tile = kvt + ((size_t)(b * 2 + kvh) * capblks + blk) * SB;

    long rowb[2]; bool val[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int key = c >> 2;
        const int pos = kst + (blk << 7) + key;
        val[i] = pos < len;
        if (val[i]) {
            const int page = pt[b * pt_stride + (pos >> pagelog)];
            rowb[i] = ((long)page << pagelog) + (pos & ((1 << pagelog) - 1));
        } else rowb[i] = 0;
    }
    if (!old_blk) asm volatile("griddepcontrol.wait;" ::: "memory");   // the new tokens' cache rows come from qkf
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
    if (old_blk) asm volatile("griddepcontrol.wait;" ::: "memory");   // our completion must imply qkf's
}

}  // namespace pfg

static void pg_fmha4(torch::Tensor q, torch::Tensor kvt, torch::Tensor out, torch::Tensor po, torch::Tensor pml,
                     torch::Tensor lens, torch::Tensor k_scale, torch::Tensor v_scale, double smqk_log2e,
                     int64_t B, int64_t L, int64_t capblks, int64_t S, int64_t win, int64_t nc,
                     const __nv_bfloat16* gate, int64_t gstride, double ng_eps) {
    dim3 grid3((L + 23) / 24, B * 2, S), grid4((L + 31) / 32, B * 2, S);
    auto st = at::cuda::getCurrentCUDAStream();
    auto qp = reinterpret_cast<const unsigned char*>(q.data_ptr());
    auto kp = reinterpret_cast<const unsigned char*>(kvt.data_ptr());
    auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    auto pop = reinterpret_cast<__half*>(po.data_ptr());
    auto pmp = reinterpret_cast<float2*>(pml.data_ptr());
    const int* lp = lens.data_ptr<int>();
    const float *ks = k_scale.data_ptr<float>(), *vsp = v_scale.data_ptr<float>();
    if (nc == 4)
        pgfa::run_fmha7<4706009687LL>(grid4, st, qp, kp, op, pop, pmp, lp, ks, vsp, (float)smqk_log2e, (int)B, (int)L,
                                  (int)capblks, (int)S, (int)win, gate, (int)gstride, (float)ng_eps);
    else
        pgfa::run_fmha7<4496294487LL>(grid3, st, qp, kp, op, pop, pmp, lp, ks, vsp, (float)smqk_log2e, (int)B, (int)L,
                                  (int)capblks, (int)S, (int)win, gate, (int)gstride, (float)ng_eps);
}

void fmha4n_split(torch::Tensor q, torch::Tensor kvt, torch::Tensor out, torch::Tensor po, torch::Tensor pml,
                  torch::Tensor lens, torch::Tensor k_scale, torch::Tensor v_scale, double smqk_log2e,
                  int64_t B, int64_t L, int64_t capblks, int64_t S, int64_t win, int64_t nc) {
    pg_fmha4(q, kvt, out, po, pml, lens, k_scale, v_scale, smqk_log2e, B, L, capblks, S, win, nc, nullptr, 0, 0.0);
}

void combine(torch::Tensor po, torch::Tensor pml, torch::Tensor out, int64_t rows, int64_t S) {
    const int blocks = (int)((rows * 8 + 255) / 256);
    pgfa::combine<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(po.data_ptr()), reinterpret_cast<const float2*>(pml.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)rows, (int)S);
}

void gatherp(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
             torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
             int64_t capblks, int64_t B, int64_t L, int64_t win) {
    pfg::gather_kp<<<dim3((unsigned)capblks, (unsigned)(B * 2)), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const unsigned char*>(k_cache.data_ptr()),
        reinterpret_cast<const unsigned char*>(v_cache.data_ptr()),
        reinterpret_cast<unsigned char*>(kvt.data_ptr()),
        (const int*)page_table.data_ptr<int>(), (const int*)lens.data_ptr<int>(),
        seq.numel() ? (const int*)seq.data_ptr<int>() : (const int*)nullptr,
        lens_out.numel() ? lens_out.data_ptr<int>() : (int*)nullptr,
        (int)pt_stride, (int)pagelog, (int)capblks, (int)L, (int)win);
}
"""
