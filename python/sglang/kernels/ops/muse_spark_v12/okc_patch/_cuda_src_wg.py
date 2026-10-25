"""CUDA source for the wgmma W4A16 grouped-GEMM MoE pair (wg_kernel graft).

Kernel code identical to the certified kernels/muse_prefill_layer/wg_kernel.py
(v10 lane: ~2x Marlin on 16K-token prefill chunks, 67%/61% of bf16 peak, 3 full
courts, 48/48 checks OK), except two inline-asm lines re-wrapped at C string /
operand boundaries for the 400-char diff line limit (same preprocessed tokens).
Entry points: gather (x bf16 -> fp16 tile images in sorted-slot order), gg1
(gate+up GEMM, silu*mul fused), gg2 (down GEMM, scatter by sorted_ids).
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DINL __device__ __forceinline__

// ------------------------------------------------------------------ cp.async
DINL void cpa16(void* smem, const void* gmem) {
    unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sm), "l"(gmem));
}
DINL void cpa4(void* smem, const void* gmem) {
    unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sm), "l"(gmem));
}
// "memory" clobbers: these are COMPILER barriers. Without them ptxas may hoist
// the next iteration's LDS/decode (and the register defs that will reuse the
// in-flight wgmma group's A-fragment physical registers) above the wait -
// measured as run-to-run nondeterministic ~1e-2 errors that vanish under
// compute-sanitizer. CUTLASS's warpgroup intrinsics carry the same clobbers.
DINL void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DINL void cpa_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

// ------------------------------------------------------------------ wgmma plumbing
DINL void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DINL void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
// No "memory" clobber on the wait: the next iteration's LDS is already anchored
// by its __syncthreads (BAR.SYNC is not reorderable), and the clobber was measured
// to serialize the decode of iter kt+1 behind group kt-1's completion.
template <int N> DINL void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N)); }
// cp.async writes are generic-proxy; wgmma reads smem through the async proxy.
DINL void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// ------------------------------------------------------------------ mbarriers
DINL void mbar_init(unsigned bar, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
DINL void mbar_arrive(unsigned bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(bar) : "memory");
}
// arrive once ALL of this thread's prior cp.asyncs have landed. .noinc: the
// completion-arrive counts against the init count (the default form increments
// the expected count first and nets zero -> the phase never completes).
DINL void mbar_arrive_cpa(unsigned bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(bar) : "memory");
}
DINL void mbar_expect_tx(unsigned bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(bar), "r"(bytes) : "memory");
}
// 1D TMA bulk copy global->smem, completion via mbarrier transaction bytes.
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

// Register-lifetime fences (CUTLASS warpgroup_fence_operand): wgmma.mma_async
// reads its register operands ASYNCHRONOUSLY until the group completes, but the
// compiler's model frees an "r" input at issue - without these empty asms it
// reuses A-fragment/accumulator registers inside the async window (measured:
// ~5% NaN outputs at full speed, clean under compute-sanitizer).
DINL void fence_u(unsigned& x) { asm volatile("" : "+r"(x)); }
DINL void fence_bank(unsigned* b) {
#pragma unroll
    for (int i = 0; i < 16; i++) fence_u(b[i]);
}
DINL void fence_acc(float* d) {
#pragma unroll
    for (int i = 0; i < 64; i++) asm volatile("" : "+f"(d[i]));
}

// B-matrix shared memory descriptor (PTX 9.7.14 matrix descriptor format).
DINL unsigned long long make_desc(unsigned saddr, unsigned lbo, unsigned sbo) {
    return (unsigned long long)((saddr >> 4) & 0x3FFF)
         | ((unsigned long long)((lbo >> 4) & 0x3FFF) << 16)
         | ((unsigned long long)((sbo >> 4) & 0x3FFF) << 32);
    // swizzle bits 62-63 = 0 (core matrices stored contiguous, no swizzle needed)
}

// wgmma.mma_async m64n128k16 f32 += f16(A regs) x f16(B desc).
// A-fragment regs are "+r": the hardware reads them ASYNCHRONOUSLY until the
// group completes, so their live ranges must extend past the issue point - a
// plain "r" input lets ptxas reuse the registers inside the async window.
template <int TRN>
DINL void wgmma128(float* d, unsigned* a, unsigned long long db, int accum) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %70, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "{%64,%65,%66,%67}, %68, p, 1, 1, %69;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
          "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
        : "l"(db), "n"(TRN), "r"(accum));
}


// wgmma.mma_async m64n256k16 f32 += f16(A regs) x f16(B desc).
template <int TRN>
DINL void wgmma256(float* d, unsigned* a, unsigned long long db, int accum) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %134, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,"
        "%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127},"
        "{%128,%129,%130,%131}, %132, p, 1, 1, %133;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
          "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]),
          "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]),
          "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]),
          "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]),
          "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127]),
          "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
        : "l"(db), "n"(TRN), "r"(accum));
}

// ------------------------------------------------------------------ e2m1 decode
DINL __half2 u32h2(unsigned v) { return *reinterpret_cast<__half2*>(&v); }
DINL unsigned h2u32(__half2 v) { return *reinterpret_cast<unsigned*>(&v); }

// 8 e2m1 values -> 4 half2, each = true_value * 2^-14 exactly (nibble order set
// by the offline packer so d[0..3] are wgmma A-fragment regs a0..a3 directly).
// Multiplies (= shifts mod 2^32) so ptxas can place them on the FMA pipe (IMAD).
DINL void dq8(unsigned t, __half2 d[4]) {
    unsigned o0 = ((t * 512u)  & 0x0E000E00u) | ((t * 4096u) & 0x80008000u);
    unsigned o1 = ((t * 32u)   & 0x0E000E00u) | ((t * 256u)  & 0x80008000u);
    unsigned o2 = ((t * 2u)    & 0x0E000E00u) | ((t * 16u)   & 0x80008000u);
    unsigned o3 = ((t >> 3)    & 0x0E000E00u) | ( t          & 0x80008000u);
    d[0] = u32h2(o0); d[1] = u32h2(o1); d[2] = u32h2(o2); d[3] = u32h2(o3);
}

DINL __half2 sfh2(unsigned short v) {   // 2 x e4m3 -> 2 x fp16 (exact, hw op)
    unsigned r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return u32h2(r);
}

// ------------------------------------------------------------------ main kernel
// Tile: 128 weight rows (A, decoded in registers) x 128 tokens (B, fp16 smem) x K.
// GATED=1: w13 gate/up pairs (gate at wgmma row r, up at r+8, same thread),
//          epilogue silu(g*gs)*u*gs -> inter fp16 (x 2^-6).
// GATED=0: w2, plain epilogue -> y bf16 scattered by sorted_ids (x 2^20).
constexpr int BK = 64;                  // k per stage (= 4 wgmma k16 = 4 sf groups)
// XNT = tokens per tile (= moe block_m, wgmma n): 128, or 256 for the big-T
// shapes (halves the per-FLOP smem B re-reads AND the staged bytes - the
// measured walls at XN=128: ~836cy of smem port traffic vs 790cy of MMA).
// MJ = m64 tiles per warpgroup (WM = MJ*128 weight rows per CTA).
constexpr int XBf(int xnt) { return xnt * BK * 2; }
constexpr int WBf(int mj) { return mj * 4096; }   // packed-W stage bytes
constexpr int FBf(int mj) { return mj * 512; }    // sf stage bytes
constexpr int SBf(int mj, int xnt) {              // 1KB-aligned stage stride
    return (XBf(xnt) + WBf(mj) + FBf(mj) + 1023) / 1024 * 1024;
}
constexpr int XN = 128;                 // probe_cpa uses the 128-token MJ=1 image
constexpr int XB = XBf(128);
constexpr int WB = WBf(1);
constexpr int SB = SBf(1, 128);

struct StageSrc {
    const char* x;          // tile-image base for this block: xg_t + pid_m*NKT*XB
    const uint4*  w;        // wm  base for (e, wt)
    const unsigned* s;      // sfp base for (e, wt)
};

// X arrives PRE-TILED (xg_t): the gather kernel materializes, per (block, kt),
// the exact 16KB smem stage image ([kc][t] 16B chunks). Staging is then linear
// 16B chunks on BOTH sides. Probe (10-23, this box): row-strided src + [kc][t]
// dst = 2.7 TB/s; linear src + linear dst = 9.7 TB/s aggregate.
// X/W/sf arrive as CONTIGUOUS blocks (tile images), so one elected thread
// issues three 1D TMA bulk copies per stage: replaces ~1536 cp.asyncs per stage
// (their issue+wait pressure was the measured pipeline bubble) and the data
// lands via the async proxy, which wgmma reads natively.
template <int MJ, int XNT>
DINL void stage_in(char* smem, int sl, int kt, int tid, const StageSrc& g, unsigned full_bar) {
    if (tid == 0) {
        char* st = smem + sl * SBf(MJ, XNT);
        constexpr unsigned XBv = XBf(XNT);
        mbar_expect_tx(full_bar, XBv + MJ * 4096 + MJ * 512);
        bulk_g2s(st, g.x + (size_t)kt * XBv, XBv, full_bar);
        bulk_g2s(st + XBv, g.w + (size_t)kt * (MJ * 256), MJ * 4096, full_bar);
        bulk_g2s(st + XBv + WBf(MJ), g.s + (size_t)kt * (128 * MJ), MJ * 512, full_bar);
    }
}

// Rolling mbarrier pipeline state (no block-wide barriers in the mainloop:
// warps wait per-stage mbarriers and run decoupled - the __syncthreads version
// measured 46% tensor-core idle with every warp lockstepped on the barrier).
struct Pipe {
    int sl, cph;        // consume slot + full-phase parity
    int psl, pround;    // produce slot + its round (empty-wait parity = round-1)
    int rsl;            // release slot = (kt - WD) mod NST
    unsigned mb;        // smem addr of full[0]; empty[i] at mb + NST*8
};

template <int NST>
DINL void pipe_adv(Pipe& p) {
    p.sl++;  if (p.sl == NST)  { p.sl = 0; p.cph ^= 1; }
    p.psl++; if (p.psl == NST) { p.psl = 0; p.pround++; }
    p.rsl++; if (p.rsl == NST) p.rsl = 0;
}

template <int GATED, int TRN, int MODE, int NST, int MJ, int WD, int XNT>
DINL void iter_body(char* smem, int kt, int tid, const StageSrc& src,
                    int r0l, int r1l, float* acc, unsigned (&ab)[16 * MJ],
                    unsigned sbase, unsigned long long dhi, int NKT, Pipe& p) {
    constexpr int PF = NST - 1 - WD;   // prefetch distance (slot released at kt-1)
    const int P = kt + PF;
    if (P < NKT) {
        if (p.pround > 0 && tid == 0) mbar_wait(p.mb + NST * 8 + p.psl * 8, (p.pround - 1) & 1);
        stage_in<MJ, XNT>(smem, p.psl, P, tid, src, p.mb + p.psl * 8);
    }
    mbar_wait(p.mb + p.sl * 8, p.cph);

    if (MODE == 0 || MODE == 2 || MODE == 4) {
        const char* st = smem + p.sl * SBf(MJ, XNT);
#pragma unroll
        for (int jj = 0; jj < MJ; jj++) {
            uint4 w = *reinterpret_cast<const uint4*>(st + XBf(XNT) + (jj * 256 + tid) * 16);
            unsigned sfr0 = *reinterpret_cast<const unsigned*>(st + XBf(XNT) + WBf(MJ) + (r0l + jj * 64) * 4);
            unsigned sfr1 = *reinterpret_cast<const unsigned*>(st + XBf(XNT) + WBf(MJ) + (r1l + jj * 64) * 4);
            const unsigned* wq = reinterpret_cast<const unsigned*>(&w);
#pragma unroll
            for (int q = 0; q < 4; q++) {
                __half2 d[4];
                dq8(wq[q], d);
                __half2 sf2 = sfh2((unsigned short)__byte_perm(sfr0, sfr1, 0x0040 + 0x11 * q));
                __half2 s0 = __half2half2(__low2half(sf2));
                __half2 s1 = __half2half2(__high2half(sf2));
                ab[jj * 16 + 4 * q + 0] = h2u32(__hmul2(d[0], s0));
                ab[jj * 16 + 4 * q + 1] = h2u32(__hmul2(d[1], s1));
                ab[jj * 16 + 4 * q + 2] = h2u32(__hmul2(d[2], s0));
                ab[jj * 16 + 4 * q + 3] = h2u32(__hmul2(d[3], s1));
            }
        }
    }
    if (MODE == 0 || MODE == 1 || MODE == 4) {
        unsigned xaddr = sbase + p.sl * SBf(MJ, XNT);
        wgmma_fence();
#pragma unroll
        for (int q = 0; q < 4; q++) {
            unsigned long long db = dhi | ((xaddr + q * (XNT * 32)) >> 4 & 0x3FFF);
#pragma unroll
            for (int jj = 0; jj < MJ; jj++) {
                if (XNT == 128)
                    wgmma128<TRN>(acc + jj * 64, ab + jj * 16 + 4 * q, db,
                                  (kt > 0 || q > 0) ? 1 : 0);
                else
                    wgmma256<TRN>(acc + jj * 128, ab + jj * 16 + 4 * q, db,
                                  (kt > 0 || q > 0) ? 1 : 0);
            }
        }
        wgmma_commit();
        if (MODE == 4) wgmma_wait<0>(); else wgmma_wait<WD>();
    } else {
#pragma unroll
        for (int jj = 0; jj < MJ; jj++)
            fence_bank(ab + jj * 16);      // keep the decode alive in MODE 2
    }
    if (kt >= WD) mbar_arrive(p.mb + NST * 8 + p.rsl * 8);
    pipe_adv<NST>(p);
}

// MODE: 0 = full; 1 = no-decode probe (constant A regs); 2 = no-mma probe;
//       3 = copy-only probe (staging + syncs + epilogue); 4 = wait<0> per iter.
// NST: cp.async stages. OCC: min CTAs/SM for launch bounds.
// MJ:  m64 tiles per warpgroup. D: deep MMA pipeline (wait<2>, 4 banks).
// XNT: tokens per tile (wgmma n; = moe block_m).
template <int GATED, int TRN, int MODE, int NST, int OCC, int MJ, int D, int XNT>
__global__ void __launch_bounds__(256, OCC) wg_moe(
    const __half* __restrict__ xg,       // tiled stage images [cap/XNT, NKT, XNT*8, 16B]
    const uint4* __restrict__ wm,        // [E, NWT, NKT, MJ*256] uint4 packed nibbles
    const unsigned* __restrict__ sfp,    // [E, NWT, NKT, MJ*128] u32 of 4 sf bytes
    const float* __restrict__ gs,        // [E] global scales (GATED only)
    void* __restrict__ out,              // GATED: inter tiled images; else bf16 y [numel, NWT*MJ*128]
    const int* __restrict__ sorted_ids,  // flat route ids (GATED=0 scatter)
    const int* __restrict__ block_experts,
    const int* __restrict__ n_padded,
    int num_pid_m, int NWT, int NKT, int numel,
    unsigned lbo, unsigned sbo, int gsz)
{
    // Rasterization: groups of gsz blocks sweep all w-tiles together, pid_m fastest
    // inside the group -> the group's X tiles AND the expert's W slabs stay
    // L2-resident instead of re-streaming W from HBM per block. gsz=1: wt fastest.
    const int grp = blockIdx.x / (gsz * NWT);
    const int rem = blockIdx.x % (gsz * NWT);
    const int first = grp * gsz;
    const int size = min(num_pid_m - first, gsz);
    const int pid_m = first + rem % size;
    const int wt    = rem / size;
    if (pid_m * XNT >= *n_padded) return;
    const int e = block_experts[pid_m];
    if (e < 0 || e >= 32) return;

    extern __shared__ __align__(1024) char smem[];
    const int tid = threadIdx.x;
    const int wg = tid >> 7;              // warpgroup 0/1

    StageSrc src;
    src.x = reinterpret_cast<const char*>(xg) + (size_t)pid_m * NKT * XBf(XNT);
    src.w = wm + ((size_t)e * NWT + wt) * NKT * (MJ * 256);
    src.s = sfp + ((size_t)e * NWT + wt) * NKT * (MJ * 128);

    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned mb = sbase + NST * SBf(MJ, XNT);
    if (tid < NST) {
        mbar_init(mb + tid * 8, 1);                  // full[s]: expect_tx arrival
        mbar_init(mb + NST * 8 + tid * 8, 256);      // empty[s]: all consumers
    }
    __syncthreads();

    constexpr int WDv = D ? 2 : 1;
    constexpr int PF = NST - 1 - WDv;
    // prologue: stages 0..PF-1
    for (int kt = 0; kt < PF; kt++)
        stage_in<MJ, XNT>(smem, kt, kt, tid, src, mb + kt * 8);

    // thread coords within the warpgroup (PTX wgmma A-fragment layout)
    const int lw = (tid >> 5) & 3;        // warp in wg: rows 16*lw + ...
    const int lane = tid & 31;
    const int r0l = wg * (64 * MJ) + lw * 16 + (lane >> 2);   // sf row, tile j adds 64j
    const int r1l = r0l + 8;

    constexpr int ACC = (XNT / 2) * MJ;   // f32 acc regs per thread
    float acc[ACC];
    unsigned bA[16 * MJ], bB[16 * MJ];
    // B-desc strides are layout constants: k-core slab stride = XNT*16B,
    // n-core (8-token) stride = 128B. (lbo/sbo args kept as an override: 0 = use these.)
    const unsigned lboe = lbo ? lbo : XNT * 16;
    const unsigned sboe = sbo ? sbo : 128;
    const unsigned long long dhi =
        ((unsigned long long)((lboe >> 4) & 0x3FFF) << 16) |
        ((unsigned long long)((sboe >> 4) & 0x3FFF) << 32);

    unsigned bC[D ? 16 * MJ : 16], bD[D ? 16 * MJ : 16];
    if (MODE == 1) {   // constant 1.0 A-fragments: pipeline-without-decode probe
#pragma unroll
        for (int i = 0; i < 16 * MJ; i++) { bA[i] = 0x3C003C00u; bB[i] = 0x3C003C00u; }
        if constexpr (D)
#pragma unroll
            for (int i = 0; i < 16 * MJ; i++) { bC[i] = 0x3C003C00u; bD[i] = 0x3C003C00u; }
    }
    // Banks bA..bD are COMPILE-TIME selected: a runtime abank[kt&1] demotes the
    // fragments to local memory and the async wgmma then reads reused temp regs.
    Pipe p;
    p.sl = 0; p.cph = 0;
    p.psl = PF % NST; p.pround = PF / NST;
    p.rsl = NST - WDv;
    p.mb = mb;
    if constexpr (D) {
        for (int kt = 0; kt < NKT; kt += 4) {
            iter_body<GATED, TRN, MODE, NST, MJ, 2, XNT>(smem, kt, tid, src, r0l, r1l, acc, bA, sbase, dhi, NKT, p);
            iter_body<GATED, TRN, MODE, NST, MJ, 2, XNT>(smem, kt + 1, tid, src, r0l, r1l, acc, bB, sbase, dhi, NKT, p);
            iter_body<GATED, TRN, MODE, NST, MJ, 2, XNT>(smem, kt + 2, tid, src, r0l, r1l, acc, bC, sbase, dhi, NKT, p);
            iter_body<GATED, TRN, MODE, NST, MJ, 2, XNT>(smem, kt + 3, tid, src, r0l, r1l, acc, bD, sbase, dhi, NKT, p);
        }
    } else {
        for (int kt = 0; kt < NKT; kt += 2) {
            iter_body<GATED, TRN, MODE, NST, MJ, 1, XNT>(smem, kt, tid, src, r0l, r1l, acc, bA, sbase, dhi, NKT, p);
            iter_body<GATED, TRN, MODE, NST, MJ, 1, XNT>(smem, kt + 1, tid, src, r0l, r1l, acc, bB, sbase, dhi, NKT, p);
        }
    }
    wgmma_wait<0>();
#pragma unroll
    for (int jj = 0; jj < MJ; jj++) {
        fence_bank(bA + jj * 16);
        fence_bank(bB + jj * 16);
    }
#pragma unroll
    for (int a0 = 0; a0 < ACC; a0 += 64) fence_acc(acc + a0);
    if constexpr (D)
#pragma unroll
        for (int jj = 0; jj < MJ; jj++) { fence_bank(bC + jj * 16); fence_bank(bD + jj * 16); }
    __syncthreads();

    const int t4 = (lane & 3) * 2;        // acc cols 8j + t4 (+1)
    constexpr int NJ = XNT / 8;           // n8 blocks per m64 tile
    if (GATED) {
        // rows: r0 = gate feature f, r1 = up of the SAME f (offline row pairing)
        const float gsc = gs[e] * 16384.0f;          // 2^14 decode comp + global scale
        __half* ism = reinterpret_cast<__half*>(smem);   // [XNT tok][MJ*64 feat]
#pragma unroll
        for (int jj = 0; jj < MJ; jj++) {
            const int fl = wg * (MJ * 32) + jj * 32 + lw * 8 + (lane >> 2);
            const float* ac = acc + jj * (XNT / 2);
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                float g0 = ac[4 * j] * gsc, g1 = ac[4 * j + 1] * gsc;
                float u0 = ac[4 * j + 2] * gsc, u1 = ac[4 * j + 3] * gsc;
                float i0 = g0 / (1.f + __expf(-g0)) * u0 * 0.015625f;   // x 2^-6
                float i1 = g1 / (1.f + __expf(-g1)) * u1 * 0.015625f;
                int tok = 8 * j + t4;
                ism[tok * (MJ * 64) + fl] = __float2half_rn(i0);
                ism[(tok + 1) * (MJ * 64) + fl] = __float2half_rn(i1);
            }
        }
        __syncthreads();
        // inter is written PRE-TILED for gg2: MJ tile images (64 features each).
        char* op = reinterpret_cast<char*>(out) + ((size_t)pid_m * (NWT * MJ) + wt * MJ) * XBf(XNT);
#pragma unroll
        for (int m = 0; m < MJ; m++)
#pragma unroll
            for (int i = 0; i < XNT / 32; i++) {
                int c = i * 256 + tid;                   // c = kc*XNT + t
                int t = c & (XNT - 1), kc = c / XNT;
                *reinterpret_cast<uint4*>(op + m * XBf(XNT) + c * 16) =
                    *reinterpret_cast<const uint4*>(ism + t * (MJ * 64) + m * 64 + kc * 8);
            }
    } else {
        const float sc2 = 1048576.0f;                // 2^14 decode comp * 2^6 inter comp
        __nv_bfloat16* ysm = reinterpret_cast<__nv_bfloat16*>(smem);  // [XNT tok][MJ*128 feat]
#pragma unroll
        for (int jj = 0; jj < MJ; jj++) {
            const int f0 = wg * (MJ * 64) + jj * 64 + lw * 16 + (lane >> 2);
            const float* ac = acc + jj * (XNT / 2);
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                int tok = 8 * j + t4;
                ysm[tok * (MJ * 128) + f0] = __float2bfloat16_rn(ac[4 * j] * sc2);
                ysm[(tok + 1) * (MJ * 128) + f0] = __float2bfloat16_rn(ac[4 * j + 1] * sc2);
                ysm[tok * (MJ * 128) + f0 + 8] = __float2bfloat16_rn(ac[4 * j + 2] * sc2);
                ysm[(tok + 1) * (MJ * 128) + f0 + 8] = __float2bfloat16_rn(ac[4 * j + 3] * sc2);
            }
        }
        __syncthreads();
        __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(out);
#pragma unroll
        for (int i = 0; i < 8 * MJ * (XNT / 128); i++) {
            int c = tid * (8 * MJ * (XNT / 128)) + i;    // (tok, ch) over [XNT][16*MJ]
            int tok = c / (16 * MJ), ch = c % (16 * MJ);
            int rid = sorted_ids[pid_m * XNT + tok];
            if (rid < numel)
                *reinterpret_cast<uint4*>(yp + (size_t)rid * (NWT * MJ * 128) + wt * (MJ * 128) + ch * 8) =
                    *reinterpret_cast<const uint4*>(ysm + tok * (MJ * 128) + ch * 8);
        }
    }
}

// ------------------------------------------------------------------ x gather
// bf16 x rows -> fp16 tile images ([kc][t] 16B chunks per (block, kt)), via an
// smem bounce: row-coalesced loads, linear 32KB stores. (Triton versions of this
// transpose measured 527-741us at 16K tokens; floor is ~95us.)
template <int BM>
__global__ void __launch_bounds__(256) xg_gather(
    const __nv_bfloat16* __restrict__ x, __half* __restrict__ xt,
    const int* __restrict__ sorted_ids, const int* __restrict__ block_experts,
    const int* __restrict__ n_padded, int numel, int topk, int K) {
    extern __shared__ __align__(16) char sm[];           // [8 kc][BM][16B]
    const int pid = blockIdx.y, kt = blockIdx.x;   // kt fastest: full x rows per wave
    // moe_align's worst-case grid covers all 256 experts' blocks: only LOCAL
    // routed blocks are ever read by the GEMMs (without this guard the gather
    // did 5.6x the work at the 16K shapes: 640us vs 115us)
    if (pid * BM >= *n_padded) return;
    const int e = block_experts[pid];
    if (e < 0 || e >= 32) return;
    const int tid = threadIdx.x;
    constexpr int NC = BM * 8;
    constexpr int NI = NC / 256;          // 4 (BM=128) or 8 (BM=256)
    const size_t obase = ((size_t)pid * gridDim.x + kt) * (size_t)(BM * 64);
    // batch the loads: rid -> x-row is a 2-deep dependent miss chain; issuing
    // all NI chains independently keeps NI loads in flight per thread
    // (the serial version measured 519us at 16K tokens; floor ~95us).
    int rids[NI];
#pragma unroll
    for (int i = 0; i < NI; i++) rids[i] = sorted_ids[pid * BM + ((i * 256 + tid) >> 3)];
    uint4 raw[NI];
#pragma unroll
    for (int i = 0; i < NI; i++) {
        int c = i * 256 + tid;
        if (rids[i] < numel)   // topk == 8 (host-asserted): shift, not IDIV
            raw[i] = *reinterpret_cast<const uint4*>(
                x + (size_t)(rids[i] >> 3) * K + kt * 64 + (c & 7) * 8);
        else
            raw[i] = make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int i = 0; i < NI; i++) {
        int c = i * 256 + tid;
        int r = c >> 3, kc = c & 7;
        __half2 h[4];
        if (rids[i] < numel) {
            const __nv_bfloat162* s = reinterpret_cast<const __nv_bfloat162*>(&raw[i]);
#pragma unroll
            for (int j = 0; j < 4; j++) h[j] = __float22half2_rn(__bfloat1622float2(s[j]));
        } else {
#pragma unroll
            for (int j = 0; j < 4; j++) h[j] = __float2half2_rn(0.f);
        }
        *reinterpret_cast<uint4*>(sm + ((size_t)kc * BM + r) * 16) = *reinterpret_cast<const uint4*>(h);
    }
    __syncthreads();
    const uint4* s4 = reinterpret_cast<const uint4*>(sm);
    uint4* d4 = reinterpret_cast<uint4*>(xt + obase);
#pragma unroll
    for (int i = 0; i < NI; i++) { int c = i * 256 + tid; d4[c] = s4[c]; }
}

void gather(torch::Tensor x, torch::Tensor xt, torch::Tensor sorted_ids, torch::Tensor block_experts,
            torch::Tensor n_padded, int64_t num_pid_m, int64_t numel, int64_t topk, int64_t bm) {
    const int K = x.size(1);
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(xg_gather<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
        cudaFuncSetAttribute(xg_gather<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
        set = true;
    }
    dim3 grid(K / 64, num_pid_m);
    auto fn = (bm == 128) ? xg_gather<128> : xg_gather<256>;
    fn<<<grid, 256, bm * 8 * 16, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__half*>(xt.data_ptr()),
        sorted_ids.data_ptr<int>(), block_experts.data_ptr<int>(), n_padded.data_ptr<int>(),
        numel, topk, K);
}

// ------------------------------------------------------------ staging microbench
// Isolates the cp.async staging rate of the exact X+W+sf pattern.
// variant bit0: 1 = linear global source (contiguous per CTA) instead of X rows
// variant bit1: 1 = linear smem dst (no [kc][t] layout)
// variant bit2: 1 = sync per iter (wait<NST-3> + barrier) else wait-at-end only
__global__ void __launch_bounds__(256, 1) probe_cpa(
    const __half* __restrict__ xg, const uint4* __restrict__ wm,
    const unsigned* __restrict__ sfp, float* __restrict__ sink,
    int niter, int variant, int K) {
    extern __shared__ __align__(1024) char smem[];
    const int tid = threadIdx.x;
    const size_t xbase = (size_t)(blockIdx.x % 188) * XN * K;
    const uint4* wsrc = wm + (size_t)(blockIdx.x % 2048) * 256;
    const unsigned* ssrc = sfp + (size_t)(blockIdx.x % 2048) * 128;
    int sl = 0;
    for (int kt = 0; kt < niter; kt++) {
        char* st = smem + sl * SB;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int c = i * 256 + tid;
            if (variant & 1) {
                cpa16(st + c * 16, xg + xbase + (size_t)(kt % 8) * (XN * BK) + c * 8);
            } else {
                int kc = c & 7, t = c >> 3;
                char* dst = (variant & 2) ? st + c * 16 : st + kc * (XN * 16) + t * 16;
                cpa16(dst, xg + xbase + (size_t)t * K + (kt % 64) * BK + kc * 8);
            }
        }
        cpa16(st + XB + tid * 16, wsrc + (size_t)(kt % 64) * 256 + tid);
        if (tid < 128) cpa4(st + XB + WB + tid * 4, ssrc + (size_t)(kt % 64) * 128 + tid);
        cpa_commit();
        if (variant & 4) {
            cpa_wait<5>();
            __syncthreads();
        }
        sl++; if (sl == 8) sl = 0;
    }
    cpa_wait<0>();
    __syncthreads();
    if (tid == 0) sink[blockIdx.x] = *reinterpret_cast<float*>(smem);
}

void probe_copy(torch::Tensor xg, torch::Tensor wm, torch::Tensor sfp, torch::Tensor sink,
                int64_t grid, int64_t niter, int64_t variant, int64_t K) {
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(probe_cpa, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * SB);
        set = true;
    }
    probe_cpa<<<grid, 256, 8 * SB, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(xg.data_ptr()),
        reinterpret_cast<const uint4*>(wm.data_ptr()),
        reinterpret_cast<const unsigned*>(sfp.data_ptr()),
        sink.data_ptr<float>(), niter, variant, K);
}

// ------------------------------------------------------------------ launchers
template <int GATED, int MODE, int NST, int OCC, int MJ, int D = 0, int XNT = 128>
static void launch(const torch::Tensor& xg, const torch::Tensor& wm, const torch::Tensor& sfp,
                   const float* gs, void* out, const int* sorted_ids,
                   const torch::Tensor& block_experts, const torch::Tensor& n_padded,
                   int num_pid_m, int numel, unsigned lbo, unsigned sbo, int gsz = 8) {
    const int NWT = wm.size(1), NKT = wm.size(2);
    static bool attr_set = false;
    auto fn = wg_moe<GATED, 0, MODE, NST, OCC, MJ, D, XNT>;
    if (!attr_set) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, NST * SBf(MJ, XNT) + 256);
        attr_set = true;
    }
    fn<<<num_pid_m * NWT, 256, NST * SBf(MJ, XNT) + 256, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(xg.data_ptr()),
        reinterpret_cast<const uint4*>(wm.data_ptr()),
        reinterpret_cast<const unsigned*>(sfp.data_ptr()),
        gs, out, sorted_ids,
        block_experts.data_ptr<int>(), n_padded.data_ptr<int>(),
        num_pid_m, NWT, NKT, numel, lbo, sbo, gsz);
}

// cfg codes: 4 = XN128/MJ1/NST4/2 CTAs-per-SM; 8 = XN128/MJ1/NST8;
// 25 = XN128/MJ2/NST5; 50 = XN256/MJ1/NST4; 51 = XN256/MJ1/NST5.
#define GG_DISPATCH(G, OUTP, SIDS, NUMEL)                                                  \
    if (cfg == 4)                                                                          \
        launch<G, 0, 4, 2, 1>(xg, wm, sfp, gsp, OUTP, SIDS, block_experts, n_padded,      \
                              num_pid_m, NUMEL, lbo, sbo, (int)gsz);                                 \
    else if (cfg == 25)                                                                    \
        launch<G, 0, 5, 1, 2>(xg, wm, sfp, gsp, OUTP, SIDS, block_experts, n_padded,      \
                              num_pid_m, NUMEL, lbo, sbo, (int)gsz);                                 \
    else if (cfg == 50)                                                                    \
        launch<G, 0, 4, 1, 1, 0, 256>(xg, wm, sfp, gsp, OUTP, SIDS, block_experts,        \
                                      n_padded, num_pid_m, NUMEL, lbo, sbo, (int)gsz);               \
    else if (cfg == 51)                                                                    \
        launch<G, 0, 5, 1, 1, 0, 256>(xg, wm, sfp, gsp, OUTP, SIDS, block_experts,        \
                                      n_padded, num_pid_m, NUMEL, lbo, sbo, (int)gsz);               \
    else if (cfg == 52)                                                                    \
        launch<G, 0, 6, 1, 1, 0, 256>(xg, wm, sfp, gsp, OUTP, SIDS, block_experts,        \
                                      n_padded, num_pid_m, NUMEL, lbo, sbo, (int)gsz);               \
    else if (cfg == 53)                                                                    \
        launch<G, 0, 6, 1, 1, 1, 256>(xg, wm, sfp, gsp, OUTP, SIDS, block_experts,        \
                                      n_padded, num_pid_m, NUMEL, lbo, sbo, (int)gsz);               \
    else                                                                                   \
        launch<G, 0, 8, 1, 1>(xg, wm, sfp, gsp, OUTP, SIDS, block_experts, n_padded,      \
                              num_pid_m, NUMEL, lbo, sbo, (int)gsz);

void gg1(torch::Tensor xg, torch::Tensor wm, torch::Tensor sfp, torch::Tensor gs,
         torch::Tensor inter, torch::Tensor block_experts, torch::Tensor n_padded,
         int64_t num_pid_m, int64_t cfg, int64_t lbo, int64_t sbo, int64_t gsz) {
    const float* gsp = gs.data_ptr<float>();
    GG_DISPATCH(1, inter.data_ptr(), nullptr, 0)
}

void gg1p(torch::Tensor xg, torch::Tensor wm, torch::Tensor sfp, torch::Tensor gs,
          torch::Tensor inter, torch::Tensor block_experts, torch::Tensor n_padded,
          int64_t num_pid_m, int64_t mode, int64_t lbo, int64_t sbo) {
    const float* gsp = gs.data_ptr<float>();
    if (mode == 1)
        launch<1, 1, 8, 1, 1>(xg, wm, sfp, gsp, inter.data_ptr(), nullptr,
                              block_experts, n_padded, num_pid_m, 0, lbo, sbo);
    else if (mode == 2)
        launch<1, 2, 8, 1, 1>(xg, wm, sfp, gsp, inter.data_ptr(), nullptr,
                              block_experts, n_padded, num_pid_m, 0, lbo, sbo);
    else if (mode == 3)
        launch<1, 3, 8, 1, 1>(xg, wm, sfp, gsp, inter.data_ptr(), nullptr,
                              block_experts, n_padded, num_pid_m, 0, lbo, sbo);
    else if (mode == 51)   // XN256 no-decode
        launch<1, 1, 4, 1, 1, 0, 256>(xg, wm, sfp, gsp, inter.data_ptr(), nullptr,
                                      block_experts, n_padded, num_pid_m, 0, lbo, sbo);
    else                   // 53: XN256 copy-only
        launch<1, 3, 4, 1, 1, 0, 256>(xg, wm, sfp, gsp, inter.data_ptr(), nullptr,
                                      block_experts, n_padded, num_pid_m, 0, lbo, sbo);
}

void gg2(torch::Tensor xg, torch::Tensor wm, torch::Tensor sfp,
         torch::Tensor y, torch::Tensor sorted_ids, torch::Tensor block_experts,
         torch::Tensor n_padded, int64_t num_pid_m, int64_t numel,
         int64_t cfg, int64_t lbo, int64_t sbo, int64_t gsz) {
    const float* gsp = nullptr;
    GG_DISPATCH(0, y.data_ptr(), sorted_ids.data_ptr<int>(), numel)
}
"""
