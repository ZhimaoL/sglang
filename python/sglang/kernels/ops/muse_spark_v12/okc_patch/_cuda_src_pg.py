"""CUDA source for the pg grouped-GEMM MoE pair (prefill-gemm graft, BF16 end to end).

Kernels pg_tiles / pg_gather / pg_gemm (namespace pgk) copied from the pg_kernel module of the prefill
bank kernels/muse_prefill_layer/plt_c3.py (sha256 2d1fad0d0b7c42f1). Edits: five stale comments that still said
fp16 / 4608 B (the bank's code is bf16 with WSB = 5120 B blocks) and an unused cuda_fp16.h include dropped.
Serving edits: the scales are 8-bit codes (WSB 4608), decoded to the same bf16 values in registers; the code words
use the DQ3 bit placement (pack_wsf._B3), 8 decode ops per word instead of the bank's 11, same values.
Not carried over: the pgkw P2 kernel, the pg_n4 64-token probe kernel and the probe / split
configurations (none is on the bank's default cfg 350 path).
Entry points: tiles (live block list), gather (bf16 x rows -> bf16 swizzled token images), gg1
(gate+up GEMM, silu*mul fused, bf16 intermediate), gg2 (down GEMM, scatter by sorted_ids, bf16 rows).
Weights stay packed NVFP4 (pack_wsf in pg_moe.py: e2m1 codes reordered, e4m3 block scales stored
exactly as bf16) and are decoded in registers.
"""

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
// (5 LOP3 + 2 SHF + 1 IMAD; the bit layout of the kernel lane's rd_pack_codes_k dq=3, the v7 one took 11).
DINL void dq3(unsigned t, __nv_bfloat162 d[4]) {
    d[0] = u32b2(t & 0x81C081C0u);
    d[1] = u32b2(__funnelshift_l(t, t, 10) & 0x81C081C0u);
    d[2] = u32b2(__funnelshift_l(t, t, 13) & 0x81C081C0u);
    d[3] = u32b2(((t & 0x001B001Bu) * 0x1084u) & 0x81C081C0u);
}

DINL void stsm_x4_trans(unsigned addr, unsigned r0, unsigned r1, unsigned r2, unsigned r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

@@WGMMA@@

// ------------------------------------------------------------------ geometry
constexpr int XNT = 256;                 // tokens per tile (wgmma n)
constexpr int BK  = 64;                  // k per stage
constexpr int NKT = 64;                  // k-tiles (K = 4096 for both GEMMs)
constexpr int XB  = XNT * BK * 2;        // 32768: one k-tile token image [256 t][128 B], 128B-swizzled K-major
constexpr int WSB = 4608;                // merged W(4096) + 8-bit scale codes (512) per (wt, kt); pg_moe.pack_wsf
constexpr int SB  = 37888;               // stage stride: X 32768 + W|sf 4608 padded to a whole number of 1024 B atoms
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
// live[0] = number of live blocks (local expert, inside n_padded); live[1 + i] = pid_m | (wcode << 20).
// wcode = n/64 - 1 for the block's wgmma width n = the smallest allowed width (wmask bit0: 64, bit1: 128,
// bit2: 192; 256 always) covering its valid-token prefix (moe_align pads each expert's range at its end).
// Order: widest class first, pid_m ascending inside a class (round-robin CTAs then end on cheap narrow items).
__global__ void __launch_bounds__(1024) pg_tiles(const int* __restrict__ block_experts,
                                                 const int* __restrict__ n_padded,
                                                 const int* __restrict__ sorted_ids, int numel,
                                                 int num_pid_m, int bm, int wmask, int* __restrict__ live) {
    __shared__ int wsum[32];
    __shared__ int total_s;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int npm = min(num_pid_m, *n_padded / bm);
    if (tid == 0) total_s = 0;
    __syncthreads();
    for (int cls = 3; cls >= 0; cls--) {
        for (int base = 0; base < npm; base += 1024) {
            const int pm = base + tid;
            const bool live_blk = pm < npm && (unsigned)block_experts[pm] < 32u;
            int code = 3;
            if (live_blk && bm == 256) {
                const int* sid = sorted_ids + (size_t)pm * 256;
                if ((wmask & 4) && sid[192] >= numel) code = 2;
                if ((wmask & 2) && sid[128] >= numel) code = 1;
                if ((wmask & 1) && sid[64] >= numel) code = 0;
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

// effective wgmma width of a block: the smallest COMPILED width (WSET bit0 64, bit1 128, bit2 192; 256 always)
// covering the width code from pg_tiles. Producer and consumers must agree (the producer stages nv columns).
template <int WSET> DINL int eff_nv(int wcode) {
    if ((WSET & 1) && wcode == 0) return 64;
    if ((WSET & 2) && wcode <= 1) return 128;
    if ((WSET & 4) && wcode <= 2) return 192;
    return 256;
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
    else wgmma_n64(d, a, db, accum);
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
        // 8-bit codes of the (r0, r1) scale pairs for k16 steps 0..3: bf16(e4m3 * 2^118) has exponent field 236..253 and
        // a 3-bit mantissa, so code = bf16 bits / 16 - 0x75F and each bf16 lane is code * 16 + 0x75F0 (exact, pack_wsf).
        const uint2 sfp = *reinterpret_cast<const uint2*>(stp + WOFF + 4096 + c.sfo);
        const unsigned sfw[2] = {sfp.x, sfp.y};
        const unsigned* wq = reinterpret_cast<const unsigned*>(&w);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            __nv_bfloat162 d[4];
            dq3(wq[q], d);
            const __nv_bfloat162 sf2 = u32b2(__byte_perm(sfw[q >> 1], 0u, (q & 1) ? 0x4342u : 0x4140u) * 16u + 0x75F075F0u);
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

// One item (block pm x w-tile wt) of width NV: mainloop + this warpgroup's epilogue. Same consumer structure as
// wg12 (ptxas keeps the two A banks in distinct registers): item-scoped banks/accumulators, compile-time bank
// selection, unconditional wait<1>, every consumer thread releases the stage of k-tile kt-1 (and the item's
// last stage after its final wait<0>).
// acc / bA / bB are ONE set shared by all width specializations (each width uses a prefix of acc): with
// per-width arrays ptxas kept every width's accumulators apart and ran out of the 232 registers (C7512, spills).
// gsv = the expert's global scale (gg1), rids = the item's 256 route ids in smem (gg2): both staged by the
// producer with the item's first stage, so no consumer ever waits on a global load.
template <int GATED, int NST, int MODE, int NV, int PAIR64, int ACC = 128, int N4 = 0>
DINL void item_body(Ctx& c, float (&acc)[ACC], unsigned (&bA)[16], unsigned (&bB)[16], int pm, int wt, float gsv,
                    const int* rids, void* __restrict__ out, int NWT, int numel) {
    constexpr int PR = (PAIR64 && NV == 64) ? 1 : 0;
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
        if (wtid == 0) bulk_wait_read0();                // previous item's bulk store finished reading epi
        named_bar_sync(bar, 128);
        // this warpgroup's 32 features = one 64 B row per token (SW64 half-tile, independent of the other WG):
        // stmatrix.trans matrix m of an x4 = i-block 4s+m (tokens 8i..8i+7); lane 8m+j -> token t = 8i+j gets this
        // warp's 8 features = chunk lw of the row, stored at chunk lw ^ ((t >> 1) & 3) (64B swizzle; t & 7 = lane & 7)
        const unsigned lanebase = epi + ((lane >> 3) * 8 + (lane & 7)) * 64 + ((lw ^ (((lane & 7) >> 1) & 3)) << 4);
#pragma unroll
        for (int s = 0; s < NV / 32; s++) {
            unsigned r[4];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const int i = 4 * s + m;
                // silu(g) u 2^-6 = (g u 2^-7)(1 + tanh(g / 2)), g / u = acc * gsc
                float t0, t1;
                asm("tanh.approx.f32 %0, %1;" : "=f"(t0) : "f"(acc[4 * i] * hgs));
                asm("tanh.approx.f32 %0, %1;" : "=f"(t1) : "f"(acc[4 * i + 1] * hgs));
                const float p0 = acc[4 * i] * acc[4 * i + 2] * gus, p1 = acc[4 * i + 1] * acc[4 * i + 3] * gus;
                r[m] = b2u32(__floats2bfloat162_rn(fmaf(p0, t0, p0), fmaf(p1, t1, p1)));
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
        const char* epp = c.smem + (epi - c.sbase);
#pragma unroll
        for (int p = 0; p < NV / 64; p++) {              // 64-token passes
            named_bar_sync(bar, 128);                 // previous pass's readers are done
#pragma unroll
            for (int s = 0; s < 4; s++) {                // 4 x4-groups = 16 matrices (8 i-blocks x 2 row halves)
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
            for (int cc = 0; cc < 4; cc++) {
                const int idx = cc * 128 + wtid;
                const int row = idx >> 3, ch = idx & 7;
                const int rid = rids[64 * p + row];
                if (rid < numel)
                    *reinterpret_cast<uint4*>(yp + (size_t)rid * (NWT * 128) + wt * 128 + cw * 64 + ch * 8) =
                        *reinterpret_cast<const uint4*>(epp + row * EPI2_PITCH + ch * 16);
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
    int NWT, int numel, int snake)
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
                const bool pr = PAIR64 && nv == 64;     // two k-tiles per slot (see kstep)
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
                        mbar_expect_tx(mb + sl * 8, 2 * 8192 + 2 * WSB + ridb);
#pragma unroll
                        for (int sub = 0; sub < 2; sub++) {
                            const char* xk = xs + (size_t)(kt + sub) * XB;
                            if (GATED) {
                                bulk_g2s(st + sub * 8192, xk, 8192, mb + sl * 8);
                            } else {
                                bulk_g2s(st + sub * 8192, xk, 4096, mb + sl * 8);
                                bulk_g2s(st + sub * 8192 + 4096, xk + XB / 2, 4096, mb + sl * 8);
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
    c.sfo = 8 * (c.cw * 32 + c.lw * 8 + (c.lane >> 2));    // this thread's (r0, r1) scale-code pairs, 8 B per k-tile
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
        float acc[128];
        unsigned bA[16], bB[16];
        if (MODE != 0) {
#pragma unroll
            for (int i = 0; i < 16; i++) { bA[i] = 0x3F803F80u; bB[i] = 0x3F803F80u; }
        }
        const int nv = eff_nv<WSET>(wcode);
        if ((WSET & 1) && nv == 64)
            item_body<GATED, NST, MODE, 64, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel);
        else if ((WSET & 2) && nv == 128)
            item_body<GATED, NST, MODE, 128, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel);
        else if ((WSET & 4) && nv == 192)
            item_body<GATED, NST, MODE, 192, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel);
        else
            item_body<GATED, NST, MODE, 256, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel);
    }
    if (GATED && wtid == 0) bulk_wait0();               // stores complete before the CTA retires its smem
}

}  // namespace pgk

// ------------------------------------------------------------------ host
static int pg_num_sms() {
    static int n = 0;
    if (!n) n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    return n;
}

void tiles(torch::Tensor block_experts, torch::Tensor n_padded, torch::Tensor sorted_ids, torch::Tensor live,
           int64_t num_pid_m, int64_t bm, int64_t numel, int64_t wmask) {
    pgk::pg_tiles<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
        block_experts.data_ptr<int>(), n_padded.data_ptr<int>(), sorted_ids.data_ptr<int>(), (int)numel,
        (int)num_pid_m, (int)bm, (int)wmask, live.data_ptr<int>());
}

void gather(torch::Tensor x, torch::Tensor xg, torch::Tensor sorted_ids, torch::Tensor live, int64_t numel) {
    TORCH_CHECK(x.size(1) == pgk::NKT * 64 && numel == x.size(0) * 8, "pg gather: K 4096, topk 8");
    const int npm = live.size(0) - 1;
    pgk::pg_gather<3, 8><<<dim3(pgk::NKT / 8, npm), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<char*>(xg.data_ptr()),
        sorted_ids.data_ptr<int>(), live.data_ptr<int>(), (int)numel);
}

template <int GATED, int NST, int MODE, int WSET, int PAIR64 = 1, int SPL = 0>
static void launch(const torch::Tensor& xg, const torch::Tensor& wsf, const float* gs, void* out,
                   const int* sorted_ids, const torch::Tensor& block_experts, const torch::Tensor& live,
                   int numel, int grid, int snake) {
    auto fn = pgk::pg_gemm<GATED, NST, MODE, WSET, PAIR64, SPL>;
    constexpr int smem = pgk::smem_bytes<GATED, MODE>(NST);
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    const int NWT = wsf.size(1);
    TORCH_CHECK(wsf.size(2) == pgk::NKT && wsf.size(3) == pgk::WSB, "pg: wsf layout");
    fn<<<grid > 0 ? grid : pg_num_sms(), 384, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const char*>(xg.data_ptr()), reinterpret_cast<const char*>(wsf.data_ptr()),
        gs, out, sorted_ids, block_experts.data_ptr<int>(), live.data_ptr<int>(), NWT, numel, snake);
}

// snake item order for small calls (few items per CTA): tokens = numel / 8 <= pgk_snake_max_t
static int pgk_snake_max_t = 4096;
void set_snake_max_t(int64_t t) { pgk_snake_max_t = (int)t; }

// cfg 350 = the bank's shipped configuration (widths 64 / 128 / 256, 5 stages), the only one built here.
void gg1(torch::Tensor xg, torch::Tensor wsf, torch::Tensor gs, torch::Tensor inter,
         torch::Tensor block_experts, torch::Tensor live, int64_t numel, int64_t cfg, int64_t grid) {
    TORCH_CHECK(cfg == 350, "pg: only cfg 350 is built");
    const int snake = (numel / 8 <= pgk_snake_max_t) ? 1 : 0;
    launch<1, 5, 0, 3>(xg, wsf, gs.data_ptr<float>(), inter.data_ptr(), nullptr, block_experts, live, 0, grid,
                       snake);
}

void gg2(torch::Tensor xg, torch::Tensor wsf, torch::Tensor y, torch::Tensor sorted_ids,
         torch::Tensor block_experts, torch::Tensor live, int64_t numel, int64_t cfg, int64_t grid) {
    TORCH_CHECK(cfg == 350, "pg: only cfg 350 is built");
    const int snake = (numel / 8 <= pgk_snake_max_t) ? 1 : 0;
    launch<0, 5, 0, 3>(xg, wsf, nullptr, y.data_ptr(), sorted_ids.data_ptr<int>(), block_experts, live,
                       (int)numel, grid, snake);
}
"""

_CUDA_SRC = _CUDA_SRC.replace("@@WGMMA@@", "".join(_wgmma_asm(n) for n in (256, 192, 128, 64)))
