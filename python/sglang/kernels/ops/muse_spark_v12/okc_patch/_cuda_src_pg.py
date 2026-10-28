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

// bf16 weight word (pack_wsf): codes c0..c7 (A-fragment order: c0 / c4 -> register 0 lo / hi, c1 / c5 -> 1, c2 / c6
// -> 2, c3 / c7 -> 3) with magnitude bits at m and sign bit at s:
//   c0 m 6-8   s 15 | c4 m 22-24 s 31 -> o0 = t & 0x81C081C0                                  (in place)
//   c1 m 0-2   s 9  | c5 m 16-18 s 25 -> o1 = (t << 6) & 0x81C081C0
//   c2 m 3-5   s 10 | c6 m 19-21 s 26 -> o2 = ((t << 3) & 0x01C001C0) | ((t << 5) & 0x80008000)
//   c3 m 11-13 s 14 | c7 m 27-29 s 30 -> o3 = ((t >> 5) & 0x01C001C0) | ((t << 1) & 0x80008000)
// = e2m1 x 2^-126 as bf16 (magnitude to bf16 bits 6-8, sign to 15; e = 0, m = 1 is the subnormal 2^-127): 11 ops.
DINL void dq8r(unsigned t, __nv_bfloat162 d[4]) {
    d[0] = u32b2(t & 0x81C081C0u);
    d[1] = u32b2((t << 6) & 0x81C081C0u);
    d[2] = u32b2(((t << 3) & 0x01C001C0u) | ((t << 5) & 0x80008000u));
    d[3] = u32b2(((t >> 5) & 0x01C001C0u) | ((t << 1) & 0x80008000u));
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
constexpr int WSB = 5120;                // merged W(4096) + bf16 scale pairs (1024) per (wt, kt)
constexpr int SB  = 37888;               // stage stride: X 32768 + W|sf 5120 (a whole number of 1024 B atoms)
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
    for (int cls = 3; cls >= 0; cls--) {
        for (int base = 0; base < npm; base += 1024) {
            const int pm = base + tid;
            const bool live_blk = pm < npm && (unsigned)block_experts[pm] < (unsigned)nle;
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

// ------------------------------------------------------------------ small-T routing front end
// One CTA in place of topk_ids -> local ids (expert map), moe_align_block_size and pg_tiles, for numel <= RT_MAX
// routes (verify / decode): local expert e (e < nl <= RT_E: 32 home slots + replica slots) gets ceil(count_e / 256)
// consecutive blocks, experts ascending, and no block exists for non-local routes; a block's 256 sorted_ids slots hold
// its routes in route order, then the numel sentinel. block_experts / live are what pg_tiles writes for the same counts
// (live: widest class first, block ascending inside a class), so gather / gg1 / gg2 run unchanged; every route's row
// is computed by the same code as on the moe_align path, only the block numbering differs. local_ids[r] = e or -1 (the
// finalize's slot ids). No serial loops: a route's rank among same-expert routes of its warp comes from
// __match_any_sync, the per-expert bases over the 64 warp groups from one warp-wide scan per expert (warp w: experts w
// and w + 32), the block / live lists from one warp's ballots over two 32-expert halves.
constexpr int RT_MAX = 2048, RT_E = 64, RT_NB = RT_E + RT_MAX / XNT;
// a block's width code from its valid routes v (pg_tiles' rule: the smallest listed width covering v) and the live
// list's classes, widest first; the narrow build (_cuda_src_pgn) swaps both for its 16-token codes
constexpr int RT_NCLS = 4, RT_FULL = 3;
DINL int route_code(int v, int wmask) {
    int code = RT_FULL;
    if ((wmask & 4) && v <= 192) code = 2;
    if ((wmask & 2) && v <= 128) code = 1;
    if ((wmask & 1) && v <= 64) code = 0;
    return code;
}
DINL int route_cls(int i) { return RT_FULL - i; }
__global__ void __launch_bounds__(1024) pg_route(const int* __restrict__ topk_ids, const int* __restrict__ emap,
                                                 int n_global, int numel, int wmask, int nl,
                                                 int* __restrict__ local_ids, int* __restrict__ sorted_ids,
                                                 int* __restrict__ block_experts, int* __restrict__ live) {
    __shared__ int wc[RT_MAX / 32][RT_E];   // per (32-route group g = k * 32 + warp, expert): count -> base
    __shared__ int cnt[RT_E], bstart[RT_E], bexp[RT_NB], nblk;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const unsigned lt = (1u << lane) - 1u;
    wc[w][lane] = 0;                         // rows w and 32 + w are written by this warp only
    wc[w][32 + lane] = 0;
    wc[32 + w][lane] = 0;
    wc[32 + w][32 + lane] = 0;
    __syncwarp();
    pdl_wait();
    pdl_trigger();
    int er[2], rk[2];
#pragma unroll
    for (int k = 0; k < 2; k++) {
        const int r = k * 1024 + tid;
        int e = -1;
        if (r < numel) {
            const int g = topk_ids[r];
            if ((unsigned)g < (unsigned)n_global) e = emap ? emap[g] : g;
            if ((unsigned)e >= (unsigned)nl) e = -1;
            local_ids[r] = e;
        }
        const unsigned peers = __match_any_sync(0xFFFFFFFFu, e);
        er[k] = e;
        rk[k] = __popc(peers & lt);
        if (e >= 0 && rk[k] == 0) wc[k * 32 + w][e] = __popc(peers);
    }
    __syncthreads();
#pragma unroll
    for (int x = w; x < RT_E; x += 32) {   // warp w = experts w, w + 32: exclusive scan of the 64 group counts
        const int c0 = wc[2 * lane][x], c1 = wc[2 * lane + 1][x];
        int inc = c0 + c1;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xFFFFFFFFu, inc, o);
            if (lane >= o) inc += v;
        }
        const int ex = inc - c0 - c1;
        wc[2 * lane][x] = ex;
        wc[2 * lane + 1][x] = ex + c0;
        if (lane == 31) cnt[x] = inc;
    }
    __syncthreads();
    if (w == 0) {   // lane l = experts l (half 0) and 32 + l (half 1): blocks, block list, live list
        int nbx[2], code[2], b0[2], carry = 0;
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
            bstart[x] = b0[hf];
            for (int j = 0; j < nbx[hf]; j++) { bexp[b0[hf] + j] = x; block_experts[b0[hf] + j] = x; }
            // the expert's last block holds v = c - XNT (nbx - 1) routes, the others XNT (the full class)
            const int vl = nbx[hf] > 0 ? c - (nbx[hf] - 1) * XNT : 0;
            code[hf] = route_code(vl, wmask);
        }
        const int nb = carry;
        int base = 0;
        for (int ci = 0; ci < RT_NCLS; ci++) {
            const int cls = route_cls(ci);
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                // blocks of class cls this expert contributes: its full blocks (RT_FULL) and / or its last block
                const int n = nbx[hf];
                const int k_c = n == 0 ? 0 : (cls == RT_FULL ? n - 1 : 0) + (code[hf] == cls ? 1 : 0);
                int inc = k_c;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const int v = __shfl_up_sync(0xFFFFFFFFu, inc, o);
                    if (lane >= o) inc += v;
                }
                int pos = base + inc - k_c;
                if (cls == RT_FULL) for (int j = 0; j < n - 1; j++) live[1 + pos++] = (b0[hf] + j) | (RT_FULL << 20);
                if (n > 0 && code[hf] == cls) live[1 + pos] = (b0[hf] + n - 1) | (code[hf] << 20);
                base += __shfl_sync(0xFFFFFFFFu, inc, 31);
            }
        }
        if (lane == 0) { live[0] = nb; nblk = nb; }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 2; k++) {
        const int e = er[k];
        if (e >= 0) sorted_ids[bstart[e] * XNT + wc[k * 32 + w][e] + rk[k]] = k * 1024 + tid;
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
        const uint4 sfp = *reinterpret_cast<const uint4*>(stp + WOFF + 4096 + c.sfo);   // bf16 (r0, r1) x 4
        const unsigned* sfq = reinterpret_cast<const unsigned*>(&sfp);
        const unsigned* wq = reinterpret_cast<const unsigned*>(&w);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            __nv_bfloat162 d[4];
            dq8r(wq[q], d);
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
    for (int t = w; t < nv; t += 8) {
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


// ------------------------------------------------------------------ large-T routing front end
// Local-slot align for large calls (the prefill bank's sk_route align, plt_c3 / plt_c4b skr::align_count /
// align_scan / align_scatter): count the routes into nl <= AL_MAX local-slot counters -> one-CTA scan (ranges padded to
// BM, block -> slot map, padding sentinels = numel) -> CTA-aggregated scatter. Same layout as moe_align_block_size over
// the nl local slots, minus the 256-expert sort; the order of routes inside a slot's range is free (pg computes every
// route's rows independently). ids are global expert ids mapped by emap (global -> local slot, -1 remote), or with no
// emap already local slot ids; either way ids outside [0, n_global) and slots outside [0, nl) count as remote, and
// local_ids[r] receives the slot or -1 (the finalize's slot ids).
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
// tab[e] for the 256 experts: bits 0-7 the expert's slot on this rank (255: not held here), 8-10 its home rank,
// bit 11 replicated, 12-14 the replica's rank; tab[256 + j] = the j-th replicated expert in balancing order. Every
// rank takes the same decision from the same top-k ids (only the slot field differs per rank): load[r] = distinct
// active non-replicated experts homed on r; then each active replicated expert, in order, goes to whichever of its
// two holders has the lower load (tie: home) and counts there. out[i] = route i's local slot if this rank computes
// it, else -1 (the static map's convention); emap_out[e] (if given) = the same decision per expert, the forward's
// global -> local map for pg_route.
template <typename IdT>
__global__ void __launch_bounds__(256) pg_rep_assign(const IdT* __restrict__ ids, int n_routes,
                                                     const int* __restrict__ tab, int n_rep, int me,
                                                     int* __restrict__ out, int* __restrict__ emap_out) {
    __shared__ int tabs[256], ord[256], act[256], owner[256], load[8];
    const int t = threadIdx.x;
    const int v = tab[t];
    tabs[t] = v;
    if (t < n_rep) ord[t] = tab[256 + t];
    act[t] = 0;
    owner[t] = (v >> 8) & 7;
    if (t < 8) load[t] = 0;
    __syncthreads();
    for (int i = t; i < n_routes; i += 256) {
        const int e = (int)ids[i];
        if ((unsigned)e < 256u) act[e] = 1;
    }
    __syncthreads();
    if (act[t] && !((v >> 11) & 1)) atomicAdd(&load[(v >> 8) & 7], 1);
    __syncthreads();
    if (t == 0) {
        unsigned long long L = 0;   // 8 bits per rank (a load is at most 32 + n_rep < 256)
        for (int r = 0; r < 8; r++) L |= (unsigned long long)load[r] << (8 * r);
        for (int j = 0; j < n_rep; j++) {
            const int e = ord[j], ve = tabs[e];
            const int h = (ve >> 8) & 7, r = (ve >> 12) & 7;
            const int pick = ((L >> (8 * r)) & 0xFF) < ((L >> (8 * h)) & 0xFF) ? r : h;
            if (act[e]) { owner[e] = pick; L += 1ull << (8 * pick); }
        }
    }
    __syncthreads();
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
// At a long prefill every expert is active and a rank's pg time follows its routed tokens, so the routes of a replicated
// expert are split between its two holders instead of handed to one. Every rank takes the same decision from the same
// top-k ids (integer arithmetic only): c[e] = routes of expert e; load[r] = routes of the non-replicated experts homed on
// r; then the replicated experts by route count (descending, ties: lower id), each in turn: x = clamp((load[home] +
// c[e] - load[replica]) / 2, 0, c[e]); the replica takes e's routes of the tokens before cut[e], the end of the first
// sb-token block at which e's cumulative count reaches x (0 if x is 0), the home the rest, and both loads grow by their
// parts. pg_rep_count (one CTA per sb-token block) counts the routes of every expert (global atomics into cnt, zeroed by
// the caller) and of each replicated expert per block (bc[j][b]: the block is the CTA's own, so no atomics); pg_rep_cut
// (one CTA) takes the decision and writes cut[256] (0 = the home takes every route); pg_rep_split maps route i (token
// i / k) to its local slot on this rank, or -1.
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
        (int)num_pid_m, (int)bm, (int)wmask, (int)nle, live.data_ptr<int>());
}

// out: [numel] per-route slots, emap: [256] per-expert map; either may be empty (not written)
void rep_assign(torch::Tensor ids, torch::Tensor tab, torch::Tensor out, torch::Tensor emap, int64_t me) {
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
                                                             tab.data_ptr<int>(), n_rep, (int)me, o, m);
    else
        pgk::pg_rep_assign<int><<<1, 256, 0, stream>>>(ids.data_ptr<int>(), (int)ids.numel(), tab.data_ptr<int>(),
                                                       n_rep, (int)me, o, m);
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
    pg_launch(pgk::pg_gather<3, 8>, dim3(pgk::NKT / 8, npm), dim3(256), 0, pdl != 0,
              reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<char*>(xg.data_ptr()),
              sorted_ids.data_ptr<int>(), live.data_ptr<int>(), (int)numel);
}

// small-T front end: topk_ids [T, 8] int32 (global ids), emap [n_global] int32 global -> local (-1 = remote) or
// an empty tensor (ids already local); outputs local_ids [T * 8], sorted_ids [nb_max * 256], block_experts
// [nb_max], live [nb_max + 1] with nb_max = 32 + ceil(numel / 256).
void route(torch::Tensor topk_ids, torch::Tensor emap, torch::Tensor local_ids, torch::Tensor sorted_ids,
           torch::Tensor block_experts, torch::Tensor live, int64_t numel, int64_t n_global, int64_t nl, int64_t pdl) {
    TORCH_CHECK(numel <= pgk::RT_MAX && topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous() &&
                nl >= 1 && nl <= pgk::RT_E, "pg route: int32 ids, at most 2048 routes, 1..64 local slots");
    TORCH_CHECK(sorted_ids.numel() >= (nl + (numel + pgk::XNT - 1) / pgk::XNT) * pgk::XNT &&
                live.numel() >= nl + (numel + pgk::XNT - 1) / pgk::XNT + 1, "pg route: output sizes");
    const int* em = emap.numel() > 0 ? emap.data_ptr<int>() : nullptr;
    pg_launch(pgk::pg_route, dim3(1), dim3(1024), 0, pdl != 0, topk_ids.data_ptr<int>(), em, (int)n_global,
              (int)numel, 3 /* = _WMASK */, (int)nl, local_ids.data_ptr<int>(), sorted_ids.data_ptr<int>(),
              block_experts.data_ptr<int>(), live.data_ptr<int>());
}

// Large-T front end: ids [numel] int32 / int64 global ids (emap [n_global] int32, global -> local slot) or local slot ids
// (emap empty); writes local_ids and moe_align_block_size's outputs over nl slots; cnt [nl] must be zero on entry.
void align(torch::Tensor ids, torch::Tensor emap, torch::Tensor local_ids, torch::Tensor cnt, torch::Tensor off,
           torch::Tensor cur, torch::Tensor sorted_ids, torch::Tensor block_experts, torch::Tensor n_padded,
           int64_t numel, int64_t BM, int64_t nl, int64_t n_global) {
    TORCH_CHECK(nl >= 1 && nl <= pgk::AL_MAX && cnt.numel() >= nl, "pg align: 1..64 local slots");
    TORCH_CHECK(ids.is_contiguous() && local_ids.numel() >= numel && local_ids.scalar_type() == at::kInt,
                "pg align: ids / local_ids");
    TORCH_CHECK(emap.numel() == 0 || (emap.scalar_type() == at::kInt && emap.numel() >= n_global), "pg align: emap");
    auto st = at::cuda::getCurrentCUDAStream();
    const int* em = emap.numel() > 0 ? emap.data_ptr<int>() : nullptr;
    const int nblocks = (int)block_experts.numel();
    const int chunk = 4096;
    const int nscat = (int)((numel + chunk - 1) / chunk);
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
                 torch::Tensor live, int64_t numel, int64_t pdl) {
    TORCH_CHECK(x.size(1) == pgk::NKT * 64 && numel == x.size(0) * 8 && x.is_contiguous(), "pg gather: K 4096, topk 8");
    TORCH_CHECK(gamma.numel() == pgk::NKT * 64 && gamma.scalar_type() == at::kBFloat16 && gamma.is_contiguous(),
                "pg gather_norm: bf16 [4096] weight");
    const int npm = live.size(0) - 1;
    pg_launch(pgk::pg_gather_norm<3>, dim3(pgk::NKT / 8, npm), dim3(256), 0, pdl != 0,
              reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()), (float)eps,
              reinterpret_cast<char*>(xg.data_ptr()), sorted_ids.data_ptr<int>(), live.data_ptr<int>(), (int)numel);
}

template <int GATED, int NST, int MODE, int WSET, int PAIR64 = 1, int SPL = 0>
static void launch(const torch::Tensor& xg, const torch::Tensor& wsf, const float* gs, void* out,
                   const int* sorted_ids, const torch::Tensor& block_experts, const torch::Tensor& live,
                   int numel, int grid, int snake, bool pdl) {
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
              gs, out, sorted_ids, block_experts.data_ptr<int>(), live.data_ptr<int>(), NWT, numel, snake);
}

// snake item order for small calls (few items per CTA): tokens = numel / 8 <= pgk_snake_max_t
static int pgk_snake_max_t = 4096;
void set_snake_max_t(int64_t t) { pgk_snake_max_t = (int)t; }

// cfg 350 = the bank's shipped configuration (widths 64 / 128 / 256, 5 stages), the only one built here.
void gg1(torch::Tensor xg, torch::Tensor wsf, torch::Tensor gs, torch::Tensor inter,
         torch::Tensor block_experts, torch::Tensor live, int64_t numel, int64_t cfg, int64_t grid, int64_t pdl) {
    TORCH_CHECK(cfg == 350, "pg: only cfg 350 is built");
    const int snake = (numel / 8 <= pgk_snake_max_t) ? 1 : 0;
    launch<1, 5, 0, 3>(xg, wsf, gs.data_ptr<float>(), inter.data_ptr(), nullptr, block_experts, live, 0, grid,
                       snake, pdl != 0);
}

void gg2(torch::Tensor xg, torch::Tensor wsf, torch::Tensor y, torch::Tensor sorted_ids,
         torch::Tensor block_experts, torch::Tensor live, int64_t numel, int64_t cfg, int64_t grid, int64_t pdl) {
    TORCH_CHECK(cfg == 350, "pg: only cfg 350 is built");
    const int snake = (numel / 8 <= pgk_snake_max_t) ? 1 : 0;
    launch<0, 5, 0, 3>(xg, wsf, nullptr, y.data_ptr(), sorted_ids.data_ptr<int>(), block_experts, live,
                       (int)numel, grid, snake, pdl != 0);
}
"""

_CUDA_SRC = _CUDA_SRC.replace("@@WGMMA@@", "".join(_wgmma_asm(n) for n in (256, 192, 128, 64)))
