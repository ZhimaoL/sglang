"""CUDA source of the st small-T MoE pair (okc_patch/st_moe.py; developed in muse-opt serving/kernels/smallt-moe).

st_gemm<1>: gate/up over the call's active local experts, silu*mul into a bf16 intermediate; st_gemm<2>: down into
the per-route bf16 rows. Both read pg_moe.pack_wsf's blocks (no other weight copy) and decode them with pg's dq8r and
the pack's block scales into mma.sync m16n8k16 A fragments (one warp's 16 rows of a pg block are exactly that fragment).
SCALE_BF16 selects the pack's scale format (one constant, matching the tree's pack_wsf): 1 = bf16 (r0, r1) pairs,
16 B per row pair and k-tile (WSB 5120, cand6c / cand7); 0 = 8-bit codes, 8 B (WSB 4608, cand7a). Local experts: up
to ELOC (32 home slots + hot-expert replica slots), the count from wsf13.
"""

SCALE_BF16 = 1

_CUDA_SRC = r"""
#define ST_SF16 @@SF16@@
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace stk {
#define DINL __device__ __forceinline__

constexpr int H = 4096;
constexpr int NKT = 64;           // k-tiles of 64 (K = 4096 in both GEMMs)
// pg block per (expert, w-tile, k-tile): 4096 B codes + the 64 row pairs' scales (SFB bytes each)
constexpr int SFB = ST_SF16 ? 16 : 8;
constexpr int WSB = 4096 + 64 * SFB;
constexpr int NW = 16;            // warps per CTA; warp w owns k-tiles KPW*w .. KPW*w + KPW - 1 of every slab
constexpr int KPW = NKT / NW;     // 4
constexpr int NTHR = NW * 32;
constexpr int ELOC = 64;          // local slots at most (32 home + replicas); the call's count nl comes from wsf13
constexpr int NUMEL_MAX = 256;    // T <= 32, top-8
constexpr int VAMAX = ELOC + NUMEL_MAX / 8;   // passes (<= 8 tokens each): <= nl + numel / 8
constexpr int R = 4;              // partial ring slots
constexpr int D = 8;              // weight k-tiles in flight per warp (2 slabs)

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
#if ST_SF16
typedef uint4 sf_t;
DINL sf_t lds8(const char* p) {          // bf16 scale pairs (the 4 lanes of a row pair read the same 16 B)
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
#else
typedef uint2 sf_t;
DINL sf_t lds8(const char* p) {          // scale codes (the 4 lanes of a row pair read the same 8 B)
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
#endif
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

// pg's dq8r (pg_moe.pack_wsf bit placement): one u32 of codes -> the 4 A registers as e2m1 x 2^-126 in bf16
DINL void dq8r(unsigned t, __nv_bfloat162 d[4]) {
    d[0] = u32b2(t & 0x81C081C0u);
    d[1] = u32b2((t << 6) & 0x81C081C0u);
    d[2] = u32b2(((t << 3) & 0x01C001C0u) | ((t << 5) & 0x80008000u));
    d[3] = u32b2(((t >> 5) & 0x01C001C0u) | ((t << 1) & 0x80008000u));
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
#if ST_SF16
    const unsigned sfw[4] = {sfp.x, sfp.y, sfp.z, sfp.w};
#else
    const unsigned sfw[2] = {sfp.x, sfp.y};
#endif
#pragma unroll
    for (int q = 0; q < 4; q++) {
        __nv_bfloat162 d[4];
        dq8r(wq[q], d);
#if ST_SF16
        // bf16 scale pair (row r0, row r0 + 8) of k16 step q, as pg's kstep reads it
        const __nv_bfloat162 sf2 = u32b2(sfw[q]);
#else
        // scale code pair (row r0, row r0 + 8) of k16 step q -> bf16 bits code * 16 + 0x75F0 (pg_moe.pack_wsf)
        const __nv_bfloat162 sf2 = u32b2(__byte_perm(sfw[q >> 1], 0u, (q & 1) ? 0x4342u : 0x4140u) * 16u + 0x75F075F0u);
#endif
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
    int va_rt[VAMAX][8];      // route index (token * 8 + k) of the pass's tokens, route order
    float gs[ELOC];
};

// Routing table of the call from local_ids (-1 = not local), identical in every CTA of both kernels: an expert's
// routes in route order, split in passes of 8 (ascending expert, then pass). No atomics.
DINL void build_tab(const int* __restrict__ ids, int numel, int nl, const float* __restrict__ gs, Tab& tb,
                    int (*wcnt)[ELOC], int* vbase) {
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    for (int i = tid; i < 8 * ELOC; i += NTHR) (&wcnt[0][0])[i] = 0;
    int id = -1;
    if (tid < NUMEL_MAX && tid < numel) {
        id = ids[tid];
        if ((unsigned)id >= (unsigned)nl) id = -1;
    }
    if (gs != nullptr && tid >= NTHR - ELOC && tid - (NTHR - ELOC) < nl)
        tb.gs[tid - (NTHR - ELOC)] = gs[tid - (NTHR - ELOC)];
    __syncthreads();
    int rank = 0;
    if (w < NUMEL_MAX / 32) {
        const unsigned m = __match_any_sync(0xFFFFFFFFu, id);
        rank = __popc(m & ((1u << lane) - 1u));
        if (id >= 0 && rank == 0) wcnt[w][id] = __popc(m);
    }
    __syncthreads();
    if (w == 0) {                                        // warp 0: per-expert totals, pass counts, scan
        int carry = 0;                                   // lane l: experts l, then 32 + l
#pragma unroll
        for (int hf = 0; hf < ELOC / 32; hf++) {
            const int e = 32 * hf + lane;
            int s = 0;
#pragma unroll
            for (int ww = 0; ww < 8; ww++) { const int c = wcnt[ww][e]; wcnt[ww][e] = s; s += c; }
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
    if (id >= 0) {
        const int k = wcnt[w][id] + rank;
        tb.va_rt[vbase[id] + (k >> 3)][k & 7] = tid;
    }
    __syncthreads();
}

// KIND 1: gate/up (W13, 64 w-tiles; B = x rows of the tokens) -> h[va * 8 + n][f] = silu(gate) * up (pg gg1 math)
// KIND 2: down (W2, 32 w-tiles; B = h rows) -> y[route][row] = acc * 2^8 (pg gg2 math)
// Work: slabs = 16 weight rows (one warp's piece of a pg block) x full K of the passes, an equal contiguous range per
// CTA; inside the CTA the 16 warps split each slab's K; their partials meet in a 4-slot smem ring and one warp per slab
// (slab % 16) sums them in warp order (deterministic) one slab later and runs the epilogue.
template <int KIND>
__global__ void __launch_bounds__(NTHR, 1) st_gemm(const __nv_bfloat16* __restrict__ xin, const char* __restrict__ wsf,
                                                   const float* __restrict__ gs, __nv_bfloat16* __restrict__ out,
                                                   const int* __restrict__ ids, int numel, int nl, int pdl) {
    constexpr int NWT = KIND == 1 ? 64 : 32;
    constexpr int LSPE = KIND == 1 ? 9 : 8;            // log2 slabs per pass (NWT x 8 slabs)
    __shared__ Tab tb;
    __shared__ __align__(16) float4 part[R][NW][32];   // per-warp slab partials
    __shared__ __align__(16) unsigned short scr[NW][128];
    __shared__ __align__(8) unsigned long long bars[2 * R];   // full[R] | empty[R]
    __shared__ int s_wcnt[8][ELOC];
    __shared__ int s_vbase[ELOC];

    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    if (KIND == 1 && pdl) pdl_trigger();               // KIND 2 CTAs only fit on SMs this kernel has left
    if (tid < R) {
        mbar_init(smem_u32(&bars[tid]), NTHR);
        mbar_init(smem_u32(&bars[R + tid]), 32);
    }
    build_tab(ids, numel, nl, KIND == 1 ? gs : nullptr, tb, s_wcnt, s_vbase);   // ends with __syncthreads

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

    uint4 wr[D];
    sf_t sr[D];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const char* p = slab_ptr(h);
        const int so = sofs_of(h);
#pragma unroll
        for (int i = 0; i < KPW; i++) {
            wr[4 * h + i] = ldw(p + (size_t)i * WSB + lofs);
            sr[4 * h + i] = lds8(p + (size_t)i * WSB + so);
        }
    }
    if (KIND == 2 && pdl) pdl_wait();                  // the intermediate is complete from here on

    unsigned b[KPW][4][2];
    int cva = -1;
    auto load_b = [&](int va) {
        const bool live = g < tb.va_n[va];             // padded token columns stay zero (their outputs are dropped)
        const __nv_bfloat16* row = KIND == 1 ? xin + (size_t)(tb.va_rt[va][live ? g : 0] >> 3) * H
                                             : xin + (size_t)(va * 8 + g) * H;
        row += KPW * w * 64 + 2 * t;
#pragma unroll
        for (int i = 0; i < KPW; i++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                if (KIND == 1) {
                    b[i][q][0] = live ? ldb32(row + i * 64 + 16 * q) : 0u;
                    b[i][q][1] = live ? ldb32(row + i * 64 + 16 * q + 8) : 0u;
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
    for (int j0 = 0; j0 < nslab; j0 += 2) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int jl = j0 + h;
            if (jl < nslab) {
                const int va = (s0 + jl) >> LSPE;
                if (va != cva) { cva = va; load_b(va); }
                const char* p = slab_ptr(jl + 2);
                const int so = sofs_of(jl + 2);
#pragma unroll
                for (int i = 0; i < KPW; i++) {
                    ktile(acc, wr[4 * h + i], sr[4 * h + i], b[i]);
                    wr[4 * h + i] = ldw(p + (size_t)i * WSB + lofs);     // unconditional refill (clamped slab)
                    sr[4 * h + i] = lds8(p + (size_t)i * WSB + so);
                }
                const int slot = jl % R;
                if (jl >= R) mbar_wait(emptyb + slot * 8, ((jl / R) - 1) & 1);
                part[slot][w][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
                mbar_arrive(fullb + slot * 8);
                acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
                if (jl >= 1 && (jl - 1) % NW == w) reduce(jl - 1);
            }
        }
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
void st_moe(torch::Tensor x, torch::Tensor ids, torch::Tensor wsf13, torch::Tensor gs13, torch::Tensor wsf2,
            torch::Tensor h, torch::Tensor y, int64_t pdl) {
    const int numel = (int)ids.numel();
    TORCH_CHECK(numel <= stk::NUMEL_MAX && numel == x.size(0) * 8 && x.size(1) == stk::H, "st: T <= 32, top-8, H 4096");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.is_contiguous() && x.is_contiguous(), "st: int32 ids, contiguous x");
    const int nl = (int)wsf13.size(0);
    TORCH_CHECK(nl >= 1 && nl <= stk::ELOC && wsf13.size(1) == 64 && wsf13.size(2) == stk::NKT &&
                wsf13.size(3) == stk::WSB, "st: wsf13 layout (local slots <= 64, block bytes = this build's WSB)");
    TORCH_CHECK(wsf2.size(0) == nl && wsf2.size(1) == 32 && wsf2.size(2) == stk::NKT && wsf2.size(3) == stk::WSB,
                "st: wsf2 layout");
    TORCH_CHECK(h.size(0) >= (nl + numel / 8) * 8 && gs13.numel() == nl, "st: h rows / gs13");
    const int G = st_num_sms();
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto* hp = reinterpret_cast<__nv_bfloat16*>(h.data_ptr());
    auto* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
    const int* ip = ids.data_ptr<int>();
    stk::st_gemm<1><<<G, stk::NTHR, 0, st>>>(xp, reinterpret_cast<const char*>(wsf13.data_ptr()),
                                            gs13.data_ptr<float>(), hp, ip, numel, nl, (int)pdl);
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
                                          numel, nl, 1));
    } else {
        stk::st_gemm<2><<<G, stk::NTHR, 0, st>>>(hp, reinterpret_cast<const char*>(wsf2.data_ptr()), nullptr, yp, ip,
                                                numel, nl, 0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
"""

_CUDA_SRC = _CUDA_SRC.replace("@@SF16@@", str(SCALE_BF16))
