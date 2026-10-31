"""CUDA source of OKC_SW_WSC2 (okc_patch/wsc2.py): KQ3 (qkv_gate at m 17-32) and KO2b (o_proj at m 1-32), from muse-opt
serving/kernels/megakernel/mkw1b_src.py (md5 906b1eb9; LOG M5.3 / M5.4) with the probe-only kernels left out.

KQ3: 108 CTAs = 36 row tiles x a cluster of 3 (split-K 22 / 21 / 21 k-stages of 128), an 8-stage ring of [16 KB W | 8 KB
x] (205,960 B, so a CTA fits beside an all-gather-norm CTA), wgmma m64n32k16 (W rows as A, tokens as B); the producer issues the W atoms of the first 9 stages before
griddepcontrol.wait (one expect_tx of W + x per stage) and their x atoms after it (14.2 MB staged); fp32 partials reduced by push-mode
DSMEM (warp w's rows owned by rank (w == 3 ? 0 : w); own + lower + higher rank, fixed order), bf16 out.
KO2b: 128 CTAs, each loads its 64-row weight slice (16 atoms of 64 k, one barrier each) before griddepcontrol.wait, then
streams x through a 3-slot ring of 64-k atoms. 135-145 KB, so a CTA fits beside a VP2 CTA (81,984 B).
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <map>
#include <tuple>

#define DINL __device__ __forceinline__

namespace wsc2 {

constexpr int QN = 2304, QK = 8192, ON = 8192, OK = 1024;
constexpr int QTILES = QN / 64, OTILES = ON / 64;
constexpr int KT = 128, QKT_ALL = QK / KT;
constexpr int WATOM = 64 * 128, STAGE_W = 2 * WATOM, OW_ATOMS = OK / 64, OW_BYTES = OW_ATOMS * WATOM;
constexpr int RINGQ = 224 * 1024, NXS = 3;
constexpr int QS3 = 3, QCTAS3 = QTILES * QS3;

DINL unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
DINL void mbar_init(unsigned bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
DINL void mbar_arrive(unsigned bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
DINL void mbar_expect_tx(unsigned bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
DINL void mbar_wait(unsigned bar, unsigned phase) {
    asm volatile("{\n.reg .pred p;\nWAIT_%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra WAIT_%=;\n}\n" :: "r"(bar), "r"(phase) : "memory");
}
DINL void tma_2d(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"((uint64_t)map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
DINL void wg_fence()  { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DINL void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
DINL void wg_wait0()  { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
DINL void wg_wait1()  { asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory"); }
DINL uint64_t desc_sw128(unsigned saddr) {
    return (uint64_t)((saddr >> 4) & 0x3FFF) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
DINL void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

DINL void wgmma_n8(float* acc, uint64_t da, uint64_t db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, %4, %5, p, 1, 1, 0, 0;\n}\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]) : "l"(da), "l"(db), "r"(1));
}
DINL void wgmma_n16(float* acc, uint64_t da, uint64_t db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0;\n}\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
        : "l"(da), "l"(db), "r"(1));
}
DINL void wgmma_n32(float* acc, uint64_t da, uint64_t db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 0, 0;\n}\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]),
          "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]) : "l"(da), "l"(db), "r"(1));
}
template <int NT> DINL void wgmma_nt(float* acc, uint64_t da, uint64_t db) {
    if constexpr (NT == 8) wgmma_n8(acc, da, db);
    else if constexpr (NT == 16) wgmma_n16(acc, da, db);
    else wgmma_n32(acc, da, db);
}

// 8 stages, not the 9 that fit in 224 KB: a KQ3 CTA must sit beside an all-gather-norm CTA (1,232 B static) on one SM
// (233,472 B, 1 KB reserved per CTA), and 9 stages miss by 360 B.
template <int NT> struct LayQ3 {
    static constexpr int XATOM = NT * 128;
    static constexpr int SB = STAGE_W + 2 * XATOM;
    static constexpr int RX = 2 * 2 * 16 * NT * 4;            // [owned slot][sender][16 rows][NT] fp32
    static constexpr int NSTQ = (RINGQ - RX) / SB < 8 ? (RINGQ - RX) / SB : 8;
    static constexpr int RING_BYTES = NSTQ * SB + RX;
    static constexpr int SMEM = RING_BYTES + 8 * (2 * NSTQ + 1) + 1024;
};
template <int NT> struct LayO2b {
    static constexpr int XATOM = NT * 128;
    static constexpr int SMEM = OW_BYTES + NXS * XATOM + 8 * (OW_ATOMS + 2 * NXS) + 1024;
};

// bf16 out[t][col0 + r] for the wgmma m64 accumulator fragment of warp w (rows 16 w + g, + 8; tokens 8 j + 2 c, + 1)
template <int NT>
DINL void store_frag(const float* acc, __nv_bfloat16* out, int ld, int col0, int warp, int lane, int m) {
    const int g = lane >> 2, c = lane & 3;
#pragma unroll
    for (int j = 0; j < NT / 8; j++) {
        const int r = 16 * warp + g, t = 8 * j + 2 * c;
        if (t < m) { out[(size_t)t * ld + col0 + r] = __float2bfloat16_rn(acc[4 * j]); out[(size_t)t * ld + col0 + r + 8] = __float2bfloat16_rn(acc[4 * j + 2]); }
        if (t + 1 < m) { out[(size_t)(t + 1) * ld + col0 + r] = __float2bfloat16_rn(acc[4 * j + 1]); out[(size_t)(t + 1) * ld + col0 + r + 8] = __float2bfloat16_rn(acc[4 * j + 3]); }
    }
}

DINL void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
DINL void cluster_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
DINL void st_dsmem2(unsigned local_addr, unsigned rank, float a, float b) {
    unsigned remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
    asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};" :: "r"(remote), "f"(a), "f"(b) : "memory");
}
DINL void mbar_arrive_remote_release(unsigned local_bar, unsigned rank) {
    unsigned remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_bar), "r"(rank));
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
}
DINL void mbar_wait_acq_cluster(unsigned bar, unsigned phase) {
    asm volatile("{\n.reg .pred p;\nWAIT_%=:\n"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra WAIT_%=;\n}\n" :: "r"(bar), "r"(phase) : "memory");
}

// ---------------- KQ3: qkv, 108 CTAs, cluster 3 ----------------
template <int NT, bool WFIRST>
__global__ void __launch_bounds__(160, 1) kq3_k(const __grid_constant__ CUtensorMap wqm, const __grid_constant__ CUtensorMap xqm,
                                              __nv_bfloat16* __restrict__ qout, int m)
{
    using L = LayQ3<NT>;
    constexpr int NSTQ = L::NSTQ;
    extern __shared__ char smem_raw[];
    const unsigned sraw = smem_u32(smem_raw);
    const unsigned sbase = (sraw + 1023) & ~1023u;
    char* sgen = smem_raw + (sbase - sraw);
    const unsigned ring = sbase, rx = sbase + NSTQ * L::SB;
    const float* rxp = reinterpret_cast<const float*>(sgen + NSTQ * L::SB);
    const unsigned qfull = sbase + L::RING_BYTES, qempty = qfull + 8 * NSTQ, rxbar = qempty + 8 * NSTQ;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int b = blockIdx.x, qrt = b / QS3, qrank = b % QS3;
    if (tid == 0) {
        for (int i = 0; i < NSTQ; i++) { mbar_init(qfull + 8 * i, 1); mbar_init(qempty + 8 * i, 4); }
        mbar_init(rxbar, (qrank == 0 ? 2 : 1) * 2 * 32);       // owned warps x 2 senders x 32 lanes
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    cluster_arrive_relaxed();
    const int ks = (qrank * QKT_ALL) / QS3, ke = ((qrank + 1) * QKT_ALL) / QS3;
    const int u0 = qrt * QKT_ALL + ks, n = ke - ks;
    if (warp == 4) {
        if (lane == 0) {
            const int npre = n < NSTQ ? n : NSTQ;
            if constexpr (WFIRST) {
                for (int i = 0; i < npre; i++) {
                    const int u = u0 + i;
                    const unsigned st = ring + i * L::SB, full = qfull + 8 * i;
                    mbar_expect_tx(full, STAGE_W + 2 * L::XATOM);
                    for (int a = 0; a < 2; a++) tma_2d(st + a * WATOM, &wqm, (u % QKT_ALL) * KT + 64 * a, (u / QKT_ALL) * 64, full);
                }
            }
            pdl_wait();
            for (int i = 0; i < n; i++) {
                const int u = u0 + i, slot = i % NSTQ;
                const unsigned st = ring + slot * L::SB, full = qfull + 8 * slot;
                const int k = (u % QKT_ALL) * KT;
                if (i >= NSTQ) mbar_wait(qempty + 8 * slot, ((i / NSTQ) - 1) & 1);
                const bool wdone = WFIRST && i < npre;
                if (!wdone) mbar_expect_tx(full, STAGE_W + 2 * L::XATOM);
                for (int a = 0; a < 2; a++) {
                    if (!wdone) tma_2d(st + a * WATOM, &wqm, k + 64 * a, (u / QKT_ALL) * 64, full);
                    tma_2d(st + STAGE_W + a * L::XATOM, &xqm, k + 64 * a, 0, full);
                }
            }
        }
    } else {
        pdl_wait();
        float acc[NT / 2];
#pragma unroll
        for (int i = 0; i < NT / 2; i++) acc[i] = 0.f;
        for (int i = 0; i < n; i++) {
            const int slot = i % NSTQ;
            mbar_wait(qfull + 8 * slot, (i / NSTQ) & 1);
            const unsigned st = ring + slot * L::SB;
            wg_fence();
#pragma unroll
            for (int a = 0; a < 2; a++)
#pragma unroll
                for (int s4 = 0; s4 < 4; s4++)
                    wgmma_nt<NT>(acc, desc_sw128(st + a * WATOM + 32 * s4), desc_sw128(st + STAGE_W + a * L::XATOM + 32 * s4));
            wg_commit();
            if (i > 0) {
                wg_wait1();
                __syncwarp();
                if (lane == 0) mbar_arrive(qempty + 8 * ((i - 1) % NSTQ));
            }
        }
        wg_wait0();
        // push-mode split-K over 3: warp w's rows belong to rank own(w) = (w == 3 ? 0 : w), slot = (w == 3 ? 1 : 0)
        cluster_wait();
        const int g = lane >> 2, c = lane & 3;
        const int own = warp == 3 ? 0 : warp, oslot = warp == 3 ? 1 : 0;
        if (own != qrank) {
            const int sidx = qrank < own ? qrank : qrank - 1;
            const int base = (oslot * 2 + sidx) * 16 * NT;
#pragma unroll
            for (int j = 0; j < NT / 8; j++) {
                const int col = 8 * j + 2 * c;
                st_dsmem2(rx + 4 * (base + g * NT + col), own, acc[4 * j], acc[4 * j + 1]);
                st_dsmem2(rx + 4 * (base + (g + 8) * NT + col), own, acc[4 * j + 2], acc[4 * j + 3]);
            }
            mbar_arrive_remote_release(rxbar, own);
        } else {
            mbar_wait_acq_cluster(rxbar, 0);
            const int grow = qrt * 64 + 16 * warp + g;
            const int b0 = (oslot * 2 + 0) * 16 * NT, b1 = (oslot * 2 + 1) * 16 * NT;
#pragma unroll
            for (int j = 0; j < NT / 8; j++) {
                const int t = 8 * j + 2 * c;
                const float v0 = acc[4 * j] + rxp[b0 + g * NT + t] + rxp[b1 + g * NT + t];
                const float v1 = acc[4 * j + 1] + rxp[b0 + g * NT + t + 1] + rxp[b1 + g * NT + t + 1];
                const float v2 = acc[4 * j + 2] + rxp[b0 + (g + 8) * NT + t] + rxp[b1 + (g + 8) * NT + t];
                const float v3 = acc[4 * j + 3] + rxp[b0 + (g + 8) * NT + t + 1] + rxp[b1 + (g + 8) * NT + t + 1];
                if (t < m) { qout[(size_t)t * QN + grow] = __float2bfloat16_rn(v0); qout[(size_t)t * QN + grow + 8] = __float2bfloat16_rn(v2); }
                if (t + 1 < m) { qout[(size_t)(t + 1) * QN + grow] = __float2bfloat16_rn(v1); qout[(size_t)(t + 1) * QN + grow + 8] = __float2bfloat16_rn(v3); }
            }
        }
    }
}

// ---------------- KO2b: KO2 with one barrier per W atom ----------------
template <int NT>
__global__ void __launch_bounds__(160, 1) ko2b_k(const __grid_constant__ CUtensorMap wom, const __grid_constant__ CUtensorMap xom,
                                               __nv_bfloat16* __restrict__ oout, int m)
{
    using L = LayO2b<NT>;
    extern __shared__ char smem_raw[];
    const unsigned sraw = smem_u32(smem_raw);
    const unsigned sbase = (sraw + 1023) & ~1023u;
    const unsigned A = sbase, X = sbase + OW_BYTES;
    const unsigned wst = X + NXS * L::XATOM, xfull = wst + 8 * OW_ATOMS, xempty = xfull + 8 * NXS;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ort = blockIdx.x;
    if (tid == 0) {
        for (int a = 0; a < OW_ATOMS; a++) mbar_init(wst + 8 * a, 1);
        for (int i = 0; i < NXS; i++) { mbar_init(xfull + 8 * i, 1); mbar_init(xempty + 8 * i, 4); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp == 4) {
        if (lane == 0) {
            for (int a = 0; a < OW_ATOMS; a++) { mbar_expect_tx(wst + 8 * a, WATOM); tma_2d(A + a * WATOM, &wom, 64 * a, ort * 64, wst + 8 * a); }
            pdl_wait();
            for (int a = 0; a < OW_ATOMS; a++) {
                const int slot = a % NXS;
                if (a >= NXS) mbar_wait(xempty + 8 * slot, ((a / NXS) - 1) & 1);
                mbar_expect_tx(xfull + 8 * slot, L::XATOM);
                tma_2d(X + slot * L::XATOM, &xom, 64 * a, 0, xfull + 8 * slot);
            }
        }
    } else {
        pdl_wait();
        float acc[NT / 2];
#pragma unroll
        for (int i = 0; i < NT / 2; i++) acc[i] = 0.f;
        for (int a = 0; a < OW_ATOMS; a++) {
            const int slot = a % NXS;
            mbar_wait(wst + 8 * a, 0);
            mbar_wait(xfull + 8 * slot, (a / NXS) & 1);
            wg_fence();
#pragma unroll
            for (int s4 = 0; s4 < 4; s4++)
                wgmma_nt<NT>(acc, desc_sw128(A + a * WATOM + 32 * s4), desc_sw128(X + slot * L::XATOM + 32 * s4));
            wg_commit();
            if (a > 0) {
                wg_wait1();
                __syncwarp();
                if (lane == 0) mbar_arrive(xempty + 8 * ((a - 1) % NXS));
            }
        }
        wg_wait0();
        store_frag<NT>(acc, oout, ON, ort * 64, warp, lane, m);
    }
}

}  // namespace wsc2

static PFN_cuTensorMapEncodeTiled_v12000 g_encode = nullptr;
static CUtensorMap make_map(const void* p, int rows, int cols, int box_rows) {
    if (!g_encode) {
        cudaDriverEntryPointQueryResult qr;
        TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_encode, cudaEnableDefault, &qr) ==
                    cudaSuccess && g_encode, "wsc2: no cuTensorMapEncodeTiled");
    }
    CUtensorMap mp;
    cuuint64_t gdim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
    cuuint64_t gstr[1] = {(cuuint64_t)cols * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g_encode(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), gdim, gstr, box, es,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "wsc2: tensor map encode failed ", (int)r);
    return mp;
}
// By value: a later lookup may clear the cache, which must not invalidate a map taken before it (qa-wsc W3 P1).
static std::map<std::tuple<const void*, int, int, int>, CUtensorMap> g_maps;
static CUtensorMap cached_map(const void* p, int rows, int cols, int box_rows) {
    auto key = std::make_tuple(p, rows, cols, box_rows);
    auto it = g_maps.find(key);
    if (it != g_maps.end()) return it->second;
    if (g_maps.size() > 4096) g_maps.clear();
    const CUtensorMap mp = make_map(p, rows, cols, box_rows);
    g_maps.emplace(key, mp);
    return mp;
}

// PDL-launched: each kernel issues its weight before griddepcontrol.wait and reads x only after it.
template <typename F>
static void launch(F fn, int grid, int smem, int cluster, const CUtensorMap& wm, const CUtensorMap& xm, torch::Tensor& out, int m) {
    static std::map<const void*, bool> set;
    if (!set[(const void*)fn]) {
        TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "wsc2: smem");
        set[(const void*)fn] = true;
    }
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(grid, 1, 1);
    lc.blockDim = dim3(160, 1, 1);
    lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute la[2];
    int na = 0;
    la[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[na].val.programmaticStreamSerializationAllowed = 1;
    na++;
    if (cluster > 1) {
        la[na].id = cudaLaunchAttributeClusterDimension;
        la[na].val.clusterDim.x = cluster; la[na].val.clusterDim.y = 1; la[na].val.clusterDim.z = 1;
        na++;
    }
    lc.attrs = la; lc.numAttrs = na;
    TORCH_CHECK(cudaLaunchKernelEx(&lc, fn, wm, xm, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), m) == cudaSuccess,
                "wsc2: launch");
}

// kind 0: qkv_gate [2304, 8192] at m 17-32 (KQ3); kind 1: o_proj [8192, 1024] at m 1-32 (KO2b)
void wsc2_launch(int64_t kind, torch::Tensor W, torch::Tensor x, torch::Tensor out) {
    const int N = (int)W.size(0), K = (int)W.size(1), m = (int)x.size(0);
    TORCH_CHECK(W.is_contiguous() && x.is_contiguous() && out.is_contiguous() && x.size(1) == K && out.size(0) == m &&
                out.size(1) == N, "wsc2: layout");
    if (kind == 0) {
        TORCH_CHECK(N == wsc2::QN && K == wsc2::QK && m >= 17 && m <= 32, "wsc2: kq3 shape");
        const CUtensorMap wm = cached_map(W.data_ptr(), N, K, 64);
        const CUtensorMap xm = cached_map(x.data_ptr(), m, K, 32);
        launch(wsc2::kq3_k<32, true>, wsc2::QCTAS3, wsc2::LayQ3<32>::SMEM, wsc2::QS3, wm, xm, out, m);
    } else {
        TORCH_CHECK(N == wsc2::ON && K == wsc2::OK && m >= 1 && m <= 32, "wsc2: ko2b shape");
        const CUtensorMap wm = cached_map(W.data_ptr(), N, K, 64);
        if (m <= 8) launch(wsc2::ko2b_k<8>, wsc2::OTILES, wsc2::LayO2b<8>::SMEM, 1, wm, cached_map(x.data_ptr(), m, K, 8), out, m);
        else if (m <= 16) launch(wsc2::ko2b_k<16>, wsc2::OTILES, wsc2::LayO2b<16>::SMEM, 1, wm, cached_map(x.data_ptr(), m, K, 16), out, m);
        else launch(wsc2::ko2b_k<32>, wsc2::OTILES, wsc2::LayO2b<32>::SMEM, 1, wm, cached_map(x.data_ptr(), m, K, 32), out, m);
    }
}
"""
