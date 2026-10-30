"""CUDA source of the W-staged small-m GEMM (OKC_SW_WSTAGE; okc_patch/smallm.py dispatches it): _cuda_src_smallm's rmw
kernel with a PDL-aware producer. The weight slice does not depend on the previous kernel, so the producer TMA's the
first NST stages of W before griddepcontrol.wait and x after it; launched with programmatic stream serialization, the
CTAs land while the predecessor (a collective or a norm) still runs and stream W under its latency. The consumer loop,
the cluster split-K reduction and the epilogue are rmw's, and a stage holds the same k-tile as in rmw at every NST, so
outputs are bitwise rmw's (muse-opt serving/kernels/wstage).

TABLE_WS: per (N, K) and NT, rmw's config with NST raised so a CTA can hold its whole W slice where shared memory and
residency allow (qkv m 8: 8 of 16 k-tiles, keeping three CTAs per SM; qkv m 16: 5 of 16; pooler m 64: 6 of 8)."""

TABLE_WS = {(512, 8192): {8: (8, 8, 8, 128, 1, 1, 1),
                          16: (16, 8, 8, 128, 1, 1, 0),
                          32: (32, 4, 8, 256, 1, 1, 1),
                          64: (64, 6, 8, 128, 1, 1, 1)},
            (1024, 4096): {8: (8, 4, 8, 128, 1, 1, 0), 16: (16, 4, 8, 128, 1, 1, 0), 32: (32, 4, 8, 128, 1, 1, 0)},
            (2304, 8192): {8: (8, 8, 8, 64, 1, 1, 1), 16: (16, 5, 4, 128, 1, 1, 0)}}

_CUDA_SRC_WSM = r"""

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <map>
#include <tuple>

#define DINL __device__ __forceinline__

namespace wsm {

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
// PDL: wait until the previous kernel has completed and its writes are visible (no-op without a PDL launch)
DINL void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DINL void wg_fence()  { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DINL void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
DINL void wg_wait0()  { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
DINL void wg_wait1()  { asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory"); }
// K-major 128B-swizzled smem operand: start >> 4, LBO 1 (unused), SBO 1024 B (8 rows x 128 B), layout 1 (SW128)
DINL uint64_t desc_sw128(unsigned saddr) {
    return (uint64_t)((saddr >> 4) & 0x3FFF) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
DINL void cluster_sync() {
    asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;" ::: "memory");
}
DINL float ld_dsmem(unsigned local_addr, unsigned rank) {
    unsigned remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
    float v;
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");
    return v;
}

DINL void wgmma_n8(float* acc, uint64_t da, uint64_t db, int scale_d) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, %4, %5, p, 1, 1, 0, 0;\n}\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]) : "l"(da), "l"(db), "r"(scale_d));
}
DINL void wgmma_n16(float* acc, uint64_t da, uint64_t db, int scale_d) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0;\n}\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]) : "l"(da), "l"(db), "r"(scale_d));
}
DINL void wgmma_n32(float* acc, uint64_t da, uint64_t db, int scale_d) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 0, 0;\n}\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]),
          "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]) : "l"(da), "l"(db), "r"(scale_d));
}
DINL void wgmma_n64(float* acc, uint64_t da, uint64_t db, int scale_d) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]),
          "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
          "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]), "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
          "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]), "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]) : "l"(da), "l"(db), "r"(scale_d));
}


template <int NT> DINL void wgmma_nt(float* acc, uint64_t da, uint64_t db, int sd) {
    if constexpr (NT == 8) wgmma_n8(acc, da, db, sd);
    else if constexpr (NT == 16) wgmma_n16(acc, da, db, sd);
    else if constexpr (NT == 32) wgmma_n32(acc, da, db, sd);
    else wgmma_n64(acc, da, db, sd);
}

template <int NT, int KT, int R> struct Lay {
    static constexpr int ATOMS = KT / 64;
    static constexpr int WB = R * 128;            // one W atom: R rows x 128 B
    static constexpr int XB = NT * 128;           // one x atom: NT rows x 128 B
    static constexpr int SB = ATOMS * (WB + XB);  // stage bytes (all TMA)
};

// rmw (NT tokens, NST stages of KT k, S CTAs per cluster splitting k, R = 64 RT NWG rows per CTA) with a W-first
// producer: stages 0 .. min(nkt, NST) - 1 get their W atoms before griddepcontrol.wait and their x atoms after it, both
// on the stage's one full barrier (expect_tx counts W + x); later stages run as in rmw. Every other thread waits on the
// grid dependency before its first ring read, so no thread touches the predecessor's output or writes out before the
// predecessor has completed.
template <int NT, int NST, int S, int KT, int RT, int NWG, int PIPE>
__global__ void __launch_bounds__(128 * NWG + 32, 1) rmw_ws(const __grid_constant__ CUtensorMap wmap,
                                                const __grid_constant__ CUtensorMap xmap,
                                                __nv_bfloat16* __restrict__ out, int N, int K, int m)
{
    constexpr int R = 64 * RT * NWG;
    constexpr int NTH = 128 * NWG + 32;
    using L = Lay<NT, KT, R>;
    extern __shared__ char smem_raw[];
    const unsigned sraw = smem_u32(smem_raw);
    const unsigned sbase = (sraw + 1023) & ~1023u;
    char* sgen = smem_raw + (sbase - sraw);
    const unsigned bars = sbase + NST * L::SB;    // full[NST], empty[NST]
    float* tile = reinterpret_cast<float*>(sgen);  // epilogue: [R][NT] fp32 over the ring (drained by then)
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row0 = blockIdx.x * R;
    const int q = blockIdx.y;                      // = rank in the (1, S, 1) cluster
    const int ntot = K / KT;
    const int kt0 = q * ntot / S, kt1 = (q + 1) * ntot / S, nkt = kt1 - kt0;
    if (tid == 0) {
        for (int i = 0; i < NST; i++) {
            mbar_init(bars + 8 * i, 1);
            mbar_init(bars + 8 * (NST + i), 4 * NWG);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp == 4 * NWG) {
        if (lane == 0) {
            const int npre = nkt < NST ? nkt : NST;
            for (int kt = 0; kt < npre; kt++) {   // W: constant weights, independent of the predecessor
                const unsigned full = bars + 8 * kt;
                mbar_expect_tx(full, L::SB);
                const unsigned st = sbase + kt * L::SB;
#pragma unroll
                for (int a = 0; a < L::ATOMS; a++) tma_2d(st + a * L::WB, &wmap, (kt0 + kt) * KT + 64 * a, row0, full);
            }
            pdl_wait();
            for (int kt = 0; kt < npre; kt++) {   // x: the predecessor's output
                const unsigned full = bars + 8 * kt;
                const unsigned st = sbase + kt * L::SB;
#pragma unroll
                for (int a = 0; a < L::ATOMS; a++)
                    tma_2d(st + L::ATOMS * L::WB + a * L::XB, &xmap, (kt0 + kt) * KT + 64 * a, 0, full);
            }
            for (int kt = npre; kt < nkt; kt++) {
                const int slot = kt % NST;
                mbar_wait(bars + 8 * (NST + slot), ((kt / NST) - 1) & 1);
                const unsigned full = bars + 8 * slot;
                mbar_expect_tx(full, L::SB);
                const unsigned st = sbase + slot * L::SB;
#pragma unroll
                for (int a = 0; a < L::ATOMS; a++) {
                    const int k = (kt0 + kt) * KT + 64 * a;
                    tma_2d(st + a * L::WB, &wmap, k, row0, full);
                    tma_2d(st + L::ATOMS * L::WB + a * L::XB, &xmap, k, 0, full);
                }
            }
        } else {
            pdl_wait();
        }
    } else {
        pdl_wait();
        const int wg = warp >> 2, wq = warp & 3;
        float acc[RT][NT / 2];
#pragma unroll
        for (int t = 0; t < RT; t++)
#pragma unroll
            for (int i = 0; i < NT / 2; i++) acc[t][i] = 0.f;
        for (int kt = 0; kt < nkt; kt++) {
            const int slot = kt % NST;
            mbar_wait(bars + 8 * slot, (kt / NST) & 1);
            const unsigned st = sbase + slot * L::SB;
            wg_fence();
#pragma unroll
            for (int a = 0; a < L::ATOMS; a++)
#pragma unroll
                for (int s4 = 0; s4 < 4; s4++)
#pragma unroll
                    for (int t = 0; t < RT; t++)
                        wgmma_nt<NT>(acc[t], desc_sw128(st + a * L::WB + (wg * RT + t) * 64 * 128 + 32 * s4),
                                     desc_sw128(st + L::ATOMS * L::WB + a * L::XB + 32 * s4), 1);
            wg_commit();
            if constexpr (PIPE) {   // keep this stage's group in flight; retire and release the previous stage's
                if (kt > 0) {
                    wg_wait1();
                    __syncwarp();
                    if (lane == 0) mbar_arrive(bars + 8 * (NST + (kt - 1) % NST));
                }
            } else {
                wg_wait0();
                __syncwarp();
                if (lane == 0) mbar_arrive(bars + 8 * (NST + slot));
            }
        }
        if constexpr (PIPE) wg_wait0();
        // park the fp32 tile: row tile (wg, t) warp wq rows 16 wq + (g, g + 8); n8 block j cols 8 j + 2 c + (0, 1)
        asm volatile("bar.sync 1, %0;" :: "n"(128 * NWG) : "memory");   // all consumers past their last ring read
        const int g = lane >> 2, c = lane & 3;
#pragma unroll
        for (int t = 0; t < RT; t++)
#pragma unroll
            for (int j = 0; j < NT / 8; j++) {
                const int r = (wg * RT + t) * 64 + 16 * wq + g, col = 8 * j + 2 * c;
                tile[r * NT + col] = acc[t][4 * j];
                tile[r * NT + col + 1] = acc[t][4 * j + 1];
                tile[(r + 8) * NT + col] = acc[t][4 * j + 2];
                tile[(r + 8) * NT + col + 1] = acc[t][4 * j + 3];
            }
    }
    __syncwarp();
    if constexpr (S > 1) {
        cluster_sync();   // every CTA's tile is parked
        constexpr int RQ = R / S;
        const unsigned tl = smem_u32(tile);
        for (int e = tid; e < RQ * NT; e += NTH) {
            const int r = q * RQ + e / NT, t = e % NT;
            float v = 0.f;
#pragma unroll
            for (int src = 0; src < S; src++) v += ld_dsmem(tl + 4 * (r * NT + t), src);
            if (t < m) out[(size_t)t * N + row0 + r] = __float2bfloat16_rn(v);
        }
        cluster_sync();   // peers may still read this CTA's tile
    } else {
        __syncthreads();
        for (int e = tid; e < R * NT; e += NTH) {
            const int t = e / R, r = e % R;
            if (t < m) out[(size_t)t * N + row0 + r] = __float2bfloat16_rn(tile[r * NT + t]);
        }
    }
}

}  // namespace wsm

static PFN_cuTensorMapEncodeTiled_v12000 g_encode = nullptr;

static CUtensorMap make_map(const void* p, int rows, int K, int box_rows) {
    if (!g_encode) {
        cudaDriverEntryPointQueryResult qr;
        TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_encode, cudaEnableDefault, &qr) ==
                    cudaSuccess && g_encode, "wsm: no cuTensorMapEncodeTiled");
    }
    CUtensorMap m;
    cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)rows};
    cuuint64_t gstr[1] = {(cuuint64_t)K * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g_encode(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), gdim, gstr, box, es,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "wsm: tensor map encode failed ", (int)r);
    return m;
}

static std::map<std::tuple<const void*, int, int, int>, CUtensorMap> g_maps;

static const CUtensorMap& cached_map(const void* p, int rows, int K, int box_rows) {
    auto key = std::make_tuple(p, rows, K, box_rows);
    auto it = g_maps.find(key);
    if (it != g_maps.end()) return it->second;
    if (g_maps.size() > 4096) g_maps.clear();
    return g_maps.emplace(key, make_map(p, rows, K, box_rows)).first->second;
}

template <int NT, int NST, int S, int KT, int RT, int NWG, int PIPE>
static void launch(const torch::Tensor& W, const torch::Tensor& x, torch::Tensor& out, bool pdl) {
    const int N = (int)W.size(0), K = (int)W.size(1), m = (int)x.size(0);
    constexpr int R = 64 * RT * NWG;
    using L = wsm::Lay<NT, KT, R>;
    const int smem = NST * L::SB + 2 * NST * 8 + 1024;
    TORCH_CHECK(N % R == 0, "wsm: rows");
    auto fn = wsm::rmw_ws<NT, NST, S, KT, RT, NWG, PIPE>;
    static bool set = false;
    if (!set) {
        TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                    "wsm: smem attribute");
        if (S > 8)
            TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess,
                        "wsm: non-portable cluster");
        set = true;
    }
    const CUtensorMap& wm = cached_map(W.data_ptr(), N, K, R);
    const CUtensorMap& xm = cached_map(x.data_ptr(), m, K, NT);
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(N / R, S, 1);
    lc.blockDim = dim3(128 * NWG + 32, 1, 1);
    lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute la[2];
    int na = 0;
    if (S > 1) {
        la[na].id = cudaLaunchAttributeClusterDimension;
        la[na].val.clusterDim.x = 1;
        la[na].val.clusterDim.y = S;
        la[na].val.clusterDim.z = 1;
        na++;
    }
    if (pdl) {
        la[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        la[na].val.programmaticStreamSerializationAllowed = 1;
        na++;
    }
    lc.attrs = la;
    lc.numAttrs = na;
    TORCH_CHECK(cudaLaunchKernelEx(&lc, fn, wm, xm, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), N, K, m) ==
                cudaSuccess, "wsm: launch");
}

void wsm_launch(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t NT, int64_t NST, int64_t S, int64_t KT,
                int64_t RT, int64_t NWG, int64_t PIPE, int64_t pdl) {
    const int N = (int)W.size(0), K = (int)W.size(1), m = (int)x.size(0);
    TORCH_CHECK(W.is_contiguous() && x.is_contiguous() && m >= 1 && m <= NT && N % 64 == 0 && K % KT == 0 &&
                (K / KT) >= S, "wsm: shape");
    const bool p = pdl != 0;
    if (NT == 8 && NST == 8 && S == 8 && KT == 128 && RT == 1 && NWG == 1 && PIPE == 1) { launch<8, 8, 8, 128, 1, 1, 1>(W, x, out, p); return; }
    if (NT == 16 && NST == 8 && S == 8 && KT == 128 && RT == 1 && NWG == 1 && PIPE == 0) { launch<16, 8, 8, 128, 1, 1, 0>(W, x, out, p); return; }
    if (NT == 32 && NST == 4 && S == 8 && KT == 256 && RT == 1 && NWG == 1 && PIPE == 1) { launch<32, 4, 8, 256, 1, 1, 1>(W, x, out, p); return; }
    if (NT == 64 && NST == 6 && S == 8 && KT == 128 && RT == 1 && NWG == 1 && PIPE == 1) { launch<64, 6, 8, 128, 1, 1, 1>(W, x, out, p); return; }
    if (NT == 8 && NST == 4 && S == 8 && KT == 128 && RT == 1 && NWG == 1 && PIPE == 0) { launch<8, 4, 8, 128, 1, 1, 0>(W, x, out, p); return; }
    if (NT == 16 && NST == 4 && S == 8 && KT == 128 && RT == 1 && NWG == 1 && PIPE == 0) { launch<16, 4, 8, 128, 1, 1, 0>(W, x, out, p); return; }
    if (NT == 32 && NST == 4 && S == 8 && KT == 128 && RT == 1 && NWG == 1 && PIPE == 0) { launch<32, 4, 8, 128, 1, 1, 0>(W, x, out, p); return; }
    if (NT == 8 && NST == 8 && S == 8 && KT == 64 && RT == 1 && NWG == 1 && PIPE == 1) { launch<8, 8, 8, 64, 1, 1, 1>(W, x, out, p); return; }
    if (NT == 16 && NST == 5 && S == 4 && KT == 128 && RT == 1 && NWG == 1 && PIPE == 0) { launch<16, 5, 4, 128, 1, 1, 0>(W, x, out, p); return; }
    TORCH_CHECK(false, "wsm: config not built");
}

"""
