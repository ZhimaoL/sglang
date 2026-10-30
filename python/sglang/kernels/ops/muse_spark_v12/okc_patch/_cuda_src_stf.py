"""CUDA source of the one-launch st pair (OKC_ST_FUSE; cuda_ext.st_fused_ext): st_gemm<1> and st_gemm<2> as two phases
of one cooperative kernel with a grid barrier between them, compiled after _cuda_src_st's text (cuda_sources
[_CUDA_SRC, this]), so the st module itself is unchanged.

In the two-launch form, K1 holds the whole register file, so K2's CTAs become resident only as K1's CTAs exit. Each then
rebuilds the routing table (and the replica balancer) before its first slab, and the late ones finish their equal slab
share last: ~7 us per call after K1's end (serving/kernels/extend/STFUSE.md). Here every CTA stays resident.
- The table is built once.
- Phase 1 is st_gemm<1>'s slab loop; at its end each CTA releases its h rows and arrives.
- Phase 2 prefetches its first W2 slab and then waits for the last arrival (where st_gemm<2> ran griddepcontrol.wait),
  before running st_gemm<2>'s loop.
stf_phase<KIND, FOLD> is st_gemm<KIND>'s body after build_tab, verbatim apart from that wait and the norm fold as a
template argument (host dispatch on gamma; the served verify path, whose pooler gather emits the norm, runs FOLD 0 and
compiles leaner than st_gemm<1>): the same slab ranges, K split, partial ring and epilogues, so y and the slots are
bit-identical to the pair.
Barrier: sense-reversing over stf_arrive / stf_gen. Each CTA's thread 0 arrives with one atom.acq_rel, which releases the
CTA's h rows (after a bar.sync); the last arriver also acquires every arrival, resets the count and st.release-s the next
generation. The others spin relaxed, then do one ld.acquire. There is no reset race: the next launch starts after this
one completes. The arrive comes before any phase-2 load is issued, so its fence waits only for phase 1's last refill
(L2 hits).
"""

_CUDA_SRC_STF = r"""
namespace stk {
__device__ unsigned stf_arrive, stf_gen;   // grid barrier: arrivals of the running launch, its generation
// st_fused's shared memory (namespace scope: the phases address it directly, as st_gemm does its own)
__shared__ Tab stf_tb;
__shared__ __align__(16) float4 stf_part[R][NW][32];
__shared__ __align__(16) unsigned short stf_scr[NW][128];
__shared__ __align__(8) unsigned long long stf_bars[2][2 * R];   // per phase: full[R] | empty[R]
__shared__ float stf_rs[2][8];
__shared__ __align__(8) unsigned long long stf_xbar[2];
extern __shared__ __align__(16) __nv_bfloat16 stf_x[];

DINL unsigned ld_relaxed_u32(const unsigned* p) {
    unsigned v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
DINL unsigned ld_acquire_u32(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
DINL unsigned atom_add_acq_rel_u32(unsigned* p, unsigned v) {
    unsigned old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}
DINL void st_relaxed_u32(unsigned* p, unsigned v) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
DINL void st_release_u32(unsigned* p, unsigned v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// st_gemm<KIND>'s body after build_tab. KIND 2 waits out generation gen before its first h read (wait = 0: the last
// arriver, which acquired every arrival already)
template <int KIND, bool FOLD>
DINL void stf_phase(const __nv_bfloat16* __restrict__ xin, const char* __restrict__ wsf, __nv_bfloat16* __restrict__ out,
                    const __nv_bfloat16* __restrict__ gamma_, float eps, unsigned gen, int wait) {
    const __nv_bfloat16* __restrict__ gamma = FOLD ? gamma_ : nullptr;   // compile-time null without the fold
    Tab& tb = stf_tb;
    auto& part = stf_part;
    auto& scr = stf_scr;
    auto& bars = stf_bars[KIND - 1];
    auto& s_rs = stf_rs;
    auto& s_xbar = stf_xbar;
    __nv_bfloat16* s_x = stf_x;
    constexpr int NWT = KIND == 1 ? 64 : 32;
    constexpr int LSPE = KIND == 1 ? 9 : 8;            // log2 slabs per pass (NWT x 8 slabs)
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int U = tb.nva << LSPE;
    const int s0 = (int)(((long long)U * blockIdx.x) / gridDim.x);
    const int nslab = (int)(((long long)U * (blockIdx.x + 1)) / gridDim.x) - s0;
    if (nslab <= 0) return;                            // no h read: no wait (this CTA has arrived)
    const unsigned fullb = smem_u32(&bars[0]), emptyb = smem_u32(&bars[R]);

    auto slab_ptr = [&](int jl) -> const char* {
        const int s = s0 + min(jl, nslab - 1);
        const int va = s >> LSPE, loc = s & ((1 << LSPE) - 1);
        const int e = tb.va_e[va], wt = loc >> 3, wi = loc & 7;
        return wsf + ((size_t)(e * NWT + wt) * NKT + KPW * w) * WSB + wi * 512;
    };
    const int lofs = lane * 16, sofs = 4096 + g * SFB;
    auto sofs_of = [&](int jl) -> int { return sofs - ((s0 + min(jl, nslab - 1)) & 7) * (512 - 8 * SFB); };

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
    if (KIND == 2) {                                   // st_gemm<2>'s pdl_wait: every CTA's h rows are released
        if (tid == 0 && wait) {
            while (ld_relaxed_u32(&stf_gen) == gen) {
            }
            (void)ld_acquire_u32(&stf_gen);
        }
        __syncthreads();
    }

    unsigned b[KPW][4][2];
    int cva = -1;
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
        const bool live = g < tb.va_n[va];
        const __nv_bfloat16* row = KIND == 1 ? xin + (size_t)(tb.va_rt[va][live ? g : 0] >> 3) * H
                                             : xin + (size_t)(va * 8 + g) * H;
        const __nv_bfloat16* grow = gamma;
        float rs = 0.f;
        if (KIND == 1 && gamma != nullptr) {
            const int nv = tb.va_n[va], bi = nbuf;
            if (pva != va && tid == 0) stage(va, bi, false);
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
            row = xb + g * XROW;
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
            const float gsc = tb.gs[tb.va_e[va]] * 256.0f;
            const float ngl = -1.4426950408889634f * gsc, gsq = gsc * gsc;
            scr[w][(2 * t) * 8 + g] = __bfloat16_as_ushort(__float2bfloat16_rn(okc_silu_mul(v.x, v.z, ngl, gsq)));
            scr[w][(2 * t + 1) * 8 + g] = __bfloat16_as_ushort(__float2bfloat16_rn(okc_silu_mul(v.y, v.w, ngl, gsq)));
            __syncwarp();
            if (lane < tb.va_n[va])
                *reinterpret_cast<uint4*>(out + (size_t)(va * 8 + lane) * H + wt * 64 + wi * 8) =
                    *reinterpret_cast<const uint4*>(&scr[w][lane * 8]);
        } else {
            const __nv_bfloat162 r0 = __floats2bfloat162_rn(v.x * 256.0f, v.y * 256.0f);
            const __nv_bfloat162 r1 = __floats2bfloat162_rn(v.z * 256.0f, v.w * 256.0f);
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
        if (KIND == 1 && gamma != nullptr && pva < 0) {
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
            wr[i] = ldw(p + (size_t)i * WSB + lofs);
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

template <bool FOLD>
__global__ void __launch_bounds__(NTHR, 1) st_fused(const __nv_bfloat16* __restrict__ x, const char* __restrict__ wsf13,
                                                    const float* __restrict__ gs, __nv_bfloat16* __restrict__ h,
                                                    const char* __restrict__ wsf2, __nv_bfloat16* __restrict__ y,
                                                    const int* __restrict__ ids, int numel, int nl,
                                                    const __nv_bfloat16* __restrict__ gamma, float eps,
                                                    const int* __restrict__ rtab, int n_rep, int me, int passes,
                                                    int* __restrict__ lids) {
    Tab& tb = stf_tb;
    __shared__ unsigned short s_wcnt[NUMEL_MAX / 32][ELOC];
    __shared__ int s_vbase[ELOC];
    __shared__ int s_rtab[256], s_rcnt[256], s_rown[256], s_rord[256], s_rload[8];
    __shared__ unsigned s_gen;
    __shared__ int s_wait;

    const int tid = threadIdx.x;
    if (tid < 2 * R) {
        mbar_init(smem_u32(&stf_bars[tid / R][tid % R]), NTHR);
        mbar_init(smem_u32(&stf_bars[tid / R][R + tid % R]), 32);
    }
    if (tid == 2 * R || tid == 2 * R + 1) mbar_init(smem_u32(&stf_xbar[tid - 2 * R]), 1);
    if (rtab != nullptr) rep_owner<RPT>(ids, numel, rtab, n_rep, passes, s_rtab, s_rcnt, s_rown, s_rord, s_rload);
    build_tab(ids, numel, nl, gs, tb, s_wcnt, s_vbase, rtab != nullptr ? s_rtab : nullptr, s_rown, me,
              blockIdx.x == 0 ? lids : nullptr);   // ends with __syncthreads
    if (tb.nva == 0) return;                           // no local route here: every CTA sees it, nobody arrives

    stf_phase<1, FOLD>(x, wsf13, h, gamma, eps, 0u, 0);
    __syncthreads();                                   // this CTA's h rows are written
    if (tid == 0) {
        const unsigned gen = ld_relaxed_u32(&stf_gen);   // changes only after every CTA of this launch arrived
        s_gen = gen;
        s_wait = 1;
        // release this CTA's h rows (cumulative after the bar.sync); the last arriver also acquires every arrival
        if (atom_add_acq_rel_u32(&stf_arrive, 1u) == gridDim.x - 1) {
            st_relaxed_u32(&stf_arrive, 0u);
            st_release_u32(&stf_gen, gen + 1u);       // open the next generation
            s_wait = 0;
        }
    }
    __syncthreads();                                   // phase 2's loads only after the release
    stf_phase<2, false>(h, wsf2, y, nullptr, 0.f, s_gen, s_wait);
}
}  // namespace stk

// st_moe's arguments and contract (pdl unused: one launch), on st_fused
void st_moe_fused(torch::Tensor x, torch::Tensor ids, torch::Tensor wsf13, torch::Tensor gs13, torch::Tensor wsf2,
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
    const bool rep = tab.numel() > 0;
    const int n_rep = rep ? (int)tab.numel() - 256 : 0;
    TORCH_CHECK(!rep || (tab.scalar_type() == at::kInt && tab.is_contiguous() && n_rep >= 0 && n_rep <= 256 &&
                         me >= 0 && me < 8 && lids.scalar_type() == at::kInt && lids.is_contiguous() &&
                         lids.numel() == numel),
                "st: replica table int32 [256 + n_rep], rank < 8, int32 lids like ids");
    static bool attr = false;
    if (!attr) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(stk::st_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            stk::XSTAGE_BYTES));
        attr = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(st_num_sms());                  // one CTA per SM, all resident: the grid barrier needs it
    cfg.blockDim = dim3(stk::NTHR);
    cfg.dynamicSmemBytes = fold ? stk::XSTAGE_BYTES : 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeCooperative;
    at[0].val.cooperative = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fold ? stk::st_fused<true> : stk::st_fused<false>,
                                      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                      reinterpret_cast<const char*>(wsf13.data_ptr()), (const float*)gs13.data_ptr<float>(),
                                      reinterpret_cast<__nv_bfloat16*>(h.data_ptr()),
                                      reinterpret_cast<const char*>(wsf2.data_ptr()),
                                      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), (const int*)ids.data_ptr<int>(),
                                      numel, nl, fold ? reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()) : nullptr,
                                      (float)eps, rep ? (const int*)tab.data_ptr<int>() : nullptr, n_rep, (int)me,
                                      (int)passes, rep ? lids.data_ptr<int>() : nullptr));
}
"""
