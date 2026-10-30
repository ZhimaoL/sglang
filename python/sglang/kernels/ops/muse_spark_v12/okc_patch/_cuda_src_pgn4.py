"""CUDA source of pg's narrow64 gg1 on four consumer warpgroups (OKC_PG64_N4; cuda_ext.pg_n4_ext).

The prefill lane's pg_n4 (kernels/muse_prefill_layer/pg_kernel9.py) on the narrow64 build: items of 256 weight rows
(w-tile pair wp of a live block) instead of 128, warpgroup c4 = 0..3 computing rows 64 (c4 & 1) .. + 63 of w-tile
2 wp + (c4 >> 1), all four on the same X k-tiles. 640 threads, setmaxnreg 24 / 112 (a <= 64-token item needs 32
accumulators). Built as one translation unit after _cuda_src_pg's narrow64 text (cuda_sources [_CUDA_SRC, this]), so
_cuda_src_pg is unchanged and no other module rebuilds. kstep4 / n4_epi are pg_gemm's PAIR k-step (MODE 0, LAG 1) and
its narrow gg1 epilogue with the N4 slot stride, the width switched per k-step inside one k-loop (see kstep4): per
thread the same W|sf bytes, decode, wgmma sequence and K order as pg_gemm's consumer of that row half, so every inter
element is bit-identical. A PAIR slot holds X k-tiles kt / kt + 1 at 0 / 8 KB and W|sf (kt, kt + 1) of w-tile 2 wp at
16 KB, of 2 wp + 1 at 16 KB + 2 WSB (36 KB); five slots (six do not fit in 227 KB with the four epilogue buffers).
"""

_CUDA_SRC_N4 = r"""
#if !defined(PGN_WSET) || PGN_XNT != 64
#error pg_n4 builds on the narrow64 instance only (-DPGN_WSET -DPGN_XNT=64)
#endif
namespace pgk {
constexpr int N4_NST = 5, N4_SB = 16384 + 4 * WSB, N4_PREGS = 24, N4_CREGS = 112;
constexpr int N4_SMEM = N4_NST * N4_SB + BAR_BYTES + 4 * EPI1;   // ring | full / empty / meta | 4 epilogue half-tiles
static_assert(N4_SB % 1024 == 0 && 2 * N4_NST * 8 + 32 <= BAR_BYTES && N4_SMEM <= 232448, "n4 smem layout");
static_assert(128 * N4_PREGS + 512 * N4_CREGS <= 640 * 96, "n4 setmaxnreg budget exceeds the launch allocation");

// one k16 group of 4 wgmmas of width NV (fence and commit inside the width's branch of kstep4)
template <int NV>
DINL void mma4(float* acc, unsigned* ab, unsigned dlo0, unsigned dhi1, int kt) {
    wgmma_fence();
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const unsigned long long db = ((unsigned long long)dhi1 << 32) | (unsigned long long)(dlo0 + q * 2);
        wgmma_nv<NV>(acc, ab + 4 * q, db, (kt > 0 || q > 0) ? 1 : 0);
    }
    wgmma_commit();
}

// kstep<1, NV, NST, 0, 1, 1, SUB, 1> with the N4 slot stride (c.woff selects the warpgroup's w-tile in the slot) and the
// width nv chosen per k-step: ONE k-loop for every width, so both A banks keep one register set. With a k-loop per
// width (pg_gemm's form) ptxas gives bank B a register range per width and serializes every wgmma at 112 registers
// (C7512; three widths need ~128, two fit); a commit after the width branches costs a dummy group per k-step.
template <int SUB>
DINL void kstep4(Ctx& c, int kt, int nv, float* acc, unsigned (&ab)[16]) {
    if (SUB == 0) mbar_wait(c.mb + c.sl * 8, c.cph);
    const char* stp = c.smem + c.sl * N4_SB + 16384 + SUB * WSB;   // c.ctid / c.sfo: byte offsets incl. c.woff
    const uint4 w = *reinterpret_cast<const uint4*>(stp + c.ctid);
    const uint4 sfp = *reinterpret_cast<const uint4*>(stp + c.sfo);   // bf16 (r0, r1) x 4
    const unsigned* sfq = reinterpret_cast<const unsigned*>(&sfp);
    const unsigned* wq = reinterpret_cast<const unsigned*>(&w);
#pragma unroll
    for (int q = 0; q < 4; q++) {
        __nv_bfloat162 d[4];
        dq3(wq[q], d);
        const __nv_bfloat162 sf2 = u32b2(sfq[q]);
        const __nv_bfloat162 s0 = __bfloat162bfloat162(__low2bfloat16(sf2));
        const __nv_bfloat162 s1 = __bfloat162bfloat162(__high2bfloat16(sf2));
        ab[4 * q + 0] = b2u32(__hmul2(d[0], s0));
        ab[4 * q + 1] = b2u32(__hmul2(d[1], s1));
        ab[4 * q + 2] = b2u32(__hmul2(d[2], s0));
        ab[4 * q + 3] = b2u32(__hmul2(d[3], s1));
    }
    const unsigned xaddr = c.sbase + c.sl * N4_SB + SUB * 8192;
    const unsigned dlo0 = (unsigned)c.dhi + (xaddr >> 4);
    const unsigned dhi1 = (unsigned)(c.dhi >> 32);
    if ((PGN_WSET & 16) && nv == 16) mma4<16>(acc, ab, dlo0, dhi1, kt);
    else if ((PGN_WSET & 8) && nv == 32) mma4<32>(acc, ab, dlo0, dhi1, kt);
    else mma4<XNT>(acc, ab, dlo0, dhi1, kt);
    wgmma_wait<1>();
    if (SUB == 0 && kt >= 2) {
        mbar_arrive(c.mb + N4_NST * 8 + c.rsl * 8);
        if (++c.rsl == N4_NST) c.rsl = 0;
    }
    if (SUB == 1) {
        if (++c.sl == N4_NST) { c.sl = 0; c.cph ^= 1; }
    }
}

// item_body<1, N4_NST, 0, NV, 1>'s narrow gg1 epilogue: this warpgroup's SW64 half-tile of w-tile wt
template <int NV>
DINL void n4_epi(Ctx& c, float (&acc)[XNT / 2], int pm, int wt, float gsv, void* __restrict__ out, int NWT) {
    fence_acc(acc, NV / 2);
    const int lw = c.lw, lane = c.lane, wtid = c.wtid, cw = c.hf, bar = c.bar;
    const unsigned epi = c.epi;
    const float gsc = gsv * 256.0f;                      // 2^8 decode compensation x global scale
    const float ngl = -1.4426950408889634f * gsc, gsq = gsc * gsc;
    if (wtid == 0) bulk_wait_read0();                    // previous item's bulk store finished reading epi
    named_bar_sync(bar, 128);
    const unsigned lanebase = epi + ((lane >> 3) * 8 + (lane & 7)) * 64 + ((lw ^ (((lane & 7) >> 1) & 3)) << 4);
    if constexpr (NV == 16) {
        unsigned r[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            r[i] = b2u32(__floats2bfloat162_rn(okc_silu_mul(acc[4 * i], acc[4 * i + 2], ngl, gsq),
                                                okc_silu_mul(acc[4 * i + 1], acc[4 * i + 3], ngl, gsq)));
        }
        stsm_x2_trans(lanebase, r[0], r[1]);
    }
#pragma unroll
    for (int s = 0; s < NV / 32; s++) {
        unsigned r[4];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int i = 4 * s + m;
            r[m] = b2u32(__floats2bfloat162_rn(okc_silu_mul(acc[4 * i], acc[4 * i + 2], ngl, gsq),
                                                okc_silu_mul(acc[4 * i + 1], acc[4 * i + 3], ngl, gsq)));
        }
        stsm_x4_trans(lanebase + s * 32 * 64, r[0], r[1], r[2], r[3]);
    }
    fence_proxy_async();
    named_bar_sync(bar, 128);
    if (wtid == 0) {                                     // the half-tile's first NV token rows are contiguous
        bulk_s2g(reinterpret_cast<char*>(out) + ((size_t)pm * NWT + wt) * XB + cw * (XB / 2), epi, NV * 64);
        bulk_commit();
    }
}

// pg_gemm<1, PGN_NST, 0, PGN_WSET> on items (live block, w-tile pair wp), static round-robin / snake as pg_gemm: one
// producer thread stages each slot (the shared X k-tiles + both w-tiles' W|sf), all 512 consumer threads release it
__global__ void __launch_bounds__(640, 1) pg_n4(
    const char* __restrict__ xg, const char* __restrict__ wsf, const float* __restrict__ gs, void* __restrict__ out,
    const int* __restrict__ block_experts, const int* __restrict__ live, int NWT, int snake)
{
    extern __shared__ __align__(1024) char smem[];
    const int tid = threadIdx.x;
    const unsigned sbase = smem_u32(smem);
    if (sbase & 1023) __trap();
    const unsigned mb = sbase + N4_NST * N4_SB;         // full[N4_NST] | empty[N4_NST] | meta[2][4]
    int* meta = reinterpret_cast<int*>(smem + N4_NST * N4_SB + 2 * N4_NST * 8);
    const unsigned ebase = mb + BAR_BYTES;
    if (tid < N4_NST) {
        mbar_init(mb + tid * 8, 1);
        mbar_init(mb + N4_NST * 8 + tid * 8, 512);      // empty: every consumer thread arrives
    }
    pdl_wait();
    pdl_trigger();
    __syncthreads();
    const int NWP = NWT / 2;
    const int n_items = live[0] * NWP;
    const int wg = tid >> 7;

    if (wg == 0) {
        reg_dec<N4_PREGS>();
        if (tid == 0) {
            int sl = 0, rnd = 0;
            int item = sched_item(0, blockIdx.x, gridDim.x, snake);
            int ent_n = item < n_items ? live[1 + item / NWP] : 0;
            int e_n = item < n_items ? block_experts[ent_n & 0xFFFFF] : 0;
            float g_n = item < n_items ? gs[e_n] : 0.f;
            for (int j = 0; item < n_items; j++) {
                const int wp = item % NWP;
                const int ent = ent_n, e = e_n;
                const float gsv = g_n;
                const int pm = ent & 0xFFFFF, wcode = ent >> 20;
                const int nv = eff_nv<PGN_WSET>(wcode);
                const int nxt = sched_item(j + 1, blockIdx.x, gridDim.x, snake);
                if (nxt < n_items) ent_n = live[1 + nxt / NWP];
                int* mt = meta + (j & 1) * 4;           // published before this item's first stage arrives
                mt[0] = pm; mt[1] = wp; mt[2] = wcode; mt[3] = __float_as_int(gsv);
                const char* xs = xg + (size_t)pm * NKT * XB;
                const char* ws = wsf + ((size_t)e * NWT + 2 * wp) * (size_t)NKT * WSB;   // w-tile 2 wp, then 2 wp + 1
                for (int ks = 0; ks < NKT / 2; ks++) {
                    if (rnd > 0) mbar_wait(mb + N4_NST * 8 + sl * 8, (rnd - 1) & 1);
                    const unsigned st = sbase + sl * N4_SB, fb = mb + sl * 8;
                    const int kt = 2 * ks;
                    mbar_expect_tx(fb, 2 * nv * 128 + 4 * WSB);
                    bulk_g2s(st, xs + (size_t)kt * XB, nv * 128, fb);
                    bulk_g2s(st + 8192, xs + (size_t)(kt + 1) * XB, nv * 128, fb);
                    bulk_g2s(st + 16384, ws + (size_t)kt * WSB, 2 * WSB, fb);
                    bulk_g2s(st + 16384 + 2 * WSB, ws + (size_t)(NKT + kt) * WSB, 2 * WSB, fb);
                    if (++sl == N4_NST) { sl = 0; rnd++; }
                    if (ks == 8 && nxt < n_items) e_n = block_experts[ent_n & 0xFFFFF];
                    if (ks == 16 && nxt < n_items) g_n = gs[e_n];
                }
                item = nxt;
            }
        }
        return;
    }

    reg_inc<N4_CREGS>();
    const int c4 = wg - 1;
    Ctx c;
    c.smem = smem;
    c.sbase = sbase;
    c.mb = mb;
    c.cw = c4 & 1;                          // row half of the w-tile (A fragments, scales, output half-tile)
    c.bar = 1 + c4; c.hf = c.cw; c.woff = (c4 >> 1) * 2 * WSB;
    c.wtid = tid & 127;
    c.lw = (tid >> 5) & 3;
    c.lane = tid & 31;
    // the W|sf bytes pg_gemm's consumer cw reads (ctid = cw * 128 + tid & 127, sfo), as offsets in the slot's W|sf half;
    // opaque to NVVM, which otherwise re-derives both from tid inside the k-loop (~10 instructions per 2 k-tiles)
    c.ctid = c.woff + (c.cw * 128 + (tid & 127)) * 16;
    c.sfo = c.woff + 4096 + 16 * (c.cw * 32 + c.lw * 8 + (c.lane >> 2));
    asm volatile("" : "+r"(c.ctid), "+r"(c.sfo));
    c.dhi = (1ull << 16) | ((unsigned long long)(1024 >> 4) << 32) | (1ull << 62);
    c.epi = ebase + c4 * EPI1;
    c.sl = 0; c.cph = 0;
    c.rsl = 0;
    for (int j = 0, item = sched_item(0, blockIdx.x, gridDim.x, snake); item < n_items;
         j++, item = sched_item(j, blockIdx.x, gridDim.x, snake)) {
        mbar_wait(c.mb + c.sl * 8, c.cph);  // the item's first stage (and its metadata) has landed
        const volatile int* mt = meta + (j & 1) * 4;
        const int pm = mt[0], wt = 2 * mt[1] + (c4 >> 1), wcode = mt[2];
        const float gsv = __int_as_float(mt[3]);
        float acc[XNT / 2];
        unsigned bA[16], bB[16];
        const int nv = eff_nv<PGN_WSET>(wcode);
        for (int kt = 0; kt < NKT; kt += 2) {
            kstep4<0>(c, kt, nv, acc, bA);
            kstep4<1>(c, kt + 1, nv, acc, bB);
        }
        wgmma_wait<0>();
        release_rest<N4_NST>(c, 1);
        fence_bank(bA); fence_bank(bB);
        if ((PGN_WSET & 16) && nv == 16)
            n4_epi<16>(c, acc, pm, wt, gsv, out, NWT);
        else if ((PGN_WSET & 8) && nv == 32)
            n4_epi<32>(c, acc, pm, wt, gsv, out, NWT);
        else
            n4_epi<XNT>(c, acc, pm, wt, gsv, out, NWT);
    }
    if (c.wtid == 0) bulk_wait0();          // stores complete before the CTA retires its smem
}
}  // namespace pgk

// gg1's entry with the same arguments, on pg_n4 (OKC_PG64_N4)
void gg1n4(torch::Tensor xg, torch::Tensor wsf, torch::Tensor gs, torch::Tensor inter,
           torch::Tensor block_experts, torch::Tensor live, int64_t numel, int64_t cfg, int64_t grid, int64_t pdl) {
    TORCH_CHECK(cfg == 350, "pg: only cfg 350 is built");
    const int NWT = wsf.size(1);
    TORCH_CHECK(wsf.size(2) == pgk::NKT && wsf.size(3) == pgk::WSB && NWT % 2 == 0, "pg n4: wsf layout");
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pgk::pg_n4, cudaFuncAttributeMaxDynamicSharedMemorySize, pgk::N4_SMEM);
        set = true;
    }
    pg_launch(pgk::pg_n4, dim3(grid > 0 ? grid : pg_num_sms()), dim3(640), pgk::N4_SMEM, pdl != 0,
              reinterpret_cast<const char*>(xg.data_ptr()), reinterpret_cast<const char*>(wsf.data_ptr()),
              gs.data_ptr<float>(), inter.data_ptr(), block_experts.data_ptr<int>(), live.data_ptr<int>(), NWT,
              (numel / 8 <= pgk_snake_max_t) ? 1 : 0);
}
"""
