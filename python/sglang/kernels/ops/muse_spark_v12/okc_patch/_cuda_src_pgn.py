"""Narrow-width build of the pg MoE kernels for follow-up extends (generated from _cuda_src_pg at import).

A follow-up extend (~1K tokens) gives each rank ~25 active local experts with a long tail of tiny ones (session M
routing, rank 0 at T 1024: tokens per expert p10 / p50 / p90 = 2 / 16 / 96), and a pg item costs about the same
whatever share of its 64 token columns is real. This build adds wgmma widths 16 and 32 to the item widths, so a
<= 16-token expert runs m64n16 items. Same weights, same accumulation per output element: the real tokens' outputs are
bit-identical to the 64-wide build (serving/kernels/extend/pg_mid_bench.py --check); only padded columns are skipped.

The text edits below apply to the shipped source and each must match exactly once; the shipped extension
(cuda_ext.pg_ext) is untouched and keeps every other T. Width codes become nv / 16 - 1 in the live list, wmask bits
are 0 = 64, 1 = 128, 2 = 192, 3 = 32, 4 = 16, and every item of <= 64 tokens stages two k-tiles per ring slot.
ptxas keeps the wgmmas unserialized for at most three width loops (C7512 beyond), so this build also halves the
block: moe_align blocks of 128 tokens (XNT 128), widths 16 / 64 / 128 (PGN_WSET = 17: bits 16 and 64, the block width
always). A > 128-token expert gets several 128-token blocks instead of one 256-wide item, and the 16 KB X stage
leaves room for PGN_NST = 7 ring slots (5 at XNT 256) in the same shared memory.
"""

from sglang.kernels.ops.muse_spark_v12.okc_patch import _cuda_src_pg as _pg


def _rep(s: str, old: str, new: str) -> str:
    if s.count(old) != 1:
        raise RuntimeError(f"pg narrow build: source edit does not match once: {old[:80]!r}")
    return s.replace(old, new)


def narrow_source() -> str:
    s = _pg._CUDA_SRC
    s = _rep(s, "// ------------------------------------------------------------------ geometry",
             _pg._wgmma_asm(32) + _pg._wgmma_asm(16)
             + "\n// ------------------------------------------------------------------ geometry")
    s = _rep(s, """    else if constexpr (NV == 128) wgmma_n128(d, a, db, accum);
    else wgmma_n64(d, a, db, accum);""", """    else if constexpr (NV == 128) wgmma_n128(d, a, db, accum);
    else if constexpr (NV == 64) wgmma_n64(d, a, db, accum);
    else if constexpr (NV == 32) wgmma_n32(d, a, db, accum);
    else wgmma_n16(d, a, db, accum);""")
    s = _rep(s, "DINL void stsm_x4_trans(unsigned addr, unsigned r0, unsigned r1, unsigned r2, unsigned r3) {",
             """DINL void stsm_x2_trans(unsigned addr, unsigned r0, unsigned r1) {
    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};"
                 :: "r"(addr), "r"(r0), "r"(r1) : "memory");
}
DINL void stsm_x4_trans(unsigned addr, unsigned r0, unsigned r1, unsigned r2, unsigned r3) {""")
    # live list: width code = nv / 16 - 1, classes widest first
    s = _rep(s, """    for (int cls = 3; cls >= 0; cls--) {
        for (int base = 0; base < npm; base += 1024) {
            const int pm = base + tid;
            const bool live_blk = pm < npm && (unsigned)block_experts[pm] < 32u;
            int code = 3;
            if (live_blk && bm == 256) {
                const int* sid = sorted_ids + (size_t)pm * 256;
                if ((wmask & 4) && sid[192] >= numel) code = 2;
                if ((wmask & 2) && sid[128] >= numel) code = 1;
                if ((wmask & 1) && sid[64] >= numel) code = 0;
            }""", """    const int CLS[6] = {15, 11, 7, 3, 1, 0};
    const int CLM[6] = {0, 4, 2, 1, 8, 16};
    for (int ci = 0; ci < 6; ci++) {
        const int cls = CLS[ci];
        if (cls != XNT / 16 - 1 && !(wmask & CLM[ci])) continue;   // the full-block class always runs
        for (int base = 0; base < npm; base += 1024) {
            const int pm = base + tid;
            const bool live_blk = pm < npm && (unsigned)block_experts[pm] < 32u;
            int code = 15;
            if (live_blk && bm == 256) {
                const int* sid = sorted_ids + (size_t)pm * 256;
                if ((wmask & 4) && sid[192] >= numel) code = 11;
                if ((wmask & 2) && sid[128] >= numel) code = 7;
                if ((wmask & 1) && sid[64] >= numel) code = 3;
                if ((wmask & 8) && sid[32] >= numel) code = 1;
                if ((wmask & 16) && sid[16] >= numel) code = 0;
            }""")
    s = _rep(s, """template <int WSET> DINL int eff_nv(int wcode) {
    if ((WSET & 1) && wcode == 0) return 64;
    if ((WSET & 2) && wcode <= 1) return 128;
    if ((WSET & 4) && wcode <= 2) return 192;
    return 256;
}""", """template <int WSET> DINL int eff_nv(int wcode) {
    const int need = 16 * (wcode + 1);
    if ((WSET & 16) && need <= 16) return 16;
    if ((WSET & 8) && need <= 32) return 32;
    if ((WSET & 1) && need <= 64) return 64;
    if ((WSET & 2) && need <= 128) return 128;
    if ((WSET & 4) && need <= 192) return 192;
    return 256;
}""")
    # two k-tiles per ring slot for every item of <= 64 tokens, copying only its nv token rows
    s = _rep(s, "    constexpr int PR = (PAIR64 && NV == 64) ? 1 : 0;", "    constexpr int PR = (PAIR64 && NV <= 64) ? 1 : 0;")
    s = _rep(s, "                const bool pr = PAIR64 && nv == 64;     // two k-tiles per slot (see kstep)",
             "                const bool pr = PAIR64 && nv <= 64;     // two k-tiles per slot (see kstep)")
    s = _rep(s, """                        mbar_expect_tx(mb + sl * 8, 2 * 8192 + 2 * WSB + ridb);
#pragma unroll
                        for (int sub = 0; sub < 2; sub++) {
                            const char* xk = xs + (size_t)(kt + sub) * XB;
                            if (GATED) {
                                bulk_g2s(st + sub * 8192, xk, 8192, mb + sl * 8);
                            } else {
                                bulk_g2s(st + sub * 8192, xk, 4096, mb + sl * 8);
                                bulk_g2s(st + sub * 8192 + 4096, xk + XB / 2, 4096, mb + sl * 8);
                            }
                        }""", """                        mbar_expect_tx(mb + sl * 8, 2 * nv * 128 + 2 * WSB + ridb);
#pragma unroll
                        for (int sub = 0; sub < 2; sub++) {
                            const char* xk = xs + (size_t)(kt + sub) * XB;
                            if (GATED) {
                                bulk_g2s(st + sub * 8192, xk, nv * 128, mb + sl * 8);
                            } else {
                                bulk_g2s(st + sub * 8192, xk, nv * 64, mb + sl * 8);
                                bulk_g2s(st + sub * 8192 + 4096, xk + XB / 2, nv * 64, mb + sl * 8);
                            }
                        }""")
    # gg1 epilogue: n16 = one x2 stmatrix group
    s = _rep(s, """#pragma unroll
        for (int s = 0; s < NV / 32; s++) {
            unsigned r[4];""", """        if constexpr (NV == 16) {
            unsigned r[2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                float t0, t1;
                asm("tanh.approx.f32 %0, %1;" : "=f"(t0) : "f"(acc[4 * i] * hgs));
                asm("tanh.approx.f32 %0, %1;" : "=f"(t1) : "f"(acc[4 * i + 1] * hgs));
                const float p0 = acc[4 * i] * acc[4 * i + 2] * gus, p1 = acc[4 * i + 1] * acc[4 * i + 3] * gus;
                r[i] = b2u32(__floats2bfloat162_rn(fmaf(p0, t0, p0), fmaf(p1, t1, p1)));
            }
            stsm_x2_trans(lanebase, r[0], r[1]);
        }
#pragma unroll
        for (int s = 0; s < NV / 32; s++) {
            unsigned r[4];""")
    # gg2 epilogue: one partial pass below 64 tokens
    s = _rep(s, """#pragma unroll
        for (int p = 0; p < NV / 64; p++) {              // 64-token passes
            named_bar_sync(bar, 128);                 // previous pass's readers are done
#pragma unroll
            for (int s = 0; s < 4; s++) {                // 4 x4-groups = 16 matrices (8 i-blocks x 2 row halves)""",
             """        constexpr int NPS = (NV + 63) / 64, NSG = NV >= 64 ? 4 : NV / 16;   // passes, x4 groups per pass
#pragma unroll
        for (int p = 0; p < NPS; p++) {                  // 64-token passes (one partial pass below 64)
            named_bar_sync(bar, 128);                 // previous pass's readers are done
#pragma unroll
            for (int s = 0; s < NSG; s++) {              // 4 x4-groups = 16 matrices (8 i-blocks x 2 row halves)""")
    s = _rep(s, """#pragma unroll
            for (int cc = 0; cc < 4; cc++) {
                const int idx = cc * 128 + wtid;""", """#pragma unroll
            for (int cc = 0; cc < NSG; cc++) {
                const int idx = cc * 128 + wtid;""")
    s = _rep(s, """        const int nv = eff_nv<WSET>(wcode);
        if ((WSET & 1) && nv == 64)
            item_body<GATED, NST, MODE, 64, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel);""",
             """        const int nv = eff_nv<WSET>(wcode);
        if ((WSET & 16) && nv == 16)
            item_body<GATED, NST, MODE, 16, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel);
        else if ((WSET & 8) && nv == 32)
            item_body<GATED, NST, MODE, 32, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel);
        else if ((WSET & 1) && nv == 64)
            item_body<GATED, NST, MODE, 64, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel);""")
    # host: same entry points, compile-time width set
    s = _rep(s, "        (int)num_pid_m, (int)bm, (int)wmask, live.data_ptr<int>());",
             "        (int)num_pid_m, (int)bm, (int)PGN_WSET, live.data_ptr<int>());")
    s = _rep(s, "    pgk::pg_gather<3, 8><<<", "    pgk::pg_gather<PGN_WSET, 8><<<")
    s = _rep(s, "    launch<1, 5, 0, 3>(", "    launch<1, PGN_NST, 0, PGN_WSET>(")
    s = _rep(s, "    launch<0, 5, 0, 3>(", "    launch<0, PGN_NST, 0, PGN_WSET>(")
    # 128-token blocks
    s = _rep(s, "constexpr int XNT = 256;                 // tokens per tile (wgmma n)",
             "constexpr int XNT = 128;                 // tokens per block (narrow build: the widest item)")
    sb = [ln for ln in s.split("\n") if ln.startswith("constexpr int SB  = ")]
    if len(sb) != 1:
        raise RuntimeError("pg narrow build: no unique SB line in the shipped source")
    s = _rep(s, sb[0], "constexpr int SB  = (16384 + 2 * WSB + 1023) / 1024 * 1024;   // PAIR stage: X 2 x 8 KB + W|sf 2 x WSB")
    s = _rep(s, """            int code = 15;
            if (live_blk && bm == 256) {
                const int* sid = sorted_ids + (size_t)pm * 256;
                if ((wmask & 4) && sid[192] >= numel) code = 11;
                if ((wmask & 2) && sid[128] >= numel) code = 7;""", """            int code = 7;
            if (live_blk && bm == XNT) {
                const int* sid = sorted_ids + (size_t)pm * XNT;""")
    s = _rep(s, """    if ((WSET & 4) && need <= 192) return 192;
    return 256;""", """    return XNT;""")
    s = _rep(s, "template <int GATED, int NST, int MODE, int NV, int PAIR64, int ACC = 128, int N4 = 0>",
             "template <int GATED, int NST, int MODE, int NV, int PAIR64, int ACC = XNT / 2, int N4 = 0>")
    s = _rep(s, "        float acc[128];", "        float acc[XNT / 2];")
    s = _rep(s, """        else if ((WSET & 2) && nv == 128)
            item_body<GATED, NST, MODE, 128, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel);
        else if ((WSET & 4) && nv == 192)
            item_body<GATED, NST, MODE, 192, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel);
        else
            item_body<GATED, NST, MODE, 256, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel);""", """        else
            item_body<GATED, NST, MODE, XNT, PAIR64>(c, acc, bA, bB, pm, wt, gsv, rids, out, NWT, numel);""")
    return s
