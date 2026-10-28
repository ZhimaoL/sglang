"""The small-T replica balancer, one device function shared by pg_rep_assign (_cuda_src_pg.py) and st's kernels
(_cuda_src_st.py), pasted into both sources at their @@REP@@ line so the two cannot drift."""

_CUDA_SRC_REP = r"""
#include <cuda_runtime.h>
// tab[e]: bits 0-7 expert e's slot on this rank (255: not held), 8-10 its home rank, bit 11 replicated, 12-14 the
// replica's rank; tab[256 + j] = the j-th replicated expert in balancing order. Same decision on every rank (and every
// CTA) from the same ids. An active expert weighs 1, or with passes its st passes ceil(routes / 8) (st's time is c0 +
// c1 x passes); load[r] = weight of the active non-replicated experts homed on r; each active replicated expert, in
// order, goes to the holder with the lower load (tie: home). Leaves owner[e] = the rank computing expert e's routes
// (the slot is tabs[e] & 0xFF there). One CTA of >= 256 threads; tabs / cnt / owner / ord shared [256], load [8]. The
// first NPRE ids per thread are loaded with the table's trip.
template <int NPRE, typename IdT>
__device__ __forceinline__ void rep_owner(const IdT* __restrict__ ids, int n_routes, const int* __restrict__ tab,
                                          int n_rep, int passes, int* tabs, int* cnt, int* owner, int* ord,
                                          int* load) {
    const int t = threadIdx.x, nt = blockDim.x;
    int v = 0, pre[NPRE];
    if (t < 256) v = tab[t];
    if (t < n_rep) ord[t] = tab[256 + t];
#pragma unroll
    for (int j = 0; j < NPRE; j++) pre[j] = t + j * nt < n_routes ? (int)ids[t + j * nt] : -1;
    if (t < 256) {
        tabs[t] = v;
        cnt[t] = 0;
        owner[t] = (v >> 8) & 7;
    }
    if (t < 8) load[t] = 0;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPRE; j++)
        if ((unsigned)pre[j] < 256u) atomicAdd(&cnt[pre[j]], 1);
    for (int i = t + NPRE * nt; i < n_routes; i += nt) {
        const int e = (int)ids[i];
        if ((unsigned)e < 256u) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (t < 256) {
        const int w = passes ? (cnt[t] + 7) >> 3 : (cnt[t] > 0);
        if (w && !((v >> 11) & 1)) atomicAdd(&load[(v >> 8) & 7], w);
    }
    __syncthreads();
    if (t == 0) {
        // 16 bits per rank in two registers (a passes-weighted load stays below 2^16); no local-memory arrays: this
        // serial loop sits on every rank's critical path. A batch's shared reads go first, then its serial steps.
        unsigned long long L0 = 0, L1 = 0;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            L0 |= (unsigned long long)load[r] << (16 * r);
            L1 |= (unsigned long long)load[4 + r] << (16 * r);
        }
        for (int j0 = 0; j0 < n_rep; j0 += 8) {
            int eb[8], cb[8], vb[8];
#pragma unroll
            for (int u = 0; u < 8; u++) eb[u] = j0 + u < n_rep ? ord[j0 + u] : 0;
#pragma unroll
            for (int u = 0; u < 8; u++) {
                cb[u] = j0 + u < n_rep ? cnt[eb[u]] : 0;
                vb[u] = tabs[eb[u]];
            }
#pragma unroll
            for (int u = 0; u < 8; u++) {
                const int e = eb[u], c = cb[u];
                if (!c) continue;
                const int ve = vb[u], h = (ve >> 8) & 7, r = (ve >> 12) & 7;
                const unsigned lh = (unsigned)(((h & 4) ? L1 : L0) >> (16 * (h & 3))) & 0xFFFFu;
                const unsigned lr = (unsigned)(((r & 4) ? L1 : L0) >> (16 * (r & 3))) & 0xFFFFu;
                const int pick = lr < lh ? r : h;
                owner[e] = pick;
                const unsigned long long add = (unsigned long long)(passes ? (c + 7) >> 3 : 1) << (16 * (pick & 3));
                if (pick & 4) L1 += add; else L0 += add;
            }
        }
    }
    __syncthreads();
}
"""
