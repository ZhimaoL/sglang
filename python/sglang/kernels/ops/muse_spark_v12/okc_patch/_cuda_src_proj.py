"""muse_bf16_proj v5 CUDA source, byte-identical to the certified kernel-track lane
(same load_inline name and flags -> torch_extensions build-cache hit)."""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

// bf16 pair (packed in a u32) -> two fp32, 1 int op each.
__device__ __forceinline__ float blo(unsigned u) { return __int_as_float(u << 16); }
__device__ __forceinline__ float bhi(unsigned u) { return __int_as_float(u & 0xffff0000u); }

__device__ __forceinline__ float dot8(uint4 w, uint4 x, float acc) {
  acc = fmaf(blo(w.x), blo(x.x), acc);
  acc = fmaf(bhi(w.x), bhi(x.x), acc);
  acc = fmaf(blo(w.y), blo(x.y), acc);
  acc = fmaf(bhi(w.y), bhi(x.y), acc);
  acc = fmaf(blo(w.z), blo(x.z), acc);
  acc = fmaf(bhi(w.z), bhi(x.z), acc);
  acc = fmaf(blo(w.w), blo(x.w), acc);
  acc = fmaf(bhi(w.w), bhi(x.w), acc);
  return acc;
}

// out[i, n] = sum_k x[i, k] * W[n, k];  warp owns rows [wid*R, wid*R+R), full K.
// U = k-steps per load group (all U*R W loads + U*MT x loads issued back-to-back
// for latency hiding); requires kvec % (32*U) == 0 (python-guarded).
template <int MT, int R, int U>
__global__ void gemv_ffma_k(const uint4* __restrict__ W, const uint4* __restrict__ X,
                            __nv_bfloat16* __restrict__ out, int m, int N, int kvec) {
  int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  size_t row0 = (size_t)wid * R;
  if (row0 >= (size_t)N) return;
  const uint4* wp = W + row0 * kvec;
  float acc[2][R][MT];
#pragma unroll
  for (int p = 0; p < 2; p++)
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
      for (int i = 0; i < MT; i++) acc[p][r][i] = 0.f;

  for (int kv = lane; kv < kvec; kv += 32 * U) {
    uint4 wv[U][R], xf[U][MT];
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int r = 0; r < R; r++) wv[u][r] = __ldcs(wp + (size_t)r * kvec + kv + 32 * u);
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int i = 0; i < MT; i++)
        xf[u][i] = (MT == 1 || i < m) ? __ldg(X + (size_t)i * kvec + kv + 32 * u)
                                      : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int i = 0; i < MT; i++)
          acc[u & 1][r][i] = dot8(wv[u][r], xf[u][i], acc[u & 1][r][i]);
  }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int i = 0; i < MT; i++) {
      float v = acc[0][r][i] + acc[1][r][i];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == i && (MT == 1 || i < m))
        out[(size_t)i * N + row0 + r] = __float2bfloat16(v);
    }
}

// Router: Wp = [2*NR, K] bf16, rows 2j / 2j+1 = hi_j / lo_j. Warp (j, sk) owns the
// pair over K-slice sk; fp32 atomicAdd into pre-zeroed out[m, NR].
template <int MT, int U>
__global__ void gemv_router_k(const uint4* __restrict__ Wp, const uint4* __restrict__ X,
                              float* __restrict__ out, int m, int NR, int kvec, int SK) {
  int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  int j = wid / SK, sk = wid - j * SK;
  if (j >= NR) return;
  int kchunk = kvec / SK;
  int k0 = sk * kchunk;
  const uint4* hp = Wp + (size_t)(2 * j) * kvec + k0;
  const uint4* lp = hp + kvec;
  float acc[2][MT];
#pragma unroll
  for (int p = 0; p < 2; p++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[p][i] = 0.f;

  for (int kv = lane; kv < kchunk; kv += 32 * U) {
    uint4 hv[U], lv[U], xf[U][MT];
#pragma unroll
    for (int u = 0; u < U; u++) {
      hv[u] = __ldcs(hp + kv + 32 * u);
      lv[u] = __ldcs(lp + kv + 32 * u);
    }
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int i = 0; i < MT; i++)
        xf[u][i] = (MT == 1 || i < m) ? __ldg(X + (size_t)i * kvec + k0 + kv + 32 * u)
                                      : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int i = 0; i < MT; i++)
        acc[u & 1][i] = dot8(lv[u], xf[u][i], dot8(hv[u], xf[u][i], acc[u & 1][i]));
  }
#pragma unroll
  for (int i = 0; i < MT; i++) {
    float v = acc[0][i] + acc[1][i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == i && (MT == 1 || i < m)) atomicAdd(out + (size_t)i * NR + j, v);
  }
}

// ------------------------------------------------------------------ mma stream
// One warp owns an 8-column strip of out (n0 = ncol*8) over K/SK. W is PREPACKED
// (prepare-time) so each 64-k group of a strip is one contiguous 1KB block holding
// the exact mma B-fragments: lane t reads 2 uint4 = b-pairs for 4 mmas. x is staged
// per-CTA in shared (bf16, zero-padded rows >= m) and read via ldmatrix.
__device__ __forceinline__ void mma16816(float* c, unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldm_x4(unsigned& a0, unsigned& a1, unsigned& a2,
                                       unsigned& a3, unsigned saddr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
               : "r"(saddr));
}
__device__ __forceinline__ void ldm_x2(unsigned& a0, unsigned& a2, unsigned saddr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(a0), "=r"(a2)
               : "r"(saddr));
}
__device__ __forceinline__ void cp16(__nv_bfloat16* smem, const __nv_bfloat16* gmem) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

// MR: padded A rows (8 / 16 / 32). SPLIT: 0 -> bf16 direct store; 1 -> fp32 partials
// P[sk][m][NOUT]. G: 64k-groups in flight. A-fragments come straight from x via
// predicated __ldg u32 loads: x is tiny (<= 512KB) and the same addresses are read
// by every n-strip warp, so after first touch they are L1/L2-hot; W (__ldcs,
// evict-first) never displaces them.
template <int MR, int SPLIT, int G>
__global__ void __launch_bounds__(256, 1)
mma_stream_k(const uint4* __restrict__ Wp, const __nv_bfloat16* __restrict__ X,
             void* __restrict__ out, int m, int NOUT, int K, int SK) {
  extern __shared__ __nv_bfloat16 xs[];  // [MR][kslice + 8] (pad -> 16B ldmatrix skew)
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int WPC = blockDim.x >> 5;
  const int nb = blockIdx.x / SK, sk = blockIdx.x - nb * SK;
  const int ncol = nb * WPC + warp;
  const int kslice = K / SK, kstart = sk * kslice;
  const int g = lane >> 2, cc = lane & 3;
  const int SP = kslice + 8;

  float c0[4] = {0.f, 0.f, 0.f, 0.f};
  float c1[4] = {0.f, 0.f, 0.f, 0.f};  // MR==32 second m-tile

  {  // stage x[0..MR)[kstart..kstart+kslice) once: cp.async (deep HW queue, no
     // per-thread load->store register chains); zero-pad rows >= m with plain stores
    const int nu4 = kslice >> 3;
    for (int idx = threadIdx.x; idx < MR * nu4; idx += blockDim.x) {
      int r = idx / nu4, kq = idx - r * nu4;
      if (r < m)
        cp16(xs + (size_t)r * SP + (kq << 3), X + (size_t)r * K + kstart + (kq << 3));
      else
        *reinterpret_cast<uint4*>(xs + (size_t)r * SP + (kq << 3)) =
            make_uint4(0u, 0u, 0u, 0u);
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
  }
  unsigned sa_lo, sa_hi = 0;
  if (MR == 8)
    sa_lo = (unsigned)__cvta_generic_to_shared(xs + (lane & 7) * SP + ((lane >> 3) & 1) * 8);
  else {
    sa_lo = (unsigned)__cvta_generic_to_shared(xs + (lane & 15) * SP + (lane >> 4) * 8);
    if (MR == 32)
      sa_hi = (unsigned)__cvta_generic_to_shared(xs + ((lane & 15) + 16) * SP + (lane >> 4) * 8);
  }

  const uint4* wb = Wp + (((size_t)ncol * (K >> 6) + (kstart >> 6)) << 6) + (lane << 1);
  for (int kk = 0; kk < kslice; kk += 64 * G) {
    uint4 u[G][2];
#pragma unroll
    for (int gg = 0; gg < G; gg++) {
      const uint4* p = wb + (((kk >> 6) + gg) << 6);
      u[gg][0] = __ldcs(p);
      u[gg][1] = __ldcs(p + 1);
    }
#pragma unroll
    for (int gg = 0; gg < G; gg++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        unsigned b0, b1;
        if (q == 0) { b0 = u[gg][0].x; b1 = u[gg][0].y; }
        else if (q == 1) { b0 = u[gg][0].z; b1 = u[gg][0].w; }
        else if (q == 2) { b0 = u[gg][1].x; b1 = u[gg][1].y; }
        else { b0 = u[gg][1].z; b1 = u[gg][1].w; }
        const unsigned koff = (unsigned)((kk + gg * 64 + q * 16) * 2);
        unsigned a0, a1 = 0u, a2, a3 = 0u;
        if (MR == 8) ldm_x2(a0, a2, sa_lo + koff);
        else ldm_x4(a0, a1, a2, a3, sa_lo + koff);
        mma16816(c0, a0, a1, a2, a3, b0, b1);
        if (MR == 32) {
          unsigned e0, e1, e2, e3;
          ldm_x4(e0, e1, e2, e3, sa_hi + koff);
          mma16816(c1, e0, e1, e2, e3, b0, b1);
        }
      }
  }
  // epilogue: D rows (g, g+8[, g+16, g+24]), cols n0 + 2*cc + {0,1}
  const int col = ncol * 8 + 2 * cc;
  if (SPLIT == 0) {
    __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(out);
    if (g < m)
      *reinterpret_cast<__nv_bfloat162*>(o + (size_t)g * NOUT + col) =
          __float22bfloat162_rn(make_float2(c0[0], c0[1]));
    if (g + 8 < m)
      *reinterpret_cast<__nv_bfloat162*>(o + (size_t)(g + 8) * NOUT + col) =
          __float22bfloat162_rn(make_float2(c0[2], c0[3]));
    if (MR == 32) {
      if (g + 16 < m)
        *reinterpret_cast<__nv_bfloat162*>(o + (size_t)(g + 16) * NOUT + col) =
            __float22bfloat162_rn(make_float2(c1[0], c1[1]));
      if (g + 24 < m)
        *reinterpret_cast<__nv_bfloat162*>(o + (size_t)(g + 24) * NOUT + col) =
            __float22bfloat162_rn(make_float2(c1[2], c1[3]));
    }
  } else {
    float* P = reinterpret_cast<float*>(out) + (size_t)sk * m * NOUT;
    if (g < m) *reinterpret_cast<float2*>(P + (size_t)g * NOUT + col) = make_float2(c0[0], c0[1]);
    if (g + 8 < m)
      *reinterpret_cast<float2*>(P + (size_t)(g + 8) * NOUT + col) = make_float2(c0[2], c0[3]);
    if (MR == 32) {
      if (g + 16 < m)
        *reinterpret_cast<float2*>(P + (size_t)(g + 16) * NOUT + col) = make_float2(c1[0], c1[1]);
      if (g + 24 < m)
        *reinterpret_cast<float2*>(P + (size_t)(g + 24) * NOUT + col) = make_float2(c1[2], c1[3]);
    }
  }
}

// 13-bit compressed variant: mant plane Wm = per-(8-row strip, 64k) 512B blocks
// (lane-major 16B, byte j = sign<<7|mant7 of weight j in B-fragment order); exp
// plane We = per-(strip, 512k) blocks of 32 lanes x 80B: 128 5-bit codes as an LE
// bitstream in consumption order. bf16 bits = (mb&0x80)<<8 | e8<<7 | (mb&0x7f),
// e8 = v ? v + EB : 0 (v==0 encodes e==0: exact zeros / denormals).
template <int MR, int SPLIT>
__global__ void __launch_bounds__(256, 1)
mma_cstream_k(const uint4* __restrict__ Wm, const uint4* __restrict__ We,
              const __nv_bfloat16* __restrict__ X, void* __restrict__ out, int m,
              int NOUT, int K, int SK, int EB) {
  extern __shared__ char smem[];
  unsigned* lutE = reinterpret_cast<unsigned*>(smem);  // 1024 x u32: (e0<<7 | e1<<23)
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem + 4096);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int WPC = blockDim.x >> 5;
  const int nb = blockIdx.x / SK, sk = blockIdx.x - nb * SK;
  const int ncol = nb * WPC + warp;
  const int kslice = K / SK, kstart = sk * kslice;
  const int g = lane >> 2, cc = lane & 3;
  const int SP = kslice + 8;

  float c0[4] = {0.f, 0.f, 0.f, 0.f};
  float c1[4] = {0.f, 0.f, 0.f, 0.f};

  {
    const int nu4 = kslice >> 3;
    for (int idx = threadIdx.x; idx < MR * nu4; idx += blockDim.x) {
      int r = idx / nu4, kq = idx - r * nu4;
      if (r < m)
        cp16(xs + (size_t)r * SP + (kq << 3), X + (size_t)r * K + kstart + (kq << 3));
      else
        *reinterpret_cast<uint4*>(xs + (size_t)r * SP + (kq << 3)) =
            make_uint4(0u, 0u, 0u, 0u);
    }
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
      unsigned v0 = i & 31u, v1 = (unsigned)i >> 5;
      lutE[i] = ((v0 ? v0 + (unsigned)EB : 0u) << 7) | ((v1 ? v1 + (unsigned)EB : 0u) << 23);
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
  }
  unsigned sa_lo, sa_hi = 0;
  if (MR == 8)
    sa_lo = (unsigned)__cvta_generic_to_shared(xs + (lane & 7) * SP + ((lane >> 3) & 1) * 8);
  else {
    sa_lo = (unsigned)__cvta_generic_to_shared(xs + (lane & 15) * SP + (lane >> 4) * 8);
    if (MR == 32)
      sa_hi = (unsigned)__cvta_generic_to_shared(xs + ((lane & 15) + 16) * SP + (lane >> 4) * 8);
  }

  for (int kk = 0; kk < kslice; kk += 512) {
    const size_t g64b = (size_t)ncol * (K >> 6) + ((kstart + kk) >> 6);
    const size_t g512 = (size_t)ncol * (K >> 9) + ((kstart + kk) >> 9);
    uint4 um[8], ue4[5];
#pragma unroll
    for (int gg = 0; gg < 8; gg++) um[gg] = __ldcs(Wm + (g64b + gg) * 32 + lane);
#pragma unroll
    for (int j = 0; j < 5; j++) ue4[j] = __ldcs(We + g512 * 160 + j * 32 + lane);
    unsigned ew[20];
#pragma unroll
    for (int j = 0; j < 5; j++) {
      ew[4 * j] = ue4[j].x;
      ew[4 * j + 1] = ue4[j].y;
      ew[4 * j + 2] = ue4[j].z;
      ew[4 * j + 3] = ue4[j].w;
    }
#pragma unroll
    for (int gg = 0; gg < 8; gg++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        unsigned b[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
          const int p = q * 2 + i;
          const int off = (gg * 8 + p) * 10, wd = off >> 5, sh = off & 31;
          unsigned v10;
          if (sh <= 22)
            v10 = (ew[wd] >> sh) & 0x3FFu;
          else
            v10 = __funnelshift_r(ew[wd], ew[wd + 1], sh) & 0x3FFu;
          const unsigned mw = (p < 2) ? um[gg].x : (p < 4) ? um[gg].y
                              : (p < 6) ? um[gg].z : um[gg].w;
          const unsigned M = __byte_perm(mw, 0u, (p & 1) ? 0x7372u : 0x7170u);
          b[i] = ((M & 0x00800080u) << 8) | lutE[v10] | (M & 0x007F007Fu);
        }
        const unsigned koff = (unsigned)((kk + gg * 64 + q * 16) * 2);
        unsigned a0, a1 = 0u, a2, a3 = 0u;
        if (MR == 8) ldm_x2(a0, a2, sa_lo + koff);
        else ldm_x4(a0, a1, a2, a3, sa_lo + koff);
        mma16816(c0, a0, a1, a2, a3, b[0], b[1]);
        if (MR == 32) {
          unsigned e0_, e1_, e2_, e3_;
          ldm_x4(e0_, e1_, e2_, e3_, sa_hi + koff);
          mma16816(c1, e0_, e1_, e2_, e3_, b[0], b[1]);
        }
      }
  }
  const int col = ncol * 8 + 2 * cc;
  if (SPLIT == 0) {
    __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(out);
    if (g < m)
      *reinterpret_cast<__nv_bfloat162*>(o + (size_t)g * NOUT + col) =
          __float22bfloat162_rn(make_float2(c0[0], c0[1]));
    if (g + 8 < m)
      *reinterpret_cast<__nv_bfloat162*>(o + (size_t)(g + 8) * NOUT + col) =
          __float22bfloat162_rn(make_float2(c0[2], c0[3]));
    if (MR == 32) {
      if (g + 16 < m)
        *reinterpret_cast<__nv_bfloat162*>(o + (size_t)(g + 16) * NOUT + col) =
            __float22bfloat162_rn(make_float2(c1[0], c1[1]));
      if (g + 24 < m)
        *reinterpret_cast<__nv_bfloat162*>(o + (size_t)(g + 24) * NOUT + col) =
            __float22bfloat162_rn(make_float2(c1[2], c1[3]));
    }
  } else {
    float* P = reinterpret_cast<float*>(out) + (size_t)sk * m * NOUT;
    if (g < m) *reinterpret_cast<float2*>(P + (size_t)g * NOUT + col) = make_float2(c0[0], c0[1]);
    if (g + 8 < m)
      *reinterpret_cast<float2*>(P + (size_t)(g + 8) * NOUT + col) = make_float2(c0[2], c0[3]);
    if (MR == 32) {
      if (g + 16 < m)
        *reinterpret_cast<float2*>(P + (size_t)(g + 16) * NOUT + col) = make_float2(c1[0], c1[1]);
      if (g + 24 < m)
        *reinterpret_cast<float2*>(P + (size_t)(g + 24) * NOUT + col) = make_float2(c1[2], c1[3]);
    }
  }
}

// FOLD=0: out bf16 [m,NOUT] = sum_sk P. FOLD=1 (router): out fp32 [m,NOUT/2],
// out[i][j] = sum_sk P[sk][i][j] + P[sk][i][j+NOUT/2].
template <int FOLD>
__global__ void reduce_k(const float* __restrict__ P, void* __restrict__ out, int m, int NOUT,
                         int SK) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t plane = (size_t)m * NOUT;
  if (FOLD == 0) {
    int total = (int)(plane >> 1);
    if (tid >= total) return;
    const float2* p2 = reinterpret_cast<const float2*>(P);
    float2 s = make_float2(0.f, 0.f);
    for (int si = 0; si < SK; si++) {
      float2 v = p2[(plane >> 1) * si + tid];
      s.x += v.x;
      s.y += v.y;
    }
    reinterpret_cast<__nv_bfloat162*>(out)[tid] = __float22bfloat162_rn(s);
  } else {
    int half = NOUT >> 1;
    int total = m * half;
    if (tid >= total) return;
    int i = tid / half, j = tid - i * half;
    float s = 0.f;
    for (int si = 0; si < SK; si++) {
      const float* p = P + plane * si + (size_t)i * NOUT;
      s += p[j] + p[j + half];
    }
    reinterpret_cast<float*>(out)[(size_t)i * half + j] = s;
  }
}

void gemv_bf16_launch(at::Tensor W, at::Tensor X, at::Tensor out, int64_t m, int64_t n,
                      int64_t k, int64_t MT, int64_t R, int64_t U, int64_t WPC) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int kvec = (int)(k >> 3);
  dim3 grid((unsigned)(n / R / WPC)), block((unsigned)(WPC * 32));
  const uint4* w = reinterpret_cast<const uint4*>(W.data_ptr());
  const uint4* x = reinterpret_cast<const uint4*>(X.data_ptr());
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define GCASE(mt, r, u)                                                               \
  if (MT == mt && R == r && U == u) {                                                 \
    gemv_ffma_k<mt, r, u><<<grid, block, 0, stream>>>(w, x, o, (int)m, (int)n, kvec); \
    return;                                                                           \
  }
  GCASE(1, 1, 2) GCASE(1, 1, 4) GCASE(1, 1, 8)
  TORCH_CHECK(false, "no gemv instantiation");
}

void mma_launch(at::Tensor Wp, at::Tensor X, at::Tensor out, int64_t m, int64_t nout,
                int64_t k, int64_t MR, int64_t SPLIT, int64_t SK, int64_t WPC, int64_t G) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)((nout / 8 / WPC) * SK)), block((unsigned)(WPC * 32));
  const uint4* w = reinterpret_cast<const uint4*>(Wp.data_ptr());
  const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
  void* o = out.data_ptr();
  int sh = (int)(MR * (k / SK + 8) * 2);
#define MCASE(mr, sp, g)                                                                   \
  if (MR == mr && SPLIT == sp && G == g) {                                                 \
    cudaFuncSetAttribute(mma_stream_k<mr, sp, g>,                                          \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 135168);             \
    mma_stream_k<mr, sp, g><<<grid, block, sh, stream>>>(w, x, o, (int)m, (int)nout,       \
                                                         (int)k, (int)SK);                 \
    return;                                                                                \
  }
  MCASE(8, 0, 2) MCASE(8, 0, 4) MCASE(8, 0, 8) MCASE(8, 1, 2) MCASE(8, 1, 4) MCASE(8, 1, 8)
  MCASE(16, 0, 2) MCASE(16, 0, 4) MCASE(16, 1, 2) MCASE(16, 1, 4)
  MCASE(32, 0, 2) MCASE(32, 0, 4) MCASE(32, 1, 2) MCASE(32, 1, 4)
  TORCH_CHECK(false, "no mma instantiation");
}

void cmma_launch(at::Tensor Wm, at::Tensor We, at::Tensor X, at::Tensor out, int64_t m,
                 int64_t nout, int64_t k, int64_t MR, int64_t SPLIT, int64_t SK,
                 int64_t WPC, int64_t EB) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)((nout / 8 / WPC) * SK)), block((unsigned)(WPC * 32));
  const uint4* wm = reinterpret_cast<const uint4*>(Wm.data_ptr());
  const uint4* we = reinterpret_cast<const uint4*>(We.data_ptr());
  const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
  void* o = out.data_ptr();
  int sh = (int)(MR * (k / SK + 8) * 2) + 4096;
#define CCASE(mr, sp)                                                                      \
  if (MR == mr && SPLIT == sp) {                                                           \
    cudaFuncSetAttribute(mma_cstream_k<mr, sp>,                                            \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 139264);             \
    mma_cstream_k<mr, sp><<<grid, block, sh, stream>>>(wm, we, x, o, (int)m, (int)nout,    \
                                                       (int)k, (int)SK, (int)EB);          \
    return;                                                                                \
  }
  CCASE(8, 0) CCASE(8, 1) CCASE(16, 0) CCASE(16, 1) CCASE(32, 0) CCASE(32, 1)
  TORCH_CHECK(false, "no cmma instantiation");
}

void reduce_launch(at::Tensor P, at::Tensor out, int64_t m, int64_t nout, int64_t SK,
                   int64_t FOLD) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int total = (int)(FOLD ? m * (nout >> 1) : (m * nout) >> 1);
  dim3 grid((unsigned)((total + 255) / 256)), block(256u);
  const float* p = reinterpret_cast<const float*>(P.data_ptr());
  if (FOLD)
    reduce_k<1><<<grid, block, 0, stream>>>(p, out.data_ptr(), (int)m, (int)nout, (int)SK);
  else
    reduce_k<0><<<grid, block, 0, stream>>>(p, out.data_ptr(), (int)m, (int)nout, (int)SK);
}

void gemv_router_launch(at::Tensor Wp, at::Tensor X, at::Tensor out, int64_t m, int64_t k,
                        int64_t MT, int64_t SK, int64_t U, int64_t WPC) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int kvec = (int)(k >> 3);
  int NR = (int)(Wp.size(0) / 2);
  dim3 grid((unsigned)((int64_t)NR * SK / WPC)), block((unsigned)(WPC * 32));
  const uint4* w = reinterpret_cast<const uint4*>(Wp.data_ptr());
  const uint4* x = reinterpret_cast<const uint4*>(X.data_ptr());
  auto* o = reinterpret_cast<float*>(out.data_ptr());
#define RCASE(mt, u)                                                                        \
  if (MT == mt && U == u) {                                                                 \
    gemv_router_k<mt, u><<<grid, block, 0, stream>>>(w, x, o, (int)m, NR, kvec, (int)SK);   \
    return;                                                                                 \
  }
  RCASE(1, 1) RCASE(1, 2) RCASE(1, 4)
  TORCH_CHECK(false, "no router instantiation");
}
"""

_CPP_SRC = """
void gemv_bf16_launch(at::Tensor W, at::Tensor X, at::Tensor out, int64_t m, int64_t n,
                      int64_t k, int64_t MT, int64_t R, int64_t U, int64_t WPC);
void gemv_router_launch(at::Tensor Wp, at::Tensor X, at::Tensor out, int64_t m, int64_t k,
                        int64_t MT, int64_t SK, int64_t U, int64_t WPC);
void mma_launch(at::Tensor Wp, at::Tensor X, at::Tensor out, int64_t m, int64_t nout,
                int64_t k, int64_t MR, int64_t SPLIT, int64_t SK, int64_t WPC, int64_t G);
void cmma_launch(at::Tensor Wm, at::Tensor We, at::Tensor X, at::Tensor out, int64_t m,
                 int64_t nout, int64_t k, int64_t MR, int64_t SPLIT, int64_t SK,
                 int64_t WPC, int64_t EB);
void reduce_launch(at::Tensor P, at::Tensor out, int64_t m, int64_t nout, int64_t SK,
                   int64_t FOLD);
"""
