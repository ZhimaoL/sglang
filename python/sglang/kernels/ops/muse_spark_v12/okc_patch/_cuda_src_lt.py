"""cuBLASLt entries for okc_patch/lt.py: descriptors + configured algo built and AlgoCheck'd once (lt_register), then
cublasLtMatmul only (lt_mm). Layout 0: col-major, A = W (op T), B = x; 1: row order, A = x, B = W (op T)."""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <deque>
#include <mutex>
#define OK(x) ((x) == CUBLAS_STATUS_SUCCESS)
struct E { int64_t m, n, k, l; cublasLtMatmulDesc_t op = nullptr; cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
           cublasLtMatmulAlgo_t algo; size_t ws = 0; };
static std::mutex mu;
static std::deque<E> es;
static void drop(E& e) {
  for (auto l : {e.a, e.b, e.d}) if (l) cublasLtMatrixLayoutDestroy(l);
  if (e.op) cublasLtMatmulDescDestroy(e.op);
}
static bool mk(E& e) {
  bool r = OK(cublasLtMatmulDescCreate(&e.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
  if (e.l == 1) {
    r = r && OK(cublasLtMatrixLayoutCreate(&e.a, CUDA_R_16BF, e.m, e.k, e.k)) &&
        OK(cublasLtMatrixLayoutCreate(&e.b, CUDA_R_16BF, e.n, e.k, e.k)) &&
        OK(cublasLtMatrixLayoutCreate(&e.d, CUDA_R_16BF, e.m, e.n, e.n));
    cublasLtOrder_t o = CUBLASLT_ORDER_ROW;
    for (auto l : {e.a, e.b, e.d}) r = r && OK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_ORDER, &o, sizeof(o)));
    std::swap(t, n);
  } else {
    r = r && OK(cublasLtMatrixLayoutCreate(&e.a, CUDA_R_16BF, e.k, e.n, e.k)) &&
        OK(cublasLtMatrixLayoutCreate(&e.b, CUDA_R_16BF, e.k, e.m, e.k)) &&
        OK(cublasLtMatrixLayoutCreate(&e.d, CUDA_R_16BF, e.n, e.m, e.n));
  }
  return r && OK(cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t))) &&
         OK(cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_TRANSB, &n, sizeof(n)));
}
int64_t lt_version() { return (int64_t)cublasLtGetVersion(); }
// cfg = {algo, tile, splitk, reduction, swizzle, custom, stages, inner, cluster} -> {index (< 0: rejected), ws, align A, B}
std::vector<int64_t> lt_register(int64_t m, int64_t n, int64_t k, int64_t l, std::vector<int64_t> c, int64_t ws_max) {
  E e;
  e.m = m; e.n = n; e.k = k; e.l = l;
  auto h = at::cuda::getCurrentCUDABlasLtHandle();
  bool r = c.size() == 9 && mk(e) && OK(cublasLtMatmulAlgoInit(h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                                                              CUDA_R_16BF, CUDA_R_16BF, (int)c[0], &e.algo));
  for (int i = 1; r && i < 9; i++) {
    uint32_t v32 = (uint32_t)c[i];
    uint16_t v16 = (uint16_t)c[i];
    auto at = (cublasLtMatmulAlgoConfigAttributes_t)i;
    r = i >= 7 ? OK(cublasLtMatmulAlgoConfigSetAttribute(&e.algo, at, &v16, 2))
               : OK(cublasLtMatmulAlgoConfigSetAttribute(&e.algo, at, &v32, 4));
  }
  cublasLtMatmulHeuristicResult_t res = {};
  r = r && OK(cublasLtMatmulAlgoCheck(h, e.op, e.a, e.b, e.d, e.d, &e.algo, &res)) && OK(res.state) &&
      (int64_t)res.workspaceSize <= ws_max;
  if (!r) { drop(e); return {-1, 0, 0, 0}; }
  e.ws = res.workspaceSize;
  uint32_t al[2] = {0, 0};
  size_t w;
  cublasLtMatmulAlgoCapGetAttribute(&e.algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES, &al[0], 4, &w);
  cublasLtMatmulAlgoCapGetAttribute(&e.algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES, &al[1], 4, &w);
  std::lock_guard<std::mutex> g(mu);
  es.push_back(e);
  return {(int64_t)es.size() - 1, (int64_t)e.ws, al[0], al[1]};
}
// out[m][n] = x @ W^T on the current stream; false = refused, nothing enqueued
bool lt_mm(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, int64_t i) {
  const E* e;
  { std::lock_guard<std::mutex> g(mu); TORCH_CHECK(i >= 0 && i < (int64_t)es.size()); e = &es[i]; }
  TORCH_CHECK(x.size(0) == e->m && x.size(1) == e->k && w.size(0) == e->n && out.size(1) == e->n);
  const float one = 1.f, zero = 0.f;
  const void* A = e->l == 1 ? x.data_ptr() : w.data_ptr();
  const void* B = e->l == 1 ? w.data_ptr() : x.data_ptr();
  return OK(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), e->op, &one, A, e->a, B, e->b, &zero, out.data_ptr(),
                           e->d, out.data_ptr(), e->d, &e->algo, e->ws ? ws.data_ptr() : nullptr, e->ws,
                           at::cuda::getCurrentCUDAStream()));
}
"""
