"""Lazy loaders for the OKC inline CUDA extensions, each built once per box (~1-2 min of nvcc at first use).
pg and wst carry hand wgmma PTX, which needs the sm_90a feature target."""

import os
import threading

_lock = threading.Lock()
_PG_EXT = None
_PG_NARROW_EXT = None
_ST_EXT = None
_PROJ_EXT = None
_SMALLM_EXT = None
_FMHA_EXT = None
_GLUE_EXT = None
_WST_EXT = None
_VFG_EXT = None
_DRAFT_EXT = None

_PROJ_CPP = """
void gemv_bf16_launch(at::Tensor W, at::Tensor X, at::Tensor out, int64_t m, int64_t n,
                      int64_t k, int64_t MT, int64_t R, int64_t U, int64_t WPC);
void gemv_router_launch(at::Tensor Wp, at::Tensor X, at::Tensor out, int64_t m, int64_t k,
                        int64_t MT, int64_t SK, int64_t U, int64_t WPC);
void reduce_launch(at::Tensor P, at::Tensor out, int64_t m, int64_t nout, int64_t SK,
                   int64_t FOLD);
void proj_set_pdl(int64_t on);
"""


def glue_ext():
    global _GLUE_EXT
    if _GLUE_EXT is None:
        with _lock:
            if _GLUE_EXT is None:
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch import SW_PDL
                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_glue import _CUDA_SRC

                ext = load_inline(
                    name="okc_verify_glue_v1",
                    cpp_sources="void tail_boundary(torch::Tensor, torch::Tensor, torch::Tensor, double, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor);\n"
                    "void glue_set_pdl(int64_t);",
                    cuda_sources=_CUDA_SRC,
                    functions=["tail_boundary", "glue_set_pdl"],
                    extra_cuda_cflags=["-O3"],
                    verbose=False,
                )
                ext.glue_set_pdl(1 if SW_PDL else 0)
                _GLUE_EXT = ext
    return _GLUE_EXT


# Host entry points of _cuda_src_pg (and of the narrow build generated from it): T = torch::Tensor, i = int64_t,
# d = double.
_PG_SIG = {"align": "TTTTTTTTTiiii", "tiles": "TTTTiiiii", "rep_assign": "TTTTi", "rep_split": "TTTTTTii",
           "gather": "TTTTii", "route": "TTTTTTiiii", "gather_norm": "TTdTTTii", "gg1": "TTTTTTiiii",
           "gg2": "TTTTTTiiii", "set_snake_max_t": "i"}
_ARG = {"T": "torch::Tensor", "i": "int64_t", "d": "double"}


def _pg_load(name, src, fns, defines=()):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    from torch.utils.cpp_extension import load_inline

    cpp = "".join(f"void {f}({', '.join(_ARG[c] for c in _PG_SIG[f])});\n" for f in fns)
    return load_inline(
        name=name, cpp_sources=cpp, cuda_sources=src, functions=list(fns),
        extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a", *defines],
        verbose=False,
    )


def pg_ext():
    global _PG_EXT
    if _PG_EXT is None:
        with _lock:
            if _PG_EXT is None:
                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_pg import _CUDA_SRC

                # The name differs from the prefill bank's build and from the lanes' branches: the small-T front end,
                # the PDL launch flags, the local-slot align, tiles' local-slot count and the replica balancers.
                _PG_EXT = _pg_load("okc_pg_moe_c350_int_d3", _CUDA_SRC, _PG_SIG)
    return _PG_EXT


def pg_narrow_ext():
    """The narrow build of pg (_cuda_src_pgn: 128-token blocks, widths 16 / 64 / 128) for mid-T calls."""
    global _PG_NARROW_EXT
    if _PG_NARROW_EXT is None:
        with _lock:
            if _PG_NARROW_EXT is None:
                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_pgn import narrow_source

                _PG_NARROW_EXT = _pg_load(
                    "okc_pg_moe_n17x128s7_int_d3", narrow_source(),
                    ["tiles", "gather", "gg1", "gg2", "route", "gather_norm", "set_snake_max_t"],
                    ["-DPGN_WSET=17", "-DPGN_NST=7"],
                )
    return _PG_NARROW_EXT


def st_ext():
    """The st small-T MoE pair (okc_patch/st_moe.py): reads pg's packs, hand mma.sync PTX + griddepcontrol (sm_90a)."""
    global _ST_EXT
    if _ST_EXT is None:
        with _lock:
            if _ST_EXT is None:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch import _cuda_src_st as _st_src

                # bf16 scales and the 64-slot table (the lane's okc_st_moe_c1 reads 8-bit codes, 32 slots)
                _ST_EXT = load_inline(
                    name="okc_st_moe_c4_s16_e64",
                    cpp_sources="void st_moe(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, int64_t);",
                    cuda_sources=_st_src._CUDA_SRC,
                    functions=["st_moe"],
                    extra_cuda_cflags=[
                        "-O3",
                        "--use_fast_math",
                        "-gencode=arch=compute_90a,code=sm_90a",
                    ],
                    verbose=False,
                )
    return _ST_EXT


def proj_ext():
    global _PROJ_EXT
    if _PROJ_EXT is None:
        with _lock:
            if _PROJ_EXT is None:
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_proj import (
                    _CUDA_SRC,
                )

                # Flags match the certified kernel-track build; the name differs because the router GEMV and
                # reduce_k<2> changed (a shared name would rebuild over that lane's torch_extensions cache), and
                # again for the PDL launches (proj_set_pdl) and once more without the MMA-pack / 13-bit streams.
                _PROJ_EXT = load_inline(
                    name="okc_v5_stream_det_pdl_g",
                    cpp_sources=[_PROJ_CPP],
                    cuda_sources=[_CUDA_SRC],
                    functions=[
                        "gemv_bf16_launch",
                        "gemv_router_launch",
                        "reduce_launch",
                        "proj_set_pdl",
                    ],
                    extra_cuda_cflags=["-O3"],
                    verbose=False,
                )
                from sglang.kernels.ops.muse_spark_v12.okc_patch import SW_PDL

                _PROJ_EXT.proj_set_pdl(1 if SW_PDL else 0)
    return _PROJ_EXT


_FMHA_CPP = """
void fmha4n_split(torch::Tensor q, torch::Tensor kvt, torch::Tensor out, torch::Tensor po, torch::Tensor pml,
                  torch::Tensor lens, torch::Tensor k_scale, torch::Tensor v_scale, double smqk_log2e,
                  int64_t B, int64_t L, int64_t capblks, int64_t S, int64_t win, int64_t nc);
void combine(torch::Tensor po, torch::Tensor pml, torch::Tensor out, int64_t rows, int64_t S);
void gatherp(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
             torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
             int64_t capblks, int64_t B, int64_t L, int64_t win);
"""


def fmha_ext(exact: bool = False):
    """exact=True: the same kernels with every exp2 on MUFU (-DPFMHA_EXACT=1, its own build)."""
    global _FMHA_EXT
    key = bool(exact)
    if _FMHA_EXT is None or key not in _FMHA_EXT:
        with _lock:
            if _FMHA_EXT is None:
                _FMHA_EXT = {}
            if key not in _FMHA_EXT:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_fmha import (
                    _CUDA_SRC,
                )

                # The prefill lane's flags for this module (fmha7's TMA / wgmma PTX needs sm_90a).
                _FMHA_EXT[key] = load_inline(
                    name="okc_pfmha_c4b_exact" if key else "okc_pfmha_c4b",
                    cpp_sources=[_FMHA_CPP],
                    cuda_sources=[_CUDA_SRC],
                    functions=["fmha4n_split", "combine", "gatherp"],
                    extra_cuda_cflags=[
                        "-O3",
                        "--use_fast_math",
                        "-gencode=arch=compute_90a,code=sm_90a",
                    ]
                    + (["-DPFMHA_EXACT=1"] if key else []),
                    verbose=False,
                )
    return _FMHA_EXT[key]


def smallm_ext():
    global _SMALLM_EXT
    if _SMALLM_EXT is None:
        with _lock:
            if _SMALLM_EXT is None:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_smallm import (
                    _CUDA_SRC,
                )

                # TMA tensor maps (cuTensorMapEncodeTiled via cudaGetDriverEntryPoint) + wgmma: sm_90a.
                _SMALLM_EXT = load_inline(
                    name="okc_smallm_rmw_v1",
                    cpp_sources="void rmw_launch(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t);",
                    cuda_sources=[_CUDA_SRC],
                    functions=["rmw_launch"],
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                    extra_ldflags=["-lcuda"],
                    verbose=False,
                )
    return _SMALLM_EXT


_WST_CPP = """
void wst_attn_fqbt(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t,
                   int64_t, int64_t, int64_t, double, double, double, torch::Tensor, torch::Tensor, torch::Tensor);
void wst_attn_bal(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t,
                  int64_t, int64_t, int64_t, double, double, double, torch::Tensor, torch::Tensor, torch::Tensor);
"""


def wst_ext():
    """Verify-full attention (attention lane's wst_ct14 / wst_ct14b, q_len 8): see _cuda_src_wst.py. One build per
    partial precision (OKC_WST_ACC16), so only the configured one is compiled."""
    global _WST_EXT
    if _WST_EXT is None:
        with _lock:
            if _WST_EXT is None:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch import WST_ACC16
                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_wst import (
                    _CUDA_SRC,
                )

                # The bank's flags (v16c_m7g_rlx3 _wst_ext); a name of our own so the lane's cache is never reused.
                _WST_EXT = load_inline(
                    name="okc_wst_verify_v1_" + ("p16" if WST_ACC16 else "p32"),
                    cpp_sources=[_WST_CPP],
                    cuda_sources=[_CUDA_SRC],
                    functions=["wst_attn_fqbt", "wst_attn_bal"],
                    extra_cuda_cflags=[
                        "-O3",
                        "--use_fast_math",
                        "-gencode=arch=compute_90a,code=sm_90a",
                        "-DOKC_WST_ACC32=" + ("0" if WST_ACC16 else "1"),
                    ],
                    verbose=False,
                )
    return _WST_EXT


_VFG_CPP = """
void muse_vfg_launch(int64_t qkv, int64_t page_table, int64_t pt_stride, int64_t seq_lens, int64_t cache_loc,
                     int64_t k_cache, int64_t v_cache, int64_t inv_freq, int64_t out, int64_t B, int64_t NS,
                     int64_t window, double c2, double k_rscale, double v_rscale, double v_scale, double eps,
                     int64_t ws, int64_t dbgp, int64_t stream);
void muse_vp2_launch(int64_t qkv, int64_t page_table, int64_t pt_stride, int64_t seq_lens, int64_t cache_loc,
                     int64_t k_cache, int64_t v_cache, int64_t inv_freq, int64_t out, int64_t B, int64_t window,
                     double c2, double k_rscale, double v_rscale, double v_scale, double eps, int64_t dbgp,
                     int64_t stream);
"""


def vfg_ext():
    """Verify attention on the attention lane's VFG / VP2 kernels (q_len 8): see _cuda_src_vfg.py."""
    global _VFG_EXT
    if _VFG_EXT is None:
        with _lock:
            if _VFG_EXT is None:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_vfg import (
                    _CUDA_SRC,
                )

                # The bank's flags for its svsd module (v16c_m7g_rlx3 _svsd_ext); a name of our own.
                _VFG_EXT = load_inline(
                    name="okc_vfg_verify_v2",
                    cpp_sources=[_VFG_CPP],
                    cuda_sources=[_CUDA_SRC],
                    functions=["muse_vfg_launch", "muse_vp2_launch"],
                    extra_cuda_cflags=[
                        "-O3",
                        "-gencode=arch=compute_90a,code=sm_90a",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                    ],
                    verbose=False,
                )
    return _VFG_EXT


def draft_ext():
    """DFlash draft-side glue (_cuda_src_draft.py): candidate pack / merge, draft context-KV commit."""
    global _DRAFT_EXT
    if _DRAFT_EXT is None:
        with _lock:
            if _DRAFT_EXT is None:
                from torch.utils.cpp_extension import load_inline

                from sglang.kernels.ops.muse_spark_v12.okc_patch._cuda_src_draft import _CPP_SRC, _CUDA_SRC

                # No fast-math: cand_merge's tanh and products must match torch's float kernels.
                _DRAFT_EXT = load_inline(
                    name="okc_draft_glue_v1",
                    cpp_sources=_CPP_SRC,
                    cuda_sources=[_CUDA_SRC],
                    functions=["cand_pack", "cand_merge", "kv_write"],
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                    verbose=False,
                )
    return _DRAFT_EXT
