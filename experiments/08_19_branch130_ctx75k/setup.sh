#!/usr/bin/env bash
# setup.sh -- per-node runtime environment for this experiment. Source it; do not execute it.
#
#   source setup.sh cuda-compat     before activating the virtual environment (cluster A only)
#   source setup.sh cuda-toolkit    after activating the virtual environment
#
# The launch scripts source it in every compute-node process: the Ray daemons started by
# submit_*.sbatch and the driver (and the flashinfer warm-up) started by run_attach_*.sh.
case "${1:-}" in
cuda-compat)
# before activating the venv. Single place that makes this cu129 stack work on cluster A.
#
# (1) MIXED DRIVERS. One allocation can span nodes with different drivers (e.g. 580.x /
#     CUDA 13.0 on some nodes, 565.x / CUDA 12.7 on others).
#     On the 565 nodes a plain torch op dies with "CUDA error: the provided PTX was compiled
#     with an unsupported toolchain" unless the CUDA 12.9 forward-compat driver libs are on
#     LD_LIBRARY_PATH. Those libs are 575.57.08 and forward-compat only applies when the KERNEL
#     driver is OLDER, so on the 580 node they break CUDA outright (torch.cuda.is_available()
#     False), which surfaces later as verl building "cpu:gloo,cpu:nccl" -> "ValueError:
#     Duplicate device type cpu". Neither always-on nor always-off works: decide per node.
#
# (2) JIT HEADERS. flashinfer/DeepGEMM JIT-compile at engine init and need the CUDA C++ core
#     headers; without them the judge dies with "fatal error: nv/target: No such file or
#     directory". They ship in the VENV (nvidia/cuda_cccl/include), which the cuda-toolkit section does not
#     add -- its list covers cublas/curand/cufft/cusolver/cusparse/nvrtc/nvjitlink only.
CLUSTER_A_COMPAT=${AC2_CLUSTER_A_CUDA_COMPAT}
_drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ -n "$_drv" ] && [ "$_drv" -lt 575 ] 2>/dev/null && [ -f "$CLUSTER_A_COMPAT/libcuda.so.1" ]; then
  export LD_LIBRARY_PATH="$CLUSTER_A_COMPAT:${LD_LIBRARY_PATH:-}"
  echo "[cuda-shim] $(hostname): driver $_drv < 575 -> CUDA 12.9 forward-compat libs"
else
  echo "[cuda-shim] $(hostname): driver ${_drv:-unknown} -> native CUDA (no compat)"
fi

_CH=/usr/local/cuda-12.9
[ -x "$_CH/bin/nvcc" ] || _CH=/usr/local/cuda
export CUDA_HOME="$_CH" CUDA_PATH="$_CH" CUDACXX="$_CH/bin/nvcc" TRTLLM_DG_NVCC_COMPILER="$_CH/bin/nvcc"
case ":$PATH:" in *":$_CH/bin:"*) : ;; *) export PATH="$_CH/bin:$PATH" ;; esac

_NS="${SELF_PLAY_ROOT:-${AC2_CLUSTER_A_ROOT}/self-play}/.venv/lib/python3.12/site-packages/nvidia"
_INC=()
for c in cuda_cccl cuda_runtime cublas curand cufft cusolver cusparse cusparselt cuda_nvrtc nvjitlink; do
  [ -d "$_NS/$c/include" ] && _INC+=("$_NS/$c/include")
done
[ -d "$_CH/include" ] && _INC+=("$_CH/include")
if [ "${#_INC[@]}" -gt 0 ]; then
  export CPATH="$(IFS=:; echo "${_INC[*]}")${CPATH:+:$CPATH}"
  export NVCC_PREPEND_FLAGS="$(printf -- '-I%s ' "${_INC[@]}")${NVCC_PREPEND_FLAGS:-}"
fi

# flashinfer writes its JIT workspace under ~/.cache by default, and cluster A home directories
# are at quota (Errno 122 on ${AC2_CLUSTER_A_HOME}/.cache/flashinfer). The Ray runtime_env sets
# this for the workers, but the driver-srun AOT preflight runs OUTSIDE that env, so it must be
# set here.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${AC2_CLUSTER_A_ROOT}/.cache/ds4_vllm_023/flashinfer}"

# HOME REDIRECT. When the home directory is over quota (so badly that even `ln -s` fails) or
# ~/.cache is missing, libraries that call expanduser("~/.cache") directly ignore
# XDG_CACHE_HOME, so setting the cache vars is not enough: at TP=4 this killed Worker_TP0 with
# Errno 122 mid-init. Pointing HOME at scratch makes every ~-based default land somewhere
# writable. Scoped to compute-node engine processes; the login shell keeps the real HOME.
if [ -n "${SP_SCRATCH_HOME:-}" ] || [ ! -w "${HOME:-/}" ] || [ ! -e "$HOME/.cache" ]; then
  export HOME="${SP_SCRATCH_HOME:-${AC2_CLUSTER_A_ROOT}/fakehome}"
  mkdir -p "$HOME/.cache" 2>/dev/null || true
fi
  ;;
cuda-toolkit)
#
# At engine init, flashinfer/vLLM JIT-compile the DS4 FP8 block-scale GEMM kernels whenever
# the warm JIT cache misses (e.g. after a .venv rebuild, or a kernel/shape not previously
# warmed). nvcc then needs the CUDA toolkit AND the pip `nvidia/*/include` headers
# (cublasLt.h, curand, cusolver, ...) on its include path; without them the build fails with
# `fatal error: cublasLt.h: No such file`.
#
# Source this AFTER activating the venv (it reads the venv's site-packages/nvidia includes).
# Evaluated per-node so it picks whatever CUDA toolkit that node actually has. Idempotent.
_VENV="${VIRTUAL_ENV:-${AC2_CLUSTER_B_ROOT}/self-play/.venv}"
# Toolkit preference: system cuda-12.9 (cluster B) -> a user-space 12.9 install in $HOME
# (cluster A nodes ship only <=12.6, whose nvcc cannot build the DeepGEMM DS4 fp8 kernels:
# "NVCC compilation failed"; an install on an inode-limited scratch filesystem can end up
# truncated -> "fatal error: cuda/std/utility") -> whatever /usr/local/cuda points at.
if [ -x /usr/local/cuda-12.9/bin/nvcc ]; then _CH=/usr/local/cuda-12.9
elif [ -x "$HOME/cuda-12.9/bin/nvcc" ]; then _CH="$HOME/cuda-12.9"
else _CH=/usr/local/cuda; fi
export CUDA_HOME="$_CH" CUDA_PATH="$_CH" CUDAToolkit_ROOT="$_CH" CUDACXX="$_CH/bin/nvcc" TRTLLM_DG_NVCC_COMPILER="$_CH/bin/nvcc"
case ":$PATH:" in *":$_CH/bin:"*) : ;; *) export PATH="$_CH/bin:$PATH" ;; esac
_NS="$_VENV/lib/python3.12/site-packages/nvidia"
_INC=()
for c in cublas curand cufft cusolver cusparse cusparselt cuda_nvrtc nvjitlink; do
  [ -d "$_NS/$c/include" ] && _INC+=("$_NS/$c/include")
done
if [ "${#_INC[@]}" -gt 0 ]; then
  export CPATH="$(IFS=:; echo "${_INC[*]}")"
  export NVCC_PREPEND_FLAGS="$(printf -- '-I%s ' "${_INC[@]}")"
fi
  ;;
*)
  echo "usage: source setup.sh cuda-compat|cuda-toolkit" >&2
  return 1 2>/dev/null || exit 1
  ;;
esac
