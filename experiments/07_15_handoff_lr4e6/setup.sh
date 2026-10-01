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
  : # no forward-compatibility libraries are needed on this cluster
  ;;
cuda-toolkit)
export VIRTUAL_ENV="${VIRTUAL_ENV:-${AC2_CLUSTER_C_ROOT}/self-play/.venv}"
# setup.sh cuda-toolkit — CUDA JIT-compile environment for the DeepSeek-V4-Flash judge stack.
#
# At engine init, flashinfer/vLLM JIT-compile the DS4 FP8 block-scale GEMM kernels whenever
# the warm JIT cache misses (e.g. after a .venv rebuild, or a kernel/shape not previously
# warmed). nvcc then needs the CUDA toolkit AND the pip `nvidia/*/include` headers
# (cublasLt.h, curand, cusolver, ...) on its include path. Do not assume the cache is always
# warm: after a .venv rebuild the compile fails with `fatal error: cublasLt.h: No such file`
# unless these paths are exported.
#
# Source this AFTER activating the venv (it reads the venv's site-packages/nvidia includes).
# Evaluated per-node so it picks whatever CUDA toolkit that node actually has. Idempotent.
_VENV="${VIRTUAL_ENV:-${AC2_CLUSTER_B_ROOT}/self-play/.venv}"
# Toolkit preference: system cuda-12.9 (cluster B) -> user-space 12.9.1 install in cluster A $HOME
# (cluster A nodes ship only <=12.6, whose nvcc can't build the DeepGEMM DS4 fp8 kernels:
# "NVCC compilation failed"; an install on an inode-quota-limited filesystem can be truncated
# mid-cccl -> "fatal error: cuda/std/utility") -> whatever /usr/local/cuda points at.
if [ -n "${CUDA_HOME_PREF:-}" ] && [ -x "$CUDA_HOME_PREF/bin/nvcc" ]; then _CH="$CUDA_HOME_PREF"
elif [ -x /usr/local/cuda-12.9/bin/nvcc ]; then _CH=/usr/local/cuda-12.9
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
