#!/usr/bin/env bash
# Source before every reproduction command; all mutable caches are project-local.
export AC2_SCRATCH=/scratch/richard1xur/ac2
export TMPDIR="$AC2_SCRATCH/tmp" TMP="$AC2_SCRATCH/tmp" TEMP="$AC2_SCRATCH/tmp"
export UV_CACHE_DIR="$AC2_SCRATCH/uv" UV_PYTHON_INSTALL_DIR="$AC2_SCRATCH/python"
export HF_HOME="$AC2_SCRATCH/hf" HF_HUB_CACHE="$AC2_SCRATCH/hf/hub"
export RAY_TMPDIR="$AC2_SCRATCH/r"
export XDG_CACHE_HOME="$AC2_SCRATCH/cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TORCH_HOME="$XDG_CACHE_HOME/torch" CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb" WANDB_DATA_DIR="$XDG_CACHE_HOME/wandb-data"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export WANDB_MODE=offline DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export VENV_DIR="$AC2_SCRATCH/venv"
export FA_BUILD_DIR="$AC2_SCRATCH/fa_build"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$XDG_CACHE_HOME"
