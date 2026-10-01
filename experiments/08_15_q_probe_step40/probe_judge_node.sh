#!/usr/bin/env bash
# One node: bring up a DS4-Flash TP=8 judge with a direct `vllm serve`, then score this node's
# shard against it. Deviating from the env and flags below makes engine start-up fail with
# "Engine core initialization failed"; the reasons are worth recording because none of them
# are guessable:
#
#   * DG_JIT_CACHE_DIR, not SP_DG_JIT_CACHE_BASE. The latter is resolved by verl's
#     vLLMHttpServer only. A direct `vllm serve` never reads it, so pinning it leaves the
#     DeepGEMM JIT cache pointed at vLLM's shared default, where a concurrent first-use
#     compile from another node can be read as a 0-byte cubin (CUDA_ERROR_INVALID_IMAGE).
#   * DS4 needs `--tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4`, AND
#     `--kv-cache-dtype fp8`: dropping the latter fails with "DeepseekV4 FlashMLA fp8 layout
#     only supports fp8 kv-cache, got auto". Change the flag set below only with a reason.
#   * The JIT toolchain needs CUDA_HOME plus CPATH/NVCC_PREPEND_FLAGS pointing at the venv's
#     nvidia/*/include trees, or nvcc cannot find curand.h/cuda atomics and DeepGEMM's
#     compile fails inside the worker.
#
# Ray is stopped first: TP=8 capture dies against a live raylet holding all 8 GPUs and
# ~200 GB of /dev/shm. Safe for a paused training run on the same allocation -- attaching a
# driver always restarts Ray.
set -uo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }

REPO="${SELF_PLAY_ROOT:-${AC2_CLUSTER_A_ROOT}/self-play}"
E="$REPO/experiments/08_15_q_probe_step40"   # scripts live here
# data (model_hf, probe_set, gen/, q/, judged/) can live elsewhere so a second
# probe at a different checkpoint does not overwrite the first one's artifacts
D="${SP_PROBE_DIR:-$E}"
VENV="$REPO/.venv"
JUDGE="${SP_JUDGE_MODEL:-${AC2_CLUSTER_A_ROOT}/hf_home/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1}"
HF="${SP_HF_HOME:-${AC2_CLUSTER_A_ROOT}/hf_home}"
CACHE="${SP_PROBE_CACHE:-${AC2_CLUSTER_A_ROOT}/.cache/ds4_vllm_023}"
RANK="${SLURM_PROCID:-0}"
NSHARDS="${SP_JUDGE_NSHARDS:-4}"
PORT="${SP_JUDGE_PORT:-8731}"
OUTDIR="${SP_JUDGE_OUTDIR:-$D/judged}"
GENDIR="${SP_JUDGE_GEN_DIR:-$D/gen}"
OUTTAG="${SP_JUDGE_OUTTAG:-$RANK}"

unset ROCR_VISIBLE_DEVICES
source "$REPO/experiments/08_13_tiedq_seed192/setup.sh" cuda-compat 2>/dev/null || true
source "$VENV/bin/activate"
unset PYTHONPATH RAY_ADDRESS
unset CPATH NVCC_PREPEND_FLAGS

# CUDA toolkit: let setup.sh cuda-toolkit DETECT it per node. cluster A nodes ship <=12.6, whose nvcc
# cannot build the DeepGEMM DS4 fp8 kernels ("NVCC compilation failed"), so setup.sh cuda-toolkit falls
# back to $HOME/cuda-12.9. Never pin a toolkit path here: /usr/local/cuda-12.9 is the cluster B
# path, absent on cluster A, and pinning it produces exactly that NVCC failure.
SETUP_SH="$REPO/experiments/08_15_q_probe_step40/setup.sh"
[ -f "$SETUP_SH" ] && source "$SETUP_SH" cuda-toolkit || echo "[judge-node] WARN: setup.sh cuda-toolkit missing"
echo "[judge-node] CUDA_HOME=${CUDA_HOME:-unset} nvcc=$(command -v nvcc || echo MISSING)"

# stale per-kernel toggles from an inherited env silently change the served numerics
while IFS='=' read -r name _; do
  case "$name" in
    VLLM_TRITON_MLA_SPARSE*) unset "$name" ;;
    VLLM_ATTENTION_BACKEND|VLLM_USE_BREAKABLE_CUDAGRAPH|VLLM_USE_DEEP_GEMM|VLLM_MOE_USE_DEEP_GEMM|VLLM_USE_FLASHINFER_MOE_FP8|VLLM_FLASHINFER_MOE_BACKEND|VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER|VLLM_TEST_FORCE_FP8_MARLIN|VLLM_DISABLED_KERNELS|TORCH_COMPILE_DISABLE|VLLM_USE_DEEP_GEMM_E8M0|VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES|VLLM_MARLIN_USE_ATOMIC_ADD|VLLM_MARLIN_INPUT_DTYPE|VLLM_MXFP4_USE_MARLIN|VLLM_USE_FLASHINFER_MOE_FP16|VLLM_USE_FLASHINFER_MOE_FP4|VLLM_USE_FLASHINFER_MOE_INT4|VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8|VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS|VLLM_USE_FLASHINFER_MOE_MXFP4_BF16|VLLM_USE_FLASHINFER_SAMPLER|VLLM_MLA_DISABLE|VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD|VLLM_SPARSE_INDEXER_MAX_LOGITS_MB|VLLM_USE_MEGA_AOT_ARTIFACT) unset "$name" ;;
  esac
done < <(env)

export HF_HOME="$HF" HF_HUB_CACHE="$HF/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
export XDG_CACHE_HOME="$CACHE/xdg" TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" VLLM_CACHE_ROOT="$CACHE/vllm"
export TORCH_HOME="$CACHE/torch" CUDA_CACHE_PATH="$CACHE/nv"
export TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions" FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer"
export TRTLLM_DG_CACHE_DIR="$CACHE/trtllm_dg" TRTLLM_DG_NVCC_COMPILER="$CUDA_HOME/bin/nvcc"
export TILELANG_CACHE_DIR="$CACHE/tilelang" TILELANG_TMP_DIR="$CACHE/tilelang/tmp"
export VLLM_DISABLE_COMPILE_CACHE=1 VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_RPC_TIMEOUT=600000
export TILELANG_CLEANUP_TEMP_FILES=1 PYTHONUNBUFFERED=1

host="$(hostname)"
# DG_JIT_CACHE_DIR IS THE PARENT OF DeepGEMM's OWN cache/ SUBDIR -- getting this wrong gives
# "NVCC compilation failed". verl does exactly this (vllm_async_server.py):
# os.path.join(SP_DG_JIT_CACHE_BASE, socket.gethostname()), with NO /cache suffix. DeepGEMM
# then creates and reads <dir>/cache/ itself. Pointing at <base>/<host>/cache instead -- which
# is tempting, because that is where the warm kernels visibly live -- makes DeepGEMM look in
# <base>/<host>/cache/cache, find nothing, and JIT-compile everything, on nodes whose nvcc
# cannot build these kernels.
export DG_JIT_CACHE_DIR="$CACHE/dg_jit_nodes/$host"
mkdir -p "$DG_JIT_CACHE_DIR/cache" "$TILELANG_TMP_DIR" "$OUTDIR"
echo "[judge-node $RANK/$host] DG_JIT_CACHE_DIR=$DG_JIT_CACHE_DIR (warm kernels: $(ls "$DG_JIT_CACHE_DIR/cache" 2>/dev/null | wc -l))"

echo "[judge-node $RANK/$host] stopping ray so TP=8 can capture"
ray stop --force >/dev/null 2>&1 || true
sleep 5
pkill -f "[v]llm serve" 2>/dev/null || true
sleep 2

echo "[judge-node $RANK/$host] serving DS4 TP=8 on :$PORT"
# EXACTLY the engine_kwargs the training run gives its judge (the reward_model engine kwargs
# and the pinned _JUDGE_COMPILATION_CONFIG in 08_13_tiedq_seed192/runner.py).
# WHY THIS MUST MATCH EXACTLY, including the batch shapes: the run never COMPILES anything.
# cluster A nodes cannot build the DeepGEMM DS4 fp8 kernels (nvcc fails), so the whole design
# relies on the per-node dg_jit cache already holding every kernel the engine asks for. A
# kernel is keyed by its GEMM shape, and the shapes follow max_num_seqs / max_num_batched_
# tokens. Other values (e.g. 3072/32768) force an uncached shape whose JIT dies with "NVCC
# compilation failed". The run uses 256/16384 (the runner's SP_REWARD_MAX_NUM_SEQS /
# SP_REWARD_MAX_NUM_BATCHED_TOKENS defaults), so we do too.
setsid vllm serve "$JUDGE" --served-model-name judge \
  --tensor-parallel-size 8 \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --kv-cache-dtype fp8 --block-size 256 --moe-backend marlin --async-scheduling \
  --max-model-len 98304 --gpu-memory-utilization 0.80 \
  --max-num-seqs 256 --max-num-batched-tokens 16384 \
  --enable-prefix-caching --enable-chunked-prefill \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]}' \
  --disable-uvicorn-access-log --trust-remote-code \
  --host 127.0.0.1 --port "$PORT" > "$OUTDIR/serve_${host}.log" 2>&1 &
SPID=$!

ok=0
for i in $(seq 1 480); do            # up to 40 min: DS4 capture is slow and flaky
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  sleep 5
done
if [ "$ok" != "1" ]; then
  echo "[judge-node $RANK/$host] FATAL: judge never healthy; tail of serve log:"
  grep -E "Error|error:|Exception|raise |NVCC|CUDA" "$OUTDIR/serve_${host}.log" | tail -15 || true
  exit 0
fi
echo "[judge-node $RANK/$host] judge healthy after ~$((i*5))s"

SP_JUDGE_MAX_INFLIGHT="${SP_JUDGE_MAX_INFLIGHT:-160}" \
SELF_PLAY_JUDGE_URL="http://127.0.0.1:$PORT/v1/chat/completions" \
python "$E/probe_judge.py" --gen-dir "$GENDIR" --shard "$RANK" --num-shards "$NSHARDS" \
    --out "$OUTDIR/judged.shard${OUTTAG}.jsonl" \
    --judge-url "http://127.0.0.1:$PORT/v1/chat/completions" \
    > "$OUTDIR/client_${OUTTAG}.log" 2>&1
echo "[judge-node $RANK/$host] client exited rc=$?"
kill -TERM -- "-$SPID" 2>/dev/null || kill "$SPID" 2>/dev/null || true
exit 0
