#!/usr/bin/env bash
# run_attach_cluster_c.sh <ALLOC_JOBID> [KEY=VAL ...] -- launch the 07_15_handoff_lr4e6 training driver
# into a running allocation started by submit_cluster_c_8node.sbatch.
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
JOBID="${1:?usage: run_attach_cluster_c.sh <alloc_jobid> [KEY=VAL ...]}"
REPO_ROOT="${SELF_PLAY_ROOT:-${AC2_CLUSTER_C_ROOT}/self-play}"
EXP="$REPO_ROOT/experiments/07_15_handoff_lr4e6"
INFO="$EXP/alloc_${JOBID}.info"
[ -f "$INFO" ] || { echo "no allocation info for $JOBID"; exit 1; }
HEAD_NODE=$(grep '^HEAD_NODE=' "$INFO" | cut -d= -f2-)
RAY_ADDRESS=$(grep '^RAY_ADDRESS=' "$INFO" | cut -d= -f2-)

CNT="$EXP/run_${JOBID}.attempts"
A=$(( $(cat "$CNT" 2>/dev/null || echo 0) + 1 )); echo "$A" > "$CNT"
LOG="$EXP/run_${JOBID}_attempt${A}.log"

# ---- val-before-train fires on the GENUINE first launch only --------------------------
# verl runs val_before_train unconditionally at the top of fit(), NOT only at step 0
# (ray_trainer.py: `if self.config.trainer.get("val_before_train", True): self._validate()`).
# Leaving it True would therefore re-run a full 60-problem x n=16 IMO-ProofBench validation
# on EVERY driver restart — roughly an hour of the whole allocation, every time. So: True
# only while no checkpoint exists (the real step-0 launch, where the base model's val IS
# the baseline this experiment is measured against), False on every relaunch after that.
if [ -f "$EXP/run_data/checkpoints/latest_checkpointed_iteration.txt" ]; then
  VBT=False
else
  VBT=True
  echo "[attach] FRESH START: base Qwen3-4B-Thinking-2507, fresh optimizer, step 0"
  echo "[attach] val_before_train=True — the base model's val@0 is this run's baseline"
fi

ENVS=(
  "RAY_ADDRESS=$RAY_ADDRESS" "NNODES=8" "N_GPUS_PER_NODE=8"

  # ---- pins that would otherwise come from ac2.clusters ------------------------
  # detect_cluster() knows only cluster A and cluster B and falls back to "cluster_a" on anything
  # else, so WITHOUT these the runner would hand offline Ray workers a ${AC2_CLUSTER_A_MARKER}
  # model cache that does not exist here and set WANDB_MODE=online on a host with no
  # wandb credentials. HF_HUB_CACHE as well as HF_HOME: they are separate keys in that
  # registry and only pinning HF_HOME leaves the hub cache on the cluster A path.
  "HF_HOME=${AC2_CLUSTER_C_ROOT}/hf-cache" "HF_HUB_CACHE=${AC2_CLUSTER_C_ROOT}/hf-cache/hub"
  "XDG_CACHE_HOME=/tmp/${USER}/sp_cache"
  "WANDB_MODE=offline"
  # Both models pinned EXPLICITLY. The runner's judge default is a hard-coded snapshot
  # path under the cluster B scratch, which does not exist here.
  "SP_ACTOR_MODEL=Qwen/Qwen3-4B-Thinking-2507"
  "SP_JUDGE_MODEL=deepseek-ai/DeepSeek-V4-Flash"
  "SELF_PLAY_DATA_DIR=${AC2_CLUSTER_C_ROOT}/data/fineproofs"
  # The val map default is ~/data/fineproofs/val_map.json — a home path that does not
  # exist here, and it is only opened at the FIRST validation, so a wrong value fails
  # silently (val graded with no reference solution) rather than at startup.
  "SP_VAL_MAP=${AC2_CLUSTER_C_ROOT}/data/fineproofs/val_map.json"
  # 51200 is the 80 GB H100 value. It is slightly below a full packed sequence
  # (2048 prompt + 50000 response = 52048); that is deliberate, not an oversight — 98304
  # OOMs the actor update on 80 GB.
  "SP_PPO_MAX_TOKEN_LEN=51200"
  "SP_ACTOR_MAX_NUM_SEQS=96"
  # Rollout tensor parallelism 4, not the runner default of 1: $GPUS_PER_NODE/4 replicas per
  # node instead of one per GPU. In a rollout benchmark TP2 gave 967.6 vs TP1 603.6
  # tok/s/node, where TP1's loss was replica IMBALANCE (11.4x spread, most GPUs idle while one
  # replica finished its tail) rather than per-GPU speed, and attention (which TP shards) was
  # ~75% of rollout GPU time. Throughput only: it changes no sampling and no gradient.
  #
  # REQUIRES the flashinfer AOT preflight below. At TP>1 vLLM's allreduce fusion pass
  # JIT-builds a flashinfer module once per rank under a single FileLock, so engine init costs
  # TP x build-time at 0% GPU util and wedges outright at high TP. Set SP_ROLLOUT_TP=1 to opt
  # out of both.
  "SP_ROLLOUT_TP=4"
  # Retention: the trainer keeps every 10th step's weights permanently (SP_CKPT_KEEP_EVERY,
  # default 10) on top of the last $CKPT_KEEP + best. ~47 GB per retained step, so a 500-step
  # run holds ~50 of them. Set 0 to get last-N + best only.
  "SP_CKPT_KEEP_EVERY=10"

  "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1"
  "VLLM_NO_USAGE_STATS=1" "DO_NOT_TRACK=1" "VERL_STAGGER_ENGINE_INIT=1"
  "PYTHONUNBUFFERED=1"
  "SP_EXPERIMENT_NAME=07_15_handoff_lr4e6" "SP_RUN_DATA_DIR=run_data"
  "VERL_STEP_CACHE_DIR=$EXP/run_data/step_cache"
  # Fills a drained rollout replica's GPU with a benign matmul while stragglers finish, so
  # the job's average utilisation does not fall into low-util-reclaim territory. Scheduling
  # only — it changes no sampling and no gradient.
  "SP_ROLLOUT_BACKFILL=1"

  # ---- science -------------------------------------------------------------------
  "SP_JUDGE_STANDALONE=0"
  "SP_JUDGE_MAX_INFLIGHT=80" "SP_PASS_POINTS_MIN=6"
  "SP_TRAIN_BATCH_SIZE=256" "SP_ROLLOUT_N=16" "SP_PPO_MINI_BATCH=128"
  # THE learning rate.
  #
  # The three LR-sweep arms (07_15_handoff, 07_15_handoff_lr1e6, 07_15_handoff_lr4e6)
  # differ only in the learning rate, the run name and the KV arena below.
  "SP_LR=4e-6" "SP_ENTROPY_COEFF=0.0"
  "SP_ADAPTIVE_ENTROPY=1" "SP_AEC_TARGET_H=0.28" "SP_AEC_DELTA=0.02"
  "SP_AEC_KMAX=0.08" "SP_AEC_KMIN=-0.08" "SP_AEC_KINIT=0.06"
  "SP_LENPEN_ENABLE=0"
  "SP_DIFF_SAMPLING=0"
  "SP_KEEP_BEST_CKPT=1"
  # The context-length group: the 50k-token response budget used by AC2 as well.
  "SP_MAX_PROMPT_LEN=2048" "SP_MAX_RESPONSE_LEN=50000"
  # The KV arena. It is the first thing to lower when generation cannot coexist with the
  # resident FSDP actor ("sample_tokens RPC timed out" with NO OOM line).
  "SP_ACTOR_GPU_MEM_UTIL=0.7"

  "SP_TOTAL_STEPS=500" "SP_SAVE_FREQ=1" "SP_TEST_FREQ=10"
  "SP_VAL_BEFORE_TRAIN=$VBT"
  "SP_MAX_CKPT_KEEP=3"
)
for kv in "${@:2}"; do ENVS+=("$kv"); done

echo "[attach] attempt $A on allocation $JOBID (head $HEAD_NODE) -> $LOG"
# ---- flashinfer AOT preflight: REQUIRED whenever SP_ROLLOUT_TP > 1 ------------------------
# vLLM's AllReduceRMSFusionPass -> flashinfer get_trtllm_comm_module -> JitSpec.build_and_load
# runs a FileLock-wrapped build-then-load once per RANK, so a TP group serializes on one lock
# and each rank re-runs ninja: TP=2 ~4 min, TP=4 ~8-10 min, TP=8 >17 min, after which the ranks
# parked at the c10d barrier blow their 600s budget and the engine wedges. All at 0% GPU util,
# because it is nvcc on the CPU. Promoting the modules to AOT once makes build_and_load a plain
# load() with compilation still fully enabled. Idempotent, so it runs on every attach.
#
# It MUST see the same FLASHINFER_WORKSPACE_BASE the engines get, or it warms a different AOT
# directory, reports READY about that one, and the engines stampede anyway. flashinfer's default
# is HOME-based, which would also put .so files against a home quota.
_TP="$(printf '%s\n' "${ENVS[@]}" | sed -n 's/^SP_ROLLOUT_TP=//p' | tail -1)"
if [ "${_TP:-1}" -gt 1 ] && [ "${SP_SKIP_AOT_WARM:-0}" != "1" ]; then
  export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/${USER}/sp_cache/flashinfer}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/${USER}/sp_cache}"
  echo "[attach] flashinfer AOT warm (TP=$_TP, workspace $FLASHINFER_WORKSPACE_BASE)..."
  if [ ! -f "$EXP/flashinfer_aot_warm.py" ]; then
    echo "[attach] REFUSING: SP_ROLLOUT_TP=$_TP needs $EXP/flashinfer_aot_warm.py and it is"
    echo "         missing. Restore it, or set SP_ROLLOUT_TP=1."
    exit 1
  fi
  _AOT_OUT="$("$REPO_ROOT/.venv/bin/python" "$EXP/flashinfer_aot_warm.py" 2>&1)" || true
  echo "$_AOT_OUT" | tail -5
  case "$_AOT_OUT" in
    *"[aot] READY"*) echo "[attach] flashinfer AOT: READY" ;;
    *) echo "[attach] REFUSING: the warm step did not report '[aot] READY'. Launching now would"
       echo "         have every TP rank rebuild the module under one lock: minutes of 0%-util"
       echo "         startup per engine, and a wedge at high TP. Fix it, set SP_ROLLOUT_TP=1, or"
       echo "         SP_SKIP_AOT_WARM=1 to override deliberately."
       exit 1 ;;
  esac
fi
# setup.sh cuda-toolkit is NOT optional. At engine init flashinfer/vLLM JIT-compile the DS4 FP8
# block-scale GEMM kernels whenever the JIT cache misses -- and on a fresh cluster the
# cache is ALWAYS cold. nvcc then needs the CUDA toolkit plus the venv's nvidia/*/include
# headers on its include path, or it dies with "fatal error: cublasLt.h: No such file".
# It is evaluated per-node so it picks up whatever toolkit that node actually has.
nohup srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 --mem=0 -w "$HEAD_NODE" \
  bash -c "unset ROCR_VISIBLE_DEVICES; source $REPO_ROOT/.venv/bin/activate && source $EXP/setup.sh cuda-toolkit && cd $REPO_ROOT && env ${ENVS[*]} python $EXP/runner.py" \
  > "$LOG" 2>&1 &
echo "[attach] driver pid $!"
