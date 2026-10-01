#!/usr/bin/env bash
# run_attach_cluster_b.sh <JOBID> [KEY=VAL ...] -- launch Prefix GRPO (08_11_ablation1_replay_noq)
# into a running 8-node allocation on cluster B (submit_cluster_b_8node.sbatch).
#
# Runs the preflight checks, pins every training environment variable, and starts runner.py
# on the allocation's head node. Trailing KEY=VAL arguments are appended to the driver
# environment and override the pinned values (SP_VAL_BEFORE_TRAIN excepted).
#
# This is AC2's launch environment (08_13_tiedq_seed192/run_attach_cluster_b.sh) with every
# critic setting removed and SP_Q_ENABLE=0; verify_ablation_diff.sh compares the two. The
# critic is removed, not configured away: the train dataset is replay_dataset.py /
# SPReplayNoQDataset, and the runner refuses to start with any behaviour-changing SP_Q_* still
# exported. Every continuation runs to termination and every reward is a judge score, so the
# per-step generation cost is higher than AC2's; compare on decoding cost, not only per step.
#
# Shared with AC2: n_batch = n_refill = 192 (SP_TRAIN_BATCH_SIZE=384, SP_REPLAY_N=192, PPO
# minibatch 96), group size 16, LR 2e-6, prefix cuts at multiples of 10,000 tokens up to 0.90,
# a global FIFO of 256 trajectories with ungated admission and question-uniform draws, empty at
# step 1 (cold bootstrap), the base Qwen3-4B-Thinking-2507 policy, no length penalty, adaptive
# entropy control, difficulty sampling off, a 50,000-token response budget, the judge at the
# pinned revision 60d8d707, and rollout TP=4 behind the flashinfer AOT preflight.
#
# The replay buffer is one global FIFO (BUCKETING=global, ROTATION=global_fifo, bounded by
# SP_REPLAY_BOUND) with no per-problem cap; SP_REPLAY_CAPACITY applies only to per_question
# bucketing and is not in force here.
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
REPO_ROOT=${AC2_CLUSTER_B_ROOT}/self-play
EXP="$REPO_ROOT/experiments/08_11_ablation1_replay_noq"
# No other experiment directory is referenced: a from-scratch run inherits no weights and no
# seed data.
VENV="$REPO_ROOT/.venv"
CACHE=${AC2_CLUSTER_B_ROOT}/.cache/ds4_vllm_023
HF_HOME_DIR=${AC2_CLUSTER_B_SCRATCH}/.cache/huggingface
HF_HUB_CACHE_DIR="$HF_HOME_DIR/hub"

JOBID="${1:?usage: run_attach_cluster_b.sh <HOLDER_JOBID> [KEY=VAL ...]}"; shift || true
# Trailing KEY=VAL overrides are appended to the driver env far below, but PREFLIGHT runs first
# and consults some of them -- SP_JUDGE_MODEL in particular, whose whole purpose is to bypass
# the hub-cache lookup. Without this, `run_attach_cluster_b.sh <jid> SP_JUDGE_MODEL=/path` would
# fail the cache check before the override is read. Only the preflight-relevant keys are
# lifted (an unrestricted eval of caller args would shadow this script's own variables).
for _kv in "$@"; do
  case "$_kv" in
    SP_JUDGE_MODEL=*|SP_JUDGE_HF_REPO=*|SP_JUDGE_REVISION=*|SP_REPLAY_COLD_BOOTSTRAP=*|SP_ALLOW_GRAFT=*|SP_FORCE_NNODES=*)
      export "${_kv?}"
      echo "[preflight] caller override honored before preflight: ${_kv%%=*}"
      ;;
  esac
done
JUDGE_HF_REPO="${SP_JUDGE_HF_REPO:-deepseek-ai/DeepSeek-V4-Flash}"
JUDGE_REVISION="${SP_JUDGE_REVISION-60d8d70770c6776ff598c94bb586a859a38244f1}"
COLD="${SP_REPLAY_COLD_BOOTSTRAP:-1}"
INFO="$EXP/sandbox_${JOBID}.info"
[ -f "$INFO" ] || INFO="$HOME/sandbox_${JOBID}.info"
[ -f "$INFO" ] || { echo "no holder info for $JOBID"; exit 1; }
HEAD_NODE=$(grep '^HEAD_NODE=' "$INFO" | head -1 | cut -d= -f2-)
RAY_ADDRESS=$(grep '^RAY_ADDRESS=' "$INFO" | head -1 | cut -d= -f2-)
# Node count comes from the allocation's .info file, not a constant: a hardcoded NNODES=8 on
# a smaller allocation hangs in ray/torch rendezvous.
NODES_LINE=$(grep '^NODES=' "$INFO" | head -1 | cut -d= -f2- | tr -d "'")
NNODES_H="${SP_FORCE_NNODES:-$(echo $NODES_LINE | wc -w)}"
[ "${NNODES_H:-0}" -ge 1 ] || { echo "[preflight] could not derive node count from $INFO"; exit 1; }
echo "[preflight] holder $JOBID has $NNODES_H node(s): $NODES_LINE"

# ---- flashinfer AOT preflight: REQUIRED for SP_ROLLOUT_TP>1 -------------------------------
# vLLM's AllReduceRMSFusionPass -> flashinfer get_trtllm_comm_module -> JitSpec.build_and_load
# does `with FileLock(...): build(); load()` UNCONDITIONALLY, once per rank. Every rank of a TP
# group serializes on that one lock and each re-runs ninja, so startup costs TP x build-time --
# TP=2 ~4 min, TP=4 ~8-10 min, TP=8 >17 min, at which point the ranks parked at the c10d store
# barrier blow their 600s budget and the engine WEDGES. All of it at 0% GPU util, because the
# work is nvcc on the CPU. TP=1 never enters this path.
#
# JitSpec.is_aot short-circuits build_and_load to a plain load(), so building each module ONCE
# single-process and promoting it into FLASHINFER_AOT_DIR gives the SAME artifact with
# compilation fully enabled. Idempotent, so this runs on every attach. Alternatives that do not
# work: disable_custom_all_reduce and VLLM_ALLREDUCE_USE_FLASHINFER=0 do not avoid the path,
# and compilation_config mode 0 avoids it only by disabling compilation.
#
# The warm step MUST run with the same FLASHINFER_WORKSPACE_BASE the engines get, or it
# promotes into a different AOT dir, prints READY about that dir, and the engines stampede
# anyway -- a gate that reports success while protecting nothing. flashinfer derives
# FLASHINFER_AOT_DIR from this variable, and its default is HOME-based, which on this cluster
# would also write .so files against the home quota. Exported here (not just listed in ENVS
# below, which only reaches the driver) and referenced from ENVS so the two cannot drift.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE/flashinfer}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE/torch_extensions}"
_AOT_WARM="$REPO_ROOT/experiments/08_11_ablation1_replay_noq/flashinfer_aot_warm.py"
if [ "${SP_ROLLOUT_TP:-4}" -gt 1 ] && [ "${SP_SKIP_AOT_WARM:-0}" != "1" ]; then
  echo "[preflight] flashinfer workspace: $FLASHINFER_WORKSPACE_BASE"
  if [ ! -f "$_AOT_WARM" ]; then
    echo "[preflight] REFUSING: SP_ROLLOUT_TP>1 needs $_AOT_WARM and it is"
    echo "            missing. Without the AOT promotion every rollout engine JIT-rebuilds"
    echo "            flashinfer's trtllm_comm once per rank and startup wedges."
    exit 1
  fi
  echo "[preflight] flashinfer AOT warm (idempotent; required at TP>1)..."
  _AOT_OUT="$("$VENV/bin/python" "$_AOT_WARM" 2>&1)" || true
  echo "$_AOT_OUT" | tail -5
  case "$_AOT_OUT" in
    *"[aot] READY"*) echo "[preflight] flashinfer AOT: READY" ;;
    *) echo "[preflight] REFUSING: flashinfer_aot_warm.py did not report '[aot] READY'."
       echo "            Launching anyway would have every TP rank rebuild the module under one"
       echo "            FileLock: minutes of 0%-util startup per engine, and a wedge at high TP."
       echo "            Fix the warm step first (SP_SKIP_AOT_WARM=1 overrides, deliberately)."
       exit 1 ;;
  esac
fi

SEED="$EXP/replay_seed_cold"   # EMPTY buffer seed, built by build_cold_artifacts.py
RUNNER="$EXP/runner.py"
CKPT="$EXP/run_data/checkpoints"
RUN_DATA_DIR="$EXP/run_data"

# ---- preflight ----
[ -f "$RUNNER" ] || { echo "[preflight] runner.py missing: $RUNNER"; exit 1; }
# The cold artifact is a ONE-TIME build: ONE empty seed (the replay buffer) with a valid
# manifest. AC2 builds three (buffer, critic buffer, reference bank); without a critic only the
# buffer exists. sp_replay's loader requires a manifest and one shard even when the shard
# holds zero records; without them the harness raises FileNotFoundError deep inside dataset
# construction, ~20 minutes into engine startup.
[ -f "$SEED/replay_buffer_manifest.json" ] || {
  echo "[preflight] cold replay seed missing: $SEED — run once:"
  echo "  $VENV/bin/python $EXP/build_cold_artifacts.py --out $EXP"
  exit 1
}
# This configuration builds NO critic artifacts: there is no critic buffer and no reference
# bank, so AC2's q_seed/ + reference_bank/ manifest and emptiness checks are gone. Only the
# replay seed is verified (above). Critic knobs are refused by the runner rather than by a seed
# check.
# ---- resume topology check ------------------------------------------------------------
# AC2 probes the checkpoint's world size from sp_q_optim/rank_*.pt and refuses a mismatch.
# This configuration writes no critic optimizer, so the probe reads the ACTOR shards instead.
# There is deliberately NO q_state.json / sp_q_optim requirement here: a checkpoint from this
# run never has them, and one that DOES came from an AC2 run and must not be resumed here.
_CKPT_DIR="$EXP/run_data/checkpoints/global_step_$(cat "$EXP/run_data/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)"
if [ -d "$_CKPT_DIR" ]; then
  if [ -e "$_CKPT_DIR/q_state.json" ] || [ -d "$_CKPT_DIR/sp_q_optim" ]; then
    echo "[preflight] REFUSING: $_CKPT_DIR carries q_state.json / sp_q_optim/, so it was"
    echo "            written by a run WITH a Q function. Resuming it here would continue a"
    echo "            main-run trajectory while every log line claimed the ablation."
    exit 1
  fi
  _CKPT_WS=$(ls "$_CKPT_DIR"/actor/*rank_*.pt 2>/dev/null | wc -l | tr -d ' ')
  _WANT_WS=$(( NNODES_H * 8 ))
  if [ "${_CKPT_WS:-0}" -gt 0 ] && [ "$_CKPT_WS" -ne "$_WANT_WS" ]; then
    echo "[preflight] REFUSING: checkpoint at $_CKPT_DIR has world size $_CKPT_WS but this"
    echo "            allocation is $_WANT_WS. Reshard it before resuming."
    exit 1
  fi
fi

CNT="$EXP/run_${JOBID}.attempts"; A=$(( $(cat "$CNT" 2>/dev/null || echo 0) + 1 )); echo "$A" > "$CNT"
LOG="$EXP/run_${JOBID}_attempt${A}.log"

ENVS=(
  "RAY_ADDRESS=$RAY_ADDRESS" "NNODES=$NNODES_H" "N_GPUS_PER_NODE=8"
  "HF_HOME=$HF_HOME_DIR" "HF_HUB_CACHE=$HF_HUB_CACHE_DIR"
  "SELF_PLAY_DATA_DIR=${AC2_CLUSTER_B_ROOT}/data/fineproofs"
  "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1" "WANDB_MODE=offline"
  "XDG_CACHE_HOME=$XDG_CACHE_HOME" "TRITON_CACHE_DIR=$CACHE/triton" "TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor"
  "VLLM_CACHE_ROOT=$CACHE/vllm" "TORCH_HOME=$CACHE/torch" "CUDA_CACHE_PATH=$CACHE/nv"
  "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR" "FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE"
  "MPLCONFIGDIR=$CACHE/matplotlib" "DO_NOT_TRACK=1" "VLLM_NO_USAGE_STATS=1" "VERL_STAGGER_ENGINE_INIT=1"
  # The judge runs as world_size // SP_JUDGE_TP replicas (one TP=8 replica per node).
  # SP_JUDGE_MAX_INFLIGHT caps the judge calls each reward worker keeps in flight; 160 was sized
  # for 8 replicas (~20 concurrent requests per replica).
  "SP_JUDGE_MAX_INFLIGHT=160" "SP_PASS_POINTS_MIN=6"
  # This run's OWN experiment name: it is not a continuation of any wandb run.
  "SP_EXPERIMENT_NAME=08_11_ablation1_replay_noq" "SP_RUN_DATA_DIR=run_data"
  "VERL_STEP_CACHE_DIR=$EXP/run_data/step_cache"
  # ==== judge: model AND revision pinned, so the reward function cannot drift ====
  "SP_JUDGE_HF_REPO=$JUDGE_HF_REPO" "SP_JUDGE_REVISION=$JUDGE_REVISION"
  # ---- policy optimization ----
  "SP_REPLAY_ENABLE=1"
  # policy LR 2e-6 = the runner default 1.4142e-6 x sqrt(2), for a trained batch of 192 groups
  # instead of the default 96
  "SP_LR=2e-6" "SP_ENTROPY_COEFF=0.0"
  "SP_ADAPTIVE_ENTROPY=1" "SP_AEC_TARGET_H=0.28" "SP_AEC_DELTA=0.02"
  "SP_AEC_KMAX=0.08" "SP_AEC_KMIN=-0.08" "SP_AEC_KINIT=0.06"
  # ==== batch shape ====
  # Each step: SP_REPLAY_N = 192 replayed prefixes (trained) plus TRAIN_BATCH - REPLAY_N = 192
  # fresh problems in the inflow lane. SP_SCRATCH_INFLOW_ONLY=1 drops the inflow lane before the
  # loss, so the TRAINED batch is 192 groups x 16 = 3,072 sequences in 2 PPO minibatches of 96
  # (which is why mini 96 satisfies the REPLAY_N % PPO_MINI assertion in runner.py); the inflow
  # lane only feeds the replay buffer and the difficulty EMA.
  "SP_TRAIN_BATCH_SIZE=384" "SP_REPLAY_N=192" "SP_PPO_MINI_BATCH=96"
  "SP_SCRATCH_INFLOW_ONLY=1"
  # Prefix cuts at multiples of 10,000 tokens, up to 0.90 of the stored trajectory's length.
  "SP_REPLAY_CUT_LOW=0" "SP_REPLAY_CUT_HIGH=0.90" "SP_REPLAY_CUT_GRAIN=10000"
  # ==== replay management: one global FIFO buffer, admission regardless of correctness ====
  "SP_REPLAY_BUCKETING=global" "SP_REPLAY_ADMISSION=ungated" "SP_REPLAY_ROTATION=global_fifo"
  # ==== replay draw: global FIFO of 256 trajectories, uniform over QUESTIONS ====
  # The default "entry" mode permutes ENTRIES, which makes a problem's draw probability
  # proportional to how many trajectories it currently holds -- an implicit OCCUPANCY weighting
  # with no difficulty signal in it (under admission=ungated, occupancy tracks how recently/often
  # a problem entered the inflow lane). "question" permutes the DISTINCT problems and then picks
  # one of that problem's entries uniformly, so every problem in the buffer is equally likely per
  # slot.
  #
  # NOT the same thing as the buffer's difficulty weighting (w = 3 - 2.5*EMA): that lives in
  # the per_question draw only and is unreachable under bucketing=global, in either mode.
  # Watch replay/distinct_questions and replay/question_repeat_factor -- with 192 replay slots
  # and fewer than 192 distinct problems live, some problems appear more than once per step (at
  # different prefixes). At 256 entries that factor is ~1.0-2.3 depending on occupancy.
  "SP_REPLAY_BOUND=256" "SP_REPLAY_GLOBAL_SAMPLING=question"
  # ==== cold start: EMPTY buffer. The seed carries zero entries, and the cold-bootstrap flag
  # is what lets the global draw report a full shortfall at step 1 instead of raising
  # "sp_replay global buffer is empty".
  "SP_REPLAY_SEED_DIR=$SEED"
  "SP_REPLAY_COLD_BOOTSTRAP=1"
  # ==== no critic settings: every SP_Q_* knob of AC2's launch environment (readiness gates,
  # critic buffer and prompts, chunk length, audit fraction, LR ladder) is removed here, and
  # SP_Q_ENABLE is pinned to 0 below.
  # ==== rollout tensor parallelism 4 (inference efficiency) ====
  # One rollout engine per 4 GPUs instead of one per GPU. Single-GPU replicas lose throughput to
  # replica IMBALANCE (in a replay benchmark, an 11.4x spread: most GPUs idle while one replica
  # finished its tail; TP=2 reached 967.6 vs 603.6 tok/s/node for TP=1), and TP>1 also shards
  # attention, the largest share (~75%) of rollout GPU time.
  #
  # Requires the flashinfer AOT preflight above -- at TP>1 the engine cannot start reliably
  # without it.
  "SP_ROLLOUT_TP=4"
  "SP_ROLLOUT_PRIORITY=0"
  # ==== response budget: must match AC2's (verify_ablation_diff.sh checks it). There is no
  # critic, so no C_Q to reconcile with the rollout max_model_len (2048 + 50000 + 1248 = 53296).
  #
  # SP_PPO_MAX_TOKEN_LEN=51200 is ~0.98x a full packed row (2048 + 50000 = 52048). The invariant
  # that matters is ONE maximum-length row per GPU micro-batch; the budget need not sit strictly
  # below a full row.
  "SP_MAX_RESPONSE_LEN=50000"
  "SP_PPO_MAX_TOKEN_LEN=51200"
  # KV arena 0.7, between the runner default (0.6) and the 0.75 used at longer budgets. At
  # 50k/TP=4 it holds ~28 full-length (53,296-token) sequences per engine, up from ~24 at 0.6,
  # so SP_ACTOR_MAX_NUM_SEQS=96 is nowhere near binding and the extra arena is headroom for
  # short sequences.
  #
  # This is the knob to LOWER if you see "sample_tokens RPC timed out" with NO OOM line: that
  # signature means the arena cannot coexist with the resident FSDP actor. Raising it is the
  # riskier direction, which is why 0.7 rather than 0.75 at 50k.
  "SP_ACTOR_GPU_MEM_UTIL=0.7"
  # ==== NO length penalty -- pinned to 0, not merely omitted.
  # ds4_finegrained_judge reads SP_LENPEN_ENABLE straight from the process environment, the
  # allocation does not use --export=NONE, and the runner only forwards the variable when it is
  # non-empty. So submitting from a shell that still exports SP_LENPEN_ENABLE=1 would silently
  # reinstate a length penalty with nothing in the log saying so. Setting it explicitly makes the
  # intended value the one that reaches the judge, and the runner asserts it. Same reasoning for
  # the other science-changing switches below.
  "SP_LENPEN_ENABLE=0"
  # The critic is ABSENT here, so SP_Q_ENABLE is pinned OFF rather than omitted: the runner
  # forwards it into runtime_env and the dataset asserts on it, and an inherited 1 from an AC2
  # shell would otherwise install critic hooks against a dataset that routes everything "full".
  "SP_Q_ENABLE=0"
  "SP_DIFF_SAMPLING=0" "SP_REPLAY_POLICY_OVERRIDE=0"
  # ==== run length / validation ====
  "SP_TOTAL_STEPS=500" "SP_SAVE_FREQ=1" "SP_TEST_FREQ=10"
  "SP_ROLLOUT_BACKFILL=1" "PYTHONUNBUFFERED=1"
)

# ---- validation gate ---------------------------------------------------------------
# Two jobs here, both about not losing a validation:
#   (a) FIRST ATTACH of a from-scratch run: validate before training, so the base model's
#       IMO-ProofBench score is this run's own step-0 baseline. Every later attach must NOT
#       repeat it (it costs a full val pass and would overwrite nothing useful).
#   (b) ray_trainer saves the checkpoint BEFORE validating, so a relaunch between the save
#       and the end of validation LOSES that step's val row -- not retried, and the next one
#       is TEST_FREQ steps away. Turn it back on for exactly that case: the resumed checkpoint
#       is a validation step AND metrics.jsonl has no val-* row for it.
_VBT="$("$VENV/bin/python" - "$EXP" <<'PYVBT' 2>/dev/null || echo False
import json, sys
E = sys.argv[1]
TF = 10                     # SP_TEST_FREQ
try:
    ck = int(open(f"{E}/run_data/checkpoints/latest_checkpointed_iteration.txt").read().strip())
except Exception:
    ck = 0
try:
    _hist = sum(1 for _l in open(f"{E}/run_data/metrics.jsonl", errors="ignore") if _l.strip())
except OSError:
    _hist = 0
# (a) fresh start: no checkpoint AND no history -> baseline validation.
if ck <= 0:
    print("True" if _hist == 0 else "False"); raise SystemExit
if ck % TF:
    print("False"); raise SystemExit
if _hist == 0:
    # a checkpoint with no history is the graft case; the attach refuses it above.
    print("False"); raise SystemExit
try:
    for line in open(f"{E}/run_data/metrics.jsonl", errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        d = r.get("data", {})
        if not any(k.startswith("val-") for k in d):
            continue
        # Read BOTH step fields: in-loop validation carries training/global_step, while the
        # val_before_train path logs only val_metrics with a top-level "step". Checking one
        # would make the gate blind to its own validation and re-fire every restart.
        step = d.get("training/global_step")
        if step is None:
            step = r.get("step")
        try:
            step = int(step)
        except (TypeError, ValueError):
            continue
        if step == ck:
            print("False"); raise SystemExit      # already validated
except FileNotFoundError:
    print("False"); raise SystemExit
print("True")
PYVBT
)"
case "$_VBT" in True) : ;; *) _VBT=False ;; esac   # fail closed
echo "[attach] validation gate: SP_VAL_BEFORE_TRAIN=$_VBT (ckpt ${_LATEST:-0})"

# Caller-supplied KEY=VALUE args win for everything EXCEPT SP_VAL_BEFORE_TRAIN, which is
# dropped so the gate above decides.
for kv in "$@"; do
  case "$kv" in SP_VAL_BEFORE_TRAIN=*) continue ;; esac
  ENVS+=("$kv")
done
ENVS+=("SP_VAL_BEFORE_TRAIN=$_VBT")

echo "[attach] attempt $A on holder $JOBID (head $HEAD_NODE); log: $LOG"
nohup srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 --mem=0 -w "$HEAD_NODE" \
    bash -c "unset ROCR_VISIBLE_DEVICES; source $VENV/bin/activate && source $REPO_ROOT/experiments/08_11_ablation1_replay_noq/setup.sh cuda-toolkit && cd $REPO_ROOT && \
        env ${ENVS[*]} python $RUNNER" > "$LOG" 2>&1 &
echo "[attach] driver login-node pid $! ; tail -f $LOG"
