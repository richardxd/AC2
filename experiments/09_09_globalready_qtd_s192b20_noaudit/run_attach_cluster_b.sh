#!/usr/bin/env bash
# run_attach_cluster_b.sh <HOLDER_JOBID> [KEY=VAL ...]
# Launch script for AC2 w/o Group & Audit & local readiness
# (09_09_globalready_qtd_s192b20_noaudit) on cluster B: a branch of the AC2 main run
# (08_13_tiedq_seed192) at its step-20 checkpoint, on one 4-node allocation (ws32, the parent's
# training world size, so no resharding). Normally run inline on the first node by
# launch_cluster_b.sh (SP_DRIVER_INLINE=1); SP_DRYRUN=1 only imports runner.py with the final
# environment. ENVS below is the authoritative record of the run's settings.
#
# On first use it constructs the branch: the parent's step-20 checkpoint directory (actor,
# sp_q_optim/, q_state.json, sp_replay_state.json) is copied as a real writable directory
# together with the parent's two delta logs, and the checkpoint tracker is set. The harness
# replays the delta logs up to the branch step to rebuild the replay buffer, critic buffer and
# reference bank. Construction is keyed on the tracker, so later attaches simply resume.
#
# Changes relative to the parent from step 21:
#   1. No local readiness (SP_Q_READY_MODE=global, SP_Q_READY_GLOBAL_LATCH=1): once the global
#      criterion holds (critic error pooled over 5 steps < 0.20), every replay slot is ready
#      and stays so. The per-problem table is still maintained (q/ready_problems) but decides
#      nothing.
#   2. No auditing (SP_Q_AUDIT_DEN=0, SP_Q_AUDIT_CUT=0).
#   3. No grouping on ready slots (SP_Q_TD_ENABLE=1, SP_Q_TD_LANE=short, SP_ADV_ESTIMATOR=
#      sp_segment, SP_Q_TD_CUT_GRAIN=1000, SP_Q_TD_ADMIT_PER_SLOT=8), as in
#      09_01_scratch_qtd_prefix16: 16 single-continuation action chunks at 16 distinct prefixes
#      of one stored trajectory, advantage A_i = v_i - Q(p_i).
# The parent's global criterion was not met at step 20 (it had been met at earlier steps), so
# all slots get grouped, judge-scored rollouts until it first holds in the branch; from then
# on every replay slot is a TD slot. Everything else follows the parent, including b = 10000
# and the critic-side seed 804001. Cluster B specifics: no DeepGEMM cache seeding, the AOT
# warm-up runs in place, and rollout-engine throughput settings (grouped cascade attention,
# prefix affinity, piecewise CUDA graphs, judge max_num_seqs 128) that do not change the
# objective.
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
REPO_ROOT=${AC2_CLUSTER_B_ROOT}/self-play
EXP="$REPO_ROOT/experiments/09_09_globalready_qtd_s192b20_noaudit"
# parent: 08_13_tiedq_seed192, branched at its step-20 checkpoint (see header)
E13="$REPO_ROOT/experiments/08_13_tiedq_seed192"
BRANCH=20
FROZEN="$E13/run_data/checkpoints/global_step_${BRANCH}"
VENV="$REPO_ROOT/.venv"
CACHE=${AC2_CLUSTER_B_ROOT}/.cache/ds4_vllm_023
HF_HOME_DIR=${AC2_CLUSTER_B_SCRATCH}/.cache/huggingface   # shared, READ-ONLY use (HF_HUB_OFFLINE)
HF_HUB_CACHE_DIR="$HF_HOME_DIR/hub"

JOBID="${1:?usage: run_attach_cluster_b.sh <HOLDER_JOBID> [KEY=VAL ...]}"; shift || true

# ---- SINGLE-WRITER GUARD ------------------------------------------------------------------
# Two drivers must never write one checkpoint tree. Two callers can attach to a fresh holder:
# the allocation's own launch step (started from INSIDE the allocation, so its wrapper runs on
# a compute node and is invisible to pgrep on the login node) and an external watcher that
# re-attaches on a poll. If both pass preflight (the AOT warm-up takes minutes, before any
# driver exists), two drivers would write the same run_data. The guard lives HERE because every
# launch path goes through this script; the named job step (sp-driver) is the liveness signal
# regardless of which side launched the driver.
# ---- CROSS-HOLDER attach lock ---------------------------------------------------------
# The per-holder check below is necessary but NOT sufficient: a successor holder can start
# while its predecessor's driver is still running, and each holder's `squeue -s -j $JOBID`
# test only sees its own steps. Two drivers on one run_data leave duplicated dataset steps with
# conflicting contents in the append-only delta logs. The job name cannot discriminate (every
# experiment's driver step is called `sp-driver`), so the lock is scoped by run_data instead:
# whoever attaches records its holder, and a later attach refuses while that holder still has
# a live driver. A lock left by a dead holder is detected and overwritten.
# The lock path uses $EXP, NOT $RUN_DATA_DIR: the latter is assigned further down, so here it
# would expand to "" and silently test /.attach.lock -- a lock that always passes.
_LOCK="$EXP/run_data/.attach.lock"
if [ -f "$_LOCK" ]; then
  _lk_job=$(sed -n 's/^JOBID=//p' "$_LOCK" | head -1)
  if [ -n "${_lk_job:-}" ] && [ "$_lk_job" != "$JOBID" ]; then
    _lk_drv=$(squeue -s -j "$_lk_job" -h -o '%j' 2>/dev/null | grep -c '^sp-driver$' || true)
    if [ "${_lk_drv:-0}" -gt 0 ] && [ "${SP_FORCE_ATTACH:-0}" != "1" ]; then
      echo "[attach] REFUSING: holder $_lk_job already drives this run_data (${_lk_drv} live"
      echo "        sp-driver step). Two holders writing one checkpoint tree is how steps 54-58"
      echo "        and 67-69 ended up with two conflicting delta rows each. Kill that driver"
      echo "        first, or pass SP_FORCE_ATTACH=1 if you have confirmed it is gone."
      exit 1
    fi
    echo "[attach] stale attach lock from holder $_lk_job (no live driver) — taking it over"
  fi
fi

_existing_drv=$(squeue -s -j "$JOBID" -h -o '%j' 2>/dev/null | grep -c '^sp-driver$' || true)
if [ "${_existing_drv:-0}" -gt 0 ] && [ "${SP_FORCE_ATTACH:-0}" != "1" ]; then
  echo "[attach] REFUSING: holder $JOBID already has ${_existing_drv} live sp-driver step(s)."
  echo "        Another attach path got there first; a second driver would mean two writers on"
  echo "        one checkpoint tree. Set SP_FORCE_ATTACH=1 only after confirming the other is dead."
  exit 3
fi
# Trailing KEY=VAL overrides are appended to the driver env far below, but PREFLIGHT runs first
# and consults some of them -- SP_JUDGE_MODEL in particular, whose whole purpose is to bypass
# the hub-cache lookup, so it must be visible before the cache check. Only the
# preflight-relevant keys are lifted (an unrestricted eval of caller args would shadow this
# script's own variables).
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
# Node count comes from the HOLDER, not a constant: a hardcoded NNODES=8 on a smaller
# allocation hangs in ray/torch rendezvous.
NODES_LINE=$(grep '^NODES=' "$INFO" | head -1 | cut -d= -f2- | tr -d "'")
NNODES_H="${SP_FORCE_NNODES:-$(echo $NODES_LINE | wc -w)}"
[ "${NNODES_H:-0}" -ge 1 ] || { echo "[preflight] could not derive node count from $INFO"; exit 1; }
echo "[preflight] holder $JOBID has $NNODES_H node(s): $NODES_LINE"

# ---- DeepGEMM JIT cache seeding: not needed on cluster B ------------------------------------
# The cluster A launch scripts seed the per-node DeepGEMM JIT caches before the driver boots;
# on cluster B the judge runs without deep_gemm, so there is nothing to seed.

# ---- flashinfer AOT preflight: REQUIRED for SP_ROLLOUT_TP>1 -------------------------------
# vLLM's AllReduceRMSFusionPass -> flashinfer get_trtllm_comm_module -> JitSpec.build_and_load
# does `with FileLock(...): build(); load()` UNCONDITIONALLY, once per rank. Every rank of a TP
# group serializes on that one lock and each re-runs ninja, so startup costs TP x build-time --
# TP=2 ~4 min, TP=4 ~8-10 min, TP=8 >17 min, at which point the ranks parked at the c10d store
# barrier blow their 600s budget and the engine WEDGES. All of it at 0% GPU util, because the
# work is nvcc on the CPU. TP=1 does not take this path.
#
# The fix is not a workaround: JitSpec.is_aot short-circuits build_and_load to a plain load(),
# so building each module ONCE single-process and promoting it into FLASHINFER_AOT_DIR gives
# the SAME artifact with compilation fully enabled. Idempotent, so this runs every attach.
# (Alternatives do not help: disable_custom_all_reduce and VLLM_ALLREDUCE_USE_FLASHINFER=0 do
# not avoid the path, and compilation_config mode 0 avoids it only by disabling compilation.)
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
_AOT_WARM="$REPO_ROOT/experiments/09_09_globalready_qtd_s192b20_noaudit/flashinfer_aot_warm.py"
if [ "${SP_ROLLOUT_TP:-4}" -gt 1 ] && [ "${SP_SKIP_AOT_WARM:-0}" != "1" ]; then
  echo "[preflight] flashinfer workspace: $FLASHINFER_WORKSPACE_BASE"
  if [ ! -f "$_AOT_WARM" ]; then
    echo "[preflight] REFUSING: SP_ROLLOUT_TP>1 needs $_AOT_WARM and it is"
    echo "            missing. Without the AOT promotion every rollout engine JIT-rebuilds"
    echo "            flashinfer's trtllm_comm once per rank and startup wedges."
    exit 1
  fi
  echo "[preflight] flashinfer AOT warm (idempotent; required at TP>1)..."
  # The warm-up needs a GPU (arch detection needs a device; a login node has none -> "No
  # supported CUDA architectures"). Under launch_cluster_b.sh this script already runs on the
  # first compute node, so the warm-up runs in place.
  _AOT_OUT="$("$VENV/bin/python" "$_AOT_WARM" 2>&1)" || true   # runs in place (GPU node)
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

RUNNER="$EXP/runner.py"
CKPT="$EXP/run_data/checkpoints"
RUN_DATA_DIR="$EXP/run_data"

# ---- preflight the branch inputs ----
# Tied weights: the critic optimizer state lives in sp_q_optim/, not in a separate q_model/
# directory, so that is what the checkpoint must contain.
[ -f "$RUNNER" ]                      || { echo "[preflight] runner.py missing: $RUNNER"; exit 1; }
# The parent's step-$BRANCH checkpoint and delta logs must be present in the parent's folder
# (08_13_tiedq_seed192/run_data) on this cluster. The frozen step-$BRANCH is needed ONLY when
# branch construction has to run (no resume pointer yet), so the parent checks are gated on the
# same tracker.
if [ ! -f "$CKPT/latest_checkpointed_iteration.txt" ]; then
[ -d "$FROZEN/actor" ]                || { echo "[preflight] step-$BRANCH actor missing: $FROZEN/actor"; exit 1; }
[ -d "$FROZEN/sp_q_optim" ]           || { echo "[preflight] step-$BRANCH sp_q_optim missing (tied Q): $FROZEN"; exit 1; }
[ -f "$FROZEN/q_state.json" ]         || { echo "[preflight] step-$BRANCH q_state.json missing (readiness lives IN-STATE)"; exit 1; }
[ -f "$FROZEN/sp_replay_state.json" ] || { echo "[preflight] step-$BRANCH sp_replay_state.json missing"; exit 1; }
else
  echo "[preflight] resume pointer present ($(cat "$CKPT/latest_checkpointed_iteration.txt")); parent step-$BRANCH not required"
fi

# ---- branch construction (ONE-TIME) ----
# Keyed on the TRACKER, never on global_step_$BRANCH: on an established run retention may have
# pruned the branch-step dir, and re-running construction would clobber the grown delta logs
# with the parent's copies.
mkdir -p "$CKPT" "$RUN_DATA_DIR"
if [ ! -f "$CKPT/latest_checkpointed_iteration.txt" ]; then
  echo "[prep] branch construction (one-time): copy parent step-$BRANCH + BOTH delta logs"
  rm -rf "$CKPT/.g${BRANCH}.tmp"
  # -L and a real dir, never a symlink: retention rmtree's global_step_N recursively.
  cp -rL "$FROZEN" "$CKPT/.g${BRANCH}.tmp"
  chmod -R u+w "$CKPT/.g${BRANCH}.tmp"
  mv "$CKPT/.g${BRANCH}.tmp" "$CKPT/global_step_${BRANCH}"
  # Both logs come across: the harness replays each up to the branch cursor and truncates the
  # tail, reconstructing the exact step-$BRANCH replay buffer and Q FIFO. Without them this would be
  # a weights-only graft with an empty buffer.
  cp "$E13/run_data/q_state_deltas.jsonl"      "$RUN_DATA_DIR/q_state_deltas.jsonl"
  cp "$E13/run_data/replay_buffer_deltas.jsonl" "$RUN_DATA_DIR/replay_buffer_deltas.jsonl"
  echo ${BRANCH} > "$CKPT/latest_checkpointed_iteration.txt"
fi
[ "$(cat "$CKPT/latest_checkpointed_iteration.txt")" -ge ${BRANCH} ] || { echo "[prep] latest < ${BRANCH}?"; exit 1; }
[ ! -L "$CKPT/global_step_${BRANCH}" ] || { echo "[prep] global_step_${BRANCH} must be a real dir, not a symlink"; exit 1; }

# The Q delta log must cover through BRANCH-1: the FIFO/bank rebuild consumes deltas with
# dataset_step < BRANCH, and a short log silently yields a smaller FIFO than the parent had.
"$VENV/bin/python" - "$RUN_DATA_DIR/q_state_deltas.jsonl" ${BRANCH} <<'PYEOF' || { echo "[prep] q delta log does not cover through branch-1"; exit 1; }
import json, sys
path, branch = sys.argv[1], int(sys.argv[2])
steps = [json.loads(l).get("dataset_step", -1) for l in open(path, encoding="utf-8") if l.strip()]
mx = max(steps) if steps else None
assert steps and mx is not None and mx >= branch - 1, \
    f"q_state_deltas: max dataset_step={mx} < {branch - 1}"
print(f"[prep] q_state_deltas: {len(steps)} deltas, max dataset_step={mx} (>= {branch - 1} OK)")
PYEOF

# Judge weights must already be in the local hub cache: the run is HF_HUB_OFFLINE=1, so a
# missing snapshot would otherwise surface as an opaque vLLM init failure ~20 min in.
if [ -n "${SP_JUDGE_MODEL:-}" ]; then
  # Explicit path wins (the runner honors it too). Validate THAT instead of the cache: a
  # directory with a config.json is what vLLM needs, and failing here beats failing in engine
  # init 20 minutes later.
  if [ ! -f "${SP_JUDGE_MODEL%/}/config.json" ]; then
    echo "[preflight] SP_JUDGE_MODEL=$SP_JUDGE_MODEL has no config.json — not a model dir."
    exit 1
  fi
  echo "[preflight] judge: SP_JUDGE_MODEL override -> $SP_JUDGE_MODEL (hub-cache check skipped)"
  _JUDGE_DIR=""
else
_JUDGE_DIR="$HF_HUB_CACHE_DIR/models--${JUDGE_HF_REPO//\//--}"
if [ ! -d "$_JUDGE_DIR/snapshots" ]; then
  echo "[preflight] judge $JUDGE_HF_REPO is not in the hub cache ($_JUDGE_DIR)."
  echo "            fetch it from a login node with network, then re-attach:"
  echo "              HF_HUB_ENABLE_HF_TRANSFER=1 hf download $JUDGE_HF_REPO"
  echo "            (or pass SP_JUDGE_MODEL=<snapshot dir> to point at a manual copy)"
  exit 1
fi
echo "[preflight] judge cache: $(du -sh "$_JUDGE_DIR" 2>/dev/null | cut -f1) in $_JUDGE_DIR"
fi

mkdir -p "$CKPT" "$RUN_DATA_DIR"
# BRANCH RUN: a grafted checkpoint with no metrics history is EXPECTED here on the first
# attach -- that is precisely what branch construction just built. The from-scratch guard the
# parent carries would refuse every first launch of this experiment, so it is replaced by its
# mirror image: the branch step must be PRESENT.
_LATEST="$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)"
if [ "${_LATEST:-0}" -lt "$BRANCH" ]; then
  echo "[preflight] REFUSING: latest ckpt ${_LATEST:-0} < branch $BRANCH — construction did not run."
  exit 1
fi
echo "[preflight] resume state: latest ckpt = ${_LATEST:-0} ($([ "${_LATEST:-0}" -eq 0 ] && echo 'FRESH START' || echo 'resume'))"

# ---- topology check (NOT a reshard) -----------------------------------------------------------
# The actor shards AND O_Q's shards are split across the training world size, and both
# loaders hard-assert it: a checkpoint written at ws64 refuses to load on anything else.
# Compare the checkpoint's own world size against this holder's. At matching topology (the
# normal relaunch) there is nothing to do. On a mismatch, REFUSE: there is no resharder for the
# tied-weight layout (critic optimizer in sp_q_optim/, no separate q_model/), and an untested
# one written at relaunch time is worse than waiting for a correctly sized holder.
if [ "${_LATEST:-0}" -gt 0 ]; then
  _CKPT_DIR="$CKPT/global_step_${_LATEST}"
  # world size == the number of per-rank O_Q shards (the driver writes exactly one per rank)
  _CKPT_WS=$(ls "$_CKPT_DIR"/sp_q_optim/rank_*.pt 2>/dev/null | wc -l | tr -d ' ')
  if [ "${_CKPT_WS:-0}" -eq 0 ]; then
    # fall back to the actor's own shard count before assuming anything
    _CKPT_WS=$(ls "$_CKPT_DIR"/actor/*rank_*.pt 2>/dev/null | wc -l | tr -d ' ')
  fi
  _WANT_WS=$(( NNODES_H * 8 ))
  if [ "${_CKPT_WS:-0}" -eq 0 ]; then
    echo "[preflight] WARNING: could not determine the world size of $_CKPT_DIR"
    echo "            (no sp_q_optim/rank_*.pt and no actor/*rank_*.pt). Proceeding — the"
    echo "            loaders will assert if it is wrong — but check that checkpoint."
  elif [ "$_CKPT_WS" -ne "$_WANT_WS" ]; then
    echo "[preflight] REFUSING: checkpoint $_CKPT_DIR was written at world size $_CKPT_WS,"
    echo "            but this holder gives $_WANT_WS ($NNODES_H nodes x 8). The actor and O_Q"
    echo "            shards are world-size-locked and the loaders will reject them."
    echo "            There is NO resharder for the tied-Q layout: the generic FSDP reshard tool"
    echo "            handles the separate-Q q_model/ layout, which this run does not write."
    echo "            Get a $(( _CKPT_WS / 8 ))-node holder, or write + verify a tied-Q"
    echo "            resharder (sp_q_optim/ + actor/) first. Do not force this."
    exit 1
  else
    echo "[preflight] topology matches: checkpoint and holder are both world size $_CKPT_WS"
  fi
  # sp_q's loader treats a MISSING q_state.json as a legitimate fresh-critic branch point. This
  # branch inherits the parent's critic state: the copied step-20 checkpoint and every later one
  # carry q_state.json, so a missing file means a damaged/incomplete checkpoint. Failing open
  # there would reset readiness and the error window, and -- because delta replay does not
  # advance q_seq_next -- risk REUSING record sequence numbers. Refuse instead.
  if [ ! -f "$_CKPT_DIR/q_state.json" ]; then
    echo "[preflight] REFUSING: $_CKPT_DIR has no q_state.json."
    echo "            This run is not a branch run: every checkpoint after step 0 must carry Q"
    echo "            state. Resuming would silently reset readiness + the MAE window and could"
    echo "            reuse q_seq_next values. Resume an earlier complete checkpoint."
    exit 1
  fi
  [ -d "$_CKPT_DIR/sp_q_optim" ] || {
    echo "[preflight] REFUSING: $_CKPT_DIR has no sp_q_optim/ (the tied-Q optimizer)."
    echo "            A resume without O_Q is fatal by design, not license to"
    echo "            reinitialize it. Resume an earlier complete checkpoint."
    exit 1
  }
  echo "[preflight] checkpoint carries q_state.json + sp_q_optim/"
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
  # Judge requests in flight: 160, the main run's value; the judge runs one TP=8 replica per
  # node (num_replicas = world_size // TP).
  "SP_JUDGE_MAX_INFLIGHT=160" "SP_PASS_POINTS_MIN=6"
  # NO SP_EXPERIMENT_NAME: runner.py defaults it to the directory name, which is what a branch
  # wants. The parent's name would also set WANDB_RUN_ID to the parent's, and the branch's steps
  # would then interleave with the parent's history in W&B. Checkpoint paths derive from
  # SP_RUN_DATA_DIR, not from the name.
  "SP_RUN_DATA_DIR=run_data"
  "VERL_STEP_CACHE_DIR=$EXP/run_data/step_cache"
  # ==== judge: model AND revision pinned (identical in every run, so rewards are comparable) ====
  "SP_JUDGE_HF_REPO=$JUDGE_HF_REPO" "SP_JUDGE_REVISION=$JUDGE_REVISION"
  # ---- training hyperparameters (main-run values) ----
  "SP_REPLAY_ENABLE=1"
  # policy LR: 1.4142e-6 (runner default) * sqrt(2) = 2e-6 for twice the runner-default
  # trained batch (192 vs 96 groups)
  "SP_LR=2e-6" "SP_ENTROPY_COEFF=0.0"
  "SP_ADAPTIVE_ENTROPY=1" "SP_AEC_TARGET_H=0.28" "SP_AEC_DELTA=0.02"
  "SP_AEC_KMAX=0.08" "SP_AEC_KMIN=-0.08" "SP_AEC_KINIT=0.06"
  # ==== batch shape ====
  # 384 problems per step: n_orig = TRAIN_BATCH - REPLAY_N = 192 refill problems, rolled out
  # once and dropped before the loss (SP_SCRATCH_INFLOW_ONLY), plus REPLAY_N = 192 replayed
  # prefixes x 16 continuations = 3,072 trained sequences in 2 PPO minibatches of 96 groups.
  # The refill rollouts feed the replay buffer, the difficulty EMA and the reference bank.
  "SP_TRAIN_BATCH_SIZE=384" "SP_REPLAY_N=192" "SP_PPO_MINI_BATCH=96"
  "SP_SCRATCH_INFLOW_ONLY=1"
  # Prefix cuts: multiples of 10,000 tokens in [0, 0.90 L].
  "SP_REPLAY_CUT_LOW=0" "SP_REPLAY_CUT_HIGH=0.90" "SP_REPLAY_CUT_GRAIN=10000"
  # ==== replay management policy: one global FIFO buffer ====
  "SP_REPLAY_BUCKETING=global" "SP_REPLAY_ADMISSION=ungated" "SP_REPLAY_ROTATION=global_fifo"
  # ==== replay draw: FIFO 256, uniform over QUESTIONS ====
  # BOUND 256 (runner default 128): a larger global FIFO, so more distinct problems are live at
  # draw time.
  #
  # GLOBAL_SAMPLING question (runner default entry). "entry" permutes ENTRIES, which makes a
  # problem's draw probability proportional to how many trajectories it currently holds -- an
  # implicit OCCUPANCY weighting with no difficulty signal in it (and under admission=ungated,
  # occupancy tracks how recently/often a problem entered the inflow lane). "question" permutes
  # the DISTINCT problems and then picks one of that problem's entries uniformly, so every
  # problem in the buffer is equally likely per slot.
  #
  # NOT the same thing as the buffer's difficulty weighting (w = 3 - 2.5*EMA): that lives in
  # the per_question draw only and is unreachable under bucketing=global, in either mode.
  # Watch replay/distinct_questions and replay/question_repeat_factor -- with B1=192 slots and
  # fewer than 192 distinct problems live, some problems appear more than once per step (at
  # different prefixes). At 256 entries that factor is ~1.0-2.3 depending on occupancy.
  "SP_REPLAY_BOUND=256" "SP_REPLAY_GLOBAL_SAMPLING=question"
  # ==== replay buffer: rebuilt by replaying the parent's delta log to the branch step. No
  # SP_REPLAY_SEED_DIR (a cold seed dir would be unused and misleading), and cold bootstrap is
  # OFF, so an empty buffer raises instead of silently turning the branch into a from-scratch
  # run.
  "SP_REPLAY_COLD_BOOTSTRAP=0"
  # A branch IS a graft: runner.py carries its own from-scratch guard (independent of the one
  # in this script) that refuses a checkpoint tree holding step N with an empty metrics.jsonl.
  # That is exactly what branch construction produces on the first attach, so the guard must be
  # told this is intentional rather than removed.
  "SP_ALLOW_GRAFT=1"
  # ==== gates ====
  "SP_Q_REQUIRE_NONZERO=1"
  # critic: the critic buffer and reference bank are rebuilt from the inherited delta log,
  # not from cold seed dirs (which the from-scratch parent used).
  "SP_Q_ENABLE=1"
  # Train BOTH critic prompt variants (SP_Q_TRAIN_NOREF=1): records WITHOUT a resolvable
  # reference are trained via the no-reference variant at weight 1 instead of being skipped --
  # the case for every problem not solved yet (the bank holds only solved problems).
  # q/loss_noref becomes a real number; the critic phase does up to 2x the rows.
  # b = 10000 as in the parent. Auditing off: sp_q_readiness treats SP_Q_AUDIT_DEN <= 0 as "no
  # audit lane at all", so no ready problem is routed to audit and audit_cut never fires;
  # SP_Q_AUDIT_CUT=0 makes the intent explicit rather than relying on that side effect. The
  # critic-side seed equals the parent's (804001).
  "SP_Q_BUDGET_G=10000" "SP_Q_AUDIT_CUT=0" "SP_Q_TRAIN_NOREF=1" "SP_Q_AUDIT_DEN=0"
  "SP_Q_RNG_SEED=804001"
  # ==== critic training ====
  # ---- readiness thresholds. GLOBAL 0.20 (tau_global, critic error pooled over 5 steps)
  # gates only whether readiness may open at all and is looser than the per-problem value,
  # because with empty buffers and the base policy a tighter gate risks never opening.
  # PER-PROBLEM 0.18 (tau_local) decides whether a specific problem's continuations are cut to
  # action chunks and scored by the critic instead of the judge, which is where a wrong call
  # costs reward. SP_Q_READY_THRESH stays as the legacy single knob / default for both.
  "SP_Q_READY_THRESH=0.18"
  "SP_Q_READY_THRESH_GLOBAL=0.2" "SP_Q_READY_THRESH_PROBLEM=0.18"
  # Readiness additionally requires the problem to be SOLVED (present in the add-once reference
  # bank, which only ever fills from a judged-PASSING row). Without it readiness keys purely on
  # the critic's prediction ERROR, so a problem the policy never solves goes ready the moment
  # the critic confidently predicts its z of 0 -- 305 of 1,415 ready problems in the parent
  # were exactly that, and a probe at step 40 found the critic over-predicting their reward by
  # +0.26, worse than a constant. In THIS run the flag only shapes the logged table
  # (q/ready_problems): routing follows the global gate, so it is kept at the value of
  # 09_05_qtd_ready_s192b50_noaudit and decides nothing. q/ready_blocked_no_bank and
  # q/ready_require_bank are emitted automatically -- metrics, not knobs.
  "SP_Q_READY_REQUIRE_BANK=1"
  # Critic buffer 1,920 records (20 steps x 96 groups at the runner-default batch shape);
  # 768 training records per step; a target needs >= 8 valid continuations.
  "SP_Q_FIFO_CAP=1920" "SP_Q_MIN_VALID=8" "SP_Q_TRAIN_N=768"
  "SP_Q_GRAD_CLIP=0.2"
  # order: PPO minibatch 1 -> critic-only step (LEFT APPLIED) -> PPO minibatch 2
  "SP_Q_INTERLEAVE=1" "SP_Q_INTERLEAVE_AFTER=1"
  # step-size control: the persistent halving LR ladder. SP_Q_LR_BASE/SP_Q_RHO_CAP (the
  # non-interleaved movement cap) are NOT set: on the interleaved path they are dead knobs, and
  # exporting them would read as if the cap were still doing something.
  # Q ladder bounds also scale by sqrt(2) with the doubled batch:
  # initial 2e-6 -> 2*sqrt(2)e-6, floor 5e-7 -> 5*sqrt(2)e-7.
  "SP_Q_LR_LADDER=1" "SP_Q_LR_INITIAL=2.8284271247461903e-6" "SP_Q_LR_FLOOR=7.071067811865476e-7"
  # Ratio threshold sqrt(2) (runner default 1.0): the ladder halves the critic LR only when
  # the critic moves more than sqrt(2)x the policy per interleaved step, letting it track a
  # faster target.
  "SP_Q_LR_RATIO_MAX=1.4142135623730951" "SP_Q_LR_BREACH_PATIENCE=2" "SP_Q_LR_REDUCTION_FACTOR=0.5"
  # Persist every critic query (the context it saw and the text it returned).
  "SP_Q_DUMP_WAVE=1"
  # ==== no-group TD on the READY lane (change 3; see header). Read by the harness and the
  # dataset through the Hydra config (+data.sp_q_td_*), so no Ray runtime_env forwarding is
  # needed; SP_ADV_ESTIMATOR is read by the runner on the driver. Verify with the driver log line
  # "[sp_q_td] rewrote K ... slot(s)" and q/td_targets_admitted > 0.
  "SP_Q_TD_ENABLE=1" "SP_Q_TD_LANE=short" "SP_Q_TD_CUT_GRAIN=1000" "SP_Q_TD_ADMIT_PER_SLOT=8"
  "SP_ADV_ESTIMATOR=sp_segment"
  # ==== per-problem readiness OFF (change 1; see header). Read by sp_q_readiness through the
  # Hydra config (+data.sp_q_ready_mode / +data.sp_q_ready_global_latch via runner.py), so no
  # Ray runtime_env forwarding is needed. With TD_LANE=short this means: once the global gate
  # latches, EVERY replay slot is a TD slot ("[sp_q_td] rewrote 192 ready slot(s)"). Verify with
  # "[sp_q] READY MODE = global (latch=1): gate_open=0 latched=0 -> NO replay slots route to Q"
  # on the first attach, then "[sp_q] GLOBAL gate first opened at dataset_step=N ... latched",
  # q/global_ready_effective=1 and q/ready_fraction_sampled = 1.0 from the step after.
  # LATCH=0 would make it a LIVE gate (ready iff the current mae5 < 0.20, can close again).
  "SP_Q_READY_MODE=global" "SP_Q_READY_GLOBAL_LATCH=1"
  # ==== ROLLOUT TENSOR PARALLELISM 4 (inference efficiency; runner default 1) ====
  # Two 4-GPU rollout replicas per node instead of eight single-GPU ones. At TP=1 throughput is
  # limited by replica IMBALANCE (most GPUs idle while one replica finishes its longest
  # sequences; TP=2 gave 967.6 vs 603.6 tok/s/node in a replay benchmark), and TP=4 also shards
  # attention, which took ~75% of rollout GPU time. Measured generation time was 1.6-2.3x lower
  # than at TP=1.
  #
  # Requires the flashinfer AOT preflight above -- at TP>1 the engine cannot start reliably
  # without it.
  "SP_ROLLOUT_TP=4"
  "SP_ROLLOUT_PRIORITY=0"
  # ==== context: 50k response budget. Change these together or not at all -- the runner
  # asserts SP_Q_CTX_LIMIT == 2048 + response + 1248 (2048 + 50000 + 1248 = 53296 = C_Q), so
  # moving the response budget alone crashes at import rather than mis-training, and
  # SP_Q_MAX_TOKEN_LEN must clear C_Q (critic prompts reach it). SP_PPO_MAX_TOKEN_LEN=51200 is
  # ~0.98x a full packed 52,048-token row; the invariant is ONE maximum-length row per GPU
  # micro-batch. These equal the runner's own defaults; this script is still the single place
  # the effective values live.
  "SP_MAX_RESPONSE_LEN=50000"
  "SP_Q_CTX_LIMIT=53296" "SP_Q_MAX_TOKEN_LEN=53360"
  "SP_PPO_MAX_TOKEN_LEN=51200"
  # Rollout KV-cache arena: 0.7 of GPU memory. At 50k/TP=4 it holds ~28 full-length
  # (53,296-token) sequences per engine, so SP_ACTOR_MAX_NUM_SEQS=96 is not binding. LOWER it
  # if "sample_tokens RPC timed out" appears with NO OOM line: that signature means the arena
  # cannot coexist with the resident FSDP actor. Raising it is the riskier direction.
  "SP_ACTOR_GPU_MEM_UTIL=0.7"
  # ==== rollout-engine throughput settings (they do not change the training objective) ====
  # Grouped cascade attention: requests that share a prefix are grouped and attention over the
  # shared prefix is computed once per group, then merged with the per-request suffix attention
  # (LSE merge). The gain grows with coverage (the fraction of rows with a shareable prefix);
  # at low coverage the fixed per-batch cost of the extra kernels and the merge can outweigh it.
  # Measured with the kernel as the only change: single attention calls replayed from this
  # configuration 1.25x faster (70 of 71 batches), a whole decoding step 1.06x faster at
  # coverage 0.375 rising to 1.24x at 1.0.
  # PIECEWISE CUDA graphs are mandatory with cascade, not a tuning choice: a FULL graph captures
  # stock attention and the patched kernel never runs (PIECEWISE itself costs ~0.1%).
  # AFFINITY+PILOT are the precondition -- affinity puts a group's members on one replica, the
  # pilot makes them share ONE physical prefix copy at ref_cnt=n, and without that refcount
  # there is no shared block for the kernel to exploit.
  "SP_GROUPED_CASCADE=1" "SP_ROLLOUT_CUDAGRAPH_MODE=PIECEWISE" "SP_GC_OVERLAP=0"
  "SP_PREFIX_AFFINITY=1" "SP_PREFIX_PILOT=1"
  # Both optional routing modes stay OFF: GROUP_SLOTS was measured to LOWER bulk coverage
  # (0.455 -> 0.280) by throttling grouped rows while prefix-less rows took the seats;
  # UNGROUPED_LANES is unit-tested but has not been used in training.
  "SP_PREFIX_GROUP_SLOTS=0" "SP_PREFIX_UNGROUPED_LANES=0"
  # REQUIRED WITH CASCADE, not optional. At the default 256 the cascade engine's higher
  # resident footprint leaves too little memory for the colocated judge's DeepGEMM fp8 GEMM at
  # wake (CUDA driver error 2 on one engine, whose EngineCore then fails every request routed to
  # it with EngineDeadError). 128 avoids it; judge concurrency costs judge wall clock only.
  # Never fix this with eager mode or a lower judge gpu_memory_utilization.
  "SP_REWARD_MAX_NUM_SEQS=128"
  # SP_ATTN_TIME / SP_GC_TIME are deliberately NOT set: they synchronize a CUDA event every
  # 64th layer call, which is fine for a short measurement and pure overhead in training.
  # ==== NO length penalty -- pinned to 0, not merely omitted.
  # ds4_finegrained_judge reads SP_LENPEN_ENABLE straight from the process environment, the
  # holder does not use --export=NONE, and the runner only forwards the variable when it is
  # non-empty. So submitting from a shell that still exports SP_LENPEN_ENABLE=1 would silently
  # reinstate a length penalty with nothing in the log saying so. Setting it explicitly makes
  # the intended value the one that reaches the judge, and the runner asserts it. Same
  # reasoning for the other objective-changing switches below.
  "SP_LENPEN_ENABLE=0"
  "SP_DIFF_SAMPLING=0" "SP_Q_SEPARATE=0"
  # ==== critic prompt variant: asks for the expected rubric credit within the remaining token
  # budget (recorded in q_state.json; a resume under a different variant is refused).
  "SP_Q_PROMPT_VARIANT=reward_horizon"
  # Tier-1 Q references must be JUDGED-CORRECT trajectories. Under ungated admission the buffer
  # holds failed attempts too, and an unchecked tier-1 lookup would present one to Q as a
  # "reference correct proof" -- outranking the correctness-gated bank. Watch
  # q/ref_rejected_unpassed.
  "SP_Q_REF_REQUIRE_PASS=1"
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

# ---- replay-policy drift gate (SELF-LIMITING) ---------------------------------------------
# The copied parent checkpoint carries the PARENT's replay policy, which records
# cold_bootstrap=True; this branch runs cold_bootstrap=0, so sp_replay's drift guard aborts the
# first attach. The drift is deliberate and the safer direction: cold_bootstrap only ever fires
# when the replay buffer is EMPTY (see sp_replay.py), where True silently back-fills fresh
# problems and False raises. A branch is supposed to inherit a populated buffer from the
# parent's delta log, so if that buffer is empty this run must DIE rather than quietly reinvent
# itself as a from-scratch run under a branch's name.
#
# The override is therefore needed exactly once. sp_replay's own comment is that it is for
# manual repair and must not be left on permanently -- pinning it to 1 would blind every later
# resume to genuine drift. So it is gated on the parent's stamp still being the governing one:
# once we save step >BRANCH, that checkpoint carries OUR policy and the guard re-arms by itself.
if [ "${_LATEST:-0}" -le "$BRANCH" ]; then
  _POL_OVR=1
  echo "[attach] replay policy: OVERRIDE=1 for this attach only (ckpt ${_LATEST:-0} is the"
  echo "        grafted parent stamp; re-arms automatically once step >$BRANCH is saved)"
else
  _POL_OVR=0
  echo "[attach] replay policy: guard ACTIVE (ckpt ${_LATEST:-0} carries this branch's stamp)"
fi
ENVS+=("SP_REPLAY_POLICY_OVERRIDE=$_POL_OVR")

# Caller-supplied KEY=VALUE args win for everything EXCEPT SP_VAL_BEFORE_TRAIN, which a
# relaunch wrapper may hard-code to False; drop that one so the gate above decides.
for kv in "$@"; do
  case "$kv" in SP_VAL_BEFORE_TRAIN=*) continue ;; esac
  ENVS+=("$kv")
done
ENVS+=("SP_VAL_BEFORE_TRAIN=$_VBT")

# ---- SP_DRYRUN: import the runner on the LOGIN NODE with the EXACT env the driver gets ----
# Import-time guards in runner.py (e.g. missing cold-artifact stubs, the graft guard) would
# otherwise cost one full attach cycle each to discover; on CPU they are reachable in ~30s.
# This lives inside the attach rather than in a sibling script on purpose: it consumes the
# same ENVS array, so it cannot drift from what actually launches.
if [ "${SP_DRYRUN:-0}" = "1" ]; then
  echo "[attach] DRYRUN: importing runner with ${#ENVS[@]} env vars (no srun, no driver)"
  ( source "$VENV/bin/activate" && cd "$REPO_ROOT" && \
    env "${ENVS[@]}" python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('runner', '$RUNNER')
m = importlib.util.module_from_spec(spec); sys.modules['runner'] = m
spec.loader.exec_module(m)
print('DRYRUN_IMPORT_OK overrides=%d' % len(getattr(m, 'OVERRIDES', [])))
" )
  rc=$?
  echo "[attach] DRYRUN rc=$rc"
  exit "$rc"
fi

echo "[attach] attempt $A on holder $JOBID (head $HEAD_NODE); log: $LOG"
# --job-name=sp-driver so the holder can SEE this step. Without it the driver inherits the job
# name and is indistinguishable from the four `ray start` sruns, which live for the whole
# allocation -- a liveness check would then read "alive" forever and never detect a dead driver.
# Direct node-side log: the dying driver's traceback never survives srun's IO-abandonment;
# the tee'd file does.
if [ "${SP_DRIVER_INLINE:-0}" = "1" ]; then
  # INLINE (launch_cluster_b.sh): this script already runs as task 0 of the allocation's launch
  # step on the head node, so no inner srun; run the driver in the foreground and return its rc.
  printf 'JOBID=%s\nHOST=%s\nWHEN=%s\nLOG=%s\n' "$JOBID" "$(hostname -s)" "$(date -Is)" "$LOG" > "$EXP/run_data/.attach.lock"
  echo "[attach] INLINE driver on $(hostname -s); log: $LOG"
  ( unset ROCR_VISIBLE_DEVICES; source "$VENV/bin/activate" && source "$REPO_ROOT/experiments/09_09_globalready_qtd_s192b20_noaudit/setup.sh" cuda-toolkit \
    && cd "$REPO_ROOT" && env "${ENVS[@]}" python "$RUNNER" ) > "$LOG" 2>&1
  _rc=$?
  echo "[attach] INLINE driver exited rc=$_rc"
  exit "$_rc"
fi
nohup srun --jobid="$JOBID" --job-name=sp-driver --overlap --nodes=1 --ntasks=1 --mem=0 -w "$HEAD_NODE" \
    bash -c "unset ROCR_VISIBLE_DEVICES; source $VENV/bin/activate && source $REPO_ROOT/experiments/09_09_globalready_qtd_s192b20_noaudit/setup.sh cuda-toolkit && cd $REPO_ROOT && \
        env ${ENVS[*]} python $RUNNER" > "$LOG" 2>&1 &
printf 'JOBID=%s\nHOST=%s\nWHEN=%s\nLOG=%s\n' "$JOBID" "$HEAD_NODE" "$(date -Is)" "$LOG" > "$EXP/run_data/.attach.lock"
echo "[attach] driver login-node pid $! ; tail -f $LOG"
