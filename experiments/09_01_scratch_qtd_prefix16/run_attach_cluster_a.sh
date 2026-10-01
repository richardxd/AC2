#!/usr/bin/env bash
# run_attach_cluster_a.sh <HOLDER_JOBID> [KEY=VAL ...]
# Launch script for AC2 w/o Group & Audit (09_01_scratch_qtd_prefix16) on cluster A.
# Runs the preflight checks (cold artifacts, judge cache, checkpoint topology, flashinfer AOT
# warm-up) and starts runner.py on the head node of a running holder allocation
# (submit_cluster_a_*.sbatch) with the environment in ENVS below, which is the authoritative
# record of the run's settings; trailing KEY=VAL arguments are appended to it.
#
# Mechanism (grep "09_01 DELTA"): SP_Q_TD_ENABLE=1 with SP_ADV_ESTIMATOR=sp_segment. The 16
# copies of a ready replay slot become single-continuation requests at 16 DISTINCT prefixes p_i
# of the same stored trajectory (cut grid SP_Q_TD_CUT_GRAIN=1000 tokens; unready slots keep
# 10,000), each an action chunk of at most b = 10,000 new tokens with advantage
#     A_i = v_i - Q(p_i)
# where v_i is the critic's value at the chunk endpoint (the judge reward if the trajectory ends
# inside the chunk) and Q(p_i) the critic's probe at the row's own prefix. The critic is fit to
# the same endpoint, Q(p_i) <- grid(v_i), with the min_valid floor lowered from 8 to 1 for these
# rows and at most SP_Q_TD_ADMIT_PER_SLOT=8 targets per slot. A row whose probe has no valid
# prediction is dropped (mask 0) rather than trained on its raw reward
# (q/td_dropped_no_baseline). No auditing. Everything else equals AC2 w/o Audit
# (08_31_scratch_g10k_noaudit), including the critic-side seed, so the two runs share their
# random draws until the first problem becomes ready.
#
# Node count comes from the holder (NNODES_H below), never from this script. The launch target
# is 7 nodes (ws56, submit_cluster_a_7node.sbatch), which divides none of the per-step row
# counts, so SP_DP_PAD=1 pads the data-parallel dispatch (see ENVS); the 8-node holder also
# works (padding inert at ws64). train_batch_size is global, so the node count changes only the
# step time. Checkpoint shards are written at the training world size and asserted on load, so
# the first checkpoint fixes the node count.
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
# Cluster A specifics: cluster A paths (scratch repo, venv and caches; see clusters.py
# "cluster_a"), and the srun driver line sources setup.sh cuda-compat (nodes with the 565 driver
# need the CUDA 12.9 forward-compatibility libraries and the redirected HOME) and tees to a
# node-side log (srun abandons its IO channel on shutdown and loses the final traceback).
REPO_ROOT=${AC2_CLUSTER_A_ROOT}/self-play
EXP="$REPO_ROOT/experiments/09_01_scratch_qtd_prefix16"
# No other experiment is referenced: this run starts from the base model with empty buffers.
VENV="$REPO_ROOT/.venv"
CACHE=${AC2_CLUSTER_A_ROOT}/.cache/ds4_vllm_023
HF_HOME_DIR=${AC2_CLUSTER_A_ROOT}/hf_home
HF_HUB_CACHE_DIR="$HF_HOME_DIR/hub"

JOBID="${1:?usage: run_attach_cluster_b.sh <HOLDER_JOBID> [KEY=VAL ...]}"; shift || true
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

# ---- DeepGEMM JIT cache seeding ------------------------------------------------------------
# dg_jit_nodes/<host> caches are PER-NODE; an engine on a never-seen node JIT-compiles and, on
# these nodes, FAILS ("NVCC compilation failed"), costing one crashed attempt per new holder.
# The cache lives on shared Lustre, so seed every holder node from the LARGEST existing bucket
# right here on the login node, before the driver boots. Idempotent: --ignore-existing never
# overwrites.
_DG_BASE="$CACHE/dg_jit_nodes"
if [ -d "$_DG_BASE" ]; then
  _DONOR=$(for _d in "$_DG_BASE"/*/cache; do [ -d "$_d" ] && echo "$(ls "$_d" 2>/dev/null | wc -l) $_d"; done | sort -rn | head -1 | awk '{print $2}')
  if [ -n "${_DONOR:-}" ]; then
    for _n in $NODES_LINE; do
      mkdir -p "$_DG_BASE/$_n/cache"
      rsync -a --ignore-existing "$_DONOR/" "$_DG_BASE/$_n/cache/" 2>/dev/null || true
      echo "[preflight] DG cache: $_n has $(ls "$_DG_BASE/$_n/cache" 2>/dev/null | wc -l) kernels (donor: $_DONOR)"
    done
  else
    echo "[preflight] DG cache: no donor bucket found under $_DG_BASE (first-ever run?) — cold compiles possible"
  fi
fi

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
_AOT_WARM="$REPO_ROOT/experiments/09_01_scratch_qtd_prefix16/flashinfer_aot_warm.py"
if [ "${SP_ROLLOUT_TP:-4}" -gt 1 ] && [ "${SP_SKIP_AOT_WARM:-0}" != "1" ]; then
  echo "[preflight] flashinfer workspace: $FLASHINFER_WORKSPACE_BASE"
  if [ ! -f "$_AOT_WARM" ]; then
    echo "[preflight] REFUSING: SP_ROLLOUT_TP>1 needs $_AOT_WARM and it is"
    echo "            missing. Without the AOT promotion every rollout engine JIT-rebuilds"
    echo "            flashinfer's trtllm_comm once per rank and startup wedges."
    exit 1
  fi
  echo "[preflight] flashinfer AOT warm (idempotent; required at TP>1)..."
  # The warm-up must run on a GPU node (arch detection needs a device; the login node has
  # none -> "No supported CUDA architectures"), with setup.sh cuda-compat sourced.
  _AOT_OUT="$(srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
      bash -c "source $EXP/setup.sh cuda-compat; FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE XDG_CACHE_HOME=$XDG_CACHE_HOME TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR $VENV/bin/python $_AOT_WARM" 2>&1)" || true
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
# The cold artifacts are a ONE-TIME build: three EMPTY seeds (replay buffer, Q FIFO, reference
# bank) with valid manifests. Their loaders require a manifest and one shard even when the shard
# holds zero records; without them the harness raises FileNotFoundError deep inside dataset
# construction, ~20 minutes into engine startup.
[ -f "$SEED/replay_buffer_manifest.json" ] || {
  echo "[preflight] cold replay seed missing: $SEED — run once:"
  echo "  $VENV/bin/python $EXP/build_cold_artifacts.py --out $EXP"
  exit 1
}
for _mf in q_seed_manifest.json reference_bank_manifest.json; do
  [ -f "$EXP/$_mf" ] || {
    echo "[preflight] $_mf missing — run once:"
    echo "  $VENV/bin/python $EXP/build_cold_artifacts.py --out $EXP"
    exit 1
  }
done
# ... and ALL THREE must actually be EMPTY, counted -- not merely present: a stale non-empty
# q_seed/ or reference_bank/ from a copied directory would seed the critic with another
# policy's data while every log line still said "cold".
_count_records() {  # $1 = dir holding shard_*.jsonl
  cat "$1"/shard_*.jsonl 2>/dev/null | grep -c . || true
}
_N_REPLAY=$(_count_records "$SEED/replay_buffer")
_N_QSEED=$(_count_records "$EXP/q_seed")
_N_BANK=$(_count_records "$EXP/reference_bank")
if [ "$COLD" = "1" ]; then
  _BAD=0
  for _pair in "replay seed:${_N_REPLAY:-0}" "q seed:${_N_QSEED:-0}" "reference bank:${_N_BANK:-0}"; do
    if [ "${_pair##*:}" -ne 0 ]; then
      echo "[preflight] REFUSING: ${_pair%%:*} holds ${_pair##*:} records, but this run is the"
      echo "            NOTHING-INHERITED arm (all three artifacts must be empty)."
      _BAD=1
    fi
  done
  if [ "$_BAD" -ne 0 ]; then
    echo "            Rebuild with: $VENV/bin/python $EXP/build_cold_artifacts.py --out $EXP"
    echo "            Or pass SP_REPLAY_COLD_BOOTSTRAP=0 to run a warm-start config on purpose."
    exit 1
  fi
  echo "[preflight] cold artifacts VERIFIED empty: replay seed ${_N_REPLAY:-0}, q seed ${_N_QSEED:-0}, reference bank ${_N_BANK:-0} records"
else
  echo "[preflight] SP_REPLAY_COLD_BOOTSTRAP=0: warm-start config, artifacts hold ${_N_REPLAY:-0} / ${_N_QSEED:-0} / ${_N_BANK:-0} records (replay / q seed / bank)"
fi
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
# FROM-SCRATCH GUARD. A checkpoint tree with no metrics history means someone
# grafted weights in; that would silently make this a branch run. The runner refuses too
# (SP_ALLOW_GRAFT), but say it here where it is cheap to see.
_LATEST="$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)"
if [ "${_LATEST:-0}" -gt 0 ] && [ ! -s "$RUN_DATA_DIR/metrics.jsonl" ]; then
  echo "[preflight] REFUSING: $CKPT holds step $_LATEST but metrics.jsonl is empty — that is a"
  echo "            grafted checkpoint, and this run trains from scratch. Pass SP_ALLOW_GRAFT=1"
  echo "            only if you are recovering a run whose metrics file was lost."
  [ "${SP_ALLOW_GRAFT:-0}" = "1" ] || exit 1
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
  # sp_q's loader treats a MISSING q_state.json as a legitimate fresh-critic branch point (it
  # is, for a branch that starts a new critic). This run has no branch point: every checkpoint
  # from step 1 on writes critic state, so a missing file means a damaged/incomplete checkpoint.
  # Failing open there would reset readiness and the error window, and -- because delta replay
  # does not advance q_seq_next -- risk REUSING record sequence numbers. Refuse instead.
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
  # cluster A: the FineProofs data (train/test parquet, val_map, rubric_map) lives under the
  # redirected HOME that setup.sh cuda-compat sets.
  "SELF_PLAY_DATA_DIR=${AC2_CLUSTER_A_ROOT}/fakehome/data/fineproofs"
  "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1" "WANDB_MODE=offline"
  "XDG_CACHE_HOME=$XDG_CACHE_HOME" "TRITON_CACHE_DIR=$CACHE/triton" "TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor"
  # Pin the DG JIT base to the SAME path the preflight seeds ($CACHE/dg_jit_nodes). Without
  # this the runner defaults it to $XDG_CACHE_HOME/dg_jit_nodes -- a DIFFERENT, EMPTY tree
  # when XDG_CACHE_HOME=.cache while the seeder fills .cache/ds4_vllm_023/dg_jit_nodes -- so
  # every judge engine would see a cold cache and JIT-compile the mhc/fp8 kernels, and that
  # compile fails on these nodes ("NVCC compilation failed"). A seeded cache means no compiles.
  "SP_DG_JIT_CACHE_BASE=$CACHE/dg_jit_nodes"
  "VLLM_CACHE_ROOT=$CACHE/vllm" "TORCH_HOME=$CACHE/torch" "CUDA_CACHE_PATH=$CACHE/nv"
  "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR" "FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE"
  "MPLCONFIGDIR=$CACHE/matplotlib" "DO_NOT_TRACK=1" "VLLM_NO_USAGE_STATS=1" "VERL_STAGGER_ENGINE_INIT=1"
  # judge replicas = world_size // TP = one per node; 20 requests in flight PER REPLICA
  # (140 on 7 nodes, 160 on 8).
  "SP_JUDGE_MAX_INFLIGHT=$(( NNODES_H * 20 ))" "SP_PASS_POINTS_MIN=6"
  # 7-node topology (ws56): NONE of this run's per-step row counts divides 56
  # (pre-drop 3264, trained 3072, minibatch 1536 -- no factor of 7 in 384/192/96 x 16). With
  # SP_DP_PAD=1 verl pads every dp dispatch with zero-loss-mask duplicates (exactly zero
  # contribution to the token-mean loss and its all-reduced denominator) and the driver assigns
  # minibatch membership so both PPO minibatches still hold exactly 1536 REAL sequences. The
  # training shape does not move; no-group TD keeps the row count (16 singletons per ready slot)
  # and its per-copy rewrite runs before the pad, which is stripped before advantage estimation.
  # Inert on a dividing world size (8 nodes): the runner prints "[sp_dp_pad] ... inert" and the
  # unpadded code path runs unchanged. Read inside the TaskRunner actor, so the runner
  # forwards it into the Ray runtime_env.
  "SP_DP_PAD=1"
  # Readiness also requires the problem to be SOLVED (present in the add-once reference bank,
  # which only fills from a judge-passing rollout). Without it a never-solved problem can turn
  # ready once the critic confidently predicts ~0 for it: in the main run 305 of 1,415 ready
  # problems had never been solved, and a probe at step 40 found the critic over-predicting the
  # reward of action chunks by +0.26, worse than a constant predictor.
  # This run has no auditing, so nothing downstream would catch that failure mode: the gate
  # matters more here than in an audited run. q/ready_blocked_no_bank and
  # q/ready_require_bank are emitted automatically -- if the former stays 0, the flag is
  # doing nothing.
  "SP_Q_READY_REQUIRE_BANK=1"
  # This run's OWN experiment name: it is not a continuation of any wandb run.
  "SP_EXPERIMENT_NAME=09_01_scratch_qtd_prefix16" "SP_RUN_DATA_DIR=run_data"
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
  # Admission is UNGATED, as in the main run (08_26_scratch_correctonly is the variant that
  # admits only judged-correct trajectories). All 192 refill rollouts are admitted every step,
  # so the buffer reaches its 256 bound by ~step 2 and turns over every ~1.33 steps.
  "SP_REPLAY_BUCKETING=global" "SP_REPLAY_ADMISSION=ungated" "SP_REPLAY_ROTATION=global_fifo"
  # Global FIFO of 256 trajectories with a uniform draw over DISTINCT PROBLEMS ("question"
  # mode permutes distinct problems, then picks one entry of that problem) instead of the
  # runner default (bound 128, uniform over entries). Watch replay/distinct_questions and
  # replay/question_repeat_factor.
  "SP_REPLAY_BOUND=256" "SP_REPLAY_GLOBAL_SAMPLING=question"
  # ==== cold start: EMPTY buffer. The seed carries zero entries, and the cold-bootstrap flag
  # is what lets the global draw report a full shortfall at step 1 instead of raising
  # "sp_replay global buffer is empty".
  "SP_REPLAY_SEED_DIR=$SEED"
  "SP_REPLAY_COLD_BOOTSTRAP=1"
  # ==== gates ====
  "SP_Q_REQUIRE_NONZERO=1"
  # ---- critic: empty critic-buffer seed and reference bank (build_cold_artifacts.py) ----
  "SP_Q_ENABLE=1" "SP_Q_SEED_DIR=$EXP" "SP_Q_BANK_DIR=$EXP"
  # Train BOTH critic prompt variants (SP_Q_TRAIN_NOREF=1): records WITHOUT a resolvable
  # reference are trained via the no-reference variant at weight 1 instead of being skipped --
  # the case for every problem not solved yet (the bank holds only solved problems).
  # q/loss_noref becomes a real number; the critic phase does up to 2x the rows.
  # Auditing off (DEN=0/CUT=0) and b = 10000 are the values of AC2 w/o Audit
  # (08_31_scratch_g10k_noaudit); every q/audit_* panel is blank in both runs.
  "SP_Q_BUDGET_G=10000" "SP_Q_AUDIT_CUT=0" "SP_Q_TRAIN_NOREF=1" "SP_Q_AUDIT_DEN=0"
  # ==== 09_01 DELTA A: no-group TD on ready problems ====
  # td_enable flips the whole mechanism (dataset stamps distinct cuts, trainer rewrites the
  # copies, probe relays into the seg stamps); the estimator that consumes the stamps is
  # SP_ADV_ESTIMATOR=sp_segment, set below with the science switches. td_siblings is pinned
  # to rollout.n by the runner, not exported here, so the two cannot drift.
  "SP_Q_TD_ENABLE=1" "SP_Q_TD_CUT_GRAIN=1000" "SP_Q_TD_ADMIT_PER_SLOT=8"
  # HELD EQUAL to AC2 w/o Audit (08_31_scratch_g10k_noaudit) ON PURPOSE: from scratch at an
  # identical configuration, the two runs share every draw -- cold_fill, bank_pick,
  # q_train_sample, routing -- until the first problem goes ready, so early divergence measures
  # the mechanism, not the random draws. The TD cut draws use their own rng purpose ("td_cut"),
  # which 08_31 never consumes, so holding the seed equal costs nothing there either.
  # sp_replay's own draw seed (714001) is the runner default.
  "SP_Q_RNG_SEED=831001"
  # ==== critic training ====
  # ---- readiness thresholds. GLOBAL 0.20 (tau_global, critic error pooled over 5 steps)
  # gates only whether readiness may open at all and is looser than the per-problem value,
  # because with empty buffers and the base policy a tighter gate risks never opening.
  # PER-PROBLEM 0.18 (tau_local) decides whether a specific problem's continuations are cut to
  # action chunks and scored by the critic instead of the judge, which is where a wrong call
  # costs reward. SP_Q_READY_THRESH stays as the legacy single knob / default for both.
  "SP_Q_READY_THRESH=0.18"
  "SP_Q_READY_THRESH_GLOBAL=0.2" "SP_Q_READY_THRESH_PROBLEM=0.18"
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
  # ==== NO length penalty -- pinned to 0, not merely omitted.
  # ds4_finegrained_judge reads SP_LENPEN_ENABLE straight from the process environment, the
  # holder does not use --export=NONE, and the runner only forwards the variable when it is
  # non-empty. So submitting from a shell that still exports SP_LENPEN_ENABLE=1 would silently
  # reinstate a length penalty with nothing in the log saying so. Setting it explicitly makes
  # the intended value the one that reaches the judge, and the runner asserts it. Same
  # reasoning for the other objective-changing switches below.
  "SP_LENPEN_ENABLE=0"
  "SP_DIFF_SAMPLING=0" "SP_Q_SEPARATE=0" "SP_REPLAY_POLICY_OVERRIDE=0"
  # The TD stamps' consumer. Pinned explicitly for the same reason as SP_LENPEN_ENABLE:
  # a shell that still exports SP_ADV_ESTIMATOR=grpo from another experiment would leave the
  # TD stamps unconsumed and hand every ready singleton its RAW reward as advantage -- the
  # runner refuses that pairing at import, but only if the value actually reaches it.
  # SP_Q_SEG_ENABLE stays 0 (default): segment stamps and TD are mutually exclusive.
  "SP_ADV_ESTIMATOR=sp_segment"
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

# Caller-supplied KEY=VALUE args win for everything EXCEPT SP_VAL_BEFORE_TRAIN, which a
# relaunch wrapper may hard-code to False; drop that one so the gate above decides.
for kv in "$@"; do
  case "$kv" in SP_VAL_BEFORE_TRAIN=*) continue ;; esac
  ENVS+=("$kv")
done
ENVS+=("SP_VAL_BEFORE_TRAIN=$_VBT")

echo "[attach] attempt $A on holder $JOBID (head $HEAD_NODE); log: $LOG"
# Direct node-side log: the dying driver's traceback never survives srun's IO-abandonment;
# the tee'd file does.
DIRECT_LOG="$EXP/driver_direct_attempt${A}.log"
nohup srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 --mem=0 -w "$HEAD_NODE" \
    bash -c "unset ROCR_VISIBLE_DEVICES; source $EXP/setup.sh cuda-compat; source $VENV/bin/activate && source $REPO_ROOT/experiments/09_01_scratch_qtd_prefix16/setup.sh cuda-toolkit && cd $REPO_ROOT && \
        env ${ENVS[*]} python $RUNNER 2>&1 | tee $DIRECT_LOG" > "$LOG" 2>&1 &
echo "[attach] driver login-node pid $! ; tail -f $LOG"
