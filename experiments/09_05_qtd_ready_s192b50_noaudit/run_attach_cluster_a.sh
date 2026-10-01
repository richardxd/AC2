#!/usr/bin/env bash
# run_attach_cluster_a.sh <HOLDER_JOBID> [KEY=VAL ...] -- 09_05_qtd_ready_s192b50_noaudit: BRANCH of
# 08_13_tiedq_seed192 at its step-50 checkpoint, 4 cluster A nodes (ws32, the parent's native
# training world size, no reshard). Two deltas vs the parent:
#
#   1. NO AUDIT LANE: SP_Q_AUDIT_DEN=0 / SP_Q_AUDIT_CUT=0, the configuration of
#      08_26_s192b40_g10k_noaudit.
#   2. NO-GROUP TD ON THE READY LANE (the 09_01_scratch_qtd_prefix16 mechanism, unchanged):
#      SP_Q_TD_ENABLE=1, SP_Q_TD_LANE=short (+ SP_ADV_ESTIMATOR=sp_segment, the stamp consumer;
#      SP_Q_TD_CUT_GRAIN=1000, entailed; SP_Q_TD_ADMIT_PER_SLOT=8). A READY replay slot no longer
#      runs a 16-sibling group from one prefix: its 16 copies become 16 single-rollout requests at
#      16 DISTINCT cuts of the stored trajectory, each capped at its own prefix + g and scored by
#      Q, and each row's advantage is A_i = r_i - Q(p_i): the consumed Q at the cut (or the judge
#      score if the row finished before the cap) minus the Q readiness probe at its own prefix.
#      NOT-ready slots keep the grouped full-budget lane untouched. Q's admitted records from TD
#      rows are one-macro-step backups Q(p_i) <- grid(Q(p_i + g)) (terminal where the row
#      finished), floor 1 valid member, capped at 8 per slot.
#
#   At step 50 the parent's ready fraction was 0.39, so ~75 of 192 replay slots per step take
#   the mechanism, and the share grows as readiness grows. Same trained row count and token budget
#   as the parent per slot; prefill is no longer shared across the 16 copies.
#
#   Everything else is the parent's: g = SP_Q_BUDGET_G = 10000 (the action-chunk length), seed
#   804001, tied Q, the interleaved Q update with its LR ladder, judge and its pinned revision,
#   50k context, no length penalty, batch 384/192/96. Comparison runs: 08_13 itself from step 50
#   (audit on, grouped), and 08_26_s192b40_g10k_noaudit (audit off, grouped) from its own branch.
#
# Branch mechanics are those of 08_26_s192b40_g10k_noaudit (copy the frozen parent step dir as a
# REAL writable directory plus BOTH delta logs, then set the tracker; keyed on the tracker, never
# on global_step_50).
# JUDGE: the parent's, unchanged -- deepseek-ai/DeepSeek-V4-Flash at revision 60d8d707.
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
# ============================ CLUSTER A version ============================================
# The science ENVS are identical to run_attach_cluster_b.sh (keep the two ENVS arrays in sync);
# the differences are infrastructure only:
#   * cluster A paths (cluster A scratch repo/venv/caches, fakehome fineproofs)
#   * DeepGEMM JIT cache seeding: dg_jit_nodes/<host> is PER-NODE, and an engine on a
#     never-seen node JIT-compiles and dies ("NVCC compilation failed") unless seeded here.
#   * SP_DG_JIT_CACHE_BASE pinned to the SAME path the preflight seeds (the runner would
#     otherwise derive it from XDG_CACHE_HOME and read an EMPTY tree).
#   * flashinfer AOT warm runs ON A GPU NODE via srun (the login node has no device, so arch
#     detection fails).
#   * the driver srun sources setup.sh cuda-compat (mixed 565-driver nodes need the CUDA 12.9
#     forward-compat libs + the fakehome HOME) and tees to a node-side direct log (srun
#     abandons its IO channel on shutdown and eats the dying traceback).
# ==========================================================================================
REPO_ROOT=${AC2_CLUSTER_A_ROOT}/self-play
EXP="$REPO_ROOT/experiments/09_05_qtd_ready_s192b50_noaudit"
# parent: 08_13_tiedq_seed192, branched at its step-50 checkpoint
E13="$REPO_ROOT/experiments/08_13_tiedq_seed192"
BRANCH=50
FROZEN="$E13/run_data/checkpoints/global_step_${BRANCH}"
VENV="$REPO_ROOT/.venv"
CACHE=${AC2_CLUSTER_A_ROOT}/.cache/ds4_vllm_023
HF_HOME_DIR=${AC2_CLUSTER_A_ROOT}/hf_home
HF_HUB_CACHE_DIR="$HF_HOME_DIR/hub"

JOBID="${1:?usage: run_attach_cluster_b.sh <HOLDER_JOBID> [KEY=VAL ...]}"; shift || true

# ---- SINGLE-WRITER GUARD ------------------------------------------------------------------
# TWO independent callers attach on a fresh holder and they race:
#   1. submit_cluster_b_4node.sbatch SELF-ATTACHes ~45s after boot;
#   2. an external login-node watcher (if one is used) attaches on its next poll, with no
#      driver check.
# A watcher cannot see caller 1 with pgrep -- that driver is launched from INSIDE the
# allocation, so its wrapper lives on a compute node, not the login node. If a watcher polls
# while caller 1 is still in preflight (AOT warm takes minutes, before any driver exists), both
# proceed and two drivers write one checkpoint tree.
#
# The guard lives HERE because this script is the single choke point both paths call, and it is
# re-read on every invocation. The named job step is the liveness signal that works regardless
# of which side launched the driver.
# ---- CROSS-HOLDER attach lock ---------------------------------------------------------
# The per-holder check below is necessary but NOT sufficient: two holders of one experiment can
# RUN at once (a successor whose sbatch auto-attaches the moment it STARTS, not when its
# predecessor ends). Each passes its own `squeue -s -j $JOBID` test because a different
# holder's driver is invisible to it, both drive the same run_data, and the append-only delta
# logs get duplicated dataset_steps with DIFFERENT contents. Job name cannot discriminate
# (every experiment's driver step is called `sp-driver`), so the lock is scoped by run_data
# instead: whoever attaches records its holder, and a later attach refuses while that holder
# still has a live driver. A holder that died leaves a stale lock,
# which is detected and overwritten rather than requiring manual cleanup.
# $EXP, NOT $RUN_DATA_DIR: the latter is assigned well below this check, so using it here
# would expand to "" and silently test /.attach.lock -- a lock that always passes. Exactly the
# fail-open this guard exists to prevent.
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
# the hub-cache lookup. Without this, `run_attach_cluster_b.sh <jid> SP_JUDGE_MODEL=/path` failed the
# cache check before the override was ever read. Only the preflight-relevant keys are lifted
# (an unrestricted eval of caller args would shadow this script's own variables).
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

# ---- DeepGEMM JIT cache seeding (CLUSTER A) ------------------------------------------------
# dg_jit_nodes/<host> caches are PER-NODE; an engine on a never-seen node JIT-compiles and
# FAILS ("NVCC compilation failed"). The cache lives on shared scratch, so seed every holder
# node from the LARGEST existing bucket here on the login node, before the driver boots.
# Idempotent: --ignore-existing never overwrites.
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
    echo "[preflight] DG cache: no donor bucket under $_DG_BASE (first-ever run?) -- cold compiles possible"
  fi
fi

# ---- flashinfer AOT preflight: REQUIRED for SP_ROLLOUT_TP>1 -------------------------------
# vLLM's AllReduceRMSFusionPass -> flashinfer get_trtllm_comm_module -> JitSpec.build_and_load
# does `with FileLock(...): build(); load()` UNCONDITIONALLY, once per rank. Every rank of a TP
# group serializes on that one lock and each re-runs ninja, so startup costs TP x build-time --
# TP=2 ~4 min, TP=4 ~8-10 min, TP=8 >17 min, at which point the ranks parked at the c10d store
# barrier blow their 600s budget and the engine WEDGES. All of it at 0% GPU util, because the
# work is nvcc on the CPU. TP=1 never enters this path.
#
# The fix is not a workaround: JitSpec.is_aot short-circuits build_and_load to a plain load(),
# so building each module ONCE single-process and promoting it into FLASHINFER_AOT_DIR gives
# the SAME artifact with compilation fully enabled. Idempotent, so this runs every attach.
# (Alternatives that do not help: disable_custom_all_reduce and VLLM_ALLREDUCE_USE_FLASHINFER=0
# do not avoid the path, and compilation_config mode 0 avoids it only by disabling compilation.)
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
_AOT_WARM="$REPO_ROOT/experiments/09_05_qtd_ready_s192b50_noaudit/flashinfer_aot_warm.py"
if [ "${SP_ROLLOUT_TP:-4}" -gt 1 ] && [ "${SP_SKIP_AOT_WARM:-0}" != "1" ]; then
  echo "[preflight] flashinfer workspace: $FLASHINFER_WORKSPACE_BASE"
  if [ ! -f "$_AOT_WARM" ]; then
    echo "[preflight] REFUSING: SP_ROLLOUT_TP>1 needs $_AOT_WARM and it is"
    echo "            missing. Without the AOT promotion every rollout engine JIT-rebuilds"
    echo "            flashinfer's trtllm_comm once per rank and startup wedges."
    exit 1
  fi
  echo "[preflight] flashinfer AOT warm (idempotent; required at TP>1)..."
  # CLUSTER A: the warm must run on a GPU node (arch detection needs a device; the login node
  # has none -> "No supported CUDA architectures"), with the shim sourced.
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

RUNNER="$EXP/runner.py"
CKPT="$EXP/run_data/checkpoints"
RUN_DATA_DIR="$EXP/run_data"

# ---- preflight the branch inputs ----
# TIED Q: the Q optimiser state lives in sp_q_optim/, NOT in a separate q_model/ dir (that is
# the separate-Q layout). Checking for q_model here would fail on a perfectly good checkpoint.
[ -f "$RUNNER" ]                      || { echo "[preflight] runner.py missing: $RUNNER"; exit 1; }
[ -d "$FROZEN/actor" ]                || { echo "[preflight] step-$BRANCH actor missing: $FROZEN/actor"; exit 1; }
[ -d "$FROZEN/sp_q_optim" ]           || { echo "[preflight] step-$BRANCH sp_q_optim missing (tied Q): $FROZEN"; exit 1; }
[ -f "$FROZEN/q_state.json" ]         || { echo "[preflight] step-$BRANCH q_state.json missing (readiness lives IN-STATE)"; exit 1; }
[ -f "$FROZEN/sp_replay_state.json" ] || { echo "[preflight] step-$BRANCH sp_replay_state.json missing"; exit 1; }

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
  # tail, reconstructing the exact branch-step replay buffer and Q FIFO. Without them this would
  # be a weights-only graft with an empty buffer.
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

# ---- topology check (NOT an unconditional reshard) ---------------------------------------
# The actor shards AND O_Q's shards are sharded across the training world size, and both
# loaders hard-assert it: a checkpoint written at one world size refuses to load on any other.
#
# There is no resharder for the tied-Q layout (resharding helpers for the SEPARATE-Q layout
# hard-require a `q_model/` directory, which this tied-Q run never writes; its Q optimizer
# lives in `sp_q_optim/`). So: compare the checkpoint's own world size against this holder's
# and only act on a real mismatch. At matching topology (the normal roll) there is nothing to
# do. On a mismatch, REFUSE -- an untested resharder at relaunch time is worse than waiting
# for a correctly sized holder.
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
  # sp_q's loader treats a MISSING q_state.json as a legitimate fresh-Q branch point. Here
  # every checkpoint carries Q state (the copied branch step included), so a missing file
  # means a damaged/incomplete checkpoint. Failing open there would reset readiness and the
  # MAE window, and -- because delta replay does not advance q_seq_next -- risk REUSING record
  # sequence numbers. Refuse instead.
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
  "SELF_PLAY_DATA_DIR=${AC2_CLUSTER_A_ROOT}/fakehome/data/fineproofs"
  "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1" "WANDB_MODE=offline"
  "XDG_CACHE_HOME=$XDG_CACHE_HOME" "TRITON_CACHE_DIR=$CACHE/triton" "TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor"
  # Pin the DG JIT base to the SAME path the preflight seeds. Without this the runner derives
  # it from XDG_CACHE_HOME -- a DIFFERENT, EMPTY tree -- so every judge engine sees a cold cache
  # and JIT-compiles, and the compile itself dies on these nodes.
  "SP_DG_JIT_CACHE_BASE=$CACHE/dg_jit_nodes"
  "VLLM_CACHE_ROOT=$CACHE/vllm" "TORCH_HOME=$CACHE/torch" "CUDA_CACHE_PATH=$CACHE/nv"
  "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR" "FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE"
  "MPLCONFIGDIR=$CACHE/matplotlib" "DO_NOT_TRACK=1" "VLLM_NO_USAGE_STATS=1" "VERL_STAGGER_ENGINE_INIT=1"
  # One TP=8 judge replica per node (num_replicas = world_size // TP). SP_JUDGE_MAX_INFLIGHT
  # caps concurrent judge calls per reward-worker process (see ac2/rewards/prover_judge.py).
  "SP_JUDGE_MAX_INFLIGHT=160" "SP_PASS_POINTS_MIN=6"
  # This run's OWN experiment name: it is not a continuation of any wandb run.
  # NO SP_EXPERIMENT_NAME: runner.py defaults it to the directory name, which is what a branch
  # wants. The parent's name would also set WANDB_RUN_ID to the parent's, so on any sync this
  # branch's post-branch steps would interleave into the parent's history. Checkpoint paths
  # derive from SP_RUN_DATA_DIR, not from the name.
  "SP_RUN_DATA_DIR=run_data"
  "VERL_STEP_CACHE_DIR=$EXP/run_data/step_cache"
  # ==== judge: the parent's model AND revision (pinned so it cannot drift) ====
  "SP_JUDGE_HF_REPO=$JUDGE_HF_REPO" "SP_JUDGE_REVISION=$JUDGE_REVISION"
  # ---- training hyperparameters (the parent's) ----
  "SP_REPLAY_ENABLE=1"
  # policy LR: 1.4142e-6 * sqrt(2) = 2e-6, scaled by sqrt(2) for the doubled batch
  "SP_LR=2e-6" "SP_ENTROPY_COEFF=0.0"
  "SP_ADAPTIVE_ENTROPY=1" "SP_AEC_TARGET_H=0.28" "SP_AEC_DELTA=0.02"
  "SP_AEC_KMAX=0.08" "SP_AEC_KMIN=-0.08" "SP_AEC_KINIT=0.06"
  # ==== batch shape ====
  # 384 problems per step: SP_REPLAY_N=192 replayed prefixes (trained) + 192 fresh problems in
  # the inflow-only lane (rolled out once to refill the buffer, not trained). Mini 96 gives 2
  # PPO minibatches per step. The policy and critic LRs are scaled by sqrt(2) relative to a
  # 96-prefix batch.
  "SP_TRAIN_BATCH_SIZE=384" "SP_REPLAY_N=192" "SP_PPO_MINI_BATCH=96"
  "SP_SCRATCH_INFLOW_ONLY=1"
  # Prefix cuts: multiples of 10k tokens, at most 0.90 of the stored trajectory.
  "SP_REPLAY_CUT_LOW=0" "SP_REPLAY_CUT_HIGH=0.90" "SP_REPLAY_CUT_GRAIN=10000"
  # ==== replay management policy: one global FIFO, admission regardless of correctness ====
  "SP_REPLAY_BUCKETING=global" "SP_REPLAY_ADMISSION=ungated" "SP_REPLAY_ROTATION=global_fifo"
  # ==== replay draw: FIFO 256, uniform over QUESTIONS ====
  # BOUND 256: a larger global FIFO, so more distinct problems are live at draw time.
  #
  # GLOBAL_SAMPLING=question. The default (entry) permutes ENTRIES, which makes a
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
  # ==== replay seed: none. The buffer is rebuilt by replaying the parent delta log to the
  # branch cursor (no SP_REPLAY_SEED_DIR: a cold seed dir would be both unused and misleading).
  # Cold bootstrap OFF: an empty buffer at this point must raise "sp_replay global buffer is
  # empty" rather than be back-filled with fresh problems.
  "SP_REPLAY_COLD_BOOTSTRAP=0"
  # A branch IS a graft: runner.py carries its own from-scratch guard (independent of the one
  # in this script) that refuses a checkpoint tree holding step N with an empty metrics.jsonl.
  # That is exactly what branch construction produces on the first attach, so the guard must be
  # told this is intentional rather than removed.
  "SP_ALLOW_GRAFT=1"
  # ==== gates ====
  "SP_Q_REQUIRE_NONZERO=1"
  # generative-Q: the FIFO and reference bank are rebuilt from the inherited delta log,
  # not from cold seed dirs (which is what the parent, a from-scratch arm, used).
  "SP_Q_ENABLE=1"
  # Train BOTH prompt variants. Records WITHOUT a resolvable reference are trained via the
  # no-reference variant at weight 1 instead of being skipped -- which is what the reference
  # bank produces for every unsolved problem. q/loss_noref becomes a real number; the Q phase
  # does up to 2x the rows.
  # THE AUDIT KNOBS (see the header):
  #   g = 10000: the parent's action-chunk length (each TD row is capped at its prefix + g).
  #   AUDIT_DEN 4 -> 0: sp_q_readiness treats den<=0 as "no audit lane at all", so no group
  #     is routed audit and audit_cut never fires. AUDIT_CUT=0 too, so the intent is explicit
  #     rather than relying on the empty-audit-set side effect.
  #   SEED 804001: the parent's.
  "SP_Q_BUDGET_G=10000" "SP_Q_AUDIT_CUT=0" "SP_Q_TRAIN_NOREF=1" "SP_Q_AUDIT_DEN=0"
  "SP_Q_RNG_SEED=804001"
  # ==== Q (critic) learning and readiness ====
  # ---- readiness: TWO thresholds. GLOBAL 0.20 gates only whether readiness may open at all
  # (pooled 5-step MAE) and is deliberately the looser one: from a cold start (cold FIFO, cold
  # bank, base policy) a tighter gate risks never opening. PER-PROBLEM 0.18 decides whether a
  # specific problem's rollouts get cut short and scored by Q instead of the judge, which is
  # where a wrong call costs real reward.
  # SP_Q_READY_THRESH stays as the legacy single knob / default for both.
  "SP_Q_READY_THRESH=0.18"
  "SP_Q_READY_THRESH_GLOBAL=0.2" "SP_Q_READY_THRESH_PROBLEM=0.18"
  # Readiness additionally requires the problem to be SOLVED (present in the add-once reference
  # bank, which only ever fills from a judged-PASSING row). Without it readiness keys purely on
  # Q's prediction ERROR, so a problem the prover never solves goes ready the moment Q
  # confidently predicts its z of 0 -- in the parent, 305 of 1,415 ready problems were exactly
  # that, and a step-40 critic probe found Q over-predicting their reward by +0.26, worse than
  # a constant. This arm has NO audit lane, so nothing here would catch that; the gate matters
  # more here than in the parent, not less.
  # q/ready_blocked_no_bank and q/ready_require_bank are emitted automatically -- metrics, not
  # knobs. If the former stays 0, the flag is doing nothing.
  "SP_Q_READY_REQUIRE_BANK=1"
  # FIFO = a 20-step horizon x 96 groups = 1920 records; 768 records/step; >=8 valid rewards
  # per target.
  "SP_Q_FIFO_CAP=1920" "SP_Q_MIN_VALID=8" "SP_Q_TRAIN_N=768"
  "SP_Q_GRAD_CLIP=0.2"
  # order: PPO minibatch 1 -> Q-only step (LEFT APPLIED) -> PPO minibatch 2
  "SP_Q_INTERLEAVE=1" "SP_Q_INTERLEAVE_AFTER=1"
  # size control: the persistent halving ladder. Do NOT also set SP_Q_LR_BASE/SP_Q_RHO_CAP (the
  # older per-step movement cap) here — on the interleaved path they are dead knobs, and
  # exporting them would read as if the cap were still doing something.
  # Q ladder bounds also scale by sqrt(2) with the doubled batch:
  # initial 2e-6 -> 2*sqrt(2)e-6, floor 5e-7 -> 5*sqrt(2)e-7.
  "SP_Q_LR_LADDER=1" "SP_Q_LR_INITIAL=2.8284271247461903e-6" "SP_Q_LR_FLOOR=7.071067811865476e-7"
  # Ratio threshold sqrt(2): the ladder halves Q's LR only when Q moves more than sqrt(2)x
  # the policy per interleaved step, letting Q track a faster target.
  "SP_Q_LR_RATIO_MAX=1.4142135623730951" "SP_Q_LR_BREACH_PATIENCE=2" "SP_Q_LR_REDUCTION_FACTOR=0.5"
  # Persist every completion sent to Q for scoring (ctx it saw + text it returned).
  "SP_Q_DUMP_WAVE=1"
  # ==== THE ARM: no-group TD on the READY lane (see header). Read by the harness and the
  # dataset through the Hydra config (+data.sp_q_td_*), so no Ray runtime_env forwarding is
  # needed; SP_ADV_ESTIMATOR is read by the runner on the driver. Verify with the driver log line
  # "[sp_q_td] rewrote K ... slot(s)" and q/td_targets_admitted > 0.
  "SP_Q_TD_ENABLE=1" "SP_Q_TD_LANE=short" "SP_Q_TD_CUT_GRAIN=1000" "SP_Q_TD_ADMIT_PER_SLOT=8"
  "SP_ADV_ESTIMATOR=sp_segment"
  # ==== ROLLOUT TENSOR PARALLELISM 4 (inference efficiency) ====
  # world_size / 4 replicas of 4 GPUs instead of one single-GPU replica per GPU. In a rollout
  # benchmark TP2 gave 967.6 vs TP1 603.6 tok/s/node, with TP1's loss traced to replica
  # IMBALANCE (11.4x spread -- most GPUs idle while one replica finished its tail). TP=4 also
  # shards attention, which the profile put at ~75% of rollout GPU time; measured 2,558s
  # generation vs a 4,100-6,000s TP=1 baseline (1.6-2.3x).
  #
  # Requires the flashinfer AOT preflight above -- at TP>1 the engine cannot start reliably
  # without it.
  "SP_ROLLOUT_TP=4"
  "SP_ROLLOUT_PRIORITY=0"
  # ==== CONTEXT: 50k response budget. Change these together or not at all -- the runner
  # asserts SP_Q_CTX_LIMIT (C_Q) == 2048 + response + 1248, so moving the response budget alone
  # crashes at import rather than mis-training, and SP_Q_MAX_TOKEN_LEN must clear C_Q (Q
  # prompts reach it). The runner's own defaults are the same 50k set; the attach is the single
  # place the effective value lives.
  #
  # SP_PPO_MAX_TOKEN_LEN=51200 is ~0.98x a full packed 52,048-token row (2048 + 50000). The
  # invariant that matters is ONE maximum-length row per GPU micro-batch, not that the budget
  # sits strictly below a full row (a 75k budget uses 78336, ~1.02x its 77,048-token row).
  "SP_MAX_RESPONSE_LEN=50000"
  "SP_Q_CTX_LIMIT=53296" "SP_Q_MAX_TOKEN_LEN=53360"
  "SP_PPO_MAX_TOKEN_LEN=51200"
  # KV arena 0.7: between 0.6 (used at a 50k budget) and 0.75 (used at 75k/100k budgets).
  # At 50k/TP=4 it holds ~28 full-length (53,296-token) sequences per engine, up from ~24 at
  # 0.6 -- so SP_ACTOR_MAX_NUM_SEQS=96 is still nowhere near binding, and the extra arena is
  # pure headroom for short sequences.
  #
  # This is the knob to LOWER if you see "sample_tokens RPC timed out" with NO OOM line: that
  # signature means the arena cannot coexist with the resident FSDP actor. Raising it is the
  # riskier direction, which is why 0.7 rather than 0.75 at 50k.
  "SP_ACTOR_GPU_MEM_UTIL=0.7"
  # ==== GROUPED CASCADE attention, ON (throughput only: it changes no sampling and no
  # gradient). The siblings of a group share one prefix, and the cascade kernel reads the shared
  # prefix once. The benefit is prefix-driven: the kernel pays a FIXED per-batch cost (three
  # kernels + the LSE merge + the non-identity dispatch, since many rows have no shareable
  # prefix) and only earns it back above some prefix coverage. Measured with the kernel as the
  # only change: attention calls replayed from this run 1.253x faster (70 of 71 batches), a
  # whole decoding step 1.058x at coverage 0.375 rising to 1.239x at coverage 1.00.
  #
  # PIECEWISE is mandatory, not a tuning choice: under FULL cudagraph the captured graph holds
  # stock attention and the python patch never runs, so the flag would be on and the kernel
  # would not be. AFFINITY+PILOT are the precondition -- affinity puts a group's siblings on one
  # replica, the pilot makes them share ONE physical prefix copy at ref_cnt=n, and without that
  # refcount there is no shared block for the kernel to exploit. PIECEWISE itself costs ~0.1%.
  "SP_GROUPED_CASCADE=1" "SP_ROLLOUT_CUDAGRAPH_MODE=PIECEWISE" "SP_GC_OVERLAP=0"
  "SP_PREFIX_AFFINITY=1" "SP_PREFIX_PILOT=1"
  # Both routing experiments are OFF and should stay off: GROUP_SLOTS was measured to LOWER
  # bulk coverage (0.455 -> 0.280) by throttling grouped rows while prefix-less rows took the
  # seats; UNGROUPED_LANES is implemented and unit-tested but has never run, and an untested
  # router does not belong in a 500-step science run.
  "SP_PREFIX_GROUP_SLOTS=0" "SP_PREFIX_UNGROUPED_LANES=0"
  # REQUIRED WITH CASCADE, not optional. At the default 256 the cascade engine's higher resident
  # footprint leaves too little memory for the colocated judge's DeepGEMM fp8 GEMM at wake:
  # CUDA driver error 2 on one engine, whose EngineCore then turns every request routed to it
  # into EngineDeadError. 128 avoids it. Judge concurrency costs judge wall clock only; never
  # fix this with eager or a lower judge gpu_memory_utilization.
  "SP_REWARD_MAX_NUM_SEQS=128"
  # SP_ATTN_TIME / SP_GC_TIME are deliberately NOT set: they synchronize a CUDA event every
  # 64th layer call, which is fine for a short measurement and pure overhead for a full run.
  # ==== NO length penalty -- pinned to 0, not merely omitted.
  # ds4_finegrained_judge reads SP_LENPEN_ENABLE straight from the process environment, the
  # holder does not use --export=NONE, and the runner only forwards the variable when it is
  # non-empty. So submitting from a shell that still exports SP_LENPEN_ENABLE=1 would silently
  # reinstate a length penalty with nothing in the log saying so. Setting it explicitly makes
  # the intended value the one that reaches the judge, and the runner asserts it. Same
  # reasoning for the other science-changing switches below.
  "SP_LENPEN_ENABLE=0"
  "SP_DIFF_SAMPLING=0" "SP_Q_SEPARATE=0"
  # ==== Q prompt whose stated quantity and horizon match the label (recorded in
  # q_state.json; a resume under a different variant is refused).
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
#   (b) ray_trainer saves the checkpoint BEFORE validating, so a holder roll between the save
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
# The grafted branch-step checkpoint carries the PARENT's replay policy, which records
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

# Caller-supplied KEY=VALUE args win for everything EXCEPT SP_VAL_BEFORE_TRAIN: a relaunch
# wrapper may hard-code it to False, so drop it and let the gate above decide.
for kv in "$@"; do
  case "$kv" in SP_VAL_BEFORE_TRAIN=*) continue ;; esac
  ENVS+=("$kv")
done
ENVS+=("SP_VAL_BEFORE_TRAIN=$_VBT")

# ---- SP_DRYRUN: import the runner on the LOGIN NODE with the EXACT env the driver gets ----
# runner.py's import-time guards (e.g. a missing replay_seed_cold, the runner-side graft guard)
# would otherwise each cost a full attach cycle to discover; they are reachable on CPU in ~30s.
# This lives inside the attach rather than in a sibling script on purpose: it consumes the same
# ENVS array, so it cannot drift from what actually launches.
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
DIRECT_LOG="$EXP/driver_direct_attempt${A}.log"
nohup srun --jobid="$JOBID" --job-name=sp-driver --overlap --nodes=1 --ntasks=1 --mem=0 -w "$HEAD_NODE" \
    bash -c "unset ROCR_VISIBLE_DEVICES; source $EXP/setup.sh cuda-compat; source $VENV/bin/activate && source $REPO_ROOT/experiments/09_05_qtd_ready_s192b50_noaudit/setup.sh cuda-toolkit && cd $REPO_ROOT && \
        env ${ENVS[*]} python $RUNNER 2>&1 | tee $DIRECT_LOG" > "$LOG" 2>&1 &
printf 'JOBID=%s\nHOST=%s\nWHEN=%s\nLOG=%s\n' "$JOBID" "$HEAD_NODE" "$(date -Is)" "$LOG" > "$EXP/run_data/.attach.lock"
echo "[attach] driver login-node pid $! ; tail -f $LOG"
