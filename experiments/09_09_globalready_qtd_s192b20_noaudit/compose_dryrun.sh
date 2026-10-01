#!/usr/bin/env bash
# compose_dryrun.sh — compose the verl config with the REAL launch env on the login node (CPU,
# seconds) before requesting GPUs.
#
# This does NOT hand-copy the training knobs into a second list. It EXTRACTS the
# `ENVS=( ... )` array literal straight out of run_attach_cluster_b.sh and evaluates it,
# so the dry-run cannot drift from what actually launches — the failure mode a duplicated list
# invites. Only the infra variables the array interpolates (Ray address, cache roots, node
# count) are stubbed here; none of them reach the Hydra overrides.
#
#   bash experiments/09_09_globalready_qtd_s192b20_noaudit/compose_dryrun.sh
#
# Expect a COMPOSE_OK line and an ARM_OK line. Anything else — ConfigAttributeError,
# ConfigCompositionException, an assertion from runner.py's own guards — is a launch that would
# have crashed ~25 minutes into an allocation instead.
#
# RELATIONSHIP TO `SP_DRYRUN=1 run_attach_cluster_b.sh <JID>`: that one already exists and already
# consumes this same ENVS array, but it stops at IMPORTING runner.py — it proves the import-time
# guards pass (cold-artifact manifests present, graft allowed, tied-Q asserted) and counts the
# overrides. It never composes them. This script goes the extra step and runs Hydra, which is
# what catches an override key that is not in-struct. Run BOTH; they fail on different things,
# and the import guards are the cheaper of the two to trip (e.g. a missing replay_seed_cold
# manifest).
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
REPO_ROOT="${SELF_PLAY_ROOT:-${AC2_CLUSTER_B_ROOT}/self-play}"
EXP="$REPO_ROOT/experiments/09_09_globalready_qtd_s192b20_noaudit"
ATTACH="$EXP/run_attach_cluster_b.sh"
VENV="$REPO_ROOT/.venv"
cd "$REPO_ROOT"

# ---- stubs for the infra vars the ENVS array interpolates -----------------------------------
# These are the ONLY values this script invents. Each is either a path the compose never reads
# or a scalar the runner passes straight through; if you find yourself adding a SCIENCE knob
# here, the extraction below has broken and the dry-run is no longer testing the real launch.
CACHE=${AC2_CLUSTER_B_ROOT}/.cache/ds4_vllm_023
HF_HOME_DIR=${AC2_CLUSTER_B_SCRATCH}/.cache/huggingface
HF_HUB_CACHE_DIR="$HF_HOME_DIR/hub"
RAY_ADDRESS="127.0.0.1:6379"
NNODES_H=4                       # the holder's real node count; ws32 = the parent's topology
XDG_CACHE_HOME="$CACHE/xdg"
TORCH_EXTENSIONS_DIR="$CACHE/torch_ext"
FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer"
# The judge pin is science, so take it from the attach script's own defaults rather than
# retyping it: eval the two assignment lines verbatim (they are plain ${VAR:-default} forms).
eval "$(grep -E '^JUDGE_(HF_REPO|REVISION)=' "$ATTACH")"
export CACHE HF_HOME_DIR HF_HUB_CACHE_DIR RAY_ADDRESS NNODES_H XDG_CACHE_HOME \
       TORCH_EXTENSIONS_DIR FLASHINFER_WORKSPACE_BASE JUDGE_HF_REPO JUDGE_REVISION EXP REPO_ROOT

# ---- extract the ENVS array literal from the attach script ----------------------------------
# awk, not sed -n 'A,Bp': the line numbers move every time the header comment is edited, and a
# stale range would silently dry-run a TRUNCATED env — which composes fine and proves nothing.
_ENVS_SRC="$(awk '/^ENVS=\(/{f=1} f{print} f&&/^\)/{exit}' "$ATTACH")"
[ -n "$_ENVS_SRC" ] || { echo "[dryrun] could not extract ENVS=( ) from $ATTACH"; exit 1; }
eval "$_ENVS_SRC"
[ "${#ENVS[@]}" -gt 40 ] || { echo "[dryrun] ENVS has only ${#ENVS[@]} entries — extraction truncated"; exit 1; }

# The knobs that DEFINE this arm. Assert them here rather than trusting the eye: a dry-run that
# composes a different g than the launch does is worse than no dry-run.
_want() {
  printf '%s\n' "${ENVS[@]}" | grep -qx -- "$1" \
    || { echo "[dryrun] REFUSING: expected $1 in ENVS, not found"; exit 1; }
}
_want "SP_Q_BUDGET_G=10000"
_want "SP_Q_AUDIT_DEN=0"
_want "SP_Q_AUDIT_CUT=0"
_want "SP_Q_RNG_SEED=804001"
_want "SP_Q_TD_ENABLE=1"
_want "SP_Q_TD_LANE=short"
_want "SP_Q_TD_CUT_GRAIN=1000"
_want "SP_Q_TD_ADMIT_PER_SLOT=8"
_want "SP_ADV_ESTIMATOR=sp_segment"
_want "SP_Q_READY_MODE=global"
_want "SP_Q_READY_GLOBAL_LATCH=1"
_want "SP_Q_READY_THRESH_GLOBAL=0.2"
echo "[dryrun] ENVS: ${#ENVS[@]} entries, arm knobs verified (g=10000, audit off, seed 804001, no-group TD on the ready lane, READY MODE global/latched)"

env "${ENVS[@]}" "$VENV/bin/python" - "$EXP" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import runner
from hydra import compose, initialize_config_dir

with initialize_config_dir(config_dir=runner.VERL_CONFIG_DIR, version_base=None):
    cfg = compose(config_name="ppo_trainer", overrides=runner.OVERRIDES)

# Echo the values whose drift breaks a launch, so a green dry-run is evidence about THIS
# config and not just "hydra parsed something".
print(
    "COMPOSE_OK",
    "train_bs=", cfg.data.train_batch_size,
    "mini=", cfg.actor_rollout_ref.actor.ppo_mini_batch_size,
    "lr=", cfg.actor_rollout_ref.actor.optim.lr,
    "tp=", cfg.actor_rollout_ref.rollout.tensor_model_parallel_size,
    "resp_len=", cfg.data.max_response_length,
    "ppo_max_tok=", cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
    "gpu_mem_util=", cfg.actor_rollout_ref.rollout.gpu_memory_utilization,
)
print(
    "ARM_OK",
    "q_budget_g=", cfg.data.sp_q_budget_g,
    "q_audit_den=", cfg.data.sp_q_audit_den,
    "q_audit_cut=", cfg.data.sp_q_audit_cut,
    "q_rng_seed=", cfg.data.sp_q_rng_seed,
    "q_td_enable=", cfg.data.sp_q_td_enable, "q_td_lane=", cfg.data.sp_q_td_lane,
    "q_ready_mode=", cfg.data.sp_q_ready_mode,
    "q_ready_global_latch=", cfg.data.sp_q_ready_global_latch,
    "q_thresh_global=", cfg.data.sp_q_ready_thresh_global,
)
PY
