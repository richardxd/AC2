#!/usr/bin/env bash
# compose_dryrun.sh — compose the verl config with the real launch environment on the login
# node (CPU, seconds) before requesting GPUs. Mirrors the training env of
# run_attach_cluster_a.sh; keep the two in sync when knobs change.
#
# NOTE: SP_Q_SEED_DIR/SP_Q_BANK_DIR must be passed EXPLICITLY and must point at the PARENT:
# runner.py defaults them to its own experiment dir, which is correct for a from-scratch run
# but not for a branch (the dry run would stop on "q_seed_manifest.json missing").
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
REPO_ROOT=${AC2_CLUSTER_A_ROOT}/self-play
EXP="$REPO_ROOT/experiments/08_19_branch130_ctx75k"
EPAR="$REPO_ROOT/experiments/08_13_tiedq_seed192"   # parent: sha-pinned replay seed lives here
cd "$REPO_ROOT"
env \
  SP_REPLAY_ENABLE=1 SP_Q_ENABLE=1 SP_Q_INTERLEAVE=1 SP_Q_LR_LADDER=1 \
  SP_Q_SEED_DIR="$EPAR" SP_Q_BANK_DIR="$EPAR" \
  SP_REPLAY_COLD_BOOTSTRAP=0 SP_REPLAY_SEED_DIR="$EPAR/replay_seed_cold" \
  SP_TRAIN_BATCH_SIZE=384 SP_REPLAY_N=192 SP_PPO_MINI_BATCH=96 \
  SP_LR=2e-6 SP_Q_LR_INITIAL=2.8284271247461903e-6 SP_Q_LR_FLOOR=7.071067811865476e-7 \
  SP_Q_LR_RATIO_MAX=1.4142135623730951 \
  SP_ROLLOUT_TP=4 SP_MAX_RESPONSE_LEN=75000 SP_Q_CTX_LIMIT=78296 \
  SP_Q_MAX_TOKEN_LEN=78360 SP_PPO_MAX_TOKEN_LEN=78336 SP_ACTOR_GPU_MEM_UTIL=0.7 \
  SP_REPLAY_BOUND=256 SP_REPLAY_GLOBAL_SAMPLING=question \
  SP_LENPEN_ENABLE=0 SP_DIFF_SAMPLING=0 SP_Q_SEPARATE=0 SP_REPLAY_POLICY_OVERRIDE=0 \
  SP_Q_PROMPT_VARIANT=reward_horizon SP_Q_REF_REQUIRE_PASS=1 SP_Q_TRAIN_NOREF=1 \
  SP_Q_READY_REQUIRE_BANK=1 \
  SP_Q_AUDIT_CUT=1 SP_Q_AUDIT_DEN=4 SP_Q_BUDGET_G=10000 \
  SP_TOTAL_STEPS=500 SP_SAVE_FREQ=1 SP_TEST_FREQ=10 \
  "$REPO_ROOT/.venv/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, os.path.expandvars("${AC2_CLUSTER_A_ROOT}/self-play/experiments/08_19_branch130_ctx75k"))
import runner
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir=runner.VERL_CONFIG_DIR, version_base=None):
    cfg = compose(config_name="ppo_trainer", overrides=runner.OVERRIDES)
print("COMPOSE_OK",
      "train_bs=", cfg.data.train_batch_size,
      "mini=", cfg.actor_rollout_ref.actor.ppo_mini_batch_size,
      "lr=", cfg.actor_rollout_ref.actor.optim.lr,
      "tp=", cfg.actor_rollout_ref.rollout.tensor_model_parallel_size)
# RESPONSE-BUDGET GROUP, printed because it is this run's only change and its main risk. The
# runner already asserts Q_CTX_LIMIT == 2048 + response + 1248, so a silent 50k would fail
# there -- but ppo_max_token_len is NOT covered by that assert and decides whether the first
# trained step (131) OOMs, so read it here rather than inferring it.
print("CONTEXT",
      "max_response=", cfg.data.max_response_length,
      "ppo_max_token_len=", cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
      "q_ctx_limit=", runner.Q_CTX_LIMIT,
      "q_max_token_len=", runner.Q_MAX_TOKEN_LEN,
      "arena=", cfg.actor_rollout_ref.rollout.gpu_memory_utilization)
PY
