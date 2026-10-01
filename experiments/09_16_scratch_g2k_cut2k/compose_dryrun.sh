#!/usr/bin/env bash
# compose_dryrun.sh — compose the verl config with the launch environment on the login node
# (CPU, seconds) before requesting GPUs. Mirrors the training env of run_attach_cluster_a.sh;
# keep the two in sync when knobs change.
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
REPO_ROOT=${AC2_CLUSTER_A_ROOT}/self-play
EXP="$REPO_ROOT/experiments/09_16_scratch_g2k_cut2k"
cd "$REPO_ROOT"
env \
  SP_REPLAY_ENABLE=1 SP_Q_ENABLE=1 SP_Q_INTERLEAVE=1 SP_Q_LR_LADDER=1 \
  SP_REPLAY_COLD_BOOTSTRAP=1 SP_REPLAY_SEED_DIR="$EXP/replay_seed_cold" \
  SP_TRAIN_BATCH_SIZE=384 SP_REPLAY_N=192 SP_PPO_MINI_BATCH=96 \
  SP_LR=2e-6 SP_Q_LR_INITIAL=2.8284271247461903e-6 SP_Q_LR_FLOOR=7.071067811865476e-7 \
  SP_Q_LR_RATIO_MAX=1.4142135623730951 \
  SP_ROLLOUT_TP=4 SP_MAX_RESPONSE_LEN=50000 SP_Q_CTX_LIMIT=53296 \
  SP_Q_MAX_TOKEN_LEN=53360 SP_PPO_MAX_TOKEN_LEN=51200 SP_ACTOR_GPU_MEM_UTIL=0.7 \
  SP_REPLAY_BOUND=256 SP_REPLAY_GLOBAL_SAMPLING=question \
  SP_LENPEN_ENABLE=0 SP_DIFF_SAMPLING=0 SP_Q_SEPARATE=0 SP_REPLAY_POLICY_OVERRIDE=0 \
  SP_Q_PROMPT_VARIANT=reward_horizon SP_Q_REF_REQUIRE_PASS=1 SP_Q_TRAIN_NOREF=1 \
  SP_Q_READY_REQUIRE_BANK=1 \
  SP_REPLAY_BUCKETING=global SP_REPLAY_ROTATION=global_fifo \
  SP_REPLAY_ADMISSION=ungated SP_Q_RNG_SEED=916001 \
  SP_Q_AUDIT_CUT=1 SP_Q_AUDIT_DEN=4 SP_Q_BUDGET_G=2000 SP_REPLAY_CUT_GRAIN=2000 \
  SP_TOTAL_STEPS=500 SP_SAVE_FREQ=1 SP_TEST_FREQ=10 \
  NNODES="${NNODES:-7}" SP_DP_PAD=1 \
  "$REPO_ROOT/.venv/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, os.path.expandvars("${AC2_CLUSTER_A_ROOT}/self-play/experiments/09_16_scratch_g2k_cut2k"))
import runner
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir=runner.VERL_CONFIG_DIR, version_base=None):
    cfg = compose(config_name="ppo_trainer", overrides=runner.OVERRIDES)
print("COMPOSE_OK",
      "train_bs=", cfg.data.train_batch_size,
      "mini=", cfg.actor_rollout_ref.actor.ppo_mini_batch_size,
      "lr=", cfg.actor_rollout_ref.actor.optim.lr,
      "tp=", cfg.actor_rollout_ref.rollout.tensor_model_parallel_size,
      "nnodes=", cfg.trainer.nnodes)
# Topology vs batch shape (7-node support). The runner already printed the
# "[sp_dp_pad] dp=... ACTIVE|inert" line at import and asserted SP_DP_PAD is set whenever the
# world size does not divide the shape; re-state it off the composed config so the dry-run
# record says which regime THIS launch is in.
from verl.trainer.ppo import sp_dp_pad as _sp_dp
_ws = int(cfg.trainer.nnodes) * int(cfg.trainer.n_gpus_per_node)
_n = int(cfg.actor_rollout_ref.rollout.n)
print("TOPOLOGY", _sp_dp.describe((384 - 192) + 192 * _n, 192 * _n,
                                   int(cfg.actor_rollout_ref.actor.ppo_mini_batch_size) * _n, _ws))
assert int(cfg.trainer.nnodes) == int(os.environ["NNODES"]), cfg.trainer.nnodes
assert cfg.actor_rollout_ref.actor.use_dynamic_bsz, "verl's startup batch-divisibility check is skipped only under dynamic bsz"
assert cfg.actor_rollout_ref.actor.loss_agg_mode == "token-mean", cfg.actor_rollout_ref.actor.loss_agg_mode
print("TOPOLOGY_OK")
# The replay and critic settings, read back off the COMPOSED config rather than the env --
# a knob that never made it into cfg.data is exactly the silent no-op this check exists for.
print("DELTAS",
      "admission=", cfg.data.sp_replay_admission,
      "bucketing=", cfg.data.sp_replay_bucketing,
      "rotation=", cfg.data.sp_replay_rotation,
      "bound=", cfg.data.sp_replay_bound,
      "q_budget_g=", cfg.data.sp_q_budget_g,
      "q_audit_den=", cfg.data.sp_q_audit_den,
      "q_audit_cut=", cfg.data.sp_q_audit_cut,
      "q_rng_seed=", cfg.data.sp_q_rng_seed)
# This run's two coupled changes relative to the main run (08_13_tiedq_seed192): b 10000 ->
# 2000 and cut grid 10000 -> 2000. Asserted off the COMPOSED config so a knob that never made
# it into cfg.data is caught here, not 25 minutes into an allocation.
assert int(cfg.data.sp_q_budget_g) == 2000, cfg.data.sp_q_budget_g
assert int(cfg.data.sp_replay_cut_grain) == 2000, cfg.data.sp_replay_cut_grain
# Main-run values, asserted so a stray export from another configuration cannot creep in:
# auditing ON at DEN=4/CUT=1 (unlike the noaudit variants), ungated admission, no TD, grpo
# estimator; the critic-side seed is this run's own (916001).
assert int(cfg.data.sp_q_audit_den) == 4, cfg.data.sp_q_audit_den
assert int(cfg.data.sp_q_audit_cut) == 1, cfg.data.sp_q_audit_cut
assert cfg.data.sp_replay_admission == "ungated", cfg.data.sp_replay_admission
assert int(getattr(cfg.data, "sp_q_td_enable", 0)) == 0
assert cfg.algorithm.adv_estimator == "grpo", cfg.algorithm.adv_estimator
assert int(cfg.data.sp_q_rng_seed) == 916001, cfg.data.sp_q_rng_seed
print("DELTAS_OK")

# sp_replay's own validator is the thing that rejects an illegal policy combination
# (this run uses ungated + global + global_fifo).
from verl.trainer.ppo.sp_replay import ReplayHarness  # noqa: E402
print("POLICY_VALIDATOR_IMPORTED")
PY
