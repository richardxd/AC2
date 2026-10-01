#!/usr/bin/env bash
# refresh_dashboard.sh — render the combined dashboard for 08_19_branch130_ctx75k (the parent
# 08_13_tiedq_seed192 through step 130 plus this run's 75k continuation from step 131) via
# build_lineage_dashboard.py.
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
REPO=${AC2_CLUSTER_A_ROOT}/self-play
export SELF_PLAY_ROOT=$REPO
exec "$REPO/.venv/bin/python" "$REPO/experiments/08_19_branch130_ctx75k/build_lineage_dashboard.py"
