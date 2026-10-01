#!/usr/bin/env bash
# refresh_dashboard.sh -- dashboard for 08_26_s192b40_g10k_noaudit: renders the parent's history
# and this branch in one dashboard (08_13 cache 0-40 + this branch's 41+ continuation) with
# build_lineage_dashboard.py.
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
REPO=${AC2_CLUSTER_B_ROOT}/self-play
export SELF_PLAY_ROOT=$REPO
exec "$REPO/.venv/bin/python" "$REPO/experiments/08_26_s192b40_g10k_noaudit/build_lineage_dashboard.py"
