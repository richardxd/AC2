#!/usr/bin/env bash
# refresh_dashboard.sh -- dashboard for 09_05_qtd_ready_s192b50_noaudit: renders the parent's history
# and this branch in one dashboard (08_13 cache 0-50 + this branch's 51+ continuation) with
# build_lineage_dashboard.py.
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
REPO=${AC2_CLUSTER_B_ROOT}/self-play
export SELF_PLAY_ROOT=$REPO
exec "$REPO/.venv/bin/python" "$REPO/experiments/09_05_qtd_ready_s192b50_noaudit/build_lineage_dashboard.py"
