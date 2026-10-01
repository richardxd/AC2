#!/usr/bin/env bash
# Stage C (judge) + analysis for the step-40 probe, as a script: a script invoked by name has
# no shell quoting to get wrong, unlike a long inline command with nested quotes.
set -uo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }

REPO="${SELF_PLAY_ROOT:-${AC2_CLUSTER_A_ROOT}/self-play}"
E="$REPO/experiments/08_15_q_probe_step40"
JID="${SP_PROBE_JOBID:?set SP_PROBE_JOBID to the allocation job id}"

# a previous attempt's server can still hold the GPUs; clear before re-serving
srun --overlap --jobid="$JID" --nodes=4 --ntasks=4 --ntasks-per-node=1 \
     bash -c 'pkill -f "[v]llm serve" 2>/dev/null; ray stop --force >/dev/null 2>&1; exit 0' \
     >/dev/null 2>&1
sleep 5

rm -f "$E/judged"/*.log "$E/judged"/*.jsonl 2>/dev/null
echo "[stage-c] launching $(date -Is)"
srun --overlap --jobid="$JID" --nodes=4 --ntasks=4 --ntasks-per-node=1 \
     bash "$E/probe_judge_node.sh" > "$E/stage_c.log" 2>&1
echo "[stage-c] srun returned $(date -Is)"

source "$REPO/.venv/bin/activate"
python "$E/analyze_probe.py" > "$E/RESULT.txt" 2>&1
echo "[stage-c] RESULT written $(date -Is)"
echo CHAIN_C_DONE
