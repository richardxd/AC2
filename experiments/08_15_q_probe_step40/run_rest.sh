#!/usr/bin/env bash
# Chain stages B (Q) and C (judge) behind stage A, then analyse. Run detached on the login
# node so no GPU sits idle waiting for a human to notice a stage finished.
#
# Each stage waits on ARTIFACTS, not on job ids: a shard file only appears when its replica
# wrote it, so "32 files exist" is a real completion test that survives a replica being
# retried or a launch command being duplicated.
set -uo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }

REPO="${SELF_PLAY_ROOT:-${AC2_CLUSTER_A_ROOT}/self-play}"
E="$REPO/experiments/08_15_q_probe_step40"
JID="${SP_PROBE_JOBID:?set SP_PROBE_JOBID to the allocation job id}"
NGEN="${SP_PROBE_NSHARDS:-32}"
NJUDGE="${SP_JUDGE_NSHARDS:-4}"

wait_for() {   # wait_for <glob> <count> <label> <max_sec>
  local pat="$1" want="$2" label="$3" max="${4:-14400}" t=0
  while [ "$(ls $pat 2>/dev/null | wc -l)" -lt "$want" ]; do
    sleep 30; t=$((t+30))
    if [ "$t" -ge "$max" ]; then
      echo "[chain] TIMEOUT waiting for $label ($(ls $pat 2>/dev/null | wc -l)/$want)"; return 1
    fi
  done
  echo "[chain] $label complete ($want) after ${t}s"; return 0
}

echo "[chain] start $(date -Is)"
wait_for "$E/gen/gen.shard*.jsonl" "$NGEN" "stage A generation" 10800 || exit 1

echo "[chain] launching stage B (Q at prefix+10k) $(date -Is)"
srun --overlap --jobid="$JID" --nodes=4 --ntasks=4 --ntasks-per-node=1 \
     bash "$E/probe_q_node.sh" > "$E/stage_b.log" 2>&1
echo "[chain] stage B srun returned $(date -Is)"
wait_for "$E/q/q.shard*.jsonl" "$NGEN" "stage B Q" 5400 || echo "[chain] proceeding with partial Q"

# Stage C stops Ray on every node (TP=8 capture needs the GPUs), so it must run AFTER stage B,
# which uses the same GPUs for TP=1 engines.
echo "[chain] launching stage C (DS4 judge) $(date -Is)"
srun --overlap --jobid="$JID" --nodes=4 --ntasks=4 --ntasks-per-node=1 \
     bash "$E/probe_judge_node.sh" > "$E/stage_c.log" 2>&1
echo "[chain] stage C srun returned $(date -Is)"

echo "[chain] analysing $(date -Is)"
source "$REPO/.venv/bin/activate"
python "$E/analyze_probe.py" > "$E/RESULT.txt" 2>&1
echo "[chain] RESULT written $(date -Is)"
tail -60 "$E/RESULT.txt"
echo "[chain] done $(date -Is)"
