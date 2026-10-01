#!/usr/bin/env bash
# launch_cluster_b.sh -- per-node launch step for 09_09_globalready_qtd_s192b20_noaudit, run by
# submit_cluster_b_4node.sbatch as ONE srun step with one task per node (8 GPUs each).
# Task 0 (first node) starts the Ray head, writes sandbox_<JOBID>.info for
# run_attach_cluster_b.sh and runs the driver INLINE; the other tasks join Ray as workers and
# wait for task 0's done marker (or the optional deadline). Every task exits 0 so
# --kill-on-bad-exit never tears the siblings down mid-checkpoint; the driver's rc is in the marker.
set -uo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
REPO_ROOT=${AC2_CLUSTER_B_ROOT}/self-play
EXP="$REPO_ROOT/experiments/09_09_globalready_qtd_s192b20_noaudit"
VENV="$REPO_ROOT/.venv"
SETUP_SH="$REPO_ROOT/experiments/09_09_globalready_qtd_s192b20_noaudit/setup.sh"
JOBID="${SLURM_JOB_ID:?launch_cluster_b.sh must run inside the allocation}"
RANK="${SLURM_PROCID:-0}"
RUN_ROOT="${SP_LAUNCH_RUN_ROOT:-$EXP/launch_run_root}"; mkdir -p "$RUN_ROOT/logs"
DEADLINE="${SP_LAUNCH_DEADLINE_EPOCH:-0}"
INFO="$EXP/sandbox_${JOBID}.info"; READY="$EXP/sandbox_${JOBID}.ready"; MARK="$RUN_ROOT/lane_${JOBID}.done"
NODES="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ' | sed 's/ $//')"
HEAD_NODE="${NODES%% *}"
ME="$(hostname -s)"
LOG="$RUN_ROOT/logs/lane_${JOBID}_rank${RANK}_${ME}.log"
exec > >(tee -a "$LOG") 2>&1
echo "[lane] job=$JOBID rank=$RANK node=$ME head=$HEAD_NODE nodes='$NODES' deadline=$DEADLINE $(date -Is)"
unset ROCR_VISIBLE_DEVICES
source "$VENV/bin/activate"; source "$SETUP_SH" cuda-toolkit 2>/dev/null || true
ray stop --force >/dev/null 2>&1 || true
if [ "$RANK" = "0" ]; then
  HEAD_IP="$(hostname --ip-address | awk '{print $NF}')"
  rm -f "$MARK" "$READY"
  ray start --head --node-ip-address="$HEAD_IP" --port=6379 --num-cpus 96 --num-gpus 8 --block \
      > "$RUN_ROOT/logs/ray_head_${JOBID}.log" 2>&1 &
  sleep 15
  printf "JOBID=%s\nHEAD_NODE=%s\nHEAD_IP=%s\nRAY_ADDRESS=%s:6379\nNODES='%s'\n" \
      "$JOBID" "$HEAD_NODE" "$HEAD_IP" "$HEAD_IP" "$NODES" > "$INFO"
  echo "[lane] wrote $INFO"
  want=$(echo "$NODES" | wc -w)
  for i in $(seq 1 60); do
    have=$(python -c 'import ray; ray.init(address="auto", ignore_reinit_error=True, logging_level=40); print(sum(1 for n in ray.nodes() if n.get("Alive")))' 2>/dev/null)
    [ "${have:-0}" -ge "$want" ] && break
    sleep 5
  done
  echo "[lane] ray nodes alive: ${have:-0}/$want"
  if [ "${have:-0}" -lt "$want" ]; then
    echo "[lane] REFUSING: Ray did not assemble $want nodes"; echo "rc=97" > "$MARK"; ray stop --force; exit 0
  fi
  touch "$READY"
  SP_POOL_MODE=1 SP_DRIVER_INLINE=1 bash "$EXP/run_attach_cluster_b.sh" "$JOBID" ${SP_LANE_ARGS:-}
  rc=$?
  echo "[lane] driver rc=$rc $(date -Is)"; echo "rc=$rc" > "$MARK"
  ray stop --force >/dev/null 2>&1 || true
  exit 0
else
  for i in $(seq 1 60); do [ -f "$INFO" ] && break; sleep 5; done
  HEAD_IP="$(sed -n 's/^HEAD_IP=//p' "$INFO" 2>/dev/null | head -1)"
  if [ -z "$HEAD_IP" ]; then echo "[lane] no head info after 300s; idling to the marker"; else
    ray start --address="$HEAD_IP:6379" --num-cpus 96 --num-gpus 8 --block \
        > "$RUN_ROOT/logs/ray_worker_${JOBID}_${ME}.log" 2>&1 &
  fi
  while [ ! -f "$MARK" ]; do
    if [ "$DEADLINE" -gt 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "[lane] deadline reached"; break; fi
    sleep 30
  done
  ray stop --force >/dev/null 2>&1 || true
  exit 0
fi
