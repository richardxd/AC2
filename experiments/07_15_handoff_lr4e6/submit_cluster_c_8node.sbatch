#!/usr/bin/env bash
# submit_cluster_c_8node.sbatch -- 8-node allocation for 07_15_handoff_lr4e6: starts a Ray head and
# workers, then writes alloc_<JOBID>.info for run_attach_cluster_c.sh.
# Submit with scripts/sbatch_env.sh, which fills in the ${AC2_*} site settings.
#SBATCH --job-name=07_15_handoff_lr4e6_alloc
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1800G
#SBATCH --time=12:00:00
#SBATCH --output=${AC2_CLUSTER_C_ROOT}/self-play/experiments/07_15_handoff_lr4e6/alloc_%j.out



set -uo pipefail
REPO_ROOT="${SELF_PLAY_ROOT:-${AC2_CLUSTER_C_ROOT}/self-play}"
EXP="$REPO_ROOT/experiments/07_15_handoff_lr4e6"

# SLURM >= 23.11 propagates SLURM_CPUS_PER_TASK into srun and then rejects the job when
# it conflicts with --cpus-per-task. Pin it and drop the TRES variant.
export SLURM_CPUS_PER_TASK=32
unset SLURM_TRES_PER_TASK

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD=${NODES[0]}
HEAD_IP=$(getent hosts "$HEAD" | awk '{print $1}' | head -1)
PORT=$(( 6379 + (SLURM_JOB_ID % 1000) ))
RAY_ADDRESS="$HEAD_IP:$PORT"

source "$REPO_ROOT/.venv/bin/activate"
export HF_HOME=${AC2_CLUSTER_C_ROOT}/hf-cache
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

echo "[alloc] job $SLURM_JOB_ID  nodes=${NODES[*]}  head=$HEAD  ray=$RAY_ADDRESS"

srun --nodes=1 --ntasks=1 -w "$HEAD" \
  ray start --head --node-ip-address="$HEAD_IP" --port="$PORT" \
            --num-gpus=8 --block &
sleep 25
for n in "${NODES[@]:1}"; do
  srun --nodes=1 --ntasks=1 -w "$n" \
    ray start --address="$RAY_ADDRESS" --num-gpus=8 --block &
  sleep 5
done
# ACTUALLY verify the cluster formed. Fixed sleeps plus a background srun tell you
# nothing: this script does not use set -e, so a worker that failed to start would still
# fall through to publishing "ready", and the driver would then attach to a short cluster
# and die. Poll Ray's own view until every node's GPUs are registered.
WANT_GPUS=$(( 8 * 8 ))
READY=0
for i in $(seq 1 60); do
  GOT=$(RAY_ADDRESS="$RAY_ADDRESS" ray status 2>/dev/null          | sed -n 's@.*/\([0-9.]*\) GPU$@\1@p' | head -1 | cut -d. -f1)
  if [ -n "$GOT" ] && [ "$GOT" -ge "$WANT_GPUS" ] 2>/dev/null; then
    echo "[alloc] ray cluster ready: $GOT/$WANT_GPUS GPUs after $(( i * 5 ))s"; READY=1; break
  fi
  sleep 5
done
if [ "$READY" != "1" ]; then
  echo "[alloc] FATAL: ray reported ${GOT:-0}/$WANT_GPUS GPUs after 300s — not publishing"
  echo "[alloc] no alloc_*.info is written, so run_attach_cluster_c.sh will refuse to attach"
  sleep infinity
fi

# Publish the address only after Ray is actually up -- run_attach_cluster_c.sh reads this
# file, so writing it early would let the driver attach to a half-formed cluster.
cat > "$EXP/alloc_${SLURM_JOB_ID}.info" <<INFO
JOBID=$SLURM_JOB_ID
HEAD_NODE=$HEAD
RAY_ADDRESS=$RAY_ADDRESS
NODES=${NODES[*]}
INFO
echo "[alloc] ready; holding"
sleep infinity
