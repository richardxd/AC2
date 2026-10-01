#!/usr/bin/env bash
# submit_cluster_b_4node.sbatch -- 4-node allocation for 08_26_s192b40_g10k_noaudit on cluster B: starts a Ray
# head and workers and writes sandbox_<JOBID>.info for run_attach_cluster_b.sh.
# Submit with scripts/sbatch_env.sh, which fills in the ${AC2_*} site settings.
#SBATCH --job-name=sp-0826b40-hold4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=1-00:00:00
#SBATCH --account=${AC2_CLUSTER_B_ACCOUNT}
#SBATCH --partition=${AC2_CLUSTER_B_PARTITION}
#SBATCH --qos=${AC2_CLUSTER_B_QOS}
#SBATCH --output=${AC2_CLUSTER_B_HOME}/sp-0826b40-hold4-%j.out
#SBATCH --error=${AC2_CLUSTER_B_HOME}/sp-0826b40-hold4-%j.err
#SBATCH --mail-user=${AC2_MAIL_USER}

set -xuo pipefail    # deliberately NO -e

REPO_ROOT="${SELF_PLAY_ROOT:-${AC2_CLUSTER_B_ROOT}/self-play}"
EXP="$REPO_ROOT/experiments/08_26_s192b40_g10k_noaudit"
VENV="$REPO_ROOT/.venv"
SETUP_SH="$REPO_ROOT/experiments/08_26_s192b40_g10k_noaudit/setup.sh"
export SLURM_CPU_BIND=none

mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 --mem=0 -w "$head_node" hostname --ip-address)
if [[ "$head_node_ip" == *" "* ]]; then read -ra ADDR <<<"$head_node_ip"; head_node_ip="${ADDR[1]:-${ADDR[0]}}"; fi
port=6379
ip_head="$head_node_ip:$port"

INFO="$EXP/sandbox_${SLURM_JOB_ID}.info"
{
  echo "JOBID=$SLURM_JOB_ID"
  echo "HEAD_NODE=$head_node"
  echo "RAY_ADDRESS=$ip_head"
  echo "NODES=${nodes_array[*]}"
  echo "STARTED=$(date -Is)"
} > "$INFO" 2>/dev/null || echo "[hold4] WARN: could not write $INFO (quota?)"
cp -f "$INFO" "${AC2_CLUSTER_B_HOME}/sp-0826b40-hold4-${SLURM_JOB_ID}.info" 2>/dev/null || true

# ---- Ray head + workers ----
ACTIVATE="unset ROCR_VISIBLE_DEVICES; source $VENV/bin/activate; source $SETUP_SH cuda-toolkit"
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "$ACTIVATE; ray start --head --node-ip-address=$head_node_ip --port=$port \
           --num-cpus 96 --num-gpus 8 --block" &
sleep 20
for ((i = 1; i < ${#nodes_array[@]}; i++)); do
  srun --overlap --nodes=1 --ntasks=1 -w "${nodes_array[$i]}" \
    bash -c "$ACTIVATE; ray start --address=$ip_head --num-cpus 96 --num-gpus 8 --block" &
  sleep 5
done
sleep 20

echo "[hold4] READY jobid=$SLURM_JOB_ID ray=$ip_head nodes=${nodes_array[*]}"

# ---- SELF-ATTACH ------------------------------------------------------------------------
# The attach runs from INSIDE the allocation rather than from a login-node watcher: a separate
# watcher process can die and leave an allocation idle, whereas this cannot desynchronise from
# the holder because it IS the holder.
if [ "${SP_NO_AUTOATTACH:-0}" = "1" ]; then
  echo "[hold4] SP_NO_AUTOATTACH=1 -- holding without attaching"
  sleep infinity
fi

echo "[hold4] auto-attaching: run_attach_cluster_b.sh $SLURM_JOB_ID"
bash "$EXP/run_attach_cluster_b.sh" "$SLURM_JOB_ID"
rc=$?
echo "[hold4] attach exited rc=$rc at $(date -Is)"

# rc is NOT a health signal: run_attach_cluster_b.sh launches the driver with nohup+srun and
# returns immediately, so it reports 0 whenever the LAUNCHER ran, even if the driver dies
# seconds later. Supervise the driver as a job STEP instead: it registers as its own step under
# this job, so its disappearance is an unambiguous death, with none of rc's ambiguity. It must
# be matched by its OWN name: the four `ray start` sruns inherit the job name and outlive any
# driver, so an exclusion filter would report a dead driver as alive for the whole allocation.
if [ "$rc" -eq 0 ]; then
  sleep 150                       # let the driver step register before the first look
  while :; do
    live=$(squeue -s -j "$SLURM_JOB_ID" -h -o '%j' 2>/dev/null | grep -c '^sp-driver$' || true)
    [ "${live:-0}" -gt 0 ] || { echo "[hold4] driver step GONE at $(date -Is) -- treating as failed attach"; rc=1; break; }
    sleep 60
  done
fi

# NEVER exit on a failed attach: the allocation took hours to schedule and is far more
# expensive than the failure. Hold it, leaving it available for a manual retry.
if [ "$rc" -ne 0 ]; then
  echo "[hold4] attach FAILED rc=$rc -- holding the allocation for retry"
fi
sleep infinity
