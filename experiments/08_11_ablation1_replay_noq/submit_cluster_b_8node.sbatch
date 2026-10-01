#!/usr/bin/env bash
# submit_cluster_b_8node.sbatch -- 8-node allocation for 08_11_ablation1_replay_noq on cluster B: starts a Ray
# head and workers and writes sandbox_<JOBID>.info for run_attach_cluster_b.sh.
# Submit with scripts/sbatch_env.sh, which fills in the ${AC2_*} site settings.
#SBATCH --job-name=sp-08_04tq-hold8
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --account=${AC2_CLUSTER_B_ACCOUNT}
#SBATCH --partition=${AC2_CLUSTER_B_PARTITION}
#SBATCH --qos=${AC2_CLUSTER_B_QOS}
#SBATCH --output=${AC2_CLUSTER_B_HOME}/sp-08_04tq-hold8-%j.out
#SBATCH --error=${AC2_CLUSTER_B_HOME}/sp-08_04tq-hold8-%j.err
#SBATCH --mail-user=${AC2_MAIL_USER}

set -xuo pipefail    # deliberately NO -e

REPO_ROOT="${SELF_PLAY_ROOT:-${AC2_CLUSTER_B_ROOT}/self-play}"
EXP="$REPO_ROOT/experiments/08_11_ablation1_replay_noq"
VENV="$REPO_ROOT/.venv"
SETUP_SH="$REPO_ROOT/experiments/08_11_ablation1_replay_noq/setup.sh"
ACTIVATE="unset ROCR_VISIBLE_DEVICES; source $VENV/bin/activate; source $SETUP_SH cuda-toolkit"
export SLURM_CPU_BIND=none
"$VENV/bin/python" -c "import vllm; assert vllm.__version__=='0.23.0', vllm.__version__" \
  || { echo "FATAL: $VENV is not the v023 stack"; exit 1; }

mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 --mem=0 -w "$head_node" hostname --ip-address)
if [[ "$head_node_ip" == *" "* ]]; then read -ra ADDR <<<"$head_node_ip"; head_node_ip="${ADDR[1]:-${ADDR[0]}}"; fi
port=6379
ip_head="$head_node_ip:$port"

INFO_BODY="JOBID=$SLURM_JOB_ID
HEAD_NODE=$head_node
HEAD_IP=$head_node_ip
RAY_ADDRESS=$ip_head
NODES='${nodes_array[*]}'"
printf '%s\n' "$INFO_BODY" > "$EXP/sandbox_${SLURM_JOB_ID}.info" || echo "WARN: EXP .info write failed (quota?)"
printf '%s\n' "$INFO_BODY" > "$HOME/sandbox_${SLURM_JOB_ID}.info" || true

# (A) Ray HEAD (v023 venv + CUDA JIT env)
srun --nodes=1 --ntasks=1 --mem=0 -w "$head_node" \
    bash -c "$ACTIVATE && ray start --head \
        --node-ip-address=$head_node_ip --port=$port \
        --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=8 --block" &
sleep 10

# (B) Ray WORKERS
for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
    node_i="${nodes_array[$i]}"
    srun --nodes=1 --ntasks=1 --mem=0 -w "$node_i" \
        bash -c "$ACTIVATE && ray start --address=$ip_head \
            --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=8 --block" &
    sleep 5
done

echo "holder ready: nodes (${nodes_array[*]}), Ray at $ip_head" || true
sleep infinity
