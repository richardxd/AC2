#!/usr/bin/env bash
# submit_cluster_a_4node.sbatch -- 4-node allocation for 08_26_s192b40_g10k_noaudit on cluster A: starts a Ray
# head and workers and writes sandbox_<JOBID>.info for run_attach_cluster_a.sh.
# Submit with scripts/sbatch_env.sh, which fills in the ${AC2_*} site settings.
#SBATCH --job-name=sp-0826b40-g10k
#SBATCH --mail-user=${AC2_MAIL_USER}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1800G
#SBATCH --time=2-00:00:00
#SBATCH --partition=${AC2_CLUSTER_A_PARTITION}
#SBATCH --account=${AC2_CLUSTER_A_ACCOUNT}
#SBATCH --output=${AC2_CLUSTER_A_ROOT}/logs/sp-0827seg-c50k-%j.out
#SBATCH --error=${AC2_CLUSTER_A_ROOT}/logs/sp-0827seg-c50k-%j.err

set -xuo pipefail    # deliberately NO -e

REPO_ROOT="${SELF_PLAY_ROOT:-${AC2_CLUSTER_A_ROOT}/self-play}"
E="$REPO_ROOT/experiments/08_26_s192b40_g10k_noaudit"
VENV="$REPO_ROOT/.venv"
SETUP_SH="$REPO_ROOT/experiments/08_26_s192b40_g10k_noaudit/setup.sh"
# The shim MUST precede ray start on every node: cu129 PTX cannot be JIT'd by the 565
# driver's compiler, and engines inherit Ray's environment. Without it every judge engine
# dies with 'PTX ... unsupported toolchain'.
ACTIVATE="unset ROCR_VISIBLE_DEVICES; source $REPO_ROOT/experiments/08_26_s192b40_g10k_noaudit/setup.sh cuda-compat; source $VENV/bin/activate; [ -f $SETUP_SH ] && source $SETUP_SH cuda-toolkit || true"

# SLURM 23.11+ propagates --cpus-per-task into srun via SLURM_TRES_PER_TASK, which then
# conflicts with the explicit --cpus-per-task on the ray srun steps and kills the holder
# seconds in. Pin the old variable and drop the new one.
export SLURM_CPUS_PER_TASK=32
unset SLURM_TRES_PER_TASK
export SLURM_CPU_BIND=none

mkdir -p ${AC2_CLUSTER_A_ROOT}/logs || true

mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
if [[ "$head_node_ip" == *" "* ]]; then read -ra ADDR <<<"$head_node_ip"; head_node_ip="${ADDR[1]:-${ADDR[0]}}"; fi
port=6379
ip_head="$head_node_ip:$port"

INFO_BODY="JOBID=$SLURM_JOB_ID
HEAD_NODE=$head_node
HEAD_IP=$head_node_ip
RAY_ADDRESS=$ip_head
NODES='${nodes_array[*]}'"
printf '%s\n' "$INFO_BODY" > "$E/sandbox_${SLURM_JOB_ID}.info" || echo "WARN: .info write failed"
printf '%s\n' "$INFO_BODY" > "${AC2_CLUSTER_A_ROOT}/sandbox_${SLURM_JOB_ID}.info" || true

# (A) Ray HEAD
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "$ACTIVATE && ray start --head \
        --node-ip-address=$head_node_ip --port=$port \
        --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=8 --block" &
sleep 10

# (B) Ray WORKERS
for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
    node_i="${nodes_array[$i]}"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash -c "$ACTIVATE && ray start --address=$ip_head \
            --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=8 --block" &
    sleep 5
done

echo "cluster_a holder ready: nodes (${nodes_array[*]}), Ray at $ip_head" || true
sleep infinity
