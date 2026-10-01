#!/usr/bin/env bash
# submit_cluster_b_4node.sbatch -- 4-node allocation for 09_05_qtd_ready_s192b50_noaudit. Runs launch_cluster_b.sh as one
# srun step with one task per node: task 0 starts the Ray head and runs the driver
# (run_attach_cluster_b.sh) inline, tasks 1-3 join Ray as workers.
# Submit with scripts/sbatch_env.sh, which fills in the ${AC2_*} site settings.
#SBATCH --job-name=sp-0905qtd-b4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --account=${AC2_CLUSTER_B_ACCOUNT}
#SBATCH --partition=${AC2_CLUSTER_B_PARTITION}
#SBATCH --qos=${AC2_CLUSTER_B_QOS}
#SBATCH --output=${AC2_CLUSTER_B_HOME}/sp-0905qtd-b4-%j.out
#SBATCH --error=${AC2_CLUSTER_B_HOME}/sp-0905qtd-b4-%j.err

set -uo pipefail
REPO_ROOT="${SELF_PLAY_ROOT:-${AC2_CLUSTER_B_ROOT}/self-play}"
export SLURM_CPU_BIND=none
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --kill-on-bad-exit=0 \
    bash "$REPO_ROOT/experiments/09_05_qtd_ready_s192b50_noaudit/launch_cluster_b.sh"
