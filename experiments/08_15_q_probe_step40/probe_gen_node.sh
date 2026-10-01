#!/usr/bin/env bash
# Fan one node's 8 GPUs into 8 independent TP=1 probe_gen replicas.
#
# Node rank comes from SLURM_PROCID (one task per node), so shard = rank*8 + gpu and the
# 4-node job covers shards 0..31 with no coordination between nodes.
#
# NO RAY, ON PURPOSE. Each replica is a self-contained single-GPU vLLM engine, so this
# needs neither the holder's Ray cluster nor the Ray restart that every attached
# driver in this repo has to do. It also means a paused training run's Ray steps on the
# same allocation can stay up untouched.
#
# The ACTIVATE sequence mirrors the holder sbatch exactly (cuda shim -> venv -> cuda_env);
# a bare `source .venv/bin/activate` is NOT enough on cluster A.
set -uo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }

REPO="${SELF_PLAY_ROOT:-${AC2_CLUSTER_A_ROOT}/self-play}"
E="$REPO/experiments/08_15_q_probe_step40"   # scripts live here
# data (model_hf, probe_set, gen/, q/, judged/) can live elsewhere so a second
# probe at a different checkpoint does not overwrite the first one's artifacts
D="${SP_PROBE_DIR:-$E}"
GPUS_PER_NODE="${SP_PROBE_GPUS:-8}"
RANK="${SLURM_PROCID:-0}"
NSHARDS="${SP_PROBE_NSHARDS:-32}"
OUTDIR="${SP_PROBE_OUTDIR:-$D/gen}"
MODEL="${SP_PROBE_MODEL:-$D/model_hf}"

unset ROCR_VISIBLE_DEVICES
source "$REPO/experiments/08_13_tiedq_seed192/setup.sh" cuda-compat 2>/dev/null || true
source "$REPO/.venv/bin/activate"
SETUP_SH="$REPO/experiments/08_15_q_probe_step40/setup.sh"
[ -f "$SETUP_SH" ] && source "$SETUP_SH" cuda-toolkit || true

mkdir -p "$OUTDIR"
host="$(hostname)"
echo "[node $RANK/$host] launching $GPUS_PER_NODE replicas (shards $((RANK*GPUS_PER_NODE))..$((RANK*GPUS_PER_NODE+GPUS_PER_NODE-1))) of $NSHARDS"

pids=()
for g in $(seq 0 $((GPUS_PER_NODE - 1))); do
  shard=$((RANK * GPUS_PER_NODE + g))
  out="$OUTDIR/gen.shard${shard}.jsonl"
  if [ -s "$out" ]; then echo "[node $RANK] shard $shard already done -- skip"; continue; fi
  # DISTINCT torch.distributed PORT PER REPLICA. Even at TP=1 each vLLM engine binds a
  # process-group port, and vLLM picks it randomly. Eight replicas per node started 2s
  # apart collide sooner or later: shard 13 of the step-57 probe died with
  # "DistNetworkError ... port: 43593 ... EADDRINUSE" and took 8 of 256 groups with it.
  # Deriving it from the shard index makes the assignment deterministic and collision-free.
  VLLM_PORT=$((51000 + shard * 8)) \
  CUDA_VISIBLE_DEVICES="$g" python "$E/probe_gen.py" \
      --model "$MODEL" --probe-set "$D/probe_set.jsonl" \
      --shard "$shard" --num-shards "$NSHARDS" --out "$out" \
      > "$OUTDIR/replica${shard}.log" 2>&1 &
  pids+=($!)
  sleep 2      # stagger engine init so 8 processes do not contend on the same weight read
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[node $RANK/$host] all replicas exited, failures=$fail"
exit 0     # never fail the srun: a single bad replica must not kill the other 3 nodes
