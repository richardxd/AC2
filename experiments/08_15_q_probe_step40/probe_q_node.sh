#!/usr/bin/env bash
# Fan one node's 8 GPUs into 8 TP=1 Q-evaluation replicas (stage B).
#
# Same shape as probe_gen_node.sh -- no Ray, one engine per GPU -- because Q is the SAME
# weights as the policy in the tied-Q main run, so nothing new has to be loaded or served.
# Each Q call generates gen_reserve=4 tokens, so this stage is dominated by prefill of the
# ~10k-token contexts, not decode.
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
NSHARDS="${SP_Q_NSHARDS:-32}"
OUTDIR="${SP_Q_OUTDIR:-$D/q}"
MODEL="${SP_PROBE_MODEL:-$D/model_hf}"

unset ROCR_VISIBLE_DEVICES
source "$REPO/experiments/08_13_tiedq_seed192/setup.sh" cuda-compat 2>/dev/null || true
source "$REPO/.venv/bin/activate"
SETUP_SH="$REPO/experiments/08_15_q_probe_step40/setup.sh"
[ -f "$SETUP_SH" ] && source "$SETUP_SH" cuda-toolkit || true

mkdir -p "$OUTDIR"
host="$(hostname)"
echo "[q-node $RANK/$host] launching $GPUS_PER_NODE replicas of $NSHARDS"

pids=()
for g in $(seq 0 $((GPUS_PER_NODE - 1))); do
  shard=$((RANK * GPUS_PER_NODE + g))
  out="$OUTDIR/q.shard${shard}.jsonl"
  [ -s "$out" ] && { echo "[q-node $RANK] shard $shard done -- skip"; continue; }
  # DISTINCT torch.distributed PORT PER REPLICA. Even at TP=1 each vLLM engine binds a
  # process-group port, and vLLM picks it randomly. Eight replicas per node started 2s
  # apart collide sooner or later: shard 13 of the step-57 probe died with
  # "DistNetworkError ... port: 43593 ... EADDRINUSE" and took 8 of 256 groups with it.
  # Deriving it from the shard index makes the assignment deterministic and collision-free.
  VLLM_PORT=$((53000 + shard * 8)) \
  CUDA_VISIBLE_DEVICES="$g" python "$E/probe_q.py" \
      --model "$MODEL" --gen-dir "$D/gen" \
      --shard "$shard" --num-shards "$NSHARDS" --out "$out" \
      --upto-step "${SP_Q_UPTO_STEP:-40}" \
      > "$OUTDIR/qreplica${shard}.log" 2>&1 &
  pids+=($!)
  sleep 2
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[q-node $RANK/$host] all replicas exited, failures=$fail"
exit 0
