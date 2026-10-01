#!/usr/bin/env bash
# Generic end-to-end Q probe at ANY 08_13_tiedq_seed192 checkpoint, on a SUBSET of a holder's
# nodes, so two probes can run side by side and keep every GPU busy.
#
# Same five stages as run_probe57.sh (merge -> probe set -> stage A gen -> stage B Q at prefix
# and prefix+10k -> stage C DS4 judge -> analysis), with three generalisations:
#   * SP_PROBE_NODES is a comma list of the nodes THIS probe may use (srun -w); NNODES is
#     derived from it, shards = 8 x NNODES, judge shards = NNODES (one TP=8 judge per node);
#   * every srun is pinned to those nodes, so a second probe on the other nodes of the same
#     holder never shares a GPU with this one (stage C stops Ray on ITS nodes only);
#   * the model merge runs on the probe's first node.
# Stage scripts are untouched (probe_gen_node.sh / probe_q_node.sh / probe_judge_node.sh read
# SP_PROBE_DIR, SP_PROBE_NSHARDS, SP_Q_NSHARDS, SP_JUDGE_NSHARDS from the environment).
#
#   SP_PROBE_STEP=80 SP_PROBE_DIR=$REPO/experiments/09_05_q_probe_step80 \
#   SP_PROBE_JOBID=<id> SP_PROBE_NODES=<node1>,<node2> \
#     nohup bash experiments/08_15_q_probe_step40/run_probe_generic.sh > .../chain80.log 2>&1 &
set -uo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }

REPO="${SELF_PLAY_ROOT:-${AC2_CLUSTER_A_ROOT}/self-play}"
E="$REPO/experiments/08_15_q_probe_step40"
STEP="${SP_PROBE_STEP:?SP_PROBE_STEP required}"
D="${SP_PROBE_DIR:?SP_PROBE_DIR required}"
JID="${SP_PROBE_JOBID:?SP_PROBE_JOBID required}"
NODES="${SP_PROBE_NODES:?SP_PROBE_NODES required (comma list)}"
NNODES=$(echo "$NODES" | tr ',' '\n' | grep -c .)
FIRST=$(echo "$NODES" | cut -d, -f1)
NSHARDS=$(( NNODES * 8 ))
RUN="$REPO/experiments/08_13_tiedq_seed192/run_data"
export SP_PROBE_DIR="$D" SP_PROBE_STEP="$STEP"
export SP_PROBE_NSHARDS="$NSHARDS" SP_Q_NSHARDS="$NSHARDS" SP_JUDGE_NSHARDS="$NNODES"
mkdir -p "$D"

say() { echo "[probe$STEP] $* $(date -Is)"; }
SRUN="srun --overlap --jobid=$JID --nodes=$NNODES --ntasks=$NNODES --ntasks-per-node=1 -w $NODES"

say "nodes=$NODES nnodes=$NNODES shards=$NSHARDS judge_shards=$NNODES dir=$D"

# ---- 0. clear stale servers on OUR nodes only ------------------------------------------
$SRUN bash -c 'pkill -f "[v]llm serve" 2>/dev/null; exit 0' >/dev/null 2>&1
sleep 3

# ---- 1. merge the step actor into an HF model (first node) -----------------------------
if [ ! -s "$D/model_hf/config.json" ]; then
  say "merging step $STEP actor on $FIRST"
  srun --overlap --jobid="$JID" --nodes=1 --ntasks=1 -w "$FIRST" bash "$E/merge_ckpt.sh" \
      > "$D/merge.log" 2>&1
  grep -q "torch_dtype=bfloat16" "$D/merge.log" || { say "MERGE FAILED"; tail -20 "$D/merge.log"; echo "PROBE${STEP}_FAILED"; exit 1; }
else
  say "model_hf already present -- skip merge"
fi

# ---- 2. probe set from this step's readiness -------------------------------------------
if [ ! -s "$D/probe_set.jsonl" ]; then
  say "building probe set at step $STEP"
  source "$REPO/.venv/bin/activate"
  python "$E/build_probe_set.py" --run-dir "$RUN" --step "$STEP" \
      --n 256 --out "$D/probe_set.jsonl" > "$D/build.log" 2>&1
  [ -s "$D/probe_set.jsonl" ] || { say "PROBE SET EMPTY"; tail -20 "$D/build.log"; echo "PROBE${STEP}_FAILED"; exit 1; }
  tail -6 "$D/build.log"
fi

# ---- 3. stage A: generation, 8 TP=1 replicas per node ---------------------------------
if [ "$(ls "$D"/gen/gen.shard*.jsonl 2>/dev/null | wc -l)" -lt "$NSHARDS" ]; then
  say "stage A: generation"
  $SRUN bash "$E/probe_gen_node.sh" > "$D/stage_a.log" 2>&1
fi
say "gen shards: $(ls "$D"/gen/gen.shard*.jsonl 2>/dev/null | wc -l)/$NSHARDS"

# ---- 4. stage B: Q at prefix and prefix+10k -------------------------------------------
if [ "$(ls "$D"/q/q.shard*.jsonl 2>/dev/null | wc -l)" -lt "$NSHARDS" ]; then
  say "stage B: Q at both sites"
  SP_Q_UPTO_STEP="$STEP" $SRUN bash "$E/probe_q_node.sh" > "$D/stage_b.log" 2>&1
fi
say "q shards: $(ls "$D"/q/q.shard*.jsonl 2>/dev/null | wc -l)/$NSHARDS"

# ---- 5. stage C: DS4 judge, one TP=8 server per node (stops Ray on OUR nodes) ----------
if [ "$(ls "$D"/judged/judged.shard*.jsonl 2>/dev/null | wc -l)" -lt "$NNODES" ]; then
  say "stage C: DS4 judge"
  $SRUN bash "$E/probe_judge_node.sh" > "$D/stage_c.log" 2>&1
fi
say "judged shards: $(ls "$D"/judged/judged.shard*.jsonl 2>/dev/null | wc -l)/$NNODES"

# ---- 6. analysis -----------------------------------------------------------------------
say "analysing"
source "$REPO/.venv/bin/activate"
python "$E/analyze_probe.py" --gen-dir "$D/gen" --q-dir "$D/q" --judged-dir "$D/judged" \
    --run-dir "$RUN" --dump-pairs > "$D/RESULT.txt" 2>&1
grep -v '^PAIRS' "$D/RESULT.txt" | sed -n '1,40p'
say "DONE -- RESULT at $D/RESULT.txt"
echo "PROBE${STEP}_DONE"
