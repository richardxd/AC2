#!/usr/bin/env bash
# End-to-end re-run of the Q probe at a LATER checkpoint, with the Q_prefix site added.
#
# Differences from the step-40 probe, all deliberate:
#   * step 57 instead of 40 -- 17 steps of further Q training;
#   * readiness re-sampled from step 57's own q_state.json, so the ready set is whatever the
#     run currently believes. NOTE this still includes problems that went ready BEFORE
#     SP_Q_READY_REQUIRE_BANK was enabled: that gate only guards the ready TRANSITION
#     (`not self.ready.get(qid, False)`), it never un-readies anything. Groups are tagged with
#     bank membership so the analysis can split solved/unsolved rather than pretending the
#     gate cleaned the population;
#   * Q is now evaluated at TWO sites -- the shared prefix (completion_index = -1) and
#     prefix+10k -- so degradation between them is measurable on the same group.
#
# All artifacts go to SP_PROBE_DIR, leaving the step-40 probe intact for comparison.
#
#   nohup bash experiments/08_15_q_probe_step40/run_probe57.sh > .../chain57.log 2>&1 &
set -uo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }

REPO="${SELF_PLAY_ROOT:-${AC2_CLUSTER_A_ROOT}/self-play}"
E="$REPO/experiments/08_15_q_probe_step40"
export SP_PROBE_DIR="${SP_PROBE_DIR:-$REPO/experiments/08_16_q_probe_step57}"
export SP_PROBE_STEP="${SP_PROBE_STEP:-57}"
JID="${SP_PROBE_JOBID:?set SP_PROBE_JOBID to the allocation job id}"
RUN="$REPO/experiments/08_13_tiedq_seed192/run_data"
D="$SP_PROBE_DIR"
mkdir -p "$D"

say() { echo "[probe57] $* $(date -Is)"; }

# ---- 0. the GPUs must be free: this runs on a training holder whose run is paused ------
say "checking holder $JID is idle"
srun --overlap --jobid="$JID" --nodes=4 --ntasks=4 --ntasks-per-node=1 \
     bash -c 'pkill -f "[v]llm serve" 2>/dev/null; exit 0' >/dev/null 2>&1
sleep 3

# ---- 1. merge the step-57 actor ------------------------------------------------------
if [ ! -s "$D/model_hf/config.json" ]; then
  say "merging step $SP_PROBE_STEP actor"
  srun --overlap --jobid="$JID" --nodes=1 --ntasks=1 bash "$E/merge_ckpt.sh" \
      > "$D/merge.log" 2>&1
  grep -q "torch_dtype=bfloat16" "$D/merge.log" || { say "MERGE FAILED"; tail -20 "$D/merge.log"; exit 1; }
else
  say "model_hf already present -- skip merge"
fi

# ---- 2. probe set from step-57 readiness ---------------------------------------------
if [ ! -s "$D/probe_set.jsonl" ]; then
  say "building probe set at step $SP_PROBE_STEP"
  source "$REPO/.venv/bin/activate"
  python "$E/build_probe_set.py" --run-dir "$RUN" --step "$SP_PROBE_STEP" \
      --n 256 --out "$D/probe_set.jsonl" > "$D/build.log" 2>&1
  [ -s "$D/probe_set.jsonl" ] || { say "PROBE SET EMPTY"; tail -20 "$D/build.log"; exit 1; }
  tail -6 "$D/build.log"
fi

# ---- 3. generation -------------------------------------------------------------------
if [ "$(ls "$D"/gen/gen.shard*.jsonl 2>/dev/null | wc -l)" -lt 32 ]; then
  say "stage A: generation"
  srun --overlap --jobid="$JID" --nodes=4 --ntasks=4 --ntasks-per-node=1 \
       bash "$E/probe_gen_node.sh" > "$D/stage_a.log" 2>&1
fi
say "gen shards: $(ls "$D"/gen/gen.shard*.jsonl 2>/dev/null | wc -l)/32"

# ---- 4. Q at BOTH sites --------------------------------------------------------------
if [ "$(ls "$D"/q/q.shard*.jsonl 2>/dev/null | wc -l)" -lt 32 ]; then
  say "stage B: Q at prefix and prefix+10k"
  SP_Q_UPTO_STEP="$SP_PROBE_STEP" \
  srun --overlap --jobid="$JID" --nodes=4 --ntasks=4 --ntasks-per-node=1 \
       bash "$E/probe_q_node.sh" > "$D/stage_b.log" 2>&1
fi
say "q shards: $(ls "$D"/q/q.shard*.jsonl 2>/dev/null | wc -l)/32"

# ---- 5. judge (stops Ray; must follow stage B) ----------------------------------------
if [ "$(ls "$D"/judged/judged.shard*.jsonl 2>/dev/null | wc -l)" -lt 4 ]; then
  say "stage C: DS4 judge"
  srun --overlap --jobid="$JID" --nodes=4 --ntasks=4 --ntasks-per-node=1 \
       bash "$E/probe_judge_node.sh" > "$D/stage_c.log" 2>&1
fi
say "judged shards: $(ls "$D"/judged/judged.shard*.jsonl 2>/dev/null | wc -l)/4"

# ---- 6. analysis ---------------------------------------------------------------------
say "analysing"
source "$REPO/.venv/bin/activate"
python "$E/analyze_probe.py" --gen-dir "$D/gen" --q-dir "$D/q" --judged-dir "$D/judged" \
    --run-dir "$RUN" --dump-pairs > "$D/RESULT.txt" 2>&1
sed -n '1,40p' "$D/RESULT.txt"
say "DONE -- RESULT at $D/RESULT.txt"
echo PROBE57_DONE
