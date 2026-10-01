#!/usr/bin/env bash
# Re-run ONE lost generation shard end-to-end (gen -> Q -> judge) and fold it back in.
#
# WHY THIS EXISTS: a generation replica can die at engine init -- e.g. shard 13 of the step-57
# probe hit "DistNetworkError ... port: 43593 ... EADDRINUSE" when two TP=1 replicas on the
# same node picked the same torch.distributed port (VLLM_PORT now derives from the shard
# index) -- leaving its 8 groups missing from an otherwise finished probe. Re-running only
# that shard is far cheaper than re-running the whole 32-shard probe.
#
# HOW IT REJOINS THE DATA: the shard's generation goes to its OWN directory, so the Q and
# judge stages see exactly those 8 groups and shard them 1-of-1. Their outputs use a distinct
# `shard90` tag, and analyze_probe.py globs q.shard*/judged.shard* and joins everything by
# (qid, completion_index) -- so the backfilled groups merge in without renumbering anything.
# The generation file is moved into gen/ last, once its Q and judge rows exist, so an
# interrupted backfill can never leave a group present in gen/ but unscored.
#
#   SP_BACKFILL_SHARD=13 bash backfill_shard.sh
set -uo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }

REPO="${SELF_PLAY_ROOT:-${AC2_CLUSTER_A_ROOT}/self-play}"
E="$REPO/experiments/08_15_q_probe_step40"
D="${SP_PROBE_DIR:-$REPO/experiments/08_16_q_probe_step57}"
JID="${SP_PROBE_JOBID:?set SP_PROBE_JOBID to the allocation job id}"
SH="${SP_BACKFILL_SHARD:-13}"
NS="${SP_PROBE_NSHARDS:-32}"
STEP="${SP_PROBE_STEP:-57}"
RUN="$REPO/experiments/08_13_tiedq_seed192/run_data"
TMP="$D/backfill${SH}"
mkdir -p "$TMP"

say() { echo "[backfill $SH] $* $(date -Is)"; }

# ---- 1. regenerate just this shard ---------------------------------------------------
if [ ! -s "$TMP/gen.shard${SH}.jsonl" ]; then
  say "generating"
  srun --overlap --jobid="$JID" --nodes=1 --ntasks=1 bash -c "
    set -uo pipefail
    unset ROCR_VISIBLE_DEVICES
    source $REPO/experiments/08_13_tiedq_seed192/setup.sh cuda-compat 2>/dev/null || true
    source $REPO/.venv/bin/activate
    source $REPO/experiments/08_15_q_probe_step40/setup.sh cuda-toolkit 2>/dev/null || true
    VLLM_PORT=\$((51000 + $SH * 8)) CUDA_VISIBLE_DEVICES=0 python $E/probe_gen.py \
      --model $D/model_hf --probe-set $D/probe_set.jsonl \
      --shard $SH --num-shards $NS --out $TMP/gen.shard${SH}.jsonl
  " > "$TMP/gen.log" 2>&1
fi
[ -s "$TMP/gen.shard${SH}.jsonl" ] || { say "GEN FAILED"; tail -12 "$TMP/gen.log"; exit 1; }
say "generated $(wc -l < "$TMP/gen.shard${SH}.jsonl") groups"

# ---- 2. Q at both sites for those groups ---------------------------------------------
if [ ! -s "$D/q/q.shard90.jsonl" ]; then
  say "Q (prefix + cut)"
  srun --overlap --jobid="$JID" --nodes=1 --ntasks=1 bash -c "
    set -uo pipefail
    unset ROCR_VISIBLE_DEVICES
    source $REPO/experiments/08_13_tiedq_seed192/setup.sh cuda-compat 2>/dev/null || true
    source $REPO/.venv/bin/activate
    source $REPO/experiments/08_15_q_probe_step40/setup.sh cuda-toolkit 2>/dev/null || true
    VLLM_PORT=\$((53000 + $SH * 8)) CUDA_VISIBLE_DEVICES=1 python $E/probe_q.py \
      --model $D/model_hf --gen-dir $TMP --shard 0 --num-shards 1 \
      --upto-step $STEP --run-dir $RUN --out $D/q/q.shard90.jsonl
  " > "$TMP/q.log" 2>&1
fi
[ -s "$D/q/q.shard90.jsonl" ] || { say "Q FAILED"; tail -12 "$TMP/q.log"; exit 1; }
say "Q rows $(wc -l < "$D/q/q.shard90.jsonl")"

# ---- 3. judge those groups (one node, one shard) --------------------------------------
if [ ! -s "$D/judged/judged.shard90.jsonl" ]; then
  say "judging"
  SP_JUDGE_GEN_DIR="$TMP" SP_JUDGE_NSHARDS=1 SP_JUDGE_OUTTAG=90 SP_PROBE_DIR="$D" \
  srun --overlap --jobid="$JID" --nodes=1 --ntasks=1 bash "$E/probe_judge_node.sh" \
      > "$TMP/judge.log" 2>&1
fi
[ -s "$D/judged/judged.shard90.jsonl" ] || { say "JUDGE FAILED"; tail -15 "$TMP/judge.log"; exit 1; }
say "judged rows $(wc -l < "$D/judged/judged.shard90.jsonl")"

# ---- 4. only now fold the generation file in ------------------------------------------
cp "$TMP/gen.shard${SH}.jsonl" "$D/gen/gen.shard${SH}.jsonl"
say "gen shards now $(ls "$D"/gen/gen.shard*.jsonl | wc -l)/32"

source "$REPO/.venv/bin/activate"
python "$E/analyze_probe.py" --gen-dir "$D/gen" --q-dir "$D/q" --judged-dir "$D/judged" \
    --run-dir "$RUN" --dump-pairs > "$D/RESULT.txt" 2>&1
grep -v '^PAIRS' "$D/RESULT.txt" | head -40
say "DONE"
echo BACKFILL_DONE
