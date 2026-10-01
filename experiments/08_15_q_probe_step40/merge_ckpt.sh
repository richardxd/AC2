#!/usr/bin/env bash
# Merge an 08_13_tiedq_seed192 FSDP actor checkpoint (global_step_$SP_PROBE_STEP, default 40;
# 32 shards, 47 GB) into an HF model vLLM can serve. The probe needs the policy AND the Q head,
# and the run is WEIGHT-TIED -- one set of weights carries both -- so this single merged model
# answers both the rollout continuations and the Q calls. No separate q_model/ exists to merge.
#
# WHY A FROZEN COPY: once training continues, checkpoint retention can prune global_step_N,
# and every number the probe produces is only interpretable against these exact weights.
# Merge once, keep the copy.
#
# THE torch_dtype TRAP: verl's model_merger emits a
# config.json with `torch_dtype: null`. vLLM then falls back to fp32, silently doubling
# memory and changing numerics. We patch it to bfloat16 and REFUSE to finish if the patch
# did not take, rather than leave a landmine for whoever serves this next.
#
#   srun --overlap --jobid=<id> --nodes=1 --ntasks=1 -w <node> bash merge_ckpt.sh
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }

REPO="${SELF_PLAY_ROOT:-${AC2_CLUSTER_A_ROOT}/self-play}"
STEP="${SP_PROBE_STEP:-40}"
E="$REPO/experiments/08_15_q_probe_step40"
# data dir: a probe at a different checkpoint writes its own model_hf/ etc.
D="${SP_PROBE_DIR:-$E}"
SRC="$REPO/experiments/08_13_tiedq_seed192/run_data/checkpoints/global_step_${STEP}/actor"
OUT="${SP_PROBE_MODEL:-$D/model_hf}"

[ -d "$SRC" ] || { echo "FATAL: $SRC missing (pruned by retention?)"; exit 1; }
if [ -f "$OUT/config.json" ] && [ -n "$(ls "$OUT"/*.safetensors 2>/dev/null)" ]; then
  echo "[merge] $OUT already populated -- nothing to do"; exit 0
fi
mkdir -p "$OUT"

source "$REPO/.venv/bin/activate"
SETUP_SH="$REPO/experiments/08_15_q_probe_step40/setup.sh"
[ -f "$SETUP_SH" ] && source "$SETUP_SH" cuda-toolkit || true

echo "[merge] $SRC -> $OUT"
python -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$OUT"

# tokenizer/chat-template come from the run's own snapshot, not the merger
for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
         vocab.json merges.txt special_tokens_map.json; do
  [ -f "$OUT/$f" ] || cp -n "$SRC/huggingface/$f" "$OUT/$f" 2>/dev/null || true
done

python - "$OUT/config.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
if c.get("torch_dtype") in (None, "null"):
    c["torch_dtype"] = "bfloat16"
    json.dump(c, open(p, "w"), indent=2)
    print("[merge] patched torch_dtype -> bfloat16")
else:
    print("[merge] torch_dtype already %r" % c["torch_dtype"])
PY

# refuse to hand off a config that would serve in fp32
python - "$OUT/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
assert c.get("torch_dtype") == "bfloat16", "torch_dtype patch did not take: %r" % c.get("torch_dtype")
print("[merge] verified torch_dtype=bfloat16")
PY

ls -la "$OUT"
echo "[merge] done step=$STEP"
