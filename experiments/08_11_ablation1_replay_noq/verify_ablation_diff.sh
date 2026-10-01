#!/usr/bin/env bash
# verify_ablation_diff.sh — prove that this arm differs from the main run in ONE factor.
#
# The value of an ablation is entirely in the size of its diff, and that diff is easy to grow
# by accident: a knob added here and not there, a context value nudged, a judge revision drift.
# So the claim is machine-checkable rather than a sentence in a README.
#
# Compares the ENVS arrays of the two attach scripts, normalizing paths and the pinned judge.
# Exits non-zero if anything outside the EXPECTED delta appears.
#
# Both configurations use the same response budget, so the CONTEXT GROUP is expected to be
# SHARED, not a permitted difference: otherwise part of any reward gap would be context rather
# than the critic. If the budget changes, it must change in both attach scripts or this fails.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN="${MAIN_RUN_DIR:-$HERE/../08_13_tiedq_seed192}"
[ -f "$MAIN/run_attach_cluster_b.sh" ] || { echo "main run not found at $MAIN (set MAIN_RUN_DIR)"; exit 1; }

_envs() {   # extract the ENVS array as KEY=VAL lines, path- and judge-normalized
  awk '/^ENVS=\(/,/^\)/' "$1" \
    | grep -o '"[A-Z_][A-Z_0-9]*=[^"]*"' | tr -d '"' \
    | sed 's|\$EXP.*|<path>|; s|\$SEED|<path>|; s|\$JUDGE_[A-Z_]*|<pinned>|' | sort
}
_envs "$MAIN/run_attach_cluster_b.sh" > /tmp/.abl_main.$$
_envs "$HERE/run_attach_cluster_b.sh" > /tmp/.abl_this.$$
trap 'rm -f /tmp/.abl_main.$$ /tmp/.abl_this.$$' EXIT

REMOVED=$(comm -23 /tmp/.abl_main.$$ /tmp/.abl_this.$$)
ADDED=$(comm -13 /tmp/.abl_main.$$ /tmp/.abl_this.$$)
SHARED=$(comm -12 /tmp/.abl_main.$$ /tmp/.abl_this.$$ | wc -l | tr -d ' ')

echo "=== removed vs the main run ($(echo "$REMOVED" | grep -c . || true)) ==="; echo "$REMOVED" | sed 's/^/  -/'
echo "=== added ($(echo "$ADDED" | grep -c . || true)) ==="; echo "$ADDED" | sed 's/^/  +/'
echo "=== shared: $SHARED ==="

rc=0
# Everything removed must be a Q knob (or the run name). Anything else means the ablation
# quietly became a multi-factor change.
while read -r kv; do
  [ -z "$kv" ] && continue
  case "${kv%%=*}" in
    SP_Q_*|SP_EXPERIMENT_NAME) ;;
    *) echo "UNEXPECTED REMOVAL: $kv"; echo "   -> not a Q knob and not part of the context group;"
       echo "      this is no longer a single-factor ablation."; rc=1 ;;
  esac
done <<< "$REMOVED"

while read -r kv; do
  [ -z "$kv" ] && continue
  case "$kv" in
    SP_Q_ENABLE=0|SP_EXPERIMENT_NAME=*) ;;
    *) echo "UNEXPECTED ADDITION: $kv"; rc=1 ;;
  esac
done <<< "$ADDED"

# Q must be OFF and pinned, not merely absent.
grep -q '"SP_Q_ENABLE=0"' "$HERE/run_attach_cluster_b.sh" || { echo "SP_Q_ENABLE is not pinned to 0"; rc=1; }

# The hypers that define the arm must be present AND shared -- absence would also produce an
# empty diff, which must not read as success.
# (a) VALUE-PINNED: these define the arm and are not expected to ever move. Absence would also
# shrink the diff, so check the exact value is present AND shared.
for k in SP_TRAIN_BATCH_SIZE=256 SP_REPLAY_N=192 SP_PPO_MINI_BATCH=96 SP_LR=2e-6 \
         SP_REPLAY_CUT_GRAIN=10000 SP_REPLAY_BOUND=256 SP_ROLLOUT_TP=4 SP_DIFF_SAMPLING=0 \
         SP_REPLAY_GLOBAL_SAMPLING=question; do
  comm -12 /tmp/.abl_main.$$ /tmp/.abl_this.$$ | grep -qx "$k" || {
    echo "CORE HYPER NOT SHARED: $k (expected identical in both arms)"; rc=1; }
done

# (b) MUST-AGREE, value free: the context group may legitimately move (e.g. 50k -> 75k in both
# configurations together), so pinning its value here would fail that change. Require only
# that both carry the SAME value, whatever it is.
for k in SP_MAX_RESPONSE_LEN SP_PPO_MAX_TOKEN_LEN SP_ACTOR_GPU_MEM_UTIL; do
  a=$(grep "^$k=" /tmp/.abl_main.$$ || true); b=$(grep "^$k=" /tmp/.abl_this.$$ || true)
  [ -n "$a" ] && [ -n "$b" ] || { echo "CONTEXT KEY MISSING: $k (main='$a' this='$b')"; rc=1; continue; }
  [ "$a" = "$b" ] || {
    echo "CONTEXT DIFFERS: $k -- main run '$a' vs this arm '$b'."
    echo "   -> the context group must be staged in BOTH arms together, or part of any reward"
    echo "      gap is context rather than p."; rc=1; }
done

[ "$rc" -eq 0 ] && echo "OK: single-factor ablation -- Q removed, everything else identical" || echo "FAILED"
exit $rc
