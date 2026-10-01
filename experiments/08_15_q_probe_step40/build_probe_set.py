"""Sample 256 randomly-truncated prefixes on READY problems, for the step-40 Q probe.

WHAT THE PROBE IS FOR. In the `short` lane the trainer throws away the rest of a rollout at
prefix+10k and uses Q's answer AS the reward. Nothing in the live run ever measures what that
substitution costs, because the moment Q is consumed the true reward is never observed --
the rollout was cut. This probe re-runs the same setup with the cut REMOVED: continue every
rollout to the end, score it with the judge, and separately ask Q what it would have said at
10k. The comparison mean_i(Q_i) vs mean_i(r_i) is then the substitution error itself.

WHY MORE THAN ONE STEP IS SCANNED. A step contains only ~192 replay groups, so a single
rollouts/<N>.jsonl cannot supply 256 DISTINCT problems. Files are scanned newest-first and
the first record seen for a uid wins, so every prefix comes from that problem's most recent
attempt available at or before the probe step. Pooling is over the SOURCE of the prefix
only -- the policy that continues it is step 40 for every group, so no arm is mixed.

READY IS READ FROM THE CHECKPOINT, NOT RECOMPUTED. q_state.json's `ready` map is the exact
gate state the trainer had at step 40 (1,415 problems), so "ready" here means what it meant
in the run rather than a reconstruction that could drift.

THE JOIN, AND THE TRAP IN IT. `ready` is keyed by `sp_qid` -- a sha1 of the problem statement,
stable across re-draws. A rollout row's `uid` is NOT that: it is a fresh uuid4 per group per
step (difficulty.qid_from_text's own docstring says so outright: "Identical across re-draws of
the same question (uid is NOT)"). Joining ready-ness on `uid` silently matches NOTHING (0 ready
uids across 12 step files). The correct path is the one
the dataset itself uses: `extra_index` is the train-parquet row index, and the trainer builds
its qids as `qid_from_messages(dataframe[i][prompt_key])` per row. So we rebuild that same
index -> qid map from the same parquet and join through it, importing verl's own hash rather
than reimplementing sha1 so the two cannot drift. The overlap is asserted non-empty.

THE TRUNCATION, and its one deliberate bias. The cut point is uniform over a fraction of the
source attempt, which is what "randomly truncated" asks for. But a cut so late that fewer
than `--min-budget` tokens remain under the 50k response cap cannot produce a rollout that
reaches 10k, so it could never yield a Q value at all -- it would enter the group as a hole
rather than as data. Those cuts are clamped back, and the count is reported. This biases the
prefix distribution slightly EARLY relative to pure uniform; that is a real caveat and is
printed rather than hidden.

    python build_probe_set.py --run-dir <run_data> --n 256 --out probe_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.expandvars("${AC2_CLUSTER_A_ROOT}/self-play/"
                                         "experiments/08_13_tiedq_seed192/run_data"))
    ap.add_argument("--step", type=int, default=40, help="checkpoint step; also the newest "
                                                         "rollouts file scanned")
    ap.add_argument("--n", type=int, default=256, help="distinct ready problems to sample")
    ap.add_argument("--scan-back", type=int, default=12,
                    help="how many earlier rollouts files to fall back through")
    ap.add_argument("--resp-cap", type=int, default=50000)
    ap.add_argument("--budget-g", type=int, default=10000,
                    help="the Q cut offset; only continuations exceeding this get a Q value")
    ap.add_argument("--min-budget", type=int, default=12000,
                    help="tokens that must remain under the cap after the cut, so the "
                         "rollout can reach budget-g and a Q value is possible")
    ap.add_argument("--min-frac", type=float, default=0.05)
    ap.add_argument("--max-frac", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=192)
    ap.add_argument("--data-dir", default=os.path.join(os.environ.get("HOME", ""),
                                                       "data/fineproofs"),
                    help="holds train.parquet; must be the SAME file the run trained on, "
                         "since the join is by row index")
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--out", default="probe_set.jsonl")
    args = ap.parse_args()

    ck = os.path.join(args.run_dir, "checkpoints", "global_step_%d" % args.step, "q_state.json")
    ready = json.load(open(ck, encoding="utf-8"))["ready"]
    ready = {k for k, v in ready.items() if v}
    print("ready problems at step %d: %d" % (args.step, len(ready)))

    # index -> qid, rebuilt exactly as the dataset does it (verl's own hash, not a copy)
    import pandas as pd
    from verl.trainer.ppo.difficulty import qid_from_messages
    train = os.path.join(args.data_dir, "train.parquet")
    df = pd.read_parquet(train)
    row_qids = [qid_from_messages(df[args.prompt_key].iloc[i]) for i in range(len(df))]
    ready_rows = {i for i, q in enumerate(row_qids) if q in ready}
    print("train.parquet rows: %d   of which ready: %d" % (len(row_qids), len(ready_rows)))
    if not ready_rows:
        raise SystemExit(
            "FATAL: zero parquet rows map into the ready set. The index->qid join is wrong "
            "(wrong --data-dir, wrong --prompt-key, or a re-generated parquet whose row order "
            "differs from the run's). Refusing to emit a probe set that would silently sample "
            "non-ready problems.")

    rng = random.Random(args.seed)
    chosen: dict[str, dict] = {}
    scanned = []
    for s in range(args.step, max(-1, args.step - args.scan_back), -1):
        if len(chosen) >= args.n:
            break
        path = os.path.join(args.run_dir, "rollouts", "%d.jsonl" % s)
        if not os.path.exists(path):
            continue
        got = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if len(chosen) >= args.n:
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = r.get("extra_index")
                if idx is None:
                    continue
                idx = int(idx)
                if idx not in ready_rows:          # not a ready problem
                    continue
                qid = row_qids[idx]
                if qid in chosen:                  # newest attempt for this problem wins
                    continue
                resp = r.get("response_token_ids") or []
                prompt = r.get("prompt_token_ids") or []
                # a source attempt with no room to cut AND still leave a continuation is
                # useless as a prefix, whatever the fraction drawn
                if len(resp) < 64 or not prompt:
                    continue
                chosen[qid] = {"qid": qid, "row_index": idx, "src_uid": str(r.get("uid")),
                               "src_step": s, "prompt_token_ids": prompt,
                               "response_token_ids": resp}
                got += 1
        scanned.append((s, got))
        print("  step %-3d -> +%d distinct ready uids (total %d)" % (s, got, len(chosen)))

    if len(chosen) < args.n:
        print("[warn] only %d distinct ready problems found (< %d); scanned %d files. "
              "Emitting what exists rather than padding with duplicates."
              % (len(chosen), args.n, len(scanned)))

    n_clamped = 0
    rows = []
    for qid in sorted(chosen):                    # sort so --seed alone fixes the sample
        rec = chosen[qid]
        resp = rec["response_token_ids"]
        L = len(resp)
        lo = max(1, int(args.min_frac * L))
        hi = max(lo + 1, int(args.max_frac * L))
        cut = rng.randrange(lo, hi)
        cap_cut = args.resp_cap - args.min_budget
        if cut > cap_cut:                          # keep a Q value reachable
            cut = max(1, cap_cut)
            n_clamped += 1
        rows.append({
            "qid": qid,
            "row_index": rec["row_index"],
            "src_uid": rec["src_uid"],
            "src_step": rec["src_step"],
            "src_resp_len": L,
            "prefix_len": cut,
            "prompt_token_ids": rec["prompt_token_ids"],
            "prefix_token_ids": resp[:cut],
            # what the continuation may still spend under the run's own cap
            "max_new_tokens": max(0, args.resp_cap - cut),
        })

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    pl = sorted(r["prefix_len"] for r in rows)
    q = lambda p: pl[min(len(pl) - 1, int(p * len(pl)))]
    print("\nwrote %s: %d groups" % (args.out, len(rows)))
    print("prefix_len  min %d  p25 %d  median %d  p75 %d  max %d"
          % (pl[0], q(.25), q(.50), q(.75), pl[-1]))
    print("clamped to keep a Q value reachable: %d of %d (%.1f%%)"
          % (n_clamped, len(rows), 100.0 * n_clamped / max(1, len(rows))))
    print("every group can in principle reach budget-g=%d (min remaining budget %d)"
          % (args.budget_g, min(r["max_new_tokens"] for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
