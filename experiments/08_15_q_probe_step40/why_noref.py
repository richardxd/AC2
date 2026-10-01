#!/usr/bin/env python3
"""Why do READY problems get no-ref Q calls?

Readiness is gated on a problem's own success rate clearing a threshold, so a ready problem
has demonstrably been solved. Yet 32-48% of consumed-site Q calls run the NO-REF prompt. That
looks contradictory, so this checks the mechanism instead of asserting it.

THE HYPOTHESIS UNDER TEST. tier 1 does not ask "has this PROBLEM been solved". It asks whether
the specific buffer entry THIS PREFIX WAS CUT FROM carries an extractable, judge-passed proof.
The replay buffer admits under `sp_replay_admission=ungated`, i.e. every non-empty response
including failed ones, and `sp_q_ref_require_pass` (default ON) then refuses the failed ones as
references. Tier 2 -- the reference bank, which is keyed by PROBLEM and would bridge the gap --
is empty in this cold run. So a prefix cut from a failed attempt at a solved problem yields
no reference, and readiness cannot prevent that.

Predictions, each checked below:
  1. a large share of LIVE buffer entries are judge-failed, or pass but carry no <proof>;
  2. the share that do carry a usable proof should track the observed ref rate;
  3. the qids receiving no-ref Q calls should nonetheless be ready, and mostly HAVE a passing
     proof elsewhere in the run's history -- which is precisely what tier 2 would have found.

    python why_noref.py --run-dir <run_data> --model <hf> --step 40
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.expandvars("${AC2_CLUSTER_A_ROOT}/self-play/"
                                         "experiments/08_13_tiedq_seed192/run_data"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=40, help="global step (rollouts/<step>)")
    ap.add_argument("--upto", type=int, default=38, help="buffer state offset")
    ap.add_argument("--data-dir", default=os.path.join(os.environ.get("HOME", ""),
                                                       "data/fineproofs"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from verl.trainer.ppo.sp_q_readiness import _extract_proof_text
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ---------- 1. what is actually IN the live buffer ----------
    entries = {}
    with open(os.path.join(args.run_dir, "replay_buffer_deltas.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if int(d.get("dataset_step", 0)) > args.upto:
                break
            for eid in d.get("replaced_entry_ids") or []:
                entries.pop(eid, None)
            for e in d.get("added_entries") or []:
                entries[e["entry_id"]] = e

    n = len(entries)
    n_fail = n_pass_noproof = n_pass_proof = 0
    for e in entries.values():
        meta = e.get("meta") or {}
        if meta.get("judge_pass", 1) == 0:
            n_fail += 1
            continue
        text = tok.decode(e.get("response_token_ids") or [], skip_special_tokens=False)
        if _extract_proof_text(text):
            n_pass_proof += 1
        else:
            n_pass_noproof += 1

    print("=" * 74)
    print("1. LIVE REPLAY BUFFER at dataset step %d  (n=%d entries)" % (args.upto, n))
    print("=" * 74)
    pc = lambda k: 100.0 * k / max(1, n)
    print("  judge-FAILED (ungated admission)      %5d  %5.1f%%   -> tier 1 refuses" % (n_fail, pc(n_fail)))
    print("  passed but NO extractable <proof>     %5d  %5.1f%%   -> tier 1 finds nothing"
          % (n_pass_noproof, pc(n_pass_noproof)))
    print("  passed WITH a usable proof            %5d  %5.1f%%   -> tier 1 supplies a ref"
          % (n_pass_proof, pc(n_pass_proof)))
    print("\n  => predicted ref rate if prefixes are drawn uniformly: %.1f%%" % pc(n_pass_proof))

    # ---------- 2. the observed ref rate at this step ----------
    wpath = os.path.join(args.run_dir, "q_wave", "%d.jsonl" % (args.step - 1))
    obs = Counter()
    noref_qids, ref_qids = set(), set()
    with open(wpath, encoding="utf-8") as fh:
        for line in fh:
            w = json.loads(line)
            if w.get("kind") != "consumed":
                continue
            v = w.get("variant")
            obs[v] += 1
            (ref_qids if v == "ref" else noref_qids).add(w.get("qid"))
    tot = sum(obs.values())
    print("\n" + "=" * 74)
    print("2. OBSERVED consumed-site Q calls at step %d  (n=%d)" % (args.step, tot))
    print("=" * 74)
    for k, c in obs.most_common():
        print("  %-8s %5d  %5.1f%%" % (k, c, 100.0 * c / max(1, tot)))

    # ---------- 3. are the no-ref problems ready, and solved elsewhere? ----------
    ck = os.path.join(args.run_dir, "checkpoints", "global_step_%d" % args.step, "q_state.json")
    ready = {k for k, v in json.load(open(ck, encoding="utf-8"))["ready"].items() if v}

    import pandas as pd
    from verl.trainer.ppo.difficulty import qid_from_messages
    df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    row_qids = [qid_from_messages(df["prompt"].iloc[i]) for i in range(len(df))]

    # does a passing proof exist ANYWHERE in this run's rollouts for these qids?
    want = noref_qids | ref_qids
    solved = set()
    import glob
    for path in sorted(glob.glob(os.path.join(args.run_dir, "rollouts", "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = r.get("extra_index")
                if idx is None:
                    continue
                q = row_qids[int(idx)]
                if q not in want or q in solved:
                    continue
                pts = r.get("rubric_points")
                if bool(r.get("acc")) or (pts is not None and float(pts) >= 6.0):
                    text = tok.decode(r.get("response_token_ids") or [],
                                      skip_special_tokens=False)
                    if _extract_proof_text(text):
                        solved.add(q)

    nr_ready = sum(1 for q in noref_qids if q in ready)
    nr_solved = sum(1 for q in noref_qids if q in solved)
    print("\n" + "=" * 74)
    print("3. THE NO-REF PROBLEMS THEMSELVES  (%d distinct qids)" % len(noref_qids))
    print("=" * 74)
    print("  ready at step %d                       %5d  %5.1f%%"
          % (args.step, nr_ready, 100.0 * nr_ready / max(1, len(noref_qids))))
    print("  HAVE a passed proof somewhere in run   %5d  %5.1f%%"
          % (nr_solved, 100.0 * nr_solved / max(1, len(noref_qids))))
    print("\n  Every one of those proofs is a reference tier 2 would have supplied.")
    print("  The bank is empty (cold start), so tier 1 is the only source and it asks a")
    print("  strictly narrower question: did THIS entry pass and carry a <proof>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
