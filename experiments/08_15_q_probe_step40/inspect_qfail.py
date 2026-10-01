#!/usr/bin/env python3
"""Look at the actual rollouts behind the worst cell of the z_Q vs z_true plot.

TARGET: groups whose z_true is exactly 0 -- every one of the 16 rollouts was judged wrong --
but whose z_Q lands in [0.2, 0.4], i.e. the trainer was told the group was worth something.
Inside those groups, sample rollouts where Q was HIGH and the judge said 0.

WHAT WE CAN AND CANNOT SHOW. Q was called at prefix+10k, so it only ever saw the prompt, the
prefix, and the first `--budget-g` tokens of the continuation. The judge saw the FULL rollout.
So "was the rollout already wrong when Q scored it?" is not a label we hold -- nothing judged
the truncated text. What this dumps instead is the evidence to decide by eye: the tail of
exactly what Q saw at decision time, the tail of how the rollout actually ended, and whether a
<proof> block had appeared by the cut (proof-before-cut would mean the run would NOT have
Q-scored this rollout at all, so its absence is worth confirming).

    python inspect_qfail.py --n 5 --q-min 0.5 --out qfail_report.txt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def has_proof(text):
    idx = text.rfind("</think>")
    post = text[idx + len("</think>"):] if idx >= 0 else text
    return re.search(r"<proof>(.*?)</proof>", post, re.DOTALL) is not None


def load(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--gen-dir", default=os.path.join(here, "gen"))
    ap.add_argument("--q-dir", default=os.path.join(here, "q"))
    ap.add_argument("--judged-dir", default=os.path.join(here, "judged"))
    ap.add_argument("--model", default=os.path.join(here, "model_hf"))
    ap.add_argument("--data-dir", default=os.path.join(os.environ.get("HOME", ""),
                                                       "data/fineproofs"))
    ap.add_argument("--budget-g", type=int, default=10000)
    ap.add_argument("--zq-lo", type=float, default=0.2)
    ap.add_argument("--zq-hi", type=float, default=0.4)
    ap.add_argument("--q-min", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=192)
    ap.add_argument("--tail-seen", type=int, default=2200, help="chars of what Q saw")
    ap.add_argument("--tail-end", type=int, default=1400, help="chars of how it ended")
    ap.add_argument("--out", default=os.path.join(here, "qfail_report.txt"))
    args = ap.parse_args()

    groups = load(os.path.join(args.gen_dir, "gen.shard*.jsonl"))
    qmap = {(r["qid"], r["completion_index"]): r
            for r in load(os.path.join(args.q_dir, "q.shard*.jsonl"))}
    jmap = {(r["qid"], r["completion_index"]): r
            for r in load(os.path.join(args.judged_dir, "judged.shard*.jsonl"))}

    cands = []
    n_groups_hit = 0
    for g in groups:
        qid = g["qid"]
        zs, zt = [], []
        for j, c in enumerate(g["completions"]):
            jr = jmap.get((qid, j))
            r = None if jr is None else jr.get("score")
            if r is None:
                continue
            zt.append(float(r))
            if c["exceeds_g"]:
                q = (qmap.get((qid, j)) or {}).get("q")
                if q is None:
                    continue
                zs.append(float(q))
            else:
                zs.append(float(r))
        if len(zt) < 8 or len(zs) < 8:
            continue
        ztm, zsm = mean(zt), mean(zs)
        if ztm > 1e-9 or not (args.zq_lo <= zsm <= args.zq_hi):
            continue
        n_groups_hit += 1
        for j, c in enumerate(g["completions"]):
            if not c["exceeds_g"]:
                continue
            qr = qmap.get((qid, j)); jr = jmap.get((qid, j))
            if not qr or not jr:
                continue
            q, r = qr.get("q"), jr.get("score")
            if q is None or r is None or q < args.q_min or r > 1e-9:
                continue
            cands.append((g, j, float(q), float(r), zsm, qr))

    print("groups with z_true==0 and z_Q in [%.2f, %.2f]: %d" % (args.zq_lo, args.zq_hi, n_groups_hit))
    print("rollouts in them with Q >= %.2f and r == 0: %d" % (args.q_min, len(cands)))
    if not cands:
        return 1
    rng = random.Random(args.seed)
    pick = rng.sample(cands, min(args.n, len(cands)))

    from transformers import AutoTokenizer
    import pandas as pd
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    ei_col = df["extra_info"] if "extra_info" in df.columns else None

    with open(args.out, "w", encoding="utf-8") as fh:
        w = lambda s: fh.write(s + "\n")
        w("Rollouts where the WHOLE group scored 0 but the trainer saw z_Q in [%.2f, %.2f]"
          % (args.zq_lo, args.zq_hi))
        w("groups matching: %d   candidate rollouts: %d   sampled: %d (seed %d)"
          % (n_groups_hit, len(cands), len(pick), args.seed))
        w("")
        for k, (g, j, q, r, zsm, qr) in enumerate(pick, 1):
            raw = {} if ei_col is None else (ei_col.iloc[int(g["row_index"])] or {})
            if hasattr(raw, "item"):
                raw = raw.item()
            raw = dict(raw) if not isinstance(raw, dict) else raw
            problem = (raw.get("theorem") or raw.get("question") or "")[:1200]
            c = g["completions"][j]
            seen_ids = list(c["cont_token_ids"])[:args.budget_g]
            seen = tok.decode(seen_ids, skip_special_tokens=False)
            full = c["text"]

            w("=" * 96)
            w("[%d] qid %s  completion %d" % (k, g["qid"][:16], j))
            w("    Q at prefix+%d = %.1f     judge on the FULL rollout = %.4f     group z_Q = %.3f"
              % (args.budget_g, q, r, zsm))
            w("    prefix_len %d   continuation %d tokens   used_ref=%s"
              % (g["prefix_len"], c["n_tokens"], qr.get("used_ref")))
            w("    <proof> present by the cut: %s      in the full rollout: %s"
              % (has_proof(seen), has_proof(full)))
            w("    finish_reason=%s" % c.get("finish_reason"))
            w("")
            w("--- PROBLEM ------------------------------------------------------------------")
            w(problem)
            w("")
            w("--- TAIL OF WHAT Q SAW (last %d chars at the cut) -----------------------------"
              % args.tail_seen)
            w(seen[-args.tail_seen:])
            w("")
            w("--- TAIL OF THE FULL ROLLOUT (last %d chars) ----------------------------------"
              % args.tail_end)
            w(full[-args.tail_end:])
            w("")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
