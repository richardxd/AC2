#!/usr/bin/env python3
"""STAGE C: score every generated rollout with the run's own DS4 judge.

Supplies r_i. Stage A produced the continuations, stage B asked Q what it would have said at
prefix+10k; this closes the pair so mean_i(Q_i) can be compared against mean_i(r_i).

THE JUDGE CLIENT IS IMPORTED, NOT REBUILT. compute_score from
ac2.rewards.ds4_finegrained_judge carries the sha-asserted fine-grained template, the
grade-parse precedence, the retry/semaphore policy and the reward semantics
(score = points/7, pass = points >= 6). Re-implementing any of that would produce numbers
that are not comparable with the run's own rewards, which is the entire point of the probe.

ONE TRAP, EXPLICITLY AVOIDED. compute_score short-circuits when
extra_info["sp_q_route_taken"] == "q_consumed": it then returns the STORED Q value instead of
judging, which is exactly the substitution this probe exists to measure. Feeding through the
rollout's original extra_info would therefore hand back Q where a judge score was wanted and
silently produce a perfect correlation. We build a minimal extra_info carrying only the
problem text, so the judge path always runs.

    python probe_judge.py --gen-dir gen --shard K --num-shards N --judge-url http://... \
        --out judged.shardK.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import time

# MUST be set before ds4_finegrained_judge -> prover_judge is imported: that module builds a
# MODULE-LEVEL asyncio.Semaphore(SP_JUDGE_MAX_INFLIGHT) at import time, defaulting to 16. A
# --concurrency higher than that is silently ignored, and the stage then crawls at ~14
# req/min/node against a server sized for max_num_seqs=256. The training runs use 160
# (SP_JUDGE_MAX_INFLIGHT in their attach scripts). Set it here as well as in the launcher so
# running this script directly is not quietly 10x slower than running it through
# probe_judge_node.sh.
os.environ.setdefault("SP_JUDGE_MAX_INFLIGHT", "160")


async def main_async(args):
    import pandas as pd
    from ac2.rewards.ds4_finegrained_judge import compute_score

    df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    ei_col = df["extra_info"] if "extra_info" in df.columns else None

    groups = []
    for p in sorted(glob.glob(os.path.join(args.gen_dir, "gen.shard*.jsonl"))):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                groups.append(json.loads(line))
    mine = [g for i, g in enumerate(groups) if i % args.num_shards == args.shard]

    tasks = []
    for g in mine:
        raw = {} if ei_col is None else (ei_col.iloc[int(g["row_index"])] or {})
        if hasattr(raw, "item"):
            raw = raw.item()
        raw = dict(raw) if not isinstance(raw, dict) else raw
        problem = raw.get("theorem") or raw.get("question") or ""
        # minimal extra_info ON PURPOSE -- see the q_consumed trap in the docstring
        ei = {"theorem": problem}
        for j, c in enumerate(g["completions"]):
            if args.only_exceeds and not c["exceeds_g"]:
                continue
            tasks.append((g["qid"], j, c["text"], ei))

    print("[judge %d] %d rollouts to score (of %d groups)"
          % (args.shard, len(tasks), len(mine)), flush=True)
    if not tasks:
        open(args.out, "w").close()
        return 0

    sem = asyncio.Semaphore(args.concurrency)
    done = [0]
    t0 = time.time()

    async def one(qid, j, text, ei):
        async with sem:
            try:
                r = await compute_score(
                    data_source=args.data_source, solution_str=text, ground_truth="",
                    extra_info=ei, judge_url=args.judge_url,
                    judge_max_tokens=args.judge_max_tokens, judge_temperature=1.0,
                    judge_top_p=1.0, judge_top_k=-1, judge_reasoning_effort="high",
                )
            except Exception as e:                       # one bad row must not kill the shard
                r = {"score": None, "_error": repr(e)[:200]}
            done[0] += 1
            if done[0] % 50 == 0:
                el = time.time() - t0
                print("[judge %d] %d/%d  %.1f/min" % (args.shard, done[0], len(tasks),
                                                      60.0 * done[0] / max(1e-9, el)), flush=True)
            d = r if isinstance(r, dict) else {"score": r}
            return {"qid": qid, "completion_index": j,
                    "score": d.get("score"), "rubric_points": d.get("rubric_points"),
                    "prover_judge_score": d.get("prover_judge_score"),
                    "judge_parse_failed": d.get("judge_parse_failed"),
                    "error": d.get("_error")}

    out = await asyncio.gather(*[one(*t) for t in tasks])
    n_ok = sum(1 for r in out if r["score"] is not None)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("[judge %d] DONE %d rows, %d scored (%.1f%% failed) in %.1f min"
          % (args.shard, len(out), n_ok, 100.0 * (len(out) - n_ok) / max(1, len(out)),
             (time.time() - t0) / 60.0), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-url", default=os.environ.get("SELF_PLAY_JUDGE_URL"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=192,
                    help="outer cap; the EFFECTIVE limit is min(this, SP_JUDGE_MAX_INFLIGHT)")
    ap.add_argument("--judge-max-tokens", type=int, default=40000)
    ap.add_argument("--data-source", default="fineproofs-rl")
    ap.add_argument("--data-dir", default=os.path.join(os.environ.get("HOME", ""),
                                                       "data/fineproofs"))
    ap.add_argument("--only-exceeds", action="store_true",
                    help="score only rollouts that passed the Q cut (the matched population)")
    args = ap.parse_args()
    if not args.judge_url:
        raise SystemExit("need --judge-url or SELF_PLAY_JUDGE_URL")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
