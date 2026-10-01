#!/usr/bin/env python3
"""THE RESULT: what does the short lane's Q-for-reward substitution actually cost?

Joins the three stages -- continuations (A), Q at prefix+10k (B), judge rewards (C) -- and
answers the question the live run structurally cannot: for rollouts that reached the cut, how
close is Q's answer to the reward those rollouts would really have earned?

TWO POPULATIONS, AND WHY BOTH ARE REPORTED.

  MATCHED (the headline). Only rollouts that exceeded g=10000 have a Q value at all, and
  those are exactly the rollouts the short lane would have cut and scored with Q. Comparing
  mean_i(Q_i) against mean_i(r_i) over that same set is the substitution error itself, with
  nothing else mixed in. This is the number to quote.

  ALL-16 (context only). The group's full mean reward, including rollouts that finished
  before the cut and would have kept their judge score. It answers a different question --
  "does Q at the cut predict the whole group" -- and is NOT the substitution error, because
  the two sides are then computed over different rollouts. Reported so the difference between
  the two framings is visible rather than hidden by a choice of denominator.

A SELECTION EFFECT THAT LIMITS THE ALL-16 ROW, stated plainly: reaching 10k is not random.
A rollout still running at the cut is one that did not finish early, and finishing early
correlates with succeeding outright. So the matched population is not a random sample of the
group, and the two columns should not be read as interchangeable estimates.

    python analyze_probe.py --gen-dir gen --q-dir q --judged-dir judged
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def pearson(ps):
    ps = [(a, b) for a, b in ps if a is not None and b is not None]
    if len(ps) < 3:
        return None
    mx = sum(p[0] for p in ps) / len(ps)
    my = sum(p[1] for p in ps) / len(ps)
    num = sum((a - mx) * (b - my) for a, b in ps)
    dx = sum((a - mx) ** 2 for a, b in ps) ** 0.5
    dy = sum((b - my) ** 2 for a, b in ps) ** 0.5
    return (num / (dx * dy)) if dx > 0 and dy > 0 else None


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


def f(x, n=4):
    return "  -   " if x is None else ("%.*f" % (n, x))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--gen-dir", default=os.path.join(here, "gen"))
    ap.add_argument("--q-dir", default=os.path.join(here, "q"))
    ap.add_argument("--judged-dir", default=os.path.join(here, "judged"))
    ap.add_argument("--run-dir", default=os.path.expandvars("${AC2_CLUSTER_A_ROOT}/self-play/"
                                         "experiments/08_13_tiedq_seed192/run_data"),
                    help="for q_state_deltas.jsonl, to rebuild the reference bank membership")
    ap.add_argument("--dump-pairs", action="store_true",
                    help="emit raw (z_true, z_Q, q_share) triples for plotting")
    ap.add_argument("--min-matched", type=int, default=2,
                    help="a group mean resting on a single rollout is a point of pure noise")
    args = ap.parse_args()

    groups = load(os.path.join(args.gen_dir, "gen.shard*.jsonl"))
    qrows = load(os.path.join(args.q_dir, "q.shard*.jsonl"))
    jrows = load(os.path.join(args.judged_dir, "judged.shard*.jsonl"))
    print("groups %d   Q calls %d   judged rows %d" % (len(groups), len(qrows), len(jrows)))

    qmap = {(r["qid"], r["completion_index"]): r for r in qrows}
    jmap = {(r["qid"], r["completion_index"]): r for r in jrows}

    n_q_valid = sum(1 for r in qrows if r.get("q") is not None)
    n_ref = sum(1 for r in qrows if r.get("used_ref"))
    print("Q parsed to grid: %d/%d (%.1f%% invalid)   with-ref share: %.1f%%"
          % (n_q_valid, len(qrows), 100.0 * (len(qrows) - n_q_valid) / max(1, len(qrows)),
             100.0 * n_ref / max(1, len(qrows))))

    per_rollout = []          # (r_i, Q_i) matched rollouts
    grp_matched, grp_all = [], []
    n_exceed = n_tot = 0
    for g in groups:
        qid = g["qid"]
        qs, rs_matched, rs_all = [], [], []
        for j, c in enumerate(g["completions"]):
            n_tot += 1
            n_exceed += int(c["exceeds_g"])
            jr = jmap.get((qid, j))
            r = None if jr is None else jr.get("score")
            if r is not None:
                rs_all.append(float(r))
            if not c["exceeds_g"]:
                continue
            qr = qmap.get((qid, j))
            q = None if qr is None else qr.get("q")
            if q is None or r is None:
                continue
            qs.append(float(q))
            rs_matched.append(float(r))
            per_rollout.append((float(r), float(q)))
        if len(qs) >= args.min_matched:
            grp_matched.append((mean(rs_matched), mean(qs), len(qs)))
            if rs_all:
                grp_all.append((mean(rs_all), mean(qs), len(rs_all)))

    print("rollouts %d, exceeded g: %d (%.1f%%)" % (n_tot, n_exceed, 100.0 * n_exceed / max(1, n_tot)))

    print("\n" + "=" * 78)
    print("HEADLINE -- matched: mean_i(Q_i) vs mean_i(r_i) over rollouts that reached the cut")
    print("=" * 78)
    if grp_matched:
        mq = mean([g[1] for g in grp_matched]); mr = mean([g[0] for g in grp_matched])
        print("groups %d   mean Q %s   mean r %s   BIAS %s   corr %s   MAE %s"
              % (len(grp_matched), f(mq), f(mr), "%+.4f" % (mq - mr),
                 f(pearson([(g[0], g[1]) for g in grp_matched])),
                 f(mean([abs(g[1] - g[0]) for g in grp_matched]))))
        cells = defaultdict(int)
        for r, q, _ in grp_matched:
            cells[(min(10, int(round(q * 10))), min(10, int(round(r * 10))))] += 1
        print("cells: " + ";".join("%d,%d,%d" % (k[0], k[1], v) for k, v in sorted(cells.items())))

    print("\nrollout level (same rollout, Q_i vs r_i)")
    if per_rollout:
        mq = mean([p[1] for p in per_rollout]); mr = mean([p[0] for p in per_rollout])
        print("  n %d   mean Q %s   mean r %s   BIAS %s   corr %s   MAE %s"
              % (len(per_rollout), f(mq), f(mr), "%+.4f" % (mq - mr),
                 f(pearson(per_rollout)), f(mean([abs(p[1] - p[0]) for p in per_rollout]))))

    print("\ncontext only -- mean_i(Q_i) vs the group's ALL-16 mean reward")
    print("(different denominators; see the selection-effect note in the docstring)")
    if grp_all:
        mq = mean([g[1] for g in grp_all]); mr = mean([g[0] for g in grp_all])
        print("  groups %d   mean Q %s   mean r %s   BIAS %s   corr %s   MAE %s"
              % (len(grp_all), f(mq), f(mr), "%+.4f" % (mq - mr),
                 f(pearson([(g[0], g[1]) for g in grp_all])),
                 f(mean([abs(g[1] - g[0]) for g in grp_all]))))

    # ---- Q AT THE SHARED PREFIX (site 1, completion_index = -1) -------------------------
    # The run's `probe` site. Three pairings, each answering a different question:
    #   vs z_true          -- the statistic an earlier audit-lane analysis reported (corr
    #                         0.839), so the two are comparable on the SAME quantity;
    #   vs mean_i(Q_i)     -- DEGRADATION between the two sites on the same group. If Q reads a
    #                         fresh prefix well but misjudges its own 10k continuation, it shows
    #                         up here and nowhere else;
    #   vs mean_i(r_i)     -- prefix-time prediction of the cut population's realised reward.
    pfx = {r["qid"]: r for r in qrows if r.get("completion_index") == -1}
    print("\n" + "=" * 78)
    print("Q_prefix (site 1): %d groups have a prefix-site Q" % len(pfx))
    print("=" * 78)
    rows_p = []
    for g in groups:
        pr = pfx.get(g["qid"])
        if pr is None or pr.get("q") is None:
            continue
        zt, qs = [], []
        for j, c in enumerate(g["completions"]):
            jr = jmap.get((g["qid"], j))
            r = None if jr is None else jr.get("score")
            if r is not None:
                zt.append(float(r))
            if c["exceeds_g"]:
                qv = (qmap.get((g["qid"], j)) or {}).get("q")
                if qv is not None:
                    qs.append(float(qv))
        if len(zt) >= 8:
            rows_p.append((float(pr["q"]), mean(zt), mean(qs) if qs else None))
    if rows_p:
        mp = mean([r[0] for r in rows_p])
        print("  mean Q_prefix %s   (n=%d groups)" % (f(mp), len(rows_p)))
        for lbl, idx in (("vs z_true (all-16 judge)", 1), ("vs mean_i(Q_i) at the cut", 2)):
            ps = [(r[idx], r[0]) for r in rows_p if r[idx] is not None]
            if len(ps) < 3:
                continue
            mo = mean([p[0] for p in ps])
            print("  %-28s n %4d  mean other %s  BIAS %s  corr %s  MAE %s"
                  % (lbl, len(ps), f(mo), "%+.4f" % (mean([p[1] for p in ps]) - mo),
                     f(pearson(ps)), f(mean([abs(p[1] - p[0]) for p in ps]))))

    # ---- THE NUMBER THAT ACTUALLY MATTERS: the group signal the short lane produces ----
    # The lane does NOT replace every reward with Q. A rollout that finishes before the cut is
    # judged normally; only one still running at prefix+10k has its reward replaced. So the
    # advantage signal a short-lane group carries is a HYBRID:
    #     z_Q = mean_i [ r_i if the rollout finished early else Q_i ]
    # and the truth -- observable only because this probe removed the cut -- is
    #     z_true  = mean_i [ r_i ]  over all 16.
    # Comparing those two IS the distortion the trainer actually sees. It is smaller than the
    # matched-population bias above, because roughly half of each group finishes early and
    # keeps its real judge score, diluting Q's contribution.
    #
    # CAVEAT, stated because it moves the number the wrong way: `exceeds_g` here is purely a
    # LENGTH test. The run additionally exempts a rollout that already emitted an extractable
    # <proof> before the cut (those are judged, not Q-scored). Emitting a proof normally ends
    # generation, so the two criteria mostly coincide -- but where they differ, this treats a
    # rollout as Q-scored that the run would have judged, slightly OVERSTATING the distortion.
    print("\n" + "=" * 78)
    print("SHORT-LANE GROUP SIGNAL: z_Q (hybrid r/Q) vs z_true (all-16 judge)")
    print("=" * 78)
    hyb = []
    for g in groups:
        zs, zt, n_sub = [], [], 0
        for j, c in enumerate(g["completions"]):
            jr = jmap.get((g["qid"], j))
            r = None if jr is None else jr.get("score")
            if r is None:
                continue
            zt.append(float(r))
            if c["exceeds_g"]:
                qr = qmap.get((g["qid"], j))
                q = None if qr is None else qr.get("q")
                if q is None:
                    continue
                zs.append(float(q)); n_sub += 1
            else:
                zs.append(float(r))
        if len(zs) >= 8 and len(zt) >= 8:
            hyb.append((mean(zt), mean(zs), n_sub / max(1, len(zs)), g["qid"]))
    if hyb:
        ms = mean([h[1] for h in hyb]); mt = mean([h[0] for h in hyb])
        print("groups %d   z_Q %s   z_true %s   BIAS %s   corr %s   MAE %s"
              % (len(hyb), f(ms), f(mt), "%+.4f" % (ms - mt),
                 f(pearson([(h[0], h[1]) for h in hyb])),
                 f(mean([abs(h[1] - h[0]) for h in hyb]))))
        print("  mean share of each group substituted by Q: %.1f%%"
              % (100.0 * mean([h[2] for h in hyb])))
        base = mean([h[0] for h in hyb])
        print("  constant predictor MAE %s   (z_Q MAE above)"
              % f(mean([abs(base - h[0]) for h in hyb])))
        if args.dump_pairs:
            # raw per-group rows for plotting: z_true, z_Q, Q-substituted share, in_bank.
            # in_bank is the SOLVED flag: membership in the add-once reference bank, which is
            # only ever filled from a judged-PASSING row. It is exactly the criterion the
            # SP_Q_READY_REQUIRE_BANK gate uses, so filtering on it shows what that gate would
            # have removed -- as opposed to filtering on z_true==0, which is only "scored zero
            # in THIS probe" and would also drop solvable problems that happened to fail here.
            bank = set()
            bpath = os.path.join(args.run_dir, "q_state_deltas.jsonl")
            if os.path.exists(bpath):
                with open(bpath, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            for b in (json.loads(line).get("bank_added") or []):
                                bank.add(b["qid"])
                        except json.JSONDecodeError:
                            continue
            print("BANKSIZE %d" % len(bank))
            # z_true, z_Q, Q-substituted share, in_bank, Q_prefix (-1 if absent)
            print("PAIRS " + ";".join(
                "%.4f,%.4f,%.3f,%d,%.4f"
                % (h[0], h[1], h[2], int(h[3] in bank),
                   (pfx.get(h[3]) or {}).get("q", -1.0)
                   if (pfx.get(h[3]) or {}).get("q") is not None else -1.0)
                for h in hyb))

    # Does the reference proof change Q's accuracy? Free to ask -- stage B records used_ref
    # per call. Worth asking because this cold run's bank is empty, so the run itself gets a
    # reference only when the SOURCE entry happened to carry an extractable proof; a ready
    # problem solved on some other attempt gets none. If the split is large, the run's
    # 52-68% ref share is itself costing accuracy.
    print("\nrollout level split by whether Q was shown a reference proof")
    byref = {True: [], False: []}
    for g in groups:
        for j, c in enumerate(g["completions"]):
            if not c["exceeds_g"]:
                continue
            qr = qmap.get((g["qid"], j)); jr = jmap.get((g["qid"], j))
            if qr is None or jr is None:
                continue
            q, r = qr.get("q"), jr.get("score")
            if q is None or r is None:
                continue
            byref[bool(qr.get("used_ref"))].append((float(r), float(q)))
    for k, lbl in ((True, "with ref"), (False, "no ref  ")):
        v = byref[k]
        if not v:
            print("  %s  n=0" % lbl); continue
        mq = mean([p[1] for p in v]); mr = mean([p[0] for p in v])
        print("  %s  n %5d   mean Q %s   mean r %s   BIAS %s   corr %s   MAE %s"
              % (lbl, len(v), f(mq), f(mr), "%+.4f" % (mq - mr), f(pearson(v)),
                 f(mean([abs(p[1] - p[0]) for p in v]))))

    # what a constant predictor would score -- the floor any useful Q must beat
    if grp_matched:
        base = mean([g[0] for g in grp_matched])
        print("\nbaselines on the matched groups")
        print("  constant predictor (mean r)      MAE %s"
              % f(mean([abs(base - g[0]) for g in grp_matched])))
        print("  Q                                MAE %s"
              % f(mean([abs(g[1] - g[0]) for g in grp_matched])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
