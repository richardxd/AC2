#!/usr/bin/env python3
"""Q probe across checkpoints of 08_13_tiedq_seed192: steps 40, 57, 80, 160 side by side.

Two rows, one column per probed step:
  row 1  z_Q vs z_true      -- the signal the short lane feeds the trainer against the all-16
                               judge mean. Coloured by the share of the group replaced by Q; a
                               group at 0.0 sits on the diagonal by construction.
  row 2  Q_prefix vs z_true -- ONE Q call at the shared prefix against the group's eventual
                               mean reward. No shared term, so this is the clean accuracy.
                               (Absent at step 40: that probe had no prefix site.)

Input CSVs are the probe pairs dumps: z_true,z_Q,share_substituted,in_bank[,Q_prefix] per group
(4 columns for step 40, 5 for the later probes; see analyze_probe.py --dump-pairs).

    python plot_compare4.py --csv 40=path 57=path 80=path 160=path --out compare4.png
"""
from __future__ import annotations

import argparse
import math


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        p = line.split(",")
        qp = float(p[4]) if len(p) > 4 else None
        if qp is not None and qp < 0:      # analyze_probe writes -1.0 when the prefix site is absent
            qp = None
        rows.append((float(p[0]), float(p[1]), float(p[2]), qp))
    return rows


def stats(x, y):
    n = len(x)
    if n < 3:
        return n, float("nan"), float("nan"), float("nan")
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x)); dy = math.sqrt(sum((b - my) ** 2 for b in y))
    corr = num / (dx * dy) if dx and dy else float("nan")
    return n, corr, my - mx, sum(abs(b - a) for a, b in zip(x, y)) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="step=path entries")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = []
    for e in a.csv:
        step, path = e.split("=", 1)
        series.append((int(step), load(path)))
    series.sort()
    k = len(series)
    fig, axs = plt.subplots(2, k, figsize=(4.3 * k, 8.6), squeeze=False)
    for c, (step, rows) in enumerate(series):
        zt = [r[0] for r in rows]; zq = [r[1] for r in rows]; sh = [r[2] for r in rows]
        ax = axs[0][c]
        sc = ax.scatter(zt, zq, c=sh, cmap="viridis", vmin=0, vmax=1, s=14, alpha=.8)
        ax.plot([0, 1], [0, 1], "k--", lw=.8)
        n, corr, bias, mae = stats(zt, zq)
        ax.set_title("step %d   z_Q vs z_true\nn=%d  corr %.3f  MAE %.3f  bias %+.3f" % (step, n, corr, mae, bias), fontsize=10)
        ax.set_xlabel("z_true (all-16 judge mean)"); ax.set_ylabel("z_Q (trainer's signal)")
        ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02); ax.grid(alpha=.3)
        if c == k - 1:
            fig.colorbar(sc, ax=ax, fraction=.046, label="share of group replaced by Q")
        ax = axs[1][c]
        qp = [(r[0], r[3]) for r in rows if r[3] is not None]
        if qp:
            x = [p[0] for p in qp]; y = [p[1] for p in qp]
            ax.scatter(x, y, s=14, alpha=.7, color="#d62728")
            ax.plot([0, 1], [0, 1], "k--", lw=.8)
            n, corr, bias, mae = stats(x, y)
            ax.set_title("step %d   Q_prefix vs z_true\nn=%d  corr %.3f  MAE %.3f  bias %+.3f" % (step, n, corr, mae, bias), fontsize=10)
        else:
            ax.text(.5, .5, "no prefix-site Q\nin this probe", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("step %d   Q_prefix vs z_true" % step, fontsize=10)
        ax.set_xlabel("z_true (all-16 judge mean)"); ax.set_ylabel("Q_prefix (one call at the prefix)")
        ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02); ax.grid(alpha=.3)
    fig.suptitle("Q probe on 08_13_tiedq_seed192 checkpoints: 256 ready prefixes x 16 full-budget rollouts, judge-scored\n"
                 "top: the hybrid signal the trainer consumes;  bottom: a single Q call at the prefix (no shared term)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print("wrote", a.out)
    for step, rows in series:
        zt = [r[0] for r in rows]; zq = [r[1] for r in rows]
        n, corr, bias, mae = stats(zt, zq)
        qp = [(r[0], r[3]) for r in rows if r[3] is not None]
        n2, corr2, bias2, mae2 = stats([p[0] for p in qp], [p[1] for p in qp]) if qp else (0, float("nan"), float("nan"), float("nan"))
        print("step %3d  z_Q vs z_true: n %3d corr %.3f MAE %.3f bias %+.3f | Q_prefix vs z_true: n %3d corr %.3f MAE %.3f bias %+.3f"
              % (step, n, corr, mae, bias, n2, corr2, mae2, bias2))


if __name__ == "__main__":
    main()
