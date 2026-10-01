"""The Q_prefix site: an INDEPENDENT read on the group, and how it relates to the cut site.

Two panels, both at step 57:

  [1] Q_prefix vs z_true -- the clean accuracy number. Unlike z_Q, Q_prefix shares NO term
      with z_true: it is one prediction made before any of the 16 rollouts existed, compared
      against the mean judge reward those rollouts went on to earn. The z_Q correlation (0.974)
      is inflated because roughly half of each group's z_Q IS its own r_i; this one is not.
      It is also the statistic an earlier audit-lane analysis reported (corr 0.839), so the
      two are comparable on the same quantity.

  [2] Q_prefix vs z_Q -- do the two Q call sites agree? They are the SAME weights answering
      the same question 10k tokens apart, so disagreement is Q changing its mind as evidence
      accumulates, not two different models. Points below the diagonal are groups Q became
      more pessimistic about after seeing the continuations.

Colour is the fraction of the group whose reward was replaced by Q, kept from the other plots
so the panels can be read side by side.

    python plot_prefix.py --csv zpairs5.csv --out prefix_panels.png
"""

from __future__ import annotations

import argparse
import os


def stats(x, y):
    n = len(x)
    if n < 3:
        return n, float("nan"), float("nan"), float("nan")
    mx = sum(x) / n; my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    c = num / (dx * dy) if dx and dy else float("nan")
    return n, c, my - mx, sum(abs(b - a) for a, b in zip(x, y)) / n


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", default="57")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zt, zq, w, qp = [], [], [], []
    with open(args.csv, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            if len(p) < 5 or float(p[4]) < 0:
                continue
            zt.append(float(p[0])); zq.append(float(p[1]))
            w.append(float(p[2])); qp.append(float(p[4]))

    fig, axs = plt.subplots(1, 2, figsize=(13.6, 6.0))
    for ax, x, y, xlab, ylab, title in (
            (axs[0], zt, qp, "z_true  —  mean judge reward over all 16",
             "Q_prefix  —  Q at the shared prefix",
             "step %s   Q_prefix vs z_true\n(no shared term — the clean accuracy)" % args.step),
            (axs[1], zq, qp, "z_Q  —  signal the trainer sees",
             "Q_prefix  —  Q at the shared prefix",
             "step %s   Q_prefix vs z_Q\n(same weights, 10k tokens apart)" % args.step)):
        n, c, bias, mae = stats(x, y)
        sc = ax.scatter(x, y, c=w, cmap="viridis", s=48, alpha=.85,
                        edgecolors="white", linewidths=.5, vmin=0, vmax=1)
        ax.plot([0, 1], [0, 1], ls="--", lw=1.6, color="#666", zorder=0, label="y = x")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title("%s\nn=%d   corr %.3f   bias %+.4f   MAE %.4f" % (title, n, c, bias, mae),
                     fontsize=11.5)
        ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03)
        ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=9)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("fraction of group replaced by Q", fontsize=9)

    fig.suptitle("08_13_tiedq_seed192 step %s — the prefix-site Q, which shares no term with "
                 "the truth" % args.step, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote %s\n" % args.out)

    for lbl, x, y in (("Q_prefix vs z_true", zt, qp), ("Q_prefix vs z_Q", zq, qp)):
        n, c, bias, mae = stats(x, y)
        print("%-24s n %4d  corr %6.3f  bias %+7.4f  MAE %6.4f" % (lbl, n, c, bias, mae))

    # Q_prefix is a grid value; how much of the range does it actually use?
    from collections import Counter
    cnt = Counter(round(v, 1) for v in qp)
    print("\nQ_prefix value distribution (grid):")
    for k in sorted(cnt):
        print("  %.1f  %4d  %5.1f%%  %s" % (k, cnt[k], 100.0 * cnt[k] / len(qp),
                                            "#" * int(round(50.0 * cnt[k] / len(qp)))))
    mid = sum(v for k, v in cnt.items() if 0.2 <= k <= 0.8)
    print("  mass in [0.2, 0.8]: %d of %d = %.1f%%" % (mid, len(qp), 100.0 * mid / len(qp)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
