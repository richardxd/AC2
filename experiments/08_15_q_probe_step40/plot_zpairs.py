"""Scatter the short lane's group signal against the truth it stands in for.

x = z_true = mean_i r_i over all 16 members (observable only because the probe removed the cut)
y = z_Q    = mean_i [ r_i if the rollout finished early else Q_i ]  -- what the trainer sees

COLOUR IS THE POINT. Each group is coloured by the FRACTION of its members whose reward was
replaced by Q. A group at 0.0 has y == x by construction (nothing was substituted, so it sits
exactly on the diagonal and carries no information about Q). A group at 1.0 is pure Q. The
colour separates the tautological part of this plot from the informative part, and is why the
headline correlation must not be read as "Q correlates that well with reward".

THE SECOND PANEL IS THE ACTUAL QUESTION: what does SP_Q_READY_REQUIRE_BANK buy?
That gate admits a problem to readiness only if it is in the add-once reference bank, which is
only ever filled from a judged-PASSING row -- i.e. the prover has solved it at least once. The
right panel keeps exactly those groups. Note this is NOT the same as dropping z_true == 0: a
solved problem can still score 0 across all 16 rollouts of one probe, and those groups SHOULD
stay, because the live gate would keep them too. Filtering on z_true == 0 instead would flatter
the gate by also deleting failures it does not prevent.

    python plot_zpairs.py --csv zpairs.csv --out zpairs.png
"""

from __future__ import annotations

import argparse
import os


def stats(zt, zs):
    n = len(zt)
    if n < 3:
        return n, float("nan"), float("nan"), float("nan")
    mt = sum(zt) / n; ms = sum(zs) / n
    num = sum((a - mt) * (b - ms) for a, b in zip(zt, zs))
    dx = sum((a - mt) ** 2 for a in zt) ** 0.5
    dy = sum((b - ms) ** 2 for b in zs) ** 0.5
    corr = num / (dx * dy) if dx and dy else float("nan")
    return n, corr, ms - mt, sum(abs(b - a) for a, b in zip(zt, zs)) / n


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=os.path.join(here, "zpairs.csv"))
    ap.add_argument("--out", default=os.path.join(here, "zpairs.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zt, zs, w, bank = [], [], [], []
    with open(args.csv, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            zt.append(float(parts[0])); zs.append(float(parts[1])); w.append(float(parts[2]))
            bank.append(int(parts[3]) if len(parts) > 3 else 1)

    keep = [i for i in range(len(zt)) if bank[i]]
    drop = [i for i in range(len(zt)) if not bank[i]]
    sub = lambda xs, idx: [xs[i] for i in idx]

    fig, axs = plt.subplots(1, 2, figsize=(13.4, 6.0))
    for ax, idx, title in (
            (axs[0], list(range(len(zt))), "ALL ready groups (current behaviour)"),
            (axs[1], keep, "SOLVED only — what SP_Q_READY_REQUIRE_BANK keeps")):
        n, corr, bias, mae = stats(sub(zt, idx), sub(zs, idx))
        sc = ax.scatter(sub(zt, idx), sub(zs, idx), c=sub(w, idx), cmap="viridis",
                        s=46, alpha=.85, edgecolors="white", linewidths=.5, vmin=0, vmax=1)
        ax.plot([0, 1], [0, 1], ls="--", lw=1.6, color="#666", zorder=0, label="perfect (y = x)")
        ax.set_xlabel("z_true  —  mean judge reward over all 16")
        ax.set_ylabel("z_Q  —  signal the trainer sees")
        ax.set_title("%s\nn=%d   corr %.3f   bias %+.4f   MAE %.4f" % (title, n, corr, bias, mae),
                     fontsize=11.5)
        ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03)
        ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=9)
        cb = fig.colorbar(sc, ax=ax); cb.set_label("fraction of group replaced by Q", fontsize=9)

    fig.suptitle("08_13_tiedq_seed192 step 40 — does requiring the problem be SOLVED fix the "
                 "Q inflation?", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote %s\n" % args.out)

    print("%-34s %5s %8s %9s %8s %10s" % ("population", "n", "corr", "bias", "MAE", "mean z_true"))
    for lbl, idx in (("ALL ready groups", list(range(len(zt)))),
                     ("SOLVED (in bank)  -> kept", keep),
                     ("UNSOLVED (no bank) -> dropped", drop)):
        if not idx:
            print("%-34s %5d" % (lbl, 0)); continue
        n, corr, bias, mae = stats(sub(zt, idx), sub(zs, idx))
        print("%-34s %5d %8.3f %+9.4f %8.4f %10.4f"
              % (lbl, n, corr, bias, mae, sum(sub(zt, idx)) / n))

    # how much of the TOTAL inflation the gate removes
    tot = sum(zs[i] - zt[i] for i in range(len(zt)))
    rem = sum(zs[i] - zt[i] for i in drop)
    print("\ntotal summed inflation %.2f; the dropped (unsolved) groups carry %.2f = %.1f%% of it"
          % (tot, rem, 100.0 * rem / tot if tot else float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
