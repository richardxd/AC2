"""Step 40 vs step 57: has 17 steps of Q training changed what the short lane feeds the trainer?

Three panels, all plotting the trainer's signal against the truth the probe uncovered:

  [1] step 40  z_Q vs z_true      -- the original measurement
  [2] step 57  z_Q vs z_true      -- the same measurement 17 steps later
  [3] step 57  Q_prefix vs z_true -- the OTHER Q site, added in the re-run

WHY PANEL 3 MATTERS. Q is called twice in a group's life: once at the shared prefix (the
`probe` site) and once at prefix+10k on each surviving rollout (the `consumed` site, which is
what feeds z_Q). The step-40 probe only measured the second. Having both on the same groups
tests whether Q degrades when reading its own long continuation -- and it does not: the cut
site is the MORE accurate of the two, which makes sense since it has 10k more tokens of
evidence.

READ THE COLOUR BEFORE THE CORRELATION. Points are coloured by the fraction of the group whose
reward was replaced by Q. A group at 0.0 has z_Q == z_true by construction and sits on the
diagonal contributing nothing about Q, so the headline correlation in panels 1-2 is inflated
by a shared term. Panel 3 has no such term: Q_prefix is a single independent prediction.

    python plot_compare.py --csv40 ../08_15_q_probe_step40/zpairs.csv --csv57 zpairs.csv
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


def load(path):
    zt, zq, w, bank, qp = [], [], [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            zt.append(float(p[0])); zq.append(float(p[1])); w.append(float(p[2]))
            bank.append(int(p[3]) if len(p) > 3 else 1)
            qp.append(float(p[4]) if len(p) > 4 else -1.0)
    return zt, zq, w, bank, qp


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv40", default=os.path.join(here, "zpairs.csv"))
    ap.add_argument("--csv57", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_zt, a_zq, a_w, _, _ = load(args.csv40)
    b_zt, b_zq, b_w, _, b_qp = load(args.csv57)
    # panel 3 uses only groups that actually have a prefix-site Q
    p_zt = [t for t, q in zip(b_zt, b_qp) if q >= 0]
    p_qp = [q for q in b_qp if q >= 0]

    fig, axs = plt.subplots(1, 3, figsize=(19.5, 6.0))
    panels = [
        (axs[0], a_zt, a_zq, a_w, "step 40   z_Q vs z_true", "z_Q  —  signal the trainer sees"),
        (axs[1], b_zt, b_zq, b_w, "step 57   z_Q vs z_true", "z_Q  —  signal the trainer sees"),
        (axs[2], p_zt, p_qp, None, "step 57   Q_prefix vs z_true", "Q_prefix  —  Q at the shared prefix"),
    ]
    for ax, x, y, w, title, ylab in panels:
        n, c, bias, mae = stats(x, y)
        if w is None:
            ax.scatter(x, y, c="#d62728", s=46, alpha=.8, edgecolors="white", linewidths=.5)
        else:
            sc = ax.scatter(x, y, c=w, cmap="viridis", s=46, alpha=.85,
                            edgecolors="white", linewidths=.5, vmin=0, vmax=1)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("fraction of group replaced by Q", fontsize=9)
        ax.plot([0, 1], [0, 1], ls="--", lw=1.6, color="#666", zorder=0, label="perfect (y = x)")
        ax.set_xlabel("z_true  —  mean judge reward over all 16")
        ax.set_ylabel(ylab)
        ax.set_title("%s\nn=%d   corr %.3f   bias %+.4f   MAE %.4f" % (title, n, c, bias, mae),
                     fontsize=11.5)
        ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03)
        ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("08_13_tiedq_seed192 — the short lane's Q signal, 17 training steps apart "
                 "(panels 1–2 share a term with the truth; panel 3 does not)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote %s\n" % args.out)

    print("%-30s %5s %8s %10s %9s %11s" % ("", "n", "corr", "bias", "MAE", "mean z_true"))
    for lbl, x, y in (("step 40  z_Q", a_zt, a_zq), ("step 57  z_Q", b_zt, b_zq),
                      ("step 57  Q_prefix", p_zt, p_qp)):
        n, c, bias, mae = stats(x, y)
        print("%-30s %5d %8.3f %+10.4f %9.4f %11.4f" % (lbl, n, c, bias, mae, sum(x) / max(1, n)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
