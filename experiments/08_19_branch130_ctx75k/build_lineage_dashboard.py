"""Combined dashboard for 08_19_branch130_ctx75k: the parent run 08_13_tiedq_seed192 through
step 130, followed by this run's 75k continuation from step 131.

Base = 08_13_tiedq_seed192's plotting cache (written by its refresh_dashboard.sh), truncated to
steps <= 130, the step this run branches from. The continuation (131+) is this run's own
incremental parse. A branch divider is drawn at 130.5.

Truncating the base matters because the parent kept training past step 130, so its cache can
contain steps 131+ of the 50k configuration. append_merge keeps the canonical value on
overlapping steps, so without the truncation the parent's 50k steps would SHADOW this run's 75k
steps of the same number. The parent started from scratch, so its cache is a plain canonical
with no earlier history or dividers to filter; the only divider added is the branch marker.

The branch changes only the response budget (50k -> 75k, carried by SP_MAX_RESPONSE_LEN /
SP_Q_CTX_LIMIT / SP_Q_MAX_TOKEN_LEN / SP_PPO_MAX_TOKEN_LEN); batch geometry, LR, judge and
replay policy match the parent, so a divergence after 130.5 is attributable to the budget.

Run on the cluster A login node:
  SELF_PLAY_ROOT=${AC2_CLUSTER_A_ROOT}/self-play \
    $REPO/.venv/bin/python experiments/08_19_branch130_ctx75k/build_lineage_dashboard.py
"""
import json
import os
import shutil
import subprocess
import sys

REPO = os.environ.get("SELF_PLAY_ROOT", os.path.expandvars("${AC2_CLUSTER_A_ROOT}/self-play"))
sys.path.insert(0, REPO + "/src")
from ac2.viz import merge_fig_data as M  # noqa: E402

E13 = f"{REPO}/experiments/08_13_tiedq_seed192"
EXP = f"{REPO}/experiments/08_19_branch130_ctx75k"
PY = f"{REPO}/.venv/bin/python"
NAME = "08_19_branch130_ctx75k"
BRANCH = 130  # this experiment branches 08_13_tiedq_seed192 at global_step 130
BASE_FIG = f"{E13}/.dash_08_13_tiedq_seed192/analysis/08_13_tiedq_seed192_fig_data.json"
STAGE = f"{EXP}/.dash_lineage"
CONT_TMP = f"{STAGE}/analysis/cont_parse.json"
CONT_CANON = f"{STAGE}/analysis/cont_canonical.json"
OUT_FIG = f"{STAGE}/analysis/{NAME}_lineage_fig_data.json"
os.makedirs(f"{STAGE}/analysis", exist_ok=True)

# 0) base = the parent's canonical plotting cache (maintained by its refresh_dashboard.sh)
if not os.path.exists(BASE_FIG):
    raise FileNotFoundError(
        f"08_13 fig cache missing: {BASE_FIG} — run 08_13's refresh_dashboard.sh once"
    )
base = json.load(open(BASE_FIG))

# 1) truncate the base to steps <= BRANCH (see the docstring): this drops real parent steps
#    above the branch point (the 50k configuration), not just a theoretical tail.
ps = base.get("per_step") or {}
steps = list(ps.get("steps") or [])
keep = [i for i, s in enumerate(steps) if s <= BRANCH]
_dropped_steps = [s for s in steps if s > BRANCH]
for k in list(ps.keys()):
    if isinstance(ps[k], list) and len(ps[k]) == len(steps):
        ps[k] = [ps[k][i] for i in keep]
g = (base.get("global") or {}).get("global_step_contrib") or {}
for side in ("train", "val"):
    if isinstance(g.get(side), dict):
        g[side] = {k: v for k, v in g[side].items() if int(k) <= BRANCH}
if _dropped_steps:
    print(f"[lineage] truncated base: dropped {len(_dropped_steps)} parent step(s) above "
          f"{BRANCH} ({_dropped_steps[0]}..{_dropped_steps[-1]}) — those are the 50k arm")

# 1b) drop any inherited divider above the branch step (the parent has none; cheap guard)
_bda = base.get("dashboard_annotations") or {}
_bvms = _bda.get("vertical_markers")
if isinstance(_bvms, list):
    _kept, _drop = [], []
    for m in _bvms:
        try:
            x = float(m.get("x"))
        except (TypeError, ValueError):
            _kept.append(m)
            continue
        (_kept if x <= BRANCH + 0.5 else _drop).append(m)
    _bda["vertical_markers"] = _kept
    if _drop:
        print(f"[lineage] dropped {len(_drop)} inherited divider(s) above step {BRANCH}")


# 2) parse THIS run's continuation (131+) INCREMENTALLY -- a full --run-dir parse re-reads
#    every train rollout dump and does not scale.
def _full_parse():
    subprocess.run([PY, "-m", "ac2.viz.parse_fig_data", "--run-dir", EXP,
                    "--out", CONT_TMP], check=True)
    return json.load(open(CONT_TMP))


prev_cont = json.load(open(CONT_CANON)) if os.path.exists(CONT_CANON) else None
start_step = None
if prev_cont:
    _rsteps = M._train_rollout_steps(prev_cont)
    if _rsteps:
        start_step = max(_rsteps)  # INCLUSIVE: re-parse the boundary step as the overlap
if start_step is not None:
    print(f"[lineage] incremental parse from train step {start_step}")
    subprocess.run([PY, "-m", "ac2.viz.parse_fig_data", "--run-dir", EXP,
                    "--out", CONT_TMP, "--start-step", str(start_step)], check=True)
    delta = json.load(open(CONT_TMP))
    if start_step not in M._train_rollout_steps(delta):
        print(f"[lineage][WARN] no train-rollout overlap at step {start_step} — "
              f"falling back to a FULL parse rather than appending across a gap")
        prev_cont, delta = None, _full_parse()
else:
    print("[lineage] no continuation cache -> FULL parse (one-time)")
    delta = _full_parse()
if prev_cont:
    _conf = M.overlap_conflicts(prev_cont, delta)
    if _conf:
        print(f"[lineage][WARN] {len(_conf)} overlap conflict(s) on the boundary step; "
              f"canonical values win: {_conf[:3]}")
    cont = M.append_merge(prev_cont, delta)
else:
    cont = delta
json.dump(cont, open(CONT_CANON, "w"))
cst = (cont.get("per_step") or {}).get("steps") or []

# 3) merge (disjoint by construction: base <= 130, cont >= 131)
if cst:
    merged = M.append_merge(base, cont)
    b_cfg = base.get("config") or {}
    m_cfg = merged.setdefault("config", {})
    for k, v in b_cfg.items():
        if m_cfg.get(k) is None and v is not None:
            m_cfg[k] = v
    if not merged.get("manifest") and base.get("manifest"):
        merged["manifest"] = base["manifest"]
    src = f"08_13 seed192(0-{BRANCH}) + branch130 cont({cst[0]}-{cst[-1]})"
else:
    merged = base
    src = f"08_13 seed192(0-{BRANCH}); no branch130 steps parsed yet"

# 4) branch divider at 130.5 -- the context extension is the ONLY intended variable.
_da = merged.setdefault("dashboard_annotations", {})
_vms = _da.setdefault("vertical_markers", [])
bx = BRANCH + 0.5
if not any(isinstance(m, dict) and abs(float(m.get("x", -1)) - bx) < 0.1 for m in _vms):
    _vms.append({"x": bx,
                 "label": "branch @130: response 50k -> 75k (batch/lr/judge unchanged)",
                 "color": "#e31a1c", "linestyle": "--", "linewidth": 1.4,
                 "alpha": 0.9, "label_all": True})

json.dump(merged, open(OUT_FIG, "w"))
mst = (merged.get("per_step") or {}).get("steps") or []
print(f"[lineage] {src}; per_step {(mst[0], mst[-1]) if mst else None} n={len(mst)} -> {OUT_FIG}")


# 5) render + mirror to the tracked experiment root
def _render(fig_path, desc, label):
    r = subprocess.run([PY, "-m", "ac2.viz.render_dashboard", fig_path,
                        f"{STAGE}/analysis", "--experiment-folder", NAME,
                        "--figure-desc", desc],
                       capture_output=True, text=True)
    print(f"[lineage] {label} render rc {r.returncode}",
          (r.stdout or "")[-200:], (r.stderr or "")[-200:])
    if r.returncode != 0:
        return False
    out = f"{NAME}_{desc}.pdf"
    shutil.copy(f"{STAGE}/analysis/{out}", f"{EXP}/{out}")
    print(f"[lineage] wrote {EXP}/{out}")
    return True


ok_full = _render(OUT_FIG, "dashboard", "full")

# 6) RECENT WINDOW: a second PDF over the last WINDOW train steps only. At ~4.5 h/step this
#    run adds steps slowly, so the window is deliberately small; a larger window would be
#    mostly the parent's 50k history and would hide this run's own steps.
WINDOW = int(os.environ.get("DASH_RECENT_WINDOW", "20"))
RECENT_FIG = f"{STAGE}/analysis/{NAME}_recent_fig_data.json"


def _window(fig, steps, keep_lo):
    """Deep-ish copy of `fig` restricted to steps >= keep_lo."""
    import copy
    w = copy.deepcopy(fig)
    ps = w.get("per_step") or {}
    idx = [i for i, s in enumerate(steps) if s >= keep_lo]
    for k, v in list(ps.items()):
        if isinstance(v, list) and len(v) == len(steps):
            ps[k] = [v[i] for i in idx]
    gg = (w.get("global") or {}).get("global_step_contrib") or {}
    for side in ("train", "val"):
        if isinstance(gg.get(side), dict):
            gg[side] = {k: val for k, val in gg[side].items() if int(k) >= keep_lo}
    da = w.get("dashboard_annotations") or {}
    vms = da.get("vertical_markers")
    if isinstance(vms, list):
        da["vertical_markers"] = [
            m for m in vms
            if not (isinstance(m, dict) and m.get("x") is not None)
            or float(m["x"]) >= keep_lo - 0.5
        ]
    return w


if mst and len(mst) > WINDOW:
    lo = mst[-WINDOW]
    recent = _window(merged, mst, lo)
    json.dump(recent, open(RECENT_FIG, "w"))
    rst = (recent.get("per_step") or {}).get("steps") or []
    print(f"[lineage] recent window: steps {rst[0]}..{rst[-1]} (n={len(rst)}) -> {RECENT_FIG}")
    _render(RECENT_FIG, f"dashboard_recent{WINDOW}", f"recent{WINDOW}")
elif mst:
    print(f"[lineage] only {len(mst)} steps (<= window {WINDOW}); skipping the recent PDF")
