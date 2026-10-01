"""Dashboard for 08_26_s192b40_g10k_noaudit, rendered together with the parent run's history.

Base = the 08_13_tiedq_seed192 plotting cache, truncated to <= 40, the step this run branches
from. Continuation (41+) is this run's own parse. Branch divider at 40.5. 08_13 is a
from-scratch run, so its own fig_data covers the whole history before the branch.

The base cache is produced where the parent ran and copied over through a private Hugging Face
repo (`python -m ac2.utils.hf_ckpt_sync pull-plot 08_13_tiedq_seed192`). Truncation to <= 40
freezes the base, so re-pulling a later snapshot of the parent cannot change the rendered
dashboard.

Run on the cluster B login node:
  SELF_PLAY_ROOT=${AC2_CLUSTER_B_ROOT}/self-play \
    $REPO/.venv/bin/python experiments/08_26_s192b40_g10k_noaudit/build_lineage_dashboard.py
"""
import json
import os
import shutil
import subprocess
import sys

REPO = os.environ.get("SELF_PLAY_ROOT", os.path.expandvars("${AC2_CLUSTER_B_ROOT}/self-play"))
sys.path.insert(0, REPO + "/src")
from ac2.viz import merge_fig_data as M  # noqa: E402

E13 = f"{REPO}/experiments/08_13_tiedq_seed192"
EXP = f"{REPO}/experiments/08_26_s192b40_g10k_noaudit"
PY = f"{REPO}/.venv/bin/python"
NAME = "08_26_s192b40_g10k_noaudit"
BRANCH = 40  # this experiment branches 08_13_tiedq_seed192 at global_step 40
BASE_FIG = f"{E13}/analysis/08_13_tiedq_seed192_fig_data.json"
STAGE = f"{EXP}/.dash_lineage"
CONT_TMP = f"{STAGE}/analysis/cont_parse.json"
CONT_CANON = f"{STAGE}/analysis/cont_canonical.json"
OUT_FIG = f"{STAGE}/analysis/{NAME}_lineage_fig_data.json"
os.makedirs(f"{STAGE}/analysis", exist_ok=True)

# 0) base = the 08_13 cache (a from-scratch run, so it IS the entire prehistory)
if not os.path.exists(BASE_FIG):
    raise FileNotFoundError(
        f"08_13 fig cache missing: {BASE_FIG} — pull it from cluster A with\n"
        "  HF_TOKEN=$(cat ${AC2_CLUSTER_B_ROOT}/.hf_token) \\\n"
        "    python -m ac2.utils.hf_ckpt_sync pull-plot 08_13_tiedq_seed192"
    )
base = json.load(open(BASE_FIG))

# 1) truncate the base to steps <= BRANCH (this run owns 41+; append_merge keeps
#    canonical on overlap, so a non-truncated base would shadow the continuation).
ps = base.get("per_step") or {}
steps = list(ps.get("steps") or [])
keep = [i for i, s in enumerate(steps) if s <= BRANCH]
for k in list(ps.keys()):
    if isinstance(ps[k], list) and len(ps[k]) == len(steps):
        ps[k] = [ps[k][i] for i in keep]
g = (base.get("global") or {}).get("global_step_contrib") or {}
for side in ("train", "val"):
    if isinstance(g.get(side), dict):
        g[side] = {k: v for k, v in g[side].items() if int(k) <= BRANCH}

# 1b) drop inherited dividers ABOVE the branch step (they annotate the parent's later steps).
_bda = base.get("dashboard_annotations") or {}
_bvms = _bda.get("vertical_markers")
if isinstance(_bvms, list):
    _kept, _dropped = [], []
    for m in _bvms:
        try:
            x = float(m.get("x"))
        except (TypeError, ValueError):
            _kept.append(m)
            continue
        (_kept if x <= BRANCH + 0.5 else _dropped).append(m)
    _bda["vertical_markers"] = _kept
    if _dropped:
        print(f"[lineage] dropped {len(_dropped)} inherited divider(s) above step {BRANCH}: "
              + ", ".join(str(d.get('label')) for d in _dropped))

# 2) parse THIS run's continuation (41+) INCREMENTALLY (a full --run-dir parse
#    re-reads every train rollout dump and does not scale).
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

# 3) merge (disjoint: base <= 40, cont >= 41)
if cst:
    merged = M.append_merge(base, cont)
    b_cfg = base.get("config") or {}
    m_cfg = merged.setdefault("config", {})
    for k, v in b_cfg.items():
        if m_cfg.get(k) is None and v is not None:
            m_cfg[k] = v
    if not merged.get("manifest") and base.get("manifest"):
        merged["manifest"] = base["manifest"]
    src = f"08_13(0-{BRANCH}) + s192b40 cont({cst[0]}-{cst[-1]})"
else:
    merged = base
    src = f"08_13(0-{BRANCH}); no branch steps parsed yet"

# 4) branch divider at BRANCH + 0.5. Steps <= BRANCH are the parent (08_13) with a 1-in-4
#    audit lane; steps above it are this run with NO audit lane (g stays 10000).
#    The marker label string below was written for the g = 5000 companion branch and
#    overstates the change for this run, which keeps g = 10000.
_da = merged.setdefault("dashboard_annotations", {})
_vms = _da.setdefault("vertical_markers", [])
bx = BRANCH + 0.5
if not any(isinstance(m, dict) and abs(float(m.get("x", -1)) - bx) < 0.1 for m in _vms):
    _vms.append({"x": bx,
                 "label": "branch @40: g 10k->5k, audit lane OFF",
                 "color": "#e31a1c", "linestyle": "--", "linewidth": 1.4,
                 "alpha": 0.9, "label_all": True})

# 4b) Only the branch divider is added here; add further markers for this run's own mid-run
#     changes if needed.

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

# 6) RECENT WINDOW: a second PDF over the last WINDOW train steps only.
WINDOW = int(os.environ.get("DASH_RECENT_WINDOW", "30"))
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
