"""Fail-closed acceptance of a completed 200-step surrogate R run."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

from surrogate_common import ROOT, profile
from local_runner import judge_accounting


def collect(run, launch):
    from r_curves import collect as collect_curve
    report = collect_curve(run)
    assert not report["engineering_smoke"]
    assert [r["step"] for r in report["training"]] == list(range(1, 201))
    assert [r["step"] for r in report["curve"]] == list(range(0, 201, 10))
    assert all(r["n_problems"] == 60 and r["samples_per_problem"] == 4 for r in report["curve"])
    assert report["match_fields"]["judge_profile"] == profile()
    manifest = json.loads((launch / "launch.json").read_text())
    result = json.loads((launch / "result.json").read_text())
    assert result["returncode"] == 0 and result["remaining_owned_processes"] == 0 and result["monitor_error"] is None
    assert manifest["task"] == run.name and manifest["physical_gpus"] == list(range(1, 8))
    cmd = manifest["command"]
    assert Path(cmd[1]).resolve() == ROOT / "repro/local_runner.py"
    for flag, value in (("--run-dir", str(run)), ("--steps", "200"), ("--save-freq", "20"),
                        ("--judge-backend", "surrogate"), ("--val-freq", "10")):
        assert cmd[cmd.index(flag)+1] == value
    assert "--smoke" not in cmd and "--initial-val" in cmd
    assert (run / "checkpoints/latest_checkpointed_iteration.txt").read_text().strip() == "200"
    checkpoints = []
    for step in range(20, 201, 20):
        actor = run / f"checkpoints/global_step_{step}/actor"
        for rank in range(7):
            for kind in ("model", "optim", "extra_state"):
                path = actor / f"{kind}_world_size_7_rank_{rank}.pt"
                assert path.is_file() and path.stat().st_size > 0
        checkpoints.append(step)
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    gradients = [r["data"]["actor/grad_norm"] for r in metrics if r["step"] > 0]
    assert len(gradients) == 200 and all(math.isfinite(x) and x >= 0 for x in gradients) and max(gradients) > 0
    text = (launch / "stdout.log").read_text()
    assert re.search(r"rank \d+ nranks 7 .*Init COMPLETE", text)
    identities = {r["physical_gpu"]: r["uuid"] for r in json.loads(
        (ROOT / "repro/receipts/e1-acceptance-uuid.json").read_text())["kernel_checks"]}
    activity = {i: 0 for i in range(1, 8)}
    with (launch / "gpu.csv").open() as stream:
        for row in csv.reader(stream):
            stamp, index, uuid, util, memory = [x.strip() for x in row]
            index = int(index)
            assert index in activity and uuid.removeprefix("GPU-") == identities[index]
            activity[index] += float(util) >= 5
    assert all(n >= 3 for n in activity.values())
    assert judge_accounting() == json.loads((launch / "api_before.json").read_text())
    if run.name == "r2":
        assert all("ready_fraction" in row and "global_gate_open_at_end" in row for row in report["training"])
        baseline_path = launch.parent / "r1/acceptance.json"
        baseline = json.loads(baseline_path.read_text())
        assert baseline["surrogate_acceptance"]["complete_steps"] == 200
        assert report["match_fields"] == baseline["match_fields"]
        peak = max(point["mean_score"] for point in baseline["curve"])
        report["baseline_comparison"] = {"baseline": str(baseline_path), "observed_peak": peak,
            "first_point_exceeding_observed_peak": next((point for point in report["curve"] if point["mean_score"] > peak), None),
            "scope": "Observed checkpoint grid only; no interpolation or statistical superiority claim."}
        report["sha256"][str(baseline_path)] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    report["surrogate_acceptance"] = {"task": run.name, "complete_steps": 200,
        "checkpoint_steps": checkpoints, "physical_gpu_activity": activity, "api_calls_added": 0,
        "launch": str(launch), "log_sha256": hashlib.sha256((launch / "stdout.log").read_bytes()).hexdigest(),
        "scope": "Full validation/FLOPs curve under a local surrogate, not the paper's DeepSeek judge."}
    return report
