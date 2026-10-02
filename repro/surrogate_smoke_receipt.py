"""S3 timing and provenance supplement to actual distributed save/resume acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from surrogate_common import ROOT, profile
from local_runner import judge_accounting


def union_seconds(intervals):
    total, end = 0., -math.inf
    for start, stop in sorted(intervals):
        assert stop >= start
        total += max(0, stop-max(start, end))
        end = max(end, stop)
    return total


def collect():
    run = ROOT / "runs/surrogate/r-smokes/r1"
    acceptance_path = ROOT / "runs/surrogate/s3-save-resume.json"
    acceptance = json.loads(acceptance_path.read_text())
    assert acceptance["world_size"] == 7 and not acceptance["zero_gradient_steps"]
    assert [x["step"] for x in acceptance["steps"]] == [1, 2]
    manifests = sorted((run / "launches").glob("*/surrogate_profile.json"))
    assert len(manifests) == 2
    metrics = {r["step"]: r["data"] for r in map(json.loads, (run / "metrics.jsonl").read_text().splitlines())}
    rows, hashes = [], {str(acceptance_path): hashlib.sha256(acceptance_path.read_bytes()).hexdigest()}
    for step, path in enumerate(manifests, 1):
        assert json.loads(path.read_text()) == profile()
        launch = path.parent
        assert json.loads((launch / "judge_before.json").read_text()) == judge_accounting()
        assert json.loads((launch / "judge_after.json").read_text()) == judge_accounting()
        before = {r["id"]: r for r in json.loads((launch / "surrogate_before.json").read_text())}
        after = {r["id"]: r for r in json.loads((launch / "surrogate_after.json").read_text())}
        assert all(after[key] == row for key, row in before.items())
        calls = [r for key,r in after.items() if key not in before]
        assert calls and all(r["status"] == 200 and r["finished"] is not None for r in calls)
        raw = [json.loads(Path(r["receipt"]).read_text()) for r in calls]
        assert all(r["profile"] == profile() for r in raw)
        train = [r for r in raw if r["route"] == "train"]
        val = [r for r in raw if r["route"] == "val"]
        assert train and (bool(val) == (step == 1))
        for call in raw:
            assert call["status"] == 200 and all(c["finish_reason"] == "stop" for c in call["response"]["choices"])
        m = metrics[step]
        busy = union_seconds([(r["dispatched"], r["finished"]) for r in train])
        pending = union_seconds([(r["started"], r["finished"]) for r in train])
        assert 0 < busy <= pending <= m["timing_s/step"]
        rows.append({"step": step, "step_s": m["timing_s/step"], "generation_with_overlapped_judging_s": m["timing_s/gen"],
                     "checkpoint_s": m["timing_s/save_checkpoint"], "train_judge_calls": len(train),
                     "initial_val_judge_calls": len(val), "judge_busy_union_s": busy,
                     "judge_pending_union_s": pending, "step_fraction_with_judge_busy": busy/m["timing_s/step"],
                     "slowest_trajectory_compute_score_s": m["timing_s/agent_loop/slowest/compute_score"],
                     "train_judge_completion_tokens": sum(r["response"]["usage"]["completion_tokens"] for r in train)})
        for file in list(launch.glob("*.json")) + [Path(r["receipt"]) for r in calls]:
            hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    return {"accepted": True, "profile": profile(), "run": str(run), "steps": rows,
            "api_calls_added": 0, "completed_api_calls": len(judge_accounting()),
            "scope": "Two-step engineering fixture, problem0 x4 initial validation. Judge intervals overlap policy generation; busy fraction is not exclusive critical-path stall. No full-validation or mature-policy timing claim.",
            "sha256": hashes}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = collect()
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != "sha256"}))
