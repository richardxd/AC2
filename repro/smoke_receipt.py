"""Assert E6 evidence from completed launches; never infer activity from exit alone."""
import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--launch", type=Path, action="append", required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--steps", default="1,2,3")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    expected_steps = [int(x) for x in args.steps.split(",")]
    root = Path(__file__).resolve().parents[1]
    identities = {r["physical_gpu"]: r["uuid"] for r in json.loads(
        (root / "repro/receipts/e1-acceptance-uuid.json").read_text())["kernel_checks"]}
    receipt = {"run": str(args.run), "world_size": args.world_size, "launches": [], "steps": []}
    combined_log = ""
    for launch in args.launch:
        result = json.loads((launch / "result.json").read_text())
        assert result["returncode"] == 0 and not result["timed_out"] and not result.get("monitor_error"), result
        manifest = json.loads((launch / "launch.json").read_text())
        gpus = [int(x) for x in manifest["gpus"]]
        assert len(gpus) == args.world_size and len(set(gpus)) == len(gpus) and 0 not in gpus
        samples = {g: [] for g in gpus}
        with (launch / "gpu.csv").open() as f:
            for row in csv.reader(f):
                assert len(row) == 6, row
                stamp, index, uuid, util, memory, power = [x.strip() for x in row]
                index = int(index)
                assert index in samples and uuid.removeprefix("GPU-") == identities[index]
                samples[index].append((datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(util), float(memory)))
        telemetry = {}
        for gpu, rows in samples.items():
            assert sum(r[1] >= 5 for r in rows) >= 3, (gpu, "insufficient activity evidence")
            telemetry[gpu] = {"samples": len(rows), "active_samples_ge5pct": sum(r[1] >= 5 for r in rows),
                              "max_util_pct": max(r[1] for r in rows), "max_memory_mib": max(r[2] for r in rows),
                              "max_sampling_gap_s": max((b[0]-a[0] for a,b in zip(rows,rows[1:])), default=0)}
        log = (launch / "stdout.log").read_text()
        combined_log += log
        assert re.search(rf"rank \d+ nranks {args.world_size} .*Init COMPLETE", log), "missing actual NCCL world size"
        placements = re.findall(r"\[placement-debug\] replica=(\d+) reward_model=False world_size=(\d+)", log)
        assert placements and sum(int(tp) for _,tp in placements) == args.world_size, placements
        receipt["launches"].append({"path": str(launch), "result": result, "physical_gpus": telemetry,
                                   "rollout_placements": placements, "log_sha256": hashlib.sha256(log.encode()).hexdigest()})
    rows = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]
    metrics = {r["step"]: r["data"] for r in rows}
    assert set(expected_steps).issubset(metrics), list(metrics)
    for step in expected_steps:
        m = metrics[step]
        assert math.isfinite(m["actor/pg_loss"]) and math.isfinite(m["actor/grad_norm"]), m
        assert m["actor/grad_norm"] >= 0
        directory = args.run / "checkpoints" / f"global_step_{step}" / "actor"
        shards = []
        for rank in range(args.world_size):
            for kind in ("model", "optim", "extra_state"):
                f = directory / f"{kind}_world_size_{args.world_size}_rank_{rank}.pt"
                assert f.is_file() and f.stat().st_size > 0, f
                shards.append({"path": str(f), "bytes": f.stat().st_size})
        receipt["steps"].append({"step": step, "metrics": {k:v for k,v in m.items()
            if k.startswith(("actor/", "timing_s/", "q/"))}, "checkpoint_shards": shards})
    if len(args.launch) > 1:
        assert "Loaded model from" in combined_log and "Loaded optimizer from" in combined_log
        assert "Loaded lr_scheduler from" in combined_log
        assert metrics[expected_steps[-1]]["actor/grad_norm"] > 0, "resume has not demonstrated a nonzero policy update"
    receipt["zero_gradient_steps"] = [s for s in expected_steps if metrics[s]["actor/grad_norm"] == 0]
    receipt["metrics_sha256"] = hashlib.sha256((args.run / "metrics.jsonl").read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(receipt, f, indent=2)
    print(f"SMOKE_ACCEPTED {args.output}")


if __name__ == "__main__":
    main()
