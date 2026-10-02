"""E7 critic/state/route acceptance; permissive readiness is labeled as a fixture."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--launch", type=Path, action="append", required=True)
    p.add_argument("--checkpoint", type=int, required=True)
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--require-branches", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    identities = {r["physical_gpu"]: r["uuid"] for r in json.loads(
        (root / "repro/receipts/e1-acceptance-uuid.json").read_text())["kernel_checks"]}
    metrics_path = args.run / "metrics.jsonl"
    metric_rows = list(map(json.loads, metrics_path.read_text().splitlines()))
    metrics = {r["step"]: r["data"] for r in metric_rows}
    assert len(metrics) == len(metric_rows) and sorted(metrics) == list(range(1, args.checkpoint + 1))
    launches, files = [], [metrics_path]
    phase_steps = []
    for path in args.launch:
        result = json.loads((path / "result.json").read_text())
        assert result["returncode"] == 0 and not result["timed_out"] and result["monitor_error"] is None
        launch = json.loads((path / "launch.json").read_text())
        cmd = launch["command"]
        assert (root / cmd[cmd.index("--run-dir") + 1]).resolve() == args.run.resolve()
        assert launch["gpus"] == list("1234567")
        samples = {i: [] for i in range(1, 8)}
        with (path / "gpu.csv").open() as f:
            for row in csv.reader(f):
                assert len(row) == 6
                stamp, index, uuid, util, memory, power = [x.strip() for x in row]
                index = int(index)
                assert index in samples and uuid.removeprefix("GPU-") == identities[index]
                samples[index].append((datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp(),
                                       float(util), float(memory)))
        telemetry = {}
        for gpu, rows in samples.items():
            assert sum(r[1] >= 5 for r in rows) >= 3, (gpu, "insufficient activity evidence")
            telemetry[gpu] = {"samples": len(rows), "active_samples_ge5pct": sum(r[1] >= 5 for r in rows),
                              "max_memory_mib": max(r[2] for r in rows),
                              "max_sampling_gap_s": max((b[0]-a[0] for a,b in zip(rows,rows[1:])), default=0)}
        log = (path / "stdout.log").read_text()
        assert "[sp_dp_pad] update_actor:" in log
        assert re.search(r"rank \d+ nranks 7 .*Init COMPLETE", log)
        completed = re.findall(r"RUN_COMPLETED metrics_added=([^\n\r]+)", log)
        assert len(completed) == 1
        added_path = Path(completed[0])
        assert added_path.resolve().is_relative_to(args.run.resolve() / "launches")
        added = [json.loads(line) for line in added_path.read_text().splitlines()]
        steps = [r["step"] for r in added]
        if phase_steps:
            assert steps[0] == phase_steps[-1] + 1
            assert f"[sp_q] resumed cursor={phase_steps[-1]}," in log
            assert "[sp_q] Q-LR ladder resumed:" in log
            for rank in range(7):
                for label, filename in [("model", "model"), ("optimizer", "optim"), ("lr_scheduler", "extra_state")]:
                    expected = args.run.resolve() / "checkpoints" / f"global_step_{phase_steps[-1]}" / "actor" / f"{filename}_world_size_7_rank_{rank}.pt"
                    assert f"[Rank {rank}] Loaded {label} from {expected}" in log
            assert log.count("[sp_q] O_Q state restored from checkpoint stash") == 7
        phase_steps.extend(steps)
        for row in added:
            for key in ["q/grad_norm", "q/delta_q_applied", "q/consumed_calls"]:
                if key in row["data"]:
                    a, b = row["data"][key], metrics[row["step"]][key]
                    assert a == b or (math.isnan(a) and math.isnan(b))
        launches.append({"path": str(path), "result": result, "metric_steps": steps,
                         "telemetry": telemetry,
                         "engineering_readiness": "--engineering-readiness" in cmd})
        files += [path / "stdout.log", path / "launch.json", path / "result.json", path / "gpu.csv", added_path]
    assert phase_steps == list(range(1, args.checkpoint + 1)), phase_steps
    updated = [s for s in phase_steps if metrics[s].get("q/delta_q_applied", 0) > 0]
    assert updated, "no applied critic update"
    for step in updated:
        assert math.isfinite(metrics[step]["q/grad_norm"]) and metrics[step]["q/grad_norm"] > 0
        assert metrics[step]["q/q_phase_skipped"] == 0
        assert metrics[step].get("q/interleave_late", 0) == 0
    ckpt = args.run / "checkpoints" / f"global_step_{args.checkpoint}"
    for rank in range(7):
        for kind in ["model", "optim", "extra_state"]:
            assert (ckpt / "actor" / f"{kind}_world_size_7_rank_{rank}.pt").stat().st_size > 0
    files += [ckpt / "q_state.json", ckpt / "sp_q_optim/_complete.json"]
    q_state = json.loads((ckpt / "q_state.json").read_text())
    assert q_state["next_dataset_step"] == args.checkpoint
    marker = json.loads((ckpt / "sp_q_optim/_complete.json").read_text())
    assert marker == {"world_size": 7, "global_step": args.checkpoint}
    optimizer_steps = {}
    for rank in range(7):
        blob = torch.load(ckpt / "sp_q_optim" / f"rank_{rank}.pt", map_location="cpu", weights_only=False, mmap=True)
        assert blob["world_size"] == 7
        steps = [float(state["step"]) for state in blob["state"]["state"].values()]
        assert steps and min(steps) == max(steps) == len(updated), "critic optimizer step count does not match applied updates"
        optimizer_steps[rank] = {"parameters_with_state": len(steps), "min_step": min(steps), "max_step": max(steps)}
        del blob
    wave_counts, valid_waves, routes = Counter(), Counter(), Counter()
    consumed_lengths = []
    for path in sorted((args.run / "q_wave").glob("*.jsonl")):
        assert int(path.stem) + 1 in phase_steps
        files.append(path)
        for row in map(json.loads, path.read_text().splitlines()):
            wave_counts[row["kind"]] += 1
            valid_waves[row["kind"]] += int(row["value"] is not None and not row["overflow"])
    for path in sorted((args.run / "rollouts").glob("*.jsonl")):
        assert int(path.stem) in phase_steps
        files.append(path)
        for row in map(json.loads, path.read_text().splitlines()):
            routes[row.get("sp_q_route", "absent")] += 1
            if row.get("sp_q_route_taken") == "q_consumed":
                length = row["response_length"] - row["sp_prefix_len"]
                assert 0 <= length <= args.chunk, (path, length)
                consumed_lengths.append(length)
    assert valid_waves["probe"] > 0
    if args.require_branches:
        assert any(metrics[s]["q/global_gate_open"] > 0 for s in phase_steps)
        assert consumed_lengths and valid_waves["consumed"] > 0
        assert routes["audit"] > 0 and valid_waves["audit_cut"] > 0
    result = {"run": str(args.run), "launches": launches, "checkpoint": args.checkpoint,
              "critic_update_steps": updated, "q_optimizer_steps": optimizer_steps,
              "ready_problems": sum(q_state["ready"].values()), "wave_counts": dict(wave_counts),
              "valid_waves": dict(valid_waves), "routes": dict(routes),
              "consumed_chunks": len(consumed_lengths), "max_consumed_chunk": max(consumed_lengths, default=None),
              "chunk_limit": args.chunk, "branch_checks_required": args.require_branches,
              "scope": "engineering branch coverage; permissive readiness is not a scientific result",
              "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ["critic_update_steps", "ready_problems", "wave_counts", "consumed_chunks"]}))


if __name__ == "__main__":
    main()
