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
    p.add_argument("--require-nonzero", action="store_true")
    p.add_argument("--require-replay", action="store_true")
    args = p.parse_args()
    expected_steps = [int(x) for x in args.steps.split(",")]
    root = Path(__file__).resolve().parents[1]
    identities = {r["physical_gpu"]: r["uuid"] for r in json.loads(
        (root / "repro/receipts/e1-acceptance-uuid.json").read_text())["kernel_checks"]}
    receipt = {"run": str(args.run), "world_size": args.world_size, "launches": [], "steps": []}
    prior_final_step = None
    logged_metrics = {}
    for launch in args.launch:
        result = json.loads((launch / "result.json").read_text())
        assert result["returncode"] == 0 and not result["timed_out"] and not result.get("monitor_error"), result
        manifest = json.loads((launch / "launch.json").read_text())
        command = manifest["command"]
        if "--run-dir" in command:
            launched_run = (root / command[command.index("--run-dir") + 1]).resolve()
        else:
            assert len(command) in {4, 6} and Path(command[1]).name == "r_launch.py"
            assert (root / command[1]).resolve() == root / "repro/r_launch.py"
            assert command[3] in {"smoke-cold", "smoke-resume"}
            from r_launch import TASKS
            assert command[2] in TASKS
            attempt = 1
            if len(command) == 6:
                assert command[4] == "--attempt"
                attempt = int(command[5])
                assert attempt >= 1
            name = command[2] if attempt == 1 else f"{command[2]}-attempt{attempt}"
            launched_run = root / "runs/r-smokes" / name
        assert launched_run == args.run.resolve(), (launched_run, args.run)
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
        log_bytes = (launch / "stdout.log").read_bytes()
        log = log_bytes.decode()
        phase_metrics = {}
        for match in re.finditer(r"\bstep:(\d+) - ([^\n\r]+)", log):
            step = int(match[1])
            if step == 0:
                assert "val-" in match[2] and "actor/grad_norm" not in match[2]
                continue
            phase_metrics[step] = {}
            for key in ("actor/pg_loss", "actor/grad_norm"):
                value = re.search(re.escape(key) + r":(?:np\.float\d+\()?([-+\d.eE]+)", match[2])
                assert value, (launch, step, key)
                phase_metrics[step][key] = float(value[1])
        checkpoint_pattern = re.escape(str(args.run.resolve() / "checkpoints" / "global_step_"))
        phase_steps = sorted(set(int(x) for x in re.findall(r"local_global_step_folder: " + checkpoint_pattern + r"(\d+)", log)))
        assert phase_steps, "no checkpoint steps logged by this launch"
        for step in phase_steps:
            state = args.run / "checkpoints" / f"global_step_{step}" / "actor" / f"extra_state_world_size_{args.world_size}_rank_0.pt"
            assert manifest["unix"] <= state.stat().st_mtime <= manifest["unix"] + result["elapsed_s"], state
        completed = re.findall(r"RUN_COMPLETED metrics_added=([^\n\r]+)", log)
        if completed:
            added = [json.loads(line) for line in Path(completed[-1]).read_text().splitlines()]
            phase_metrics = {r["step"]: {k:r["data"][k] for k in ("actor/pg_loss", "actor/grad_norm")}
                             for r in added if r["step"] > 0}
        loaded_step = None
        if prior_final_step is not None:
            loaded_step = prior_final_step
            assert min(phase_steps) == loaded_step + 1, (loaded_step, phase_steps)
            checkpoint = args.run.resolve() / "checkpoints" / f"global_step_{loaded_step}" / "actor"
            for label, kind in (("model", "model"), ("optimizer", "optim"), ("lr_scheduler", "extra_state")):
                for rank in range(args.world_size):
                    expected_path = checkpoint / f"{kind}_world_size_{args.world_size}_rank_{rank}.pt"
                    assert f"Loaded {label} from {expected_path}" in log, (launch, expected_path)
        else:
            assert min(phase_steps) == 1, "first launch must establish the pre-resume sequence"
        if args.require_replay:
            if loaded_step is None:
                assert "[sp_replay] no persisted state" in log
            else:
                state_path = args.run.resolve() / "checkpoints" / f"global_step_{loaded_step}" / "sp_replay_state.json"
                state = json.loads(state_path.read_text())
                assert f"[sp_replay] resumed cursor={loaded_step}, {len(state['ema'])} EMA entries from {state_path}" in log
                assert f"[sp_replay] delta log: applied {loaded_step}, truncated 0" in log
        prior_final_step = max(phase_steps)
        logged_metrics.update(phase_metrics)
        assert re.search(rf"rank \d+ nranks {args.world_size} .*Init COMPLETE", log), "missing actual NCCL world size"
        placements = re.findall(r"\[placement-debug\] replica=(\d+) reward_model=False world_size=(\d+)", log)
        assert placements and sum(int(tp) for _,tp in placements) == args.world_size, placements
        receipt["launches"].append({"path": str(launch), "result": result, "physical_gpus": telemetry,
                                   "rollout_placements": placements, "logged_checkpoint_steps": phase_steps,
                                   "captured_metric_steps": sorted(phase_metrics),
                                   "loaded_checkpoint_step": loaded_step,
                                   "log_sha256": hashlib.sha256(log_bytes).hexdigest()})
    rows = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]
    metrics = {r["step"]: r["data"] for r in rows}
    assert set(expected_steps).issubset(metrics), list(metrics)
    assert set(expected_steps).issubset({s for phase in receipt["launches"] for s in phase["logged_checkpoint_steps"]})
    for step, logged in logged_metrics.items():
        for key, value in logged.items():
            assert math.isclose(metrics[step][key], value, rel_tol=1e-7, abs_tol=1e-10), (step, key)
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
    if args.require_nonzero:
        assert metrics[expected_steps[-1]]["actor/grad_norm"] > 0, "resume has not demonstrated a nonzero policy update"
    receipt["zero_gradient_steps"] = [s for s in expected_steps if metrics[s]["actor/grad_norm"] == 0]
    receipt["nonzero_update_demonstrated"] = any(metrics[s]["actor/grad_norm"] > 0 for s in expected_steps)
    receipt["metrics_sha256"] = hashlib.sha256((args.run / "metrics.jsonl").read_bytes()).hexdigest()
    if args.require_replay:
        seed = args.run / "cold/replay_seed_cold/replay_buffer_manifest.json"
        seed_sha = hashlib.sha256(seed.read_bytes()).hexdigest()
        seed_data = json.loads(seed.read_text())
        assert seed_data["entries"] == 0 and seed_data["problems"] == 0
        from verl.trainer.ppo.sp_q_readiness import verify_manifest_shards
        verify_manifest_shards(str(seed.parent), seed_data["shards"],
            sorted(str(p) for p in (seed.parent / "replay_buffer").glob("shard_*.jsonl")), "smoke cold replay")
        assert all(not (seed.parent / name).read_text().strip() for name in seed_data["shards"])
        delta = args.run / "replay_buffer_deltas.jsonl"
        deltas = [json.loads(line) for line in delta.read_text().splitlines()]
        assert [r["dataset_step"] for r in deltas] == list(range(max(expected_steps)))
        sequence = 0
        active_entries = {}
        reconstructed = {}
        for row in deltas:
            step = row["dataset_step"] + 1
            m = metrics[step]
            assert len(row["added_entries"]) == m["replay/admitted"]
            assert len(row["replaced_entry_ids"]) == m["replay/replaced"]
            for entry_id in row["replaced_entry_ids"]:
                assert entry_id in active_entries
                del active_entries[entry_id]
            for entry in row["added_entries"]:
                assert entry["entry_id"] not in active_entries and entry["response_token_ids"]
                assert entry["meta"]["buffer_seq"] == sequence
                assert entry["meta"]["dataset_step"] == row["dataset_step"]
                sequence += 1
                active_entries[entry["entry_id"]] = entry["qid"]
            assert len(active_entries) == m["replay/buffer_size"]
            assert len(set(active_entries.values())) == m["replay/coverage"]
            reconstructed[step] = {"sequence": sequence, "covered": set(active_entries.values())}
        states = []
        previous_ema_keys = set()
        for step in expected_steps:
            path = args.run / "checkpoints" / f"global_step_{step}" / "sp_replay_state.json"
            state = json.loads(path.read_text())
            assert state["next_dataset_step"] == step and state["seed_manifest_sha"] == seed_sha
            assert not state["reseeds"]
            assert state["replay_policy"]["bucketing"] == "global"
            assert state["buffer_seq_next"] == reconstructed[step]["sequence"]
            assert state["ema"] and previous_ema_keys.issubset(state["ema"])
            assert all(math.isfinite(v) and v >= 0 for v in state["ema"].values())
            covered = reconstructed[step]["covered"]
            assert covered.issubset(state["ema"])
            if covered:
                assert math.isclose(sum(state["ema"][q] for q in covered) / len(covered),
                                    metrics[step]["replay/ema_mean"], rel_tol=1e-7, abs_tol=1e-10)
            else:
                assert math.isnan(metrics[step]["replay/ema_mean"])
            previous_ema_keys = set(state["ema"])
            if states:
                assert state["replay_policy"] == states[0]["replay_policy"]
            states.append({"step": step, "ema_entries": len(state["ema"]),
                "buffer_seq_next": state["buffer_seq_next"], "replay_policy": state["replay_policy"],
                "state_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        receipt["replay"] = {"states": states, "seed_sha256": seed_sha,
            "delta_sha256": hashlib.sha256(delta.read_bytes()).hexdigest(),
            "added_entries_per_step": [len(r["added_entries"]) for r in deltas]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(receipt, f, indent=2)
    print(f"MECHANICAL_CHECKS_PASSED nonzero_update={receipt['nonzero_update_demonstrated']} {args.output}")


if __name__ == "__main__":
    main()
