"""Explicit E8/E9 planning arithmetic; estimates are not runtime confidence bounds."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    for name in ["grpo", "ac2-cold", "ac2-replay", "prefix", "gate", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = {k: getattr(args, k) for k in ["grpo", "ac2_cold", "ac2_replay", "prefix", "gate"]}
    paths["e4"] = root / "repro/receipts/e4-cost-summary.json"
    raw = {k: json.loads(v.read_text()) for k, v in paths.items()}
    train_cost = raw["e4"]["stats"]["train"]["mean_cost_upper_usd"]
    val_cost = raw["e4"]["stats"]["val"]["mean_cost_upper_usd"]
    gate = raw["gate"]["4b"]
    rate = gate["tokens_per_gpu_generation_second"]
    assert rate > 0
    # Twenty periodic evaluations plus step zero; actual R launch must enable it.
    steps, evaluations, val_samples, problems, saves = 200, 21, 4, 60, 10
    val_tokens = val_samples * gate["generated_tokens"]
    val_s = val_tokens / (7 * rate) + problems * val_samples * raw["e4"]["stats"]["val"]["mean_latency_s"] / 4
    measured = {}
    cold, replay = raw["ac2_cold"], raw["ac2_replay"]
    assert Path(cold["run"]).resolve() == Path(replay["run"]).resolve(), "AC2 states must share a run"
    assert Path(cold["manifest"]).resolve() != Path(replay["manifest"]).resolve(), "distinct AC2 launches required"
    assert [r["step"] for r in cold["steps"]] == [1], "cold AC2 must be step1"
    assert [r["step"] for r in replay["steps"]] == [2], "populated AC2 must be resumed step2"
    cq, rq = cold["steps"][0]["q_metrics"], replay["steps"][0]["q_metrics"]
    assert cq["q/fifo_size"] == 0 and cq["q/q_phase_skipped"] == 1
    assert rq["q/fifo_size"] > 0 and rq["q/q_records_trained"] > 0
    assert rq["q/q_phase_skipped"] == 0
    assert rq["q/delta_q_applied"] > 0 and math.isfinite(rq["q/grad_norm"]) and rq["q/grad_norm"] > 0
    for q in [cq, rq]:
        assert all(q[k] == 0 for k in ["q/global_gate_open", "q/global_ready_effective",
                                      "q/ready_problems", "q/consumed_calls", "q/audit_cut_calls"]), "unready calibration required"
    for name in ["grpo", "ac2_cold", "ac2_replay", "prefix"]:
        receipt = raw[name]
        manifest = Path(receipt["manifest"])
        config_args = json.loads((manifest / "arguments.json").read_text())
        paths[name + "_arguments"] = manifest / "arguments.json"
        expected_method = "ac2" if name.startswith("ac2_") else name
        expected = {"method": expected_method, "model": "Qwen/Qwen3-4B-Thinking-2507",
                    "gpus": 7, "tp": 1, "batch": 16, "group": 4, "response": 16384,
                    "chunk": 4096, "q_train_n": 64, "engineering_readiness": False,
                    "ablation": "none", "data": "runs/e3/canonical", "replay_bound": 256}
        assert all(config_args[k] == v for k, v in expected.items()), (name, config_args)
        assert receipt["eq6_A"] == 8044544000 and receipt["eq6_B"] == 589824
        assert receipt["launch_result"] is not None
        phase = receipt["steps"]
        assert len(phase) == 1, "calibrate one fresh step per bounded launch"
        row = phase[0]
        timing = row["timing_s"]
        assert timing["timing_s/gen"] > 0
        checkpoint = timing.get("timing_s/save_checkpoint", 0)
        assert checkpoint > 0
        measured[name] = {
            "step_without_checkpoint_s": timing["timing_s/step"] - checkpoint,
            "checkpoint_s": checkpoint,
            "launch_overhead_s": max(0, receipt["launch_result"]["elapsed_s"] - timing["timing_s/step"]),
            "judge_upper_usd_per_step": receipt["judge_charged_upper_usd"],
            "decoding_flops_per_step": row["decoding_flops"],
        }
    # Cold and populated replay bracket only the observed states, not future readiness.
    ac2 = {k: max(measured[n][k] for n in ["ac2_cold", "ac2_replay"]) for k in measured["ac2_cold"]}
    tasks = {}
    for name, source, trajectories in [
        ("R1", measured["grpo"], 64), ("R2", ac2, 80), ("R3", measured["prefix"], 80),
        ("R4_2k", ac2, 80), ("R4_correct_only", ac2, 80), ("R4_no_audit", ac2, 80),
    ]:
        seconds = steps * source["step_without_checkpoint_s"] + saves * source["checkpoint_s"]
        seconds += source["launch_overhead_s"] + evaluations * val_s
        tasks[name] = {
            "steps": steps, "validation_events": evaluations, "validation_n": val_samples,
            "wall_days_observed_component_projection": seconds / 86400,
            "wall_days_2x_planning_allowance": 2 * seconds / 86400,
            "judge_usd_observed_train_plus_e9_val": steps * source["judge_upper_usd_per_step"] + evaluations * val_samples * gate["judge_charged_upper_usd"],
            "judge_usd_every_trajectory_graded_at_e4_mean": steps * trajectories * train_cost + evaluations * problems * val_samples * val_cost,
            "decode_flops_no_readiness_speedup_projection": steps * source["decoding_flops_per_step"],
            "basis": "measured matching arm" if not name.startswith("R4") else "AC2 proxy; ablation-specific mature timing unmeasured",
        }
    # A prospective ready-set probe, after a real R2 checkpoint exists.
    groups, continuations = 32, 4
    tasks["R5"] = {
        "ready_problems": groups, "continuations_per_problem": continuations,
        "policy_generation_hours_full_16k_envelope": groups * continuations * 16384 / (7 * rate) / 3600,
        "judge_usd_every_continuation_graded_at_e4_mean": groups * continuations * train_cost,
        "unmeasured_overhead": "checkpoint merge, critic queries and startup; bounded R5 smoke will measure these",
        "scientific_dependency": "real R2 checkpoint with at least32 ready problems; otherwise report fewer available, never invent readiness",
    }
    result = {
        "tasks": tasks, "measured_components": measured,
        "validation_seconds_per_event_projection": val_s,
        "assumptions": [
            "4B, seven TP1 replicas,16groups x4,16k response,4k chunk,Q batch64;200steps; save/evaluate every20/10steps.",
            "Four validation samples/problem per ROADMAP starting point;21events include step zero. Scale E9's60x1 observed cost/time by4 (linear planning assumption, not a measured240-rollout event). Initial validation must be enabled for fresh R launches.",
            "Generation throughput includes prefill and queueing; val runtime assumes balanced work on7GPUs plus serialized4-way judge service.",
            "Observed costs include missing-proof short circuits. All-graded E4 mean is a planning scenario, not a hard upper bound.",
            "One step/state cannot establish runtime variance, mature readiness savings or ablation effects.2x allowance is a stated margin, not a confidence interval.",
            "R5 throughput is a full16k envelope at step-zero policy speed; exact checkpoint/probe timing awaits smoke.",
            "All long runs and any spend above current engineering cap require Richard's approval; this file launches nothing.",
        ],
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
    }
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(tasks, indent=2))


if __name__ == "__main__":
    main()
