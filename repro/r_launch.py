"""Explicit local R launch commands; long mode is reserved for Richard's approval."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "r1": ("grpo", 4096, "none"), "r2": ("ac2", 4096, "none"),
    "r3": ("prefix", 4096, "none"), "r4-2k": ("ac2", 2048, "none"),
    "r4-correct-only": ("ac2", 4096, "correct-only"),
    "r4-no-audit": ("ac2", 4096, "no-audit"),
}


def command(task, mode, attempt=1):
    protocol = json.loads((ROOT / "repro/r_protocol.json").read_text())
    assert protocol["model"] == "Qwen/Qwen3-4B-Thinking-2507"
    expected = {"world_size": 7, "rollout_tp": 1, "training_groups": 16,
                "samples_per_group": 4, "response_budget": 16384, "chunk": 4096,
                "q_batch_max": 64, "replay_bound": 256, "steps": 200,
                "save_frequency": 20, "validation_frequency": 10, "validation_samples": 4,
                "smoke_seconds_per_launch": 1740, "strict_judge": True, "judge_max_tokens": 65536}
    assert all(protocol[k] == v for k, v in expected.items()), "protocol and launch implementation differ"
    method, chunk, ablation = TASKS[task]
    smoke = mode.startswith("smoke")
    assert attempt >= 1 and (smoke or attempt == 1)
    name = task if attempt == 1 else f"{task}-attempt{attempt}"
    run = ROOT / ("runs/r-smokes" if smoke else "runs/research") / name
    tracker = run / "checkpoints/latest_checkpointed_iteration.txt"
    if tracker.exists():
        step = int(tracker.read_text().strip())
        assert step > 0 and (run / f"checkpoints/global_step_{step}").is_dir(), "invalid checkpoint tracker"
    cmd = [sys.executable, str(ROOT / "repro/local_runner.py"), "--method", method,
           "--run-dir", str(run), "--model", protocol["model"], "--gpus", "7", "--tp", "1",
           "--batch", "16", "--group", "4", "--response", "16384", "--chunk", str(chunk),
           "--q-train-n", "64", "--replay-bound", "256", "--ablation", ablation,
           "--strict-judge", "--judge-max-tokens", "65536", "--val-n", "4", "--val-freq", "10"]
    if smoke:
        cmd += ["--smoke", "--save-freq", "1", "--val-data", str(ROOT / "runs/r-preflight/test-first.parquet")]
        if mode == "smoke-cold":
            assert not tracker.exists(), "cold smoke already has a checkpoint"
            cmd += ["--steps", "1", "--initial-val"]
        else:
            assert tracker.is_file() and tracker.read_text().strip() == "1", "resume must restore smoke checkpoint1"
            assert (run / "checkpoints/global_step_1").is_dir()
            cmd += ["--steps", "2"]
    else:
        cmd += ["--steps", "200", "--save-freq", "20"]
        if not tracker.exists():
            cmd += ["--initial-val"]
        if mode == "compose":
            cmd += ["--compose-only"]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=TASKS)
    p.add_argument("mode", choices=["compose", "smoke-cold", "smoke-resume", "long"])
    p.add_argument("--print-only", action="store_true")
    p.add_argument("--attempt", type=int, default=1, help="Separate smoke attempt, preserving failed outputs")
    args = p.parse_args()
    assert Path.cwd().resolve() == ROOT
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2,3,4,5,6,7"
    from judge_template_receipt import current_template
    print("JUDGE_TEMPLATE_PIN " + json.dumps(current_template(), sort_keys=True), flush=True)
    cmd = command(args.task, args.mode, args.attempt)
    print(shlex.join(cmd), flush=True)
    if args.print_only:
        return
    # This wrapper does not impose a deadline: smoke commands MUST be enclosed in
    # run_bounded.py, as documented in PROPOSAL_R and the per-task launch receipts.
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
