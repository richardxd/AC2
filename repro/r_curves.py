"""R1–R4 curves from complete raw rollouts; no cached or imputed measurements."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_exporter", ROOT / "scripts/export_paper_metrics.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def validation(rows, samples, problems, logged_mean):
    assert len(rows) == samples * problems
    groups = defaultdict(list)
    for row in rows:
        assert all(row[k] == 0 for k in ("judge_parse_failed", "judge_http_error", "judge_truncated"))
        score = row["acc"]
        assert math.isfinite(score) and 0 <= score <= 1
        groups[row["input"]].append(score)
    assert len(groups) == problems and all(len(v) == samples for v in groups.values())
    group_means = [sum(v)/samples for v in groups.values()]
    mean = sum(group_means)/problems
    assert math.isclose(mean, logged_mean, abs_tol=1e-8), "raw and logged validation means disagree"
    margin = math.sqrt(math.log(40)/(2*problems))
    return {"mean_score": mean, "n_problems": problems, "samples_per_problem": samples,
            "nonzero_fraction": sum(x > 0 for v in groups.values() for x in v)/len(rows),
            "separate_95pct_hoeffding": [max(0, mean-margin), min(1, mean+margin)],
            "problem_means": {hashlib.sha256(k.encode()).hexdigest(): sum(v)/samples for k,v in groups.items()}}


def collect(run):
    import pandas as pd
    from omegaconf import OmegaConf
    from verl.trainer.ppo.difficulty import qid_from_messages
    manifests = sorted((run / "launches").glob("*/config.yaml"))
    assert manifests
    cfg = OmegaConf.load(manifests[-1])
    arguments = json.loads((manifests[-1].parent / "arguments.json").read_text())
    assert arguments["model"] == "Qwen/Qwen3-4B-Thinking-2507" and arguments["val_n"] == 4
    assert cfg.reward.custom_reward_function.path.endswith("/repro/strict_reward.py")
    def one_file(value):
        if isinstance(value, str):
            return Path(value)
        assert len(value) == 1
        return Path(value[0])
    train_path, val_path = one_file(cfg.data.train_files), one_file(cfg.data.val_files)
    train = pd.read_parquet(train_path)
    unique_train = len({qid_from_messages(x) for x in train["prompt"]})
    problems = len(pd.read_parquet(val_path))
    metrics_path = run / "metrics.jsonl"
    metrics = {}
    for line in metrics_path.read_text().splitlines():
        record = json.loads(line)
        assert record["step"] not in metrics, "duplicate metric step, possibly repeated validation"
        metrics[record["step"]] = record["data"]
    training_steps = sorted(s for s,m in metrics.items() if "actor/grad_norm" in m)
    assert training_steps and training_steps == list(range(1, max(training_steps)+1))
    cumulative = {0: 0}
    hashes = {str(p): sha(p) for p in manifests + [metrics_path, train_path, val_path]}
    rows = []
    for step in training_steps:
        path = run / f"rollouts/{step}.jsonl"
        scanned = exporter.scan_rollout(path, step)
        assert scanned["complete"], scanned
        for line in path.read_text().splitlines():
            trajectory = json.loads(line)
            assert len(trajectory["response_token_ids"]) == trajectory["response_length"]
            assert len(trajectory["prompt_token_ids"]) == trajectory["prompt_length"]
            assert all(trajectory[k] == 0 for k in ("judge_parse_failed", "judge_http_error", "judge_truncated"))
        flat_metrics = dict(exporter.flatten(metrics[step]))
        logged_tokens = flat_metrics.get("v7__train__prover__generated_tokens_sum")
        if logged_tokens is not None:
            assert scanned["generated_tokens_sum"] == logged_tokens
        cost = 8044544000 * scanned["decode_forwards_sum"] + 589824 * scanned["decode_context_sum"]
        cumulative[step] = cumulative[step-1] + cost
        hashes[str(path)] = sha(path)
        m = metrics[step]
        row = {"step": step, "decoding_flops": cost, "cumulative_decoding_flops": cumulative[step],
               "logged_policy_token_sum_available": logged_tokens is not None}
        if "q/ready_problems" in m:
            count = m["q/ready_problems"]
            assert 0 <= count <= unique_train
            row.update(ready_problems=count, unique_train_problems=unique_train,
                       ready_fraction=count/unique_train, global_gate_open_at_end=int(m["q/global_gate_open"]))
        rows.append(row)
    curve = []
    key = "val-core/imoproofbench/acc/mean@4"
    for step,m in sorted(metrics.items()):
        if key not in m:
            continue
        assert step in cumulative
        path = run / f"val_rollouts/{step}.jsonl"
        raw = [json.loads(line) for line in path.read_text().splitlines()]
        assert all(r["step"] == step for r in raw)
        point = validation(raw, 4, problems, m[key])
        point.update(step=step, cumulative_decoding_flops=cumulative[step])
        curve.append(point)
        hashes[str(path)] = sha(path)
    assert curve and curve[0]["step"] == 0, "initial validation missing"
    return {"run": str(run), "engineering_smoke": arguments["smoke"],
        "match_fields": {"model": arguments["model"], "response": arguments["response"],
                         "groups": arguments["batch"], "samples": arguments["group"],
                         "train_sha256": sha(train_path), "val_sha256": sha(val_path)},
        "curve": curve, "training": rows,
        "first_global_gate_open_end_step": next((r["step"] for r in rows if r.get("global_gate_open_at_end")), None),
        "scope": "Eq.6 policy decoding only; excludes prefill, Q, judge, optimization and validation. Separate per-point bounds assume independent problem-level draw groups, not simultaneous coverage across checkpoints. Scientific peak crossings are point-estimate descriptions, not resolved effects. Smoke validation is only the fixed one-problem fixture.",
        "sha256": hashes}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    args = p.parse_args()
    report = collect(args.run.resolve())
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        assert report["match_fields"] == baseline["match_fields"] and report["engineering_smoke"] == baseline["engineering_smoke"]
        peak = max(p["mean_score"] for p in baseline["curve"])
        report["baseline_comparison"] = {"baseline": str(args.baseline), "observed_peak": peak,
            "first_point_exceeding_observed_peak": next((p for p in report["curve"] if p["mean_score"] > peak), None),
            "scope": "Observed checkpoint grid only; no interpolation or statistical superiority claim."}
        report["sha256"][str(args.baseline)] = sha(args.baseline)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"CURVE_VERIFIED points={len(report['curve'])} steps={len(report['training'])}")


if __name__ == "__main__":
    main()
