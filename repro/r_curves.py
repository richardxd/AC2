"""R1–R4 curves from complete raw rollouts; no cached or imputed measurements."""
import argparse
from collections import Counter, defaultdict
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


def validation(rows, samples, problems, logged_mean, expected_inputs=None):
    assert len(rows) == samples * problems
    groups = defaultdict(list)
    for row in rows:
        assert all(row[k] == 0 for k in ("judge_parse_failed", "judge_http_error", "judge_truncated"))
        score = row["acc"]
        assert math.isfinite(score) and 0 <= score <= 1
        groups[row["input"]].append(score)
    assert len(groups) == problems and all(len(v) == samples for v in groups.values())
    if expected_inputs is not None:
        assert len(set(expected_inputs)) == problems
        assert set(groups) == set(expected_inputs), "validation problems differ from configured dataset"
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
    manifests = [p for p in sorted((run / "launches").glob("*/config.yaml"))
                 if not json.loads((p.parent / "arguments.json").read_text())["compose_only"]]
    assert manifests
    cfg = OmegaConf.load(manifests[-1])
    arguments = json.loads((manifests[-1].parent / "arguments.json").read_text())
    varying = {"steps", "initial_val", "compose_only"}
    fixed = {k:v for k,v in arguments.items() if k not in varying}
    for manifest in manifests:
        prior = json.loads((manifest.parent / "arguments.json").read_text())
        assert {k:v for k,v in prior.items() if k not in varying} == fixed, "run configuration changed"
    assert arguments["model"] == "Qwen/Qwen3-4B-Thinking-2507" and arguments["val_n"] == 4
    backend = arguments.get("judge_backend", "deepseek")
    assert backend in {"deepseek", "surrogate"}
    expected_reward = "surrogate_reward.py" if backend == "surrogate" else "strict_reward.py"
    assert cfg.reward.custom_reward_function.path.endswith("/repro/" + expected_reward)
    judge_profile = None
    if backend == "surrogate":
        from surrogate_common import profile
        judge_profile = profile()
        for manifest in manifests:
            assert json.loads((manifest.parent / "surrogate_profile.json").read_text()) == judge_profile
            reward = OmegaConf.load(manifest).reward.custom_reward_function
            assert reward.path.endswith("/repro/surrogate_reward.py")
            for key in ("judge_url", "judge_payload_style", "judge_reasoning_effort", "judge_max_tokens"):
                name = {"judge_url": "url", "judge_payload_style": "payload_style",
                        "judge_reasoning_effort": "reasoning_effort", "judge_max_tokens": "max_tokens"}[key]
                assert reward.reward_kwargs[key] == judge_profile[name]
    def one_file(value):
        if isinstance(value, str):
            return Path(value)
        assert len(value) == 1
        return Path(value[0])
    train_path, val_path = one_file(cfg.data.train_files), one_file(cfg.data.val_files)
    train = pd.read_parquet(train_path)
    unique_train = len({qid_from_messages(x) for x in train["prompt"]})
    val = pd.read_parquet(val_path)
    problems = len(val)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
    template_kwargs = dict(cfg.data.get("apply_chat_template_kwargs", {}))
    expected_inputs = [tokenizer.decode(tokenizer.apply_chat_template(list(prompt),
        add_generation_prompt=True, **template_kwargs), skip_special_tokens=True) for prompt in val["prompt"]]
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
        trajectories = [json.loads(line) for line in path.read_text().splitlines()]
        expected_rows = arguments["batch"] * (arguments["group"] + (arguments["method"] != "grpo"))
        assert len(trajectories) == expected_rows, "incomplete training rollout population"
        expected_groups = Counter({arguments["group"]: arguments["batch"]})
        if arguments["method"] != "grpo":
            expected_groups[1] += arguments["batch"]
        assert Counter(Counter(r["uid"] for r in trajectories).values()) == expected_groups
        for trajectory in trajectories:
            assert trajectory["step"] == step
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
        point = validation(raw, 4, problems, m[key], expected_inputs)
        point.update(step=step, cumulative_decoding_flops=cumulative[step])
        curve.append(point)
        hashes[str(path)] = sha(path)
    assert curve and curve[0]["step"] == 0, "initial validation missing"
    return {"run": str(run), "engineering_smoke": arguments["smoke"],
        "match_fields": {"model": arguments["model"], "response": arguments["response"],
                         "groups": arguments["batch"], "samples": arguments["group"],
                         "train_sha256": sha(train_path), "val_sha256": sha(val_path),
                         **({"judge_profile": judge_profile} if judge_profile is not None else {})},
        "curve": curve, "training": rows,
        "first_global_gate_open_end_step": next((r["step"] for r in rows if r.get("global_gate_open_at_end")), None),
        "scope": "Eq.6 policy decoding only; excludes prefill, Q, judge, optimization and validation. Separate per-point bounds assume independent problem-level draw groups, not simultaneous coverage across checkpoints. Scientific peak crossings are point-estimate descriptions, not resolved effects. Smoke validation is only the fixed one-problem fixture.",
        "sha256": hashes}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--figure", type=Path, help="Optional PNG of verified score/FLOPs points")
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
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        series = [(report, "Current run")]
        if args.baseline:
            series.insert(0, (baseline, "GRPO baseline"))
        for data, label in series:
            points = data["curve"]
            ax.plot([p["cumulative_decoding_flops"]/1e18 for p in points],
                    [100*p["mean_score"] for p in points], "o-", label=label)
        ax.set(xlabel="Cumulative policy decoding FLOPs (×10¹⁸)",
               ylabel="Validation mean score (%)", ylim=(0, 100),
               title="Engineering fixture — no efficacy claim" if report["engineering_smoke"] else "Observed validation checkpoints")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout(rect=(0, .065, 1, 1))
        fig.text(.01, .01, "Point estimates; separate uncertainty bounds and provenance are in the JSON receipt.", fontsize=7)
        with args.figure.open("xb") as f:
            fig.savefig(f, format="png", dpi=180)
        plt.close(fig)
    print(f"CURVE_VERIFIED points={len(report['curve'])} steps={len(report['training'])}")


if __name__ == "__main__":
    main()
