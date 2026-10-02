"""E8: retain measured timings and exact Eq.6 trajectory costs, never impute gaps."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("exporter", ROOT / "scripts/export_paper_metrics.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.manifest / "config.yaml")
    assert Path(cfg.trainer.default_local_dir).resolve() == args.run.resolve() / "checkpoints"
    assert args.manifest.resolve().is_relative_to(args.run.resolve() / "launches")
    architecture = Path(cfg.actor_rollout_ref.model.path) / "config.json"
    model = json.loads(architecture.read_text())
    layers, width = model["num_hidden_layers"], model["hidden_size"]
    qdim = model["num_attention_heads"] * model["head_dim"]
    kvdim = model["num_key_value_heads"] * model["head_dim"]
    a = 2 * layers * (2 * width * qdim + 2 * width * kvdim + 3 * width * model["intermediate_size"]) + 2 * width * model["vocab_size"]
    b = 4 * layers * qdim
    report = exporter.export_run(ROOT, args.run.name, override=args.run.resolve(), train_only=True)
    scanned = {row["step"]: row for row in report["train"]["per_step"]}
    metrics = [json.loads(line) for line in (args.manifest / "metrics_added.jsonl").read_text().splitlines()]
    assert metrics and len({m["step"] for m in metrics}) == len(metrics)
    selected = {m["step"] for m in metrics}
    assert not selected.intersection(report["train"]["token_sum_mismatch_steps"]), "rollout/log token totals disagree"
    assert not selected.intersection(report["train"]["missing_or_incomplete_steps"]), "missing trajectory data"
    measured = []
    for record in metrics:
        step, data = record["step"], record["data"]
        row = scanned[step]
        assert row["complete"], row
        q_metrics = {k: v for k, v in data.items() if k.startswith("q/")}
        nonfinite_q = [k for k, v in q_metrics.items() if isinstance(v, (float, int)) and not math.isfinite(v)]
        for key in nonfinite_q:
            q_metrics[key] = None
        measured.append({"step": step, "generated_tokens": row["generated_tokens_sum"],
                         "decode_forwards": row["decode_forwards_sum"],
                         "decode_context_sum": row["decode_context_sum"],
                         "decoding_flops": a * row["decode_forwards_sum"] + b * row["decode_context_sum"],
                         "timing_s": {k: v for k, v in data.items() if k.startswith("timing_s/")},
                         "end_to_end_rollout_tokens_per_s": row["generated_tokens_sum"] / data["timing_s/gen"],
                         "score_mean": data["critic/score/mean"], "actor_grad_norm": data["actor/grad_norm"],
                         "q_metrics": q_metrics, "nonfinite_q_metrics": nonfinite_q})
    before = {r["id"]: r for r in json.loads((args.manifest / "judge_before.json").read_text())}
    after = {r["id"]: r for r in json.loads((args.manifest / "judge_after.json").read_text())}
    assert all(after[k] == v for k, v in before.items()), "overlapping judge activity changed earlier calls"
    calls = [v for k, v in after.items() if k not in before]
    paths = [architecture, args.manifest / "config.yaml", args.manifest / "metrics_added.jsonl",
             args.manifest / "judge_before.json", args.manifest / "judge_after.json"]
    paths += [args.run / row["source"] for row in scanned.values() if row["step"] in {m["step"] for m in measured}]
    result = {"run": str(args.run.resolve()), "manifest": str(args.manifest.resolve()),
              "eq6_A": a, "eq6_B": b, "steps": measured, "judge_calls": calls,
              "judge_charged_upper_usd": sum(r["charged_upper_usd"] for r in calls),
              "scope": "training decode only; excludes prefill/Q/judge/update/validation FLOPs. Rollout timing includes judge/queue overhead, not pure decode throughput. No extrapolation in this receipt.",
              "sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({"A": a, "B": b, "steps": [s["step"] for s in measured],
                      "judge_upper_usd": result["judge_charged_upper_usd"]}))


if __name__ == "__main__":
    main()
