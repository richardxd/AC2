"""E9 matched benchmark summaries with explicit bounded-score uncertainty."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def bound(mean, n, low=0., high=1.):
    margin = (high - low) * math.sqrt(math.log(40) / (2 * n))
    return [max(low, mean - margin), min(high, mean + margin)]


def summarize(directory):
    files = [directory / f"generation-{i}.json" for i in range(7)]
    files += [directory / "grades.jsonl", directory / "judge_before.json", directory / "judge_after.json"]
    parts = [json.loads(path.read_text()) for path in files[:7]]
    assert len({p["model"] for p in parts}) == 1 and len({p["data_sha256"] for p in parts}) == 1
    generated = {row["index"]: row for part in parts for row in part["rows"]}
    grades = [json.loads(line) for line in (directory / "grades.jsonl").read_text().splitlines()]
    assert len(grades) == 60 and len(generated) == 60
    assert sorted(r["index"] for r in grades) == list(range(60))
    scores = {}
    for row in grades:
        result = row["result"]
        assert all(result[key] == 0 for key in ["judge_parse_failed", "judge_http_error", "judge_truncated"])
        value = result["score"]
        assert 0 <= value <= 1
        scores[row["index"]] = value
    mean = sum(scores.values()) / 60
    nonzero = sum(v > 0 for v in scores.values()) / 60
    tokens = sum(len(row["response_token_ids"]) for row in generated.values())
    assert tokens == sum(part["generated_tokens"] for part in parts)
    before = {r["id"]: r for r in json.loads((directory / "judge_before.json").read_text())}
    after = {r["id"]: r for r in json.loads((directory / "judge_after.json").read_text())}
    assert all(after[k] == v for k, v in before.items()), "overlapping judge activity"
    calls = [v for k, v in after.items() if k not in before]
    return {"model": parts[0]["model"], "revision": parts[0]["revision"], "data_sha256": parts[0]["data_sha256"],
            "n_problems": 60, "samples_per_problem": 1, "mean_score": mean,
            "mean_score_95pct_hoeffding": bound(mean, 60), "nonzero_fraction": nonzero,
            "nonzero_95pct_hoeffding": bound(nonzero, 60), "scores_by_index": scores,
            "generated_tokens": tokens,
            "tokens_per_gpu_generation_second": tokens / sum(part["generation_wall_s"] for part in parts),
            "replicas": [{k: part[k] for k in ["gpu_uuid", "generation_wall_s", "generated_tokens", "tokens_per_s"]} for part in parts],
            "truncated_responses": sum(row["finish_reason"] == "length" for row in generated.values()),
            "judge_attempts": len(calls), "judge_charged_upper_usd": sum(r["charged_upper_usd"] for r in calls),
            "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--four-b", type=Path, required=True)
    p.add_argument("--small", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    large, small = summarize(args.four_b), summarize(args.small)
    assert large["model"] == "Qwen/Qwen3-4B-Thinking-2507"
    assert small["model"] == "Qwen/Qwen3-1.7B"
    assert large["data_sha256"] == small["data_sha256"]
    differences = [small["scores_by_index"][i] - large["scores_by_index"][i] for i in range(60)]
    half = [small["scores_by_index"][i] - .5 * large["scores_by_index"][i] for i in range(60)]
    result = {"4b": large, "1.7b": small,
              "paired_small_minus_large": {"mean": sum(differences) / 60,
                  "95pct_hoeffding": bound(sum(differences) / 60, 60, -1., 1.)},
              "paired_small_minus_half_large": {"mean": sum(half) / 60,
                  "95pct_hoeffding": bound(sum(half) / 60, 60, -.5, 1.)},
              "uncertainty_scope": "Each interval is a separate conservative95% bound assuming independent per-problem random draws. One sample/problem cannot estimate within-problem variance; these are not simultaneous confidence bounds. No model-quality superiority follows from overlapping bounds.",
              "timing_scope": "Per-GPU generation throughput includes prefill/queueing, excludes model startup and judge. Both use seven TP1 replicas with matched partition and concurrency4.",
              "gate": "Combine with E8 projection: retain4B if scaled run<=10days; otherwise evaluate1.7B half-score/nonzero>=0.20 criteria, retaining unresolved uncertainty."}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: {m: result[k][m] for m in ["mean_score", "nonzero_fraction", "tokens_per_gpu_generation_second"]}
                      for k in ["4b", "1.7b"]}))


if __name__ == "__main__":
    main()
