"""Measure R smoke judge costs, separating resume-only steps from cold validation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("r1", "r2", "r3", "r4-2k", "r4-correct-only", "r4-no-audit")


def collect():
    from local_runner import judge_accounting
    hashes, used_ids, rows = {}, set(), []
    previous_after = None

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for task in TASKS:
        run = ROOT / "runs/r-smokes" / ("r1-attempt2" if task == "r1" else task)
        curve = read(ROOT / f"repro/receipts/{task}-smoke-curve.json")
        assert Path(curve["run"]) == run and curve["engineering_smoke"]
        assert [r["step"] for r in curve["training"]] == [1, 2]
        launches = sorted((run / "launches").glob("*/arguments.json"))
        assert len(launches) == 2
        for step, argument_file in enumerate(launches, 1):
            args = read(argument_file)
            assert args["steps"] == step and args["initial_val"] == (step == 1)
            assert args["judge_max_tokens"] == 65536 and args["strict_judge"]
            assert args["val_freq"] > step and not args["val_only"] and not args["compose_only"]
            assert args["batch"] == 16 and args["group"] == 4 and args["response"] == 16384
            assert (run / f"checkpoints/global_step_{step}").is_dir()
            before = {r["id"]: r for r in read(argument_file.parent / "judge_before.json")}
            after = {r["id"]: r for r in read(argument_file.parent / "judge_after.json")}
            assert previous_after is None or before == previous_after, "accounting gap between serialized launches"
            previous_after = after
            assert all(after[key] == value for key, value in before.items())
            assert all(r["state"] == "complete" for r in after.values())
            added = [r for key, r in after.items() if key not in before]
            assert added, "smoke step has no judge accounting"
            total, output_tokens, max_output = 0., 0, 0
            for call in added:
                assert call["id"] not in used_ids
                used_ids.add(call["id"])
                receipt = read(Path(call["receipt"]))
                assert receipt["id"] == call["id"] and receipt["http_status"] == 200
                request = receipt["request"]
                assert request["model"] == "deepseek-flash" and request["max_tokens"] == 65536
                assert all(c["finish_reason"] == "stop" for c in receipt["response"]["choices"])
                usage, prices = receipt["response"]["usage"], receipt["prices_per_million"]
                hit, prompt, completion = usage["prompt_cache_hit_tokens"], usage["prompt_tokens"], usage["completion_tokens"]
                assert 0 <= hit <= prompt and 0 <= completion <= 65536
                cost = ((prompt-hit)*prices["input_miss"] + hit*prices["input_hit"] + completion*prices["output"]) / 1e6
                assert math.isclose(cost, receipt["cost_upper_usd"], rel_tol=0, abs_tol=1e-12)
                assert math.isclose(cost, call["charged_upper_usd"], rel_tol=0, abs_tol=1e-12)
                total += cost
                output_tokens += completion
                max_output = max(max_output, completion)
            rows.append({"task": task, "step": step, "launch": str(argument_file.parent),
                         "includes_initial_validation": step == 1, "judge_calls": len(added),
                         "judge_usd": total, "completion_tokens": output_tokens,
                         "max_call_completion_tokens": max_output,
                         "scope": "one training step plus initial problem0 x4 validation" if step == 1 else "one resumed training step; no validation"})
    accounting = judge_accounting()
    assert all(r["state"] == "complete" for r in accounting)
    assert previous_after == {r["id"]: r for r in read(ROOT / "runs/r5-smoke/judge_before.json")}
    assert {r["id"]: r for r in accounting} == {r["id"]: r for r in read(ROOT / "runs/r5-smoke/judge_after.json")}
    spend = sum(r["charged_upper_usd"] for r in accounting)
    assert spend <= 5
    r5 = read(ROOT / "repro/receipts/r5-full-pipeline.json")
    assert math.isclose(spend, r5["cumulative_judge_usd"], rel_tol=0, abs_tol=1e-12)
    resumed = [r for r in rows if not r["includes_initial_validation"]]
    return {"judge_max_tokens": 65536, "rows": rows,
            "resume_only_steps": len(resumed),
            "resume_only_total_usd": sum(r["judge_usd"] for r in resumed),
            "resume_only_pooled_usd_per_step": sum(r["judge_usd"] for r in resumed)/len(resumed),
            "completed_judge_calls": len(accounting), "completed_spend_usd": spend,
            "total_cap_usd": 5, "remaining_cap_usd": 5-spend, "sha256": hashes,
            "limitations": "One resumed early-state step per configuration; no estimate of mature readiness, later proof extraction, long reasoning tails, or full60 x4 validation. Cold rows include initial validation and are not training-only rates. Historical failed 40000-token R1 attempt excluded from these rates but retained in cumulative spend."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect()
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in {"sha256", "limitations"}}))


if __name__ == "__main__":
    main()
