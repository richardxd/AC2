"""Accept a completed bounded R5 fixture only with raw, hash-bound stage evidence."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["prepare", "generate", "critic", "judge", "analyze"]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def collect(directory, launches, completed_only=False):
    from probe_report import analyze
    from transformers import AutoTokenizer
    cfg = read(directory / "configuration.json")
    assert cfg["engineering_fixture"] and cfg["groups"] == cfg["shards"] == 2
    assert cfg["response"] == 16384 and cfg["chunk"] == 4096 and cfg["samples"] == 4
    for name, sha in cfg["pipeline_code_sha256"].items():
        assert digest(Path(name)) == sha, f"pipeline source changed: {name}"
    identities = {r["physical_gpu"]: r["uuid"] for r in read(ROOT / "repro/receipts/e1-acceptance-uuid.json")["kernel_checks"]}
    hashes, results = {}, []
    stages = STAGES[:3] if completed_only else STAGES + ["verify-resume"]
    for stage in stages:
        launch = launches / ("resume01" if stage == "verify-resume" else f"{stage}01")
        result, manifest = read(launch / "result.json"), read(launch / "launch.json")
        assert result["returncode"] == 0 and not result["timed_out"] and not result["monitor_error"]
        assert 0 < manifest["seconds"] <= 1740 and manifest["gpus"] == [str(i) for i in range(1, 8)]
        command = manifest["command"]
        assert (ROOT / command[1]).resolve() == ROOT / "repro/r5_launch.py" and command[2] == stage
        for key in ("run", "data", "out"):
            actual = (ROOT / command[command.index("--" + key) + 1]).resolve()
            assert actual == Path(cfg[key]).resolve()
        assert Path(cfg["out"]).resolve() == directory.resolve()
        for key in ("step", "groups", "shards", "verify_step", "verify_chunk"):
            assert int(command[command.index("--" + key.replace("_", "-")) + 1]) == cfg[key]
        assert "--engineering-fixture" in command
        activity = {}
        if stage in {"generate", "critic"}:
            for row in csv.reader((launch / "gpu.csv").open()):
                stamp, index, uuid, util, memory, power = [x.strip() for x in row]
                gpu = int(index)
                assert gpu in range(1, 8) and uuid.removeprefix("GPU-") == identities[gpu]
                activity[gpu] = activity.get(gpu, 0) + int(float(util) >= 5)
            assert all(activity.get(gpu, 0) >= 3 for gpu in (1, 2)), "missing actual GPU1/2 activity"
        log = (launch / "stdout.log").read_text()
        if stage == "verify-resume":
            assert "RESUME_VERIFIED all five stages; no model generation or judge calls repeated" in log
        else:
            done = read(directory / f"{stage}.done.json")
            assert done["stage"] == stage and done["engineering_fixture"]
            for name, sha in done["sha256"].items():
                assert name not in hashes or hashes[name] == sha
                hashes[name] = sha
        results.append({"stage": stage, "launch": str(launch), "result": result,
                        "active_samples_ge5pct": activity, "log_sha256": digest(launch / "stdout.log")})
    for name, sha in hashes.items():
        assert digest(Path(name)) == sha, f"changed stage artifact: {name}"
    assert "FSDP checks passed" in (directory / "merge-verify.log").read_text()
    context_log = (directory / "verify.log").read_text()
    contexts = int(re.search(r"rows checked\s*:\s*(\d+)", context_log)[1])
    matched = int(re.search(r"exact ctx match\s*:\s*(\d+)", context_log)[1])
    assert contexts > 0 and matched == contexts
    selection = [json.loads(x) for x in (directory / "probe_set.jsonl").read_text().splitlines()]
    selected = {r["qid"]: r for r in selection}
    data = {}
    patterns = {"groups": "gen/*.jsonl", "qrows": "q/*.jsonl"}
    if not completed_only:
        patterns["jrows"] = "judged/*.jsonl"
    for key, pattern in patterns.items():
        paths = sorted(directory.glob(pattern))
        assert paths
        data[key] = [json.loads(x) for p in paths for x in p.read_text().splitlines()]
    assert len(selected) == len(data["groups"]) == 2
    assert {r["qid"] for r in data["groups"]} == set(selected)
    tokenizer = AutoTokenizer.from_pretrained(directory / "model_hf", local_files_only=True)
    generated_tokens = 0
    for group in data["groups"]:
        source = selected[group["qid"]]
        for key in ("row_index", "src_step", "prefix_len", "src_resp_len", "prompt_token_ids", "prefix_token_ids"):
            assert group[key] == source[key], (group["qid"], key)
        for completion in group["completions"]:
            ids = completion["cont_token_ids"]
            assert len(ids) == completion["n_tokens"] <= source["max_new_tokens"]
            assert completion["exceeds_g"] == (len(ids) > cfg["chunk"])
            assert tokenizer.decode(group["prefix_token_ids"] + ids, skip_special_tokens=True) == completion["full_attempt_text"]
            generated_tokens += len(ids)
    if completed_only:
        expected_q = {(g["qid"], -1) for g in data["groups"]} | {
            (g["qid"], i) for g in data["groups"] for i, c in enumerate(g["completions"]) if c["exceeds_g"]}
        actual_q = {(r["qid"], r["completion_index"]) for r in data["qrows"]}
        assert actual_q == expected_q and len(actual_q) == len(data["qrows"])
        assert all(r["q"] is None or (math.isfinite(r["q"]) and 0 <= r["q"] <= 1) for r in data["qrows"])
        assert all(len(g["completions"]) == 4 for g in data["groups"])
        assert not (directory / "judged").exists() and not (launches / "judge01").exists()
        block = read(directory / "judge-approval-block.json")
        assert block["command_started"] is False and block["blocked_stage"] == "judge"
        resume = read(directory / "completed-stage-resume.json")
        assert resume["completed_stages_resumed_without_reexecution"]
        assert resume["full_resume_correctly_rejected_incomplete_pipeline"]
        assert [r["stage"] for r in resume["checks"]] == STAGES[:3] + ["verify-resume"]
        for check in resume["checks"]:
            assert digest(Path(check["log"])) == check["sha256"]
            text = Path(check["log"]).read_text()
            if check["stage"] == "verify-resume":
                assert check["returncode"] != 0 and "pipeline incomplete" in text
            else:
                assert check["returncode"] == 0 and f"RESUME_VERIFIED {check['stage']}; preserved completed outputs" in text
        from local_runner import judge_accounting
        accounting = judge_accounting()
        assert all(r["state"] == "complete" for r in accounting)
        cumulative = sum(r["charged_upper_usd"] for r in accounting)
        assert len(accounting) == resume["judge_calls"] == block["judge_state"][0][1]
        assert math.isclose(cumulative, resume["judge_usd"], abs_tol=1e-12)
        assert math.isclose(cumulative, block["judge_state"][0][2], abs_tol=1e-12) and cumulative <= 5
        return {"directory": str(directory), "engineering_fixture": True, "status": "blocked_before_judge",
                "stages": results, "pending_stages": ["judge", "analyze", "verify-resume"],
                "verified_artifacts": len(hashes), "historical_contexts_exact": contexts,
                "judge_template_pin": cfg["judge_template_pin"], "generated_tokens": generated_tokens,
                "groups": len(data["groups"]), "continuations": sum(len(g["completions"]) for g in data["groups"]),
                "q_measurements": len(data["qrows"]), "valid_q_measurements": sum(r["q"] is not None for r in data["qrows"]),
                "judge_rows": 0, "judge_calls": 0, "judge_usd": 0,
                "cumulative_judge_usd": cumulative, "full_pipeline_verified": False,
                "completed_stage_resume_verified": True,
                "sha256": {str(directory / name): digest(directory / name) for name in
                           ["configuration.json", "judge-approval-block.json", "completed-stage-resume.json"] +
                           [f"{s}.done.json" for s in STAGES[:3]]},
                "scope": "Partial two-group engineering receipt. External grading was denied before execution; scores, correlations, and full pipeline acceptance remain unmeasured."}
    report = read(directory / "report.json")
    regenerated = analyze(**data, n=cfg["samples"])
    assert regenerated == {k: v for k, v in report.items() if k != "sha256"}
    for path, sha in report["sha256"].items():
        assert digest(Path(path)) == sha
    before = {r["id"]: r for r in read(directory / "judge_before.json")}
    after = {r["id"]: r for r in read(directory / "judge_after.json")}
    assert all(after[key] == value for key, value in before.items())
    added = [value for key, value in after.items() if key not in before]
    assert all(r["state"] == "complete" for r in after.values())
    from local_runner import judge_accounting
    assert {r["id"]: r for r in judge_accounting()} == after, "judge calls changed after the judge stage"
    cumulative_usd = sum(r["charged_upper_usd"] for r in after.values())
    assert cumulative_usd <= 5
    return {"directory": str(directory), "engineering_fixture": True, "stages": results,
            "verified_artifacts": len(hashes), "judge_template_pin": cfg["judge_template_pin"],
            "historical_contexts_exact": contexts,
            "generated_tokens": generated_tokens, "groups": len(data["groups"]),
            "q_measurements": len(data["qrows"]), "judge_rows": len(data["jrows"]),
            "groups_complete_valid": report["groups_complete_valid"],
            "excluded_invalid_q_groups": report["excluded_invalid_q_groups"],
            "judge_calls": len(added), "judge_usd": sum(r["charged_upper_usd"] for r in added),
            "cumulative_judge_usd": cumulative_usd, "no_calls_after_judge_stage": True,
            "sha256": {str(directory / name): digest(directory / name) for name in
                       ["configuration.json", "report.json"] + [f"{s}.done.json" for s in STAGES]},
            "scope": "Bounded two-group pipeline/save-resume evidence; no scientific value-quality claim."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--launches", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--completed-only", action="store_true", help="Explicit partial receipt after blocked grading; never certifies the full pipeline")
    args = p.parse_args()
    receipt = collect(args.directory, args.launches, args.completed_only)
    with args.output.open("x") as f:
        json.dump(receipt, f, indent=2)
    label = "R5_PARTIAL_VERIFIED" if args.completed_only else "R5_PIPELINE_VERIFIED"
    print(label + " " + json.dumps({k: receipt[k] for k in
          ("groups", "q_measurements", "judge_rows", "judge_calls", "judge_usd")}))


if __name__ == "__main__":
    main()
