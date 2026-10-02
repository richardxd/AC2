"""Verify both accepted and rejected replay inflows against raw judge outcomes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def verify(run, steps):
    from omegaconf import OmegaConf
    files = [run / "replay_buffer_deltas.jsonl"]
    configs = sorted((run / "launches").glob("*/config.yaml"))
    assert configs
    for path in configs:
        cfg = OmegaConf.load(path)
        assert cfg.data.sp_replay_admission == "judged_correct"
        assert cfg.reward.custom_reward_function.reward_kwargs.pass_points_min == 6
    files += configs
    deltas = [json.loads(x) for x in files[0].read_text().splitlines()]
    assert [d["dataset_step"] for d in deltas] == list(range(steps))
    results = []
    for d in deltas:
        path = run / f"rollouts/{d['dataset_step'] + 1}.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        files.append(path)
        counts = Counter(r["uid"] for r in rows)
        assert len(rows) == 80 and Counter(counts.values()) == {1: 16, 4: 16}
        inflow = {r["uid"]: r for r in rows if counts[r["uid"]] == 1}
        for r in inflow.values():
            assert r["sp_prefix_len"] == 0
            assert all(r[k] == 0 for k in ("judge_http_error", "judge_parse_failed", "judge_truncated"))
            assert r["prover_judge_score"] in (0, 1)
            assert bool(r["prover_judge_score"]) == (r["rubric_points"] >= 6)
        eligible = {uid for uid, r in inflow.items() if r["prover_judge_score"] == 1 and r["response_token_ids"]}
        admitted = []
        for e in d["added_entries"]:
            kind, step, uid, index = e["entry_id"].split(":")
            assert kind == "online" and int(step) == d["dataset_step"] and int(index) >= 0
            assert uid in inflow
            r = inflow[uid]
            assert e["response_token_ids"] == r["response_token_ids"]
            assert e["meta"]["judge_pass"] == e["meta"]["judge_score"] == r["prover_judge_score"] == 1
            admitted.append(uid)
        assert len(admitted) == len(set(admitted)) and set(admitted) == eligible
        results.append({"step": d["dataset_step"] + 1, "inflow": len(inflow),
                        "admitted": len(admitted), "rejected": len(inflow) - len(admitted),
                        "admitted_points": [inflow[u]["rubric_points"] for u in admitted]})
    assert sum(r["admitted"] for r in results) > 0 and sum(r["rejected"] for r in results) > 0
    return {"run": str(run), "steps": results, "scope": "engineering admission check; no efficacy claim",
            "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = verify(args.run, args.steps)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print("ADMISSION_VERIFIED " + json.dumps(result["steps"]))


if __name__ == "__main__":
    main()
