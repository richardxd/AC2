"""Q0 check: regenerate E9 (4B, 60 problems x 1 sample, 16k) through samplers.vs + vs_adapters.

Pass criteria, fixed before running: prompt ids equal E9's for all 60 problems; truncation
count within 8 of E9's; gpt-oss mean score within 6 points of E9's outputs regraded by the
same judge; no judge errors. Token-identical responses and common-prefix lengths are
reported as observations of sampler equivalence (offline LLM in E9 vs AsyncLLM here).

Modes: generate (one GPU shard), generate-all (GPUs 1-7), grade, report.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from samplers.vs import sample_iid
from vs_adapters import VLLMSampler, make_engine, prompt_ids

ROOT = Path(__file__).resolve().parents[1]
E9 = ROOT / "runs/e9/4b-retry"
REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
SAMPLING = dict(n=1, temperature=0.8, top_p=0.95, top_k=20, max_tokens=16384)
ENGINE = dict(tensor_parallel_size=1, dtype="bfloat16", seed=192, gpu_memory_utilization=0.75,
              max_model_len=18432, max_num_seqs=4, max_num_batched_tokens=4096,
              enforce_eager=True, enable_prefix_caching=True)
JUDGE = dict(judge_url="http://127.0.0.1:18801/v1", judge_payload_style="gptoss",
             judge_reasoning_effort="high", judge_max_tokens=65536,
             val_map_path=str(ROOT / "runs/e3/canonical/val_map.json"))


def load_rows(paths, key=None):
    rows = [r for p in paths for r in (json.loads(p.read_text())[key] if key else json.loads(p.read_text()))]
    assert sorted(r["index"] for r in rows) == list(range(60))
    return sorted(rows, key=lambda r: r["index"])


def generate(shard, out):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset

    model = Path(os.environ["HF_HUB_CACHE"]) / "models--Qwen--Qwen3-4B-Thinking-2507" / "snapshots" / REVISION
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    data = RLHFDataset(str(ROOT / "runs/e9/data60/test.parquet"), tok, OmegaConf.create({
        "cache_dir": os.environ["TMPDIR"] + "/vs-e9-dataset", "max_prompt_length": 2048,
        "filter_overlong_prompts": True, "filter_overlong_prompts_workers": 1, "truncation": "error"}))
    e9 = {r["index"]: r for r in load_rows([E9 / f"generation-{i}.json" for i in range(7)], "rows")}
    rows = [data[i] for i in range(len(data)) if i % 7 == shard]

    async def run():
        engine = make_engine(str(model), **ENGINE)

        async def one(row):
            ids = prompt_ids(tok, row["raw_prompt"])
            assert ids == e9[row["index"]]["prompt_token_ids"], row["index"]
            [o] = await sample_iid(VLLMSampler(engine, dict(SAMPLING, seed=192 + row["index"])), ids, 1)
            return {"index": row["index"], "data_source": row["data_source"], "extra_info": row["extra_info"],
                    "response_token_ids": list(o.token_ids), "text": o.text, "finish_reason": o.finish_reason}
        try:
            return await asyncio.gather(*(one(r) for r in rows))
        finally:
            engine.shutdown()

    (out / f"generation-{shard}.json").write_text(json.dumps(asyncio.run(run())))


def generate_all(out):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2,3,4,5,6,7"
    children = [subprocess.Popen([sys.executable, __file__, "generate", "--shard", str(s), "--out", str(out)],
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(s + 1), VLLM_WORKER_MULTIPROC_METHOD="spawn",
                                             VLLM_PORT=str(40000 + 100 * s)),
                                 stdout=(out / f"worker-{s}.log").open("x"), stderr=subprocess.STDOUT)
                for s in range(7)]
    codes = [c.wait() for c in children]
    assert codes == [0] * 7, codes


async def grade(out):
    from ac2.rewards import ds4_finegrained_judge as judge, prover_judge
    rows = (load_rows([out / f"generation-{i}.json" for i in range(7)])
            + load_rows([E9 / f"generation-{i}.json" for i in range(7)], "rows"))
    limit = asyncio.Semaphore(16)

    async def one(row):
        async with limit:
            return await judge.compute_score(data_source=row["data_source"], solution_str=row["text"],
                                             extra_info=row["extra_info"], **JUDGE)
    results = await asyncio.gather(*(one(r) for r in rows))
    await prover_judge._HTTP_SESSION.close()
    (out / "grades.json").write_text(json.dumps({"new": results[:60], "e9": results[60:]}))


def report(out):
    new = load_rows([out / f"generation-{i}.json" for i in range(7)])
    old = load_rows([E9 / f"generation-{i}.json" for i in range(7)], "rows")
    g = json.loads((out / "grades.json").read_text())

    def stats(rows, grades):
        return {"truncated": sum(r["finish_reason"] == "length" for r in rows),
                "proofs": sum(x["proof_len_chars"] > 0 for x in grades),
                "mean_score": sum(x["score"] for x in grades) / 60,
                "judge_errors": sum(x["judge_parse_failed"] + x["judge_http_error"] + x["judge_truncated"]
                                    for x in grades)}

    def common(a, b):
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    a, b = stats(new, g["new"]), stats(old, g["e9"])
    prefix = sorted(common(n["response_token_ids"], o["response_token_ids"]) for n, o in zip(new, old))
    checks = {"prompt ids equal E9 (60/60, asserted in generate)": True,
              "truncation within 8 of E9": abs(a["truncated"] - b["truncated"]) <= 8,
              "gpt-oss mean within 6 points of E9": abs(a["mean_score"] - b["mean_score"]) <= 0.06,
              "no judge errors": a["judge_errors"] == 0 and b["judge_errors"] == 0}
    lines = ["# Q0: E9 regenerated through samplers.vs", "",
             "| | new (samplers.vs + AsyncLLM) | E9 (offline LLM) |", "|---|---|---|"]
    lines += [f"| {k} | {a[k]:.4f} | {b[k]:.4f} |" if k == "mean_score" else f"| {k} | {a[k]} | {b[k]} |" for k in a]
    lines += ["", "| check | result |", "|---|---|"] + [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in checks.items()]
    lines += ["", "Observations (not pass criteria):",
              f"- token-identical responses: {sum(n['response_token_ids'] == o['response_token_ids'] for n, o in zip(new, old))}/60",
              f"- common-prefix tokens with E9: median {prefix[30]}, min {prefix[0]}, max {prefix[-1]}"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["generate", "generate-all", "grade", "report"])
    p.add_argument("--shard", type=int, choices=range(7))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "generate":
        generate(args.shard, args.out)
    elif args.mode == "generate-all":
        generate_all(args.out)
    elif args.mode == "grade":
        asyncio.run(grade(args.out))
    else:
        report(args.out)


if __name__ == "__main__":
    main()
