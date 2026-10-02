"""E9 matched generation and grading, with generation timing isolated from judging."""
import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from local_runner import ROOT, REVISIONS, judge_accounting


def write_new(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def generate(args):
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.chat_template import apply_chat_template
    from verl.utils.tokenizer import normalize_token_ids
    from vllm import LLM, SamplingParams, TokensPrompt

    physical = str(args.shard + 1)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == physical and physical in list("1234567")
    expected = subprocess.check_output(["nvidia-smi", "-i", physical, "--query-gpu=uuid",
                                        "--format=csv,noheader"], text=True).strip().removeprefix("GPU-")
    assert torch.cuda.device_count() == 1
    assert str(torch.cuda.get_device_properties(0).uuid) == expected
    model = Path(os.environ["HF_HUB_CACHE"]) / ("models--" + args.model.replace("/", "--")) / "snapshots" / REVISIONS[args.model]
    index = json.loads((model / "model.safetensors.index.json").read_text())
    assert all((model / x).is_file() for x in set(index["weight_map"].values()))
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    source = ROOT / "runs/e9/data60/test.parquet"
    dataset = RLHFDataset(str(source), tokenizer, OmegaConf.create({
        "cache_dir": str(Path(os.environ["TMPDIR"]) / "gate-dataset"),
        "max_prompt_length": 2048, "filter_overlong_prompts": True, "truncation": "error"}))
    assert len(dataset) == 60
    rows = [dataset[i] for i in range(len(dataset)) if i % 7 == args.shard]
    prompts = [normalize_token_ids(apply_chat_template(tokenizer, r["raw_prompt"], tools=None,
                                  add_generation_prompt=True, tokenize=True)) for r in rows]
    for row, ids in zip(rows, prompts):
        assert ids == normalize_token_ids(tokenizer.apply_chat_template(row["raw_prompt"], add_generation_prompt=True,
                                                    tokenize=True, enable_thinking=True, return_dict=False))
        assert len(ids) <= 2048
        # 4B pre-fills <think>; 1.7B emits its thinking tag itself. Preserve each
        # shipped template and verify default mode equals explicit thinking above.
    llm = LLM(model=str(model), tensor_parallel_size=1, dtype="bfloat16", seed=192,
              gpu_memory_utilization=.75, max_model_len=18432, max_num_seqs=4,
              max_num_batched_tokens=4096, enforce_eager=True, enable_prefix_caching=True)
    started_unix = time.time()
    start = time.monotonic()
    outputs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                           [SamplingParams(n=1, temperature=.8, top_p=1., top_k=-1,
                                           max_tokens=16384, seed=192 + row["index"]) for row in rows])
    wall = time.monotonic() - start
    results = []
    for row, prompt, output in zip(rows, prompts, outputs):
        assert len(output.outputs) == 1 and list(output.prompt_token_ids) == prompt
        completion = output.outputs[0]
        results.append({"index": row["index"], "extra_info": row["extra_info"],
                        "data_source": row["data_source"], "prompt_token_ids": prompt,
                        "response_token_ids": list(completion.token_ids), "text": completion.text,
                        "finish_reason": completion.finish_reason})
    tokens = sum(len(r["response_token_ids"]) for r in results)
    write_new(args.out / f"generation-{args.shard}.json", {"model": args.model, "revision": REVISIONS[args.model],
        "gpu_uuid": expected, "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "generation_started_unix": started_unix,
        "generation_wall_s": wall, "generated_tokens": tokens, "tokens_per_s": tokens / wall,
        "timing_scope": "LLM.generate including prefill and queueing; excludes startup and judge",
        "configuration": {"engine_seed": 192, "request_seed": "192 + canonical problem index",
                          "temperature": .8, "top_p": 1., "top_k": -1,
                          "response_budget": 16384, "max_num_seqs": 4, "tp": 1}, "rows": results})
    print(json.dumps({"generated_tokens": tokens, "generation_wall_s": wall, "tokens_per_s": tokens / wall}), flush=True)


async def grade(args):
    from ac2.rewards import ds4_finegrained_judge as judge, prover_judge
    parts = [json.loads((args.out / f"generation-{i}.json").read_text()) for i in range(7)]
    assert all(part["model"] == args.model for part in parts)
    rows = [row for part in parts for row in part["rows"]]
    assert sorted(row["index"] for row in rows) == list(range(60))
    write_new(args.out / "judge_before.json", judge_accounting())
    limit = asyncio.Semaphore(4)
    async def one(row):
        async with limit:
            result = await judge.compute_score(data_source=row["data_source"], solution_str=row["text"],
                extra_info=row["extra_info"], judge_url="http://127.0.0.1:18791/v1",
                val_map_path=str(ROOT / "runs/e3/canonical/val_map.json"), judge_max_tokens=40000)
            record = {"index": row["index"], "result": result}
            with (args.out / "grades.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            return record
    assert not (args.out / "grades.jsonl").exists()
    results = await asyncio.gather(*(one(row) for row in rows))
    if prover_judge._HTTP_SESSION is not None:
        await prover_judge._HTTP_SESSION.close()
    write_new(args.out / "judge_after.json", judge_accounting())
    assert all(r["result"]["judge_parse_failed"] == 0 and r["result"]["judge_http_error"] == 0
               and r["result"]["judge_truncated"] == 0 for r in results)
    scores = [r["result"]["score"] for r in results]
    write_new(args.out / "scores.json", {"n": len(scores), "mean_score": sum(scores) / len(scores),
        "nonzero_fraction": sum(s > 0 for s in scores) / len(scores), "scores": scores,
        "no_proof_count": sum(r["result"]["proof_len_chars"] == 0 for r in results),
        "scope": "preregistered60x1 diagnostic; missing proofs score zero by original reward rule"})


def generate_all(args):
    import sys
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2,3,4,5,6,7"
    children = []
    for shard in range(7):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard + 1))
        log = (args.out / f"worker-{shard}.log").open("x")
        child = subprocess.Popen([sys.executable, __file__, "generate", "--model", args.model,
                                  "--out", str(args.out), "--shard", str(shard)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((child, log))
    codes = []
    for child, log in children:
        codes.append(child.wait())
        log.close()
    assert codes == [0] * 7, codes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["generate", "generate-all", "grade"])
    p.add_argument("--shard", type=int, choices=range(7), default=0)
    p.add_argument("--model", choices=REVISIONS, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out = args.out.resolve()
    assert args.out.is_relative_to(ROOT / "runs/e9")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "generate":
        assert not (args.out / f"generation-{args.shard}.json").exists()
        generate(args)
    elif args.mode == "generate-all":
        generate_all(args)
    else:
        asyncio.run(grade(args))


if __name__ == "__main__":
    main()
