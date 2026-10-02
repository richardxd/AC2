#!/usr/bin/env python3
"""One vLLM engine: generate this replica's shard of the probe's 16-rollout groups.

STAGE A of the step-40 probe. Takes randomly-truncated prefixes on ready problems and
continues each one 16 times to the run's own 50k response cap -- WITHOUT the short lane's
cut at prefix+10k. That cut is the whole point: in the live run a rollout that reaches
prefix+10k has its reward replaced by Q's answer and its true reward is never observed.
Here the rollout runs on, so the true reward becomes observable and Q's answer can be
scored against it.

WHY TP=1. The actor is a 4B model, so a whole engine fits on one GPU and 32 independent
replicas cover the allocation. That is not just faster than a big TP shape -- it sidesteps
the TP=8 cudagraph/flashinfer all-reduce deadlock entirely, because with TP=1 there is no
all-reduce to fuse.

WHY n=16 RATHER THAN 16 PRE-EXPANDED REQUESTS. All 16 siblings share the identical
prompt+prefix, so one request with n=16 lets the engine prefill that prefix once. At a
median prefix of ~8k tokens over 256 groups that is the difference between ~2M and ~33M
prefill tokens.

WHAT IS STORED, AND WHY IT IS TRUNCATED. Full continuation TEXT is kept (the judge needs
it). Continuation TOKEN IDS are kept only to `--keep-ids`, because the sole consumer of the
ids is the Q call at prefix+10k, which by construction never reads past 10k. Storing all 16
full id lists would multiply the dump ~5x for bytes nothing reads.

    python probe_gen.py --model <hf> --probe-set probe_set.jsonl \
        --shard K --num-shards N --out gen.shard<K>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe-set", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)   # SP_TEMPERATURE
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--keep-ids", type=int, default=10500,
                    help="continuation ids retained per rollout; must exceed budget-g so the "
                         "Q cut is always inside what was kept")
    ap.add_argument("--budget-g", type=int, default=10000)
    # this vLLM's ModelConfig types `seed` as int and rejects None outright (pydantic
    # ValidationError at engine construction). Default to the
    # shard index: an int, reproducible, and different per replica so 32 engines do not draw
    # correlated samples.
    ap.add_argument("--seed", type=int, default=-1,
                    help="engine seed; -1 means use the shard index")
    args = ap.parse_args()
    if args.seed < 0:
        args.seed = args.shard

    rows = [json.loads(l) for l in open(args.probe_set, encoding="utf-8")]
    mine = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    if not mine:
        print("[replica %d] no work" % args.shard, flush=True)
        open(args.out, "w").close()
        return 0
    print("[replica %d] %d groups, %d rollouts, prefix tokens %s"
          % (args.shard, len(mine), len(mine) * args.n,
             "{:,}".format(sum(r["prefix_len"] for r in mine))), flush=True)

    from vllm import LLM, SamplingParams, TokensPrompt

    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype="bfloat16",
        enable_prefix_caching=True,      # 16 siblings share the whole prompt+prefix
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        seed=args.seed,
    )
    print("[replica %d] engine up in %.1fs" % (args.shard, time.time() - t0), flush=True)
    tokenizer = llm.get_tokenizer()

    prompts, sps = [], []
    for r in mine:
        ids = list(r["prompt_token_ids"]) + list(r["prefix_token_ids"])
        # never let prompt+prefix+budget exceed the engine window
        budget = int(r["max_new_tokens"])
        assert budget > 0 and len(ids) + budget <= args.max_model_len, "context would alter response budget"
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        sps.append(SamplingParams(temperature=args.temperature, top_p=1.0, top_k=-1,
                                  max_tokens=budget, n=args.n, seed=None))

    t1 = time.time()
    outs = llm.generate(prompts, sps)
    wall = time.time() - t1

    n_seq = n_over = 0
    with open(args.out, "x", encoding="utf-8") as fh:
        for r, o in zip(mine, outs):
            comps = []
            for c in o.outputs:
                tid = list(c.token_ids)
                n_seq += 1
                over = len(tid) > args.budget_g
                n_over += int(over)
                comps.append({
                    "text": c.text,
                    "full_attempt_text": tokenizer.decode(list(r["prefix_token_ids"]) + tid,
                                                          skip_special_tokens=True),
                    "n_tokens": len(tid),
                    "exceeds_g": over,
                    "finish_reason": getattr(c, "finish_reason", None),
                    # only what the Q cut can read; see docstring
                    "cont_token_ids": tid[:args.keep_ids],
                })
            fh.write(json.dumps({
                "qid": r["qid"], "row_index": r["row_index"], "src_step": r["src_step"],
                "prefix_len": r["prefix_len"], "src_resp_len": r["src_resp_len"],
                "prompt_token_ids": r["prompt_token_ids"],
                "prefix_token_ids": r["prefix_token_ids"],
                "completions": comps,
            }) + "\n")

    print("[replica %d] DONE %d seqs in %.1fs (%.1f min); %d/%d exceed g=%d (%.1f%%)"
          % (args.shard, n_seq, wall, wall / 60.0, n_over, n_seq, args.budget_g,
             100.0 * n_over / max(1, n_seq)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
