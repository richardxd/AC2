"""E4: grade the fixed 20-case paired train/validation prompt fixture."""
import asyncio
import argparse
import json
import time
from pathlib import Path

from ac2.rewards import ds4_finegrained_judge as judge
from ac2.rewards import prover_judge


async def main(args):
    root = Path("runs/e4")
    items = json.loads((root / "candidates.json").read_text())
    assert len(items) == 20
    output = (root / f"grades{args.label}.jsonl").open("x")
    limit = asyncio.Semaphore(4)

    async def grade(item):
        async with limit:
            start = time.monotonic()
            val = item["route"] == "val"
            result = await judge.compute_score(
                data_source="imoproofbench" if val else "fineproofs-rl",
                solution_str="<proof>" + item["proof"] + "</proof>",
                extra_info={"theorem": item["problem"], "mode": "prover", "split": "test" if val else "train"},
                judge_url="http://127.0.0.1:18791/v1",
                val_map_path=str(Path("runs/e3/canonical/val_map.json").resolve()),
                judge_max_tokens=args.max_tokens,
            )
            row = {"route": item["route"], "index": item["index"], "kind": item["kind"],
                   "wall_s": time.monotonic() - start, "result": result}
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(json.dumps({k: row[k] for k in ["route", "index", "kind", "wall_s"]}), flush=True)
            return row

    results = await asyncio.gather(*(grade(x) for x in items))
    output.close()
    await prover_judge._HTTP_SESSION.close()
    success = sum(r["result"]["judge_parse_failed"] == 0 and r["result"]["judge_http_error"] == 0 for r in results)
    summary = {"count": len(results), "parsed": success,
               "truncated": sum(r["result"]["judge_truncated"] for r in results),
               "prompt_tokens": sum(r["result"]["judge_prompt_tokens"] for r in results),
               "completion_tokens": sum(r["result"]["judge_completion_tokens"] for r in results)}
    (root / f"summary{args.label}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    assert success == 20 and summary["truncated"] == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=40000)
    parser.add_argument("--label", default="-40k")
    asyncio.run(main(parser.parse_args()))
