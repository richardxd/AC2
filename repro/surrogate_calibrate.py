"""S1/S2: unchanged prompt calibration, with zero external API requests."""
import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path

import aiohttp
from surrogate_common import ROOT, URL, MAX_TOKENS, EFFORT, CONCURRENCY, HTTP_TIMEOUT, profile, route, accounting
from local_runner import judge_accounting
from ac2.rewards import ds4_finegrained_judge as judge, prover_judge as pj


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def statistics(rows):
    import pandas as pd
    valid = [r for r in rows if r["deepseek_points"] is not None and r["surrogate_points"] is not None
             and not r["deepseek_truncated"] and not r["surrogate_truncated"]]
    x, y = [r["deepseek_points"] for r in valid], [r["surrogate_points"] for r in valid]
    n = len(x)
    if not n:
        return {"total": len(rows), "paired": 0, "excluded_ids": [r["id"] for r in rows]}
    a, b = [int(v >= 6) for v in x], [int(v >= 6) for v in y]
    observed = sum(u == v for u, v in zip(a, b))/n
    pa, pb = sum(a)/n, sum(b)/n
    expected = pa*pb + (1-pa)*(1-pb)
    exact = sum(u == v for u, v in zip(x, y))/n
    expected_points = sum(x.count(point)*y.count(point) for point in set(x+y))/(n*n)
    rho = float(pd.Series(x).rank(method="average").corr(pd.Series(y).rank(method="average"))) if len(set(x)) > 1 and len(set(y)) > 1 else None
    return {"total": len(rows), "paired": n,
            "excluded_ids": [r["id"] for r in rows if r not in valid],
            "exact_points_agreement": exact,
            "points_cohen_kappa": (exact-expected_points)/(1-expected_points) if expected_points < 1 else None,
            "pass_agreement": observed, "pass_cohen_kappa": (observed-expected)/(1-expected) if expected < 1 else None,
            "mean_absolute_point_difference": sum(abs(u-v) for u, v in zip(x, y))/n,
            "spearman": rho, "deepseek_pass_fraction": pa, "surrogate_pass_fraction": pb}


async def main(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    before, local_before = judge_accounting(), accounting()
    write(out / "configuration.json", profile())
    write(out / "api_before.json", before)
    started = time.monotonic()
    limit = asyncio.Semaphore(CONCURRENCY)
    if args.stage == "e4":
        candidates = json.loads((ROOT / "runs/e4/candidates.json").read_text())
        assert len(candidates) == 20
        async def one(item):
            async with limit:
                start = time.monotonic()
                result = await judge.compute_score(
                    data_source="imoproofbench" if item["route"] == "val" else "fineproofs-rl",
                    solution_str="<proof>"+item["proof"]+"</proof>",
                    extra_info={"theorem": item["problem"], "mode": "prover", "split": "test" if item["route"] == "val" else "train"},
                    judge_url=URL, judge_payload_style="gptoss", judge_reasoning_effort=EFFORT,
                    judge_max_tokens=MAX_TOKENS, val_map_path=str(ROOT / "runs/e3/canonical/val_map.json"))
                row = {"route": item["route"], "index": item["index"], "kind": item["kind"],
                       "wall_s": time.monotonic()-start, "result": result}
                write(out / f"{item['route']}-{item['index']}-{item['kind']}.json", row)
                print(json.dumps({k:v for k,v in row.items() if k != "result"}), flush=True)
                return row
        rows = await asyncio.gather(*(one(item) for item in candidates))
        if pj._HTTP_SESSION is not None:
            await pj._HTTP_SESSION.close()
        success = sum(all(row["result"][key] == 0 for key in ("judge_http_error", "judge_parse_failed", "judge_truncated")) for row in rows)
        summary = {"count": 20, "successful": success,
                   "completion_tokens": sum(r["result"]["judge_completion_tokens"] for r in rows),
                   "mean_latency_s": sum(r["wall_s"] for r in rows)/len(rows)}
    else:
        assert len(before) == 547 and all(r["state"] == "complete" for r in before)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async def one(call):
                original = json.loads(Path(call["receipt"]).read_text())
                prompt = original["request"]["messages"][0]["content"]
                payload = pj._chat_payload(prompt, max_tokens=MAX_TOKENS, temperature=1., top_p=1., top_k=-1,
                                           reasoning_effort=EFFORT, seed=42, payload_style="gptoss")
                async with limit:
                    start = time.monotonic()
                    async with session.post(URL+"/chat/completions", json=payload) as response:
                        response.raise_for_status()
                        data = await response.json()
                    elapsed = time.monotonic()-start
                row = {"id": call["id"], "route": route(prompt), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "original_receipt": call["receipt"], "original_sha256": hashlib.sha256(Path(call["receipt"]).read_bytes()).hexdigest(),
                       "deepseek_points": judge._parse_points(pj._extract_content(original["response"]) or ""),
                       "surrogate_points": judge._parse_points(pj._extract_content(data) or ""),
                       "deepseek_truncated": pj._is_truncated(original["response"]),
                       "surrogate_truncated": pj._is_truncated(data), "latency_s": elapsed,
                       "usage": data["usage"], "response": data}
                write(out / (call["id"]+".json"), row)
                print(json.dumps({k:row[k] for k in ("id", "route", "deepseek_points", "surrogate_points", "latency_s")}), flush=True)
                return row
            rows = await asyncio.gather(*(one(call) for call in before))
        summary = {"count": len(rows), "unique_prompts": len({r["prompt_sha256"] for r in rows}),
                   "completion_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
                   "by_route": {kind: statistics([r for r in rows if r["route"] == kind]) for kind in ("train", "val")},
                   "scope": "Every historical API call regraded, including repeated prompts; agreement weighted by calls, stratified by original prompt. Invalid/truncated pairs excluded explicitly, never zero-filled. Agreement is descriptive, not a gate or independent efficacy evidence."}
    elapsed = time.monotonic()-started
    assert judge_accounting() == before, "external API accounting changed"
    previous = {r["id"] for r in local_before}
    calls = [r for r in accounting() if r["id"] not in previous]
    assert len(calls) == len(rows) and all(r["status"] == 200 and r["finished"] is not None for r in calls)
    write(out / "local_calls.json", calls)
    summary.update(elapsed_s=elapsed, output_tokens_per_wall_s=summary["completion_tokens"]/elapsed,
                   api_calls_added=0, profile=profile())
    write(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if args.stage == "e4":
        assert summary["successful"] == 20, "S1 parse acceptance failed"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["e4", "agreement"])
    p.add_argument("--output", type=Path, required=True)
    asyncio.run(main(p.parse_args()))
