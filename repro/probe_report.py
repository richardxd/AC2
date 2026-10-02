"""Complete-group scaled Fig.4 statistics; invalid Q groups remain explicit exclusions."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def mean(values):
    return sum(values) / len(values)


def stats(pairs):
    if not pairs:
        return {"n": 0, "bias": None, "mae": None, "pearson": None}
    x, y = zip(*pairs)
    mx, my = mean(x), mean(y)
    xx = sum((a-mx)**2 for a in x)
    yy = sum((a-my)**2 for a in y)
    corr = (sum((a-mx)*(b-my) for a, b in pairs) / math.sqrt(xx*yy)
            if min(x) != max(x) and min(y) != max(y) and xx and yy and len(x) >= 3 else None)
    return {"n": len(pairs), "bias": my-mx, "mae": mean([abs(b-a) for a, b in pairs]), "pearson": corr}


def analyze(groups, qrows, jrows, n):
    assert groups and n >= 2
    assert len({g["qid"] for g in groups}) == len(groups)
    qmap = {(r["qid"], r["completion_index"]): r for r in qrows}
    jmap = {(r["qid"], r["completion_index"]): r for r in jrows}
    assert len(qmap) == len(qrows) and len(jmap) == len(jrows), "duplicate measurements"
    expected_j = {(g["qid"], i) for g in groups for i in range(n)}
    expected_q = {(g["qid"], -1) for g in groups} | {
        (g["qid"], i) for g in groups for i, c in enumerate(g["completions"]) if c["exceeds_g"]}
    assert set(jmap) == expected_j and set(qmap) == expected_q, "missing or extraneous measurements"
    for r in jrows:
        assert all(r[k] == 0 for k in ("judge_http_error", "judge_parse_failed", "judge_truncated"))
        assert r["score"] is not None and math.isfinite(r["score"]) and 0 <= r["score"] <= 1
    for r in qrows:
        assert r["q"] is None or (math.isfinite(r["q"]) and 0 <= r["q"] <= 1)
    pairs = {name: [] for name in ("prefix_vs_reward", "hybrid_vs_reward", "matched_cut_vs_reward", "advantages")}
    records, excluded, no_substitution = [], [], []
    for g in groups:
        qid = g["qid"]
        assert len(g["completions"]) == n
        rewards = [jmap[qid, i]["score"] for i in range(n)]
        cut_indices = [i for i, c in enumerate(g["completions"]) if c["exceeds_g"]]
        invalid = [i for i in [-1] + cut_indices if qmap[qid, i]["q"] is None]
        if invalid:
            excluded.append({"qid": qid, "invalid_q_indices": invalid})
            continue
        values = [qmap[qid, i]["q"] if i in cut_indices else rewards[i] for i in range(n)]
        rbar, vbar = mean(rewards), mean(values)
        prefix = qmap[qid, -1]["q"]
        pairs["prefix_vs_reward"].append([rbar, prefix])
        if cut_indices:
            pairs["hybrid_vs_reward"].append([rbar, vbar])
            pairs["advantages"].extend([[r-rbar, v-vbar] for r, v in zip(rewards, values)])
        else:
            no_substitution.append(qid)
        if len(cut_indices) >= 2:
            pairs["matched_cut_vs_reward"].append([
                mean([rewards[i] for i in cut_indices]), mean([values[i] for i in cut_indices])])
        records.append({"qid": qid, "rewards": rewards, "hybrid_values": values,
                        "mean_reward": rbar, "mean_hybrid": vbar, "prefix_q": prefix,
                        "cut_indices": cut_indices, "substitution_fraction": len(cut_indices)/n})
    return {"groups_total": len(groups), "continuations_per_group": n,
        "groups_complete_valid": len(records), "excluded_invalid_q_groups": excluded,
        "no_substitution_groups": no_substitution,
        "statistics": {name: stats(p) for name, p in pairs.items()}, "pairs": pairs, "groups": records,
        "scope": "Descriptive matched statistics. Entire groups with any invalid Q are excluded and listed; never independently filter the two sides. Hybrid/advantages include only groups with substitutions; f=0 groups agree by construction. Matched-cut means require at least two cut samples. Correlation undefined for fewer than three pairs or zero variance. No significance or scientific efficacy claim from an engineering fixture."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = {key: sorted(args.directory.glob(pattern)) for key, pattern in {
        "groups": "gen/gen.shard*.jsonl", "qrows": "q/q.shard*.jsonl", "jrows": "judged/judged.shard*.jsonl"}.items()}
    assert all(paths.values())
    data = {key: [json.loads(line) for path in files for line in path.read_text().splitlines()] for key, files in paths.items()}
    report = analyze(**data, n=args.n)
    report["sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for files in paths.values() for path in files}
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report["statistics"], indent=2))


if __name__ == "__main__":
    main()
