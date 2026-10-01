"""Dump the step-57 probe artifact schemas so a scorer can be written against facts, not guesses.

Prints the field names and join keys of the probe set and the generation, critic and judge
shards, as recorded in the files themselves. Checking them once beats guessing and silently
mis-joining 248 rows onto 256 groups.

    python experiments/08_16_q_probe_step57/inspect_probe.py
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter

P = os.path.dirname(os.path.abspath(__file__))


def head(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def show(name, d, skip_big=True):
    print("   keys: %s" % sorted(d.keys()))
    for k in sorted(d):
        v = d[k]
        if isinstance(v, list):
            print("     %-22s list[%d] %s" % (k, len(v), ("" if skip_big else v[:6])))
        elif isinstance(v, str):
            print("     %-22s str[%d] %r" % (k, len(v), v[:60]))
        else:
            print("     %-22s %s %r" % (k, type(v).__name__, v))


def main():
    ps = os.path.join(P, "probe_set.jsonl")
    rows = [json.loads(l) for l in open(ps, encoding="utf-8") if l.strip()]
    print("== probe_set.jsonl  groups=%d" % len(rows))
    show("probe", rows[0])
    uids = [str(r.get("src_uid")) for r in rows]
    qids = [str(r.get("qid")) for r in rows]
    print("   distinct src_uid=%d  distinct qid=%d" % (len(set(uids)), len(set(qids))))

    for sub in ("gen", "judged", "q"):
        d = os.path.join(P, sub)
        fs = sorted(glob.glob(os.path.join(d, "*.jsonl")))
        print("\n== %s/  jsonl files=%d  (all files=%d)"
              % (sub, len(fs), len(os.listdir(d)) if os.path.isdir(d) else 0))
        if not fs:
            print("   (no jsonl; other files: %s)"
                  % sorted(os.listdir(d))[:4] if os.path.isdir(d) else "   (missing)")
            continue
        r = head(fs[0])
        if r:
            show(sub, r)
        n = 0
        for f in fs:
            with open(f, encoding="utf-8") as fh:
                n += sum(1 for l in fh if l.strip())
        print("   total rows across %d files: %d" % (len(fs), n))

    z5 = os.path.join(P, "zpairs5.csv")
    if os.path.exists(z5):
        lines = [l.strip() for l in open(z5) if l.strip()]
        print("\n== zpairs5.csv rows=%d  first=%s" % (len(lines), lines[0]))

    # the join question: do judged rows carry a key that matches probe_set?
    jf = sorted(glob.glob(os.path.join(P, "judged", "*.jsonl")))
    if jf:
        jr = head(jf[0])
        common = sorted(set(jr.keys()) & set(rows[0].keys()))
        print("\n== keys shared by judged and probe_set: %s" % common)
        for k in ("src_uid", "uid", "qid", "row_index", "group_id"):
            if k in jr:
                vals = Counter()
                with open(jf[0], encoding="utf-8") as fh:
                    for l in fh:
                        if l.strip():
                            vals[str(json.loads(l).get(k))] += 1
                print("   judged[%s]: %d distinct in shard0, e.g. %s"
                      % (k, len(vals), list(vals)[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
