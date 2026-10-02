"""Create a derived E3 dataset with one explicit IMO mirror correction.

Raw outputs remain in runs/e3/data. This is an exact, audited correction, not
fuzzy reference matching. Fail if the upstream discrepancy changes.
"""
import csv
import hashlib
import io
import json
import shutil
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ac2.data.build_rubric_map import PROOFBENCH_CSV_URL, _key, _norm
from ac2.data.prepare_fineproofs import _load_template, to_verl_record


def main():
    raw = Path("runs/e3/data")
    out = Path("runs/e3/canonical")
    out.mkdir(exist_ok=False)
    csv_bytes = urllib.request.urlopen(PROOFBENCH_CSV_URL, timeout=60).read()
    (out / "proofbench_v2.csv").write_bytes(csv_bytes)
    official = list(csv.DictReader(io.StringIO(csv_bytes.decode())))
    by_problem = {_norm(r["Problem"]): r for r in official}
    rows = pq.read_table(raw / "test.parquet").to_pylist()
    raw_map = json.loads((raw / "val_map.json").read_text())
    changes = []
    val_map = {}
    for i, row in enumerate(rows):
        problem = row["extra_info"]["theorem"]
        if _key(problem) in raw_map:
            val_map[_key(problem)] = raw_map[_key(problem)]
            continue
        assert problem.count(r"B'\neq B") == 1
        assert problem.count(r"C'\neq C") == 1
        corrected = problem.replace(r"B'\neq B", r"B'\neq C").replace(r"C'\neq C", r"C'\neq B")
        match = by_problem[_norm(corrected)]
        corrected = match["Problem"]
        rows[i] = to_verl_record(corrected, row["extra_info"]["index"], "test", "imoproofbench", _load_template())
        val_map[_key(corrected)] = {"solution": match["Solution"].strip(), "guidelines": match["Grading guidelines"].strip()}
        changes.append({"row": i, "before": problem, "after": corrected})
    assert len(changes) == 1 and len(rows) == len(val_map) == 60
    assert all(x["solution"] and x["guidelines"] for x in val_map.values())
    pq.write_table(pa.Table.from_pylist(rows), out / "test.parquet")
    shutil.copy2(raw / "train.parquet", out / "train.parquet")
    shutil.copy2(raw / "rubric_map.json", out / "rubric_map.json")
    (out / "val_map.json").write_text(json.dumps(val_map, indent=2))
    receipt = {"source_url": PROOFBENCH_CSV_URL, "changes": changes,
               "train_rows": pq.read_table(out / "train.parquet").num_rows,
               "val_rows": len(rows), "val_map_entries": len(val_map),
               "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
