"""Pre-register a matched, reduced E9 sample without observing model outputs."""
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "runs/e3/canonical"
    out = root / "runs/e9/data60"
    out.mkdir(parents=True, exist_ok=False)
    val = pd.read_parquet(source / "test.parquet")
    assert len(val) == 60
    indices = list(range(60))
    val.iloc[indices].to_parquet(out / "test.parquet", index=False)
    for name in ("train.parquet", "rubric_map.json", "val_map.json"):
        shutil.copyfile(source / name, out / name)
    receipt = {"selection": "all 60 canonical validation problems, supervisor revision before any model output",
               "source_indices": indices, "samples_per_problem_per_model": 1,
               "response_budget": 16384, "models": ["Qwen/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-1.7B"],
               "reason": "E4 observed cost supports all60x1 per model; reserve2 USD for E7/E8 and launch preparation",
               "metrics": ["mean score (points/7)", "fraction with score > 0", "generated tokens/decoding wall second"],
               "gate": "4B if measured scaled projection <=10 days; otherwise 1.7B only if mean >=half 4B and nonzero>=0.20; insufficient evidence leaves choice provisional",
               "limits": "Reduced samples60x1, not original60x4; report binomial uncertainty and paired differences, no strong conclusion from unresolved bounds",
               "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [source / "test.parquet", *sorted(out.iterdir())]}}
    (out / "selection.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
