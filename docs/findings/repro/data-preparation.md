# E3 data preparation and explicit validation correction

## Purpose

Record the source data, output hashes, and a validation mirror discrepancy before measuring scores.

The original `prepare_fineproofs --split both` and `build_rubric_map` produced 5,227 train rows and 60 val rows in `runs/e3/data/`. [Raw receipt](../../../repro/receipts/e3-raw.json) hashes every output. Training rubric lookup has 5,225 unique keys from 5,227 nonempty rubrics. The raw validation map has only 59 entries.

The missing row is zero-based index 51: the HF mirror states `B' != B` and `C' != C`, while the official [DeepMind CSV](https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/proofbench_v2.csv) states `B' != C` and `C' != B`. This is more than whitespace drift. Grading the uncorrected statement against the official reference would silently change the evaluation semantics.

Default chosen under the overnight decision rule: retain raw files and create `runs/e3/canonical/` through [prepare_eval_data.py](../../../repro/prepare_eval_data.py). It applies exactly those two substitutions to identify the official statement, requires an exact normalized match, reconstructs its prompt with the existing prover template and retains its original index. It does not use fuzzy matching. It asserts one correction, 60 rows, 60 val-map entries and nonempty reference/guidelines for all entries. [Canonical receipt](../../../repro/receipts/e3-canonical.json) includes before/after text, source and output hashes.

Research assessment: all 60 canonical problems remain, but this is a disclosed deviation from the mirrored paper input. Engineering assessment: val-map coverage is now complete and explicit. The legacy combined rubric file is preserved; evaluation uses the separate corrected `val_map.json`.

Related: [CPU baseline](cpu-tests.md), [engineering ledger](../../../repro/ENGINEERING_LEDGER.md).
