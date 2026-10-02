# Initial g16 CPU test inventory

## Purpose

Record E2's baseline pass/fail list before reproduction code changes. Source revision `466f883` has unchanged training/tests relative to `d90aa18`.

Command: project venv `python -m pytest tests experiments/08_13_tiedq_seed192/test_08_04_logic.py experiments/08_11_ablation1_replay_noq/test_ablation1_logic.py experiments/08_28_extreme_offpolicy/test_08_28_logic.py -v --tb=short --junitxml=runs/e2/results-02.xml`.

All three experiment test files were enumerated with `rg --files`; 120 tests ran: **114 passed, 6 failed**. See [machine-readable pass/fail list](../../../repro/receipts/e2-test-list.json). Raw logs and XML: `runs/e2/`. Initial collection needed `pypdf`; installed pypdf 6.19.0 in the project venv and reran the complete suite.

| Failure | First-hand diagnosis |
|---|---|
| `test_context_group_moves_together` | Expects response 75000, shipped scripts use 50000. Roadmap's expected failure. |
| `test_single_factor_diff_against_the_main_run` | Expects `SP_TRAIN_BATCH_SIZE=256`, shipped Prefix GRPO launcher uses 384, as does main AC2. README already documents it. |
| `test_panel1_mask_covers_burst_run_zero_inflow_steps` | Test unpacks 4 return values; dashboard `_inflow_trained_lane_series` returns 5 including `biased`. |
| `test_panel1_mask_unchanged_on_every_step_inflow_runs` | Same tuple arity mismatch. |
| `test_panel1_mask_leaves_grafted_parent_steps_alone` | Same tuple arity mismatch. |
| `test_panel1_mask_absent_without_sp_inflow_key` | Expects four None values, implementation returns five. |

These are observed test-contract mismatches, not evidence that end-to-end training works. None are hidden or rewritten as passing. Research-engineer assessment: changing launch constants to satisfy tests would alter the paper setup. ML-engineer assessment: retain baseline failures and validate real training separately through E6/E7 and R smokes.

Related: [engineering ledger](../../../repro/ENGINEERING_LEDGER.md), [checklist](../../../repro/CHECKLIST.md).
