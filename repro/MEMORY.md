# Reproduction findings pointers

## Purpose

Keep durable pointers to project findings, not duplicate investigations.

- Research/ML engineering: [initial CPU test inventory](../docs/findings/repro/cpu-tests.md). Baseline 114 passed, 6 failed; stale context/batch and dashboard tuple expectations.
- Research/data: [E3 data preparation](../docs/findings/repro/data-preparation.md). 5,227/60 rows; canonical derived validation corrects one mirror transcription and restores 60/60 reference coverage.
- ML engineering: [pinned environment](../docs/findings/repro/pinned-environment.md). Exact CUDA 12.9 stack installed; flash-attn kernel forward/backward verified on physical GPUs 1–7. Scratch cap 45 GB.
- Current state and decisions: [CHECKLIST](CHECKLIST.md). Environment and deviations: [ENGINEERING_LEDGER](ENGINEERING_LEDGER.md). Paper quantities: [RESULTS_LEDGER](RESULTS_LEDGER.md).
