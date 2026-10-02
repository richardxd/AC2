# Reproduction findings pointers

## Purpose

Keep durable pointers to project findings, not duplicate investigations.

- Research/ML engineering: [initial CPU test inventory](../docs/findings/repro/cpu-tests.md). Baseline 114 passed, 6 failed; stale context/batch and dashboard tuple expectations.
- Research/data: [E3 data preparation](../docs/findings/repro/data-preparation.md). 5,227/60 rows; canonical derived validation corrects one mirror transcription and restores 60/60 reference coverage.
- ML engineering: [pinned environment](../docs/findings/repro/pinned-environment.md). Exact CUDA 12.9 stack installed; flash-attn kernel forward/backward verified on physical GPUs 1–7. Scratch cap 45 GB.
- Current state and decisions: [CHECKLIST](CHECKLIST.md). Environment and deviations: [ENGINEERING_LEDGER](ENGINEERING_LEDGER.md). Paper quantities: [RESULTS_LEDGER](RESULTS_LEDGER.md).
- Research/ML engineering: [judge calibration](../docs/findings/repro/judge-calibration.md). 20/20 parsing at 40k; conservative durable $5 cap; official V4.1 judge differs from paper. Diagnostic costs are not rollout estimates.
- Research/ML engineering: [local launch](../docs/findings/repro/local-launch.md). Three original builders compose with private Ray and external judge; GPU training acceptance remains separate.
- Research/ML engineering: [multi-GPU smoke](../docs/findings/repro/multigpu-smoke.md). Mechanical one/seven-GPU coverage, uneven microbatch and long-response memory diagnoses; cached retries cannot measure fresh rollout throughput.
