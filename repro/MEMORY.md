# Reproduction findings pointers

## Purpose

Latest research/ML engineering amendment: [local surrogate judge](../docs/findings/repro/local-surrogate-judge.md). Richard approved R1 then R2 long with local gpt-oss-20b on GPU0 free memory; S1 passes20/20, API funds frozen. S2/S3 must finish before detached R1; R2 gated on200-step R1 acceptance. No other long arms approved.

Keep durable pointers to project findings, not duplicate investigations.

- Research/ML engineering: [initial CPU test inventory](../docs/findings/repro/cpu-tests.md). Baseline 114 passed, 6 failed; stale context/batch and dashboard tuple expectations.
- Research/data: [E3 data preparation](../docs/findings/repro/data-preparation.md). 5,227/60 rows; canonical derived validation corrects one mirror transcription and restores 60/60 reference coverage.
- ML engineering: [pinned environment](../docs/findings/repro/pinned-environment.md). Exact CUDA 12.9 stack installed; flash-attn kernel forward/backward verified on physical GPUs 1–7. Scratch cap 45 GB.
- Current state and decisions: [CHECKLIST](CHECKLIST.md). Environment and deviations: [ENGINEERING_LEDGER](ENGINEERING_LEDGER.md). Paper quantities: [RESULTS_LEDGER](RESULTS_LEDGER.md).
- Research/ML engineering: [judge calibration](../docs/findings/repro/judge-calibration.md). 20/20 parsing at 40k; conservative durable $5 cap; official V4.1 judge differs from paper. Diagnostic costs are not rollout estimates.
- Research/ML engineering: [local launch](../docs/findings/repro/local-launch.md). Three original builders compose with private Ray and external judge; GPU training acceptance remains separate.
- Research/ML engineering: [multi-GPU smoke](../docs/findings/repro/multigpu-smoke.md). Mechanical one/seven-GPU coverage, uneven microbatch and long-response memory diagnoses; cached retries cannot measure fresh rollout throughput.
- Research/ML engineering: [AC2 smoke](../docs/findings/repro/ac2-smoke.md). E7 six-step critic save/resume and consumed/audit/chunk coverage verified; permissive readiness is an engineering fixture, not a research result.
- Research/ML engineering: [value-probe preparation audit](../docs/findings/repro/value-probe-audit.md). Reference correctness, active bank and pre-wave versus post-checkpoint boundaries need validation; original verify exit status already fails correctly. No R5 result yet.
- Research/ML engineering: [scaled calibration](../docs/findings/repro/scaled-calibration.md). All four E8 states complete; early Q trains16/64 records, so projections include a separate capacity scenario. Final E8/E9 projection and R protocol are in [PROPOSAL_R](PROPOSAL_R.md).
- Research/ML engineering: [model gate](../docs/findings/repro/model-gate.md). E9 complete;4B retained by runtime planning, with format-sensitive scores and broad intervals. [R proposal](PROPOSAL_R.md) fixes protocol/costs; bounded launch preparation underway, long launches unapproved.
- Research/ML engineering: [R launch preparation](../docs/findings/repro/r-launch-preparation.md). All six R1–R4 configs and the complete R5 fixture pass bounded save/resume. Supervisor authorization resolved R5 grading; exact payload verification and current budget are in CHECKLIST/PROPOSAL_R. Long scientific launches remain unapproved; judge cap stays$5.
