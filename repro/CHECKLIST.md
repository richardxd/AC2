# AC2 reproduction checklist

## Purpose

Track every fixed [ROADMAP](ROADMAP.md) acceptance check using first-hand receipts. Long R launches require approval; bounded R preparation smokes are authorized by kickoff revision `466f883`. Started 2026-10-01 (US/Central), base commit `d90aa18`, branch `repro-g16`.

| Task | Status | Acceptance result | Evidence |
|---|---|---|---|
| E1 pinned environment | done | torch 2.11.0+cu129, vllm 0.23.0+cu129, flash_attn 2.8.1, verl importable; 7 GPUs. UUID mapping and real attention forward/backward verified on physical GPUs 1–7 | [acceptance](receipts/e1-acceptance-uuid.json), `runs/e1/`, [environment](env.sh), [finding](../docs/findings/repro/pinned-environment.md) |
| E2 CPU tests | done | 114 passed, 6 failed; complete pass/fail list recorded. One documented expected failure; five additional stale test assumptions, not suppressed | `runs/e2/pytest-02.log`, `runs/e2/results-02.xml`, [test list](receipts/e2-test-list.json), [finding](../docs/findings/repro/cpu-tests.md) |
| E3 data | done | 5,227 train / 60 val rows; raw maps 5,284 rubrics / 59 val references. Derived canonical set has 60/60 references after one explicit mirror correction | [raw receipt](receipts/e3-raw.json), [canonical receipt](receipts/e3-canonical.json), `runs/e3/`, [finding](../docs/findings/repro/data-preparation.md) |
| E4 judge | done | 20/20 parsed at 40k judge output; per-call tokens, latency, conservative USD and projections recorded. Earlier 16k attempt 19/20 retained. Total 40 attempts ≤$0.210647 | [cost receipt](receipts/e4-cost-summary.json), [finding](../docs/findings/repro/judge-calibration.md), `runs/e4/` |
| E5 local adaptation | done | GRPO, AC2 and Prefix GRPO compose; private local Ray head exposes seven GPUs. `.env` and cache paths filled for g16. Training remains E6 | `runs/e5/compose-02.log`, `runs/e5/ray-preflight.log`, [runner](local_runner.py) |
| E6 multi-GPU | in progress | GPU1 1.7B two steps + resume step3 saved; states restored, gradients still zero at4096. Seven-GPU 4B steps1/2 saved, actual world7/TP1/DP7 and load on all physical1–7 verified; resume to3 at16k underway. Earlier 4B one-GPU Adam OOM retained | [resume receipt](receipts/e6-17b-resume-mechanical.json), `runs/e6/nonzero-rejection.log`, `runs/e6/grpo-7gpu-initial-mechanical.json`, `runs/e6/grpo-7gpu-resume01/` |
| E7 AC2 smoke | todo | Critic queries/readiness/chunks/audit/padding; q metrics, optimizer step, chunk bound | Pending |
| E8 calibration | todo | Measured step times, Eq. 6 FLOPs and USD; R projections | Pending |
| E9 model gate | todo | Both models, 60 × 4, 16k; scores, nonzero fraction, decode throughput and gate | Pending |
| R1 GRPO | todo | Requires Richard's launch approval after proposal | Not launched |
| R2 AC2 | todo | Requires Richard's launch approval; score/FLOPs, readiness and crossing | Not launched |
| R3 Prefix GRPO | todo | Requires Richard's launch approval; score/FLOPs | Not launched |
| R4 ablations | todo | Requires Richard's launch approval and budget | Not launched |
| R5 value probe | todo | Requires Richard's launch approval and R2 checkpoint | Not launched |

## Preflight

Read kickoff, roadmap, paper text, root README, install script, metric exporter documentation, and README/runner/launch scripts for all three main arms. Rendered the local PDF with PyMuPDF 1.28.2; inspected pages 8–10, 18–19, 26–27 for Figures 2–4, 7–9 and Tables 3–4. Images and independent text extraction: `runs/paper/`.

Paper SHA256: `e89e2d289db589f0bcd7d576f9f2a82406d9f0188730028604993fff38690421` (29 pages, arXiv v2, 1 Oct 2026).

## Stopping rules

No Slurm operations, GPU0 work, other-project resources, or R runs over 30 minutes. Record blockers and finish every unblocked task, including R launch preparation and bounded save/resume smokes. E8/E9 must support `PROPOSAL_R.md`; missing measurements remain unknown. No idle watchdog is assumed.

## Open questions for Richard

- Resolved guard false positive: supervisor fixed `.codex/guard.py` to exclude URLs from path parsing; the health URL had been misread as an out-of-root path. Supervisor explicitly permits rerunning the read-only batch. No agent guard edits or bypasses; out-of-root deletion protection remains.

- Readiness gate timing: default to reference-bank requirement from step 1 (shipped configuration), disclose difference from paper's step-42 bug. Reversible configuration choice.
- One IMOProofBench mirror problem swaps the excluded vertices in `B' != B` and `C' != C`, versus official DeepMind `B' != C` and `C' != B`. Default: preserve raw E3 files; prepare an explicitly corrected derived evaluation file using the official problem and matching reference, with provenance and hashes. Never silently grade the mismatched problem using that reference.
- CPU stale tests: preserve initial 114/6 report; do not alter training settings to satisfy obsolete test constants. Investigate failures affecting actual smokes as they arise.
- Official API now identifies DeepSeek-V4.1-Flash rather than the paper's pinned V4 revision. Default: disclose this judge deviation and keep the requested official API; do not claim exact score comparability.
- 4B FP32 Adam training exceeds one 48 GB GPU at optimizer allocation even with original parameter/optimizer offload (states are materialized on GPU during update). Default: use 1.7B for E6 one-GPU save/resume, retain 4B for distributed checks; keep numerical precision unchanged. This does not decide E9's scientific model gate.
- First 1.7B smoke was stopped by the monitoring harness when nvidia-smi exceeded its five-second timeout during training. Default: increase telemetry timeout to 30 seconds (still within total <30-minute bound), keep fail-loud behavior and record monitor errors in result receipts; rerun after cleanup review.
- Thirty-second telemetry timeout also interrupted 1.7B. Default revised: run nvidia-smi as an independently owned sampler so a stalled NVML query cannot stop training or block the deadline; retain raw timestamps and require actual per-GPU activity evidence for acceptance. No missing samples are imputed. Review/test sampler cleanup before next training attempt.
- E7 branch coverage: cold readiness requires five steps and a solved-reference bank; 256-token engineering rollouts may never yield proofs. Default: first check ordinary cold critic training, then an explicitly labeled engineering-only fixture with thresholds 1.01 and bank/nonzero gates off, retaining the five-step window and real inference/optimization. Use 32 fixed training rows to revisit problems; never interpret this fixture's readiness as a research result or use these settings in a long R run.
- E9 preregistration revised by supervisor before any model generation: all60 canonical problems, one16384-token sample per model (120 possible judge calls). E4 measured cost supports this (~$0.6 planning estimate); reserve$2 for E7/E8 and R preparation. New hashes in `runs/e9/data60/selection.json`; old12-row proposal retained as superseded evidence. Seven TP1 replicas, fixed index-mod7 partition and matched per-replica concurrency4; report generation timing separately from judge time and retain uncertainty. This is60×1, not the original60×4.
- E6 short rollouts all truncated at 256 tokens, so initial saved optimizer steps have zero task gradients. Default: retain those mechanical checks; resume with 4096 tokens and require a finite nonzero gradient on the resumed step before claiming policy-update coverage. Do not add an artificial entropy objective or interpret short-run zeros as model quality.
- Seven-GPU resume extends response budget from256 to16384 after 1.7B's4096-token check also yielded only truncations. Default: preserve the original short-run receipts and explicitly distinguish changed rollout budget from unchanged optimizer state/objective. No scientific before/after comparison is implied.
