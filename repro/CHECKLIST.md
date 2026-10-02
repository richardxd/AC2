# AC2 reproduction checklist

## Purpose

Track every fixed [ROADMAP](ROADMAP.md) acceptance check using first-hand receipts. Long R launches require approval; bounded R preparation smokes are authorized by kickoff revision `466f883`. Started 2026-10-01 (US/Central), base commit `d90aa18`, branch `repro-g16`.

| Task | Status | Acceptance result | Evidence |
|---|---|---|---|
| E1 pinned environment | done | torch 2.11.0+cu129, vllm 0.23.0+cu129, flash_attn 2.8.1, verl importable; 7 GPUs. Real attention forward/backward verified on GPUs 1–7 | [acceptance](receipts/e1-acceptance.json), `runs/e1/`, [environment](env.sh), [finding](../docs/findings/repro/pinned-environment.md) |
| E2 CPU tests | done | 114 passed, 6 failed; complete pass/fail list recorded. One documented expected failure; five additional stale test assumptions, not suppressed | `runs/e2/pytest-02.log`, `runs/e2/results-02.xml`, [test list](receipts/e2-test-list.json), [finding](../docs/findings/repro/cpu-tests.md) |
| E3 data | done | 5,227 train / 60 val rows; raw maps 5,284 rubrics / 59 val references. Derived canonical set has 60/60 references after one explicit mirror correction | [raw receipt](receipts/e3-raw.json), [canonical receipt](receipts/e3-canonical.json), `runs/e3/`, [finding](../docs/findings/repro/data-preparation.md) |
| E4 judge | in progress | Gateway offline budget tests 3/3 passed; model API confirms V4.1-Flash, balance $32.43. Fixed 20 paired candidates prepared | `runs/e4/`, `repro/judge_gateway.py` |
| E5 local adaptation | todo | Local Ray, g16 `.env`, GRPO and AC2 config composition | Pending |
| E6 multi-GPU | todo | 1 then 7 GPUs, 2 steps, save/resume; world size, TP/DP, per-GPU load, continued step/loss | Pending |
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

- Readiness gate timing: default to reference-bank requirement from step 1 (shipped configuration), disclose difference from paper's step-42 bug. Reversible configuration choice.
- One IMOProofBench mirror problem swaps the excluded vertices in `B' != B` and `C' != C`, versus official DeepMind `B' != C` and `C' != B`. Default: preserve raw E3 files; prepare an explicitly corrected derived evaluation file using the official problem and matching reference, with provenance and hashes. Never silently grade the mismatched problem using that reference.
- CPU stale tests: preserve initial 114/6 report; do not alter training settings to satisfy obsolete test constants. Investigate failures affecting actual smokes as they arise.
