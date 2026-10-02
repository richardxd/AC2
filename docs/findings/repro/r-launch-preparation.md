# R launch preparation

## Purpose

Track bounded engineering preparation under the [scaled proposal](../../../repro/PROPOSAL_R.md). Long scientific runs remain unapproved. The [checklist](../../../repro/CHECKLIST.md) records the current task state; complete engineering receipts are required before calling a launch ready.

## Judge failure exposed by R1

The first R1 cold launch completed its initial one-problem validation but failed before the first policy update. One official DeepSeek call used all40000 output tokens as reasoning, returned `finish_reason=length`, and supplied no parsed score. The strict adapter raised rather than training on a substituted zero. Preserve `runs/r-smokes/r1` and `runs/r-smoke-launches/r1-cold01`. The [failure receipt](../../../repro/receipts/r1-cold01-failure.json) accounts for27 calls/$0.196956288, including that failure; cumulative spend was$1.064428284.

R preparation now requests65536 judge output tokens,540s client timeout and480s gateway timeout. The gateway still reserves the entire requested upper cost before network I/O and caps cumulative charged/reserved spend at$5. Offline checks cover both supported output budgets, above-limit requests, and near-cap refusal without a credential read or paid call. Attempt2 uses a separate directory, leaving the failed run intact. E4's20/20 success and E8's40k timings remain historical observations; they do not bound future judge lengths or revised runtime.

Research-engineer assessment: failed grades are missing measurements, not valid zero scores. The larger judge allowance is a disclosed protocol deviation, and long-run cost/timing tails remain uncertain. ML-engineer assessment: fail before PPO, retain the failed launch and API receipt, isolate retries, and verify checkpoint restoration before proceeding.

## Evidence checks

R1 attempt2 cold passed in1324.340960s, with finite nonzero actor gradient0.0332254749 and checkpoint1. Its receipt proves world7 and physical1–7 activity; the host process check after cleanup showed only the preexisting GPU0 processes. The curve checker accepted64 training rows, exact initial validation prompts and1.2173036066373632e16 step1 policy decoding FLOPs. Only the initial validation point is present in this cold smoke; it does not demonstrate learning. Evidence: `runs/r-preflight/r1-cold02-verified.json`, `runs/r-preflight/r1-cold02-curve.json`. Save/resume acceptance requires the separate resume receipt.

The training receipt binds the actual launch command to its output directory and verifies world size, physical GPU UUID activity, and all-rank model/optimizer/scheduler restoration. The AC2 checker additionally compares critic optimizer counters with observed applied updates and verifies Q-state cursor, probe records and chunk bounds. Its R-wrapper adaptation reproduced the historical E8 receipt exactly.

The curve collector checks complete training populations (64 GRPO;80 replay including16 scratch inflows), original UID group multiplicities, contiguous steps, original Eq.6 token accounting, exact validation prompts reconstructed from the configured tokenizer/data, and agreement with logged means. It rejects configuration drift across launches. Independent review found and then verified fixes for wrong-problem and partial-population false accepts. Scientific output is a checkpoint-grid description with separate problem-cluster Hoeffding bounds; engineering fixtures are explicitly labeled and cannot establish learning gains.

R5 stages hash source data, checkpoints, replay/reference-bank inputs, selected prompt IDs and six pipeline scripts. The saved configuration includes judge settings. A completed-stage resume verifies hashes and skips completed work; changed artifacts/code must fail. Historical context verification and full merged-weight comparison precede generation. The bounded E7 source is an engineering fixture, not an approved R2 research checkpoint.

## Independent review receipts

- Judge budget, attempt isolation, and48 E7 prompt-ID matches: `runs/reviews/r-judge-attempt-review-lykw_80o/`.
- Curve false-accept counterexamples and corrected positive/negative checks: `runs/reviews/curves-review-wk2_94f6/`, `runs/reviews/curves-fix-review-zmwonwj1/`.
- AC2 checkpoint regression, R5 code pins and failure accounting: `runs/reviews/r-acceptance-review-a2w2lq8a/`.
- Standalone figure layout correction: `runs/reviews/figure-fix-review-l8wvzmxn/`.
