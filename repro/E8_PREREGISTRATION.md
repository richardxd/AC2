# Throughput calibration plan

## Purpose

Fix the candidate and interpretation before measuring E8. This is engineering calibration, not approval of a long R run.

Use Qwen3-4B-Thinking-2507 at the pinned revision, seven physical GPUs1–7, FSDP world7, TP1/DP7, BF16 and the original fused torch output kernel. Candidate: response16384, chunk4096, 16 training groups ×4 samples, and for replay methods16 fresh inflow trajectories (one sample each). AC2 therefore generates at most80 policy trajectories per unready step; GRPO generates64. AC2 Q minibatch64, FIFO1920, replay bound256; readiness thresholds0.20/0.18, bank/nonzero gates on. Q minimum valid group size4. No permissive E7 fixture in these measurements.

The ROADMAP starting point32×8 generates256 training trajectories; E6's measured141.3-second actor update for32 trajectories plus startup, generation and checkpointing makes that an unsuitable first bounded calibration. Halving groups and samples reduces training to64 trajectories. This changes group statistics and problem visitation; it is not an equivalent scientific scale. Measure fresh GRPO, cold AC2 and AC2 with populated replay/Q, and Prefix GRPO. Each launch is limited to25minutes; preserve failures and shrink only with a recorded amendment.

Read exact policy trajectories with `export_paper_metrics.py` and Eq.6. Reject cached generation, missing trajectories and logged-token mismatches. Report generation including its judge/queue overhead separately from actor/Q update, checkpoint, and launch overhead. Judge clients are serialized; before/after gateway ledgers attribute calls only under that contract.

Proposal projections will use measured components with explicit assumptions for200steps, checkpoint/validation frequency and R4/R5 costs. One or two steps cannot estimate mature readiness speedups or a runtime confidence interval. Use an unready/full-horizon planning envelope, and report both observed estimates and conservative resource bounds. Long-run judge cost may exceed the current$5 engineering cap; no long run or higher cap is authorized by this plan. Keep at least$2 reserved for remaining engineering until E9's matched60×1 grading is accounted for.

R validation planning retains the ROADMAP starting point of60problems×4samples per event. E9's budget-authorized60×1 reduction applies to the gate; it does not by itself justify reducing the later scientific validation. Scale E9 observed validation cost/time by4 with an explicit linear extrapolation caveat. The earlier unused projection helper's1-sample placeholder was corrected before any real R proposal. R launch approval must cover these costs; engineering smoke subsets will be documented separately.

Research role: reduced horizons, groups, Q batch and validation sampling must be disclosed; no performance conclusion from calibration. ML role: preserve all-rank state/gradient checks and fit every launch under30minutes without GPU0.

## Timeout amendment before populated AC2

The first GRPO candidate's fresh generation/judging occupied approximately8minutes after model startup, followed by long-response actor training. Populated AC2 also restores critic state, generates16 inflow trajectories, and trains Q. Permit its single-step resume up to1740seconds (29minutes), plus the launcher's bounded owned-process cleanup. Independent safety review confirms a nominal1765-second maximum including polling/cleanup, leaving35seconds below30minutes; this cannot guarantee timing against an OS stall. Cold launches retain1500seconds. This changes only the execution deadline, not inputs or protocol. Exact measured component times will replace this preliminary timing observation in the receipt; no rate is inferred from this amendment.
