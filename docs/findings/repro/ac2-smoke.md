# AC2 critic and routing smoke

## Purpose

Verify the engineering path on seven physical GPUs1–7 without interpreting short rollouts or permissive readiness as research results. See [multi-GPU foundation](multigpu-smoke.md) and [checklist](../../../repro/CHECKLIST.md).

## Failure and minimal correction

The first cold run reached its first synthetic Q wave on step2 and failed with `KeyError: data_source`. The asynchronous agent worker attempted proof grading because `SPQAgentLoop` returned no `reward_score`; synthetic Q requests intentionally lack proof-grader metadata. The driver separately parses their four-token value responses. Set the parse-only output's `reward_score=0.0` to skip unrelated asynchronous grading. This is an asynchronous scoring sentinel, not an invented value estimate. Original Q response tokens, parsing, targets, losses and gradients remain unchanged.

Commit `e303081`; two regression tests run the actual loop with a fake inference response and verify that synthetic Q bypasses the proof judge while ordinary missing-score outputs still invoke it. Independent review reproduced both. Failed evidence is preserved in `runs/e7/ac2-7gpu-cold01/`.

## Cold and resumed coverage

The replacement run `runs/e7/ac2-7gpu-retry/` uses the first32 canonical training rows,16 groups×2 samples,16 one-sample inflow trajectories,256-token response and128-token chunk. It starts with empty replay, Q FIFO and reference bank. Cold step1 correctly skips Q training; cold step2 takes a nonzero Q optimizer step on all seven ranks and saves complete actor and critic optimizer state. Cold launch exit0,595.041seconds; checkpoint2 Q optimizer counter1 on all37 FSDP parameter states/rank. Pre-clipping gradient208.90636, applied delta0.15664919,16 valid probes.

The resumed engineering fixture changes readiness thresholds to1.01 and disables bank/nonzero requirements. It retains the five-step window, real value generation and real optimizer steps. Ordinary short rollouts cannot provide solved reference proofs reliably. This fixture is confined to `--smoke --method ac2` and must never enter a scientific R configuration. At global step5 (dataset step4), the fixture's global gate opens and16 problems become ready. Step6 produces six valid consumed-Q queries and one valid audit-cut query. All six consumed chunks are exactly128 tokens. Final readiness state has23 ready problems.

The resume exits0 in1055.900seconds with checkpoint6 complete. [Acceptance receipt](../../../repro/receipts/e7-ac2-seven-gpu.json) verifies model/optimizer/scheduler restoration on every rank, all seven Q-state restoration messages, actual critic movement on steps2–6, and optimizer counter5 on all37 parameter states/rank. It binds raw log/rollout/query hashes and physical GPU UUID/activity samples. Maximum telemetry gap42.42seconds is retained; no samples are imputed. Step6 is340.604seconds including211.528seconds checkpoint I/O. Six steps comprise80 valid probes,6 valid consumed calls and1 valid audit-cut call. No proof-judge calls were added; gateway total remains50 calls/$0.258737808 upper cost.

## Interpretation

Research role: these are software coverage results. The small repeated problem set, short response horizon and permissive readiness invalidate comparisons with the paper's readiness timing or proof scores. ML role: use state continuity, actual nonzero critic movement, masked DP padding and physical-GPU telemetry to validate the implementation. No inference about model quality follows from short-run rewards.
