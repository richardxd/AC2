# Scaled throughput calibration

## Purpose

Measure the g16 candidate before proposing long R runs. E8 is engineering calibration; E9 supplies the matched step-zero validation gate. Neither is a reproduction of the paper's training curves.

The candidate in [preregistration](../../../repro/E8_PREREGISTRATION.md) uses pinned4B weights, physical GPUs1–7, TP1/DP7,16 training groups×4 continuations, response16384, chunk4096 and maximum64 Q records. Replay methods add16 fresh inflow trajectories. Normal readiness and correctness-gated references remain enabled. Validation is disabled for these bounded training measurements and separately measured in E9.

## Measured components

| State | New policy tokens | Eq.6 decode FLOPs | Generation incl. judge (s) | Actor/interleaved Q (s) | Checkpoint (s) | Whole step (s) | API calls / USD |
|---|---|---|---|---|---|---|---|
| GRPO step1 | 926086 | 11707487528091648 | 485.316 | 225.208 | 78.219 | 875.976 | 21 / .143165772 |
| Cold AC2 step1 | 1225030 | 15621752438456320 | 575.348 | 226.187 | 207.842 | 1114.992 | 24 / .129811104 |
| Resumed AC2 step2 | 840875 | 11208923499528192 | 418.303 | 273.787 | 111.313 | 916.102 | 31 / .130034916 |
| Prefix GRPO step1 | 1244762 | 15937332632027136 | 601.690 | 220.703 | 75.812 | 1010.848 | 21 / .129121776 |

Receipts: [GRPO](../../../repro/receipts/e8-grpo-calibration.json), [cold AC2](../../../repro/receipts/e8-ac2-cold-calibration.json), [resumed AC2](../../../repro/receipts/e8-ac2-replay-calibration.json). [Prefix](../../../repro/receipts/e8-prefix-calibration.json). Launch wall times, including initialization and cleanup, are1016.688,1250.823,1105.154 and1140.573seconds respectively.

All four launches completed with nonzero actor gradients. The [AC2 checkpoint receipt](../../../repro/receipts/e8-ac2-critic-checkpoint.json) verifies physical GPU activity, world7, all-rank restoration,16 valid Q probes and one critic optimizer step on every rank. Readiness remained closed. Additional actor/checkpoint receipts: `runs/e8/grpo-hardware-checkpoint.json`, `runs/e8/ac2-hardware-checkpoint.json`, `runs/e8/prefix-hardware-checkpoint.json`.

## Interpretation limits

Eq.6 is computed from joint trajectory moments, with prefix tokens included in attention context but excluded from newly generated-token counts. It excludes prefill, critic queries, optimization, judging and validation compute. Generation wall time includes queueing and judging; it is not isolated kernel decode throughput.

The states and sampled problems differ. The decrease in new tokens on resumed AC2 is not evidence of learned critic efficiency: readiness is still closed, and replay supplies existing prefixes. There is only one step per state, so no measured runtime variance, confidence interval or model-quality conclusion follows. Training scores are not evaluation scores.

The early Q FIFO has16 records and trains all16, below the configured maximum64. Planning therefore includes a separate linear capacity scenario that multiplies the whole actor/Q phase by4. This deliberately also scales fixed PPO work because isolated Q duration is unavailable. It is not a measured saturated batch or a hard bound; reference lengths, reference/no-reference mixture and mature readiness can change cost. The2×planning allowance is an explicit margin, not a confidence interval.

Long-run projections retain60×4 validation every10steps plus step0, checkpoint every20steps,200steps, and all judge costs. E9's60×1 gate is extrapolated linearly for validation planning and uses the original evaluation sampling0.8/0.95/20. R5 has separate checkpoint-merge and query overhead. The final proposal awaits both E9 model receipts; no long run is authorized by these measurements.
