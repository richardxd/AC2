# E9 model gate and protocol limitations

## Purpose

Record the completed matched60×1 diagnostic, its instrument checks and the resulting scaled-protocol choice. [Gate](../../../repro/receipts/e9-model-gate.json), [execution](../../../repro/receipts/e9-execution.json), [format audit](../../../repro/receipts/e9-format-audit.json), [proposal](../../../repro/PROPOSAL_R.md).

Both pinned models generated on physicalGPUs1–7 with identical problem partitions, TP1/concurrency4, response16384 and evaluation0.8/0.95/20, using matched per-problem seeds192+index. Default and explicit thinking templates matched for all60 prompts. Four-B generated919849tokens at191.616tokens/GPU-generation-second;1.7B792389tokens at250.790. Timings include prefill/queueing but exclude engine startup and judging. Successful bounded launches took861.819s and550.009s. The first4B warmup failure was repaired before any outputs and preserved separately.

Four-B mean score0.119047619, nonzero10/60. One-point-seven-B mean0, nonzero0/60. All120 grade records have zero error flags. Actual API calls15+5 cost$0.076600620; no-proof outputs follow the original zero-score short circuit. Total engineering167calls/$0.867471996.

Instrument check: reran original last-thinking-close stripping and first complete proof-tag extraction over every saved response; all proof lengths exactly equal their grade records. Four-B15extractable proofs/45unextractable, with45truncations. One-point-seven-B5extractable/55unextractable,22truncations,33stopped without extractable proof;7responses contain literal proof tags only before their final thinking close. This is a response-format/truncation-sensitive protocol, not a pure mathematical-ability comparison. No changes to formatting, extraction or sampling were made after outputs.

Separate95% bounded-score intervals:4B mean[0,.29438],1.7B[0,.17533]; pairedsmall−large[−.46971,.23161]. Independence of per-problem draws is assumed; n1 cannot identify within-problem variance. No resolved superiority claim follows. Retain4B because the E8-supported Q-capacity scenario is4.89days/200steps,9.79days with an explicit2×allowance; this is close to the approximate10-day criterion and not a runtime guarantee. The1.7B fallback point criteria fail, but are not needed for that branch.

Research role: preserve the fixed protocol and disclose its format and scale limits; do not manufacture a quality verdict. ML role: the selected4B fits the measured planning scenario on seven GPUs, with mature-Q/full-validation uncertainty carried into launch preparation. Long scientific runs remain unapproved.
