# Official judge adapter and diagnostic calibration

## Purpose

E4 checks train/val prompt transport, parsing and spending instrumentation before training. It does not validate judge accuracy or estimate a representative generated-proof cost.

The gateway at `127.0.0.1:18791` adapts the existing reward payload to official DeepSeek thinking API. Key is read only into the authorization header, never receipts. A fixed repository SQLite ledger reserves worst-case cost before network I/O; one process holds an exclusive lock. Every retry by the existing reward layer is separately reserved. Failed/uncertain calls retain the reserve. Three offline tests cover arithmetic, durable cap rejection before key I/O, and exclusive gateway ownership.

`runs/e4/candidates.json` fixes the first ten canonical validation problems, alternating official reference proof and unsupported assertion, reused under both prompt templates (20 calls). Initial 16k output yielded 19/20 parsed; one reference evaluation truncated. Repeating all 20 at paper 40k output yielded 20/20 parsed, no truncation, 29,496 prompt and 82,907 completion tokens. Raw original failures remain. Grades of references vary, so parser success does not imply grader correctness.

Tracked [receipt](../../../repro/receipts/e4-cost-summary.json) contains per-call latency, tokens, conservative peak-price cost, raw receipt path and SHA256. Final train means: 26.335 seconds / $0.006651983; val: 13.608 seconds / $0.003429097. Train latency range 2.383–94.987 s; val 1.805–44.536 s. Total upper cost across all 40 attempts $0.210646032. Provider rounded balance moved $32.43→$32.33; official holiday/off-peak prices may halve the peak bounds. API `/models` reports DeepSeek-V4.1-Flash, not paper revision `60d8d707`.

Planning arithmetic only: GRPO 32×8 calls × train mean = $1.70291/step; unready AC2 (32 refill + 32×8) = $1.91577/step; validation 60×4 × val mean = $0.82298/pass. For 200 steps and 21 validations: $357.86417 / $400.43686 respectively. Actual short-budget rollouts may contain no extracted proof and incur no judge call; ready AC2 call counts also differ. These fixtures cannot support confidence intervals for a generated-proof population. E8 calibration is required before choosing a protocol.

Research role: preserve fixed inputs and both accepted/rejected proof types; disclose changed judge and diagnostic sampling. ML role: fail-closed cap, durable receipts, bounded concurrency four, accounting of failed attempts. Both require distinguishing upper cost from actual bill and parsing from correctness.
