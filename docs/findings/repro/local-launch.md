# Local single-node entry points

## Purpose

E5 preserves the original GRPO, AC2 and Prefix GRPO config builders while replacing cluster launch and colocated judge requirements for g16.

Source `repro/env.sh`, then run `bash repro/compose_dryrun.sh`. All three compose receipts are in `runs/e5/compose-02.log`; configs and arguments have timestamped launch directories. A separate real Ray preflight verified one local head with seven GPUs and shut down its own cluster (`runs/e5/ray-preflight.log`). GPU identity assertions precede actual model work. Ray assigns each worker's visible devices; the parent list is not forced onto workers.

`local_runner.py` pins cached model snapshot directories and verifies weights. The first GPU1 training attempt reached FSDP initialization but vLLM rejected an offline repo snapshot missing documentation files; selecting its existing exact snapshot path fixes that resolution issue. Successful composition makes no claim about rollout, optimizer or checkpoint execution; E6 owns those checks.

External judge changes are limited to three experiment runners: disable local reward-model launch when an explicit URL exists, forward URL to custom reward, and record API identity rather than local model revision. Existing local mode is preserved. API identity is mutable and not a pinned paper judge; see [E4](judge-calibration.md).

Research role: all scaling knobs and source snapshots are saved, cold artifacts come from original builder, no utility backfill. ML role: local Ray, fixed cache placement, eager small smoke, telemetry and deadline, no checkpoint pruning. Smoke defaults (256 response, 128 chunk, group 2) do not constitute a scientific reproduction.
