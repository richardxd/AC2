# Single-node training and resume on GPUs 1–7

## Purpose

Record E6 execution evidence, distinguishing state-machine coverage from nonzero policy updates. See [checklist](../../../repro/CHECKLIST.md) for current acceptance and [local adaptation](local-launch.md) for configuration.

## Evidence so far

All paths below are under `runs/e6/`.

| Launch | Result | Interpretation |
|---|---|---|
| `grpo-1gpu-attempt02` | 4B Adam allocation OOM on physical GPU1 | FP32 optimizer states exceed one48GB GPU in this configuration |
| `grpo-1gpu-17b-attempt03` | Steps1/2 saved,383.651s | World1/TP1/DP1 and physicalGPU1 activity; all256-token outputs truncated, zero gradients |
| `grpo-1gpu-17b-resume01` | Model/optimizer/RNG/scheduler restored, step3 saved,307.907s | Mechanical continuation;4096-token responses still truncated, zero gradients |
| `grpo-7gpu-attempt01` | Steps1/2 saved,516.841s | NCCL world7, seven TP1 replicas, physical1–7 activity; zero gradients at256 tokens |
| `grpo-7gpu-resume01` |16k generation/judging completed; actor microbatch assertion,522.286s | Synchronized split count incompatible with uneven2/3-row ranks |
| `grpo-7gpu-resume02` |Split-count check passed; backward OOM,225.665s | Two long rows materialized too much vocabulary output state |
| `grpo-7gpu-resume03` | Checkpoint2 restored, step3 saved,472.468s | Finite nonzero gradient0.0706812; world7 and all selected GPUs verified |

Logs, GPU telemetry, manifests, metrics and saved checkpoints are retained. Earlier synchronous NVML checks stalled during training. The bounded launcher now samples in its own process and preserves timing gaps. It tracks only launched processes using PID/birth identities and pidfds. Independent safety reviews and disposable timeout tests cover cleanup.

## Configuration fixes and limits

`SP_DP_PAD=1` pads32 real sequences to35. Driver-assigned actor minibatches contain2 or3 rows per rank. A one-sequence token allowance can request3 microbatches on one rank, impossible on a2-row rank. Doubling that allowance bounds the maximum at2, preserving masks and objective while increasing peak activation memory.

Backward then requested8.69GiB with6.96GiB free. The engineering wrapper had disabled the original runner's fused output kernel. Restoring `SP_USE_FUSED_KERNELS=True` uses the torch backend's512-token output chunks. GPU1 comparison against eager math used65 tokens, chunk32, signed log-prob weights and entropy gradients. FP32 maximum output error2.39e-6; BF16 log-prob/entropy errors4.93e-4/0.01453. BF16 hidden/weight gradient relative L2 errors0.004552/0.005322 satisfy the predeclared0.02 bound. Receipt: `fused-kernel-check-02.json`. Operator agreement alone does not establish full-run memory fit.

The original step cache retains resume01's post-reward batch. Later retries skip generation/judging and cannot measure fresh throughput or cost; E8 needs a fresh step. Increasing response budget on resume is a coverage extension, not a matched learning comparison.

Research-engineering conclusion: short-budget zero gradients demonstrate no learning effect; E6 training samples are not a model-quality benchmark. ML-engineering conclusion: state sharding, uneven dispatch, monitoring, output memory and checkpoint I/O require separate receipts. Scientific model choice remains E9.
