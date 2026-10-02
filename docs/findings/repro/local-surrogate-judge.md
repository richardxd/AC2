# Local surrogate judge

## Purpose

Track Richard's2026-10-02 amendment authorizing local `openai/gpt-oss-20b` for R1 then R2. This is a surrogate deviation from both the paper and prior DeepSeek engineering measurements. See [ROADMAP](../../../repro/ROADMAP.md), [CHECKLIST](../../../repro/CHECKLIST.md).

S1 [receipt](../../../repro/receipts/s1-surrogate.json) pins revision6cee5e81ee83917806bbde320786a8fb61efebee and all14 model file hashes, with weights on shared storage. Installed vLLM MXFP4 source explicitly supports SM80+ and Marlin; serving uses this native backend on Ada, without changing dependencies. The [official vLLM recipe](https://github.com/vllm-project/recipes/blob/main/OpenAI/GPT-OSS.md) is supplementary guidance; acceptance rests on actual local execution.

GPU0 hosts both protected sglang processes and the new judge. A0.55 total-memory vLLM budget leaves4722MiB free after load. Server arguments and PID/create-time identities are retained in `runs/surrogate/server02/launch.json`; first CLI-only failure remains in server01. Native gptoss/high reasoning,65536 output allowance, original no-reference and reference+rubric templates remain SHA-pinned. The recording gateway forwards exclusively to loopback vLLM and never loads an API credential.

E4 original20cases pass20/20,103323 output tokens,602.565s wall,102.961s mean latency,171.472output tokens/s at concurrency4. No API calls. S2 uses every547 historical request, including repeated prompts and earlier calibration/training phases. Call-weighted train/val agreement is descriptive; invalid/truncated pairs are explicitly excluded, never converted to zero. It is not a launch gate.

Long R1/R2 use strict local routing and unchanged proposed policy/data configuration. R2's old compose-only judge pin was safely backed up and migrated before any training. Historical compose manifests remain, but execution acceptance ignores them and verifies every actual launch's surrogate profile. Only successful200-step R1 acceptance may admit R2; other long arms require another decision.
