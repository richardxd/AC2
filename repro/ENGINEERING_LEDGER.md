# AC2 engineering ledger

## Purpose

Record g16 environment facts, changes, verification and deviations from the paper. Findings require receipts; configuration success does not establish training success. Links: [checklist](CHECKLIST.md), [paper results](RESULTS_LEDGER.md).

## Environment and changes

| Fact/change | Why / verification | Evidence / commit |
|---|---|---|
| g16.ttic.edu; 8 RTX 6000 Ada GPUs, 49,140 MiB each; driver 580.95.05 | First-hand nvidia-smi. GPU0 occupied (16,629 MiB); use only physical 1–7 | `runs/e1/host.txt`; base `d90aa18` |
| Scratch initially 100,729,585,664 free bytes; repository mount 90,087,925,743,616 | Check before large writes; shared scratch capacity can change | `runs/e1/host.txt` |
| System CUDA toolkits 12.1.66 and 13.0.88; no `/usr/local/cuda-12.9` | Flash-attn CUDA 12.9 build needs a project-local toolkit or verified compatible compiler | `runs/e1/host.txt` |
| uv 0.12.9; no pre-existing project training venv | New install, no environment cleared | `runs/e1/host.txt`, `runs/e1/install-base.log` |
| Added `repro/env.sh` | All temporary/cache paths under project scratch, GPUs 1–7, offline W&B; Python downloads also localized | Source before commands; implementation commit pending |
| Created `/scratch/richard1xur/ac2/tools-venv`, Python 3.11.2, PyMuPDF 1.28.2 | System `pdftoppm`/`pdftotext` absent; isolated PDF tooling, not training environment | `runs/paper/page-*.png`, `runs/paper/extracted.txt` |
| Base install started with `INSTALL_FA=skip`, original install.sh | Separate base dependency install from toolkit/build diagnosis; E1 stays incomplete until flash-attn verifies | `runs/e1/install-base.log` |

## Scientific deviations and interpretation

| Difference | Research-engineer assessment | ML-engineer assessment |
|---|---|---|
| DeepSeek official API rather than local revision `60d8d707` | Identity with paper judge unverified; scores cannot be claimed directly comparable | Removes colocated huge judge; requires centralized cost cap and receipts |
| 7 Ada 48 GB GPUs, single node; paper AC2 32 GPUs, GRPO launcher 64 | Decoding FLOPs enable a policy-work comparison, not proof of equal end-to-end cost | Need measured world size, per-GPU load and wall time; no utilization backfill |
| Proposed 16k / b=4k / g=8 / 32 replay + 32 refill / 4 validation samples | Only a candidate, not an approved scientific protocol; changes horizon, sampling and readiness visitation | Calibrate in E8/E9 before any R proposal |
| Paper reference-bank readiness requirement activated at step 42; shipped launcher sets it from step 1 | Must choose and disclose in proposal; do not silently reproduce the bug or silently claim exact replication | Configurable; no need to patch core algorithm |
| Prefix GRPO README documents test expecting batch 256 versus shipped 384 | Record actual failure separately from roadmap's expected 75k-context failure | Do not change scientific settings merely to make stale tests pass |
| 2k ablation changes both chunk length and prefix-cut spacing (App. A.2) | Confounded comparison; does not isolate chunk length alone | Preserve/report paired settings |

## Judge spending

No judge calls made yet. Spend by this reproduction: $0. Cap: $5. The kickoff's historical balance ($32.43) is not a current measurement.

## Commit and review receipts

Pending first completed task. Every commit needs an independent review; only push `origin repro-g16`.
