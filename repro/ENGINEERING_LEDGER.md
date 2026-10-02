# AC2 engineering ledger

## Purpose

Record g16 environment facts, changes, verification and deviations from the paper. Findings require receipts; configuration success does not establish training success. Links: [checklist](CHECKLIST.md), [paper results](RESULTS_LEDGER.md).

## Environment and changes

Supervisor storage constraint (2026-10-01): **project scratch total must remain below 45 GB**. Scratch is shared; check free bytes and project usage before each large build/download. Only hot caches and the active environment belong there. All checkpoints, rollouts, dumps, logs and extra snapshots belong under repository `runs/`. Supervisor measured 28 GB used (uv 14 GB, HF 12 GB), shared scratch 66 GB free at 99% usage.

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

## E1 acceptance

FA-only second build succeeded. `repro/verify_environment.py` verified torch 2.11.0+cu129, vllm package 0.23.0+cu129 (module 0.23.0), flash-attn 2.8.1 and vendored verl, with seven visible GPUs. Actual BF16 causal flash-attn forward/backward passed on each physical GPU 1–7, maximum absolute discrepancy versus Torch SDPA 0.00048828125, finite gradients. GPU0 was excluded. Receipt: `repro/receipts/e1-acceptance.json`; exact package freeze: `repro/receipts/e1-freeze.txt`.

The sandbox process listing did not expose host compiler processes; an escalated read-only process listing confirmed the restarted build's uv/ninja/nvcc descendants were active. Absence in sandbox `ps` must not be interpreted as host-process absence. Post-build scratch usage: 26 GB, below 45 GB; shared available bytes 73,019,699,200.

## Commit and review receipts

E2 commit `7d52571`: independent review by `review_e2` verified all 120 test-list entries against XML, all six diagnoses, experiment enumeration and shell syntax; clean within E2 scope. Paper/host facts not included in that review. Only push `origin repro-g16`.

E3 commit `982a4c9`: independent review by `review_e3` verified all ten raw/derived hashes, the sole row-51 change, prompt/metadata preservation, all 60 references against archived official CSV and 5,225 unique nonempty train rubrics. Clean within E3 scope.

## E3 data preparation

Raw generation ran both requested modules unmodified: 5,227 train rows and 60 val rows. Training rubrics collapse to 5,225 keys; no empty rubric. The raw val map matches 59/60. Row 51 has a mirror transcription discrepancy (excluded triangle vertices swapped); the official DeepMind CSV supplies the corrected statement and its reference. `repro/prepare_eval_data.py` creates a separate canonical dataset, asserts exactly this change, retains all 60 problems and checks nonempty reference/guidelines coverage. Both raw and derived hashes are tracked in `repro/receipts/e3-*.json`; originals preserved. This is an explicit evaluation-input deviation. See [data finding](../docs/findings/repro/data-preparation.md).

## Interrupted build / sandbox restart

Supervisor interrupted first flash-attn build and reported uv terminated. On resume no nvcc/cicc/ninja/ptxas processes remained in the user process listing. Reran FA-only build into the same project venv, with log `runs/e1/flash-attn-build-02.log`. Toolkit components were downloaded from NVIDIA CUDA 12.9.1 redistributables and checked against published SHA256 metadata; receipt `/scratch/richard1xur/ac2/cuda-12.9/receipt.json`. Initial extraction with shell-default Python failed because that interpreter lacked `tarfile`'s `filter` argument; reran using project Python 3.12.14. No system toolkit modified. `FLASH_ATTN_CUDA_ARCHS=80` builds the Ampere cubins compatible with Ada; actual kernel execution remains to be verified.
