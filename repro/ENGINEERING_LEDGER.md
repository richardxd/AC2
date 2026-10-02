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

E4 made 40 calls including the failed 16k-budget attempt: conservative peak-price upper cost $0.210646032, leaving $4.789353968 of the $5 cap. First-hand provider balance changed from $32.43 to $32.33 (rounded cents; off-peak billing differs). `repro/judge_gateway.py` is the single durable accounting point: reserve worst-case input bytes plus framing and output budget before each API attempt; uncertain calls retain reservation; reject before sending at cap. SQLite ledger and raw receipts: `runs/judge/`. Official [pricing](https://api-docs.deepseek.com/quick_start/pricing/) peak USD/million: cache miss 0.30, cache hit 0.006, output 1.20. These are upper costs, not claimed itemized billed charges.

E4 final 20-call fixture (10 train + 10 val prompts) parsed 20/20 with 40k output; the earlier 16k attempt parsed 19/20 due to one truncation and is preserved. Mean train latency 26.335 s and upper cost $0.006651983; val 13.608 s and $0.003429097. Fixtures pair five official references and five unsupported assertions per route, not a representative generated-proof distribution. Illustrative 32×8 GRPO step $1.703; AC2 unready 32 refill + 32×8 continuations $1.916; 60×4 validation $0.823. A hypothetical 200-step run plus 21 validations projects $357.86 GRPO / $400.44 AC2 under those assumptions; these do not establish actual run cost. E8 must replace them with rollout measurements. Details and per-call receipts: [judge finding](../docs/findings/repro/judge-calibration.md).

Official `/models` identifies DeepSeek-V4.1-Flash. Adapter uses thinking enabled/high effort and the paper's 40k judge output; local-vLLM temperature/top-p/seed settings are not forwarded to official thinking API. Scores are not exactly comparable to pinned paper judge.

## E1 acceptance

FA-only second build succeeded. `repro/verify_environment.py` verified torch 2.11.0+cu129, vllm package 0.23.0+cu129 (module 0.23.0), flash-attn 2.8.1 and vendored verl, with seven visible GPUs. Actual BF16 causal flash-attn forward/backward passed on each physical GPU 1–7, maximum absolute discrepancy versus Torch SDPA 0.00048828125, finite gradients. GPU0 was excluded. Receipt: `repro/receipts/e1-acceptance.json`; exact package freeze: `repro/receipts/e1-freeze.txt`.

The sandbox process listing did not expose host compiler processes; an escalated read-only process listing confirmed the restarted build's uv/ninja/nvcc descendants were active. Absence in sandbox `ps` must not be interpreted as host-process absence. Post-build scratch usage: 26 GB, below 45 GB; shared available bytes 73,019,699,200.

Independent E1 review flagged inferred physical GPU identity in the first receipt. Fixed by setting `CUDA_DEVICE_ORDER=PCI_BUS_ID` and asserting Torch device UUIDs equal nvidia-smi GPU1–7 UUIDs before kernel work. First UUID assertion failed only because nvidia-smi includes `GPU-` while Torch omits it; normalized that explicit prefix and reran successfully. Final authoritative receipt: `repro/receipts/e1-acceptance-uuid.json`; earlier receipt is retained but does not independently prove physical identity. The local training runner applies the same pre-kernel identity check.

## Commit and review receipts

E1 commits `8f0015b`, `cb34fda`: independent final UUID review clean; pushed. Initial identity-inference finding corrected with real UUID assertions before kernels.

E2 commit `7d52571`: independent review by `review_e2` verified all 120 test-list entries against XML, all six diagnoses, experiment enumeration and shell syntax; clean within E2 scope. Paper/host facts not included in that review. Only push `origin repro-g16`.

E3 commit `982a4c9`: independent review by `review_e3` verified all ten raw/derived hashes, the sole row-51 change, prompt/metadata preservation, all 60 references against archived official CSV and 5,225 unique nonempty train rubrics. Clean within E3 scope.

## E5 local adaptation

Independent review of E4 `9562723` clean: all 20 raw hashes, 40-call SQLite total, parser/token claims and projection arithmetic reproduced. E5 review of `8a4d6b2` found the API pin compared model only; fixed to compare model and URL as well as resolved identity, preventing silent endpoint drift on resume. Also pinned inherited `CUDACXX` to project CUDA 12.9 (the parent had CUDA 12.1 despite CUDA_HOME 12.9).

`repro/local_runner.py` composes the three original entry points with explicit GPU count, TP/DP, project data/output/cache paths, offline logging, external API judge and short smoke settings. Minimal runner changes disable the local reward model when `SELF_PLAY_JUDGE_URL` is supplied and record API identity without downloading the local judge. `compose_dryrun.sh` passes GRPO/AC2/Prefix GRPO; local Ray preflight exposed one alive node, seven GPUs, 24 CPUs, then shut down its own cluster (`runs/e5/`). Composition is not training acceptance.

Model snapshots are pinned to 4B `768f209d9ea81521153ed38c47d515654e938aea` and 1.7B `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. First E6 GPU1 attempt initialized FSDP then vLLM's offline repository completeness check rejected absent README/LICENSE/.gitattributes despite weights loading. The runner now supplies the pinned local snapshot directory and verifies all weight shards rather than requesting repository metadata. No alternate model or core exception fallback.

Engineering smoke deviations: response 256, chunk 128, group 2, batch 8 (+8 refill for replay), critic train 32, eager decoding, TP1, no validation, save every step; all appear in config manifests. Replay/critic cold seeds use the shipped generator. GPU backfill is disabled; `SP_DP_PAD=1` is forwarded into Ray workers. Checkpoint pruning is disabled for these bounded runs. These tiny settings are not a proposed scientific protocol.

`repro/run_bounded.py` captures physical-GPU samples and enforces ≤1,740 seconds plus bounded cleanup. It signals only captured descendants through PID/birth-checked pidfds. Independent safety review requested two fixes (telemetry timeout and identity-safe descendant tracking); both implemented. Disposable parent/child timeout test finished in 1.508 seconds, return -15, timed_out true. Review approved first GPU1 smoke; it failed after 85.995 seconds with the offline metadata error. Transient descendants between polling samples remain a monitoring limitation; do not infer absence from sandbox process listings.

## E3 data preparation

Raw generation ran both requested modules unmodified: 5,227 train rows and 60 val rows. Training rubrics collapse to 5,225 keys; no empty rubric. The raw val map matches 59/60. Row 51 has a mirror transcription discrepancy (excluded triangle vertices swapped); the official DeepMind CSV supplies the corrected statement and its reference. `repro/prepare_eval_data.py` creates a separate canonical dataset, asserts exactly this change, retains all 60 problems and checks nonempty reference/guidelines coverage. Both raw and derived hashes are tracked in `repro/receipts/e3-*.json`; originals preserved. This is an explicit evaluation-input deviation. See [data finding](../docs/findings/repro/data-preparation.md).

## Interrupted build / sandbox restart

Supervisor interrupted first flash-attn build and reported uv terminated. On resume no nvcc/cicc/ninja/ptxas processes remained in the user process listing. Reran FA-only build into the same project venv, with log `runs/e1/flash-attn-build-02.log`. Toolkit components were downloaded from NVIDIA CUDA 12.9.1 redistributables and checked against published SHA256 metadata; receipt `/scratch/richard1xur/ac2/cuda-12.9/receipt.json`. Initial extraction with shell-default Python failed because that interpreter lacked `tarfile`'s `filter` argument; reran using project Python 3.12.14. No system toolkit modified. `FLASH_ATTN_CUDA_ARCHS=80` builds the Ampere cubins compatible with Ada; actual kernel execution remains to be verified.
