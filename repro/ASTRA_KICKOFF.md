# Astra Kickoff Brief

## Purpose

Standing instructions for the Codex (gpt-6-astra, high) agent that executes the AC2 reproduction on g16. Written by Claude Code for Richard on 2026-10-01. Task list and acceptance checks are in `repro/ROADMAP.md`.

## Your job

You own the engineering and experiment bookkeeping for reproducing "Trust the Critic More" (arXiv 2609.39247) in this repository, `/share/data/pals/richard1xur/ac2` (fork `richardxd/AC2`; `upstream` is `WhenWen/AC2`). You are on g16 (8x RTX 6000 Ada 48 GB, 48 CPU, about 1 TB RAM).

1. Read first-hand before acting: `repro/ROADMAP.md`, the paper (local, untracked: `paper/2609.39247.pdf`; text in `paper/2609.39247.txt`, which is incomplete; render pages to PNG and view them for figures and tables), `README.md`, `install.sh`, `scripts/PAPER_METRICS.md`, and the README, runner and launch script of `experiments/08_13_tiedq_seed192`, `07_15_handoff`, `08_11_ablation1_replay_noq`.
2. Create and maintain three ledgers in `repro/`, each starting with `# Title` and a `## Purpose` section:
   1. `CHECKLIST.md`: every ROADMAP task (E1-E9, R1-R5) with status (todo / in progress / done / blocked), acceptance result, and evidence paths (logs, receipts, commits).
   2. `ENGINEERING_LEDGER.md`: every change made to run on g16 (file, why, commit, how verified), every environment fact discovered (versions, paths, failures and fixes), and every deviation from the paper's setup.
   3. `RESULTS_LEDGER.md`: a table of every quantitative paper claim (number, source figure/table/section), our measured value, the run and config that produced it, and the scaling differences. Fill the paper column now; fill ours as evidence arrives. Never fill a value without a receipt.
3. Execute E1-E9 in order (E1-E3 may overlap). Engineering jobs under 30 minutes need no approval. E6 (multi-GPU) is expected to need substantial debugging: verify GPU use with first-hand evidence (world size in logs, per-GPU nvidia-smi samples), not by the job finishing.
4. After E8 and E9, write the scaled-protocol proposal (configs, projected wall-clock and judge USD per R task) into `repro/PROPOSAL_R.md` and stop. Do not launch any R task; Richard approves each launch.
5. Commit ledger and code changes to branch `repro-g16` and push it to `origin` after each completed task. Never push to `upstream` or to `main`.

## Hard rules

1. Never `scancel` or otherwise cancel any Slurm job. Never `srun`/`sbatch`. Never kill a process you did not start; never touch g16 GPU0's sglang server (PID 3926213) or anything on g20.
2. Delete or overwrite only inside `/scratch/richard1xur/ac2/` and `/share/data/pals/richard1xur/ac2/`. Nothing else may be removed, moved or modified, including other projects' code, data, envs and caches.
3. Never create anything at the top level of `/scratch` or `/scratch/richard1xur`; work under `/scratch/richard1xur/ac2/`. Set `TMPDIR`, `TMP`, `TEMP`, `UV_CACHE_DIR`, `HF_HOME`, `RAY_TMPDIR` (keep it short), `TRITON_CACHE_DIR`, and vLLM/FlashInfer caches under it. Check free bytes before large writes (g16 scratch had 94 GB free on 2026-10-01). Checkpoints, rollouts, metrics and run logs go under `/share/data/pals/richard1xur/ac2/runs/`.
4. Use GPUs 1-7 (`CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`) until Richard frees GPU0.
5. Judge: DeepSeek official API with key file `/home-nfs/richard1xur/.codex/.deepseek_key`. Never print, copy, log or commit the key. Total judge spend cap is 5 USD (balance 32.43 USD on 2026-10-01); measure cost per call in E4 and stop judge use at the cap.
6. No fallbacks: no try/except wrappers, silent defaults or retries added to core code beyond what network I/O to the judge strictly needs. Keep changes minimal; prefer configuration over code.
7. Do not use other projects' code, data, or conda envs. Do not use W&B online logging.
8. If something blocks you, record it in `CHECKLIST.md` as blocked with evidence, continue every other unblocked task, and then stop.
