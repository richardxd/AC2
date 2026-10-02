# AC2 Reproduction Roadmap (g16)

## Purpose

Fixed task list for reproducing "Trust the Critic More" (arXiv 2609.39247, repo WhenWen/AC2) on g16 (8x RTX 6000 Ada 48 GB). Each task has an acceptance check. Status and evidence live in `CHECKLIST.md`; code/config changes in `ENGINEERING_LEDGER.md`; paper-vs-ours numbers in `RESULTS_LEDGER.md`. Approved by Richard on 2026-10-01; changes to this file need Richard's approval.

## Amendment 2026-10-02, Richard approved

Richard authorizes long R1 then R2 at the proposed scale, using local `openai/gpt-oss-20b` as a SURROGATE judge for both training reward and validation. This supersedes the original API-only judge and GPU0 exclusion only for the new judge's use of GPU0 free memory. Preserve existing sglang PID3926213 and its workers; leave memory headroom. Training remains on physical GPUs1–7. No DeepSeek API calls; preserve the remaining$2.281138832. R3/R4/R5 long runs require a new decision.

Execute in order: S1 serve pinned-revision weights under repository `models/` using the project venv/vLLM, native `gptoss` payloads and original SHA-pinned templates; record server arguments/reasoning effort and require20/20 E4-proof parses with latency/tokens-per-second. S2 regrade every saved DeepSeek-graded proof without API calls; separately report train(no-reference) and validation(reference+rubric) exact-points agreement, pass≥6 agreement/Cohen kappa, mean absolute point difference and Spearman. Agreement is a reported deviation, never a launch gate. S3 bounded R1 cold/resume smokes≤1740s each, measuring step time and judge share; update the proposal. S4 detach R1's200-step run under `runs/research/r1`, save every20steps on shared storage. Only after all200 R1 steps and acceptance receipts pass may detached R2 launch. All other hard rules remain, including scratch≤45GB and no Slurm actions.

## Fixed decisions (original; amended above)

1. Judge: DeepSeek official API, key file `/home-nfs/richard1xur/.codex/.deepseek_key` (never print, copy or commit it). `/models` lists `deepseek-flash` and `deepseek-v4-pro`; the paper pins DeepSeek-V4-Flash revision `60d8d707` served locally, so API identity with that revision is unverified: record this as a deviation. Balance on 2026-10-01: 32.43 USD. Total judge spend cap: 5 USD until Richard approves more.
2. Full scale (4 nodes x 8 GPUs, about 30 GPU-hours/step, about 200 steps) is out of reach. Experiments use a scaled-down protocol fixed after E8/E9 and approved by Richard.
3. Storage: venv, HF cache, uv cache, TMPDIR under `/scratch/richard1xur/ac2/` (g16 scratch had 94 GB free on 2026-10-01; check before writing). Checkpoints, rollouts, metrics, logs under `/share/data/pals/richard1xur/ac2/runs/`.
4. GPUs: g16 GPU0 hosts an agent_empowerment sglang server (PID 3926213); do not touch it. Use GPUs 1-7 until Richard decides about GPU0. g20 belongs to another running agent; do not use it.

## Engineering (E): run without asking; jobs under 30 min

1. E1 Environment: build the `install.sh` pinned stack on g16 (no CUDA 12.9 toolkit on the node; 12.1 and 13.0 exist). Accept: one command prints torch 2.11.0+cu129, vllm 0.23.0, flash_attn 2.8.1, verl importable, `torch.cuda.device_count()` as expected.
2. E2 CPU tests: `tests/` and every `experiments/*/test_*.py`. Accept: pass/fail list; `test_context_group_moves_together` is a documented expected failure (expects 75k budget).
3. E3 Data: `prepare_fineproofs --split both`, `build_rubric_map`. Accept: about 5,200 train / 60 val rows, sha256 of every output.
4. E4 Judge: point `SELF_PLAY_JUDGE_URL` (or the minimal code change) at the DeepSeek API for train and val prompts. Accept: 20 graded proofs with 100% parse success, latency, tokens and USD per call, projected USD per training step and per run.
5. E5 Single-node adaptation: no Slurm/sbatch; local Ray head on g16; `.env` filled for g16. Accept: `compose_dryrun.sh` passes for GRPO and AC2 main configs.
6. E6 Multi-GPU (expect heavy debugging): tiny GRPO run (short budget, about 2 steps) on 1 GPU, then on all available GPUs, then save and resume. Accept: FSDP world size equals GPU count; rollout TP/DP logged; per-GPU nvidia-smi samples show load on every GPU; resumed step and loss continue.
7. E7 AC2 smoke: tiny AC2 run exercising critic queries, readiness, chunks, audit, `sp_dp_pad`. Accept: `q/*` metrics logged; critic optimizer step taken; chunk lengths <= b.
8. E8 Throughput calibration: step time, Decoding FLOPs (Eq. 6 via `export_paper_metrics.py`) and judge USD for candidate scaled configs (starting point: response budget 16k, b=4k, n_batch=n_refill=32, g=8, 4 val samples per problem). Accept: projected wall-clock and USD for each R task; proposal sent to Richard.
9. E9 Small-model gate: step-0 IMO-ProofBench (60 problems x 4 samples, 16k budget, same judge) for Qwen3-4B-Thinking-2507 and Qwen3-1.7B (thinking mode). Accept: mean score, nonzero-score fraction, decode tokens/s for both. Rule: keep 4B if its scaled run projects <= about 10 days; else switch to 1.7B if its step-0 mean >= half of 4B's and nonzero fraction >= 20%; else stop and ask Richard.

## Experiments (R): each launch needs Richard's approval

1. R1 GRPO baseline, scaled. Accept: val mean score vs Decoding FLOPs curve.
2. R2 AC2 main, scaled. Accept: step of first global-readiness crossing (paper: end of step 7); ready fraction over steps (paper: about 70% by step 200); FLOPs at which AC2 first exceeds R1's peak (paper: 0.79e20 vs 1.99e20, 2.5x; step 90 vs 120).
3. R3 Prefix GRPO, scaled. Accept: same curve; tests whether AC2's gain exceeds the replay-buffer effect.
4. R4 Ablations if budget allows: 2k chunks, correct-only buffer, w/o Audit.
5. R5 Value probe on an R2 checkpoint, following `08_15_q_probe_step40` (paper Fig. 4).

## Paper targets for RESULTS_LEDGER

GRPO peak 18.50%; AC2 peak 20.57%; AC2 16.41% at step 50 and 16.88% at step 60; AC2 exceeds GRPO peak at 0.79e20 Decoding FLOPs vs 1.99e20; GPU-hours fit H = 6.11 + 27.41 D/1e18 (r = 0.874). Read the remaining numbers (Fig. 2-4, 7-9, Tables 3-4) from the paper itself; render figure pages to images when text extraction fails.
