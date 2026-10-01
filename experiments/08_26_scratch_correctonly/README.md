# AC2 w/ correct-only buffer

`08_26_scratch_correctonly`

**Paper reference:** Fig. 3, right panel, red line; Sec. 4.3, paragraph "AC2 w/ correct-only buffer improves early but worsens later"; App. A.2, Fig. 6; App. B, Table 3 (row "AC2 w/ correct-only buffer").

## Description

Motivated by Setlur et al. (2026), this run stores only correct trajectories in the actor replay buffer B. A refill rollout from S_refill enters B only if the training judge awards it at least six of seven points (`SP_REPLAY_ADMISSION=judged_correct` with `SP_PASS_POINTS_MIN=6`); in the main run every refill rollout enters B regardless of correctness. The admission rule is applied in `_admit_global` of `src/verl/verl/trainer/ppo/sp_replay.py`; no trainer code is changed.

Restricting admission has two side effects that come with the rule itself. First, the number of trajectories entering the 256-capacity buffer per step falls from 192 to the number of solved refill problems, so stored trajectories remain in B for more steps and the replayed prefixes are older relative to the current policy. Second, B concentrates on problems that the policy solves in a single attempt. As in the main run, the 192 prefixes of a step are formed by cycling through a random permutation of the distinct problems in B, so while B holds fewer than 192 distinct problems some problems are replayed more than once per step; fresh problems with empty responses replace the prefixes only while B is empty. All other settings, including g = 16, b = 10,000 and auditing with α = 1/4, follow the main run, and training starts from Qwen3-4B-Thinking-2507 with empty buffers.

Compute: 4 nodes × 8 GPUs (`submit_cluster_a_4node.sbatch`); an 8-node holder script is also provided.

## Configuration

| Setting | This run | AC2 main run |
|---|---|---|
| Replay-buffer admission (`SP_REPLAY_ADMISSION`) | refill rollouts with ≥ 6/7 training-judge points (`judged_correct`) | all refill rollouts (`ungated`) |
| Critic-side RNG seed (`SP_Q_RNG_SEED`) | 826001 | 804001 |
| Readiness requires a solved problem (`SP_Q_READY_REQUIRE_BANK=1`) | from step 1 | enabled partway through training (absent from the initial launch script) |

All other settings follow the main run (Table 4).

## Files

- `README.md`: this document.
- `runner.py`: entry point. Builds the verl/Hydra configuration from `SP_*` environment variables, writes the run manifest and launches training.
- `run_attach_cluster_a.sh`: launch script for the cluster A. Runs preflight checks and starts `runner.py` on a running holder allocation with this run's environment.
- `submit_cluster_a_4node.sbatch`, `submit_cluster_a_8node.sbatch`: SLURM holder jobs (4 and 8 nodes) that start a Ray cluster and keep the GPUs busy until the driver attaches.
- `q_dataset.py`: dataset class (`data.custom_cls`), identical to the main run's.
- `sp_agent_loops.yaml`: agent-loop registry for prefix-continuation rollouts and critic queries.
- `build_cold_artifacts.py`: writes the empty replay-buffer seed, critic-buffer seed and reference bank.
- `compose_dryrun.sh`: CPU-only check that composes the Hydra configuration and asserts this run's settings.
- `q_monitor_dataset.py`: analysis script; builds a per-(step, group) dataset of critic predictions, readiness state and continuation outcomes from `run_data/`.
- `q_monitor_report.py`: analysis script; validates that dataset and prints critic-quality summaries.
- `q_group_structure.py`: analysis script; within- versus between-group structure of critic values.
- `refresh_dashboard.sh`: render the training dashboard PDF from `run_data/`.
- `q_build_common.py`: shard-writing and checksum helpers imported by `build_cold_artifacts.py`.
- `flashinfer_aot_warm.py`: ahead-of-time compilation of the flashinfer kernels, run before every launch (required with rollout tensor parallelism > 1).
- `setup.sh`: per-node runtime environment, sourced by the launch scripts on every node: `setup.sh cuda-compat` adds the CUDA 12.9 forward-compatibility libraries on cluster A nodes with older drivers, `setup.sh cuda-toolkit` selects the CUDA toolkit and include paths for the judge's JIT-compiled kernels.

## Running

Prerequisites are those of the main run: the repository environment (`install.sh`); FineProofs-RL and IMO-ProofBench data prepared with `python -m ac2.data.prepare_fineproofs` and `python -m ac2.data.build_rubric_map` (read from `$SELF_PLAY_DATA_DIR` and `$SP_VAL_MAP`); and `Qwen/Qwen3-4B-Thinking-2507` and `deepseek-ai/DeepSeek-V4-Flash` (revision `60d8d70770c6776ff598c94bb586a859a38244f1`) in the local Hugging Face cache. The judge runs under vLLM on the training nodes; `SP_JUDGE_MODEL=<snapshot dir>` overrides the cache lookup.

```bash
E=experiments/08_26_scratch_correctonly
python $E/build_cold_artifacts.py --out $E
bash $E/compose_dryrun.sh                          # optional configuration check (CPU)
bash scripts/sbatch_env.sh $E/submit_cluster_a_4node.sbatch
bash $E/run_attach_cluster_a.sh <HOLDER_JOBID>
```

Outputs are written to `run_data/` in this folder. The SLURM and shell scripts contain site-specific paths and SLURM settings and use the helper files `setup.sh` and `flashinfer_aot_warm.py` in this folder; adapt them before running elsewhere.

## Results

From Sec. 4.3: this run reaches 17.35% at step 60, compared with 16.88% for AC2, but the early advantage does not persist; at step 110 its score falls to 14.67%. Decoding FLOPs for this run combine step means with retained joint length records, and its cost data extend to step 117 (Table 3).
