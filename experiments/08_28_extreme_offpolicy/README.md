# AC2 w/ stale replay buffer

`08_28_extreme_offpolicy`

**Paper reference:** Fig. 3, left panel, pink line; Sec. 4.2, paragraph "AC2 with stale replay buffer"; App. A.2, Fig. 6; App. B, Table 3 (row "AC2 w/ stale replay buffer").

## Description

This run tests how AC2 tolerates a stale initial-state distribution. The main run refills the replay buffer B every step with one full rollout on each of n_refill = 192 fresh problems, and B holds the most recent 256 trajectories. Here the refill happens in bursts: on every tenth step (global steps 1, 11, 21, ...) each of the 192 fresh problems receives 10 full rollouts, giving 1,920 rollouts from a single policy snapshot, and on the other nine steps no refill rollouts are generated. The buffer capacity is raised to 1,920, exactly one burst, so each burst replaces the previous one entirely. For the following ten steps the actor therefore cuts its prefixes from a fixed buffer whose trajectories are 1 to 10 steps old. As in the main run, prefixes are drawn by cycling through a random permutation of the distinct problems in B (here the 192 problems of the last burst), taking one of that problem's stored trajectories uniformly at random and a fresh cut on the 10,000-token grid.

The trained batch is unchanged at every step (192 prefixes × 16 continuations, with auditing at α = 1/4), and the number of refill rollouts per ten steps equals that of the main run (1,920). The run starts from Qwen3-4B-Thinking-2507 with empty replay buffer, critic buffer and reference bank; because step 1 is a burst step, the buffer is full from step 2 onward. Implementation: `q_dataset.py` stamps a per-row rollout count on refill rows (10 on burst steps, −1 on other steps), and the trainer's per-row repeat resolver (`_sp_per_row_repeat_counts` in `src/verl/verl/trainer/ppo/ray_trainer.py`) drops rows with a negative count before generation; `test_08_28_logic.py` covers the schedule, the resolver and whole-burst buffer turnover. Compute: 8 nodes × 8 GPUs (`submit_cluster_a_8node.sbatch`).

When reading this run's training metrics, note that once readiness opens, training-correctness metrics that average judge outcomes over all rollout rows without filtering on the routing lane (for example `rubric_grade_mean` and `replay/pass_rate_replay`) are biased low by the fraction of rows scored by the critic, which carry a judge score of 0. The audit-reweighted estimates computed by the dashboard parser (`replay_stream_*_full_suffix_est` in `src/ac2/viz/parse_fig_data.py`) do not have this bias. Refill-stream metrics such as `replay/pass_rate_orig` exist only on burst steps.

## Configuration

| Setting | This run | AC2 main run |
|---|---|---|
| Refill schedule (`SP_INFLOW_BURST_PERIOD`, `SP_INFLOW_BURST_N`) | 192 problems × 10 rollouts every 10 steps (`10`, `10`) | 192 problems × 1 rollout every step (not set) |
| Replay buffer capacity (`SP_REPLAY_BOUND`) | 1,920 | 256 |
| Critic-side RNG seed (`SP_Q_RNG_SEED`) | 828001 | 804001 |
| Readiness requires a solved problem (`SP_Q_READY_REQUIRE_BANK=1`) | from step 1 | enabled partway through training (absent from the initial launch script) |

All other settings follow the main run (Table 4).

## Files

- `README.md`: this document.
- `runner.py`: entry point. Builds the verl/Hydra configuration from `SP_*` environment variables, asserts that the buffer capacity equals one burst and that admission is ungated, writes the run manifest and launches training.
- `run_attach_cluster_a.sh`: launch script for the cluster A. Runs preflight checks and starts `runner.py` on a running holder allocation with this run's environment.
- `submit_cluster_a_8node.sbatch`, `submit_cluster_a_4node.sbatch`: SLURM holder jobs (8 and 4 nodes) that start a Ray cluster and keep the GPUs busy until the driver attaches. The run uses the 8-node holder.
- `q_dataset.py`: dataset class (`data.custom_cls`) with the burst refill schedule.
- `sp_agent_loops.yaml`: agent-loop registry for prefix-continuation rollouts and critic queries.
- `build_cold_artifacts.py`: writes the empty replay-buffer seed, critic-buffer seed and reference bank.
- `compose_dryrun.sh`: CPU-only check that composes the Hydra configuration and asserts this run's settings.
- `test_08_28_logic.py`: CPU-only tests of the burst schedule, the repeat resolver and buffer turnover (runs with `python` or `pytest`).
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
E=experiments/08_28_extreme_offpolicy
python $E/test_08_28_logic.py                      # CPU tests
python $E/build_cold_artifacts.py --out $E
bash $E/compose_dryrun.sh                          # optional configuration check (CPU)
bash scripts/sbatch_env.sh $E/submit_cluster_a_8node.sbatch
bash $E/run_attach_cluster_a.sh <HOLDER_JOBID>
```

Outputs are written to `run_data/` in this folder. The SLURM and shell scripts contain site-specific paths and SLURM settings and use the helper files `setup.sh` and `flashinfer_aot_warm.py` in this folder; adapt them before running elsewhere.

## Results

From Sec. 4.2: the mean score reaches 17.90% and appears to plateau there, but reaches this value far faster than GRPO. Decoding FLOPs for this run combine step means with retained joint length records, and its cost data extend to step 146 (Table 3).
