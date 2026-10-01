# AC2 w/o Group & Audit

`09_01_scratch_qtd_prefix16`

**Paper reference:** Fig. 3, left panel, purple line; Sec. 4.2, paragraph "AC2 with group size 1"; Sec. 4.3, paragraph "Without local readiness" (the reference run with local readiness); App. A.2, paragraph "Starting-prefix baseline", Fig. 8 (ready fraction) and Fig. 6; App. B, Table 3 (row "AC2 w/o Group & Audit (from base model)").

## Description

This run removes both grouping and auditing on ready problems. In the main run, a replay slot whose problem is ready yields one prefix s and g = 16 action chunks from it, with advantage v_i − mean_j v_j. Here the same slot yields 16 distinct prefixes cut from the same stored trajectory and one action chunk of at most b = 10,000 new tokens from each (g = 1), so the number of continuations per sampled problem is unchanged. Each continuation receives the advantage Â = v − V^π_θ(s), where v = V^π_θ(s · c) at the chunk endpoint, or the terminal reward r if the chunk ends the trajectory, and V^π_θ(s) is the critic's prediction at the continuation's own prefix. A continuation whose baseline V^π_θ(s) is invalid is left out of both the actor loss and the critic targets. The critic is fit to the same single endpoint: the target for V^π_θ(s) is v rounded to the value grid, with the minimum of 8 valid continuations per critic target lowered to 1 for these single-continuation groups. To bound the size of the critic-target log, at most 8 of the 16 single-continuation targets of a slot enter the critic buffer per step. Unready problems keep full-length groups of 16 continuations with terminal-reward group-mean advantages, which continue to supply average-terminal-reward critic targets. Auditing is disabled (α = 0).

The 16 prefixes of a ready slot are drawn from a 1,000-token cut grid in [0, 0.9 L] (instead of the 10,000-token grid of the main run) by a random permutation, so that every grid point is used once before any is repeated; at a 10,000-token grid a typical trajectory would offer only a few distinct cuts. Prefixes of unready slots keep the 10,000-token grid. Implementation: the dataset (`q_dataset.py`) stamps the cuts of each ready slot; the trainer rewrites the slot's 16 copies into single-continuation requests (`sp_td_rewrite_gen_copies`, `sp_td_rewrite_driver_copies` in `src/verl/verl/trainer/ppo/ray_trainer.py`); cuts come from `prefix_cuts_td` in `sp_replay.py`; advantages are computed by the `sp_segment` estimator in `core_algos.py`, whose path for rows without a critic stamp is identical to the GRPO group-mean advantage without standard-deviation normalization. Unit tests are in `tests/sp_qtd/` and `tests/sp_segment/`.

The run starts from Qwen3-4B-Thinking-2507 with empty replay buffer, critic buffer and reference bank. Apart from the mechanism above, the configuration equals AC2 w/o Audit (`08_31_scratch_g10k_noaudit`), including its critic-side seed, so the two runs share their random draws until the first problem becomes ready. Compute: the launch configuration targets 7 nodes × 8 GPUs, using the padded data-parallel dispatch described in `08_31_scratch_g10k_noaudit` (`SP_DP_PAD=1`); 4- and 8-node holder scripts are also provided.

## Configuration

| Setting | This run | AC2 main run |
|---|---|---|
| Ready-problem sampling (`SP_Q_TD_ENABLE`) | 16 prefixes × 1 action chunk (`1`) | 1 prefix × 16 action chunks (not set) |
| Ready-problem advantage | v − V^π_θ(s) | v_i − mean_j v_j |
| Advantage estimator (`SP_ADV_ESTIMATOR`) | `sp_segment` | `grpo` |
| Cut grid for ready-problem prefixes (`SP_Q_TD_CUT_GRAIN`) | 1,000 tokens | 10,000 tokens |
| Critic targets from ready problems | single endpoint, at most 8 per slot (`SP_Q_TD_ADMIT_PER_SLOT=8`) | group mean over ≥ 8 valid continuations |
| Audit fraction α (`SP_Q_AUDIT_DEN`, `SP_Q_AUDIT_CUT`) | 0 (`0`, `0`) | 1/4 (`4`, `1`) |
| Critic-side RNG seed (`SP_Q_RNG_SEED`) | 831001 | 804001 |
| Readiness requires a solved problem (`SP_Q_READY_REQUIRE_BANK=1`) | from step 1 | enabled partway through training (absent from the initial launch script) |
| Padded data-parallel dispatch (`SP_DP_PAD`) | 1 | not set |
| Judge requests in flight (`SP_JUDGE_MAX_INFLIGHT`) | 20 × number of nodes | 160 |

All other settings follow the main run (Table 4).

## Files

- `README.md`: this document.
- `runner.py`: entry point. Builds the verl/Hydra configuration from `SP_*` environment variables (including the single-continuation settings above), checks consistency, writes the run manifest and launches training.
- `run_attach_cluster_a.sh`: launch script for the cluster A. Runs preflight checks and starts `runner.py` on a running holder allocation with this run's environment.
- `submit_cluster_a_7node.sbatch`, `submit_cluster_a_8node.sbatch`, `submit_cluster_a_4node.sbatch`: SLURM holder jobs (7, 8 and 4 nodes) that start a Ray cluster and keep the GPUs busy until the driver attaches.
- `q_dataset.py`: dataset class (`data.custom_cls`); in addition to the main run's logic it materializes ready slots as 16 distinct cuts of one stored trajectory.
- `sp_agent_loops.yaml`: agent-loop registry for prefix-continuation rollouts and critic queries.
- `build_cold_artifacts.py`: writes the empty replay-buffer seed, critic-buffer seed and reference bank.
- `compose_dryrun.sh`: CPU-only check that composes the Hydra configuration and asserts this run's settings.
- `refresh_dashboard.sh`: render the training dashboard PDF from `run_data/`.
- `q_build_common.py`: shard-writing and checksum helpers imported by `build_cold_artifacts.py`.
- `flashinfer_aot_warm.py`: ahead-of-time compilation of the flashinfer kernels, run before every launch (required with rollout tensor parallelism > 1).
- `setup.sh`: per-node runtime environment, sourced by the launch scripts on every node: `setup.sh cuda-compat` adds the CUDA 12.9 forward-compatibility libraries on cluster A nodes with older drivers, `setup.sh cuda-toolkit` selects the CUDA toolkit and include paths for the judge's JIT-compiled kernels.

## Running

Prerequisites are those of the main run: the repository environment (`install.sh`); FineProofs-RL and IMO-ProofBench data prepared with `python -m ac2.data.prepare_fineproofs` and `python -m ac2.data.build_rubric_map` (read from `$SELF_PLAY_DATA_DIR` and `$SP_VAL_MAP`); and `Qwen/Qwen3-4B-Thinking-2507` and `deepseek-ai/DeepSeek-V4-Flash` (revision `60d8d70770c6776ff598c94bb586a859a38244f1`) in the local Hugging Face cache. The judge runs under vLLM on the training nodes; `SP_JUDGE_MODEL=<snapshot dir>` overrides the cache lookup.

```bash
E=experiments/09_01_scratch_qtd_prefix16
python -m pytest -q tests/sp_qtd tests/sp_segment  # unit tests for the mechanism (CPU)
python $E/build_cold_artifacts.py --out $E
bash $E/compose_dryrun.sh                          # optional configuration check (CPU)
bash scripts/sbatch_env.sh $E/submit_cluster_a_7node.sbatch
bash $E/run_attach_cluster_a.sh <HOLDER_JOBID>
```

Outputs are written to `run_data/` in this folder. The SLURM and shell scripts contain site-specific paths and SLURM settings and use the helper files `setup.sh` and `flashinfer_aot_warm.py` in this folder; adapt them before running elsewhere.

## Results

From Sec. 4.2: this run performs similarly to AC2 (Fig. 3, left). From Sec. 4.3: it reaches a mean score of 15.33% at step 40, when 18.23% of its sampled problems are ready (Fig. 8). Decoding FLOPs for this run use step-mean estimates throughout, and its cost data extend to step 186 (Table 3).
