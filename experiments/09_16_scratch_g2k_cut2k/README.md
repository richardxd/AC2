# AC2 w/ 2k chunks

`09_16_scratch_g2k_cut2k`

**Paper reference:** Fig. 3, right panel, green line; Sec. 4.3, paragraph "Smaller chunk size hurts performance"; App. A.2, paragraph "Chunk size" and Fig. 7; App. A.2, Fig. 6; App. B, Table 3 (row "AC2 w/ 2k chunks").

## Description

This run reduces the action-chunk length from b = 10,000 to b = 2,000 new tokens and, together with it, the spacing of the prefix-cut grid from 10,000 to 2,000 tokens. On ready problems each of the g = 16 continuations stops after at most 2,000 new tokens and its endpoint is scored by the critic V^π_θ, unless it finishes earlier. The finer cut grid applies to every replay prefix, ready or not, so prefixes are drawn uniformly from the multiples of 2,000 tokens in [0, 0.9 L]; the number of prefixes per replay slot (one) is unchanged, and a chunk that reaches its cap ends on a point of the cut grid, as in the main run. Because the chunk length and the cut spacing change together, the comparison does not isolate either change.

Everything else follows the AC2 main run, including grouping (g = 16), auditing (α = 1/4), local and global readiness, and the GRPO group-mean advantage; training starts from Qwen3-4B-Thinking-2507 with empty replay buffer, critic buffer and reference bank. Compute: the launch configuration targets 7 nodes × 8 GPUs (`submit_cluster_a_7node.sbatch`), using the padded data-parallel dispatch described in `08_31_scratch_g10k_noaudit` (`SP_DP_PAD=1`); 4- and 8-node holder scripts are also provided.

## Configuration

| Setting | This run | AC2 main run |
|---|---|---|
| Action-chunk length b (`SP_Q_BUDGET_G`) | 2,000 new tokens | 10,000 new tokens |
| Prefix-cut grid (`SP_REPLAY_CUT_GRAIN`) | 2,000 tokens | 10,000 tokens |
| Critic-side RNG seed (`SP_Q_RNG_SEED`) | 916001 | 804001 |
| Readiness requires a solved problem (`SP_Q_READY_REQUIRE_BANK=1`) | from step 1 | enabled partway through training (absent from the initial launch script) |
| Padded data-parallel dispatch (`SP_DP_PAD`) | 1 | not set |
| Judge requests in flight (`SP_JUDGE_MAX_INFLIGHT`) | 20 × number of nodes | 160 |

All other settings follow the main run (Table 4).

## Files

- `README.md`: this document.
- `runner.py`: entry point. Builds the verl/Hydra configuration from `SP_*` environment variables, checks consistency, writes the run manifest and launches training.
- `run_attach_cluster_a.sh`: launch script for the cluster A. Runs preflight checks and starts `runner.py` on a running holder allocation with this run's environment.
- `submit_cluster_a_7node.sbatch`, `submit_cluster_a_8node.sbatch`, `submit_cluster_a_4node.sbatch`: SLURM holder jobs (7, 8 and 4 nodes) that start a Ray cluster and keep the GPUs busy until the driver attaches.
- `q_dataset.py`: dataset class (`data.custom_cls`), identical to the main run's.
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
E=experiments/09_16_scratch_g2k_cut2k
python $E/build_cold_artifacts.py --out $E
bash $E/compose_dryrun.sh                          # optional configuration check (CPU)
bash scripts/sbatch_env.sh $E/submit_cluster_a_7node.sbatch
bash $E/run_attach_cluster_a.sh <HOLDER_JOBID>
```

Outputs are written to `run_data/` in this folder. The SLURM and shell scripts contain site-specific paths and SLURM settings and use the helper files `setup.sh` and `flashinfer_aot_warm.py` in this folder; adapt them before running elsewhere.

## Results

From Sec. 4.3 and App. A.2: this run and AC2 track each other through step 50, where they reach 16.03% and 16.41%, respectively. Beyond that step the shorter chunks stop improving: the run peaks at 16.62% at step 100 and falls to 13.57% at step 120. Decoding FLOPs for this run use step-mean estimates throughout, and its cost data extend to step 128 (Table 3).
