# AC2 w/o Group & Audit (branch at step 50)

`09_05_qtd_ready_s192b50_noaudit`

**Paper reference:** App. B, Table 3 (row "AC2 w/o Group & Audit (step-50 branch)"). This run appears in the paper only as a row of the decoding-cost coverage table; its scores are not reported in the text or figures. The variant reported in Fig. 3 trains from the base model (`09_01_scratch_qtd_prefix16`).

## Description

This run branches off the AC2 main run (`08_13_tiedq_seed192`) at its step-50 checkpoint and from step 51 on applies the AC2 w/o Group & Audit configuration of `09_01_scratch_qtd_prefix16`. The actor/critic weights, the critic optimizer state and the readiness state are copied from the checkpoint, and the replay buffer, critic buffer and reference bank are reconstructed by replaying the parent's delta logs up to step 50. On ready problems each replay slot yields 16 distinct prefixes of one stored trajectory (1,000-token cut grid) and one action chunk of at most b = 10,000 new tokens from each; each continuation receives the advantage v − V^π_θ(s), where V^π_θ(s) is the critic's prediction at its own prefix, and contributes a single-endpoint critic target (at most 8 per slot enter the critic buffer). Unready problems keep full-length groups of 16 with terminal-reward group-mean advantages. Auditing is disabled (α = 0). Readiness keeps the main run's local and global criteria, and the critic-side seed is the parent's.

Compute: 4 nodes × 8 GPUs (32 GPUs, the parent's training world size). The run was started on cluster A (`submit_cluster_a_4node.sbatch`, `run_attach_cluster_a.sh`) and resumed from step 101 on cluster B (`submit_cluster_b_4node.sbatch`, `launch_cluster_b.sh`, `run_attach_cluster_b.sh`) with identical training settings. The rollout engine uses throughput settings not used by the main run (grouped cascade attention for shared prefixes, prefix-affinity request routing, piecewise CUDA graphs, at most 128 concurrent sequences per judge engine); these do not change the training objective. We also tried enabling the grouped cascade-attention kernel (`SP_GROUPED_CASCADE=1`, implemented in `src/ac2/cascade_attn/`) in this run to speed up rollouts over the shared prefixes; the throughput gain was small.

## Configuration

| Setting | This run | AC2 main run |
|---|---|---|
| Initialization | AC2 main run, step-50 checkpoint and reconstructed buffers | Qwen3-4B-Thinking-2507, empty buffers |
| Ready-problem sampling (`SP_Q_TD_ENABLE`, `SP_Q_TD_LANE`) | 16 prefixes × 1 action chunk (`1`, `short`) | 1 prefix × 16 action chunks (not set) |
| Ready-problem advantage | v − V^π_θ(s) | v_i − mean_j v_j |
| Advantage estimator (`SP_ADV_ESTIMATOR`) | `sp_segment` | `grpo` |
| Cut grid for ready-problem prefixes (`SP_Q_TD_CUT_GRAIN`) | 1,000 tokens | 10,000 tokens |
| Critic targets from ready problems | single endpoint, at most 8 per slot (`SP_Q_TD_ADMIT_PER_SLOT=8`) | group mean over ≥ 8 valid continuations |
| Audit fraction α (`SP_Q_AUDIT_DEN`, `SP_Q_AUDIT_CUT`) | 0 (`0`, `0`) | 1/4 (`4`, `1`) |
| Rollout-engine settings (`SP_GROUPED_CASCADE`, `SP_PREFIX_AFFINITY`, `SP_PREFIX_PILOT`, `SP_ROLLOUT_CUDAGRAPH_MODE`, `SP_REWARD_MAX_NUM_SEQS`) | `1`, `1`, `1`, `PIECEWISE`, `128` | not set |

All other settings follow the main run (Table 4).

## Files

- `README.md`: this document.
- `runner.py`: entry point. Builds the verl/Hydra configuration from `SP_*` environment variables (including the single-continuation settings), checks consistency and launches training.
- `run_attach_cluster_a.sh`: launch script for cluster A. Constructs the branch on first use (copies the parent's step-50 checkpoint and both delta logs), runs preflight checks and starts `runner.py` on a running holder; supports `SP_DRYRUN=1`.
- `run_attach_cluster_b.sh`: the same launch script for cluster B (identical training environment), including the inline-driver mode used by `launch_cluster_b.sh`.
- `submit_cluster_a_4node.sbatch`: 4-node SLURM holder job for cluster A.
- `launch_cluster_b.sh`: per-node launch step run by `submit_cluster_b_4node.sbatch` (one task per node); starts Ray, writes the allocation info file and runs `run_attach_cluster_b.sh` on the first node.
- `q_dataset.py`: dataset class (`data.custom_cls`); materializes ready slots as 16 distinct cuts of one stored trajectory (lane selectable via `sp_q_td_lane`).
- `sp_agent_loops.yaml`: agent-loop registry for prefix-continuation rollouts and critic queries.
- `build_cold_artifacts.py`: writes the empty loader stubs (replay seed, critic seed, reference bank) that the runner requires; the actual buffers are reconstructed from the parent's delta logs.
- `compose_dryrun.sh`: CPU-only check that composes the Hydra configuration and asserts this run's settings.
- `build_lineage_dashboard.py`: renders one dashboard covering the parent's steps up to 50 followed by this run.
- `refresh_dashboard.sh`: render the training dashboard PDF from `run_data/`.
- `q_build_common.py`: shard-writing and checksum helpers imported by `build_cold_artifacts.py`.
- `flashinfer_aot_warm.py`: ahead-of-time compilation of the flashinfer kernels, run before every launch (required with rollout tensor parallelism > 1).
- `setup.sh`: per-node runtime environment, sourced by the launch scripts on every node: `setup.sh cuda-compat` adds the CUDA 12.9 forward-compatibility libraries on cluster A nodes with older drivers, `setup.sh cuda-toolkit` selects the CUDA toolkit and include paths for the judge's JIT-compiled kernels.
- `submit_cluster_b_4node.sbatch`: SLURM allocation for cluster B (4 nodes × 8 GPUs): runs `launch_cluster_b.sh` as one srun step with one task per node. Submit with `scripts/sbatch_env.sh`.

## Running

Inputs: the parent run's step-50 checkpoint (`experiments/08_13_tiedq_seed192/run_data/checkpoints/global_step_50`, including `sp_q_optim/`, `q_state.json` and `sp_replay_state.json`) and its delta logs (`run_data/q_state_deltas.jsonl`, `run_data/replay_buffer_deltas.jsonl`); FineProofs-RL and IMO-ProofBench data prepared with `python -m ac2.data.prepare_fineproofs` and `python -m ac2.data.build_rubric_map`; and the judge `deepseek-ai/DeepSeek-V4-Flash` (revision `60d8d70770c6776ff598c94bb586a859a38244f1`) in the local Hugging Face cache, served by vLLM on the training nodes. The allocation must have 4 nodes × 8 GPUs.

```bash
E=experiments/09_05_qtd_ready_s192b50_noaudit
python -m pytest -q tests/sp_qtd tests/sp_segment
python $E/build_cold_artifacts.py --out $E       # loader stubs
bash $E/compose_dryrun.sh                        # optional configuration check (CPU)
bash scripts/sbatch_env.sh $E/submit_cluster_a_4node.sbatch
bash $E/run_attach_cluster_a.sh <HOLDER_JOBID>
```

On cluster B, `bash scripts/sbatch_env.sh $E/submit_cluster_b_4node.sbatch` replaces the last two commands: it starts Ray and runs `run_attach_cluster_b.sh` inline. Outputs are written to `run_data/` in this folder. The scripts contain site-specific paths and SLURM settings and use the helper files `setup.sh` and `flashinfer_aot_warm.py` in this folder; adapt them before running elsewhere.

## Results

Not reported in the paper. The decoding-cost records for this run are joint length records and extend to step 117 (Table 3).
