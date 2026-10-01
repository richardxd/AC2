# AC2 w/o Audit (branch at step 40)

`08_26_s192b40_g10k_noaudit`

**Paper reference:** App. B, Table 3 (row "No audit, branch at 40"). This run appears in the paper only as a row of the decoding-cost coverage table; its scores are not reported in the text or figures. The variant reported in Fig. 3 removes auditing from initialization (`08_31_scratch_g10k_noaudit`).

## Description

This run branches off the AC2 main run (`08_13_tiedq_seed192`) at its step-40 checkpoint and removes auditing (α = 0) from step 41 on. The actor/critic weights, the critic optimizer state and the readiness state are copied from the checkpoint, and the replay buffer, critic buffer and reference bank are reconstructed by replaying the parent's delta logs up to step 40. All training settings other than auditing match the main run, including b = 10,000 and g = 16. The critic-side seed (815001) differs from the main run's; it is shared with a companion branch from the same checkpoint that uses b = 5,000 (not part of this release).

Compute: 4 nodes × 8 GPUs (32 GPUs, the parent's training world size; checkpoint shards are tied to the world size). The run was started on cluster B (`submit_cluster_b_4node.sbatch`, `run_attach_cluster_b.sh`) and continued from step 72 on cluster A (`submit_cluster_a_4node.sbatch`, `run_attach_cluster_a.sh`) with identical training settings. The rollout engine uses throughput settings not used by the main run (grouped cascade attention for shared prefixes, prefix-affinity request routing, piecewise CUDA graphs, at most 128 concurrent sequences per judge engine); these do not change the training objective. We also tried enabling the grouped cascade-attention kernel (`SP_GROUPED_CASCADE=1`, implemented in `src/ac2/cascade_attn/`) in this run to speed up rollouts over the shared prefixes; the throughput gain was small.

## Configuration

| Setting | This run | AC2 main run |
|---|---|---|
| Initialization | AC2 main run, step-40 checkpoint and reconstructed buffers | Qwen3-4B-Thinking-2507, empty buffers |
| Audit fraction α (`SP_Q_AUDIT_DEN`, `SP_Q_AUDIT_CUT`) | 0 (`0`, `0`) | 1/4 (`4`, `1`) |
| Critic-side RNG seed (`SP_Q_RNG_SEED`) | 815001 | 804001 |
| Rollout-engine settings (`SP_GROUPED_CASCADE`, `SP_PREFIX_AFFINITY`, `SP_PREFIX_PILOT`, `SP_ROLLOUT_CUDAGRAPH_MODE`, `SP_REWARD_MAX_NUM_SEQS`) | `1`, `1`, `1`, `PIECEWISE`, `128` | not set |

All other settings follow the main run (Table 4).

## Files

- `README.md`: this document.
- `runner.py`: entry point. Builds the verl/Hydra configuration from `SP_*` environment variables, checks consistency and launches training.
- `run_attach_cluster_b.sh`: launch script for cluster B. Constructs the branch on first use (copies the parent's step-40 checkpoint and both delta logs), runs preflight checks and starts `runner.py` on a running holder; supports `SP_DRYRUN=1` (import check on the login node).
- `run_attach_cluster_a.sh`: the same launch script for cluster A (identical training environment, site-specific paths and preflight).
- `submit_cluster_b_4node.sbatch`: 4-node SLURM holder job for cluster B; starts Ray and attaches the driver automatically.
- `submit_cluster_a_4node.sbatch`: 4-node SLURM holder job for cluster A.
- `q_dataset.py`: dataset class (`data.custom_cls`), identical to the main run's.
- `sp_agent_loops.yaml`: agent-loop registry for prefix-continuation rollouts and critic queries.
- `build_cold_artifacts.py`: writes the empty loader stubs (replay seed, critic seed, reference bank) that the runner requires; the actual buffers are reconstructed from the parent's delta logs.
- `compose_dryrun.sh`: CPU-only check that composes the Hydra configuration and asserts this run's settings.
- `build_lineage_dashboard.py`: renders one dashboard covering the parent's steps up to 40 followed by this run.
- `refresh_dashboard.sh`: render the training dashboard PDF from `run_data/`.
- `q_build_common.py`: shard-writing and checksum helpers imported by `build_cold_artifacts.py`.
- `flashinfer_aot_warm.py`: ahead-of-time compilation of the flashinfer kernels, run before every launch (required with rollout tensor parallelism > 1).
- `setup.sh`: per-node runtime environment, sourced by the launch scripts on every node: `setup.sh cuda-compat` adds the CUDA 12.9 forward-compatibility libraries on cluster A nodes with older drivers, `setup.sh cuda-toolkit` selects the CUDA toolkit and include paths for the judge's JIT-compiled kernels.

## Running

Inputs: the parent run's step-40 checkpoint (`experiments/08_13_tiedq_seed192/run_data/checkpoints/global_step_40`, including `sp_q_optim/`, `q_state.json` and `sp_replay_state.json`) and its delta logs (`run_data/q_state_deltas.jsonl`, `run_data/replay_buffer_deltas.jsonl`); FineProofs-RL and IMO-ProofBench data prepared with `python -m ac2.data.prepare_fineproofs` and `python -m ac2.data.build_rubric_map`; and the judge `deepseek-ai/DeepSeek-V4-Flash` (revision `60d8d70770c6776ff598c94bb586a859a38244f1`) in the local Hugging Face cache, served by vLLM on the training nodes. The allocation must have 4 nodes × 8 GPUs.

```bash
E=experiments/08_26_s192b40_g10k_noaudit
python $E/build_cold_artifacts.py --out $E       # loader stubs
bash $E/compose_dryrun.sh                        # optional configuration check (CPU)
bash scripts/sbatch_env.sh $E/submit_cluster_b_4node.sbatch                # cluster B: holder attaches run_attach_cluster_b.sh itself
# cluster A instead: bash scripts/sbatch_env.sh $E/submit_cluster_a_4node.sbatch; bash $E/run_attach_cluster_a.sh <HOLDER_JOBID>
```

Outputs are written to `run_data/` in this folder. The scripts contain site-specific paths and SLURM settings and use the helper files `setup.sh` and `flashinfer_aot_warm.py` in this folder; adapt them before running elsewhere.

## Results

Not reported in the paper. The decoding-cost records for this run are joint length records and extend to step 123 (Table 3).
