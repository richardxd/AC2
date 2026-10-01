# AC2 w/o Group & Audit & local readiness

`09_09_globalready_qtd_s192b20_noaudit`

**Paper reference:** Fig. 3, right panel, cyan line; Sec. 4.3, paragraph "Without local readiness"; App. A.2, Fig. 8 (ready fraction) and Fig. 6; App. B, Table 3 (row "AC2 w/o Group & Audit & local readiness (step-20 branch)").

## Description

This run ablates local readiness. It branches off the AC2 main run (`08_13_tiedq_seed192`) at its step-20 checkpoint: the actor/critic weights, the critic optimizer state and the readiness state are copied from that checkpoint, and the replay buffer, critic buffer and reference bank are reconstructed by replaying the parent's delta logs up to step 20. From step 21 the run uses the AC2 w/o Group & Audit configuration (`09_01_scratch_qtd_prefix16`): on ready problems each replay slot yields 16 distinct prefixes of one stored trajectory (1,000-token cut grid) and one action chunk of at most b = 10,000 new tokens from each, with advantage v − V^π_θ(s) and single-endpoint critic targets (at most 8 per slot); auditing is disabled (α = 0).

In addition, the per-problem conditions are removed from routing. With `SP_Q_READY_MODE=global`, every problem is treated as ready as soon as the global criterion holds, that is, the mean critic error ε over all prefixes sampled in the last five steps falls below τ_global = 0.20; the local threshold τ_local, the solved-problem requirement and the nonzero-prediction requirement are not applied. With `SP_Q_READY_GLOBAL_LATCH=1` the first opening is permanent, so from then on every replay prefix receives single action chunks. The per-problem readiness table is still maintained and logged (`q/ready_problems`) but does not affect routing (`src/verl/verl/trainer/ppo/sp_q_readiness.py`; tests in `tests/sp_q_ready/`). At the step-20 checkpoint the parent's global criterion was not met (it had been met at earlier steps), and the branch does not inherit an open latch; all replay prefixes therefore receive full-length groups of 16 until the criterion first holds in the branch.

Compute: one 4-node × 8-GPU allocation (32 GPUs, the parent's training world size; checkpoint shards are tied to the world size) on cluster B. The rollout engine uses throughput settings inherited from the step-50 branch (`09_05_qtd_ready_s192b50_noaudit`): grouped cascade attention for shared prefixes, prefix-affinity request routing, piecewise CUDA graphs, and at most 128 concurrent sequences per judge engine. These do not change the training objective. We also tried enabling the grouped cascade-attention kernel (`SP_GROUPED_CASCADE=1`, implemented in `src/ac2/cascade_attn/`) in this run to speed up rollouts over the shared prefixes; the throughput gain was small.

## Configuration

| Setting | This run | AC2 main run |
|---|---|---|
| Initialization | AC2 main run, step-20 checkpoint and reconstructed buffers | Qwen3-4B-Thinking-2507, empty buffers |
| Readiness routing (`SP_Q_READY_MODE`, `SP_Q_READY_GLOBAL_LATCH`) | global criterion only, latched (`global`, `1`) | per problem: global and local criteria, solved problem, nonzero prediction (not set) |
| Ready-problem sampling (`SP_Q_TD_ENABLE`, `SP_Q_TD_LANE`) | 16 prefixes × 1 action chunk (`1`, `short`) | 1 prefix × 16 action chunks (not set) |
| Ready-problem advantage | v − V^π_θ(s) | v_i − mean_j v_j |
| Advantage estimator (`SP_ADV_ESTIMATOR`) | `sp_segment` | `grpo` |
| Cut grid for ready-problem prefixes (`SP_Q_TD_CUT_GRAIN`) | 1,000 tokens | 10,000 tokens |
| Critic targets from ready problems | single endpoint, at most 8 per slot (`SP_Q_TD_ADMIT_PER_SLOT=8`) | group mean over ≥ 8 valid continuations |
| Audit fraction α (`SP_Q_AUDIT_DEN`, `SP_Q_AUDIT_CUT`) | 0 (`0`, `0`) | 1/4 (`4`, `1`) |
| Rollout-engine settings (`SP_GROUPED_CASCADE`, `SP_PREFIX_AFFINITY`, `SP_PREFIX_PILOT`, `SP_ROLLOUT_CUDAGRAPH_MODE`, `SP_REWARD_MAX_NUM_SEQS`) | `1`, `1`, `1`, `PIECEWISE`, `128` | not set |

All other settings, including the critic-side seed 804001, follow the main run (Table 4).

## Files

- `README.md`: this document.
- `runner.py`: entry point. Builds the verl/Hydra configuration from `SP_*` environment variables (including the readiness mode and the single-continuation settings), checks consistency and launches training.
- `run_attach_cluster_b.sh`: launch script for the cluster B. Constructs the branch on first use (copies the parent's step-20 checkpoint and both delta logs), runs preflight checks and starts `runner.py` on a running allocation; supports `SP_DRYRUN=1` (import check on the login node) and the inline-driver mode used by `launch_cluster_b.sh`.
- `launch_cluster_b.sh`: per-node launch step run by `submit_cluster_b_4node.sbatch` (one task per node); starts the Ray head and workers, writes the allocation info file and runs `run_attach_cluster_b.sh` on the first node.
- `q_dataset.py`: dataset class (`data.custom_cls`), identical to that of `09_05_qtd_ready_s192b50_noaudit`.
- `sp_agent_loops.yaml`: agent-loop registry for prefix-continuation rollouts and critic queries.
- `build_cold_artifacts.py`: writes the empty loader stubs (replay seed, critic seed, reference bank) that the runner requires; the actual buffers are reconstructed from the parent's delta logs.
- `compose_dryrun.sh`: CPU-only check that composes the Hydra configuration and asserts this run's settings.
- `q_build_common.py`: shard-writing and checksum helpers imported by `build_cold_artifacts.py`.
- `flashinfer_aot_warm.py`: ahead-of-time compilation of the flashinfer kernels, run before every launch (required with rollout tensor parallelism > 1).
- `setup.sh`: per-node runtime environment, sourced by the launch scripts on every node: `setup.sh cuda-compat` adds the CUDA 12.9 forward-compatibility libraries on cluster A nodes with older drivers, `setup.sh cuda-toolkit` selects the CUDA toolkit and include paths for the judge's JIT-compiled kernels.
- `submit_cluster_b_4node.sbatch`: SLURM allocation for cluster B (4 nodes × 8 GPUs): runs `launch_cluster_b.sh` as one srun step with one task per node. Submit with `scripts/sbatch_env.sh`.

## Running

Inputs: the parent run's step-20 checkpoint (`experiments/08_13_tiedq_seed192/run_data/checkpoints/global_step_20`, including `sp_q_optim/`, `q_state.json` and `sp_replay_state.json`) and its delta logs (`run_data/q_state_deltas.jsonl`, `run_data/replay_buffer_deltas.jsonl`); the FineProofs-RL and IMO-ProofBench data prepared with `python -m ac2.data.prepare_fineproofs` and `python -m ac2.data.build_rubric_map`; and the judge `deepseek-ai/DeepSeek-V4-Flash` (revision `60d8d70770c6776ff598c94bb586a859a38244f1`) in the local Hugging Face cache, served by vLLM on the training nodes. The allocation must have the parent's world size (4 nodes × 8 GPUs).

```bash
E=experiments/09_09_globalready_qtd_s192b20_noaudit
python -m pytest -q tests/sp_q_ready tests/sp_qtd tests/sp_segment
python $E/build_cold_artifacts.py --out $E
bash $E/compose_dryrun.sh                        # optional configuration check (CPU)
bash scripts/sbatch_env.sh $E/submit_cluster_b_4node.sbatch   # allocation + Ray + driver
```

`launch_cluster_b.sh` starts Ray on the four nodes, writes `sandbox_<JOBID>.info` and runs `run_attach_cluster_b.sh` inline on the first node. Outputs are written to `run_data/` in this folder. The scripts contain site-specific paths and use the helper files `setup.sh` and `flashinfer_aot_warm.py` in this folder; adapt them before running elsewhere.

## Results

From Sec. 4.3: with local readiness disabled, the mean score falls from 13.56% at step 30 to 13.10% at step 40, whereas AC2 w/o Group & Audit with local readiness reaches 15.33% at step 40. At that step 100% of sampled problems are ready in this run (by definition), compared with 18.23% with local readiness (Fig. 8). Decoding-FLOPs records for this run are joint length records and extend to step 40 (Table 3).
