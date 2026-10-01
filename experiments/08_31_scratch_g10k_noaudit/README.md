# AC2 w/o Audit

`08_31_scratch_g10k_noaudit`

**Paper reference:** Fig. 3, left panel, teal line; Sec. 4.2, paragraph "AC2 without auditing"; App. A.2, Fig. 6 (same comparison against training steps); App. B, Table 3 (row "No audit, from initialization").

## Description

This run removes auditing from AC2 (α = 0). In the main run, ⌈α n⌉ of the n ready problems sampled at each step (α = 1/4) receive full rollouts, so that terminal rewards continue to be recorded on ready problems. Here every ready problem receives action chunks of at most b = 10,000 new tokens, whose endpoints are scored by the critic V^π_θ; terminal rewards on ready problems are observed only for continuations that finish within the chunk. As a consequence the critic error on audited groups (third panel of Fig. 2) is not available for this run, and on ready problems the critic is trained only on group-mean endpoint values.

Everything else follows the AC2 main run (`08_13_tiedq_seed192`): training starts from Qwen3-4B-Thinking-2507 with an empty replay buffer, an empty critic buffer and an empty reference bank; g = 16; local and global readiness with τ_global = 0.20 and τ_local = 0.18; 192 replayed prefixes and 192 refill rollouts per step; replay buffer capacity 256; 50,000-token response budget. The seed for the critic-side random draws (audit selection, reference choice, critic-batch sampling) is changed; the replay sampler's seed is unchanged.

Compute: the launch configuration targets 7 nodes × 8 GPUs (56 GPUs). Because 56 does not divide the per-step batch (3,072 trained continuations, 1,536 per actor minibatch), the trainer pads each data-parallel dispatch with duplicate rows whose loss mask is zero and assigns minibatch membership explicitly, so that each of the two actor minibatches still contains exactly 1,536 real continuations and the update equals that of the unpadded batch (`SP_DP_PAD=1`, `src/verl/verl/trainer/ppo/sp_dp_pad.py`). The padding is inactive when the world size divides the batch. Holder scripts for 4 and 8 nodes are also provided.

## Configuration

| Setting | This run | AC2 main run |
|---|---|---|
| Audit fraction α (`SP_Q_AUDIT_DEN`, `SP_Q_AUDIT_CUT`) | 0 (`0`, `0`) | 1/4 (`4`, `1`) |
| Critic-side RNG seed (`SP_Q_RNG_SEED`) | 831001 | 804001 |
| Readiness requires a solved problem (`SP_Q_READY_REQUIRE_BANK=1`) | from step 1 | enabled partway through training (absent from the initial launch script) |
| Padded data-parallel dispatch (`SP_DP_PAD`) | 1 | not set |
| Judge requests in flight (`SP_JUDGE_MAX_INFLIGHT`) | 20 × number of nodes | 160 |

All other settings follow the main run (Table 4).

## Files

- `README.md`: this document.
- `runner.py`: entry point. Builds the verl/Hydra configuration from `SP_*` environment variables, checks consistency, writes the run manifest and launches training.
- `run_attach_cluster_a.sh`: launch script for the cluster A. Runs preflight checks (judge weights, empty start artifacts, checkpoint world size, flashinfer AOT build) and starts `runner.py` on a running holder allocation with this run's environment.
- `submit_cluster_a_7node.sbatch`, `submit_cluster_a_8node.sbatch`, `submit_cluster_a_4node.sbatch`: SLURM holder jobs (7, 8 and 4 nodes) that start a Ray cluster across the allocation and keep the GPUs busy until the driver attaches.
- `q_dataset.py`: dataset class (`data.custom_cls`): replay-prefix sampling, readiness routing, per-row chunk caps and refill rollout counts.
- `sp_agent_loops.yaml`: agent-loop registry for prefix-continuation rollouts and critic queries.
- `build_cold_artifacts.py`: writes the empty replay-buffer seed, critic-buffer seed and reference bank that a run from initialization starts from.
- `compose_dryrun.sh`: CPU-only check that composes the Hydra configuration with this run's environment and asserts its settings.
- `refresh_dashboard.sh`: render the training dashboard PDF from `run_data/`.
- `q_build_common.py`: shard-writing and checksum helpers imported by `build_cold_artifacts.py`.
- `flashinfer_aot_warm.py`: ahead-of-time compilation of the flashinfer kernels, run before every launch (required with rollout tensor parallelism > 1).
- `setup.sh`: per-node runtime environment, sourced by the launch scripts on every node: `setup.sh cuda-compat` adds the CUDA 12.9 forward-compatibility libraries on cluster A nodes with older drivers, `setup.sh cuda-toolkit` selects the CUDA toolkit and include paths for the judge's JIT-compiled kernels.

## Running

Prerequisites (shared with the main run): the repository environment (`install.sh`); the FineProofs-RL training split and IMO-ProofBench validation split, prepared with `python -m ac2.data.prepare_fineproofs` and `python -m ac2.data.build_rubric_map` (the runner reads `$SELF_PLAY_DATA_DIR/train.parquet`, `$SELF_PLAY_DATA_DIR/test.parquet` and the validation map at `$SP_VAL_MAP`); and the base model `Qwen/Qwen3-4B-Thinking-2507` and the judge `deepseek-ai/DeepSeek-V4-Flash` (revision `60d8d70770c6776ff598c94bb586a859a38244f1`) in the local Hugging Face cache, since training runs offline. The judge is served by vLLM on the training nodes; no separate judge server is required. Passing `SP_JUDGE_MODEL=<snapshot dir>` to the attach script uses a local copy instead of the cache.

```bash
E=experiments/08_31_scratch_g10k_noaudit
python $E/build_cold_artifacts.py --out $E        # empty replay buffer, critic buffer, reference bank
bash $E/compose_dryrun.sh                         # optional configuration check (CPU)
bash scripts/sbatch_env.sh $E/submit_cluster_a_7node.sbatch             # holder allocation with a Ray cluster
bash $E/run_attach_cluster_a.sh <HOLDER_JOBID>      # preflight, then start runner.py on the holder
```

`run_attach_cluster_a.sh` appends trailing `KEY=VAL` arguments to the driver environment. Checkpoints, `metrics.jsonl` and rollout dumps are written to `run_data/` in this folder. The SLURM and shell scripts are written for our clusters: they contain site-specific repository, cache, data and log paths and SLURM account and partition settings, and they use the helper files `setup.sh` and `flashinfer_aot_warm.py` in this folder. Adapt these before running elsewhere.

## Results

From Sec. 4.2: the run shows slightly larger instability than AC2 but reaches a peak mean score of 18.66% at step 160, above the GRPO baseline's peak of 18.50%. The paper still recommends auditing, since AC2 reaches a higher peak and auditing allows the critic's accuracy to be tracked during training. Decoding FLOPs for this run combine step means with retained joint length records, and its cost data extend to step 173 (Table 3).
