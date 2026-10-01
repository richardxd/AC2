# AC2 with 75k response limit (branch at step 130)

`08_19_branch130_ctx75k`

**Paper reference:** Table 3, row "75k response limit". The run is not plotted in the paper.

## Description

This run continues the AC2 main run (`../08_13_tiedq_seed192`) from its step-130 checkpoint with the response budget raised from 50,000 to 75,000 tokens. The budget counts the replayed prefix and the newly generated tokens and also applies to evaluation responses. The token limits that depend on it (critic context limit, critic and actor packing limits) are raised together; every other setting, including $g = 16$, $b = 10{,}000$, $\alpha = 1/4$, $\tau_{\text{global}} = 0.20$, $\tau_{\text{local}} = 0.18$, the replay-buffer policy, the learning rates and the judge, is unchanged from the main run, and `runner.py` and `q_dataset.py` are identical to the main run's files.

The branch is constructed once, on the first launch: the parent's `global_step_130` checkpoint (actor weights, critic optimizer state, critic/readiness state, replay-buffer state) is copied into this run's checkpoint directory, and the parent's replay-buffer and critic-state delta logs are copied and replayed up to step 130, which reconstructs the replay buffer, critic buffer, reference bank and readiness table as they were at that step. Training resumes at step 131. Because actor and critic-optimizer shards are not resharded, the branch runs on the parent's topology, 4 nodes × 8 GPUs.

## Configuration

Differences from the main run, taken from `run_attach_cluster_a.sh`; all other values are as in `../08_13_tiedq_seed192/README.md`.

| Parameter | Main run | This run | Launch variable |
|---|---|---|---|
| Initialization | base model | AC2 main run, `global_step_130` (with its replay buffer, critic buffer and reference bank) | `BRANCH=130` in `run_attach_cluster_a.sh` |
| Response budget (prefix + new tokens, training and evaluation) | 50,000 | 75,000 | `SP_MAX_RESPONSE_LEN` |
| Critic context limit (2,048 + budget + 1,248) | 53,296 | 78,296 | `SP_Q_CTX_LIMIT` |
| Critic packing limit | 53,360 | 78,360 | `SP_Q_MAX_TOKEN_LEN` |
| Actor packing limit per GPU | 51,200 | 78,336 | `SP_PPO_MAX_TOKEN_LEN` |
| Critic seed and reference bank source | own (empty) | parent's directory | `SP_Q_SEED_DIR`, `SP_Q_BANK_DIR`, `SP_REPLAY_SEED_DIR` |
| Hardware | 4 nodes × 8 GPUs | 4 nodes × 8 GPUs | `submit_cluster_a_4node.sbatch` |

## Files

| File | Description |
|---|---|
| `runner.py` | Training entry point (identical to the main run's). |
| `q_dataset.py` | Training dataset class (identical to the main run's). |
| `sp_agent_loops.yaml` | Agent-loop registry (identical to the main run's). |
| `run_attach_cluster_a.sh` | Launch script: branch construction from the parent's step-130 checkpoint on the first launch, preflight checks (branch source, judge cache, checkpoint topology), training environment, driver launch. |
| `submit_cluster_a_4node.sbatch` | SLURM allocation: 4 nodes × 8 GPUs, Ray head and workers. |
| `compose_dryrun.sh` | Composes the Hydra configuration with the launch environment on a CPU node before requesting GPUs. |
| `build_cold_artifacts.py` | Builder for empty replay/critic artifacts, copied from the main run (not needed for the branch, which uses the parent's artifacts). |
| `q_group_structure.py` | Analysis of between-group and within-group structure of critic values (copied from the main run). |
| `build_lineage_dashboard.py` | Renders a dashboard that joins the parent's history up to step 130 with this run's steps 131 onward. |
| `refresh_dashboard.sh` | Call the combined dashboard builder. |
| `q_build_common.py` | Shard-writing and checksum helpers imported by `build_cold_artifacts.py`. |
| `flashinfer_aot_warm.py` | Ahead-of-time compilation of the flashinfer kernels, run before every launch (required with rollout tensor parallelism > 1). |
| `setup.sh` | Per-node runtime environment, sourced by the launch scripts on every node: `setup.sh cuda-compat` adds the CUDA 12.9 forward-compatibility libraries on cluster A nodes with older drivers, `setup.sh cuda-toolkit` selects the CUDA toolkit and include paths for the judge's JIT-compiled kernels. |

## Running

`runner.py` is the entry point, started by `run_attach_cluster_a.sh` inside a running 4-node allocation (`bash scripts/sbatch_env.sh experiments/08_19_branch130_ctx75k/submit_cluster_a_4node.sbatch`). The parent run directory (`../08_13_tiedq_seed192`) must contain `run_data/checkpoints/global_step_130` (with `actor/`, `sp_q_optim/`, `q_state.json`, `sp_replay_state.json`), `run_data/q_state_deltas.jsonl`, `run_data/replay_buffer_deltas.jsonl`, and its `replay_seed_cold/`, `q_seed/` and `reference_bank/` artifacts. The attach script looks for a frozen copy under `run_data/protected_checkpoints/global_step_130_frozen` and otherwise requires `SP_ALLOW_UNFROZEN_BRANCH=1` to branch from the parent's regular checkpoint directory:

```bash
bash compose_dryrun.sh
bash run_attach_cluster_a.sh <JOBID> SP_ALLOW_UNFROZEN_BRANCH=1
```

Data, models and environment are as for the main run (see `../08_13_tiedq_seed192/README.md`). The SLURM and attach scripts are written for our cluster and contain site-specific absolute paths, account and partition settings that must be edited. The attach and allocation scripts also use two helper files in this folder: `setup.sh` (per-node CUDA environment for the judge's JIT-compiled kernels) and, because rollout tensor parallelism is 4, `flashinfer_aot_warm.py` (flashinfer ahead-of-time warm-up run before every launch).

## Results

The paper lists this run only in Table 3: decoding cost computed from joint length records through step 161. No score for this run is reported in the paper text.
