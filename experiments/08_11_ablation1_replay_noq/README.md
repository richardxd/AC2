# Prefix GRPO

`08_11_ablation1_replay_noq`

**Paper reference:** Prefix GRPO in Fig. 1, right panel, and Sec. 4.1 ("The replay-buffer state distribution alone does not explain AC2's gains"); Table 3.

## Description

Prefix GRPO uses the same actor replay-buffer system as AC2 but no critic. Each step draws $n_{\text{refill}} = 192$ fresh problems, each rolled out once to refill the replay buffer $\mathcal{B}$ and not trained on, and $n_{\text{batch}} = 192$ prefixes cut from trajectories in $\mathcal{B}$ (multiples of 10,000 tokens in $[0, 0.9L]$). From every prefix the policy samples $g = 16$ continuations that always run to termination, and every continuation is scored by the judge. Advantages are the GRPO advantages on terminal rewards, $r_i - \mathrm{mean}_j(r_j)$, computed within each prefix group. Replayed prefix tokens are conditioning context and receive no loss.

The launch environment is that of AC2 (`../08_13_tiedq_seed192/run_attach_cluster_b.sh`) with every critic setting removed and `SP_Q_ENABLE=0`; the dataset class is AC2's with the readiness and routing layer deleted, so every prefix is routed to a full rollout. Batch shape, buffer policy, learning rate, entropy control, response budget, judge and evaluation are identical to AC2. The run starts from the base Qwen3-4B-Thinking-2507 model with an empty buffer and uses 8 nodes × 8 GPUs.

## Configuration

Values are taken from `run_attach_cluster_b.sh` and `runner.py`.

| Parameter | Value | Launch variable / source |
|---|---|---|
| Initialization | `Qwen/Qwen3-4B-Thinking-2507`; empty replay buffer | `SP_ACTOR_MODEL` (runner default), `SP_REPLAY_COLD_BOOTSTRAP=1` |
| Critic | none; all continuations run to termination | `SP_Q_ENABLE=0` |
| Group size $g$ | 16 | `SP_ROLLOUT_N` (runner default) |
| $n_{\text{batch}}$ / $n_{\text{refill}}$ | 192 / 192 | `SP_REPLAY_N=192`, `SP_TRAIN_BATCH_SIZE=384`, `SP_SCRATCH_INFLOW_ONLY=1` |
| Replay buffer | global FIFO of 256 trajectories, admission regardless of correctness, uniform over distinct problems | `SP_REPLAY_BOUND=256`, `SP_REPLAY_ADMISSION=ungated`, `SP_REPLAY_GLOBAL_SAMPLING=question` |
| Prefix cut grid / maximum fraction | 10,000 tokens / 0.90 | `SP_REPLAY_CUT_GRAIN`, `SP_REPLAY_CUT_HIGH` |
| Response budget (prefix + new tokens) | 50,000 tokens; prompt 2,048 | `SP_MAX_RESPONSE_LEN=50000` |
| Advantage / KL | GRPO, no std normalization / 0 | `runner.py` |
| Actor optimizer | LR $2\times10^{-6}$, no warmup, weight decay 0.01, gradient clip 0.3 | `SP_LR=2e-6` |
| Actor minibatches | 2 per step, 96 groups (1,536 continuations) each | `SP_PPO_MINI_BATCH=96` |
| PPO clipping | $\epsilon_{\text{low}}=0.2$, $\epsilon_{\text{high}}^{\text{base}}=0.28$, dual-clip 3 | `runner.py` |
| Adaptive entropy | $H^\star=0.28$, $\delta_H=0.02$, $k\in[-0.08, 0.08]$, initial $k=0.06$ | `SP_AEC_*` |
| Training sampling | temperature 0.8, top-$p$ 1, top-$k$ unrestricted | `runner.py` |
| Evaluation | IMO-ProofBench, 16 samples per problem every 10 steps; temperature 0.8, top-$p$ 0.95, top-$k$ 20 | `SP_TEST_FREQ=10`, `runner.py` |
| Judge | DeepSeek-V4-Flash at pinned revision `60d8d707`; reward = points/7 | `SP_JUDGE_REVISION`, `SP_PASS_POINTS_MIN=6` |
| Hardware | 8 nodes × 8 GPUs; rollout tensor parallelism 4 | `submit_cluster_b_8node.sbatch`, `SP_ROLLOUT_TP=4` |
| Step target | 500 | `SP_TOTAL_STEPS=500` |

## Files

| File | Description |
|---|---|
| `runner.py` | Training entry point. Same configuration builder as AC2's `runner.py` with the critic removed; refuses to start if critic switches (`SP_Q_ENABLE`, `SP_Q_SEPARATE`, `SP_Q_INTERLEAVE`, `SP_Q_LR_LADDER`, `SP_Q_AUDIT_DEN`) are set. |
| `replay_dataset.py` | Training dataset class (`SPReplayNoQDataset`): AC2's replay-prefix dataset with the readiness and routing layer removed; every prefix receives a full rollout. |
| `sp_agent_loops.yaml` | Agent-loop registry (prefix-continuation loop; the critic loop is registered but unused). |
| `build_cold_artifacts.py` | Writes the empty replay-buffer seed required by the loader; run once before the first launch. |
| `run_attach_cluster_b.sh` | Launch script: preflight checks, all training environment variables, and the driver launch into a running allocation. Refuses to resume a checkpoint that carries critic state. |
| `submit_cluster_b_8node.sbatch` | SLURM allocation: 8 nodes × 8 GPUs, Ray head and workers. |
| `refresh_dashboard.sh` | Renders the training dashboard PDF incrementally with `ac2.viz`. |
| `verify_ablation_diff.sh` | Diffs this run's launch environment against an AC2 launch script (`MAIN_RUN_DIR`) and lists removed, added and shared variables; it fails if anything other than critic settings and the run name differs. Its value check also expects `SP_TRAIN_BATCH_SIZE=256`, whereas both shipped launch scripts set 384, so that check reports a failure. |
| `test_ablation1_logic.py` | CPU-only tests of the replay and empty-buffer logic shared with AC2, plus checks specific to this configuration (`python test_ablation1_logic.py`). `test_single_factor_diff_against_the_main_run` expects `SP_TRAIN_BATCH_SIZE=256` and fails against the shipped launch script. |
| `q_build_common.py` | Shard-writing and checksum helpers imported by `build_cold_artifacts.py`. |
| `flashinfer_aot_warm.py` | Ahead-of-time compilation of the flashinfer kernels, run before every launch (required with rollout tensor parallelism > 1). |
| `setup.sh` | Per-node runtime environment, sourced by the launch scripts on every node: `setup.sh cuda-compat` adds the CUDA 12.9 forward-compatibility libraries on cluster A nodes with older drivers, `setup.sh cuda-toolkit` selects the CUDA toolkit and include paths for the judge's JIT-compiled kernels. |

## Running

`runner.py` is the entry point, started by `run_attach_cluster_b.sh` inside a running allocation:

1. Prepare the environment, data and models as for AC2 (see `../08_13_tiedq_seed192/README.md`).
2. `python build_cold_artifacts.py --out .` (once).
3. `bash scripts/sbatch_env.sh experiments/08_11_ablation1_replay_noq/submit_cluster_b_8node.sbatch`, then `bash run_attach_cluster_b.sh <JOBID> [KEY=VAL ...]`.

`MAIN_RUN_DIR=../08_13_tiedq_seed192 bash verify_ablation_diff.sh` prints the difference between the two launch environments (only `SP_Q_*` variables and the run name). The SLURM and attach scripts are written for our clusters and contain site-specific absolute paths, account and partition settings that must be edited. The attach and allocation scripts also use two helper files in this folder: `setup.sh` (per-node CUDA environment for the judge's JIT-compiled kernels) and, because rollout tensor parallelism is 4, `flashinfer_aot_warm.py` (flashinfer ahead-of-time warm-up run before every launch).

## Results

From the paper: Prefix GRPO falls well behind AC2 when measured by Decoding FLOPs (Fig. 1, right). Its decoding cost combines per-step means with retained joint length records, through step 120 (Table 3).
