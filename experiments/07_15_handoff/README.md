# GRPO (learning rate $2\times10^{-6}$, selected baseline)

`07_15_handoff` (run name `07_15_rerun_baseline_handoff`)

**Paper reference:** the GRPO baseline throughout the paper: Fig. 1, right panel; reference line in Fig. 3 and Fig. 6; App. A.1 (Fig. 5); Fig. 9; App. B and Table 3 (row "GRPO, $2\times10^{-6}$").

## Description

GRPO trained from the base Qwen3-4B-Thinking-2507 model on FineProofs-RL. Each step samples 256 fresh problems and $g = 16$ responses per problem (4,096 responses). Every response is rolled out to completion and scored by the training judge. The advantage is $R(x, y_i) - \mathrm{mean}_j R(x, y_j)$, without division by the group standard deviation, and the KL coefficient is 0. There is no replay buffer and no critic. All 4,096 responses are trained on, in two actor minibatches of 2,048 responses. The actor optimizer, PPO clipping, adaptive entropy control, response budget, judge and evaluation protocol are those of AC2 (Table 4). The learning rate, $2\times10^{-6}$, is the best of the three-point sweep in App. A.1; the other two arms are `../07_15_handoff_lr1e6` and `../07_15_handoff_lr4e6`.


## Configuration

Values are taken from `run_attach_cluster_c.sh` and `runner.py`.

| Parameter | Value | Launch variable / source |
|---|---|---|
| Initialization | `Qwen/Qwen3-4B-Thinking-2507`, fresh optimizer | `SP_ACTOR_MODEL` |
| Learning rate | $2\times10^{-6}$, no warmup | `SP_LR` |
| Problems per step / group size | 256 / 16 | `SP_TRAIN_BATCH_SIZE=256`, `SP_ROLLOUT_N=16` |
| Actor minibatches | 2 per step, 128 problems (2,048 responses) each | `SP_PPO_MINI_BATCH=128` |
| Advantage / KL | GRPO, no std normalization / 0 | `algorithm.norm_adv_by_std_in_grpo=False`, `use_kl_loss=False` |
| Actor optimizer and clipping | weight decay 0.01, gradient clip 0.3, $\epsilon_{\text{low}}=0.2$, $\epsilon_{\text{high}}^{\text{base}}=0.28$, dual-clip 3 | `runner.py` |
| Adaptive entropy | $H^\star=0.28$, $\delta_H=0.02$, $k\in[-0.08, 0.08]$, initial $k=0.06$ | `SP_AEC_*` |
| Response budget | 50,000 tokens; prompt 2,048 | `SP_MAX_RESPONSE_LEN=50000`, `SP_MAX_PROMPT_LEN=2048` |
| Length penalty / difficulty sampling | off / off | `SP_LENPEN_ENABLE=0`, `SP_DIFF_SAMPLING=0` |
| Training sampling | temperature 0.8, top-$p$ 1, top-$k$ unrestricted | `runner.py` |
| Evaluation | IMO-ProofBench, 16 samples per problem every 10 steps; temperature 0.8, top-$p$ 0.95, top-$k$ 20 | `SP_TEST_FREQ=10` |
| Judge | `deepseek-ai/DeepSeek-V4-Flash` (no revision pin), colocated with tensor parallelism 8; reward = points/7 | `SP_JUDGE_MODEL`, `SP_PASS_POINTS_MIN=6` |
| Hardware | 8 nodes × 8 GPUs; rollout tensor parallelism 4 | `submit_cluster_c_8node.sbatch`, `SP_ROLLOUT_TP=4` |
| Step target | 500 (stopped earlier; the paper uses steps up to 180) | `SP_TOTAL_STEPS` |

## Files

| File | Description |
|---|---|
| `runner.py` | GRPO training entry point (identical in the three sweep folders). Builds the verl PPO/Hydra configuration from `SP_*` environment variables, writes the manifest and calls `verl.trainer.main_ppo.run_ppo`. |
| `run_attach_cluster_c.sh` | Launch script: sets the run's configuration (`SP_*` variables, including the learning rate), runs the flashinfer warm-up and starts `runner.py` on the head node of a running allocation. Trailing `KEY=VAL` arguments override the pinned values. |
| `submit_cluster_c_8node.sbatch` | SLURM allocation (8 nodes × 8 GPUs): starts a Ray head and workers and writes `alloc_<JOBID>.info` for the launch script. Submit with `scripts/sbatch_env.sh`. |
| `setup.sh` | Per-node runtime environment (`setup.sh cuda-toolkit`: CUDA toolkit and include paths for the judge's JIT-compiled kernels), sourced on the driver node. |
| `flashinfer_aot_warm.py` | Ahead-of-time compilation of the flashinfer kernels, run before every launch (required with rollout tensor parallelism > 1). |

## Running

Install the environment, prepare the data and place the base model and judge in the Hugging Face cache as described in the top-level README (the run is offline, `HF_HUB_OFFLINE=1`). Then, from the repository root:

```bash
E=experiments/07_15_handoff
bash scripts/sbatch_env.sh $E/submit_cluster_c_8node.sbatch   # allocation + Ray cluster
bash $E/run_attach_cluster_c.sh <ALLOC_JOBID>                   # start runner.py on it
```

The scripts read their site paths from `.env` (`AC2_CLUSTER_C_ROOT`: the repository is expected at `$AC2_CLUSTER_C_ROOT/self-play`, the Hugging Face cache at `$AC2_CLUSTER_C_ROOT/hf-cache` and the data at `$AC2_CLUSTER_C_ROOT/data/fineproofs`). The defaults assume 8 nodes with 8 GPUs of 80 GB each. Checkpoints, rollouts and `metrics.jsonl` are written to `run_data/` in this folder.

## Results

From the paper:

- Peak validation mean score 18.50%. GRPO reaches it at step 120, after 1.99e20 Decoding FLOPs; AC2 first exceeds it at step 90, after 0.79e20 Decoding FLOPs.
- App. A.1 selects $2\times10^{-6}$ over $1\times10^{-6}$ and $4\times10^{-6}$ for all comparisons.
- Decoding cost is estimated from per-step mean lengths through step 180 (Table 3). Where complete records exist (steps 161–180), the mean-based estimate falls 4.01% below the exact cost.
- On best-of-16 (Fig. 9), AC2 remains more compute-efficient, and the two methods plateau at roughly the same score.
