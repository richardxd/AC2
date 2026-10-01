# GRPO, learning rate $1\times10^{-6}$

`07_15_handoff_lr1e6`

**Paper reference:** App. A.1, GRPO tuning (App. A.1, Fig. 5); Table 3 (row "GRPO, $10^{-6}$").

## Description

The lower arm of the three-point GRPO learning-rate sweep ($1\times10^{-6}$, $2\times10^{-6}$, $4\times10^{-6}$) used to select the GRPO baseline. The run executes the GRPO entry point `runner.py` (identical to the one in `../07_15_handoff`) unchanged: 256 fresh FineProofs-RL problems per step, $g = 16$ responses per problem, group-mean baseline without standard-deviation normalization, no KL term, no replay buffer and no critic, with all other settings as in the selected baseline (see `../07_15_handoff/README.md`). The only difference in training configuration from the selected baseline is the actor learning rate, $1\times10^{-6}$. Training starts from the base Qwen3-4B-Thinking-2507 model on 8 nodes × 8 GPUs.


## Configuration

Values are taken from `run_attach_cluster_c.sh` and `runner.py`.

| Parameter | Value | Launch variable / source |
|---|---|---|
| Initialization | `Qwen/Qwen3-4B-Thinking-2507`, fresh optimizer | `SP_ACTOR_MODEL` |
| Learning rate | $1\times10^{-6}$, no warmup | `SP_LR` |
| Problems per step / group size | 256 / 16 | `SP_TRAIN_BATCH_SIZE=256`, `SP_ROLLOUT_N=16` |
| Actor minibatches | 2 per step, 128 problems (2,048 responses) each | `SP_PPO_MINI_BATCH=128` |
| Advantage / KL | GRPO, no std normalization / 0 | `runner.py` |
| Actor optimizer and clipping | weight decay 0.01, gradient clip 0.3, $\epsilon_{\text{low}}=0.2$, $\epsilon_{\text{high}}^{\text{base}}=0.28$, dual-clip 3 | `runner.py` |
| Adaptive entropy | $H^\star=0.28$, $\delta_H=0.02$, $k\in[-0.08, 0.08]$, initial $k=0.06$ | `SP_AEC_*` |
| Response budget | 50,000 tokens; prompt 2,048 | `SP_MAX_RESPONSE_LEN=50000`, `SP_MAX_PROMPT_LEN=2048` |
| Length penalty / difficulty sampling | off / off | `SP_LENPEN_ENABLE=0`, `SP_DIFF_SAMPLING=0` |
| Training sampling | temperature 0.8, top-$p$ 1, top-$k$ unrestricted | `runner.py` |
| Evaluation | IMO-ProofBench, 16 samples per problem every 10 steps; temperature 0.8, top-$p$ 0.95, top-$k$ 20 | `SP_TEST_FREQ=10` |
| Judge | `deepseek-ai/DeepSeek-V4-Flash` (no revision pin), colocated with tensor parallelism 8; reward = points/7 | `SP_JUDGE_MODEL`, `SP_PASS_POINTS_MIN=6` |
| Hardware | 8 nodes × 8 GPUs; rollout tensor parallelism 4 | `submit_cluster_c_8node.sbatch`, `SP_ROLLOUT_TP=4` |
| Step target | 500 | `SP_TOTAL_STEPS` |

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
E=experiments/07_15_handoff_lr1e6
bash scripts/sbatch_env.sh $E/submit_cluster_c_8node.sbatch   # allocation + Ray cluster
bash $E/run_attach_cluster_c.sh <ALLOC_JOBID>                   # start runner.py on it
```

The scripts read their site paths from `.env` (`AC2_CLUSTER_C_ROOT`: the repository is expected at `$AC2_CLUSTER_C_ROOT/self-play`, the Hugging Face cache at `$AC2_CLUSTER_C_ROOT/hf-cache` and the data at `$AC2_CLUSTER_C_ROOT/data/fineproofs`). The defaults assume 8 nodes with 8 GPUs of 80 GB each. Checkpoints, rollouts and `metrics.jsonl` are written to `run_data/` in this folder.

## Results

The paper reports this run as one curve in Fig. 5; App. A.1 selects $2\times10^{-6}$ from the sweep. Table 3 lists its decoding cost, estimated from per-step mean lengths, through step 70.
