# Value-function probe at step 40

`08_15_q_probe_step40`

**Paper reference:** App. A.4: Fig. 12 (left), Fig. 14, Fig. 17 (left), the step-40 rows of Table 2 and Table 1. This folder also contains the probe pipeline used for all four checkpoints (steps 40, 57, 80 and 160), including the step-80 diagnostics of Sec. 4.4 (Fig. 4).

## Description

The probe measures, on a checkpoint of the AC2 main run (`../08_13_tiedq_seed192`), how well the critic's values agree with terminal rewards that AC2 does not observe during training. At the checkpoint, 256 distinct problems that are ready according to the checkpoint's readiness state are selected, and each problem's latest stored trajectory is cut at a uniformly random fraction between 5% and 95% of its length to obtain a prefix $s$. The checkpoint's policy then generates 16 full continuations from $s$, without the action-chunk cut. Let $r_i$ be the judge reward of the $i$th continuation and $c_i$ its first 10,000 tokens (or fewer if it ends sooner). The probe sets $v_i = V^\pi_\theta(s \cdot c_i)$ if the continuation runs past 10,000 tokens and $v_i = r_i$ otherwise, and compares $V^\pi_\theta(s)$, $\mathrm{mean}_i(v_i)$ and $V^\pi_\theta(s \cdot c_i)$ against $\mathrm{mean}_i(r_i)$ and $r_i$ (App. A.4). Since the critic shares weights with the policy, one merged model serves both the continuations and the critic queries.

At step 40 the critic was evaluated only at the chunk endpoints $s \cdot c_i$; the prefix value $V^\pi_\theta(s)$ is included in the probes at steps 57, 80 and 160. The step-40 probe ran on 4 nodes × 8 GPUs.

## The probe pipeline

The scripts below are shared by `../08_16_q_probe_step57`, `../09_05_q_probe_step80` and `../09_05_q_probe_step160`. Each stage writes sharded JSONL files to a probe directory (`SP_PROBE_DIR`, default this folder) and skips shards that already exist.

1. **Merge the checkpoint** (`merge_ckpt.sh`). Converts the FSDP actor shards of `global_step_N` of the main run into a Hugging Face model (`model_hf/`) with `verl.model_merger`, and sets its dtype to bfloat16.
2. **Build the probe set** (`build_probe_set.py`). Reads the ready set from the checkpoint's `q_state.json`, maps problems to their latest stored trajectory by scanning the rollout dumps at or before step N (up to 12 files back), and samples 256 distinct ready problems and one cut per problem. Cut positions are uniform over $[0.05, 0.95]$ of the trajectory's response length; cuts that would leave fewer than 12,000 tokens of the 50,000-token budget are moved earlier so that a continuation can reach the 10,000-token chunk length. Output: `probe_set.jsonl`.
3. **Generate continuations** (`probe_gen.py`, launched per node by `probe_gen_node.sh`). One single-GPU vLLM engine per GPU; each prefix is continued 16 times with the training sampling parameters (temperature 0.8, top-$p$ 1, top-$k$ unrestricted) up to the 50,000-token response budget. Output: `gen/gen.shard*.jsonl`.
4. **Query the critic** (`probe_q.py`, launched per node by `probe_q_node.sh`). Builds the critic context with the training prompt code imported from `verl.trainer.ppo.sp_q_readiness`, at the prefix $s$ and at $s \cdot c_i$ for each continuation that exceeds 10,000 tokens. The reference solution shown to the critic is reconstructed from the replay-buffer and reference-bank delta logs as of step N. `python probe_q.py --verify --run-dir <run_data> --model <model_hf>` checks the reconstructed contexts token-for-token against critic calls logged during training (no GPU needed). Output: `q/q.shard*.jsonl`.
5. **Judge** (`probe_judge.py`, launched per node by `probe_judge_node.sh`). Serves DeepSeek-V4-Flash with tensor parallelism 8 on each node and scores every continuation with the training judge client (`ac2.rewards.ds4_finegrained_judge.compute_score`, no reference solution), giving $r_i$. Output: `judged/judged.shard*.jsonl`.
6. **Analyse** (`analyze_probe.py`). Joins the three stages by (problem, continuation) and reports: $\mathrm{mean}_i V^\pi_\theta(s \cdot c_i)$ against the mean $r_i$ of the same continuations for groups with at least two critic-scored continuations; $V^\pi_\theta(s \cdot c_i)$ against $r_i$ per continuation; $V^\pi_\theta(s)$ against $\mathrm{mean}_i(r_i)$; and $\mathrm{mean}_i(v_i)$ against $\mathrm{mean}_i(r_i)$ over all groups, each with bias, correlation and MAE. `--dump-pairs` additionally prints per-group tuples ($\mathrm{mean}_i(r_i)$, $\mathrm{mean}_i(v_i)$, critic-scored fraction $f$, reference-bank membership, and $V^\pi_\theta(s)$ when available), which are saved as `zpairs*.csv`. The comparison between critic-based and Prefix GRPO advantages reported in the paper uses the same per-continuation outputs but is not computed by `analyze_probe.py`.

`run_probe_generic.sh` runs stages 1–6 for any checkpoint inside an existing allocation, restricted to a list of nodes:

```bash
SP_PROBE_STEP=80 SP_PROBE_DIR=<repo>/experiments/09_05_q_probe_step80 \
SP_PROBE_JOBID=<allocation job id> SP_PROBE_NODES=<node1>,<node2> \
  bash run_probe_generic.sh
```

It uses 8 generation and critic shards per node and one judge server per node. `run_probe57.sh` is the same chain on 4 nodes for step 57. `run_rest.sh` and `run_stage_c.sh` run stages 4–6 and stages 5–6 for the step-40 probe, whose artifacts live in this folder, and `backfill_shard.sh` re-runs one lost generation shard through stages 3–5 and merges it back.

## Configuration

| Parameter | Value | Source |
|---|---|---|
| Checkpoint | AC2 main run, `global_step_40` | `merge_ckpt.sh` (`SP_PROBE_STEP`, default 40) |
| Problems | 256 distinct ready problems, latest stored trajectory each | `build_probe_set.py --n 256 --scan-back 12` |
| Cut position | uniform fraction in $[0.05, 0.95]$; at least 12,000 tokens of budget left | `--min-frac`, `--max-frac`, `--min-budget` |
| Continuations per prefix | 16, full length, response budget 50,000 tokens | `probe_gen.py --n 16`, `build_probe_set.py --resp-cap` |
| Generation sampling | temperature 0.8, top-$p$ 1, top-$k$ unrestricted | `probe_gen.py` |
| Chunk length for critic scoring | 10,000 tokens | `--budget-g 10000` |
| Critic query | training prompt (`reward_horizon` variant), reference from replay buffer / bank at step N, judge-passing references only, up to 4 generated tokens at temperature 0.8 | `probe_q.py` (`--variant`, `--gen-reserve 4`, `--temperature`) |
| Judge | DeepSeek-V4-Flash, training (no-reference) prompt, reward = points/7 | `probe_judge.py`, `probe_judge_node.sh` |
| Hardware | 4 nodes × 8 GPUs: 32 single-GPU vLLM engines; 4 judge servers (tensor parallelism 8) | `run_rest.sh` |
| Random seed for the probe set | 192 | `build_probe_set.py --seed` |

## Files

| File | Description |
|---|---|
| `merge_ckpt.sh` | Stage 1: merge a main-run checkpoint into a bfloat16 Hugging Face model. |
| `build_probe_set.py` | Stage 2: sample the ready problems and prefixes. |
| `probe_gen.py`, `probe_gen_node.sh` | Stage 3: continuation generation (one engine per GPU) and its per-node launcher. |
| `probe_q.py`, `probe_q_node.sh` | Stage 4: critic queries at the prefix and at the chunk endpoint, with `--verify` mode; per-node launcher. |
| `probe_judge.py`, `probe_judge_node.sh` | Stage 5: judge scoring and the per-node judge server plus client launcher. |
| `analyze_probe.py` | Stage 6: joins the stages and prints the summary statistics (written as `RESULT.txt` by the chain scripts). |
| `run_probe_generic.sh` | End-to-end chain for any checkpoint on a subset of an allocation's nodes (used for steps 80 and 160). |
| `run_probe57.sh` | End-to-end chain on 4 nodes for step 57. |
| `run_rest.sh`, `run_stage_c.sh` | Partial chains (stages 4–6, stages 5–6) for the step-40 probe. |
| `backfill_shard.sh` | Re-runs a single generation shard through generation, critic and judge and merges it into an existing probe. |
| `zpairs.csv` | Step-40 per-group pairs: $\mathrm{mean}_i(r_i)$, $\mathrm{mean}_i(v_i)$, critic-scored fraction $f$, reference-bank membership. |
| `plot_zpairs.py` | Plots a scatter of $\mathrm{mean}_i(v_i)$ against $\mathrm{mean}_i(r_i)$ coloured by $f$, for all groups and for problems in the reference bank. |
| `plot_compare.py` | Step 40 vs. step 57 panels (group means, and $V^\pi_\theta(s)$ at step 57). |
| `plot_compare4.py` | Group-mean and prefix-value panels for steps 40, 57, 80 and 160 side by side. |
| `plot_prefix.py` | $V^\pi_\theta(s)$ against $\mathrm{mean}_i(r_i)$ and against $\mathrm{mean}_i(v_i)$ (used for step 57). |
| `inspect_qfail.py` | Qualitative inspection script: prints continuations from groups whose 16 continuations all scored 0 but whose $\mathrm{mean}_i(v_i)$ lies in $[0.2, 0.4]$, with the text the critic saw at the cut and the end of the rollout. |
| `why_noref.py` | Diagnoses why some critic calls on ready problems carry no reference solution (source trajectory not judge-passing). |
| `setup.sh` | Per-node runtime environment, sourced by the launch scripts on every node: `setup.sh cuda-compat` adds the CUDA 12.9 forward-compatibility libraries on cluster A nodes with older drivers, `setup.sh cuda-toolkit` selects the CUDA toolkit and include paths for the judge's JIT-compiled kernels. |

The large intermediate artifacts (`model_hf/`, `probe_set.jsonl`, `gen/`, `q/`, `judged/`) are not included.

## Running

All stages run inside an existing SLURM allocation through `srun --overlap`; the node launchers source `setup.sh` in this folder, and use the repository's virtual environment. Required inputs are the main run's `run_data/` (checkpoint `global_step_N`, rollout dumps, `replay_buffer_deltas.jsonl`, `q_state_deltas.jsonl`), the FineProofs-RL `train.parquet` used in training (`--data-dir`, joined by row index), and the DeepSeek-V4-Flash weights. For a new checkpoint, use `run_probe_generic.sh` as shown above. The scripts default to absolute paths of our cluster (`SELF_PLAY_ROOT`, `--run-dir`, `--data-dir`, cache directories) that must be changed for another site.

## Results

From the paper (step 40):

- Groups with at least two critic-scored continuations: 153; MAE between their mean critic value and the mean terminal reward of the same continuations 0.312 (Table 2).
- Critic-based vs. Prefix GRPO advantages: 2,592 continuations, MAE 0.177; individual critic-scored continuations: 1,919 (Table 2).
- $\mathrm{mean}_i(v_i)$ vs. $\mathrm{mean}_i(r_i)$ over all 256 groups: MAE 0.129, signed bias +0.112; by stratum, $f=0$: 94 groups (0 by construction), $0<f<1$: 79 groups, MAE 0.188, bias +0.169; $f=1$: 83 groups, MAE 0.218, bias +0.183 (Table 1).
