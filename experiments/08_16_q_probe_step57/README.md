# Value-function probe at step 57

`08_16_q_probe_step57`

**Paper reference:** App. A.4: Fig. 10 (left), Fig. 12 (right), Fig. 15, Fig. 17 (right), the step-57 rows of Table 2 and Table 1.

## Description

The value-function probe of App. A.4 applied to the step-57 checkpoint of the AC2 main run (`../08_13_tiedq_seed192`). 256 distinct ready problems are sampled from the checkpoint's readiness state, each problem's latest stored trajectory is cut at a uniformly random fraction between 5% and 95% to give a prefix $s$, and the step-57 policy generates 16 full continuations from $s$. The critic is queried at the prefix, $V^\pi_\theta(s)$, and at the end of the first 10,000 tokens of each continuation that runs past them, $V^\pi_\theta(s \cdot c_i)$; every continuation is scored by the training judge, giving $r_i$. The pipeline, its stages and its settings are described in `../08_15_q_probe_step40/README.md`; this folder holds the outputs written by `../08_15_q_probe_step40/run_probe57.sh` (with `SP_PROBE_DIR` set to this folder) on 4 nodes × 8 GPUs (32 single-GPU vLLM engines for generation and critic queries, 4 judge servers).

248 of the 256 groups completed; the eight groups of one generation shard are missing (`backfill_shard.sh` in the pipeline folder re-runs such a shard).

## Configuration

As in `../08_15_q_probe_step40/README.md`, with:

| Parameter | Value | Source |
|---|---|---|
| Checkpoint | AC2 main run, `global_step_57` | `run_probe57.sh` (`SP_PROBE_STEP=57`) |
| Critic sites | prefix $s$ and chunk endpoint $s \cdot c_i$ | `probe_q.py` (prefix site enabled) |
| Reference-solution state | replay buffer and bank as of step 57 | `SP_Q_UPTO_STEP=57` |
| Hardware | 4 nodes × 8 GPUs | `run_probe57.sh` |

## Files

| File | Description |
|---|---|
| `RESULT.txt` | Output of `analyze_probe.py` for this probe, including the per-group `PAIRS` dump and the reference-bank size. |
| `zpairs.csv` | Per-group pairs: $\mathrm{mean}_i(r_i)$, $\mathrm{mean}_i(v_i)$, critic-scored fraction $f$, reference-bank membership. |
| `zpairs5.csv` | The same with a fifth column, $V^\pi_\theta(s)$. |
| `inspect_probe.py` | Prints the schemas of the probe artifacts (probe set, generation, critic and judge shards). |

## Running

Run the pipeline from `../08_15_q_probe_step40` with `SP_PROBE_DIR` pointing at this folder: `run_probe57.sh` (4 nodes; `SP_PROBE_JOBID` selects the allocation) or `run_probe_generic.sh` with `SP_PROBE_STEP=57`. Re-plot from the saved pairs with, e.g., `python ../08_15_q_probe_step40/plot_prefix.py --csv zpairs5.csv --out prefix_panels.png`.

## Results

From the paper (step 57):

- Groups with at least two critic-scored continuations: 144; MAE between their mean critic value and the mean terminal reward of the same continuations 0.121 (Table 2).
- Critic-based vs. Prefix GRPO advantages: 2,480 continuations, MAE 0.141; individual critic-scored continuations: 1,685 (Table 2).
- $\mathrm{mean}_i(v_i)$ vs. $\mathrm{mean}_i(r_i)$ over all 248 groups: MAE 0.050, signed bias +0.008; by stratum, $f=0$: 93 groups (0 by construction), $0<f<1$: 103 groups, MAE 0.066, bias +0.006; $f=1$: 52 groups, MAE 0.107, bias +0.026 (Table 1).
- $V^\pi_\theta(s)$ against $\mathrm{mean}_i(r_i)$ is shown for 248 prefixes in Fig. 10 (left).
