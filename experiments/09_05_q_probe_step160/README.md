# Value-function probe at step 160

`09_05_q_probe_step160`

**Paper reference:** App. A.4: Fig. 10 (right), Fig. 13 (right), Fig. 16, Fig. 18 (right), the step-160 rows of Table 2 and Table 1.

## Description

The value-function probe of App. A.4 applied to the step-160 checkpoint of the AC2 main run (`../08_13_tiedq_seed192`). 256 distinct ready problems are sampled from the checkpoint's readiness state, each problem's latest stored trajectory is cut at a uniformly random fraction between 5% and 95% to give a prefix $s$, and the step-160 policy generates 16 full continuations from $s$. The critic is queried at the prefix, $V^\pi_\theta(s)$, and at the end of the first 10,000 tokens of each continuation that runs past them, $V^\pi_\theta(s \cdot c_i)$; every continuation is scored by the training judge, giving $r_i$. The pipeline, its stages and its settings are described in `../08_15_q_probe_step40/README.md`; this probe was run with `run_probe_generic.sh` on 2 nodes × 8 GPUs (16 single-GPU vLLM engines for generation and critic queries, 2 judge servers), concurrently with the step-80 probe on two other nodes of the same allocation.

## Configuration

As in `../08_15_q_probe_step40/README.md`, with:

| Parameter | Value | Source |
|---|---|---|
| Checkpoint | AC2 main run, `global_step_160` | `SP_PROBE_STEP=160` |
| Probe set | 256 groups (ready problems at step 160: 3,637; all prefixes drawn from the step-160 rollout dump) | `build.log` |
| Critic sites | prefix $s$ and chunk endpoint $s \cdot c_i$ | `probe_q.py` (prefix site enabled) |
| Reference-solution state | replay buffer and bank as of step 160 | `SP_Q_UPTO_STEP=160` |
| Hardware | 2 nodes × 8 GPUs | `SP_PROBE_NODES` (two nodes) |

## Files

| File | Description |
|---|---|
| `RESULT.txt` | Output of `analyze_probe.py` for this probe, including the per-group `PAIRS` dump and the reference-bank size. |
| `zpairs5.csv` | Per-group values: $\mathrm{mean}_i(r_i)$, $\mathrm{mean}_i(v_i)$, critic-scored fraction $f$, reference-bank membership, $V^\pi_\theta(s)$. |

## Running

Run the pipeline from `../08_15_q_probe_step40`:

```bash
SP_PROBE_STEP=160 SP_PROBE_DIR=<repo>/experiments/09_05_q_probe_step160 \
SP_PROBE_JOBID=<allocation job id> SP_PROBE_NODES=<node1>,<node2> \
  bash ../08_15_q_probe_step40/run_probe_generic.sh
```

The main run's `global_step_160` checkpoint, rollout dumps and delta logs are required (see the pipeline README).

## Results

From the paper (step 160):

- Groups with at least two critic-scored continuations: 153; MAE between their mean critic value and the mean terminal reward of the same continuations 0.159 (Table 2).
- Critic-based vs. Prefix GRPO advantages: 2,672 continuations, MAE 0.146; individual critic-scored continuations: 1,892 (Table 2).
- $\mathrm{mean}_i(v_i)$ vs. $\mathrm{mean}_i(r_i)$ over all 256 groups: MAE 0.064, signed bias −0.041; by stratum, $f=0$: 89 groups (0 by construction), $0<f<1$: 99 groups, MAE 0.079, bias −0.059; $f=1$: 68 groups, MAE 0.126, bias −0.066 (Table 1).
- $V^\pi_\theta(s)$ against $\mathrm{mean}_i(r_i)$ is shown for 256 prefixes in Fig. 10 (right).
