# Value-function probe at step 80

`09_05_q_probe_step80`

**Paper reference:** Sec. 4.4, Critic diagnostics (Fig. 4); App. A.4: Fig. 11, Fig. 13 (left), Fig. 18 (left), the step-80 rows of Table 2 and Table 1.

## Description

The value-function probe of App. A.4 applied to the step-80 checkpoint of the AC2 main run (`../08_13_tiedq_seed192`). 256 ready training problems are sampled, one prefix $s$ is cut from each problem's latest stored trajectory at a uniformly random fraction between 5% and 95%, and the step-80 policy generates $g = 16$ full continuations from $s$. The endpoint value $v_i$ is the critic prediction after 10,000 new tokens when the continuation runs past them, and the terminal reward $r_i$ otherwise. The critic is also queried at the prefix, $V^\pi_\theta(s)$. The pipeline, its stages and its settings are described in `../08_15_q_probe_step40/README.md`; this probe was run with `run_probe_generic.sh` on 2 nodes × 8 GPUs (16 single-GPU vLLM engines for generation and critic queries, 2 judge servers), concurrently with the step-160 probe on two other nodes of the same allocation.

## Configuration

As in `../08_15_q_probe_step40/README.md`, with:

| Parameter | Value | Source |
|---|---|---|
| Checkpoint | AC2 main run, `global_step_80` | `SP_PROBE_STEP=80` |
| Probe set | 256 groups (ready problems at step 80: 2,592; prefixes drawn from the rollout dumps of steps 80 and 79) | `build.log` |
| Critic sites | prefix $s$ and chunk endpoint $s \cdot c_i$ | `probe_q.py` (prefix site enabled) |
| Reference-solution state | replay buffer and bank as of step 80 | `SP_Q_UPTO_STEP=80` |
| Hardware | 2 nodes × 8 GPUs | `SP_PROBE_NODES` (two nodes) |

## Files

| File | Description |
|---|---|
| `RESULT.txt` | Output of `analyze_probe.py` for this probe, including the per-group `PAIRS` dump and the reference-bank size. |
| `zpairs5.csv` | Per-group values: $\mathrm{mean}_i(r_i)$, $\mathrm{mean}_i(v_i)$, critic-scored fraction $f$, reference-bank membership, $V^\pi_\theta(s)$. |

## Running

Run the pipeline from `../08_15_q_probe_step40`:

```bash
SP_PROBE_STEP=80 SP_PROBE_DIR=<repo>/experiments/09_05_q_probe_step80 \
SP_PROBE_JOBID=<allocation job id> SP_PROBE_NODES=<node1>,<node2> \
  bash ../08_15_q_probe_step40/run_probe_generic.sh
```

The main run's `global_step_80` checkpoint, rollout dumps and delta logs are required (see the pipeline README).

## Results

From the paper (step 80):

- Against the mean terminal reward $\mathrm{mean}_i(r_i)$, the group-mean value $\mathrm{mean}_i(v_i)$ has MAE 0.065, compared with 0.211 for the prefix value $V^\pi_\theta(s)$ (256 prefixes; Sec. 4.4, Fig. 4 left and middle).
- Critic-based advantages $v_i - \mathrm{mean}_j(v_j)$ have MAE 0.150 relative to the Prefix GRPO advantages $r_i - \mathrm{mean}_j(r_j)$ across 2,640 responses in 165 groups containing at least one critic prediction; Pearson correlation 0.388 (Fig. 4 right).
- Groups with at least two critic-scored continuations: 157; MAE between their mean critic value and the mean terminal reward of the same continuations 0.169 (Table 2, Fig. 11); individual critic-scored continuations: 1,848.
- $\mathrm{mean}_i(v_i)$ vs. $\mathrm{mean}_i(r_i)$ by stratum: all 256 groups, MAE 0.065, signed bias −0.031; $f=0$: 91 groups (0 by construction); $0<f<1$: 98 groups, MAE 0.096, bias −0.070; $f=1$: 67 groups, MAE 0.106, bias −0.018 (Table 1).
