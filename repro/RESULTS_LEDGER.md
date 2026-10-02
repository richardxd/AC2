# AC2 paper and reproduction results

## Purpose

Inventory quantitative claims and settings from the local paper, with source, our measured value, producing run/config and scaling differences. An em dash means unmeasured, never zero. Paper values are claims, not independently reproduced findings. Engineering smokes/calibrations have run; no long R experiment has launched.

Source: local `paper/2609.39247.pdf`, arXiv v2 (1 Oct 2026), SHA256 `e89e2d289db589f0bcd7d576f9f2a82406d9f0188730028604993fff38690421`. First-hand rendered receipts: `runs/paper/page-NN.png`. Graph-only estimates below are deliberately approximate, not invented exact values.

## Performance and readiness claims

| Claim | Paper value | Source | Ours | Run/config/receipt | Scaling differences |
|---|---|---|---|---|---|
| GRPO peak mean score | 18.50% at step 120 | §1, §4.1, Fig. 1 | — | — | Pending protocol; API judge, 7 Ada |
| AC2 peak mean score | 20.57% | §1, §4.1, Fig. 1 | — | — | Same |
| AC2 first exceeds GRPO peak | step 90; 0.79e20 decoding FLOPs | §1, §4.1 | — | — | Same |
| GRPO compute at peak | 1.99e20 decoding FLOPs | §4.1 | — | — | Same |
| Compute / step reduction | 2.5× / 25% fewer steps | Abstract, §1 | — | — | Compare matched own baselines |
| Global readiness first crossing | end of step 7 | §4.1, Fig. 2 | — | — | Smaller batches change visitation |
| Ready sampled problems | about 70% at step 200 | §4.1, Fig. 2 | — | — | Same |
| Reference vs no-reference MAE, steps 40/80/120/160 | approximately 0.19/0.19/0.15/0.12 vs 0.24/0.25/0.20/0.16 (visual, ±0.01) | Fig. 2 right, p. 8 | — | — | Different judge and prefix sample |
| Reference comparison sample | about 1,500 prefixes/checkpoint from following 20 steps | §4.4 | — | — | — |
| Stale buffer peak | 17.90% | §4.2, Fig. 3 | — | — | 1,920 refill every 10 paper steps |
| No-audit peak | 18.66% at step 160 | §4.2, Fig. 3 | — | — | — |
| 2k chunks vs AC2 at step 50 | 16.03% vs 16.41% | §4.3, Fig. 7 | — | — | Cut spacing also changed |
| 2k chunks peak / final | 16.62% at step 100 / 13.57% at 120 | §4.3, Fig. 7 | — | — | Same confound |
| Correct-only vs AC2 at step 60 | 17.35% vs 16.88% | §4.3, Fig. 3 | — | — | Correct admission ≥6/7 |
| Correct-only later score | 14.67% at step 110 | §4.3 | — | — | — |
| No local readiness, no group/audit | 13.56% at step 30; 13.10% at 40 | §4.3 | — | — | Branch from main step 20 |
| With local readiness, no group/audit | 15.33% at step 40 | §4.3 | — | — | Starts from base, unlike branch |
| Ready fraction, local vs no local | 18.23% vs 100% at step 40 | §4.3, Fig. 8 | — | — | Unmatched histories |
| Best-of-16 plateau | roughly same; visually about 32–36%, no exact plateau reported | App. A.3, Fig. 9 p.19 | — | — | Best-of-4 would be a different metric |
| Prefix GRPO / no-group-no-audit curves | Prefix behind AC2; singleton comparable; exact peaks not tabulated | Figs. 1, 3, 6 | — | — | Do not manufacture exact peaks from pixels |

## Critic diagnostics

Different checkpoints sample different ready problems; cross-checkpoint differences are not a fixed-sample improvement. Groups with f=0 agree by construction and cannot demonstrate critic accuracy.

| Claim | Paper value | Source | Ours | Run/config/receipt | Scaling differences |
|---|---|---|---|---|---|
| Step-80 group endpoint MAE / prefix MAE | 0.065 / 0.211, 256 prefixes | Fig. 4, §4.4 | — | — | R5 pending checkpoint |
| Step-80 advantage MAE / Pearson r | 0.150 / 0.388 | Fig. 4, §4.4 | — | — | 2,640 responses, 165 groups with f>0 |
| Probe settings | 256 distinct ready problems; 16 full continuations; prefix fraction 5–95%; chunk 10k | App. A.4 | — | — | Prepared32×4 protocol; scientific run pending |
| Prefix vs terminal mean MAE | step 57: 0.153; step 160: 0.173 | Fig. 10 | — | — | 248 / 256 prefixes |
| Prefix vs group endpoint mean MAE | step 80: 0.186 | Fig. 13 left | — | — | 256 prefixes |
| Individual cut-state vs reward MAE | steps 40/57/80/160: 0.301/0.200/0.206/0.213 | Figs. 17–18 | — | — | Critic-scored continuations only |

Table 1: each row is `(groups, MAE, signed bias)`; source p.20. Ours and producing runs are unmeasured for every row.

| Step | All | f=0 | 0<f<1 | f=1 | Ours / receipt |
|---|---|---|---|---|---|
| 40 | 256, .129, +.112 | 94, .000, +.000 | 79, .188, +.169 | 83, .218, +.183 | — |
| 57 | 248, .050, +.008 | 93, .000, +.000 | 103, .066, +.006 | 52, .107, +.026 | — |
| 80 | 256, .065, −.031 | 91, .000, +.000 | 98, .096, −.070 | 67, .106, −.018 | — |
| 160 | 256, .064, −.041 | 89, .000, +.000 | 99, .079, −.059 | 68, .126, −.066 | — |

Table 2 (p.24): group statistic uses groups with ≥2 critic-scored continuations; centered statistic uses all continuations in f>0 groups.

| Step | Group n / MAE | Centered n / MAE | Individual n | Ours / run / receipt |
|---|---|---|---|---|
| 40 | 153 / .312 | 2592 / .177 | 1919 | — |
| 57 | 144 / .121 | 2480 / .141 | 1685 | — |
| 80 | 157 / .169 | 2640 / .150 | 1848 | — |
| 160 | 153 / .159 | 2672 / .146 | 1892 | — |

## Compute accounting and coverage

| Claim | Paper value | Source | Ours / run / receipt | Scaling differences |
|---|---|---|---|---|
| Eq. 6 coefficients | A=8,044,544,000; B=589,824 | App. B p.25 | 4B matches; [E8 receipt](receipts/e8-grpo-calibration.json) | Architecture-derived, no mean-length approximation |
| Architecture | 36 layers; hidden 2560; FFN 9728; query/KV heads 32/8; head dim 128; vocabulary 151936 | App. B | — | Verify actual model config |
| Mean-cost underestimate | GRPO 4.01% (steps 161–180), AC2 5.51% | App. B | — | Use joint moments in our exporter |
| GPU-hour regression | H=6.11+27.41(D/1e18), r=.874 | App. B, Fig. 19 | — | Not a g16 runtime predictor |
| Fit sample | 196 complete steps from 1–202; excludes six cached resumes; 32 GPUs | App. B | — | Includes judge/update/checkpoints; excludes val/startup/failures |

Table 3 (p.26), source coverage only, not outcome quality:

| Configuration | Length source | Last cost step | Ours / run / receipt |
|---|---|---|---|
| AC2 | Joint | 202 | — |
| GRPO 1e-6 | Means | 70 | — |
| GRPO 2e-6 | Means | 180 | — |
| GRPO 4e-6 | Means | 100 | — |
| Prefix GRPO | Means + joint | 120 | — |
| No audit branch at 40 | Joint | 123 | — |
| No audit from base | Means + joint | 173 | — |
| Correct-only | Means + joint | 117 | — |
| No group/audit branch at 50 | Joint | 117 | — |
| No group/audit from base | Means | 186 | — |
| No group/audit/local readiness branch at 20 | Joint | 40 | — |
| 2k chunks | Means | 128 | — |
| Stale buffer | Means + joint | 146 | — |
| 75k response branch | Joint | 161 | — |

## Paper configuration inventory

Values are from Table 4 and App. C unless otherwise stated. The populated cells below describe executed E8 calibration, not approved long R runs; [AC2 receipt](receipts/e8-ac2-replay-calibration.json) binds its resolved configuration and metrics.

| Setting | Paper value | Ours / run / receipt | Scaling differences |
|---|---|---|---|
| Train / val problems | roughly 5,200 / 60 (§4) | 5,227 / 60; [canonical receipt](receipts/e3-canonical.json) | One explicit mirror correction |
| Group, replay, refill, FIFO | 16; 192; 192; 256 | E8:4;16;16;256 | Reduced group statistics and visitation |
| Audit | 1/4, rounded up | E8 configured1/4; no ready routes yet | Actual branch coverage only in E7 fixture |
| Response / chunk / cut-grid / max prefix | 50,000 / 10,000 / 10,000 / 0.9 | E8:16384/4096/4096/.9 | Shorter horizon and cut spacing |
| Critic grid | 0, .1, …, 1; ties rounded down | — | — |
| Critic FIFO / train pairs / min valid | 1,920 / up to 768 / ≥8 | E8:1920/up to64/4; actual step2 trains16 | Lower maximum; early FIFO unsaturated |
| Actor LR / minibatch / updates | 2e-6 / 1536 continuations / 2 | E8:2e-6/32 continuations/2 | Same LR, smaller minibatch |
| Critic LR initial / floor | 2√2e-6 / 5√2e-7 | — | — |
| Critic LR controller | ratio √2, patience 2, factor .5 | — | — |
| Critic optimizer | AdamW (.9,.999), eps 1e-8, weight decay 0 | — | — |
| Critic clip / decoding / loss normalization | .2 / greedy ≤4 tokens / divide by 4 | — | — |
| Readiness window / thresholds | 5 steps / global .20 / local .18 | — | — |
| Reference bank | ≥6/7 reward; add once; gate starts step 42 in main run | — | Shipped launcher gate starts step 1 |
| Actor clip / weight decay / dual clip | .3 / .01 / 3 | E8:.3/.01/3 | Same settings |
| Actor KL / entropy bonus / epochs | 0 / 0 / 1, no minibatch shuffle | E8:0/0/1, shuffleFalse | Same settings |
| PPO low / base high clipping | .2 / .28 | E8:.2/.28 | Same settings |
| Adaptive entropy target / step / bounds / initial | .28 nats / .02 / [−.08,.08] / .06 | — | — |
| Train temperature / top-p / top-k | .8 / 1 / unrestricted | E8:.8/1/unrestricted | Same settings |
| Eval n / interval / temperature / top-p / top-k / budget | 16 / 10 / .8 / .95 / 20 / 50,000 | — | Candidate 4 samples, 16k |
| GRPO batch / group | 256 / 16 (4096 responses) | E8:16/4 (64 trained responses) | Scaled GRPO and AC2 both train64; AC2 adds16 inflow |
| LR sweep | 1e-6, 2e-6, 4e-6; selected 2e-6 (App. A.1) | — | No new selection without proposal |

## Engineering measurements, not paper performance results

| Fresh step | Generated tokens | Eq.6 decode FLOPs | Step / save seconds | Judge USD | Receipt |
|---|---|---|---|---|---|
| E8 GRPO | 926,086 | 1.1707487528091648e16 | 875.976 / 78.219 | 0.143165772 | [GRPO](receipts/e8-grpo-calibration.json) |
| E8 cold AC2 | 1,225,030 | 1.562175243845632e16 | 1114.992 / 207.842 | 0.129811104 | [cold AC2](receipts/e8-ac2-cold-calibration.json) |

Both use4B,7GPUs,16groups×4,16k response; AC2 adds16 inflow trajectories. Q was empty/skipped in cold AC2. These single-step measurements cannot establish scientific score improvement, runtime uncertainty, or mature readiness savings. E7 permissive readiness remains engineering coverage only.

Populated E8 AC2 step2:840,875 generated tokens,1.1208923499528192e16 decode FLOPs,916.102s step/111.313s save,$0.130034916 for31 calls ([receipt](receipts/e8-ac2-replay-calibration.json)). It trained16 of the configured maximum64 Q records; the proposal must account for that capacity difference. Normal readiness remains closed; these are engineering measurements.

E8 Prefix step1:1,244,762 generated tokens,1.5937332632027136e16 decode FLOPs,1010.848s step/75.812s save,$0.129121776 for21 calls ([receipt](receipts/e8-prefix-calibration.json)). This is an empty-buffer calibration; mature Prefix timing remains unmeasured.

E9 fixed60×1 step-zero protocol at16k (not paper×16):4B mean11.9048%, nonzero16.6667%,191.616tokens/GPU-generation-second;1.7B mean0%, nonzero0%,250.790tokens/GPU-generation-second ([receipt](receipts/e9-model-gate.json)). Four-B15/60 and1.7B5/60 extractable proofs;33 stopped1.7B responses have no extractable proof. Original format-sensitive zeros are preserved, so these numbers cannot establish intrinsic mathematical ability. Mean-score95% Hoeffding intervals[0,29.44%]/[0,17.53%] overlap; no quality-superiority claim. Generation times include prefill/queueing and exclude startup/judging. [Format audit](receipts/e9-format-audit.json).

E8/E9 planning only: R1/R2/R3 observed-component projections2.72/2.99/3.04days per200steps; AC2 Q-capacity scenario4.89days and2×allowance9.79days. R4 uses AC2 proxies. These are not measured scientific runs or confidence bounds. [Projection receipt](receipts/e8-r-projections.json), [protocol proposal](PROPOSAL_R.md). All paper performance targets above remain unmeasured.

R1–R4 launch preparation: all six training configs pass bounded cold/save/resume with nonzero updates and complete checkpoint/replay checks; only the fixed initial one-problem validation fixture is evaluated. These are not learning curves or ablation results. R5 complete fixture passes full weight comparison,6/6 historical contexts,2groups×4continuations/103832tokens,10/10 parsed critic values, grading/analysis and full resume ([receipt](receipts/r5-full-pipeline.json)). Four explicitly authorized API calls cost$0.019788816; four other attempts lack extractable proofs and retain original no-proof zeros. Saved report has two complete groups and undefined correlations because Q values are constant (and group-level pairs number only2); it supports no Fig.4 efficacy claim. Cumulative547completed calls cost$2.718861168 under$5, leaving$2.281138832. Scientific paper targets remain unmeasured.
