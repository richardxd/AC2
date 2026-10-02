# Scaled R protocol proposal

## Purpose

R preparation amendment (2026-10-02, before retry): judge max output65536, client timeout540s/gateway480s, strict failure checks and$5 cap unchanged. R1's first cold smoke failed before update on a40000-token reasoning-only judge response; preserve it and use `r_launch.py r1 smoke-cold --attempt 2`, followed by `smoke-resume --attempt 2`. Other tasks default attempt1. E4/E8 costs and timings below were measured at40000, so revised long-run tail cost/time is unmeasured; they are planning scenarios rather than guarantees.

Fix the protocol supported by E8/E9 before bounded launch preparation. Richard must approve every long R launch and its judge budget. No long experiment has been started. Engineering smokes remain under the existing$5 cap and at most29minutes per launch, with bounded owned-process cleanup. This proposal is available for Richard's review while preparation continues.

## Model gate and its limits

Retain **Qwen3-4B-Thinking-2507**, revision`768f209d9ea81521153ed38c47d515654e938aea`. The measured-component AC2 projection is2.99days/200steps; the populated-Q capacity scenario is4.89days, or9.79days with the explicitly chosen2×planning allowance. This satisfies the ROADMAP's approximately10-day per-run rule on the planning assumptions below. The allowance is not a confidence bound, and its proximity to10days is material. Mature-Q timing and full60×4 validation are unmeasured. No runtime guarantee is claimed.

E9 used all60 canonical problems ×1 sample/model, the supervisor-approved reduction from×4, with16384 response tokens, temperature0.8/top-p0.95/top-k20, and seed192+canonical problem index. Both pinned models used their shipped thinking templates, verified against explicit thinking mode, on seven TP1 replicas with matched partition/concurrency4. [Gate receipt](receipts/e9-model-gate.json).

| Model | Mean score | Nonzero fraction | Tokens/GPU generation second | Truncated responses | Extractable proofs |
|---|---:|---:|---:|---:|---:|
| 4B | 11.9048% | 16.6667% | 191.616 | 45/60 | 15/60 |
| 1.7B | 0% | 0% | 250.790 | 22/60 | 5/60 |

These are protocol scores, not estimates of intrinsic mathematical ability. Original scoring requires a complete `<proof>…</proof>` after the last `</think>`;33 stopped1.7B outputs lacked an extractable proof. All55 unextractable1.7B responses scored zero without an API call. Four-B's45 such outputs all truncated. The [format audit](receipts/e9-format-audit.json) reproduces the extractor's lengths against every saved grade; no post-hoc answer normalization or regrading was used. The1.7B point estimates fail the fallback thresholds, but the runtime branch already selects4B.

Separate95% Hoeffding intervals for mean score are4B[0,29.44%] and1.7B[0,17.53%]; paired1.7B−4B is[−46.97%,23.16%]. They assume independent per-problem random draws and are not simultaneous bounds. One sample/problem cannot estimate within-problem variance. The results do not establish a statistically resolved quality difference. E9 cost20 actual API calls/$0.076600620, because missing proofs short-circuit; cumulative engineering167 calls/$0.867471996, leaving$4.132528004. All actual judge calls completed without error flags.

## Fixed training configuration

All arms use one g16 node, physicalGPUs1–7, FSDP world7, TP1/DP7, BF16, the original fused torch output kernel, parameter/optimizer offload and `sp_dp_pad=1`. GPU0 remains excluded. Full canonical5,227-row training data;60-problem corrected canonical validation. No permissive readiness fixture in R1–R4.

Common settings:16 prompt groups×4 policy samples, response16384, prompt2048, PPO mini-batch8 prompts×4=32 real continuations (two minibatches), learning rate2e−6, original AEC target0.28/delta0.02/range[−0.08,0.08]/initial0.06, clipping0.2/0.28, grad clip0.3, weight decay0.01, dual clip3, one PPO epoch, shuffle off, zero entropy/KL losses. Original training sampling0.8/1/unrestricted; validation0.8/0.95/20. vLLM eager, memory fraction0.45,16sequences,4096batched tokens. Strict judge adapter raises on HTTP/parse/truncation failures before updates and preserves legitimate no-proof and consumed-Q behavior.

AC2/Prefix use16 fresh one-sample inflow trajectories plus16×4 training continuations, ungated FIFO replay bound256, question sampling, cut fraction[0,0.9] on a4096-token grid. AC2 Q batch maximum64, FIFO1920, minimum valid4, original interleaved PPO1→Q→PPO2, Q grad clip0.2 and LR ladder, readiness thresholds0.20/0.18 with bank/nonzero gates, bank enabled from step1, audit denominator4/cut on. This differs from the paper's historical bank-gating bug through step42.

Preserve original resolved seeds: data.seed isNone (Torch's default generator), rollout seed isNone (replica rank offset), actor loader42, replay714001, Q804001 and judge42. Do not describe all randomness as seed192. Correct-only/no-audit retain shipped Q seeds826001/831001; those seed changes are an additional ablation confound.

Run200steps, save every20, evaluate every10 plus step0 (21events),60×4 samples/event. Resume restores optimizer, replay, Q, readiness and AEC state and avoids repeated initial validation. All artifacts go to `runs/research/<task>`; project scratch stays below45GB. No checkpoint pruning is requested by these launch configurations.

| Task | Change from common settings | Scientific output |
|---|---|---|
| R1 `r1` | GRPO, no replay/Q;64 policy trajectories/step | Validation mean versus cumulative exact Eq.6 decoding FLOPs |
| R2 `r2` | AC2 main | Same curve, readiness crossing/fraction, first crossing of observed R1 peak |
| R3 `r3` | Prefix GRPO, replay without Q | Same curve to assess replay's contribution |
| R4 `r4-2k` | AC2 chunk and replay cut grid2048 | Matched curve/readiness; mature timing not yet measured |
| R4 `r4-correct-only` | Judged-correct admission, pass points≥6;Q seed826001 | Same curve; explicitly changed seed |
| R4 `r4-no-audit` | Audit denominator0/cut off;Q seed831001 | Same curve; explicitly changed seed |

Reduced16k versus50k response,4k versus10k chunk,16×4 versus paper groups/samples,64 versus768 Q maximum, seven versus32GPUs, and official DeepSeek-V4.1-Flash versus paper's pinned local V4 revision all limit comparability. The projected200-step decoding budgets are only approximately2.34–3.19e18 FLOPs, far below the paper's0.79e20/1.99e20 crossing budgets. A scaled curve cannot claim reproduction of those absolute crossing points.

## Wall-clock and judge planning

[Projection receipt](receipts/e8-r-projections.json) hashes the four measured E8 states and E9 input. Cold and populated AC2 component maxima supply its early-state estimate. The measured populated Q batch had16records, so the capacity scenario scales the entire interleaved actor/Q phase by64/16, deliberately overcounting fixed PPO time. This linear scenario is not a measured mature-state bound. Validation extrapolates E9 cost/time by4, with balanced seven-GPU work plus four-way judge service; R's resident training engines differ from standalone E9 engines, so transfer is an assumption. Readiness savings are not assumed.

| Task | Observed-component days | Capacity days | 2× capacity allowance days | Observed-cost USD | Every trajectory graded at E4 mean USD |
|---|---:|---:|---:|---:|---:|
| R1 | 2.72 | 2.72 | 5.45 | 34.55 | 102.43 |
| R2 | 2.99 | 4.89 | 9.79 | 31.92 | 123.71 |
| R3 | 3.04 | 3.04 | 6.08 | 31.74 | 123.71 |
| R4 2k | 2.99 | 4.89 | 9.79 | 31.92 | 123.71 |
| R4 correct-only | 2.99 | 4.89 | 9.79 | 31.92 | 123.71 |
| R4 no-audit | 2.99 | 4.89 | 9.79 | 31.92 | 123.71 |
| R5 | Policy generation0.434hours | Merge/Q/startup pending smoke | Unmeasured | Not yet measured | 0.85 |

Each R4 row is an AC2 proxy; no ablation-specific mature rate is known. USD columns are planning scenarios, not hard maxima. Later policies may produce longer proofs or more extractable proofs. All six training arms run sequentially on the same seven GPUs; the per-arm10-day gate is not a ten-day budget for the whole roadmap. The existing$5 cap remains unchanged and cannot fund a scientific run. Default: prepare and test all launches, then leave long runs and higher budgets pending Richard's decision.

## Bounded training launch preparation

`r_launch.py` provides explicit `compose`, `smoke-cold`, `smoke-resume`, and `long` modes for every task above. The machine-readable candidate is [r_protocol.json](r_protocol.json). CPU composition must pass before each smoke. Cold smoke: initial validation on canonical problem0×4 (fixed before E9 outputs), full training data, one actual update and checkpoint1. Separate resume: restore checkpoint1, run step2 and save checkpoint2, with no repeated initial validation. Each launch is enclosed in `run_bounded.py --seconds 1740`; failed/time-limited launches remain failures. Judge clients and GPU training are serialized. E7 already supplies real consumed-Q/audit branch coverage under an explicitly permissive engineering fixture; two normal-readiness steps do not establish scientific readiness.

Example commands (TASK is one of the six table names):

```bash
source repro/env.sh
python repro/r_launch.py TASK compose
python repro/run_bounded.py --seconds 1740 --receipt-dir runs/r-smoke-launches/TASK-cold01 -- python repro/r_launch.py TASK smoke-cold
python repro/run_bounded.py --seconds 1740 --receipt-dir runs/r-smoke-launches/TASK-resume01 -- python repro/r_launch.py TASK smoke-resume
```

After separate approval, the prepared long command is `python repro/r_launch.py TASK long`; it has not been executed. Preparation results and remaining blockers are maintained in [CHECKLIST](CHECKLIST.md).

## R5 value probe

Use an approved normal-readiness R2 checkpoint with at least32 ready problems,32 distinct prefixes×4 full continuations,16384 response cap,4096 Q cut, prefix fraction5–95%, at least6144 tokens remaining, selection seed192. Policy sampling0.8/1/unrestricted; Q sampling greedy0, matching live Q/Table4 rather than the old probe script's0.8 default. Select latest eligible attempts at or before checkpoint, never by observed probe quality. If32 ready problems are unavailable, report the actual count and defer/revise before scientific generation; the wrapper refuses silent count reduction.

Local stages in `r5_launch.py`: prepare (FSDP merge and full state comparison, mandatory historical token-context verification, fixed prefix selection), generate, critic, judge, analyze. Every completed stage records output/input hashes; repeated completed stages verify them and skip work. The source R2 run must remain unchanged during this staged probe; changed source delta/data hashes reject resume. Bank state uses the actual cold seed and additions through checkpoint dataset stepS−1; historical wave verification usesS−2. Selected tier1 newest proof/bank is recorded; missing source entry IDs prevent an exact historical reference-selection claim.

Grade full prefix+continuation with the original no-reference training judge. Report group means, prefix values, individual cut values, matched cut-population means (at least2), and centered advantages `v_i−mean(v)` versus `r_i−mean(r)`. Retain every early-ending reward asv_i=r_i; separatef=0 groups that agree by construction. Missing measurements/failed grades abort; invalid Q excludes and lists the whole group. Correlations are undefined for constant vectors or fewer than3pairs. Small engineering fixtures support plumbing checks only, not efficacy conclusions.

Bounded preparation uses two prefixes×4 from E7 checkpoint6, a permissive-readiness/short-training fixture, and clearly labels every receipt. Its generation/Q budgets match proposed R5; the historical verifier uses that fixture's actual128-token chunk. This is a pipeline save/resume check, not an R5 scientific result. Each stage receives its own≤29minute bound; `verify-resume` must validate all saved stages without new generation/judge calls. Example common arguments: `--run runs/e7/ac2-7gpu-retry --step 6 --data runs/e7/data32 --out runs/r5-smoke --groups 2 --shards 2 --verify-step 6 --verify-chunk 128 --engineering-fixture`. Scientific use substitutes an approved R2 checkpoint/data,32groups/seven shards and4096 historical chunk, and omits the fixture flag.
