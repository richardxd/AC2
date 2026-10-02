# Value probe preparation audit

## Purpose

Record source findings before R5 launch preparation. No value-probe measurement has been made. The original pipeline is `experiments/08_15_q_probe_step40`; its historical empty reference bank differs from the current AC2 configuration.

The research role requires correct proof labels, references available at the checkpoint, and token-for-token context verification. The ML role needs a local GPU1–7 wrapper, bounded generation and an explicit failure when a measurement stage is incomplete. Neither permits filling missing probe evidence with fabricated values.

## Source findings

- `probe_q.py:build_ref_map_rollouts` accepts `bool(acc)` as passing. Current fine-grained scores can be fractional, so a positive failed proof can become a reference. Prefer the recorded binary `prover_judge_score`, otherwise rubric points at least6 or fractional accuracy at least6/7. Check both acceptance and rejection before use.
- That harvest is called without `steps`, so a probe at an earlier checkpoint can read future rollout files. Restrict global rollout steps to the checkpoint or earlier.
- **Correction to the initial checklist:** `do_verify` already returns1 for zero rows or any context mismatch, and `main` propagates it through `SystemExit`. No exit-status fix is needed. Exercise both outcomes in preparation.
- `QHarness.trajectory_ref_proof` decodes with `skip_special_tokens=True`; the probe usesFalse. Align extraction before claiming exact reference parity.
- The original probe rebuilds tier1 replay references, but ignores the active tier2 reference bank. `QHarness.reference_for` prefers the trajectory proof, then the bank. The bank is add-once (`setdefault`) and its additions are preserved in `q_state_deltas.jsonl`; cold seed shards must also be included. Current E8 cold AC2 admitted11 bank entries, so the original empty-bank assumption demonstrably does not apply.
- Historical wave reconstruction uses the wrong boundary for current training order: global stepS maps to dataset stepS−1; the Q wave runs before `sp_replay.observe_and_update`, which appends that step's admissions/evictions. Historical wave references therefore use deltas throughS−2. A post-checkpoint probe at global stepS uses state throughS−1. Verify these distinct boundaries with an admission/eviction fixture; do not use a post-step state to reconstruct a pre-update wave.
- Rollout dumps do not identify the exact source replay entry. The original verifier tries candidate references to establish context assembly, while generation chooses the newest. This does not prove identical historical reference selection. Retain that limitation and record the actual selected reference for every probe group.
- `build_probe_set.py` joins ready qids through the actual training parquet indices and scans only checkpoint-or-earlier rollouts. It can emit fewer requested groups, and zero selected usable attempts reaches an empty-list indexing failure. The wrapper must require the intended count before GPU work.
- GPU generation defaults to65536 context and non-eager execution. Local bounded configuration must use the proposed response/chunk budgets and explicit eager execution. Judge stage output completeness and error flags require checking before analysis.

## Next acceptance

After E8/E9 and the R proposal: implement only the needed local adaptations; test failed/passed/future references, pre-wave/post-checkpoint state boundaries, and empty/mismatched verification; then execute the complete bounded probe pipeline on a clearly labeled engineering checkpoint. A real R5 result still requires an approved R2 checkpoint with normal readiness. See [AC2 smoke](ac2-smoke.md) for the permissive fixture's limitations.
