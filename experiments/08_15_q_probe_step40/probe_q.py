#!/usr/bin/env python3
"""STAGE B: ask Q, at prefix+10k, what each generated rollout will earn.

This is the measurement the live run cannot make. In the `short` lane a rollout that reaches
prefix+10k has its reward REPLACED by Q's answer and is then cut, so its true reward is never
observed and the substitution error is unobservable by construction. Stage A removed the cut;
this stage supplies the other half of the pair.

EVERYTHING THAT DEFINES Q IS IMPORTED, NOT REIMPLEMENTED. The instruction texts, the answer
prefill, the </think> marker, the grid parser and the proof extractor all come from
verl.trainer.ppo.sp_q_readiness itself. A local copy of any of them would be a second source
of truth that can silently drift from the trained format, and the format IS the trained
behaviour -- there is no prompt hash in q_state.json that would catch a swap.

THE REFERENCE PROOF is reconstructed at the requested state boundary: active tier1 replay
proofs (gated on meta.judge_pass), then the actual add-once tier2 bank from its seed and
q_state_deltas. The original historical run had an empty bank; current AC2 does not.
Verification tries all candidate live proofs because rollout dumps omit source entry IDs.
Generation selects the newest eligible tier1 proof, then the bank, and records that choice.
This establishes context assembly, not exact historical source-entry selection parity.

--verify IS NOT OPTIONAL BEFORE A REAL RUN. q_wave rows store `ctx_token_ids`: the exact
token ids the trainer sent for every historical Q call. Verify mode rebuilds the context for
those same rows and compares token-for-token. If the reconstruction is off by even the
turn-close bytes, the 4,096 Q values this stage produces are measuring some other prompt and
the whole probe is void. Run --verify first; it needs no GPU.

    python probe_q.py --verify --run-dir <run_data> --model <hf>      # no GPU
    python probe_q.py --gen-dir gen --shard K --num-shards N --out q.shardK.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def build_ref_map(run_dir, tokenizer, require_pass=True, upto_step=None):
    """qid -> [reference proof text, ...], rebuilt from the replay buffer's entries (tier 1).

    TIME MATTERS, and getting it wrong is silent. The buffer is a fixed-capacity FIFO: over
    this run 7,616 entries were evicted and only 256 are live at the end. A Q call at step N
    saw the buffer AS OF step N, so replaying every delta to the final state throws away most
    of the entries that actually supplied references (a final-state replay verifies at only
    38.7%). `upto_step` stops the replay at the dataset step being reconstructed.

    Returns a LIST per qid, newest first, because a qid can have several live entries and the
    trainer keys tier 1 on the specific source entry_id, which rollout dumps do not record.
    Verification tries every candidate; generation takes the newest.

    `sp_q_ref_require_pass` defaults ON in these runs, and for good reason: under ungated
    replay admission the buffer accepts every non-empty response, so an unchecked lookup
    hands Q a FAILED attempt labelled as a "reference correct proof". Absent judge_pass is
    treated as passing (pre-field entries and the judged_correct path are correct by
    construction); only an explicit 0 is rejected.
    """
    from verl.trainer.ppo.sp_q_readiness import _extract_proof_text

    entries = {}
    path = os.path.join(run_dir, "replay_buffer_deltas.jsonl")
    n_del = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if upto_step is not None and int(d.get("dataset_step", 0)) > upto_step:
                break
            for eid in d.get("replaced_entry_ids") or []:
                entries.pop(eid, None)
                n_del += 1
            for e in d.get("added_entries") or []:
                entries[e["entry_id"]] = e

    refs, n_rej = {}, 0
    ordered = sorted(entries.values(),
                     key=lambda e: (e.get("meta") or {}).get("buffer_seq", 0), reverse=True)
    for e in ordered:
        meta = e.get("meta") or {}
        if require_pass and meta.get("judge_pass", 1) == 0:
            n_rej += 1
            continue
        qid = e.get("qid")
        if not qid:
            continue
        text = tokenizer.decode(e.get("response_token_ids") or [], skip_special_tokens=True)
        proof = _extract_proof_text(text)
        if proof:
            refs.setdefault(qid, []).append(proof)
    print("[ref] upto_step=%s  live entries %d (evicted %d), judge-failed rejected %d, "
          "qids with >=1 usable proof %d"
          % (upto_step, len(entries), n_del, n_rej, len(refs)), flush=True)
    return refs


def build_ref_map_rollouts(run_dir, data_dir, tokenizer, prompt_key="prompt", steps=None):
    """qid -> proof, harvested from the run's own JUDGE-PASSED rollouts.

    WHY THIS EXISTS ALONGSIDE THE BUFFER MAP. The live replay buffer holds only 256 entries at
    any moment (7,616 were evicted over this run), so a qid-keyed lookup into it covers barely
    40% of the probe's 256 problems -- every miss silently downgrades that group to the no-ref
    prompt and would drag the probe's ref share far below the 51-68% the run actually runs at.
    The rollout dumps are the same trajectories before eviction, so harvesting judged-correct
    proofs from them restores coverage without changing WHAT a reference is: a correct proof
    for this problem, produced by the same run.

    Pass criterion mirrors the reward path's binary correctness label (rubric points >= 6 of 7),
    with `acc` accepted as the already-computed form of the same thing.
    """
    from verl.trainer.ppo.sp_q_readiness import _extract_proof_text
    import pandas as pd
    from verl.trainer.ppo.difficulty import qid_from_messages

    df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    row_qids = [qid_from_messages(df[prompt_key].iloc[i]) for i in range(len(df))]

    refs = {}
    files = sorted(glob.glob(os.path.join(run_dir, "rollouts", "*.jsonl")),
                   key=lambda p: -int(os.path.basename(p).split(".")[0]))
    if steps is not None:
        files = [f for f in files if int(os.path.basename(f).split(".")[0]) in steps]
    n_seen = 0
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_seen += 1
                idx = r.get("extra_index")
                if idx is None:
                    continue
                qid = row_qids[int(idx)]
                if qid in refs:
                    continue
                pts = r.get("rubric_points")
                passed = r.get("prover_judge_score")
                ok = (float(passed) >= 1 if passed is not None else
                      float(pts) >= 6 if pts is not None else float(r.get("acc") or 0) >= 6 / 7)
                if not ok:
                    continue
                text = tokenizer.decode(r.get("response_token_ids") or [],
                                        skip_special_tokens=True)
                proof = _extract_proof_text(text)
                if proof:
                    refs[qid] = proof
    print("[ref-rollouts] scanned %d rows over %d files -> %d qids with a passed proof"
          % (n_seen, len(files), len(refs)), flush=True)
    return refs


def build_bank_map(run_dir, bank_dir, upto_step):
    """Replay the actual add-once tier2 bank, never future rollout substitutes."""
    from pathlib import Path
    from verl.trainer.ppo.sp_q_readiness import verify_manifest_shards
    base = Path(bank_dir)
    manifest = json.loads((base / "reference_bank_manifest.json").read_text())
    shards = sorted(str(p) for p in base.glob("reference_bank/shard_*.jsonl"))
    verify_manifest_shards(str(base), manifest["shards"], shards, "probe reference bank")
    bank = {}
    for path in shards:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            assert row["qid"] not in bank and row["proof"]
            bank[row["qid"]] = row["proof"]
    with (Path(run_dir) / "q_state_deltas.jsonl").open() as f:
        for line in f:
            delta = json.loads(line)
            if int(delta["dataset_step"]) > upto_step:
                break
            for row in delta.get("bank_added", []):
                assert row["proof"]
                bank.setdefault(row["qid"], row["proof"])
    return bank


def make_ctx_builder(tokenizer, variant="reward_horizon"):
    """Returns build(prompt_ids, attempt_ids, ref_proof) -> ctx ids, mirroring
    QHarness.build_q_context_ids exactly (prompt + attempt + turn close + user turn +
    forced prefill)."""
    from verl.trainer.ppo.sp_q_readiness import (
        Q_PROMPT_VARIANTS, Q_ANSWER_PREFILL, THINK_CLOSE)

    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    with_ref, no_ref = Q_PROMPT_VARIANTS[variant]
    ids_think_close = enc(THINK_CLOSE)
    ids_turn_close = enc("<|im_end|>\n")
    ids_user_open = enc("<|im_start|>user\n")
    ids_assistant_open = enc("<|im_start|>assistant\n")
    ids_prefill = enc(Q_ANSWER_PREFILL)
    cache = {}

    def instr_ids(ref):
        key = ref if ref is None else hash(ref)
        if key not in cache:
            text = with_ref.format(reference_proof=ref) if ref is not None else no_ref
            if len(cache) > 4096:
                cache.clear()
            cache[key] = enc(text)
        return cache[key]

    def build(prompt_ids, attempt_ids, ref_proof):
        closed = (ids_think_close[0] in attempt_ids if len(ids_think_close) == 1
                  else _find_sub(attempt_ids, ids_think_close) >= 0)
        parts = [list(prompt_ids), list(attempt_ids)]
        if not closed:
            parts.append(ids_think_close)
        parts += [ids_turn_close, ids_user_open, instr_ids(ref_proof),
                  ids_turn_close, ids_assistant_open, ids_prefill]
        out = []
        for p in parts:
            out.extend(p)
        return out

    return build


def _find_sub(hay, needle):
    n = len(needle)
    for i in range(len(hay) - n, -1, -1):
        if hay[i:i + n] == needle:
            return i
    return -1


def do_verify(args):
    """Rebuild the context for historical q_wave rows and diff against what was really sent."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Global step S uses dataset step S-1; its Q wave precedes that step's deltas.
    upto = args.verify_upto if args.verify_upto is not None else args.verify_step - 2
    refs = build_ref_map(args.run_dir, tok, require_pass=not args.no_require_pass,
                         upto_step=upto)
    bank = build_bank_map(args.run_dir, args.bank_dir, upto)
    build = make_ctx_builder(tok, args.variant)

    # the run's own rollout rows supply prompt+attempt ids for the step being checked
    roll = {}
    rpath = os.path.join(args.run_dir, "rollouts", "%d.jsonl" % args.verify_step)
    with open(rpath, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            roll.setdefault(str(r.get("uid")), []).append(r)

    # q_wave/<N> is named by sp_dataset_step (0-based); rollouts/<N> by global_steps (1-based)
    wpath = os.path.join(args.run_dir, "q_wave", "%d.jsonl" % (args.verify_step - 1))
    n = ok = miss = var_ok = 0
    n_ref_run = 0
    by = {"ref": [0, 0], "noref": [0, 0]}   # variant -> [matched, total]
    first_bad = None
    with open(wpath, encoding="utf-8") as fh:
        for line in fh:
            w = json.loads(line)
            if w.get("kind") != "consumed" or w.get("overflow"):
                continue
            n += 1
            n_ref_run += int(w.get("variant") == "ref")
            rows = roll.get(str(w.get("uid")))
            if not rows:
                miss += 1
                continue
            ctx = w.get("ctx_token_ids") or []
            hit = False
            # the trainer keys tier 1 on a specific source entry_id that rollout dumps do not
            # record, so try every live proof for this qid; a hit on any of them confirms the
            # ASSEMBLY is right, which is what this check exists to establish
            cands = ([None] if w.get("variant") != "ref"
                     else list(refs.get(w.get("qid")) or [])[:args.ref_try])
            if w.get("variant") == "ref" and w.get("qid") in bank:
                cands.append(bank[w["qid"]])
            for r in rows:
                prompt_ids = r.get("prompt_token_ids") or []
                resp = r.get("response_token_ids") or []
                for cut in (len(resp), args.budget_g + int(r.get("sp_prefix_len") or 0)):
                    for ref in cands:
                        if build(prompt_ids, resp[:cut], ref) == ctx:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            ok += int(hit)
            _v = "ref" if w.get("variant") == "ref" else "noref"
            by[_v][0] += int(hit); by[_v][1] += 1
            if not hit and first_bad is None:
                first_bad = w
            if n >= args.verify_n:
                break

    print("\n=== VERIFY (step %d, consumed rows) ===" % args.verify_step)
    print("rows checked      : %d" % n)
    print("exact ctx match   : %d (%.1f%%)" % (ok, 100.0 * ok / max(1, n)))
    print("no rollout row    : %d" % miss)
    print("run ref share     : %.1f%%   (reconstruction has proofs for %d qids)"
          % (100.0 * n_ref_run / max(1, n), len(refs)))
    for v in ("noref", "ref"):
        m, t = by[v]
        print("  %-6s match    : %d/%d (%.1f%%)" % (v, m, t, 100.0 * m / max(1, t)))
    if first_bad is not None:
        ctx = first_bad.get("ctx_token_ids") or []
        print("\nfirst mismatch: uid=%s qid=%s variant=%s len(ctx)=%d"
              % (first_bad.get("uid"), str(first_bad.get("qid"))[:12],
                 first_bad.get("variant"), len(ctx)))
        print("  ctx tail ids : %s" % ctx[-24:])
        print("  ctx tail text: %r" % tok.decode(ctx[-64:], skip_special_tokens=False))
    return 0 if ok == n and n > 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.expandvars("${AC2_CLUSTER_A_ROOT}/self-play/"
                                         "experiments/08_13_tiedq_seed192/run_data"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--bank-dir", required=True, help="actual seed reference-bank directory")
    ap.add_argument("--variant", default="reward_horizon")
    ap.add_argument("--budget-g", type=int, default=10000)
    ap.add_argument("--no-require-pass", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-step", type=int, default=40)
    ap.add_argument("--verify-n", type=int, default=200)
    ap.add_argument("--verify-upto", type=int, default=None,
                    help="buffer state offset to reconstruct at; default verify_step-2")
    ap.add_argument("--ref-try", type=int, default=8,
                    help="candidate live proofs per qid to try in verify mode")
    # generation-mode args
    ap.add_argument("--gen-dir")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)   # rollout temperature
    ap.add_argument("--gen-reserve", type=int, default=4)       # QHarness.gen_reserve
    ap.add_argument("--data-dir", default=os.path.join(os.environ.get("HOME", ""),
                                                       "data/fineproofs"))
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--no-prefix-site", action="store_true",
                    help="skip the per-group Q at the shared prefix")
    ap.add_argument("--upto-step", type=int, default=39,
                    help="buffer state to reconstruct refs at; 39 = dataset step of ckpt 40")
    args = ap.parse_args()

    if args.verify:
        return do_verify(args)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from verl.trainer.ppo.sp_q_readiness import parse_grid_value

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    refs = build_ref_map(args.run_dir, tok, require_pass=not args.no_require_pass,
                         upto_step=args.upto_step)
    bank = build_bank_map(args.run_dir, args.bank_dir, args.upto_step)
    build = make_ctx_builder(tok, args.variant)

    groups = []
    for p in sorted(glob.glob(os.path.join(args.gen_dir, "gen.shard*.jsonl"))):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                groups.append(json.loads(line))
    mine = [g for i, g in enumerate(groups) if i % args.num_shards == args.shard]
    print("[q %d] %d groups of %d total" % (args.shard, len(mine), len(groups)), flush=True)

    prompts, keys = [], []
    n_skip_len = 0
    # ---- SITE 1: Q at the SHARED PREFIX, one call per group (completion_index = -1) -------
    # This is the run's `probe` site: build_q_context_ids with attempt_ids = the prefix alone,
    # no continuation. It exists here for two reasons the consumed site cannot serve:
    #   * it is the quantity an audit-lane analysis measures (Q_prefix vs mean_i r_i), so the
    #     probe is comparable with such an analysis on the SAME statistic;
    #   * paired with the consumed value on the same group it isolates DEGRADATION -- whether
    #     Q reads a fresh prefix well but misjudges its own 10k continuation, a hypothesis
    #     left open by the step-40 probe.
    if not args.no_prefix_site:
        for g in mine:
            _rl = refs.get(g["qid"]) or []
            ref = _rl[0] if _rl else bank.get(g["qid"])
            ctx = build(g["prompt_token_ids"], list(g["prefix_token_ids"]), ref)
            if len(ctx) + args.gen_reserve > args.max_model_len:
                n_skip_len += 1
                continue
            prompts.append(TokensPrompt(prompt_token_ids=ctx))
            keys.append((g["qid"], -1, ref, "tier1-newest" if _rl else "bank" if ref else "none", len(ctx)))

    # ---- SITE 2: Q at prefix+g, one call per rollout that reached the cut ----------------
    for g in mine:
        _rl = refs.get(g["qid"]) or []
        ref = _rl[0] if _rl else bank.get(g["qid"])
        for j, c in enumerate(g["completions"]):
            if not c["exceeds_g"]:
                continue                      # never reached the cut: no Q value is defined
            attempt = list(g["prefix_token_ids"]) + list(c["cont_token_ids"])[:args.budget_g]
            ctx = build(g["prompt_token_ids"], attempt, ref)
            if len(ctx) + args.gen_reserve > args.max_model_len:
                n_skip_len += 1               # QHarness.fits() would drop these too
                continue
            prompts.append(TokensPrompt(prompt_token_ids=ctx))
            keys.append((g["qid"], j, ref, "tier1-newest" if _rl else "bank" if ref else "none", len(ctx)))
    n_pfx = sum(1 for k in keys if k[1] == -1)
    print("[q %d] %d Q calls (%d prefix-site, %d consumed-site; %d dropped for length)"
          % (args.shard, len(prompts), n_pfx, len(prompts) - n_pfx, n_skip_len))
    print("[q %d] (legacy line) %d Q calls (%d dropped for context length)"
          % (args.shard, len(prompts), n_skip_len), flush=True)
    if not prompts:
        open(args.out, "w").close()
        return 0

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, dtype="bfloat16", enable_prefix_caching=True,
              enforce_eager=args.enforce_eager, trust_remote_code=True, seed=args.shard)
    sp = SamplingParams(temperature=args.temperature, top_p=1.0, top_k=-1,
                        max_tokens=args.gen_reserve, n=1)
    outs = llm.generate(prompts, sp)

    n_valid = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for (qid, j, ref, ref_source, ctx_len), o in zip(keys, outs):
            text = o.outputs[0].text
            v = parse_grid_value(text)
            n_valid += int(v is not None)
            fh.write(json.dumps({"qid": qid, "completion_index": j, "q": v,
                                 "gen_text": text, "used_ref": ref is not None,
                                 "reference_proof": ref, "reference_source": ref_source,
                                 "ctx_len": ctx_len}) + "\n")
    print("[q %d] DONE %d calls, %d parsed to grid (%.1f%% invalid)"
          % (args.shard, len(keys), n_valid,
             100.0 * (len(keys) - n_valid) / max(1, len(keys))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
