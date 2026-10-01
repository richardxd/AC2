"""Pure-logic tests for Prefix GRPO (08_11_ablation1_replay_noq). No torch, no GPU, no cluster.

The first group repeats AC2's checks on the logic this configuration shares with it: the
cold-bootstrap draw, the relaunch topology check, the pinned off switches, the judge revision
pin, the TP>1 AOT preflight, the response-budget group and checkpoint retention. The second
group checks this configuration's own invariants: no critic setting survives in the launch
environment, the dataset installs no critic and routes every row "full", a checkpoint carrying
critic state is refused, the cold builder writes only the replay seed, the single-factor diff
checker exists, question-uniform sampling is opt-in and resume-safe, and the replay draw policy
matches AC2's.

Run:  python experiments/08_11_ablation1_replay_noq/test_ablation1_logic.py
   or: pytest experiments/08_11_ablation1_replay_noq/test_ablation1_logic.py

Sources are loaded by AST surgery rather than by importing `verl` (which needs torch/ray), so
these run anywhere -- including a laptop -- and fail loudly if the shapes they assume change.
"""
from __future__ import annotations

import ast
import os
import re
import sys

SP_Q = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "verl", "verl", "trainer", "ppo", "sp_q_readiness.py",
)


def _load_names(path: str, class_names=(), assign_prefixes=()):
    """exec only the named classes / prefixed module assignments from `path`."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                name = getattr(t, "id", "")
                if name and any(name.startswith(p) for p in assign_prefixes):
                    keep.append(node)
                    break
    ns: dict = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[])),
                 "<sp_q>", "exec"), ns)
    return ns


# ----------------------------------------------------------- 4. the cold-bootstrap draw
def test_cold_bootstrap_fill_is_disjoint_and_full():
    """The empty-buffer branch must return zero qids and exactly n_replay fill indices,
    disjoint from the statement lane -- the dataset indexes fill[rslot - len(qids)] for
    every replay slot, so a short list would IndexError mid-epoch."""
    # stdlib `random` stands in for numpy's Generator here: this asserts the branch's
    # INVARIANTS (count, distinctness, disjointness from the statement lane), which do not
    # depend on which RNG produced the draw. Keeps the test runnable without numpy.
    import random

    num_statements, n_orig, n_replay = 5227, 32, 96
    used = set(random.Random(0).sample(range(num_statements), n_orig))
    candidates = [i for i in range(num_statements) if i not in used]
    assert len(candidates) >= n_replay
    fill = random.Random(1).sample(candidates, n_replay)

    qids: list = []
    assert len(qids) == 0
    assert len(fill) == n_replay
    assert not (set(fill) & used), "fill must not reuse this step's statement problems"
    assert len(set(fill)) == n_replay, "fill must be distinct problems"
    for rslot in range(n_replay):                       # every replay slot resolves
        assert 0 <= rslot - len(qids) < len(fill)

    src = open(os.path.join(os.path.dirname(SP_Q), "sp_replay.py"), encoding="utf-8").read()
    assert "cold bootstrap needs" in src, "the short-fill guard is gone from sp_replay"
    assert 'if not self.cold_bootstrap:' in src, "the empty-buffer branch is no longer gated"


# --------------------------------------------------- 5. the relaunch topology check
def test_relaunch_does_not_call_the_separate_q_resharder():
    """The attach must NOT invoke the separate-Q reshard helper: it hard-requires q_model/, which
    this tied-Q run never writes, so every relaunch after checkpoint 1 would fail on "source
    q_model missing". Assert the wiring directly."""
    here = os.path.dirname(os.path.abspath(__file__))
    attach = open(os.path.join(here, "run_attach_cluster_b.sh"), encoding="utf-8").read()
    # Check for an INVOCATION, not a mention: the refusal message names the helper on purpose,
    # so a bare substring test would flag its own explanatory text.
    code = "\n".join(l for l in attach.splitlines() if not l.lstrip().startswith("#"))
    for pattern in (r"bash\s+\S*reshard_to_ws16\.sh", r"^\s*RESHARD_WS=", r"RESHARD_EXP="):
        assert not re.search(pattern, code, re.M), (
            f"the attach invokes the separate-Q resharder ({pattern}); it requires q_model/ "
            "and this run writes sp_q_optim/ -- every holder roll after checkpoint 1 will fail"
        )
    # and it must decide from the checkpoint's real world size
    assert "sp_q_optim/rank_*.pt" in attach, "no world-size probe in the topology check"
    assert "REFUSING: checkpoint" in attach, "a topology mismatch must refuse, not proceed"
    # if the separate-Q helper is present, it really does require q_model
    helper = os.path.join(os.path.dirname(here), "08_01_branch230_fullaudit",
                          "reshard_to_ws16.sh")
    if os.path.exists(helper):
        assert 'source q_model missing' in open(helper, encoding="utf-8").read()


# ------------------------------------------- 6. the science switches are pinned, not omitted
def test_off_switches_are_pinned_and_asserted():
    """SP_LENPEN_ENABLE is read from the process environment by the reward module, and the
    allocation does not use --export=NONE, so omitting it is not the same as setting it to 0."""
    here = os.path.dirname(os.path.abspath(__file__))
    attach = open(os.path.join(here, "run_attach_cluster_b.sh"), encoding="utf-8").read()
    runner = open(os.path.join(here, "runner.py"), encoding="utf-8").read()
    assert '"SP_LENPEN_ENABLE=0"' in attach, "the attach must PIN the penalty off, not omit it"
    assert 'assert not _truthy("SP_LENPEN_ENABLE")' in runner, "and the runner must assert it"
    assert 'env_vars["SP_LENPEN_ENABLE"] = "0"' in runner, (
        'the forward loop only carries truthy values, so "0" must be set explicitly or the Ray '
        "actors read whatever the node inherited"
    )
    # the reward module really does read it from the environment (so the risk was real)
    reward = os.path.join(os.path.dirname(os.path.dirname(here)), "src", "ac2",
                          "rewards", "ds4_finegrained_judge.py")
    if os.path.exists(reward):
        assert 'os.environ.get("SP_LENPEN_ENABLE"' in open(reward, encoding="utf-8").read()


# ------------------------------- 11. the judge model is fixed, and its revision is pinned
def test_judge_is_the_lineage_model_at_a_pinned_revision():
    """The judge IS the reward function. The run keeps one fixed judge model so rewards stay on
    one scale, and pins the REVISION so a hub-cache re-download cannot move it. Both halves
    matter: the right repo at a floating revision is not reproducible."""
    here = os.path.dirname(os.path.abspath(__file__))
    runner = open(os.path.join(here, "runner.py"), encoding="utf-8").read()
    attach = open(os.path.join(here, "run_attach_cluster_b.sh"), encoding="utf-8").read()
    lineage_rev = "60d8d70770c6776ff598c94bb586a859a38244f1"

    # the pinned judge model -- NOT a newer release
    assert '"deepseek-ai/DeepSeek-V4-Flash"' in runner, "runner no longer defaults to the reported runs' judge"
    # Strip comments before checking for the newer judge release: a comment may name it, and a
    # bare substring test would flag that explanation (the same trap as the resharder test
    # above). Only CODE may not mention it.
    def _code(text, marker):
        return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(marker))
    assert "DeepSeek-V4-Flash-0731" not in _code(runner, "#"), \
        "the 0731 judge is back in runner CODE (not just a comment)"
    assert "DeepSeek-V4-Flash-0731" not in _code(attach, "#"), \
        "the 0731 judge is back in attach CODE (not just a comment)"
    assert "deepseek-ai/DeepSeek-V4-Flash}" in attach or \
           'JUDGE_HF_REPO="${SP_JUDGE_HF_REPO:-deepseek-ai/DeepSeek-V4-Flash}"' in attach

    # and the exact revision, pinned rather than resolved from refs/main
    assert lineage_rev in runner, "the runner does not pin the reported runs' judge revision"
    assert lineage_rev in attach, "the attach does not pin the reported runs' judge revision"
    # the resolver must PREFER the pin over refs/main
    i_pin = runner.index("if _JUDGE_REVISION:")
    i_ref = runner.index('ref = repo_dir / "refs" / "main"')
    assert i_pin < i_ref, (
        "refs/main is consulted before the pinned revision; a re-download would silently change "
        "the reward function"
    )
    # ...and the pin must be recorded per run so a later launch can detect drift
    assert '"revision": _JUDGE_REVISION' in runner, \
        "the resolved revision is not written into judge_snapshot.json"
    assert "SP_ALLOW_JUDGE_DRIFT" in runner, "no escape hatch documented for a deliberate change"


# --------------------------------- 13. TP>1 rollout is gated on the flashinfer AOT preflight
def test_tp_gt_1_is_gated_on_the_aot_preflight():
    """SP_ROLLOUT_TP=4 is only startable because the flashinfer modules are promoted to AOT
    first. Wire the two together: a future edit that keeps TP=4 but drops (or softens) the
    preflight reintroduces a wedge that shows up as a 0%-util hang minutes into engine init,
    which is exactly the failure nobody diagnoses correctly under time pressure."""
    here = os.path.dirname(os.path.abspath(__file__))
    attach = open(os.path.join(here, "run_attach_cluster_b.sh"), encoding="utf-8").read()
    runner = open(os.path.join(here, "runner.py"), encoding="utf-8").read()
    code = "\n".join(l for l in attach.splitlines() if not l.lstrip().startswith("#"))

    m = re.search(r'"SP_ROLLOUT_TP=(\d+)"', code)
    assert m, "the attach must PIN SP_ROLLOUT_TP, not inherit the runner default"
    tp = int(m.group(1))
    assert 8 % tp == 0, f"TP={tp} does not divide the 8 GPUs per node"

    # The runner default is deliberately NOT tp: it is 1, the only value that is safe without the
    # preflight. The attach is what pins 4. So a launch that bypasses the attach runs SLOWER than
    # intended rather than wedging on a cold flashinfer cache -- the failure mode you can read off
    # a throughput number instead of a 0%-util hang. Cost of that choice: the effective TP lives in
    # exactly one place, so read run_attach_cluster_b.sh, never runner.py, to know what a run used.
    dm = re.search(r'ROLLOUT_TP\s*=\s*_env\("SP_ROLLOUT_TP",\s*(\d+)', runner)
    assert dm, "runner.py no longer reads SP_ROLLOUT_TP"
    assert int(dm.group(1)) == 1, (
        f"the runner default is {dm.group(1)}, not 1; a launch that bypasses the attach would then "
        "take the TP>1 path with no guarantee the AOT preflight ever ran"
    )

    if tp > 1:
        assert "flashinfer_aot_warm.py" in code, (
            "TP>1 without the AOT preflight: every rank JIT-rebuilds flashinfer's trtllm_comm "
            "under one FileLock, so engine init burns TP x build-time at 0% GPU util and wedges"
        )
        assert "[aot] READY" in code, "the preflight must check for the READY marker"
        # and it must be FATAL -- a warning would be silently ignored on every relaunch
        gate = code[code.index("flashinfer_aot_warm.py"):]
        gate = gate[:gate.index("SEED=")] if "SEED=" in gate else gate
        assert re.search(r"^\s*exit 1", gate, re.M), (
            "the AOT preflight only warns; it must abort the launch"
        )

    # the warm script really does emit that marker and cover the module vLLM's fusion pass needs
    warm = os.path.join(os.path.dirname(here), "08_11_ablation1_replay_noq",
                        "flashinfer_aot_warm.py")
    if os.path.exists(warm):
        w = open(warm, encoding="utf-8").read()
        assert "[aot] READY" in w, "the preflight greps for a marker the warm script never prints"
        assert "trtllm_comm" in w, "the warm script skips the allreduce module, which is the one "\
                                   "the AllReduceRMSFusionPass builds at TP>1"


# --------------------------------------- 14. the context group is internally consistent at 75k
def test_context_group_moves_together():
    """The five context budgets are coupled by arithmetic the runner asserts at import
    (C_Q == 2048 + response + 1248) and by a memory invariant it does NOT check. Editing the
    response budget alone is the easy mistake -- it crashes at import if you are lucky and
    silently mis-packs if you are not -- so pin the whole group's arithmetic here."""
    here = os.path.dirname(os.path.abspath(__file__))
    attach = open(os.path.join(here, "run_attach_cluster_b.sh"), encoding="utf-8").read()
    code = "\n".join(l for l in attach.splitlines() if not l.lstrip().startswith("#"))
    v = {}
    # SP_Q_CTX_LIMIT / SP_Q_MAX_TOKEN_LEN are Q-only and correctly ABSENT in this arm -- the
    # main run pins C_Q == rollout max_model_len so Q sees the policy's horizon; with no Q there
    # is one limit and nothing to reconcile.
    for k in ("SP_MAX_RESPONSE_LEN", "SP_PPO_MAX_TOKEN_LEN", "SP_ACTOR_GPU_MEM_UTIL"):
        m = re.search(rf'"{k}=([0-9.]+)"', code)
        assert m, f"the attach must PIN {k} -- the context group moves together or not at all"
        v[k] = float(m.group(1))

    resp, prompt, head = v["SP_MAX_RESPONSE_LEN"], 2048, 1248
    assert resp == 50000, (
        f"response budget is {resp:.0f}, expected 50000 -- this arm STARTS at 50k and is\n"
        "moved to 75k by a staged relaunch. If you just did the switch, update this test\n"
        "and record the step in run_data/.context_switch_step."
    )
    # The equality runner.py asserts at import (C_Q == 2048+resp+1248) has no analogue here:
    # there is no C_Q. What remains is the memory invariant, which is arm-independent.
    # (c) the memory invariant NOTHING checks at runtime: the PPO per-GPU budget must admit
    # exactly ONE maximum-length row. Two would OOM the actor update at this model and batch;
    # fewer than one would mean a full row cannot be formed at all.
    full_row = prompt + resp
    ratio = v["SP_PPO_MAX_TOKEN_LEN"] / full_row
    assert ratio < 2, (
        f"SP_PPO_MAX_TOKEN_LEN is {ratio:.2f}x a full {full_row:.0f}-token row -- two max-length "
        "rows can pack into one GPU micro-batch and the actor update OOMs"
    )
    assert ratio > 0.9, (
        f"SP_PPO_MAX_TOKEN_LEN is only {ratio:.2f}x a full row; well below 1 starts costing "
        "throughput for no memory benefit"
    )

    # (d) the arena must hold enough full-length sequences to be worth the rollout batch, and
    # SP_ACTOR_MAX_NUM_SEQS must not be the binding constraint (else it, not KV, caps concurrency)
    tp = int(re.search(r'"SP_ROLLOUT_TP=(\d+)"', code).group(1))
    KV_PER_TOKEN, WEIGHTS, OVERHEAD, TOTAL = 2 * 36 * 8 * 128 * 2, int(4.02e9) * 2, 2 << 30, 85520809984
    per_gpu = v["SP_ACTOR_GPU_MEM_UTIL"] * TOTAL - OVERHEAD
    _model_len = prompt + resp + head          # what C_Q would have been
    kv_seqs = int((tp * per_gpu - WEIGHTS) // (KV_PER_TOKEN * _model_len))
    assert kv_seqs >= 8, (
        f"the arena holds only {kv_seqs} full-length sequences per TP={tp} engine; the rollout "
        "will serialize and generation time will not match the benchmark"
    )
    cm = re.search(r'"SP_ACTOR_MAX_NUM_SEQS=(\d+)"', code) or re.search(
        r'_env\("SP_ACTOR_MAX_NUM_SEQS",[^,]*?(\d+)\), int\)|"rollout_max_num_seqs", (\d+)\)',
        open(os.path.join(here, "runner.py"), encoding="utf-8").read())
    cap = int(next(g for g in (cm.groups() if cm else ()) if g) if cm else 96)
    assert cap > kv_seqs, (
        f"SP_ACTOR_MAX_NUM_SEQS={cap} is at or below the KV-bound {kv_seqs}; it has silently "
        "become the concurrency limit, so the documented 'KV binds first' reasoning is stale"
    )


# ------------------- 15. checkpoint retention keeps every 10th step's WEIGHTS, permanently
def test_retention_keeps_every_tenth_step():
    """The rolling window alone makes mid-run steps unrecoverable: keep_last=5 with save_freq=1
    means step 137's weights are gone by step 143, so nothing later can re-evaluate or branch
    from it. Exercise the REAL fork function (not a reimplementation) against a fake checkpoint
    tree, because the failure mode is silent and destructive -- by the time anyone notices, the
    weights are already deleted."""
    import contextlib, io, json as _json, shutil, tempfile, types

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = os.path.join(root, "src", "verl", "verl", "trainer", "ppo", "ray_trainer.py")
    assert os.path.exists(src), (
        f"cannot find the verl fork at {src}; this test silently passed for one revision "
        "because the path was one directory short -- fail loudly instead"
    )

    # Import ONLY the method's source, not the module (ray_trainer imports torch/ray).
    text = open(src, encoding="utf-8").read()
    start = text.index("    def _retain_recent_and_best(self, train_reward):")
    end = text.index("\n    def ", start + 10)
    ns = {"os": os, "json": _json}
    exec("import os, json\nclass T:\n" + text[start:end], ns)
    fn = ns["T"]._retain_recent_and_best

    def build(tmp, steps):
        for st in steps:
            for sub in ("actor", "sp_q_optim"):
                os.makedirs(os.path.join(tmp, f"global_step_{st}", sub), exist_ok=True)
                open(os.path.join(tmp, f"global_step_{st}", sub, "w.pt"), "w").close()
            open(os.path.join(tmp, f"global_step_{st}", "data.pt"), "w").close()

    def surviving(steps, cur, keep_last=5, every=None, reward=0.1):
        tmp = tempfile.mkdtemp()
        try:
            build(tmp, steps)
            cfg = types.SimpleNamespace(trainer=types.SimpleNamespace(
                default_local_dir=tmp, get=lambda k, d=None: {"max_actor_ckpt_to_keep": keep_last}.get(k, d)))
            self_ = types.SimpleNamespace(config=cfg, global_steps=cur)
            old = os.environ.get("SP_CKPT_KEEP_EVERY")
            if every is None:
                os.environ.pop("SP_CKPT_KEEP_EVERY", None)      # exercise the DEFAULT
            else:
                os.environ["SP_CKPT_KEEP_EVERY"] = str(every)
            try:
                # the real function logs one line per pruned dir; keep the suite readable
                with contextlib.redirect_stdout(io.StringIO()):
                    fn(self_, reward)
            finally:
                os.environ.pop("SP_CKPT_KEEP_EVERY", None)
                if old is not None:
                    os.environ["SP_CKPT_KEEP_EVERY"] = old
            w = {st for st in steps
                 if os.path.isdir(os.path.join(tmp, f"global_step_{st}", "actor"))}
            q = {st for st in steps
                 if os.path.isdir(os.path.join(tmp, f"global_step_{st}", "sp_q_optim"))}
            d = {st for st in steps
                 if os.path.exists(os.path.join(tmp, f"global_step_{st}", "data.pt"))}
            assert q == w, (
                f"sp_q_optim did not rotate with the weights it belongs to: actor={sorted(w)} "
                f"sp_q_optim={sorted(q)}; a tied-Q optimizer is as heavy as the model"
            )
            return w, d
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    steps = list(range(1, 44))
    # DEFAULT (no env var set) must already retain the stride -- "every active run", no opt-in
    w, d = surviving(steps, 43)
    assert {10, 20, 30, 40} <= w, f"multiples of 10 were pruned by default: kept {sorted(w)}"
    assert {39, 40, 41, 42, 43} <= w, f"the last-5 window was broken: kept {sorted(w)}"
    assert 43 in w and 7 not in w, f"a non-multiple outside the window survived: {sorted(w)}"
    # the small per-step state is NEVER pruned -- the dashboard reads every step
    assert d == set(steps), "data.pt was pruned; the dashboard would lose history"
    # (sp_q_optim/weights co-rotation is asserted inside surviving() on every call)

    # explicit stride
    w5, _ = surviving(steps, 43, every=5)
    assert {5, 10, 15, 20, 25, 30, 35, 40} <= w5, f"stride 5 not honoured: {sorted(w5)}"
    # OFF switch restores last-N + best
    w0, _ = surviving(steps, 43, every=0)
    assert 10 not in w0 and 20 not in w0, f"SP_CKPT_KEEP_EVERY=0 still retained a stride: {sorted(w0)}"
    assert {39, 40, 41, 42, 43} <= w0
    # a malformed value must fall back to the DEFAULT, not silently disable retention
    wbad, _ = surviving(steps, 43, every="ten")
    assert {10, 20, 30, 40} <= wbad, f"a bad SP_CKPT_KEEP_EVERY disabled the stride: {sorted(wbad)}"


# ---------------------------------- 16. the AOT warm must share the engines' flashinfer dir
def test_aot_preflight_shares_the_engine_workspace():
    """flashinfer derives FLASHINFER_AOT_DIR from FLASHINFER_WORKSPACE_BASE. If the warm step
    runs without the value the ENGINES get, it promotes into a different directory, prints
    '[aot] READY' truthfully about that one, and every rank still JIT-builds -- a gate that
    reports success while protecting nothing. It must be EXPORTED (the ENVS array only reaches
    the driver) and the two must read the same variable so they cannot drift."""
    here = os.path.dirname(os.path.abspath(__file__))
    attach = open(os.path.join(here, "run_attach_cluster_b.sh"), encoding="utf-8").read()
    code = "\n".join(l for l in attach.splitlines() if not l.lstrip().startswith("#"))
    if "flashinfer_aot_warm.py" not in code:
        return                                    # TP=1: no preflight to check

    m = re.search(r"^\s*export FLASHINFER_WORKSPACE_BASE=", code, re.M)
    assert m, ("FLASHINFER_WORKSPACE_BASE is not exported before the warm step, so it warms "
               "flashinfer's HOME-based default while the engines read the pinned cache dir")
    assert m.start() < code.index("flashinfer_aot_warm.py"), (
        "the export comes AFTER the warm step; it must precede it"
    )
    assert '"FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE"' in code, (
        "ENVS pins a second literal path instead of reusing the exported variable -- the warm "
        "step and the engines can silently diverge"
    )


# ============================== the ablation's own invariants ==============================

def _attach_code():
    here = os.path.dirname(os.path.abspath(__file__))
    txt = open(os.path.join(here, "run_attach_cluster_b.sh"), encoding="utf-8").read()
    return "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))


def test_q_is_absent_and_pinned_off():
    """p=1 is reached by REMOVAL. Any surviving SP_Q_* export would either configure a
    mechanism that is not installed (harmless but a lie in the log) or, for SP_Q_ENABLE,
    install Q hooks against a dataset that routes everything 'full' -- neither arm."""
    here = os.path.dirname(os.path.abspath(__file__))
    code = _attach_code()
    qs = set(re.findall(r'"(SP_Q_[A-Z_]+)=', code))
    assert qs == {"SP_Q_ENABLE"}, (
        f"the attach still exports {sorted(qs - {'SP_Q_ENABLE'})}; this arm has no Q function "
        "for them to configure"
    )
    assert '"SP_Q_ENABLE=0"' in code, (
        "SP_Q_ENABLE must be PINNED to 0, not omitted: the runner forwards it into runtime_env "
        "and the dataset asserts on it, so an inherited 1 from a main-run shell would install "
        "the Q hooks"
    )
    runner = open(os.path.join(here, "runner.py"), encoding="utf-8").read()
    assert 'assert not _truthy("SP_Q_ENABLE")' in runner, "the runner must refuse SP_Q_ENABLE"
    assert 'env_vars["SP_Q_ENABLE"] = "0"' in runner, (
        'the forward loop only carries truthy values, so "0" must be set explicitly'
    )
    # the four behaviour-changing knobs are refused, not merely unset
    for k in ("SP_Q_SEPARATE", "SP_Q_INTERLEAVE", "SP_Q_LR_LADDER", "SP_Q_AUDIT_DEN"):
        assert k in runner, f"{k} is not refused by the runner"


def test_dataset_installs_no_q_and_routes_every_row_full():
    """The dataset is q_dataset.py minus Q. It must install no Q harness, route every row
    'full' with cap 0 (that IS p=1), and KEEP the two schema keys the trainer reads
    unconditionally -- dropping them KeyErrors rather than degrades."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "replay_dataset.py"), encoding="utf-8").read()
    tree = ast.parse(src)

    # Check the IMPORTS and CALLS, not the prose -- the docstring necessarily names
    # sp_q_readiness to explain what was removed, and a substring test would flag exactly that.
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            imported |= {a.name for a in n.names}
        elif isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
    assert "sp_q_readiness" not in imported, (
        f"the dataset still imports the Q harness: {sorted(imported)}"
    )
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for bad in ("route_plan_for_step", "budget_cap_for"):
        assert bad not in called, f"the dataset still calls {bad}() -- it could truncate"
    # p=1: the cap is a literal 0 and the route a literal "full", not a variable
    assert re.search(r'row\["sp_q_max_new_tokens"\]\s*=\s*0\b', src), (
        "sp_q_max_new_tokens must be a literal 0 -- any expression here could truncate"
    )
    assert re.search(r'"sp_q_route":\s*"full"', src), (
        'sp_q_route must be the literal "full" for every row'
    )
    # the class exists under the name the runner selects
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert "SPReplayNoQDataset" in names, f"class not found; got {names}"
    runner = open(os.path.join(here, "runner.py"), encoding="utf-8").read()
    assert "custom_cls.name=SPReplayNoQDataset" in runner
    # Check the CONFIG LINE, not the file: runner.py's docstring legitimately names
    # q_dataset.py to say what this was derived from, and a whole-file substring test would flag
    # that prose (the same trap as the dataset's own Q-import check above).
    _cc = [l for l in runner.splitlines() if "custom_cls.path" in l and not l.lstrip().startswith("#")]
    assert len(_cc) == 1, f"expected exactly one custom_cls.path line, got {_cc}"
    assert "replay_dataset.py" in _cc[0] and "q_dataset" not in _cc[0], (
        f"the dataset the runner actually loads is not replay_dataset.py: {_cc[0].strip()}"
    )
    # and it refuses the half-wired pairing
    assert "_q_env_on()" in src, "the dataset must assert SP_Q_ENABLE is off"


def test_resume_refuses_a_checkpoint_that_carries_q_state():
    """A checkpoint with q_state.json / sp_q_optim/ came from a run WITH Q. Resuming it here
    would continue a main-run trajectory while every log line claimed the ablation -- invisible
    until someone tries to explain the curve later."""
    code = _attach_code()
    assert "q_state.json" in code and "sp_q_optim" in code, (
        "the attach no longer checks for Q state in the resume checkpoint"
    )
    i = code.index("q_state.json")
    window = code[max(0, i - 600):i + 600]
    assert "REFUSING" in window and "exit 1" in window, (
        "finding Q state in a checkpoint must ABORT, not warn"
    )
    # the world-size probe must read actor shards -- this arm writes no sp_q_optim to count
    assert "actor/*rank_*.pt" in code, (
        "the topology probe still counts sp_q_optim/rank_*.pt, which this arm never writes, so "
        "it would read world size 0 and skip the check entirely"
    )


def test_cold_build_writes_only_the_replay_seed():
    """No Q FIFO, no reference bank. If the builder still wrote them, the attach's emptiness
    checks are gone too, so a stale non-empty artifact would go unnoticed."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "build_cold_artifacts.py"), encoding="utf-8").read()
    body = src[src.index("def main("):]
    for bad in ('write_shards([], out, "q_seed")', 'write_shards([], out, "reference_bank")'):
        assert bad not in body, f"the cold builder still writes {bad}"
    assert 'q_seed_manifest.json' not in body and 'reference_bank_manifest.json' not in body
    assert '"replay_buffer"' in body, "the replay seed is no longer built"
    code = _attach_code()
    assert "q_seed_manifest.json" not in code, (
        "the attach still requires a q_seed manifest the builder no longer writes -- the launch "
        "would refuse every time"
    )


def test_single_factor_diff_against_the_main_run():
    """The ablation's whole value is the size of its diff. verify_ablation_diff.sh enforces it;
    this test enforces that the checker exists, is executable, and covers the cases that matter
    -- including a core hyper going MISSING, which would otherwise shrink the diff and read as
    success."""
    here = os.path.dirname(os.path.abspath(__file__))
    v = os.path.join(here, "verify_ablation_diff.sh")
    assert os.path.exists(v), "verify_ablation_diff.sh is missing"
    assert os.access(v, os.X_OK), "verify_ablation_diff.sh is not executable"
    src = open(v, encoding="utf-8").read()
    assert "UNEXPECTED REMOVAL" in src and "UNEXPECTED ADDITION" in src
    assert "CORE HYPER NOT SHARED" in src, (
        "the checker does not detect a core hyper going missing, so an empty diff would pass"
    )
    for k in ("SP_TRAIN_BATCH_SIZE=256", "SP_REPLAY_N=192", "SP_PPO_MINI_BATCH=96",
              "SP_LR=2e-6", "SP_REPLAY_CUT_GRAIN=10000", "SP_ROLLOUT_TP=4"):
        assert k in src, f"{k} is not in the checker's must-be-shared list"
    # and the arm really does pin them
    code = _attach_code()
    for k in ("SP_TRAIN_BATCH_SIZE=256", "SP_REPLAY_N=192", "SP_PPO_MINI_BATCH=96", "SP_LR=2e-6"):
        assert f'"{k}"' in code, f"the attach does not pin {k}"



def test_question_sampling_is_optin_and_resume_safe():
    """SP_REPLAY_GLOBAL_SAMPLING=question changes the replay draw, so it must be opt-in and must
    not break resuming a run that uses the default 'entry' behaviour:
      (a) the default stays 'entry';
      (b) global_sampling is stamped into policy_dict ONLY when non-default -- that dict is
          compared for EXACT EQUALITY on resume, so an unconditional key would fail every
          existing checkpoint with a spurious policy-drift error (sp_replay applies the same
          rule to cold_bootstrap)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src_p = os.path.join(root, "src", "verl", "verl", "trainer", "ppo", "sp_replay.py")
    assert os.path.exists(src_p), f"verl fork not found at {src_p}"
    src = open(src_p, encoding="utf-8").read()

    m = re.search(r'cfg\.get\("sp_replay_global_sampling",\s*"(\w+)"\)', src)
    assert m, "sp_replay no longer reads sp_replay_global_sampling"
    assert m.group(1) == "entry", (
        f"the FORK default is {m.group(1)!r}, not 'entry' -- this silently changes the replay "
        "draw for every run using bucketing=global, including other runs"
    )
    # the base policy_dict literal must NOT carry the key
    base = src[src.index("self.policy_dict = {"):]
    base = base[:base.index("}")]
    assert "global_sampling" not in base, (
        "global_sampling is in the policy_dict literal; policy_dict is compared for exact "
        "equality on resume, so every pre-existing checkpoint would fail with policy drift"
    )
    assert re.search(r'if self\.global_sampling != "entry":\s*\n\s*self\.policy_dict\[', src), (
        "global_sampling is not conditionally stamped -- see the cold_bootstrap precedent"
    )
    # invalid values must be refused, not silently treated as entry
    assert "must be 'entry' or 'question'" in src, "an invalid mode is not rejected"
    # determinism must be local to the new branch, not inherited from _flat_entries' sort
    qb = src[src.index('if self.global_sampling == "question":'):]
    qb = qb[:qb.index("else:")]
    assert 'buffer_seq' in qb and '.sort(' in qb, (
        "the question branch does not sort its buckets, so its draw depends on the order "
        "_flat_entries happens to return -- changing that distant sort would silently change "
        "which trajectory each slot draws"
    )
    for metric in ("replay/distinct_questions", "replay/question_repeat_factor"):
        assert metric in src, f"{metric} is not reported; the wraparound would be invisible"


def test_both_arms_pin_the_same_draw_policy():
    """The draw policy must move in BOTH arms together or the ablation silently becomes a
    multi-factor comparison (Q removed AND a different replay draw)."""
    here = os.path.dirname(os.path.abspath(__file__))
    main = os.path.join(os.path.dirname(here), "08_13_tiedq_seed192", "run_attach_cluster_b.sh")
    if not os.path.exists(main):
        return
    mine = _attach_code()
    theirs = "\n".join(l for l in open(main, encoding="utf-8").read().splitlines()
                        if not l.lstrip().startswith("#"))
    for k in ("SP_REPLAY_BOUND", "SP_REPLAY_GLOBAL_SAMPLING", "SP_REPLAY_N",
              "SP_REPLAY_BUCKETING", "SP_REPLAY_ROTATION", "SP_REPLAY_ADMISSION"):
        a = re.search(rf'"{k}=([^"]*)"', mine)
        b = re.search(rf'"{k}=([^"]*)"', theirs)
        assert a and b, f"{k} is not pinned in both arms (this={bool(a)} main={bool(b)})"
        assert a.group(1) == b.group(1), (
            f"{k} differs: this arm {a.group(1)!r} vs the main run {b.group(1)!r} -- the "
            "ablation is no longer single-factor"
        )
    assert re.search(r'"SP_REPLAY_GLOBAL_SAMPLING=question"', mine), (
        "the arms agree but on the wrong value: the 2026-08-11 decision is uniform over "
        "QUESTIONS, not entries"
    )
    assert re.search(r'"SP_REPLAY_BOUND=256"', mine), "the FIFO bound should be 256"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if not failures else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)
