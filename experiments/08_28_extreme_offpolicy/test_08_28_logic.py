"""Pure-logic tests for the 08_28 burst-inflow additions. No torch, no GPU, no cluster.

Covers the pieces of new decision logic that would otherwise be first exercised on a
32-GPU allocation:

  1. `_inflow_stamp` — the burst schedule (burst_n on step % period == phase, -1 otherwise,
     constant 1 with no schedule), loaded by AST surgery from q_dataset.py.
  2. `_sp_per_row_repeat_counts` — the trainer's three-valued resolver (positive = as-is,
     0 = config n, NEGATIVE = drop), loaded by AST surgery from the vendored ray_trainer.py,
     including the all-dropped assert and backward compatibility with the legacy {0,1} stamp.
  3. Zero-count row dropping keeps the two repeat sites aligned — the numpy half of
     `sample_level_repeat` (np.repeat with a 0 in counts) drops the same rows in the same
     order for any per-key array.
  4. Whole-burst FIFO turnover — a real `ReplayHarness` (numpy only) at
     bound == n_orig * burst_n: each burst's admission evicts the previous burst COMPLETELY,
     `replay_plan_for_step` under global_sampling=question covers every burst problem exactly
     once per step, and a zero-inflow step admits nothing and evicts nothing.

Run:  python experiments/08_28_extreme_offpolicy/test_08_28_logic.py
   or: pytest experiments/08_28_extreme_offpolicy/test_08_28_logic.py

Sources are loaded by AST surgery rather than by importing `verl` (which needs torch/ray), so
these run anywhere — including a laptop — and fail loudly if the shapes they assume change.
"""
from __future__ import annotations

import ast
import json
import os
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
Q_DATASET = os.path.join(HERE, "q_dataset.py")
RAY_TRAINER = os.path.join(REPO, "src", "verl", "verl", "trainer", "ppo", "ray_trainer.py")
SP_REPLAY = os.path.join(REPO, "src", "verl", "verl", "trainer", "ppo", "sp_replay.py")


def _load_method(path: str, class_name: str, method_name: str):
    """Extract one method from a class in `path` and return it as a plain function."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    mod = ast.Module(body=[item], type_ignores=[])
                    ns: dict = {"print": print}
                    exec(compile(ast.fix_missing_locations(mod), f"<{method_name}>", "exec"), ns)
                    return ns[method_name]
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


# ------------------------------------------------------- 1. the burst schedule stamp
def test_inflow_stamp_schedule():
    stamp = _load_method(Q_DATASET, "SPQReadinessDataset", "_inflow_stamp")

    # no schedule -> the constant-1 stamp on every step (main-run behavior)
    legacy = types.SimpleNamespace(_burst_period=0)
    assert all(stamp(legacy, s) == 1 for s in range(25))

    # production schedule: 10 rollouts on steps == 0 (mod 10), drop sentinel otherwise
    prod = types.SimpleNamespace(_burst_period=10, _burst_n=10, _burst_phase=0)
    got = [stamp(prod, s) for s in range(25)]
    assert got[0] == 10 and got[10] == 10 and got[20] == 10, got
    assert all(v == -1 for i, v in enumerate(got) if i % 10 != 0), got

    # phase shifts the burst, period respected
    ph = types.SimpleNamespace(_burst_period=5, _burst_n=3, _burst_phase=2)
    assert [stamp(ph, s) for s in range(7)] == [-1, -1, 3, -1, -1, -1, -1]

    # pure function of step: same input, same output (resume-rebuild contract)
    assert [stamp(prod, 10)] * 3 == [stamp(prod, 10) for _ in range(3)]
    print("ok: _inflow_stamp schedule")


# ------------------------------------------------------- 2. the trainer resolver
class _FakeCfg:
    def __init__(self, n):
        self.algorithm = types.SimpleNamespace(adv_estimator="grpo")
        self.actor_rollout_ref = types.SimpleNamespace(
            rollout=types.SimpleNamespace(n=n))


class _FakeBatch:
    def __init__(self, stamps):
        self.non_tensor_batch = {
            "extra_info": [{"sp_rollout_n": s, "sp_source_type": "x"} for s in stamps]
        }
        self.batch = None


def _resolver():
    fn = _load_method(RAY_TRAINER, "RayPPOTrainer", "_sp_per_row_repeat_counts")
    # the method references AdvantageEstimator.REMAX; provide a distinct dummy
    fn.__globals__["AdvantageEstimator"] = types.SimpleNamespace(REMAX=object())
    return fn


def test_resolver_three_valued():
    fn = _resolver()
    self = types.SimpleNamespace(_sp_inflow_only=lambda: True, config=_FakeCfg(n=16))

    # burst step: 2 inflow rows x10, 2 train rows -> config n
    assert fn(self, _FakeBatch([10, 10, 0, 0])) == [10, 10, 16, 16]
    # non-burst step: inflow rows dropped (0), train rows untouched
    assert fn(self, _FakeBatch([-1, -1, 0, 0])) == [0, 0, 16, 16]
    # legacy {0,1} stamp resolves exactly as before (runs without a burst schedule)
    assert fn(self, _FakeBatch([1, 1, 0, 0])) == [1, 1, 16, 16]
    # inflow-only off -> None (uniform repeat path)
    off = types.SimpleNamespace(_sp_inflow_only=lambda: False, config=_FakeCfg(n=16))
    assert fn(off, _FakeBatch([1, 0])) is None
    print("ok: resolver three-valued semantics")


def test_resolver_refuses_empty_step():
    fn = _resolver()
    self = types.SimpleNamespace(_sp_inflow_only=lambda: True, config=_FakeCfg(n=16))
    try:
        fn(self, _FakeBatch([-1, -1, -1]))
    except AssertionError as e:
        assert "0 rollouts" in str(e)
    else:
        raise AssertionError("all-dropped batch must be refused, not silently emptied")
    print("ok: resolver refuses an all-dropped batch")


# ------------------------------------------------------- 3. zero-count repeat alignment
def test_zero_count_repeat_alignment():
    import numpy as np

    counts = [0, 3, 0, 1, 16]
    a = np.array(["r0", "r1", "r2", "r3", "r4"], dtype=object)
    b = np.arange(5)
    ra, rb = np.repeat(a, counts, axis=0), np.repeat(b, counts, axis=0)
    assert ra.tolist() == ["r1"] * 3 + ["r3"] + ["r4"] * 16
    # every key drops the SAME rows in the SAME order -> the two repeat sites stay aligned
    assert [a.tolist().index(x) for x in ra] == rb.tolist()
    assert len(ra) == sum(counts)
    print("ok: zero-count repeat keeps per-key alignment")


# ------------------------------------------------------- 4. whole-burst FIFO turnover
def _load_sp_replay():
    """Load sp_replay.py by FILE path: the verl package __init__ imports ray/torch, but the
    module itself needs only stdlib + numpy."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("sp_replay_isolated", SP_REPLAY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_harness(tmp, n_base=8, n_replay=4, bound=12):
    """Real ReplayHarness on an empty cold seed (numpy only; no torch anywhere)."""
    sp_replay = _load_sp_replay()

    os.environ["SP_REPLAY_ENABLE"] = "1"
    seed = os.path.join(tmp, "replay_seed_cold")
    os.makedirs(os.path.join(seed, "replay_buffer"), exist_ok=True)
    shard = os.path.join(seed, "replay_buffer", "shard_0000.jsonl")
    open(shard, "w").close()
    manifest = {"shards": {os.path.relpath(shard, seed): sp_replay.sha256_file(shard)}}
    with open(os.path.join(seed, sp_replay.MANIFEST_NAME), "w") as f:
        json.dump(manifest, f)
    cfg = {
        "sp_replay_seed_dir": seed,
        "sp_replay_delta_dir": os.path.join(tmp, "run_data"),
        "train_batch_size": n_base,
        "sp_replay_n": n_replay,
        "sp_replay_bucketing": "global",
        "sp_replay_admission": "ungated",
        "sp_replay_rotation": "global_fifo",
        "sp_replay_bound": bound,
        "sp_replay_global_sampling": "question",
        "sp_replay_cold_bootstrap": 1,
        "sp_replay_rng_seed": 714001,
    }
    qids = [f"q{i:03d}" for i in range(40)]
    h = sp_replay.ReplayHarness(cfg, num_statements=40, row_qids=qids)
    h.finalize_from_checkpoint(None)
    return h


def _admit_burst(h, step, problems, per_problem):
    """Drive _admit_global with a fabricated statement lane (ungated admits everything)."""
    extra, passes, pjs, uids, toks = [], [], [], [], []
    for q in problems:
        for r in range(per_problem):
            extra.append({"sp_source_type": "statement", "sp_qid": q, "sp_prefix_len": 0})
            passes.append(1)
            pjs.append(1.0)
            uids.append(f"u{step}:{q}:{r}")
            toks.append([1, 2, 3, step])
    added, replaced, _ = h._admit_global(
        extra, passes, pjs, uids, step, lambda i: toks[i]
    )
    return added, replaced


def test_whole_burst_fifo_turnover():
    n_orig, burst_n = 4, 3           # bound = 12 = exactly one burst
    with tempfile.TemporaryDirectory() as tmp:
        h = _mk_harness(tmp, n_base=8, n_replay=4, bound=n_orig * burst_n)

        # burst 0 (dataset step 0): buffer empty -> cold shortfall, then 12 admits
        qids0, fill0 = h.replay_plan_for_step(0)
        assert qids0 == [] and len(fill0) == 4, (qids0, fill0)   # cold_scratch fill
        p0 = [f"q{i:03d}" for i in range(n_orig)]
        added, replaced = _admit_burst(h, 0, p0, burst_n)
        assert len(added) == 12 and replaced == [], (len(added), replaced)
        assert sum(len(b) for b in h.entries.values()) == 12

        # steps 1..9: draws cover every burst problem exactly once per step (question mode),
        # and a zero-inflow step admits nothing and evicts nothing.
        for step in range(1, 4):
            h._global_slot_cache = None       # per-step cache is keyed by step anyway
            qids, fill = h.replay_plan_for_step(step)
            assert fill == [] and sorted(qids) == sorted(p0), (step, qids)
            added, replaced = _admit_burst(h, step, [], burst_n)   # no statement rows
            assert added == [] and replaced == []
            assert sum(len(b) for b in h.entries.values()) == 12

        # burst 1 (step 10): 12 new admits evict the previous burst COMPLETELY
        p1 = [f"q{i:03d}" for i in range(20, 20 + n_orig)]
        added, replaced = _admit_burst(h, 10, p1, burst_n)
        assert len(added) == 12 and len(replaced) == 12
        live = [e for b in h.entries.values() for e in b]
        assert len(live) == 12
        assert {e["meta"]["dataset_step"] for e in live} == {10}, "old burst survived"
        assert sorted(h.entries) == sorted(p1)

        # draws after the roll come from burst 1 only
        h._global_slot_cache = None
        qids, fill = h.replay_plan_for_step(11)
        assert fill == [] and sorted(qids) == sorted(p1)
    print("ok: whole-burst FIFO turnover (admit 12 -> evict exactly the previous 12)")


def test_partial_bound_would_collapse_diversity():
    """Documents WHY the runner hard-asserts bound == n_orig * burst_n: at a smaller bound
    the FIFO keeps the tail of the admission walk, i.e. the LAST few problems x burst_n."""
    n_orig, burst_n = 4, 3
    with tempfile.TemporaryDirectory() as tmp:
        h = _mk_harness(tmp, n_base=8, n_replay=4, bound=6)      # half a burst
        p0 = [f"q{i:03d}" for i in range(n_orig)]
        _admit_burst(h, 0, p0, burst_n)
        live_q = sorted(h.entries)
        # only the last 2 of 4 problems survive -- the collapse the assert exists to prevent
        assert live_q == ["q002", "q003"], live_q
        assert all(len(h.entries[q]) == burst_n for q in live_q)
    print("ok: sub-burst bound keeps only the tail problems (assert justified)")


# ------------------------------------------------- 5. panel-1 inflow-lane mask (dashboard)
DASHBOARD = os.path.join(REPO, "src", "ac2", "viz", "single_run_dashboard.py")


def _load_function(path: str, func_name: str, inject: dict):
    """Extract one MODULE-LEVEL function from `path` (AST surgery, no heavy imports)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            mod = ast.Module(body=[node], type_ignores=[])
            ns = dict(inject)
            exec(compile(ast.fix_missing_locations(mod), f"<{func_name}>", "exec"), ns)
            return ns[func_name]
    raise AssertionError(f"{func_name} not found in {path}")


class _FakeF:
    def __init__(self, n, keys):
        self.n, self.keys = n, dict(keys)

    def arr(self, key, scale=1.0):
        import numpy as np
        a = np.asarray(self.keys.get(key, np.full(self.n, np.nan)), dtype=float)
        return a * scale


def _lane_series():
    import numpy as np
    return _load_function(
        DASHBOARD, "_inflow_trained_lane_series",
        {"np": np, "has": lambda a: a is not None and np.asarray(a).size > 0
                                    and np.isfinite(a).any()},
    )


def _steps(n, finite_at, val=1.0):
    import numpy as np
    a = np.full(n, np.nan)
    for i in finite_at:
        a[i] = val
    return a


def test_panel1_mask_covers_burst_run_zero_inflow_steps():
    """Burst-run metrics with sp_inflow + orig keys only on burst steps {0,10};
    naive replay keys every step; HT keys from step 6. Every step must map to the
    trained lane (mask all True), HT preferred where present."""
    import numpy as np
    fn = _lane_series()
    n = 12
    F = _FakeF(n, {
        "sp_inflow__trained_rows": _steps(n, [0, 10], 192),
        "replay__pass_rate_orig": _steps(n, [0, 10], 0.30),
        "replay__pass_rate_replay": np.full(n, 0.40),
        "replay__best_at_n_replay": np.full(n, 0.60),
        "v7__train__prover__replay_stream_pass_full_suffix_est": _steps(n, range(6, 12), 0.45),
    })
    mask, pm, bn, _s7 = fn(F)
    assert mask.all(), mask
    assert np.allclose(pm[:6], 40.0) and np.allclose(pm[6:], 45.0), pm   # HT wins where present
    assert np.allclose(bn, 60.0), bn
    print("ok: panel-1 mask covers a burst run's zero-inflow steps (HT preferred)")


def test_panel1_mask_unchanged_on_every_step_inflow_runs():
    """Every-step refill runs (e.g. the main run): sp_inflow logged every step -> mask must
    equal the original definition exactly (unchanged rendering for those runs)."""
    import numpy as np
    fn = _lane_series()
    n = 8
    F = _FakeF(n, {
        "sp_inflow__trained_rows": np.full(n, 192.0),
        "replay__pass_rate_orig": np.full(n, 0.30),
        "replay__pass_rate_replay": np.full(n, 0.40),
        "replay__best_at_n_replay": np.full(n, 0.60),
    })
    mask, pm, _bn, _s7 = fn(F)
    assert mask.all() and np.allclose(pm, 40.0)
    print("ok: panel-1 mask unchanged on every-step-inflow runs")


def test_panel1_mask_leaves_grafted_parent_steps_alone():
    """A branched run whose parent steps (0-4) predate the sp_inflow metrics: they have
    replay estimates and a statement stream but NO sp_inflow key -> they must keep their
    orig-stream rendering (mask False); the child's steps (5-11) join the lane."""
    fn = _lane_series()
    n = 12
    F = _FakeF(n, {
        "sp_inflow__trained_rows": _steps(n, range(5, 12), 192),
        "replay__pass_rate_orig": _steps(n, range(0, 12), 0.30),   # statement stream everywhere
        "replay__pass_rate_replay": _steps(n, range(0, 12), 0.40),
    })
    mask, _pm, _bn, _s7 = fn(F)
    assert not mask[:5].any(), mask   # parent steps stay orig-stream
    assert mask[5:].all(), mask
    print("ok: panel-1 mask leaves older grafted parent steps alone")


def test_panel1_mask_absent_without_sp_inflow_key():
    fn = _lane_series()
    F = _FakeF(6, {"replay__pass_rate_replay": _steps(6, range(6), 0.4)})
    assert fn(F) == (None, None, None, None)
    print("ok: panel-1 lane disabled on runs that never logged sp_inflow keys")


if __name__ == "__main__":
    test_inflow_stamp_schedule()
    test_resolver_three_valued()
    test_resolver_refuses_empty_step()
    test_zero_count_repeat_alignment()
    test_whole_burst_fifo_turnover()
    test_partial_bound_would_collapse_diversity()
    test_panel1_mask_covers_burst_run_zero_inflow_steps()
    test_panel1_mask_unchanged_on_every_step_inflow_runs()
    test_panel1_mask_leaves_grafted_parent_steps_alone()
    test_panel1_mask_absent_without_sp_inflow_key()
    print("ALL OK")
