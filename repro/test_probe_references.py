"""Bidirectional checkpoint/reference/context checks for R5 measurement."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_q", ROOT / "experiments/08_15_q_probe_step40/probe_q.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode()


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class ProbeReferences(unittest.TestCase):
    def setUp(self):
        base = ROOT / "runs/probe-tests"
        base.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(dir=base))
        self.tok = Tokenizer()
        self.bank = self.root / "seed"
        shard = self.bank / "reference_bank/shard_0000.jsonl"
        lines(shard, [{"qid": "seed", "proof": "Seed proof"}])
        (self.bank / "reference_bank_manifest.json").write_text(json.dumps({"shards": {
            "reference_bank/shard_0000.jsonl": hashlib.sha256(shard.read_bytes()).hexdigest()}}))
        lines(self.root / "q_state_deltas.jsonl", [
            {"dataset_step": 0, "bank_added": [{"qid": "q", "proof": "First bank proof"}]},
            {"dataset_step": 1, "bank_added": [{"qid": "q", "proof": "Replacement forbidden"},
                                              {"qid": "future", "proof": "Future proof"}]}])
        def entry(eid, qid, passed):
            return {"entry_id": eid, "qid": qid, "meta": {"judge_pass": passed, "buffer_seq": 0},
                    "response_token_ids": self.tok.encode("<proof>Trajectory proof</proof>")}
        lines(self.root / "replay_buffer_deltas.jsonl", [
            {"dataset_step": 0, "added_entries": [entry("e1", "q", 1), entry("e2", "failed", 0)]},
            {"dataset_step": 1, "replaced_entry_ids": ["e1"], "added_entries": [entry("e3", "future", 1)]}])

    def test_reference_state_boundary_and_bank_add_once(self):
        before = probe.build_ref_map(str(self.root), self.tok, upto_step=0)
        after = probe.build_ref_map(str(self.root), self.tok, upto_step=1)
        self.assertEqual(before, {"q": ["Trajectory proof"]})
        self.assertEqual(after, {"future": ["Trajectory proof"]})
        self.assertEqual(probe.build_bank_map(self.root, self.bank, 0),
                         {"seed": "Seed proof", "q": "First bank proof"})
        self.assertEqual(probe.build_bank_map(self.root, self.bank, 1)["q"], "First bank proof")
        self.assertNotIn("future", probe.build_bank_map(self.root, self.bank, 0))

    def test_verify_bank_context_and_reject_missing_or_wrong_context(self):
        prompt, response = [65], [66]
        ctx = probe.make_ctx_builder(self.tok)(prompt, response, "First bank proof")
        lines(self.root / "rollouts/2.jsonl", [{"uid": "u", "prompt_token_ids": prompt,
                                              "response_token_ids": response}])
        wave = {"uid": "u", "qid": "q", "kind": "consumed", "variant": "ref", "ctx_token_ids": ctx}
        args = SimpleNamespace(model="unused", run_dir=str(self.root), bank_dir=str(self.bank),
            verify_upto=None, verify_step=2, no_require_pass=False, variant="reward_horizon",
            ref_try=8, verify_n=200, budget_g=1)
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=self.tok):
            lines(self.root / "q_wave/1.jsonl", [wave])
            self.assertEqual(probe.do_verify(args), 0)
            lines(self.root / "q_wave/1.jsonl", [dict(wave, ctx_token_ids=ctx + [67])])
            self.assertEqual(probe.do_verify(args), 1)
            # A row missing from rollout evidence must fail, even with valid bank state.
            lines(self.root / "q_wave/1.jsonl", [dict(wave, uid="missing")])
            self.assertEqual(probe.do_verify(args), 1)
            lines(self.root / "q_wave/1.jsonl", [])
            self.assertEqual(probe.do_verify(args), 1)

    def test_fractional_failed_and_future_rollouts_rejected(self):
        import pandas as pd
        from verl.trainer.ppo.difficulty import qid_from_messages
        prompts = [[{"role": "user", "content": s}] for s in ("Failed", "Passed", "Future")]
        pd.DataFrame({"prompt": prompts}).to_parquet(self.root / "train.parquet")
        def row(index, acc, passed):
            return {"extra_index": index, "acc": acc, "prover_judge_score": passed,
                    "response_token_ids": self.tok.encode("<proof>Proof</proof>")}
        lines(self.root / "rollouts/1.jsonl", [row(0, 1/7, 0), row(1, 6/7, 1)])
        lines(self.root / "rollouts/2.jsonl", [row(2, 1, 1)])
        refs = probe.build_ref_map_rollouts(self.root, self.root, self.tok, steps={1})
        self.assertEqual(refs, {qid_from_messages(prompts[1]): "Proof"})
        self.assertEqual(probe.build_ref_map_rollouts(self.root, self.root, self.tok, steps=set()), {})


class ProbeJudge(unittest.IsolatedAsyncioTestCase):
    async def test_full_attempt_and_error_rejection(self):
        import pandas as pd
        from ac2.rewards import ds4_finegrained_judge as judge
        spec = importlib.util.spec_from_file_location("probe_judge", ROOT / "experiments/08_15_q_probe_step40/probe_judge.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base = ROOT / "runs/probe-tests"
        base.mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(dir=base))
        pd.DataFrame({"extra_info": [{"theorem": "Problem"}]}).to_parquet(root / "train.parquet")
        full = "<proof>Prefix and continuation</proof>"
        lines(root / "gen.shard0.jsonl", [{"qid": "q", "row_index": 0, "completions": [
            {"text": "continuation</proof>", "full_attempt_text": full, "exceeds_g": True}]}])
        args = SimpleNamespace(data_dir=str(root), gen_dir=str(root), num_shards=1, shard=0,
            only_exceeds=False, concurrency=4, out=str(root / "judged.jsonl"),
            data_source="fineproofs-rl", judge_url="unused", judge_max_tokens=40000)
        result = judge._empty_extras()
        with patch.object(judge, "compute_score", AsyncMock(return_value=result)) as call:
            self.assertEqual(await module.main_async(args), 0)
            self.assertEqual(call.call_args.kwargs["solution_str"], full)
            self.assertEqual(call.call_args.kwargs["extra_info"], {"theorem": "Problem"})
        for flag in ("judge_parse_failed", "judge_http_error", "judge_truncated"):
            args.out = str(root / (flag + ".jsonl"))
            with patch.object(judge, "compute_score", AsyncMock(return_value=dict(result, **{flag: 1}))), \
                 self.assertRaises(AssertionError):
                await module.main_async(args)


if __name__ == "__main__":
    unittest.main()
