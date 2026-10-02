"""Bidirectional checks of gate evidence acceptance before model generation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from gate_report import summarize
from local_runner import REVISIONS


class GateEvidenceTest(unittest.TestCase):
    def setUp(self):
        # Preserve fixtures as review receipts rather than deleting artifacts.
        root = Path(__file__).resolve().parents[1] / "runs/e9/validator"
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="case-", dir=root))
        self.parts = []
        for shard in range(7):
            rows = [{"index": i, "response_token_ids": [11, 12], "finish_reason": "stop"}
                    for i in range(shard, 60, 7)]
            self.parts.append({"model": "Qwen/Qwen3-4B-Thinking-2507",
                "revision": REVISIONS["Qwen/Qwen3-4B-Thinking-2507"],
                "data_sha256": "fixture", "gpu_uuid": str(shard),
                "configuration": {"engine_seed": 192, "request_seed": "192 + canonical problem index",
                    "temperature": .8, "top_p": 1., "top_k": -1, "response_budget": 16384,
                    "max_num_seqs": 4, "tp": 1},
                "rows": rows, "generated_tokens": len(rows) * 2,
                "generation_wall_s": 10., "tokens_per_s": len(rows) / 5})
        grades = [{"index": i, "result": {"score": (i % 8) / 7,
                  "judge_parse_failed": 0, "judge_http_error": 0, "judge_truncated": 0}}
                  for i in range(60)]
        (self.directory / "grades.jsonl").write_text("".join(json.dumps(r) + "\n" for r in grades))
        (self.directory / "judge_before.json").write_text("[]")
        self.calls = [{"id": "fixture", "state": "complete", "charged_upper_usd": .1}]

    def evaluate(self):
        for shard, part in enumerate(self.parts):
            (self.directory / f"generation-{shard}.json").write_text(json.dumps(part))
        (self.directory / "judge_after.json").write_text(json.dumps(self.calls))
        return summarize(self.directory)

    def test_valid(self):
        result = self.evaluate()
        self.assertEqual(result["generated_tokens"], 120)
        self.assertAlmostEqual(result["tokens_per_gpu_generation_second"], 120 / 70)
        self.assertEqual(result["judge_charged_upper_usd"], .1)

    def test_invalid_generation(self):
        cases = [
            ("revision", "wrong"), ("gpu_uuid", "1"), ("generation_wall_s", 0),
            ("generation_wall_s", float("nan")), ("generated_tokens", 1),
            ("configuration", {}),
        ]
        original = copy.deepcopy(self.parts[0])
        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.parts[0] = copy.deepcopy(original)
                self.parts[0][key] = value
                with self.assertRaises(AssertionError):
                    self.evaluate()

    def test_duplicate_row(self):
        self.parts[0]["rows"].append(copy.deepcopy(self.parts[0]["rows"][0]))
        self.parts[0]["generated_tokens"] += 2
        with self.assertRaises(AssertionError):
            self.evaluate()

    def test_incomplete_accounting(self):
        self.calls[0]["state"] = "pending"
        with self.assertRaises(AssertionError):
            self.evaluate()

    def test_aborted_generation(self):
        self.parts[0]["rows"][0]["finish_reason"] = "abort"
        with self.assertRaises(AssertionError):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()
