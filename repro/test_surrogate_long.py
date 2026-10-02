"""Synthetic admission tests; these never claim a real 200-step run occurred."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import r_curves
import surrogate_long_acceptance as accept
from surrogate_common import ROOT, profile
from surrogate_smoke_receipt import union_seconds
from surrogate_chain import source_identity


class LongAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Path(tempfile.mkdtemp(prefix="long-admission-", dir=ROOT / "runs/surrogate"))
        cls.run_dir, cls.launch = cls.base / "r1", cls.base / "launch"
        cls.run_dir.mkdir(); cls.launch.mkdir()
        cls.report = {"engineering_smoke": False, "training": [{"step": i} for i in range(1, 201)],
                      "curve": [{"step": i, "n_problems": 60, "samples_per_problem": 4} for i in range(0, 201, 10)],
                      "match_fields": {"judge_profile": profile()}}
        cmd = ["python", str(ROOT / "repro/local_runner.py"), "--run-dir", str(cls.run_dir), "--steps", "200",
               "--save-freq", "20", "--judge-backend", "surrogate", "--val-freq", "10", "--initial-val"]
        cls.values = {"launch.json": {"task": "r1", "physical_gpus": list(range(1, 8)), "command": cmd},
                      "result.json": {"returncode": 0, "remaining_owned_processes": 0, "monitor_error": None}, "api_before.json": []}
        for name, value in cls.values.items():
            (cls.launch / name).write_text(json.dumps(value))
        (cls.launch / "stdout.log").write_text("rank 0 nranks 7 fixture Init COMPLETE\n")
        identities = json.loads((ROOT / "repro/receipts/e1-acceptance-uuid.json").read_text())["kernel_checks"]
        (cls.launch / "gpu.csv").write_text("".join(f"fixture,{r['physical_gpu']},GPU-{r['uuid']},50,100\n" for r in identities for _ in range(3)))
        for step in range(20, 201, 20):
            actor = cls.run_dir / f"checkpoints/global_step_{step}/actor"
            actor.mkdir(parents=True)
            for rank in range(7):
                for kind in ("model", "optim", "extra_state"):
                    (actor / f"{kind}_world_size_7_rank_{rank}.pt").write_bytes(b"fixture")
        (cls.run_dir / "checkpoints/latest_checkpointed_iteration.txt").write_text("200")
        (cls.run_dir / "metrics.jsonl").write_text("".join(json.dumps({"step": i, "data": {"actor/grad_norm": .1}})+"\n" for i in range(1, 201)))

    def collect(self, report=None):
        with patch.object(r_curves, "collect", return_value=copy.deepcopy(report or self.report)), patch.object(accept, "judge_accounting", return_value=[]):
            return accept.collect(self.run_dir, self.launch)

    def test_positive_wrapper(self):
        self.assertEqual(self.collect()["surrogate_acceptance"]["complete_steps"], 200)

    def test_missing_step_validation_and_wrong_judge(self):
        for field in ("training", "curve"):
            changed = copy.deepcopy(self.report); changed[field].pop()
            with self.assertRaises(AssertionError): self.collect(changed)
        changed = copy.deepcopy(self.report); changed["match_fields"]["judge_profile"]["payload_style"] = "deepseek_v4"
        with self.assertRaises(AssertionError): self.collect(changed)

    def test_checkpoint_and_exit_gate(self):
        original = Path.is_file
        with patch.object(Path, "is_file", lambda p: False if p.name == "optim_world_size_7_rank_6.pt" else original(p)):
            with self.assertRaises(AssertionError): self.collect()
        read = Path.read_text
        with patch.object(Path, "read_text", lambda p,*a,**kw: json.dumps({"returncode": 1, "remaining_owned_processes": 0}) if p == self.launch / "result.json" else read(p,*a,**kw)):
            with self.assertRaises(AssertionError): self.collect()

    def test_union_overlap_not_summed_latency(self):
        self.assertEqual(union_seconds([(1, 4), (2, 3), (3, 8), (10, 12)]), 9)
        self.assertEqual(union_seconds([]), 0)

    def test_directory_symlink_inventory(self):
        link = self.base / "directory-link"
        link.symlink_to(self.run_dir, target_is_directory=True)
        self.assertEqual(source_identity(link), {"kind": "symlink", "target": str(self.run_dir)})
        self.assertEqual(source_identity(self.launch / "stdout.log")["kind"], "file")
        with self.assertRaises(AssertionError):
            source_identity(self.base / "absent")


if __name__ == "__main__":
    unittest.main()
