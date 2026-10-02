import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from r5_launch import ROOT, validate_source


class ProbeSourceTests(unittest.TestCase):
    def fixture(self, change=None):
        base = ROOT / "runs/r5-source-tests"
        base.mkdir(exist_ok=True)
        run = Path(tempfile.mkdtemp(dir=base))
        source = ROOT / "runs/research/r2"
        shutil.copytree(source / "cold", run / "cold")
        original = sorted((source / "launches").glob("*/arguments.json"))[-1]
        data = json.loads(original.read_text())
        if change:
            data.update(change)
        path = run / "launches/fixture/arguments.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(data))
        return SimpleNamespace(run=run, data=(ROOT / data["data"]).resolve(), engineering_fixture=False)

    def test_normal_r2_acceptance(self):
        self.assertEqual(len(validate_source(self.fixture())), 1)

    def test_ablation_and_data_mismatch_rejected(self):
        for change in ({"ablation": "no-audit"}, {"ablation": "correct-only"},
                       {"engineering_readiness": True}, {"q_train_n": 32}):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                validate_source(self.fixture(change))
        args = self.fixture()
        args.data = ROOT / "runs/e7/data32"
        with self.assertRaises(AssertionError):
            validate_source(args)

    def test_added_or_changed_seed_shards_rejected(self):
        args = self.fixture()
        (args.run / "cold/reference_bank/shard_9999.jsonl").write_text('{}\n')
        with self.assertRaises(ValueError):
            validate_source(args)
        args = self.fixture()
        with (args.run / "cold/reference_bank/shard_0000.jsonl").open('a') as f:
            f.write('{}\n')
        with self.assertRaises(ValueError):
            validate_source(args)


if __name__ == "__main__":
    unittest.main()
