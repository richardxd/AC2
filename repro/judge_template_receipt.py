"""Verify the training judge's module pin before launch and in archived R sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = "src/ac2/rewards/templates/finegrained_noref_judge.txt"
MODULE = "src/ac2/rewards/ds4_finegrained_judge.py"


def checked_digest(data, pin):
    actual = hashlib.sha256(data).hexdigest()
    assert actual == pin, f"training judge template drift: {actual} != {pin}"
    return actual


def current_template():
    from ac2.rewards import ds4_finegrained_judge as judge
    path = Path(judge._DEFAULT_TEMPLATE_PATH).resolve()
    assert path == ROOT / TEMPLATE, "judge module resolves outside the pinned source"
    return {"path": str(path), "sha256": checked_digest(path.read_bytes(), judge._FINEGRAINED_TEMPLATE_SHA256)}


def archived_template(manifest):
    from omegaconf import OmegaConf
    archive = manifest / "source/code.zip"
    config = manifest / "config.yaml"
    cfg = OmegaConf.load(config)
    kwargs = cfg.reward.custom_reward_function.reward_kwargs
    configured = kwargs.get("finegrained_template_path")
    assert configured is None or Path(configured).resolve() == ROOT / TEMPLATE
    assert not kwargs.get("train_rubric_template_path"), "alternate train prompt configured"
    with zipfile.ZipFile(archive) as z:
        assignments = [n for n in ast.parse(z.read(MODULE)).body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "_FINEGRAINED_TEMPLATE_SHA256" for t in n.targets)]
        assert len(assignments) == 1
        pin = ast.literal_eval(assignments[0].value)
        sha = checked_digest(z.read(TEMPLATE), pin)
    # Compare the independently stored config copy, not an inferred runner default.
    assert OmegaConf.to_container(cfg) == OmegaConf.to_container(OmegaConf.load(manifest / "source/config.yaml"))
    return {"manifest": str(manifest), "template_sha256": sha, "module_pin": pin,
            "configured_template_path": configured, "verification": "archived source bytes",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifests = sorted((args.run / "launches").glob("*/source/code.zip"))
    assert manifests
    receipt = {"run": str(args.run), "launches": [archived_template(p.parent.parent) for p in manifests]}
    with args.output.open("x") as f:
        json.dump(receipt, f, indent=2)
    print(f"JUDGE_TEMPLATES_VERIFIED launches={len(manifests)}")


if __name__ == "__main__":
    main()
