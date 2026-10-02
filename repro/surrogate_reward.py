"""Strict surrogate reward: reject API routing, changed prompts, and failed grades."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from surrogate_common import ROOT, URL, MAX_TOKENS, EFFORT, templates
from strict_reward import compute_score as strict_score


async def compute_score(*args, **kwargs):
    templates()
    assert kwargs["judge_url"] == URL and kwargs["judge_payload_style"] == "gptoss"
    assert kwargs["judge_max_tokens"] == MAX_TOKENS and kwargs["judge_reasoning_effort"] == EFFORT
    for key, name in (("finegrained_template_path", "finegrained_noref_judge.txt"),
                      ("val_judge_template_path", "imo_proofautograder.txt")):
        canonical = ROOT / "src/ac2/rewards/templates" / name
        assert Path(kwargs.get(key) or canonical).resolve() == canonical, "surrogate template override rejected"
    assert not kwargs.get("train_rubric_template_path"), "alternate train rubric rejected"
    return await strict_score(*args, **kwargs)
