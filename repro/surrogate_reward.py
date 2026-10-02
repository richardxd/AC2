"""Strict surrogate reward: reject API routing, changed prompts, and failed grades."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from surrogate_common import URL, MAX_TOKENS, EFFORT, templates
from strict_reward import compute_score as strict_score


async def compute_score(*args, **kwargs):
    templates()
    assert kwargs["judge_url"] == URL and kwargs["judge_payload_style"] == "gptoss"
    assert kwargs["judge_max_tokens"] == MAX_TOKENS and kwargs["judge_reasoning_effort"] == EFFORT
    return await strict_score(*args, **kwargs)
