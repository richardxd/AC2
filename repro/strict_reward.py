"""R entry-point adapter: failed judge calls must abort before policy updates."""
from ac2.rewards.ds4_finegrained_judge import compute_score as original_compute_score


async def compute_score(*args, **kwargs):
    result = await original_compute_score(*args, **kwargs)
    failures = {key: result[key] for key in (
        "judge_http_error", "judge_parse_failed", "judge_truncated") if result[key] != 0}
    if failures:
        raise RuntimeError(f"Judge failed; refusing a policy update: {failures}")
    return result
