import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from surrogate_calibrate import statistics
from surrogate_common import URL, MAX_TOKENS, EFFORT, templates, route
import surrogate_reward


class SurrogateTests(unittest.TestCase):
    def rows(self, x, y):
        return [{"id": str(i), "deepseek_points": a, "surrogate_points": b,
                 "deepseek_truncated": False, "surrogate_truncated": False}
                for i, (a, b) in enumerate(zip(x, y))]

    def test_agreement_both_directions(self):
        same = statistics(self.rows([0, 1, 6, 7], [0, 1, 6, 7]))
        self.assertEqual((same["exact_points_agreement"], same["pass_cohen_kappa"], same["spearman"]), (1, 1, 1))
        opposite = statistics(self.rows([0, 1, 6, 7], [7, 6, 1, 0]))
        self.assertEqual((opposite["pass_agreement"], opposite["pass_cohen_kappa"], opposite["spearman"]), (0, -1, -1))
        self.assertEqual(opposite["mean_absolute_point_difference"], 6)

    def test_invalid_and_constant_not_zero_filled(self):
        rows = self.rows([0, 0, None], [0, 0, 7])
        stats = statistics(rows)
        self.assertEqual(stats["paired"], 2)
        self.assertEqual(stats["excluded_ids"], ["2"])
        self.assertIsNone(stats["pass_cohen_kappa"])
        self.assertIsNone(stats["spearman"])
        rows[0]["surrogate_truncated"] = True
        self.assertEqual(statistics(rows)["paired"], 1)

    def test_template_routes(self):
        for kind, template in templates().items():
            prompt = template.format(problem="P", proof="C", solution="R", guidelines="G")
            self.assertEqual(route(prompt), kind)
        with self.assertRaises(AssertionError):
            route("unrecognized prompt")

    def test_reward_rejects_api_and_preserves_call(self):
        kwargs = dict(judge_url=URL, judge_payload_style="gptoss", judge_max_tokens=MAX_TOKENS,
                      judge_reasoning_effort=EFFORT)
        with patch.object(surrogate_reward, "strict_score", new=AsyncMock(return_value={"score": 1})) as call:
            self.assertEqual(asyncio.run(surrogate_reward.compute_score(**kwargs)), {"score": 1})
            self.assertEqual(call.call_count, 1)
            for change in (dict(judge_url="http://127.0.0.1:18791/v1"), dict(judge_payload_style="deepseek_v4"),
                           dict(judge_max_tokens=40000), dict(judge_reasoning_effort="low"),
                           dict(finegrained_template_path="/tmp/custom-train.txt"),
                           dict(val_judge_template_path="/tmp/custom-val.txt"),
                           dict(train_rubric_template_path="/tmp/custom-rubric.txt")):
                with self.assertRaises(AssertionError):
                    asyncio.run(surrogate_reward.compute_score(**(kwargs | change)))
            self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
