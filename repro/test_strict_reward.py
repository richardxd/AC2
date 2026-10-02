"""Exercise actual judge parsing through the fail-loud R adapter, without HTTP."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import strict_reward
from ac2.rewards import ds4_finegrained_judge as judge


class StrictRewardTests(unittest.IsolatedAsyncioTestCase):
    async def score(self, response, http_error=0, truncated=0):
        outcome = SimpleNamespace(response=response, http_error=http_error,
                                  truncated=truncated, prompt_tokens=10, completion_tokens=10)
        with patch.object(judge, "_get_session", AsyncMock(return_value=None)), \
             patch.object(judge, "_run_judge_call", AsyncMock(return_value=outcome)):
            return await strict_reward.compute_score(
                solution_str="<proof>A proof</proof>", extra_info={"theorem": "A problem"},
                judge_url="http://127.0.0.1:18791/v1")

    async def test_successful_scores_preserved(self):
        for points in (0, 1, 6, 7):
            result = await self.score(f"<points>{points} out of 7</points>")
            self.assertEqual(result["score"], points / 7)
            self.assertEqual(result["prover_judge_score"], int(points >= 6))

    async def test_failures_raise(self):
        for response, http, trunc in (("invalid", 0, 0), ("", 1, 0),
                                      ("<points>7</points>", 0, 1)):
            with self.subTest(http=http, trunc=trunc), self.assertRaises(RuntimeError):
                await self.score(response, http, trunc)

    async def test_no_proof_and_consumed_q_preserved_without_http(self):
        with patch.object(judge, "_get_session", AsyncMock(side_effect=AssertionError("HTTP forbidden"))):
            result = await strict_reward.compute_score(
                solution_str="unfinished thinking", extra_info={"theorem": "A problem"})
            self.assertEqual(result["score"], 0)
            for invalid in (0, 1):
                result = await strict_reward.compute_score(extra_info={
                    "sp_q_route_taken": "q_consumed", "sp_q_value": .3, "sp_q_invalid": invalid})
                self.assertEqual(result["score"], 0 if invalid else .3)
                self.assertEqual(result["sp_q_invalid"], invalid)


if __name__ == "__main__":
    unittest.main()
