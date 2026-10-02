"""Q queries bypass proof grading; ordinary proof outputs still reach the reward loop."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, AgentLoopWorker
from verl.experimental.agent_loop.sp_q_agent_loop import SPQAgentLoop


class RewardRoutingTests(unittest.IsolatedAsyncioTestCase):
    def worker(self):
        remote = AsyncMock(return_value={"reward_score": .5, "reward_extra_info": {"acc": .5}})
        worker = SimpleNamespace(
            reward_loop_worker_handles=[SimpleNamespace(compute_score=SimpleNamespace(remote=remote))],
            _compute_multi_modal_inputs=lambda *args: {},
            _compute_position_ids=lambda ids, *args: torch.zeros_like(ids),
            _get_mm_processor_kwargs=lambda *args: None,
        )
        return worker, remote

    async def test_q_generation_remains_parse_only(self):
        agent = object.__new__(SPQAgentLoop)
        agent.response_length = 256
        agent._get_mm_processor_kwargs = lambda *args: None
        agent.server_manager = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(
            token_ids=[10, 11], num_preempted=0, extra_fields={})))
        output = await SPQAgentLoop.run.__wrapped__(agent, {}, sp_q_ctx_token_ids=[1, 2, 3], sp_q_max_tokens=4)
        self.assertEqual(output.response_ids, [10, 11])
        self.assertEqual(output.reward_score, 0.)
        self.assertEqual(agent.server_manager.generate.call_args.kwargs["prompt_ids"], [1, 2, 3])
        worker, remote = self.worker()
        # No data_source/reward_model: a Q-wave row intentionally has neither.
        await AgentLoopWorker._compute_score(worker, [output], kwargs={})
        remote.assert_not_awaited()

    async def test_proof_output_still_graded(self):
        output = AgentLoopOutput(prompt_ids=[1], response_ids=[2], response_mask=[1], num_turns=2, metrics={})
        worker, remote = self.worker()
        await AgentLoopWorker._compute_score(worker, [output], kwargs={
            "data_source": "fineproofs-rl", "reward_model": {"ground_truth": ""}})
        remote.assert_awaited_once()
        self.assertEqual(output.reward_score, .5)
        self.assertEqual(output.extra_fields["reward_extra_info"]["acc"], .5)


if __name__ == "__main__":
    unittest.main()
