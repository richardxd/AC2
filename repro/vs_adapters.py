"""ac2-side adapters that plug a tokenizer and vLLM into samplers.vs.

Prompts are built exactly as verl's agent loop builds them. Replay states are formed at
the token level (root prompt ids + generated ids), never by re-rendering a partial
assistant message: Qwen3-Thinking's chat template inserts an empty <think></think> block
before an assistant prefix whose reasoning has not closed.
"""
import uuid

from verl.utils.chat_template import apply_chat_template
from verl.utils.tokenizer import normalize_token_ids

THINK_END = "</think>"


def prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Ids that verl's agent loop sends the engine for a fresh assistant turn."""
    return normalize_token_ids(
        apply_chat_template(tokenizer, messages, tools=None, add_generation_prompt=True, tokenize=True))


def replay_state(root_ids: list[int], response_ids: list[int], cut: int) -> list[int]:
    """Replay state: the root prompt followed by the first `cut` generated tokens."""
    assert 0 < cut <= len(response_ids)
    return root_ids + response_ids[:cut]


def final_text(text: str) -> str:
    """Model output after its last </think>; raises if the reasoning never closed."""
    _, sep, tail = text.rpartition(THINK_END)
    if not sep:
        raise ValueError("output has no </think>")
    return tail.strip()


def one_line(text: str) -> str:
    """The single non-empty line after </think> (one plan per call); raises otherwise."""
    lines = [line.strip() for line in final_text(text).splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected one line after </think>, got {len(lines)}")
    return lines[0]


def make_engine(model: str, **engine_kwargs):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    return AsyncLLM.from_engine_args(AsyncEngineArgs(model=model, **engine_kwargs))


class VLLMSampler:
    """Token-in/token-out sampler over vLLM AsyncLLM, called as verl's vllm_async_server calls it."""

    def __init__(self, engine, params: dict):
        self.engine, self.params = engine, params

    async def __call__(self, ids: list[int]):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        final = None
        async for out in self.engine.generate(prompt=TokensPrompt(prompt_token_ids=ids),
                                              sampling_params=SamplingParams(**self.params),
                                              request_id=uuid.uuid4().hex):
            final = out
        assert final is not None and len(final.outputs) == 1
        return final.outputs[0]
