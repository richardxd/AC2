import unittest

from transformers import AutoTokenizer

from vs_adapters import THINK_END, final_text, one_line, prompt_ids, replay_state

MODEL = "Qwen/Qwen3-4B-Thinking-2507"
REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
MESSAGES = [{"role": "user", "content": "Find all integers n such that n^2+1 divides n+3."}]
PREFIX = "Okay, let me think. First consider small values of n."


class TestPromptAndState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)

    def test_prompt_matches_native_thinking_template(self):
        ids = prompt_ids(self.tok, MESSAGES)
        native = self.tok.apply_chat_template(MESSAGES, add_generation_prompt=True, tokenize=True,
                                              enable_thinking=True, return_dict=False)
        self.assertEqual(ids, list(native))
        self.assertTrue(self.tok.decode(ids).endswith("<|im_start|>assistant\n<think>\n"))

    def test_replay_state_is_plain_token_concatenation(self):
        root = prompt_ids(self.tok, MESSAGES)
        gen = self.tok(PREFIX, add_special_tokens=False)["input_ids"]
        state = replay_state(root, gen, len(gen))
        self.assertEqual(state, root + gen)
        self.assertNotIn(THINK_END, self.tok.decode(state))

    def test_rendering_partial_assistant_message_inserts_empty_think(self):
        """Regression: the template path that must never build mid-reasoning states."""
        rendered = self.tok.apply_chat_template(MESSAGES + [{"role": "assistant", "content": PREFIX}],
                                                tokenize=False, continue_final_message=True)
        self.assertIn("<think>\n\n</think>\n\n" + PREFIX, rendered)


class TestExtract(unittest.TestCase):
    def test_reads_only_after_last_think_end(self):
        text = "PLAN: draft inside reasoning\n</think>\n\nPLAN: final plan\n"
        self.assertEqual(final_text(text), "PLAN: final plan")
        self.assertEqual(one_line(text), "PLAN: final plan")

    def test_unclosed_reasoning_raises(self):
        with self.assertRaisesRegex(ValueError, "no </think>"):
            one_line("PLAN: still thinking")

    def test_zero_or_two_lines_raise(self):
        for text in ["</think>\n\n", "</think>\nPLAN: a\nPLAN: b"]:
            with self.assertRaisesRegex(ValueError, "expected one line"):
                one_line(text)


if __name__ == "__main__":
    unittest.main()
