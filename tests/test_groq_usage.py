"""request_json: optional usage return and reasoning_effort pass-through (fake client, no network)."""

import json
import unittest
from types import SimpleNamespace

from src.llm.groq_client import LLMConfigurationError, request_json

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["x"], "properties": {"x": {"type": "integer"}}}


class Client:
    def __init__(self, use="default"):
        self.calls = []
        self.use = use
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        use = self.use
        if use == "default":
            use = SimpleNamespace(prompt_tokens=50, completion_tokens=20, total_tokens=70,
                                  completion_tokens_details=SimpleNamespace(reasoning_tokens=12))
        msg = SimpleNamespace(content=json.dumps({"x": 1}), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=use)


def call(client, **kw):
    return request_json(client=client, model=kw.pop("model", "openai/gpt-oss-20b"), name="t", schema=SCHEMA,
                        system="s", payload={"q": 1}, **kw)


class UsageTests(unittest.TestCase):
    def test_default_return_is_unchanged(self):
        client = Client()
        self.assertEqual(call(client), {"x": 1})
        self.assertNotIn("reasoning_effort", client.calls[0])

    def test_return_usage(self):
        data, usage = call(Client(), return_usage=True)
        self.assertEqual(data, {"x": 1})
        self.assertEqual(usage, {"prompt_tokens": 50, "completion_tokens": 20, "reasoning_tokens": 12, "total_tokens": 70})

    def test_missing_or_odd_usage_becomes_none(self):
        _, usage = call(Client(use=None), return_usage=True)
        self.assertEqual(set(usage.values()), {None})
        odd = SimpleNamespace(prompt_tokens="50", completion_tokens=-1, total_tokens=True)
        _, usage = call(Client(use=odd), return_usage=True)
        self.assertEqual(set(usage.values()), {None})  # only plain non-negative ints are kept

    def test_reasoning_effort_for_gpt_oss_only(self):
        client = Client()
        call(client, reasoning_effort="low")
        self.assertEqual(client.calls[0]["reasoning_effort"], "low")
        client = Client()
        call(client, model="qwen/qwen3.8-27b", reasoning_effort="high")
        self.assertNotIn("reasoning_effort", client.calls[0])

    def test_invalid_reasoning_effort(self):
        with self.assertRaises(LLMConfigurationError) as ctx:
            call(Client(), reasoning_effort="max")
        self.assertEqual(ctx.exception.code, "invalid_reasoning_effort")


if __name__ == "__main__":
    unittest.main()
