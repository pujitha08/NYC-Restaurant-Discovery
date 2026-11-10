"""OpenRouter provider path: request_json(provider="openrouter"), client config and the judge flag.

Fake clients and a fake openai module only: no network, no API key.
"""

import csv
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from eval import llm_judge, make_pool
from src.llm import openrouter_client
from src.llm.groq_client import LLMAPIError, LLMConfigurationError, LLMResponseError, _safe_api_error, request_json

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["x"], "properties": {"x": {"type": "integer"}}}


class Client:
    """OpenAI-compatible fake returning {"x": 1}; `served` sets OpenRouter's top-level provider field."""

    def __init__(self, served="Groq", content=None, error=None):
        self.calls, self.served, self.content, self.error = [], served, content, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        msg = SimpleNamespace(content=self.content or json.dumps({"x": 1}), refusal=None)
        use = SimpleNamespace(prompt_tokens=40, completion_tokens=10, total_tokens=50,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=4))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=use,
                               provider=self.served)


def call(client, **kw):
    return request_json(client=client, model=kw.pop("model", "openai/gpt-oss-120b"), name="t", schema=SCHEMA,
                        system="s", payload={"q": 1}, provider="openrouter", **kw)


class RequestTests(unittest.TestCase):
    def test_strict_schema_require_parameters_and_reasoning_mapping(self):
        client = Client()
        call(client, reasoning_effort="low")
        kw = client.calls[0]
        self.assertEqual(kw["response_format"]["type"], "json_schema")
        self.assertTrue(kw["response_format"]["json_schema"]["strict"])
        self.assertEqual(kw["response_format"]["json_schema"]["schema"], SCHEMA)
        self.assertEqual(kw["extra_body"], {"provider": {"require_parameters": True}, "reasoning": {"effort": "low"}})
        self.assertNotIn("reasoning_effort", kw)  # Groq-style top-level parameter is not used
        self.assertEqual(kw["model"], "openai/gpt-oss-120b")

    def test_no_reasoning_block_without_effort_or_for_other_models(self):
        client = Client()
        call(client)
        self.assertEqual(client.calls[0]["extra_body"], {"provider": {"require_parameters": True}})
        client = Client()
        call(client, model="mistralai/some-model", reasoning_effort="high")
        self.assertNotIn("reasoning", client.calls[0]["extra_body"])

    def test_groq_allow_list_does_not_apply(self):
        self.assertEqual(call(Client(), model="vendor/not-on-groq"), {"x": 1})

    def test_usage_and_served_by(self):
        data, usage = call(Client(served="DeepInfra"), return_usage=True)
        self.assertEqual(data, {"x": 1})
        self.assertEqual(usage, {"prompt_tokens": 40, "completion_tokens": 10, "reasoning_tokens": 4,
                                 "total_tokens": 50, "served_by": "DeepInfra"})
        for odd in (None, 42, "x" * 65, "Evil\nProvider", "<script>"):
            with self.subTest(odd=odd):
                self.assertIsNone(call(Client(served=odd), return_usage=True)[1]["served_by"])

    def test_same_strict_parsing_and_typed_errors(self):
        with self.assertRaises(LLMResponseError):
            call(Client(content='{"x": 1, "x": 2}'))  # duplicate key
        with self.assertRaises(LLMResponseError):
            call(Client(content="not json"))
        with self.assertRaises(LLMAPIError) as ctx:
            call(Client(error=TimeoutError()))
        self.assertEqual((ctx.exception.category, ctx.exception.code), ("timeout", "request_timeout"))

    def test_invalid_provider_and_effort(self):
        with self.assertRaises(LLMConfigurationError) as ctx:
            request_json(client=Client(), model="m", name="t", schema=SCHEMA, system="s", payload={}, provider="azure")
        self.assertEqual(ctx.exception.code, "invalid_provider")
        with self.assertRaises(LLMConfigurationError):
            call(Client(), reasoning_effort="max")

    def test_groq_default_is_unchanged(self):
        client = Client()
        request_json(client=client, model="openai/gpt-oss-20b", name="t", schema=SCHEMA, system="s", payload={})
        self.assertNotIn("extra_body", client.calls[0])
        data, usage = request_json(client=Client(), model="openai/gpt-oss-20b", name="t", schema=SCHEMA,
                                   system="s", payload={}, return_usage=True)
        self.assertNotIn("served_by", usage)


class ErrorNamingTests(unittest.TestCase):
    def test_messages_name_the_provider_and_402_is_billing(self):
        self.assertIn("OpenRouter", str(_safe_api_error(ConnectionError(), "OpenRouter")))
        self.assertIn("GroqCloud", str(_safe_api_error(ConnectionError())))
        import openai  # the HTTP status is only read from real SDK errors
        import httpx
        request = httpx.Request("POST", openrouter_client.BASE_URL + "/chat/completions")
        err = openai.APIStatusError("secret body", response=httpx.Response(402, request=request), body=None)
        mapped = _safe_api_error(err, "OpenRouter")
        self.assertEqual((mapped.category, mapped.code, mapped.http_status), ("billing", "payment_required", 402))
        self.assertNotIn("secret", str(mapped))


class ClientConfigTests(unittest.TestCase):
    def test_model_default_env_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(openrouter_client.resolve_model(), "openai/gpt-oss-120b")
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "vendor/model"}, clear=True):
            self.assertEqual(openrouter_client.resolve_model(), "vendor/model")
            self.assertEqual(openrouter_client.resolve_model("other/model"), "other/model")

    def test_missing_or_placeholder_key_is_rejected(self):
        for env in ({}, {"OPENROUTER_API_KEY": "your-openrouter-api-key-here"}, {"OPENROUTER_API_KEY": "a b"}):
            with self.subTest(env=list(env)), patch.dict(os.environ, env, clear=True):
                with self.assertRaises(LLMConfigurationError) as ctx:
                    openrouter_client.get_client()
                self.assertEqual(ctx.exception.code, "invalid_api_key_configuration")

    def test_client_points_at_openrouter_without_retries(self):
        sdk = SimpleNamespace(OpenAI=Mock(return_value="client"))
        with patch.dict(sys.modules, {"openai": sdk}), patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-sentinel"}):
            self.assertEqual(openrouter_client.get_client(), "client")
        kwargs = sdk.OpenAI.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual((kwargs["max_retries"], kwargs["timeout"]), (0, 30.0))
        self.assertNotIn("default_headers", kwargs)  # no attribution headers

    def test_client_init_failure_does_not_leak(self):
        sdk = SimpleNamespace(OpenAI=Mock(side_effect=RuntimeError("offline-sentinel leaked")))
        with patch.dict(sys.modules, {"openai": sdk}), patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-sentinel"}):
            with self.assertRaises(LLMConfigurationError) as ctx:
                openrouter_client.get_client()
        self.assertNotIn("offline-sentinel", str(ctx.exception))
        self.assertIsNone(ctx.exception.__cause__)


class JudgeProviderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        row = {c: "" for c in make_pool.COLUMNS}
        row.update(query_id="Q1", query="vegan", gmap_id="G1", name="R1", borough="Brooklyn",
                   evidence_status="retrieved", evidence_review_ids="11", evidence_1="[review 11, 2020-01] vegan menu")
        empty = {**row, "gmap_id": "G2", "evidence_status": "none", "evidence_review_ids": "", "evidence_1": ""}
        with open(self.dir / "pool.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=make_pool.COLUMNS)
            w.writeheader()
            w.writerows([row, empty])

    def run_judge(self, client, *extra):
        out = self.dir / "labels.jsonl"
        argv = ["--pool", str(self.dir / "pool.csv"), "--out", str(out), "--errors", str(self.dir / "err.jsonl"),
                "--rpm", "0", "--tpm", "0", *extra]
        with redirect_stdout(io.StringIO()) as stdout:
            llm_judge.main(argv, client=client)
        return [json.loads(l) for l in open(out)], stdout.getvalue()

    def answer(self):
        return Client(content=json.dumps({"label": 2, "reason": "Vegan menu.", "review_ids": [11]}), served="Groq")

    def test_openrouter_provider_is_used_and_recorded(self):
        client = self.answer()
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "openai/gpt-oss-120b"}):
            recs, out = self.run_judge(client, "--provider", "openrouter")
        self.assertIn("Provider: openrouter; model: openai/gpt-oss-120b", out)
        self.assertNotIn("Note: judging with", out)  # Groq allow-list note is Groq-only
        kw = client.calls[0]
        self.assertEqual(kw["extra_body"]["reasoning"], {"effort": "low"})
        by_gid = {r["gmap_id"]: r for r in recs}
        self.assertEqual({r["provider"] for r in recs}, {"openrouter"})  # rule record too
        self.assertEqual(by_gid["G1"]["usage"]["served_by"], "Groq")
        self.assertEqual(by_gid["G1"]["model"], "openai/gpt-oss-120b")
        self.assertEqual(by_gid["G2"]["source"], "rule")

    def test_groq_is_the_default_and_recorded(self):
        client = self.answer()
        recs, _ = self.run_judge(client, "--model", "openai/gpt-oss-20b")
        self.assertEqual({r["provider"] for r in recs}, {"groq"})
        self.assertEqual(client.calls[0]["reasoning_effort"], "low")
        self.assertNotIn("extra_body", client.calls[0])


if __name__ == "__main__":
    unittest.main()
