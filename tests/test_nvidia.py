"""NVIDIA provider path: request_json(provider="nvidia"), client config and the judge flag (no network, no key)."""

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
from src.llm import nvidia_client
from src.llm.groq_client import LLMAPIError, LLMConfigurationError, _safe_api_error, request_json

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["x"], "properties": {"x": {"type": "integer"}}}


class Client:
    def __init__(self, content=None, error=None):
        self.calls, self.content, self.error = [], content, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        msg = SimpleNamespace(content=self.content or json.dumps({"x": 1}), refusal=None)
        use = SimpleNamespace(prompt_tokens=30, completion_tokens=8, total_tokens=38,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=3))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=use)


def call(client, **kw):
    return request_json(client=client, model=kw.pop("model", "openai/gpt-oss-120b"), name="t", schema=SCHEMA,
                        system="s", payload={"q": 1}, provider="nvidia", **kw)


class RequestTests(unittest.TestCase):
    def test_strict_schema_and_top_level_reasoning_effort(self):
        client = Client()
        data, usage = call(client, reasoning_effort="low", return_usage=True)
        kw = client.calls[0]
        self.assertEqual(data, {"x": 1})
        self.assertEqual(kw["model"], "openai/gpt-oss-120b")
        self.assertEqual(kw["reasoning_effort"], "low")  # top-level, per NVIDIA's API reference
        self.assertNotIn("extra_body", kw)
        self.assertEqual(kw["response_format"]["type"], "json_schema")
        self.assertTrue(kw["response_format"]["json_schema"]["strict"])
        self.assertEqual(usage, {"prompt_tokens": 30, "completion_tokens": 8, "reasoning_tokens": 3, "total_tokens": 38})

    def test_no_effort_for_non_gpt_oss_and_no_groq_allow_list(self):
        client = Client()
        call(client, model="meta/llama-3.3-70b-instruct", reasoning_effort="high")
        self.assertNotIn("reasoning_effort", client.calls[0])

    def test_upstream_pin_is_openrouter_only(self):
        with self.assertRaises(LLMConfigurationError):
            call(Client(), openrouter_upstream="crusoe")

    def test_errors_name_nvidia(self):
        with self.assertRaises(LLMAPIError) as ctx:
            call(Client(error=TimeoutError()))
        self.assertEqual(ctx.exception.code, "request_timeout")
        self.assertIn("NVIDIA", str(_safe_api_error(ConnectionError(), "NVIDIA")))


class ClientConfigTests(unittest.TestCase):
    def test_model_default_and_env(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(nvidia_client.resolve_model(), "openai/gpt-oss-120b")
        with patch.dict(os.environ, {"NVIDIA_MODEL": "vendor/other"}, clear=True):
            self.assertEqual(nvidia_client.resolve_model(), "vendor/other")

    def test_key_checks_and_client_settings(self):
        for env in ({}, {"NVIDIA_API_KEY": "your-nvidia-api-key-here"}, {"NVIDIA_API_KEY": "a b"}):
            with self.subTest(env=list(env)), patch.dict(os.environ, env, clear=True):
                with self.assertRaises(LLMConfigurationError) as ctx:
                    nvidia_client.get_client()
                self.assertEqual(ctx.exception.code, "invalid_api_key_configuration")
        sdk = SimpleNamespace(OpenAI=Mock(return_value="client"))
        with patch.dict(sys.modules, {"openai": sdk}), patch.dict(os.environ, {"NVIDIA_API_KEY": "offline-sentinel"}):
            self.assertEqual(nvidia_client.get_client(), "client")
        kw = sdk.OpenAI.call_args.kwargs
        self.assertEqual(kw["base_url"], "https://integrate.api.nvidia.com/v1")
        self.assertEqual((kw["max_retries"], kw["timeout"]), (0, 60.0))

    def test_init_failure_does_not_leak(self):
        sdk = SimpleNamespace(OpenAI=Mock(side_effect=RuntimeError("offline-sentinel leaked")))
        with patch.dict(sys.modules, {"openai": sdk}), patch.dict(os.environ, {"NVIDIA_API_KEY": "offline-sentinel"}):
            with self.assertRaises(LLMConfigurationError) as ctx:
                nvidia_client.get_client()
        self.assertNotIn("offline-sentinel", str(ctx.exception))


class JudgeTests(unittest.TestCase):
    def test_provider_recorded_and_pacing_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            row = {c: "" for c in make_pool.COLUMNS}
            row.update(query_id="Q1", query="vegan", gmap_id="G1", name="R1", borough="Brooklyn",
                       evidence_status="retrieved", evidence_review_ids="11", evidence_1="[review 11, 2020-01] vegan menu")
            empty = {**row, "gmap_id": "G2", "evidence_status": "none", "evidence_review_ids": "", "evidence_1": ""}
            with open(d / "pool.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=make_pool.COLUMNS)
                w.writeheader()
                w.writerows([row, empty])
            client = Client(content=json.dumps({"label": 2, "reason": "Vegan menu.", "review_ids": [11]}))
            paced = []
            real = llm_judge.Pacer
            with patch.object(llm_judge, "Pacer", side_effect=lambda rpm, tpm, **k: paced.append((rpm, tpm)) or real(0, 0)), \
                    redirect_stdout(io.StringIO()) as out:
                llm_judge.main(["--pool", str(d / "pool.csv"), "--out", str(d / "l.jsonl"), "--errors", str(d / "e.jsonl"),
                                "--provider", "nvidia"], client=client)
            recs = [json.loads(l) for l in open(d / "l.jsonl")]
        self.assertEqual({r["provider"] for r in recs}, {"nvidia"})
        self.assertEqual({r["openrouter_upstream"] for r in recs}, {None})
        self.assertEqual(paced, [(30, 60_000)])
        self.assertEqual(client.calls[0]["reasoning_effort"], "low")
        self.assertIn("Provider: nvidia; model: openai/gpt-oss-120b", out.getvalue())
        self.assertNotIn("served_by", next(r for r in recs if r["source"] == "llm")["usage"])


if __name__ == "__main__":
    unittest.main()
