"""A failing live search gives up quickly, shows progress, and can use an optional fallback model (mocks only; no network)."""

import json
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import pipeline
from src.schemas import ParsedQuery
from tests.test_app_pipeline import FakeClient, FakeRetriever, MODEL, rate_limit_error


class ByModel(FakeClient):
    """429 for every model except `ok_model`; records the model of each call."""

    def __init__(self, ok_model=None, **kw):
        super().__init__(**kw)
        self.ok_model, self.models = ok_model, []

    def create(self, **kwargs):
        self.models.append(kwargs["model"])
        if kwargs["model"] != self.ok_model:
            raise rate_limit_error()
        return super().create(**kwargs)


def view():
    ex = [pipeline.Excerpt(100 + i, "2019-01", f"Great vegan food {i}", 5) for i in range(3)]
    return pipeline.ResultView(1, "G1", "Place 1", "Brooklyn", "", 4.0, 40, False, 1.0, ex)


def result():
    return FakeRetriever().search("x")[0]


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class DeadlineTests(unittest.TestCase):
    def test_the_limits_are_about_fifteen_seconds(self):
        self.assertLessEqual(pipeline.SUMMARY_DEADLINE_S, 15)
        self.assertLessEqual(pipeline.CALL_TIMEOUT_S, 12)
        self.assertLessEqual(sum(pipeline.BACKOFF_SECONDS), 8)

    def test_a_retry_that_would_start_after_the_deadline_is_not_made(self):
        clock = FakeClock()
        client = pipeline.UsageRecorder(ByModel(location=None))
        with patch.object(pipeline, "BACKOFF_SECONDS", (1.0, 2.0, 4.0)), patch.object(pipeline, "_backoff_delay", lambda d: d):
            error = pipeline.summarize_with_backoff(view(), result(), ParsedQuery(), client, MODEL, sleep=clock.sleep,
                                                    deadline=3.5, clock=clock)
        self.assertEqual(error["category"], "rate_limit")
        self.assertEqual(clock.t, 3.0)  # waited 1 s and 2 s; the 4 s wait would pass the deadline
        self.assertEqual(len(client.take()), 3)

    def test_results_not_started_by_the_deadline_get_a_timeout_error_and_no_call(self):
        client = pipeline.UsageRecorder(FakeClient(location=None))
        out = pipeline.SearchOutcome("r", "vegan", "Both", "semantic_capped", ParsedQuery(), ParsedQuery(), None,
                                     [view()], model=MODEL)
        pipeline.summarize_results(out, [result()], client, 6, deadline_s=0.0)
        self.assertEqual(out.results[0].summary_error["category"], "timeout")
        self.assertIsNone(out.results[0].summary)
        self.assertEqual(client._client.calls, [])

    def test_a_search_where_every_call_is_rate_limited_ends_with_errors_and_not_a_hang(self):
        client = pipeline.UsageRecorder(ByModel(location=None))
        out, _ = pipeline.run_search("vegan options", "Both", "semantic_capped", retriever=FakeRetriever(), client=client,
                                     model=MODEL, log_path=None, auto_summaries=3)
        self.assertTrue(all(v.summary_error for v in out.results[:3]))
        self.assertTrue(all(v.summary_error["category"] in ("rate_limit", "timeout") for v in out.results[:3]))


class ProgressTests(unittest.TestCase):
    def test_progress_is_reported_on_the_calling_thread_from_zero_to_total(self):
        calls, threads = [], set()

        def progress(done, total):
            calls.append((done, total))
            threads.add(threading.get_ident())
        client = pipeline.UsageRecorder(FakeClient(location=None))
        out, _ = pipeline.run_search("vegan options", "Both", "semantic_capped", retriever=FakeRetriever(), client=client,
                                     model=MODEL, log_path=None, auto_summaries=3, on_summary_progress=progress)
        n = len([v for v in out.results if v.rank <= 3])
        self.assertEqual(calls, [(i, n) for i in range(n + 1)])
        self.assertEqual(threads, {threading.get_ident()})

    def test_no_progress_when_nothing_is_summarized(self):
        calls = []
        client = pipeline.UsageRecorder(FakeClient(location=None))
        pipeline.run_search("vegan options", "Both", "semantic_capped", retriever=FakeRetriever(), client=client, model=MODEL,
                            log_path=None, auto_summaries=0, on_summary_progress=lambda *a: calls.append(a))
        self.assertEqual(calls, [])


class TimeoutTests(unittest.TestCase):
    def test_the_client_is_made_with_the_per_call_timeout(self):
        seen = {}
        with patch.dict(os.environ, {"GROQ_MODEL": MODEL, "GROQ_API_KEY": "test-key-not-real"}), \
                patch.object(pipeline, "load_env"), \
                patch("src.llm.groq_client.get_client", lambda timeout=30.0: seen.setdefault("timeout", timeout) or object()):
            pipeline.make_llm_client()
        self.assertEqual(seen["timeout"], pipeline.CALL_TIMEOUT_S)

    def test_get_client_passes_the_timeout_to_the_sdk(self):
        captured = []
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key-not-real"}), \
                patch("openai.OpenAI", lambda **kw: captured.append(kw) or object()):
            from src.llm import groq_client
            groq_client.get_client()
            groq_client.get_client(timeout=7)
        self.assertEqual([c["timeout"] for c in captured], [30.0, 7])
        self.assertTrue(all(c["max_retries"] == 0 for c in captured))


class FallbackModelTests(unittest.TestCase):
    def test_off_by_default_and_an_unsupported_name_is_ignored(self):
        self.assertIsNone(pipeline.fallback_model({}))
        self.assertIsNone(pipeline.fallback_model({"GROQ_FALLBACK_MODEL": "  "}))
        self.assertIsNone(pipeline.fallback_model({"GROQ_FALLBACK_MODEL": "some/unsupported-model"}))
        self.assertEqual(pipeline.fallback_model({"GROQ_FALLBACK_MODEL": " openai/gpt-oss-120b "}), "openai/gpt-oss-120b")

    def test_without_the_env_var_there_is_no_second_model(self):
        client = ByModel(ok_model="openai/gpt-oss-120b", location=None)
        v = view()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_FALLBACK_MODEL", None)
            error = pipeline.summarize_with_backoff(v, result(), ParsedQuery(), pipeline.UsageRecorder(client), MODEL)
        self.assertEqual(error["category"], "rate_limit")
        self.assertEqual(set(client.models), {MODEL})

    def test_with_the_env_var_the_fallback_model_gets_one_try_after_the_main_model_failed(self):
        client = ByModel(ok_model="openai/gpt-oss-120b", location=None)
        v = view()
        with patch.dict(os.environ, {"GROQ_FALLBACK_MODEL": "openai/gpt-oss-120b"}):
            error = pipeline.summarize_with_backoff(v, result(), ParsedQuery(dietary_restrictions=["vegan"]),
                                                    pipeline.UsageRecorder(client), MODEL)
        self.assertIsNone(error)
        self.assertIsNotNone(v.summary)
        self.assertEqual(client.models[-1], "openai/gpt-oss-120b")
        self.assertEqual(client.models.count("openai/gpt-oss-120b"), 1)
        self.assertEqual(set(client.models[:-1]), {MODEL})

    def test_other_errors_do_not_use_the_fallback(self):
        client = FakeClient(summary="ok", location=None)
        seen = []
        with patch.dict(os.environ, {"GROQ_FALLBACK_MODEL": "openai/gpt-oss-120b"}), \
                patch.object(pipeline, "summarize", lambda *a, **k: seen.append(a[4]) or {"category": "response", "code": "x", "message": "m"}):
            pipeline.summarize_with_backoff(view(), result(), ParsedQuery(), client, MODEL)
        self.assertEqual(seen, [MODEL])

    def test_the_parser_uses_the_fallback_model_after_a_rate_limit(self):
        client = ByModel(ok_model="openai/gpt-oss-120b", location=None)
        with patch.dict(os.environ, {"GROQ_FALLBACK_MODEL": "openai/gpt-oss-120b"}):
            parsed, error = pipeline.parse("vegan options", pipeline.UsageRecorder(client), MODEL)
        self.assertIsNone(error)
        self.assertEqual(parsed.dietary_restrictions, ["vegan"])
        self.assertEqual(client.models, [MODEL, "openai/gpt-oss-120b"])

    def test_the_parser_error_is_kept_when_there_is_no_fallback(self):
        client = ByModel(location=None)
        os.environ.pop("GROQ_FALLBACK_MODEL", None)
        parsed, error = pipeline.parse("vegan options", pipeline.UsageRecorder(client), MODEL)
        self.assertIsNone(parsed)
        self.assertEqual(error["category"], "rate_limit")
        self.assertEqual(client.models, [MODEL])


if __name__ == "__main__":
    unittest.main()
