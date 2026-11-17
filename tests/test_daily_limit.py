"""A daily token-limit 429 is not retried and stops the search; a parser 429 means no summary call at all (mocks only)."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

from app import demo_cache, pipeline, view_model as vm
from src.llm import groq_client
from src.schemas import ParsedQuery
from tests.test_app_pipeline import FakeClient, FakeRetriever, MODEL

DAILY = ("Rate limit reached for model `openai/gpt-oss-20b` in organization `org_PRIVATE` service tier `on_demand` on tokens per day "
         "(TPD): Limit 200000, Used 199245, Requested 1387. Please try again in 6m33.984s. Need more tokens? Upgrade to Dev Tier.")
MINUTE = ("Rate limit reached for model `openai/gpt-oss-20b` in organization `org_PRIVATE` service tier `on_demand` on tokens per minute "
          "(TPM): Limit 8000, Used 7900, Requested 1387. Please try again in 1.2s.")


def limit_error(text):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return openai.RateLimitError(text, response=httpx.Response(429, request=request), body={"message": text, "code": "rate_limit_exceeded"})


class Limited(FakeClient):
    """429 with `text` on the parser call and/or every summary call; counts the calls by schema name."""

    def __init__(self, text, parser=False, summary=True, **kw):
        super().__init__(location=None, **kw)
        self.text, self.limit_parser, self.limit_summary, self.counts = text, parser, summary, {"parsed_query": 0, "evidence_summary": 0}

    def create(self, **kwargs):
        name = kwargs["response_format"]["json_schema"]["name"]
        self.counts[name] += 1
        if (name == "parsed_query" and self.limit_parser) or (name == "evidence_summary" and self.limit_summary):
            raise limit_error(self.text)
        return super().create(**kwargs)


def search(client, **kw):
    return pipeline.run_search("vegan options", "Both", "semantic_capped", retriever=FakeRetriever(), client=pipeline.UsageRecorder(client),
                               model=MODEL, log_path=None, **kw)


class ClassificationTests(unittest.TestCase):
    def test_a_daily_limit_is_told_apart_from_a_per_minute_one_and_the_wait_is_kept_as_a_number(self):
        daily = groq_client._safe_api_error(limit_error(DAILY))
        self.assertEqual((daily.category, daily.code, daily.http_status), ("rate_limit", "daily_limit", 429))
        self.assertAlmostEqual(daily.retry_after, 6 * 60 + 33.984, places=2)
        minute = groq_client._safe_api_error(limit_error(MINUTE))
        self.assertEqual((minute.category, minute.code), ("rate_limit", "rate_limited"))
        self.assertAlmostEqual(minute.retry_after, 1.2, places=2)

    def test_no_provider_text_survives(self):
        for text in (DAILY, MINUTE):
            error = groq_client._safe_api_error(limit_error(text))
            self.assertNotIn("PRIVATE", repr(vars(error)) + str(error))
            self.assertNotIn("org_", repr(vars(error)) + str(error))

    def test_a_limit_message_without_a_wait_or_without_a_body(self):
        plain = groq_client._safe_api_error(limit_error("Rate limit reached (TPD)."))
        self.assertEqual((plain.code, plain.retry_after), ("daily_limit", None))
        none = groq_client._safe_api_error(openai.RateLimitError("private provider text", response=httpx.Response(
            429, request=httpx.Request("POST", "https://x")), body=None))
        self.assertEqual((none.code, none.retry_after), ("rate_limited", None))

    def test_wait_parts(self):
        for text, seconds in (("try again in 4m33.024s", 273.024), ("try again in 2h1m", 7260.0), ("try again in 577ms", 0.577),
                              ("try again in 45s", 45.0)):
            self.assertAlmostEqual(groq_client._rate_limit_detail(limit_error(f"TPD {text}"))[1], seconds, places=3)

    def test_error_info_carries_the_wait_and_a_daily_message(self):
        info = pipeline.error_info(groq_client._safe_api_error(limit_error(DAILY)))
        self.assertEqual((info["category"], info["code"]), ("rate_limit", "daily_limit"))
        self.assertEqual(info["message"], "The daily limit resets gradually; try again in about 7 minutes.")
        self.assertAlmostEqual(info["retry_seconds"], 393.984, places=2)

    def test_human_wait_rounds_up(self):
        self.assertEqual(pipeline.human_wait(20), "about a minute")
        self.assertEqual(pipeline.human_wait(60), "about a minute")
        self.assertEqual(pipeline.human_wait(61), "about 2 minutes")
        self.assertEqual(pipeline.human_wait(393.984), "about 7 minutes")
        self.assertEqual(pipeline.human_wait(3 * 3600), "about 3 hours")
        self.assertEqual(pipeline.daily_limit_message(None), "The daily limit resets gradually; try again later.")


class DailyLimitSearchTests(unittest.TestCase):
    def test_a_daily_429_is_not_retried_and_no_further_call_starts(self):
        client = Limited(DAILY)
        with patch.object(pipeline, "SUMMARY_WORKERS", 1):
            out, _ = search(client, auto_summaries=6)
        self.assertEqual(client.counts["evidence_summary"], 1)  # exactly one summary call, not 1 + 3 retries, not one per restaurant
        self.assertEqual(client.counts["parsed_query"], 1)
        failed = [v for v in out.results if v.summary_error]
        self.assertEqual([v.summary_error["code"] for v in failed], ["daily_limit"])
        self.assertTrue(all(v.summary_skipped for v in out.results if not v.summary_error))
        self.assertEqual(vm.live_unavailable(out), "daily limit")

    def test_with_two_workers_only_calls_already_in_flight_can_happen_and_none_is_retried(self):
        client = Limited(DAILY)
        out, _ = search(client, auto_summaries=6)
        self.assertLessEqual(client.counts["evidence_summary"], pipeline.SUMMARY_WORKERS)
        self.assertEqual(vm.live_unavailable(out), "daily limit")

    def test_the_daily_limit_uses_no_fallback_model(self):
        client = Limited(DAILY)
        with patch.dict("os.environ", {"GROQ_FALLBACK_MODEL": "openai/gpt-oss-120b"}), patch.object(pipeline, "SUMMARY_WORKERS", 1):
            search(client, auto_summaries=3)
        self.assertEqual(client.counts["evidence_summary"], 1)

    def test_the_wait_time_is_in_the_message(self):
        client = Limited(DAILY)
        out, _ = search(client, auto_summaries=3)
        reason, wait = vm.live_unavailable(out), vm.live_wait_seconds(out)
        self.assertAlmostEqual(wait, 393.984, places=2)
        self.assertEqual(vm.unavailable_message(reason, wait),
                         "Live search is unavailable right now (daily limit). Try one of the saved examples. "
                         "The daily limit resets gradually; try again in about 7 minutes.")
        self.assertEqual(vm.unavailable_message("daily limit"),
                         "Live search is unavailable right now (daily limit). Try one of the saved examples. "
                         "The daily limit resets gradually; try again later.")
        self.assertEqual(vm.unavailable_message("rate limit", 60),
                         "Live search is unavailable right now (rate limit). Try one of the saved examples.")  # per-minute: unchanged

    def test_a_daily_limit_on_one_summary_with_others_ok_still_shows_the_unavailable_page(self):
        a = pipeline.ResultView(1, "G1", "A", "Brooklyn", "", 4.0, 40, False, 1.0, [])
        b = pipeline.ResultView(2, "G2", "B", "Brooklyn", "", 4.0, 40, False, 1.0, [])
        b.summary = pipeline.Summary(pipeline.STATUS_INSUFFICIENT, [], [])
        a.summary_error = {"category": "rate_limit", "code": "daily_limit", "message": "m", "retry_seconds": 120.0}
        out = pipeline.SearchOutcome("r", "q", "Both", "semantic_capped", ParsedQuery(), ParsedQuery(), None, [a, b])
        self.assertEqual(vm.live_unavailable(out), "daily limit")
        self.assertEqual(vm.live_wait_seconds(out), 120.0)


class ParserLimitTests(unittest.TestCase):
    def test_a_parser_429_of_either_kind_makes_zero_summary_calls(self):
        for text in (DAILY, MINUTE):
            client = Limited(text, parser=True, summary=False)  # summaries would work, but must not be asked for
            out, _ = search(client, auto_summaries=6)
            self.assertEqual(client.counts["evidence_summary"], 0, text[:60])
            self.assertEqual(client.counts["parsed_query"], 1)  # and the parser is not retried either
            self.assertEqual(out.parse_error["category"], "rate_limit")
            self.assertTrue(vm.live_unavailable(out))
            self.assertTrue(all(v.summary_skipped for v in out.results))
            self.assertTrue(out.results)  # the retrieval itself still ran

    def test_other_parser_errors_still_search_with_the_exact_words(self):
        client = FakeClient(parser="bad", location=None)  # not a 429: unchanged behavior
        out, _ = search(client, auto_summaries=3)
        self.assertEqual(client.calls.count("evidence_summary"), 3)

    def test_the_parser_is_not_retried_with_the_fallback_model_on_a_daily_limit(self):
        client = Limited(DAILY, parser=True, summary=False)
        with patch.dict("os.environ", {"GROQ_FALLBACK_MODEL": "openai/gpt-oss-120b"}):
            search(client, auto_summaries=3)
        self.assertEqual(client.counts["parsed_query"], 1)


class PerMinuteLimitStillRetriesTests(unittest.TestCase):
    def test_a_per_minute_429_keeps_the_backoff(self):
        client = Limited(MINUTE)
        with patch.object(pipeline, "SUMMARY_WORKERS", 1):
            out, _ = search(client, auto_summaries=1)
        self.assertEqual(client.counts["evidence_summary"], 1 + len(pipeline.BACKOFF_SECONDS))
        self.assertEqual(out.results[0].summary_error["code"], "rate_limited")

    def test_a_per_minute_429_that_clears_recovers(self):
        class Clears(Limited):
            def create(self, **kwargs):
                if kwargs["response_format"]["json_schema"]["name"] == "evidence_summary" and self.counts["evidence_summary"] >= 2:
                    self.limit_summary = False
                return super().create(**kwargs)
        client = Clears(MINUTE)
        with patch.object(pipeline, "SUMMARY_WORKERS", 1):
            out, _ = search(client, auto_summaries=1)
        self.assertIsNotNone(out.results[0].summary)
        self.assertIsNone(vm.live_unavailable(out))


@unittest.skipIf(AppTest is None, "streamlit not installed")
class PageTests(unittest.TestCase):
    def run_page(self, client):
        import tempfile
        st.cache_resource.clear()
        st.cache_data.clear()
        self.addCleanup(st.cache_resource.clear)
        self.addCleanup(st.cache_data.clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        llm = (pipeline.UsageRecorder(client), MODEL, None)
        for obj, name, value in ((pipeline, "LOG_PATH", Path(tmp.name) / "log.jsonl"), (demo_cache, "CACHE_DIR", Path(tmp.name) / "c"),
                                 (pipeline, "load_retriever", lambda n: FakeRetriever()), (pipeline, "make_llm_client", lambda: llm),
                                 (pipeline, "data_dir_missing", lambda: False), (pipeline, "available_methods", lambda: list(pipeline.METHODS))):
            p = patch.object(obj, name, value)
            p.start()
            self.addCleanup(p.stop)
        at = AppTest.from_file(str(Path(pipeline.REPO) / "app" / "streamlit_app.py"), default_timeout=30)
        at.run()
        at.text_input[0].input("vegan options for a group")
        next(b for b in at.button if b.label == "Search").click()
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def test_the_page_shows_the_unavailable_message_with_the_wait_and_the_saved_examples(self):
        for limited in (dict(parser=False, summary=True), dict(parser=True, summary=False)):
            client = Limited(DAILY, **limited)
            at = self.run_page(client)
            text = " ".join(m.value for m in at.markdown).replace("&#x27;", "'")
            self.assertIn("Live search is unavailable right now (daily limit). Try one of the saved examples. "
                          "The daily limit resets gradually; try again in about 7 minutes.", text)
            self.assertNotIn("restaurants checked", text)
            self.assertIn("Peanut", [b.label for b in at.button])


if __name__ == "__main__":
    unittest.main()
