"""A bad request or an unparseable answer for one restaurant is retried once with shortened evidence, then becomes State A.

Mocked calls only: no Groq. The failure that motivated it: Groq answered 400 output_parse_failed because gpt-oss-20b wrote prose
("We have excerpts all mention kosher...") instead of the JSON for one kosher restaurant, on some runs and not on others.
"""

import json
import unittest
import unittest.mock
from types import SimpleNamespace

import httpx
import openai

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

from app import pipeline
from src.schemas import ParsedQuery, RetrievalResult, Snippet
from tests.test_app_pipeline import FakeClient, FakeRetriever, MODEL

LONG = " ".join(f"word{i}" for i in range(120))


def parse_failed():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"code": "output_parse_failed", "message": "private provider text", "failed_generation": "We have excerpts..."}
    return openai.BadRequestError("private provider text", response=httpx.Response(400, request=request), body=body)


class Scripted(FakeClient):
    """Summary calls: fail (as scripted by `plan`, one entry per call) or answer as FakeClient does. Records the payloads."""

    def __init__(self, plan, **kw):
        super().__init__(**kw)
        self.plan, self.payloads = list(plan), []

    def create(self, **kwargs):
        if kwargs["response_format"]["json_schema"]["name"] == "evidence_summary":
            self.payloads.append(json.loads(kwargs["messages"][1]["content"]))
            step = self.plan.pop(0) if self.plan else "ok"
            if step == "bad_request":
                raise parse_failed()
            if step == "prose":
                return self.response("We have excerpts all mention kosher. No contradictions. Provide support claims.")
        return super().create(**kwargs)


def result_with(*texts):
    return RetrievalResult("G1", 1.0, "m", [Snippet(t, 100 + i, 1546300800000, 5) for i, t in enumerate(texts)])


def view():
    return pipeline.ResultView(1, "G1", "Place 1", "Brooklyn", "", 4.0, 40, False, 1.0,
                               [pipeline.Excerpt(100 + i, "2019-01", t) for i, t in enumerate(["Great vegan food", "Vegan again", "Third"])])


def run(client, result):
    v = view()
    error = pipeline.summarize(v, result, ParsedQuery(dietary_restrictions=["vegan"]), pipeline.UsageRecorder(client), MODEL,
                               False, "vegan options")
    return v, error


class ShortenTests(unittest.TestCase):
    def test_sanitize_removes_control_characters_and_collapses_whitespace(self):
        self.assertEqual(pipeline.sanitize_text("a\x00b​c\n\n d\te\x07"), "a b c d e")
        self.assertEqual(pipeline.sanitize_text("Kosher כשר café ★"), "Kosher כשר café ★")  # text is kept

    def test_shortened_keeps_two_excerpts_cut_to_forty_words(self):
        short = pipeline.shortened(result_with("one", LONG, "three", "four"))
        self.assertEqual([s.review_id for s in short.evidence_snippets], [100, 101])
        self.assertEqual(len(short.evidence_snippets[1].text.split()), pipeline.SHORT_WORDS + 1)  # 40 words and the ellipsis
        self.assertTrue(short.evidence_snippets[1].text.endswith("…"))
        self.assertEqual(short.evidence_snippets[0].text, "one")

    def test_empty_after_sanitizing_is_dropped_and_the_original_is_not_changed(self):
        original = result_with("\x00\x07", "ok text")
        short = pipeline.shortened(original)
        self.assertEqual([s.text for s in short.evidence_snippets], ["ok text"])
        self.assertEqual(original.evidence_snippets[0].text, "\x00\x07")


class RetryTests(unittest.TestCase):
    def test_a_bad_request_is_retried_once_with_shorter_sanitized_evidence_and_then_succeeds(self):
        client = Scripted(["bad_request"], location=None)
        v, error = run(client, result_with("Great vegan\x00 food " + LONG, "Vegan again", "Third", "Fourth"))
        self.assertIsNone(error)
        self.assertIsNotNone(v.summary)
        self.assertEqual(len(client.payloads), 2)
        first, second = client.payloads
        self.assertEqual(len(first["snippets"]), 4)
        self.assertEqual(len(second["snippets"]), pipeline.SHORT_EXCERPTS)
        self.assertLess(len(second["snippets"][0]["text"]), len(first["snippets"][0]["text"]))
        self.assertNotIn("\x00", second["snippets"][0]["text"])
        self.assertEqual(v.summary.claims[0].excerpts[0].review_id, 100)  # the card still quotes the full retrieved review

    def test_an_unparseable_answer_is_retried_too(self):
        client = Scripted(["prose"], location=None)
        v, error = run(client, result_with("Great vegan food", "Vegan again"))
        self.assertIsNone(error)
        self.assertEqual(len(client.payloads), 2)

    def test_if_the_retry_also_fails_the_error_stays_and_it_is_tried_only_twice(self):
        client = Scripted(["bad_request", "bad_request"], location=None)
        v, error = run(client, result_with("Great vegan food", "Vegan again", "Third"))
        self.assertEqual((error["category"], error["code"]), ("bad_request", "output_parse_failed"))
        self.assertIsNone(v.summary)
        self.assertEqual(v.summary_error, error)
        self.assertNotIn("private provider text", json.dumps(error))
        self.assertEqual(len(client.payloads), 2)

    def test_other_errors_are_not_retried_this_way(self):
        for category in ("rate_limit", "authentication", "provider", "timeout"):
            with self.subTest(category=category):
                calls = []

                def once(view_, *a, **kw):
                    calls.append(1)
                    view_.summary_error = {"category": category, "code": "x", "message": "m"}
                    return view_.summary_error
                with unittest.mock.patch.object(pipeline, "_summarize_once", once):
                    pipeline.summarize(view(), result_with("a", "b"), ParsedQuery(), None, MODEL)
                self.assertEqual(len(calls), 1)

    def test_a_result_with_nothing_left_to_shorten_is_not_retried(self):
        client = Scripted(["bad_request"], location=None)
        v, error = run(client, result_with("\x00"))  # the only excerpt is control characters
        self.assertEqual(len(client.payloads), 1)
        self.assertEqual(error["category"], "bad_request")

    def test_one_failing_restaurant_does_not_fail_the_query(self):
        client = Scripted(["bad_request", "bad_request"], location=None)  # the first two calls (restaurant 1 and its retry) fail
        out, _ = pipeline.run_search("vegan options for a group", "Both", "semantic_capped", retriever=FakeRetriever(),
                                     client=pipeline.UsageRecorder(client), model=MODEL, log_path=None, auto_summaries=3)
        failed = [v for v in out.results if v.summary_error]
        done = [v for v in out.results if v.summary]
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(done), 2)
        self.assertEqual(out.parse_error, None)


@unittest.skipIf(AppTest is None, "streamlit not installed")
class StateATests(unittest.TestCase):
    def test_the_failing_card_is_state_a_and_the_others_are_normal(self):
        from pathlib import Path
        from unittest.mock import patch
        from app import demo_cache
        import tempfile
        st.cache_resource.clear()
        st.cache_data.clear()
        self.addCleanup(st.cache_resource.clear)
        self.addCleanup(st.cache_data.clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        client = Scripted(["bad_request", "bad_request"], location=None)
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
        html = " ".join(m.value for m in at.markdown)
        self.assertEqual(html.count("The AI summary isn&#x27;t available right now."), 1)
        self.assertEqual(html.count('class="care-card"'), 3)
        self.assertEqual([b.label for b in at.button].count("Try the summary again"), 1)
        self.assertEqual(len(at.error), 0)


if __name__ == "__main__":
    unittest.main()
