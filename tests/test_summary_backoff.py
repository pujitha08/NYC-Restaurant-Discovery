"""Live summaries: top 6 only, two at a time, rate-limited calls retried with exponential backoff (mocked; no Groq)."""

import json
import threading
import time
import unittest
from unittest.mock import patch

from app import pipeline
from tests.test_app_pipeline import FakeClient, FakeRetriever, META, MODEL, rate_limit_error


class Flaky(FakeClient):
    """Rate-limits the first `fail` summary calls (only those about restaurant `only`, if given), then answers."""

    def __init__(self, fail=2, only=None, **kw):
        super().__init__(**kw)
        self.fail, self.only, self.failed, self.attempts, self.lock = fail, only, 0, 0, threading.Lock()
        self.active = self.max_active = 0

    def create(self, **kwargs):
        if kwargs["response_format"]["json_schema"]["name"] == "evidence_summary":
            ids = [s["review_id"] for s in json.loads(kwargs["messages"][1]["content"])["snippets"]]
            with self.lock:
                self.active += 1
                self.attempts += 1
                self.max_active = max(self.max_active, self.active)
                limited = self.failed < self.fail and (self.only is None or self.only in ids)
                self.failed += limited
            time.sleep(0.02)
            with self.lock:
                self.active -= 1
            if limited:
                raise rate_limit_error()
        return super().create(**kwargs)


def many_gids(n):
    for i in range(7, n + 1):
        META[f"G{i}"] = {"name": f"Place {i}", "borough": "Brooklyn", "address": f"{i} Main St", "avg_rating": 4.1,
                         "num_clean_reviews": 40, "low_evidence": False}
    return tuple(f"G{i}" for i in range(1, n + 1) if i != 5)  # G5 is low-evidence


class BackoffTests(unittest.TestCase):
    def search(self, client, **kw):
        retriever = FakeRetriever(gids=many_gids(12))
        return pipeline.run_search("vegan options", "Both", "semantic_capped", retriever=retriever,
                                   client=pipeline.UsageRecorder(client), model=MODEL, log_path=None, **kw)

    def test_defaults(self):
        self.assertEqual(pipeline.AUTO_SUMMARIES, 6)
        self.assertEqual(pipeline.SUMMARY_WORKERS, 2)
        self.assertEqual(pipeline.BACKOFF_SECONDS[:1], (0.0,))  # tests run without real waiting (tests/__init__.py)

    def test_only_the_top_six_are_summarized_and_the_rest_are_marked(self):
        out, _ = self.search(FakeClient(location=None))
        self.assertEqual(len(out.results), 10)
        self.assertTrue(all(v.summary for v in out.results[:6]))
        for v in out.results[6:]:
            self.assertIsNone(v.summary)
            self.assertEqual(v.summary_skipped, "not requested")

    def test_two_summaries_at_a_time_at_most(self):
        client = Flaky(fail=0, location=None)
        self.search(client)
        self.assertEqual(client.max_active, 2)

    def test_a_rate_limit_is_retried_and_then_succeeds(self):
        client = Flaky(fail=3, location=None)
        out, _ = self.search(client)
        self.assertTrue(all(v.summary and not v.summary_error for v in out.results[:6]))
        self.assertEqual(client.attempts, 6 + 3)  # 3 rate-limited attempts, each retried

    def test_waits_double_between_attempts(self):
        waits = []
        view = pipeline.ResultView(1, "G1", "Place 1", "Brooklyn", "", 4.0, 40, False, 1.0, [])
        result = FakeRetriever().search("x")[0]
        client = pipeline.UsageRecorder(Flaky(fail=99, location=None))
        with patch.object(pipeline, "BACKOFF_SECONDS", (2.0, 4.0, 8.0)), patch.object(pipeline, "_backoff_delay", lambda d: d):
            error = pipeline.summarize_with_backoff(view, result, pipeline.ParsedQuery(), client, MODEL, sleep=waits.append)
        self.assertEqual(waits, [2.0, 4.0, 8.0])
        self.assertEqual(error["category"], "rate_limit")  # still limited after the last attempt
        self.assertEqual(view.summary_error["category"], "rate_limit")

    def test_other_errors_are_not_retried(self):
        waits = []
        view = pipeline.ResultView(1, "G1", "Place 1", "Brooklyn", "", 4.0, 40, False, 1.0, [])
        result = FakeRetriever().search("x")[0]
        client = pipeline.UsageRecorder(FakeClient(summary="bad", location=None))
        with patch.object(pipeline, "summarize", return_value={"category": "response", "code": "x", "message": "m"}) as call:
            error = pipeline.summarize_with_backoff(view, result, pipeline.ParsedQuery(), client, MODEL, sleep=waits.append)
        self.assertEqual((call.call_count, waits, error["category"]), (1, [], "response"))

    def test_jitter_stays_within_a_quarter(self):
        for _ in range(50):
            self.assertTrue(1.5 <= pipeline._backoff_delay(2.0) <= 2.5)

    def test_a_result_that_still_fails_keeps_an_error_and_the_others_still_run(self):
        client = Flaky(fail=99, only=101, location=None)  # restaurant G1 (review 101) never gets through
        out, _ = self.search(client)
        errors = [v for v in out.results[:6] if v.summary_error]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].summary_error["category"], "rate_limit")
        self.assertEqual(len([v for v in out.results[:6] if v.summary]), 5)


if __name__ == "__main__":
    unittest.main()
