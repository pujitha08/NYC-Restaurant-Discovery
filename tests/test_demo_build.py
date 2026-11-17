"""Cache builds with a larger pool (--top-k): paced under a token budget per minute, stopped by a daily limit, nothing saved on failure
(fake retriever and fake model; no network)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import demo_cache, pipeline
from tests.test_app_pipeline import FakeClient, FakeRetriever, META, MODEL
from tests.test_daily_limit import DAILY, MINUTE, Limited

QUERY = "restaurant careful about peanut allergies"


def retriever(n):
    for i in range(7, n + 3):
        META[f"G{i}"] = {"name": f"Place {i}", "borough": "Brooklyn", "address": f"{i} Main St", "avg_rating": 4.1,
                         "num_clean_reviews": 40, "low_evidence": False}
    return FakeRetriever(gids=tuple(f"G{i}" for i in range(1, n + 3) if i != 5))  # G5 is low evidence


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def view(rank):
    return pipeline.ResultView(rank, f"G{rank}", f"Place {rank}", "Brooklyn", "", 4.0, 40, False, 1.0, [])


def outcome_of(n):
    return pipeline.SearchOutcome("r", QUERY, "Both", "semantic_capped", None, None, None, [view(i) for i in range(1, n + 1)], model=MODEL)


class PacingTests(unittest.TestCase):
    def run_paced(self, n=14, tokens=1500, budget=7000, daily_at=None):
        clock = FakeClock()
        client = pipeline.UsageRecorder(FakeClient())
        calls = []

        def fake(v, result, context, client_, model, guarded, user_request, sleep=None, deadline=None, clock_=None):
            calls.append((clock(), v.rank))
            client.calls.append({"call": "evidence_summary", "ok": True, "latency_s": 1.0, "usage": {"total_tokens": tokens}})
            clock.t += 1.0  # a call takes a second
            v.summary = pipeline.Summary(pipeline.STATUS_INSUFFICIENT, [], [])
            if daily_at and v.rank == daily_at:
                return {"category": "rate_limit", "code": "daily_limit", "message": "m"}
            return None
        with patch.object(pipeline, "summarize_with_backoff", fake):
            total, error = demo_cache.paced_summaries(outcome_of(n), [None] * n, client, budget=budget, clock=clock, sleep=clock.sleep)
        return clock, calls, total, error

    def test_no_sixty_second_window_exceeds_the_budget(self):
        clock, calls, total, error = self.run_paced()
        self.assertIsNone(error)
        self.assertEqual(total, 14 * 1500)
        for start, _ in calls:  # the calls that began inside any 60 s window
            in_window = [t for t, _ in calls if start <= t < start + 60]
            self.assertLessEqual(len(in_window) * 1500, 7000, (start, in_window))
        self.assertGreater(clock.t, 60)  # it had to wait: 14 calls cannot fit in one minute

    def test_a_smaller_pool_is_not_slowed_down(self):
        clock, calls, total, _ = self.run_paced(n=4)
        self.assertEqual(clock.t, 4.0)  # four calls (6,000 tokens) fit in one window: no sleep

    def test_the_first_daily_limit_stops_everything_at_once(self):
        clock, calls, total, error = self.run_paced(daily_at=3)
        self.assertEqual(error["code"], "daily_limit")
        self.assertEqual([r for _, r in calls], [1, 2, 3])  # nothing after the call that hit it
        self.assertEqual(total, 3 * 1500)

    def test_results_are_marked_as_asked_for(self):
        o = outcome_of(2)
        for v in o.results:
            v.summary_skipped = "not requested"
        client = pipeline.UsageRecorder(FakeClient())
        with patch.object(pipeline, "summarize_with_backoff", lambda v, *a, **k: None):
            demo_cache.paced_summaries(o, [None, None], client)
        self.assertTrue(all(v.summary_skipped is None for v in o.results))


class BuildTopKTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def build(self, client, n=8, **kw):
        recorder = pipeline.UsageRecorder(client)
        with patch.object(pipeline, "make_llm_client", lambda: (recorder, MODEL, None)), \
                patch.object(pipeline, "load_retriever", lambda name: retriever(n)):
            return demo_cache.build(self.dir, gap=0, sleep=lambda s: None, only=[QUERY], **kw)

    def test_the_default_build_is_unchanged_top_ten_top_six(self):
        seen = {}
        original = pipeline.run_search

        def spy(*a, **kw):
            seen.update(kw)
            return original(*a, **kw)
        with patch.object(pipeline, "run_search", spy):
            self.build(FakeClient(location=None), n=10)
        self.assertNotIn("top_k", seen)
        self.assertNotIn("auto_summaries", seen)
        cached = demo_cache.load_cached(QUERY, "Both", "semantic_capped", self.dir)
        self.assertEqual((len(cached.results), len([v for v in cached.results if v.summary])), (10, 6))

    def test_top_k_builds_a_larger_pool_with_every_restaurant_summarized(self):
        done = self.build(FakeClient(location=None), n=12, top_k=12)
        self.assertEqual([q for q, _ in done], [QUERY])
        self.assertGreater(done[0][1], 0)  # tokens were counted (parser and summaries)
        cached = demo_cache.load_cached(QUERY, "Both", "semantic_capped", self.dir)
        self.assertEqual(len(cached.results), 12)
        self.assertTrue(all(v.summary for v in cached.results))
        self.assertEqual(len({v.gmap_id for v in cached.results}), 12)
        self.assertEqual([p.name for p in self.dir.iterdir()], [demo_cache.cache_path(QUERY, "Both", "semantic_capped").name])  # only this one

    def test_only_builds_just_the_named_queries(self):
        with patch.object(pipeline, "make_llm_client", lambda: (pipeline.UsageRecorder(FakeClient(location=None)), MODEL, None)), \
                patch.object(pipeline, "load_retriever", lambda name: retriever(4)):
            demo_cache.build(self.dir, gap=0, sleep=lambda s: None, only=["  RESTAURANT careful about PEANUT allergies "], top_k=4)
        self.assertEqual(len(list(self.dir.iterdir())), 1)

    def test_a_daily_limit_stops_the_build_and_saves_nothing(self):
        client = Limited(DAILY)
        with self.assertRaises(SystemExit) as stop:
            self.build(client, n=12, top_k=12)
        message = str(stop.exception)
        self.assertIn("Daily token limit reached", message)
        self.assertIn("The daily limit resets gradually", message)
        self.assertIn("Nothing was saved", message)
        self.assertEqual(client.counts["evidence_summary"], 1)  # the first call hit it; nothing after
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_a_per_minute_429_is_retried_once_per_call_and_the_normal_backoff_is_restored(self):
        client = Limited(MINUTE)
        before = pipeline.BACKOFF_SECONDS
        with self.assertRaises(SystemExit):  # every call stays limited, so the result is incomplete and not saved
            self.build(client, n=3, top_k=3)
        self.assertEqual(client.counts["evidence_summary"], 3 * 2)  # one try and one retry for each of the 3 restaurants
        self.assertEqual(pipeline.BACKOFF_SECONDS, before)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_a_parser_failure_saves_nothing(self):
        with self.assertRaises(SystemExit):
            self.build(FakeClient(parser="bad", location=None), n=4, top_k=4)
        self.assertEqual(list(self.dir.iterdir()), [])


class CommandLineTests(unittest.TestCase):
    def test_top_k_and_only_are_passed_to_build(self):
        with patch.object(demo_cache, "build") as build:
            demo_cache.main(["--build", "--top-k", "30", "--only", "a query", "--only", "another"])
        build.assert_called_once_with(only_missing=False, top_k=30, only=["a query", "another"])

    def test_defaults_are_unchanged(self):
        with patch.object(demo_cache, "build") as build:
            demo_cache.main(["--build"])
        build.assert_called_once_with(only_missing=False, top_k=None, only=None)
        with patch.object(demo_cache, "build") as build:
            demo_cache.main(["--build", "--missing"])
        build.assert_called_once_with(only_missing=True, top_k=None, only=None)


if __name__ == "__main__":
    unittest.main()
