"""Cached demo results: round trip, exact matching, demo mode and the rate-limit fallback (fake retriever and LLM)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import demo_cache, pipeline
from tests.test_app_pipeline import FakeClient, FakeRetriever

MODEL = "openai/gpt-oss-20b"
QUERY, BOROUGH, METHOD = "vegan options for a group", "Both", "semantic_capped"


class _CacheFixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def live(self, client=None, **kwargs):
        client = client or pipeline.UsageRecorder(FakeClient())
        return dict(retriever=FakeRetriever(), client=client, model=MODEL, log_path=None, cache_dir=self.dir, **kwargs)

    def saved(self, query=QUERY, **client_kwargs):
        outcome, _ = pipeline.run_search(query, BOROUGH, METHOD, retriever=FakeRetriever(),
                                         client=pipeline.UsageRecorder(FakeClient(**client_kwargs)), model=MODEL,
                                         log_path=None, auto_summaries=3)
        demo_cache.save_outcome(outcome, self.dir)
        return outcome


class DemoCacheTests(_CacheFixture):
    def test_round_trip_keeps_results_excerpts_and_summaries(self):
        original = self.saved()
        cached = demo_cache.load_cached("  Vegan  options for a GROUP ", BOROUGH, METHOD, self.dir)
        self.assertTrue(cached.cached)
        self.assertEqual(cached.parsed, original.parsed)
        self.assertEqual([v.name for v in cached.results], [v.name for v in original.results])
        first, orig = cached.results[0], original.results[0]
        self.assertEqual(first.excerpts, orig.excerpts)
        self.assertEqual(first.summary, orig.summary)
        self.assertIn("cached demo result", cached.notices[0])

    def test_saved_file_has_no_run_specific_data(self):
        self.saved()
        record = json.loads(next(self.dir.glob("*.json")).read_text())
        self.assertEqual(record["outcome"]["llm_calls"], [])
        self.assertEqual(record["outcome"]["timings"], {})
        self.assertEqual(record["outcome"]["request_id"], "")

    def test_unsummarized_results_cannot_be_explained(self):
        self.saved()
        cached = demo_cache.load_cached(QUERY, BOROUGH, METHOD, self.dir)
        self.assertIsNone(cached.results[3].summary)
        self.assertIn("cached demo result", cached.results[3].summary_skipped)

    def test_match_is_exact_on_query_borough_and_method(self):
        self.saved()
        self.assertIsNone(demo_cache.load_cached("vegan options", BOROUGH, METHOD, self.dir))
        self.assertIsNone(demo_cache.load_cached(QUERY, "Brooklyn", METHOD, self.dir))
        self.assertIsNone(demo_cache.load_cached(QUERY, BOROUGH, "semantic_log", self.dir))

    def test_unreadable_cache_file_is_treated_as_missing(self):
        self.saved()
        next(self.dir.glob("*.json")).write_text("{not json")
        self.assertIsNone(demo_cache.load_cached(QUERY, BOROUGH, METHOD, self.dir))

    def test_demo_mode_serves_cache_without_calling_the_model(self):
        self.saved()
        client = pipeline.UsageRecorder(FakeClient())
        outcome, raw = demo_cache.search_with_fallback(QUERY, BOROUGH, METHOD, demo_mode=True,
                                                       **self.live(client))
        self.assertTrue(outcome.cached)
        self.assertEqual(raw, [])
        self.assertEqual(client._client.calls, [])

    def test_demo_mode_without_a_cached_query_runs_live_silently(self):
        outcome, raw = demo_cache.search_with_fallback("tacos", BOROUGH, METHOD, demo_mode=True, **self.live())
        self.assertFalse(outcome.cached)
        self.assertTrue(raw)
        self.assertEqual(outcome.notices, [])  # no banner: a live search just runs

    def test_rate_limit_falls_back_to_the_cache_for_an_exact_match(self):
        self.saved()
        client = pipeline.UsageRecorder(FakeClient(summary="rate_limit"))
        outcome, _ = demo_cache.search_with_fallback(QUERY, BOROUGH, METHOD, **self.live(client))
        self.assertTrue(outcome.cached)
        self.assertIn("language model was unavailable", outcome.notices[0])
        self.assertTrue(all(v.summary_error is None for v in outcome.results))

    def test_rate_limit_without_a_cache_keeps_the_live_outcome(self):
        client = pipeline.UsageRecorder(FakeClient(summary="rate_limit"))
        outcome, raw = demo_cache.search_with_fallback("tacos", BOROUGH, METHOD, **self.live(client))
        self.assertFalse(outcome.cached)
        self.assertTrue(raw)
        self.assertEqual(outcome.results[0].summary_error["category"], "rate_limit")

    def test_other_errors_do_not_trigger_the_fallback(self):
        self.saved()
        outcome, _ = demo_cache.search_with_fallback(QUERY, BOROUGH, METHOD, **self.live(pipeline.UsageRecorder(FakeClient(parser="bad"))))
        self.assertFalse(outcome.cached)  # "response" error: not rate_limit / timeout / network

    def test_successful_live_search_is_not_replaced_by_the_cache(self):
        self.saved()
        outcome, _ = demo_cache.search_with_fallback(QUERY, BOROUGH, METHOD, **self.live())
        self.assertFalse(outcome.cached)

    def test_one_failed_summary_among_several_does_not_swap_in_the_cache(self):
        self.saved()
        errors = {"category": "rate_limit", "code": "x", "message": "m"}
        o = self.saved()
        o.results[0].summary_error, o.results[0].summary = errors, None
        self.assertFalse(demo_cache.live_failed(o))
        for v in o.results:
            v.summary_error, v.summary = errors, None
        self.assertTrue(demo_cache.live_failed(o))

    def test_every_chip_query_is_a_demo_query(self):
        from app import examples
        queries = {q for q, _, _ in demo_cache.DEMO_QUERIES}
        for query in examples.chip_queries():
            self.assertIn(query, queries)
        self.assertEqual(len(demo_cache.DEMO_QUERIES), len(set(demo_cache.DEMO_QUERIES)))

    def test_build_missing_skips_queries_that_have_a_file(self):
        called = []

        def fake(query, borough, method, **kw):
            called.append(query)
            raise SystemExit("stop after the first query that is built")
        first = demo_cache.DEMO_QUERIES[0]
        demo_cache.cache_path(*first, self.dir).write_text("{}", encoding="utf-8")  # a file exists for the first query
        with patch.object(demo_cache.pipeline, "make_llm_client", lambda: (object(), "m", None)), \
                patch.object(demo_cache.pipeline, "load_retriever", lambda name: object()), \
                patch.object(demo_cache.pipeline, "run_search", fake), self.assertRaises(SystemExit):
            demo_cache.build(self.dir, gap=0, only_missing=True)
        self.assertEqual(called, [demo_cache.DEMO_QUERIES[1][0]])

    def test_live_failed_categories(self):
        err = lambda c: {"category": c, "code": "x", "message": "m"}
        for category, expected in (("rate_limit", True), ("timeout", True), ("network", True), ("provider", False)):
            outcome = pipeline.SearchOutcome("r", "q", "Both", METHOD, None, None, None, [], parse_error=err(category))
            self.assertEqual(demo_cache.live_failed(outcome), expected, category)


class CacheNoticeTests(_CacheFixture):
    def test_exactly_one_notice_with_the_saved_date_when_served_in_demo_mode(self):
        self.saved()
        outcome, _ = demo_cache.search_with_fallback(QUERY, BOROUGH, METHOD, demo_mode=True, **self.live())
        self.assertEqual(len(outcome.notices), 1)
        self.assertRegex(outcome.notices[0], r"cached demo result \(saved \d{4}-\d{2}-\d{2}\), not a live search")
        self.assertEqual(outcome.cached_at, outcome.notices[0].split("saved ")[1][:10])

    def test_fallback_after_a_rate_limit_also_has_exactly_one_notice(self):
        self.saved()
        client = pipeline.UsageRecorder(FakeClient(summary="rate_limit"))
        outcome, _ = demo_cache.search_with_fallback(QUERY, BOROUGH, METHOD, **self.live(client))
        self.assertEqual(len(outcome.notices), 1)
        self.assertIn("language model was unavailable", outcome.notices[0])
        self.assertRegex(outcome.notices[0], r"\(saved \d{4}-\d{2}-\d{2}\)")

    def test_progress_reports_the_cache_stage(self):
        self.saved()
        seen = []
        demo_cache.search_with_fallback(QUERY, BOROUGH, METHOD, demo_mode=True, progress=seen.append, **self.live())
        self.assertEqual(seen, ["cache"])


class CommittedCacheTests(unittest.TestCase):
    def test_every_demo_query_has_a_committed_cache_file(self):
        for query, borough, method in demo_cache.SAVED_BEFORE_CHIPS:  # the chip queries are built later
            cached = demo_cache.load_cached(query, borough, method)
            self.assertIsNotNone(cached, (query, borough, method))
            self.assertGreaterEqual(len([v for v in cached.results if v.summary]), 3)  # the saved results hold the top 3 (built before the top-6 change)

    def test_allergy_demo_results_never_show_supported(self):
        for query, borough, method in demo_cache.DEMO_QUERIES:
            cached = demo_cache.load_cached(query, borough, method)
            if cached is not None and cached.allergy_sensitive:
                self.assertNotIn(pipeline.STATUS_SUPPORTED, [v.summary.status for v in cached.results if v.summary])


if __name__ == "__main__":
    unittest.main()
