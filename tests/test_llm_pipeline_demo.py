"""Offline wiring tests. Block network and real-client initialization in every test."""

import unittest
from unittest.mock import patch

from examples.llm_pipeline_demo import FakeRetriever, FakeSDKClient, RAW_QUERY, borough_filters, run_pipeline


class PipelineDemoTests(unittest.TestCase):
    def setUp(self):
        guard = patch("socket.socket.connect", side_effect=AssertionError("Network forbidden"))
        guard.start()
        self.addCleanup(guard.stop)
        guard = patch("src.llm.groq_client.get_client", side_effect=AssertionError("Live client forbidden"))
        guard.start()
        self.addCleanup(guard.stop)
        self.lines = []

    def test_success_preserves_raw_text_boroughs_and_valid_citations(self):
        retriever, client = FakeRetriever(), FakeSDKClient()
        self.assertEqual(run_pipeline(RAW_QUERY, retriever, client, self.lines.append), "complete")
        self.assertEqual(retriever.calls, [(RAW_QUERY, ["Brooklyn"])])
        self.assertEqual(len(client.requests), 2)
        self.assertIn("support [reviews 101]", "\n".join(self.lines))
        self.assertIn("contradiction [reviews 102]", "\n".join(self.lines))
        self.assertIn("do not establish allergen safety", "\n".join(self.lines))

    def test_parser_failure_stops_before_search(self):
        retriever, client = FakeRetriever(), FakeSDKClient("parser-error")
        self.assertEqual(run_pipeline(RAW_QUERY, retriever, client, self.lines.append), "parser_error")
        self.assertEqual(retriever.calls, [])
        self.assertEqual(len(client.requests), 1)
        self.assertIn("response/invalid_response", "\n".join(self.lines))

    def test_empty_search_skips_summary(self):
        client = FakeSDKClient()
        self.assertEqual(run_pipeline(RAW_QUERY, FakeRetriever(empty=True), client, self.lines.append), "no_results")
        self.assertEqual(len(client.requests), 1)
        self.assertIn("No summarizer request made", "\n".join(self.lines))

    def test_summary_failure_shows_safe_error_without_claims(self):
        client = FakeSDKClient("summary-error")
        self.assertEqual(run_pipeline(RAW_QUERY, FakeRetriever(), client, self.lines.append), "summary_error")
        output = "\n".join(self.lines)
        self.assertIn("timeout/request_timeout", output)
        self.assertNotIn("SIMULATED_PRIVATE_DETAIL", output)
        self.assertNotIn("[reviews", output)
        self.assertEqual(len(client.requests), 2)

    def test_matched_restaurant_without_evidence_is_not_no_results_or_failure(self):
        retriever, client = FakeRetriever(empty_evidence=True), FakeSDKClient()
        self.assertEqual(run_pipeline(RAW_QUERY, retriever, client, self.lines.append), "insufficient_evidence")
        output = "\n".join(self.lines)
        self.assertIn("Synthetic Brooklyn Cafe", output)
        self.assertIn("Status: insufficient_evidence", output)
        self.assertIn("No review snippets supplied", output)
        self.assertIn("do not establish allergen safety", output)
        self.assertNotIn("No search results", output)
        self.assertNotIn("Summary unavailable", output)
        self.assertNotIn("[reviews", output)
        self.assertEqual(len(client.requests), 1)  # only the parser; no summary SDK call

    def test_only_explicit_supported_boroughs_are_filters(self):
        self.assertEqual(borough_filters(" brooklyn ", RAW_QUERY), ["Brooklyn"])
        self.assertIsNone(borough_filters("Queens", "Dinner in Queens"))
        self.assertIsNone(borough_filters("Williamsburg", "Dinner in Williamsburg"))
        self.assertIsNone(borough_filters("Brooklyn", "Dinner somewhere"))
        self.assertIsNone(borough_filters(None, RAW_QUERY))

    def test_unresolved_location_keeps_raw_search_without_borough_filter(self):
        retriever, client = FakeRetriever(), FakeSDKClient(location="Queens")
        raw = "Vegetarian dinner in Queens for a friend with a sesame allergy."
        self.assertEqual(run_pipeline(raw, retriever, client, self.lines.append), "complete")
        self.assertEqual(retriever.calls, [(raw, None)])
        self.assertEqual(len(client.requests), 3)  # parser plus two synthetic restaurant summaries
        self.assertIn("[reviews 201]", "\n".join(self.lines))


if __name__ == "__main__":
    unittest.main()
