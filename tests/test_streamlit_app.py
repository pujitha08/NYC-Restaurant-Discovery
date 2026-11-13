"""Streamlit smoke tests with streamlit.testing.v1.AppTest (fake retriever and LLM; no data or network)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ImportError:  # streamlit is optional for the rest of the suite
    AppTest = None

from app import pipeline
from tests.test_app_pipeline import FakeClient, FakeRetriever

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@unittest.skipIf(AppTest is None, "streamlit not installed")
class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log_path = Path(tmp.name) / "app_requests.jsonl"
        log = patch.object(pipeline, "LOG_PATH", self.log_path)
        log.start()
        self.addCleanup(log.stop)
        self.addCleanup(st.cache_resource.clear)

    def start(self, client=FakeClient(location=None), setup_error=None):
        llm = (pipeline.UsageRecorder(client), "openai/gpt-oss-20b", None) if client else (None, None, setup_error)
        for target, value in (("load_retriever", lambda name: FakeRetriever()), ("make_llm_client", lambda: llm)):
            p = patch.object(pipeline, target, value)
            p.start()
            self.addCleanup(p.stop)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def search(self, at, query="vegan options for a group"):
        at.text_input[0].input(query)
        at.button[0].click()  # the form's Search button
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def test_initial_page(self):
        at = self.start()
        self.assertEqual(at.title[0].value, "NYC Restaurant Discovery")
        self.assertIn("through September 2021", at.warning[0].value)
        self.assertIn("always confirm with the restaurant", at.warning[0].value)
        self.assertEqual(at.selectbox[0].value, pipeline.DEFAULT_METHOD)
        self.assertEqual(at.radio[0].options, ["Both", "Manhattan", "Brooklyn"])
        self.assertEqual(len(at.error), 0)

    @staticmethod
    def results(at):
        """Result expanders (labels start with the rank), in order."""
        return [e for e in at.expander if e.label[:1].isdigit()]

    @staticmethod
    def text(block):
        return " ".join(m.value for m in block.markdown)

    def test_results_render_collapsed_with_header(self):
        at = self.search(self.start())
        results = self.results(at)
        self.assertEqual(len(results), 4)
        self.assertEqual([e.proto.expanded for e in results], [False] * 4)
        self.assertRegex(results[0].label, r"^1\. Place 1 — (Brooklyn|Manhattan) · (★ \d\.\d|no rating) · ✓ Supported")
        self.assertIn("Not summarized yet", results[3].label)  # rank 4: not summarized automatically
        self.assertNotIn("✓", results[3].label)
        self.assertEqual(at.subheader[0].value, "4 restaurants")
        self.assertTrue(any(c.value.startswith("Method ") for c in at.caption))  # timings above the list

    def test_guarded_allergy_header_has_no_check_mark(self):
        at = self.search(self.start(), "peanut allergy friendly dinner")
        for e in self.results(at):
            self.assertNotIn("✓", e.label)
            self.assertNotIn("Supported", e.label)
        self.assertIn(f"ℹ {pipeline.STATUS_NEUTRAL}", self.results(at)[0].label)
        self.assertIn(pipeline.ALLERGY_NOTE, " ".join(w.value for w in self.results(at)[0].warning))

    def test_expanding_shows_summary(self):
        at = self.search(self.start())
        outcome, _ = at.session_state["last_search"]
        at.session_state[f"open_{outcome.request_id}_{outcome.results[0].gmap_id}"] = True  # what a header click does
        at.run()
        first = self.results(at)[0]
        self.assertTrue(first.proto.expanded)
        self.assertFalse(any(e.proto.expanded for e in self.results(at)[1:]))
        body = self.text(first)
        self.assertIn("supported", body)
        self.assertIn("One reviewer reported vegan options for a group.", body)
        self.assertIn("Review from 2019-01", body)
        self.assertIn("All retrieved excerpts (2)", body)

    def test_explain_keeps_result_open(self):
        at = self.search(self.start())
        explain = [b for b in at.button if b.label == "Explain"]
        self.assertEqual(len(explain), 1)  # rank 4 only; top 3 summarized automatically
        explain[0].click()
        at.run()
        self.assertFalse(at.exception, at.exception)
        results = self.results(at)
        self.assertEqual([e.proto.expanded for e in results], [False, False, False, True])
        self.assertNotIn("Not summarized yet", results[3].label)
        self.assertIn("One reviewer reported", self.text(results[3]))
        self.assertEqual(len([b for b in at.button if b.label == "Explain"]), 0)
        at.run()  # an unrelated rerun keeps it open
        self.assertTrue(self.results(at)[3].proto.expanded)
        events = [line.split('"event": "')[1].split('"')[0] for line in self.log_path.read_text().splitlines()]
        self.assertEqual(events, ["search", "explain"])  # both logged to the temp file

    def test_expand_all_collapse_all_and_new_search_resets(self):
        at = self.search(self.start())
        next(b for b in at.button if b.label == "Expand all").click()
        at.run()
        self.assertEqual([e.proto.expanded for e in self.results(at)], [True] * 4)
        at.run()
        self.assertEqual([e.proto.expanded for e in self.results(at)], [True] * 4)  # survives reruns
        next(b for b in at.button if b.label == "Collapse all").click()
        at.run()
        self.assertEqual([e.proto.expanded for e in self.results(at)], [False] * 4)
        next(b for b in at.button if b.label == "Expand all").click()
        at.run()
        self.search(at, "vegan brunch")  # a new search starts collapsed
        self.assertEqual([e.proto.expanded for e in self.results(at)], [False] * 4)

    def test_missing_key_shows_setup_message_and_still_searches(self):
        setup = {"category": "configuration", "code": "invalid_api_key_configuration",
                 "message": pipeline.ERROR_MESSAGES["configuration"]}
        at = self.search(self.start(client=None, setup_error=setup))
        self.assertIn("GROQ_API_KEY", at.error[0].value)
        self.assertIn("Search still works", at.error[0].value)
        self.assertTrue(self.results(at)[0].label.startswith("1. Place 1"))
        self.assertIn("Not summarized yet", self.results(at)[0].label)

    def test_empty_query_and_markdown_escaping(self):
        at = self.start()
        at.button[0].click()
        at.run()
        self.assertIn("Type what you are looking for", " ".join(w.value for w in at.warning))
        from app.streamlit_app import md  # noqa: imported after AppTest so st is configured
        self.assertEqual(md("$3 *happy* hour"), r"\$3 \*happy\* hour")


if __name__ == "__main__":
    unittest.main()
