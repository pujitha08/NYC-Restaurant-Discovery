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

    def test_search_shows_results_summaries_and_explain(self):
        at = self.search(self.start())
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("1. Place 1", text)
        self.assertIn("supported", text)
        self.assertIn("One reviewer reported vegan options for a group.", text)
        self.assertIn("Review from 2019-01", text)
        explain = [b for b in at.button if b.label == "Explain"]
        self.assertEqual(len(explain), 1)  # rank 4 only; top 3 summarized automatically
        explain[0].click()
        at.run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len([b for b in at.button if b.label == "Explain"]), 0)
        events = [line.split('"event": "')[1].split('"')[0] for line in self.log_path.read_text().splitlines()]
        self.assertEqual(events, ["search", "explain"])  # both logged to the temp file

    def test_missing_key_shows_setup_message_and_still_searches(self):
        setup = {"category": "configuration", "code": "invalid_api_key_configuration",
                 "message": pipeline.ERROR_MESSAGES["configuration"]}
        at = self.search(self.start(client=None, setup_error=setup))
        self.assertIn("GROQ_API_KEY", at.error[0].value)
        self.assertIn("Search still works", at.error[0].value)
        self.assertIn("1. Place 1", " ".join(m.value for m in at.markdown))

    def test_empty_query_and_markdown_escaping(self):
        at = self.start()
        at.button[0].click()
        at.run()
        self.assertIn("Type what you are looking for", " ".join(w.value for w in at.warning))
        from app.streamlit_app import md  # noqa: imported after AppTest so st is configured
        self.assertEqual(md("$3 *happy* hour"), r"\$3 \*happy\* hour")


if __name__ == "__main__":
    unittest.main()
