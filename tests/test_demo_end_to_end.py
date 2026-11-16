"""The chips end to end against the real committed data/demo_cache (fake retriever and model).

A chip is answered from the saved result without any retriever or model call, so this runs the real page (chips, status, cards,
columns, allergy guard) on a real saved outcome. The chips whose cache file has not been built yet run live (fake retriever).
"""

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

from app import demo_cache, examples, pipeline
from tests.test_app_pipeline import FakeClient, FakeRetriever

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
SAVED = [(label, query) for label, query in examples.ALLERGY_CHIPS + examples.DIET_CHIPS
         if demo_cache.load_cached(query, examples.CHIP_BOROUGH, examples.CHIP_METHOD) is not None]


class NoRetriever(FakeRetriever):
    """A saved result must not search: any call fails the test."""

    def search(self, *a, **kw):
        raise AssertionError("the retriever was called for a saved result")


@unittest.skipIf(AppTest is None, "streamlit not installed")
class ChipsFromTheSavedResultsTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        st.cache_data.clear()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.client = FakeClient()
        llm = (pipeline.UsageRecorder(self.client), "openai/gpt-oss-20b", None)
        for obj, name, value in ((pipeline, "LOG_PATH", Path(tmp.name) / "log.jsonl"),
                                 (pipeline, "load_retriever", lambda n: NoRetriever()),
                                 (pipeline, "make_llm_client", lambda: llm)):
            p = patch.object(obj, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(st.cache_resource.clear)
        self.addCleanup(st.cache_data.clear)

    def click(self, label):
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        next(b for b in at.button if b.label == label).click()
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def test_the_peanut_chip_is_saved(self):
        self.assertIn("Peanut", [label for label, _ in SAVED])  # the one chip query that was already cached

    def test_a_saved_chip_runs_without_a_model_or_retriever_call(self):
        for label, query in SAVED:
            with self.subTest(chip=label):
                at = self.click(label)
                self.assertEqual(at.text_input[0].value, query)
                self.assertEqual(self.client.calls, [])
                self.assertEqual(at.status[0].label, "Done")
                html = " ".join(m.value for m in at.markdown)
                self.assertRegex(html, r"How this was found: a saved result from \d{4}-\d{2}-\d{2} \(no live search was run\)")
                count = re.search(r"(\d+) restaurants checked · (\d+) mention [^·]+ · (\d+) report problems · (\d+) not enough evidence", html)
                self.assertIsNotNone(count)
                self.assertEqual(int(count.group(1)), 10)
                self.assertGreaterEqual(html.count('class="care-card"') + html.count('class="problems-card"'), 1)
                self.assertEqual(html.count("not safety guarantees"), 1)  # the one footer line

    def test_saved_allergy_chips_never_say_supported_or_safe(self):
        for label, query in SAVED:
            if not any(w in query for w in ("allerg", "celiac")):
                continue
            with self.subTest(chip=label):
                html = " ".join(m.value for m in self.click(label).markdown)
                app_voice = re.sub(r"<blockquote.*?</blockquote>", " ", html, flags=re.S)  # a reviewer's own words may say "safe"
                self.assertNotRegex(app_voice, r"(?i)supported|\bsafe\b")
                self.assertIn("Reviews mention allergy care", html)
                self.assertIn("Reviews report problems", html)


if __name__ == "__main__":
    unittest.main()
