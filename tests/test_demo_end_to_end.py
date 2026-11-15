"""The four example buttons end to end in Demo mode, against the real committed data/demo_cache (fake retriever and model).

Demo mode serves an exact (query, borough, method) match from the cache without any retriever or model call, so this
runs the real page (chips, status, cards, allergy guard) on the real saved outcomes.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

from app import demo_cache, pipeline
from tests.test_app_pipeline import FakeClient, FakeRetriever

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
ALLERGY_EXAMPLES = {"Peanut allergy", "Celiac safe"}


class NoRetriever(FakeRetriever):
    """Demo mode must not search: any call fails the test."""

    def search(self, *a, **kw):
        raise AssertionError("the retriever was called in Demo mode")


@unittest.skipIf(AppTest is None, "streamlit not installed")
class ExampleButtonsInDemoModeTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        st.cache_data.clear()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for target, value in (("LOG_PATH", Path(tmp.name) / "log.jsonl"),):
            p = patch.object(pipeline, target, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = FakeClient()
        llm = (pipeline.UsageRecorder(self.client), "openai/gpt-oss-20b", None)
        for target, value in (("load_retriever", lambda name: NoRetriever()), ("make_llm_client", lambda: llm)):
            p = patch.object(pipeline, target, value)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(st.cache_resource.clear)
        self.addCleanup(st.cache_data.clear)

    def click(self, label):
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        next(c for c in at.checkbox if c.label.startswith("Demo mode")).check()
        next(b for b in at.button if b.label == label).click()
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    @staticmethod
    def cards(at):
        return [e for e in at.expander if e.label[:1].isdigit()]

    def test_every_example_runs_from_the_saved_outcome(self):
        from app.streamlit_app import EXAMPLES
        for label, query, borough in EXAMPLES:
            with self.subTest(example=label):
                at = self.click(label)
                self.assertEqual((at.text_input[0].value, at.pills[0].value), (query, borough))  # the chip filled the form
                self.assertEqual(self.client.calls, [])  # no model call at all
                cards = self.cards(at)
                self.assertEqual(len(cards), 10)
                self.assertEqual([c.proto.expanded for c in cards], [True] + [False] * 9)  # the top card is open
                self.assertEqual(at.status[0].label, "Done")
                notices = [m.value for m in at.markdown if 'class="notice-cache"' in m.value]
                self.assertEqual(len(notices), 1)  # one notice, neutral style, with the saved date
                self.assertRegex(notices[0], r"\(saved \d{4}-\d{2}-\d{2}\), not a live search")
                self.assertFalse(any("cached demo result" in i.value for i in at.info))
                self.assertFalse(any("cached demo result" in w.value.lower() for w in at.warning))
                chips = next(m.value for m in at.markdown if 'class="chips"' in m.value)
                self.assertNotIn("restaurants", chips)  # no count chip ...
                self.assertEqual(at.subheader[0].value, "10 restaurants")  # ... the heading carries the count
                for card in cards:
                    self.assertLessEqual(card.label.count("★"), 1)
                    if card.caption:
                        self.assertNotIn("★", card.caption[0].value)  # the rating appears once per card
                self.assertNotIn("Explain", [b.label for b in at.button])  # cached ranks 4+ have no live Explain

    def test_allergy_examples_keep_the_guard_and_the_full_caveat(self):
        for label in sorted(ALLERGY_EXAMPLES):
            with self.subTest(example=label):
                at = self.click(label)
                self.assertTrue(any("always confirm with the restaurant" in w.value for w in at.warning))  # full caveat
                main = " ".join(m.value for m in at.main.markdown)
                self.assertNotIn('class="claim claim-positive"', main)
                self.assertNotIn("✓", main)
                for card in self.cards(at):
                    self.assertNotIn("Supported", card.label)
                    self.assertNotIn("✓", card.label)
                summarized = [c for c in self.cards(at) if c.warning]
                self.assertGreaterEqual(len(summarized), 3)
                for card in summarized:
                    row = [w.value for w in card.warning if pipeline.ALLERGY_NOTE in w.value]
                    self.assertEqual(len(row), 1, card.label)  # the verdict and the allergy reminder are one row

    def test_non_allergy_examples_have_only_the_slim_note(self):
        for label in ("Vegan group", "Halal Chinese"):
            with self.subTest(example=label):
                at = self.click(label)
                self.assertFalse(any("always confirm with the restaurant" in w.value for w in at.warning))
                self.assertTrue(any("not safety guarantees" in c.value for c in at.caption))


if __name__ == "__main__":
    unittest.main()
