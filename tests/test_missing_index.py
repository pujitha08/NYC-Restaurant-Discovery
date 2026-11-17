"""A missing index or data folder is a clear message or a hidden option, never a stack trace."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

from app import pipeline
from eval import make_pool
from tests.test_app_pipeline import FakeClient

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


class LoaderTests(unittest.TestCase):
    def test_tfidf_loaders_return_none_when_the_index_is_missing(self):
        with patch.object(make_pool, "doc_tfidf_exists", return_value=False), \
                patch.object(make_pool, "chunk_tfidf_exists", return_value=False):
            self.assertIsNone(make_pool.load_doc_tfidf())
            self.assertIsNone(make_pool.load_chunk_tfidf())
            self.assertIsNone(pipeline.load_retriever("doc_tfidf"))

    def test_tfidf_loaders_load_when_the_index_is_there(self):
        sentinel = object()
        with patch.object(make_pool, "doc_tfidf_exists", return_value=True), \
                patch("src.retrieval.tfidf.TfidfRetriever.load", return_value=sentinel):
            self.assertIs(make_pool.load_doc_tfidf(), sentinel)


class AvailableMethodsTests(unittest.TestCase):
    def available(self, **exists):
        checks = {"semantic": True, "semantic_stance": True, "doc_tfidf": True, "chunk_tfidf": True, **exists}
        fakes = {name: (lambda v=v: v) for name, v in checks.items()}
        with patch.dict(pipeline.INDEX_EXISTS, fakes):
            return pipeline.available_methods()

    def test_all_methods_when_every_index_is_there(self):
        self.assertEqual(self.available(), list(pipeline.METHODS))

    def test_methods_without_their_index_are_hidden(self):
        got = self.available(doc_tfidf=False, chunk_tfidf=False)
        self.assertEqual({pipeline.METHODS[m][0] for m in got}, {"semantic", "semantic_stance"})
        self.assertIn(pipeline.DEFAULT_METHOD, got)
        self.assertEqual(pipeline.DEFAULT_METHOD, "semantic_capped")

    def test_semantic_methods_are_hidden_when_the_semantic_index_is_missing(self):
        got = self.available(semantic=False, semantic_stance=False)
        self.assertNotIn(pipeline.DEFAULT_METHOD, got)

    def test_data_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(pipeline, "DATA_DIR", Path(tmp)):
                self.assertFalse(pipeline.data_dir_missing())
            with patch.object(pipeline, "DATA_DIR", Path(tmp) / "nope"):
                self.assertTrue(pipeline.data_dir_missing())


@unittest.skipIf(AppTest is None, "streamlit not installed")
class AppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        st.cache_data.clear()
        self.addCleanup(st.cache_resource.clear)
        self.addCleanup(st.cache_data.clear)
        llm = (pipeline.UsageRecorder(FakeClient(location=None)), "openai/gpt-oss-20b", None)
        for target, value in (("make_llm_client", lambda: llm),):
            p = patch.object(pipeline, target, value)
            p.start()
            self.addCleanup(p.stop)

    def run_app(self, **patches):
        for target, value in patches.items():
            p = patch.object(pipeline, target, value)
            p.start()
            self.addCleanup(p.stop)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def test_selector_lists_only_available_methods_and_defaults_to_semantic_capped(self):
        only = ["semantic_capped", "semantic_log", "semantic_capped_stance"]
        at = self.run_app(available_methods=lambda: only, data_dir_missing=lambda: False)
        at.toggle[0].set_value(True).run()  # the method dropdown is part of Evaluation tools
        box = next(b for b in at.selectbox if b.label == "Retrieval method")
        self.assertEqual(list(box.options), only)
        self.assertEqual(box.value, "semantic_capped")

    def test_missing_data_folder_shows_a_message_not_a_traceback(self):
        at = self.run_app(data_dir_missing=lambda: True)
        self.assertEqual(len(at.error), 1)
        self.assertIn("data folder was not found", at.error[0].value)
        self.assertIn("NYC_DATA_DIR", at.error[0].value)
        self.assertEqual(len(at.text_input), 0)  # the search form is not drawn


if __name__ == "__main__":
    unittest.main()
