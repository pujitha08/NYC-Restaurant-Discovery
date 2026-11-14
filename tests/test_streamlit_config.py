"""The Streamlit theme file and page settings: light theme, one accent, WCAG AA contrast, wide layout, collapsed sidebar."""

import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

from tests.test_app_pipeline import FakeClient, FakeRetriever
from app import pipeline

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / ".streamlit" / "config.toml"
APP = str(ROOT / "app" / "streamlit_app.py")


def luminance(hex_color):
    channels = [int(hex_color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class ThemeTests(unittest.TestCase):
    theme = tomllib.loads(CONFIG.read_text())["theme"]

    def test_light_theme_with_one_accent(self):
        self.assertEqual(self.theme["base"], "light")
        for key in ("primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor"):
            self.assertRegex(self.theme[key], r"^#[0-9A-Fa-f]{6}$")

    def test_text_and_accent_have_aa_contrast(self):
        t = self.theme
        self.assertGreaterEqual(contrast(t["textColor"], t["backgroundColor"]), 7)  # high contrast (AAA)
        self.assertGreaterEqual(contrast(t["textColor"], t["secondaryBackgroundColor"]), 7)
        self.assertGreaterEqual(contrast(t["primaryColor"], t["backgroundColor"]), 4.5)  # accent text/links on white
        self.assertGreaterEqual(contrast("#FFFFFF", t["primaryColor"]), 4.5)  # white label on an accent button

    def test_accent_is_warm(self):
        r, g, b = (int(self.theme["primaryColor"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        self.assertTrue(r > g > b)

    def test_no_usage_statistics_and_no_web_fonts(self):
        text = CONFIG.read_text()
        self.assertFalse(tomllib.loads(text)["browser"]["gatherUsageStats"])
        self.assertNotIn("http", text.replace("https://docs", ""))
        self.assertIn(self.theme["font"], ("sans serif", "serif", "monospace"))  # Streamlit's built-in families only


@unittest.skipIf(AppTest is None, "streamlit not installed")
class PageConfigTests(unittest.TestCase):
    def test_wide_layout_and_collapsed_sidebar(self):
        st.cache_resource.clear()
        llm = (pipeline.UsageRecorder(FakeClient()), "openai/gpt-oss-20b", None)
        with patch.object(pipeline, "load_retriever", lambda name: FakeRetriever()), \
                patch.object(pipeline, "make_llm_client", lambda: llm), \
                patch.object(st, "set_page_config") as set_config:
            AppTest.from_file(APP, default_timeout=30).run()
        kwargs = set_config.call_args.kwargs
        self.assertEqual((kwargs["layout"], kwargs["initial_sidebar_state"]), ("wide", "collapsed"))
        st.cache_resource.clear()


if __name__ == "__main__":
    unittest.main()
