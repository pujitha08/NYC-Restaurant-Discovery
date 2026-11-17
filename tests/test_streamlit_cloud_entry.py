"""deploy/streamlit_cloud_app.py, the main file for Streamlit Community Cloud (AppTest with fakes: no network, no data download, no model)."""

import ast
import inspect
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

from app import pipeline
from deploy import space_data
from tests.test_app_pipeline import FakeClient, FakeRetriever

ROOT = Path(__file__).resolve().parent.parent
ENTRY = str(ROOT / "deploy" / "streamlit_cloud_app.py")
SECRETS = {"HF_TOKEN": "hf_FAKE_TOKEN_0123456789", "GROQ_API_KEY": "gsk_FAKE_KEY_0123456789", "GROQ_MODEL": "model-x",
           "GROQ_FALLBACK_MODEL": "model-y"}


def page_text(at):
    parts = []
    for group in (at.markdown, at.error, at.info, at.warning, at.caption, at.text):
        parts += [e.value for e in group]
    return "\n".join(parts)


@unittest.skipIf(AppTest is None, "streamlit not installed")
class EntryTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        st.cache_data.clear()
        self.addCleanup(st.cache_resource.clear)
        self.addCleanup(st.cache_data.clear)
        env = patch.dict(os.environ)  # the script sets variables in this process: put them back afterwards
        env.start()
        self.addCleanup(env.stop)
        for key in ("OMP_NUM_THREADS", "NYC_LOG_REQUESTS", "NYC_DATA_DIR", *SECRETS):
            os.environ.pop(key, None)
        self.calls = []
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        llm = (pipeline.UsageRecorder(FakeClient(location=None)), "openai/gpt-oss-20b", None)
        for obj, name, value in ((pipeline, "make_llm_client", lambda: llm), (pipeline, "load_retriever", lambda n: FakeRetriever()),
                                 (pipeline, "data_dir_missing", lambda: False), (pipeline, "LOG_PATH", self.tmp / "log.jsonl")):
            p = patch.object(obj, name, value)
            p.start()
            self.addCleanup(p.stop)

    def fake_ensure(self, result=(None, 0), lines=()):
        def ensure(environ, repo_root, default_dir=None, downloader=None, out=print):
            self.calls.append({"environ_has": {k: bool(environ.get(k)) for k in SECRETS}, "root": Path(repo_root)})
            for line in lines:
                out(line)
            return result
        p = patch.object(space_data, "ensure_data", ensure)
        p.start()
        self.addCleanup(p.stop)

    def run_entry(self, secrets=None):
        at = AppTest.from_file(ENTRY, default_timeout=30)
        for key, value in (secrets or {}).items():
            at.secrets[key] = value
        at.run()
        return at

    # ---- the app runs under the entry file --------------------------------------------------------------------------------
    def test_the_app_is_drawn_and_the_page_is_configured_exactly_once(self):
        self.fake_ensure()
        at = self.run_entry()
        # a second st.set_page_config (or one that is not the first Streamlit call) would raise StreamlitAPIException
        self.assertFalse(at.exception, at.exception)
        self.assertIn("Find restaurants whose reviews show they take your food allergy seriously.", page_text(at))
        self.assertEqual(len(self.calls), 1)

    def test_the_apps_main_is_told_the_page_is_already_configured(self):
        from app import streamlit_app
        self.fake_ensure()
        spy = MagicMock()
        with patch.object(streamlit_app, "main", spy):
            self.run_entry()
        spy.assert_called_once_with(configure_page=False)

    def test_run_directly_the_app_still_configures_its_own_page(self):
        from app import streamlit_app
        from app.page_config import PAGE_CONFIG
        self.assertTrue(inspect.signature(streamlit_app.main).parameters["configure_page"].default)
        with patch.object(st, "set_page_config") as config:
            at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=30)
            at.run()
        config.assert_called_once_with(**PAGE_CONFIG)
        self.assertFalse(at.exception, at.exception)

    # ---- environment and secrets ------------------------------------------------------------------------------------------
    def test_the_environment_defaults_are_set_and_a_value_already_present_wins(self):
        self.fake_ensure()
        self.run_entry()
        self.assertEqual((os.environ["OMP_NUM_THREADS"], os.environ["NYC_LOG_REQUESTS"]), ("1", "0"))
        os.environ["OMP_NUM_THREADS"], os.environ["NYC_LOG_REQUESTS"] = "4", "1"
        st.cache_resource.clear()
        self.run_entry()
        self.assertEqual((os.environ["OMP_NUM_THREADS"], os.environ["NYC_LOG_REQUESTS"]), ("4", "1"))

    def test_the_four_secrets_reach_the_environment_and_never_the_page(self):
        self.fake_ensure()
        at = self.run_entry(secrets=SECRETS)
        for key, value in SECRETS.items():
            self.assertEqual(os.environ[key], value)
        self.assertTrue(all(self.calls[0]["environ_has"].values()))  # the download sees them
        shown = page_text(at)
        for value in SECRETS.values():
            self.assertNotIn(value, shown)

    def test_without_secrets_the_app_still_starts(self):
        self.fake_ensure()
        at = self.run_entry()
        self.assertFalse(at.exception, at.exception)
        self.assertFalse(any(os.environ.get(k) for k in SECRETS))

    # ---- the data ----------------------------------------------------------------------------------------------------------
    def test_the_data_is_prepared_once_per_server_process(self):
        self.fake_ensure()
        at = self.run_entry()
        at.run()
        self.run_entry()  # a second session in the same process
        self.assertEqual(len(self.calls), 1)

    def test_nyc_data_dir_is_exported_before_the_app_is_imported_or_run(self):
        from app import streamlit_app
        folder = self.tmp / "data"
        self.fake_ensure(result=(folder, 0))
        seen = []
        with patch.object(streamlit_app, "main", lambda **kw: seen.append(os.environ.get("NYC_DATA_DIR"))):
            self.run_entry()
        self.assertEqual(seen, [str(folder)])

    def test_when_the_default_folder_is_right_nyc_data_dir_is_left_alone(self):
        self.fake_ensure(result=(None, 0))
        self.run_entry()
        self.assertNotIn("NYC_DATA_DIR", os.environ)

    def test_a_failure_shows_the_missing_file_without_the_token_and_the_app_does_not_start(self):
        from app import streamlit_app
        token = SECRETS["HF_TOKEN"]
        self.fake_ensure(result=(None, 1), lines=[f"ERROR: The search data is incomplete: missing processed/semantic/embeddings.npy in /x. {token}"])
        spy = MagicMock()
        with patch.object(streamlit_app, "main", spy):
            at = self.run_entry(secrets=SECRETS)
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.error), 1)
        self.assertIn("processed/semantic/embeddings.npy", at.error[0].value)
        self.assertNotIn(token, at.error[0].value)
        spy.assert_not_called()
        self.assertNotIn("Find restaurants", page_text(at))

    def test_a_failure_is_not_cached_so_the_next_load_tries_again(self):
        from app import streamlit_app
        self.fake_ensure(result=(None, 1), lines=["ERROR: missing processed/review_chunks.jsonl"])
        with patch.object(streamlit_app, "main", MagicMock()):
            at = self.run_entry()
        self.assertEqual(len(at.error), 1)
        self.fake_ensure(result=(None, 0))
        at2 = self.run_entry()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(at2.error), 0)
        self.assertIn("Find restaurants", page_text(at2))

    def test_the_loading_message_is_the_cached_functions_spinner_text(self):
        # shown only while the data is being prepared (the first start), after the page is configured
        self.assertIn("Loading data, first start takes a few minutes",
                      (ROOT / "deploy" / "streamlit_cloud_app.py").read_text())


class EntrySourceTests(unittest.TestCase):
    source = (ROOT / "deploy" / "streamlit_cloud_app.py").read_text()

    def test_no_app_or_src_module_but_the_page_config_is_imported_before_the_data_is_ready(self):
        tree = ast.parse(self.source)
        first_app_import = None
        before = []
        for node in tree.body:
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            if any(n == "app" or n.startswith("app.") or n == "src" or n.startswith("src.") for n in names) or (
                    isinstance(node, ast.ImportFrom) and node.module == "app"):
                if isinstance(node, ast.ImportFrom) and node.module == "app" and any(a.name == "streamlit_app" for a in node.names):
                    first_app_import = node.lineno
                    break
                before.append(ast.unparse(node))
        self.assertIsNotNone(first_app_import)
        self.assertEqual(before, ["from app.page_config import PAGE_CONFIG"])
        self.assertLess(self.source.index("prepare_data()\n"), self.source.index("from app import streamlit_app"))
        self.assertLess(self.source.index("os.environ[\"NYC_DATA_DIR\"]"), self.source.index("from app import streamlit_app"))

    def test_the_environment_defaults_come_before_torch_streamlit_or_the_app_can_be_imported(self):
        self.assertLess(self.source.index('os.environ.setdefault("OMP_NUM_THREADS", "1")'), self.source.index("import streamlit"))
        self.assertLess(self.source.index('os.environ.setdefault("NYC_LOG_REQUESTS", "0")'), self.source.index("import streamlit"))

    def test_set_page_config_is_the_first_streamlit_call(self):
        code = self.source[self.source.index("import streamlit as st"):]
        calls = [i for i in (code.find("st.set_page_config"), code.find("st.secrets"), code.find("st.cache_resource"),
                             code.find("st.error"), code.find("st.stop")) if i >= 0]
        self.assertEqual(code.index("st.set_page_config"), min(calls))
        self.assertEqual(code.count("st.set_page_config("), 1)

    def test_no_secret_is_printed_or_logged(self):
        for token in ("print(", "logging", "st.write", "st.code", "st.json", "st.text("):
            self.assertNotIn(token, self.source)

    def test_the_page_config_module_imports_nothing_from_src(self):
        code = "import sys; sys.path.insert(0, '.'); import app.page_config; print(sorted(m for m in sys.modules if m.split('.')[0] in ('src', 'torch')))"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        self.assertEqual(out, "[]")


if __name__ == "__main__":
    unittest.main()
