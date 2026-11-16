"""NYC_DATA_DIR: the data directory is configurable, and no module builds its own data path."""

import importlib
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from src import config

REPO = Path(__file__).resolve().parent.parent


class ResolveTests(unittest.TestCase):
    def test_default_is_the_repo_data_folder(self):
        self.assertEqual(config.resolve_data_dir({}), config.PROJECT_ROOT / "data")
        self.assertEqual(config.resolve_data_dir({"NYC_DATA_DIR": "   "}), config.PROJECT_ROOT / "data")  # blank = unset

    def test_env_var_overrides(self):
        self.assertEqual(config.resolve_data_dir({"NYC_DATA_DIR": "/cache/nyc"}), Path("/cache/nyc"))
        self.assertEqual(config.resolve_data_dir({"NYC_DATA_DIR": " /cache/nyc "}), Path("/cache/nyc"))
        self.assertEqual(config.resolve_data_dir({"NYC_DATA_DIR": "~/nyc"}), Path("~/nyc").expanduser())


class ModulesFollowTheSettingTests(unittest.TestCase):
    """Fresh interpreters, so every module-level path is computed under the variable."""

    def run_py(self, code, data_dir):
        env = {**os.environ, "NYC_DATA_DIR": str(data_dir), "HF_HUB_OFFLINE": "1"}
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        return out.stdout.strip().splitlines()

    def test_every_runtime_path_is_under_the_data_dir(self):
        code = ("from app import pipeline, demo_cache\n"
                "from src.retrieval import evidence, semantic, tfidf, tfidf_chunks, stance_rerank\n"
                "for p in (evidence.REVIEWS_PATH, evidence.INDEX_PATH, semantic.INDEX_DIR, semantic.CHUNKS_PATH, tfidf.INDEX_DIR,\n"
                "          tfidf_chunks.INDEX_DIR, tfidf_chunks.CHUNKS_PATH, stance_rerank.CACHE_PATH, stance_rerank.STANCE_MODEL_PATH):\n"
                "    print(p)\n")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for line in self.run_py(code, tmp):
                self.assertTrue(line.startswith(str(Path(tmp))), line)

    def test_data_stats_reads_the_relocated_folder(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "processed").mkdir()
            (Path(tmp) / "processed" / "cleaning_stats.json").write_text(json.dumps({"restaurants": 123, "kept": 4567}))
            out = self.run_py("from app import pipeline; print(pipeline.data_stats())", tmp)
            self.assertIn("'restaurants': 123", out[-1])

    def test_demo_cache_follows_the_data_dir_and_falls_back_to_the_repo_copy(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_py("from app import demo_cache; print(demo_cache.CACHE_DIR)", tmp)[-1],
                             str(REPO / "data" / "demo_cache"))  # none in the relocated folder: use the tracked copy
            (Path(tmp) / "demo_cache").mkdir()
            self.assertEqual(self.run_py("from app import demo_cache; print(demo_cache.CACHE_DIR)", tmp)[-1],
                             str(Path(tmp) / "demo_cache"))


class NoHardCodedDataPathsTests(unittest.TestCase):
    """No module under app/, src/ or eval/ builds a data path of its own: only config.DATA_DIR names the folder."""

    def test_no_module_spells_out_the_data_folder(self):
        pattern = re.compile(r'(REPO|PROJECT_ROOT|Path\(__file__\)[^\n]*)\s*/\s*"data"')
        offenders = []
        for sub in ("app", "src", "eval"):
            for path in (REPO / sub).rglob("*.py"):
                if path.name == "config.py" and sub == "src":
                    continue  # the one definition
                for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if path.name == "demo_cache.py" and "is_dir()" in line:
                        continue  # the one deliberate fallback: the demo results tracked in the repository
                    if pattern.search(line) and not line.lstrip().startswith("#"):
                        offenders.append(f"{path.relative_to(REPO)}:{n}: {line.strip()}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
