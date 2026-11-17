"""requirements.txt (read by Streamlit Community Cloud) lists exactly what the app imports at run time; the development tools are in requirements-dev.txt."""

import ast
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Module name -> distribution name, for the modules whose names differ.
DIST = {"sklearn": "scikit-learn", "dotenv": "python-dotenv", "sentence_transformers": "sentence-transformers"}
# Files that make up the running app: the app itself, retrieval, the LLM clients and the shared modules (+ eval.make_pool, which the app imports).
RUNTIME = ([*(REPO / "app").glob("*.py"), *(REPO / "src" / "retrieval").glob("*.py"), *(REPO / "src" / "llm").glob("*.py")]
           + [REPO / "src" / "allergy.py", REPO / "src" / "config.py", REPO / "src" / "schemas.py", REPO / "eval" / "make_pool.py"]
           + [REPO / "deploy" / "space_data.py", REPO / "deploy" / "streamlit_cloud_app.py"])  # the Streamlit Cloud entry and its data download


def requirement_names(path):
    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[<>=!~\[ ]", line)[0].lower().replace("_", "-"))
    return names


def imported_third_party():
    local = {"app", "src", "eval", "tests", "deploy"}
    found = {}
    for path in RUNTIME:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else [])
            for m in mods:
                top = m.split(".")[0]
                if top not in sys.stdlib_module_names and top not in local:
                    found.setdefault(top, path.relative_to(REPO).as_posix())
    return found


class RequirementsSpaceTests(unittest.TestCase):
    def setUp(self):
        self.space = requirement_names(REPO / "requirements.txt")

    def test_every_third_party_import_of_the_app_is_listed(self):
        missing = {m: where for m, where in imported_third_party().items() if DIST.get(m, m).lower().replace("_", "-") not in self.space}
        self.assertEqual(missing, {}, "imported at run time but missing from requirements.txt")

    def test_no_development_tools(self):
        self.assertEqual(self.space & {"jupyter", "matplotlib", "datasets", "nltk"}, set())

    def test_the_app_does_not_import_the_development_tools(self):
        self.assertEqual(set(imported_third_party()) & {"matplotlib", "datasets", "nltk", "jupyter", "IPython"}, set())

    def test_scikit_learn_is_pinned_to_the_version_the_indexes_were_built_with(self):
        text = (REPO / "requirements.txt").read_text()
        self.assertRegex(text, r"(?m)^scikit-learn==1\.6\.1\b")

    def test_torch_comes_from_the_cpu_index(self):
        text = (REPO / "requirements.txt").read_text()
        self.assertRegex(text, r"(?m)^--extra-index-url https://download\.pytorch\.org/whl/cpu\s*$")
        self.assertIn("torch", self.space)

    def test_the_data_download_dependency_is_listed(self):
        self.assertIn("huggingface-hub", self.space)

    def test_development_tools_are_in_requirements_dev_txt(self):
        dev = requirement_names(REPO / "requirements-dev.txt")
        self.assertTrue({"jupyter", "matplotlib", "datasets", "nltk"} <= dev)
        self.assertEqual(self.space & {"jupyter", "matplotlib", "datasets", "nltk", "pandas-stubs"}, set())

    def test_requirements_dev_txt_is_the_old_development_list(self):
        if subprocess.run(["git", "rev-parse", "--verify", "-q", "feature/final-polish"], cwd=REPO, capture_output=True).returncode:
            self.skipTest("the branch point is not available")
        old = subprocess.run(["git", "show", "feature/final-polish:requirements.txt"], cwd=REPO, capture_output=True, text=True).stdout
        self.assertEqual((REPO / "requirements-dev.txt").read_text(), old)

    def test_there_is_a_single_slim_list_and_no_leftover_space_file(self):
        self.assertFalse((REPO / "requirements-space.txt").exists())


if __name__ == "__main__":
    unittest.main()
