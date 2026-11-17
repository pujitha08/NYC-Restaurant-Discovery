"""The data download and check run before the app starts on a hosted deployment (deploy/space_data.py): no network; the checks run against
a temporary fake data folder and a fake downloader."""

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deploy import space_data

ROOT = Path(__file__).resolve().parent.parent
NAMED = ["processed/review_chunks.jsonl", "processed/reviews_clean.jsonl", "processed/reviews_clean_index.npz",
         "processed/semantic/embeddings.npy", "processed/semantic/meta.jsonl"]
TOKEN = "hf_FAKE_TOKEN_for_the_test_0123456789"


def fake_sizes():
    """The sizes fake_data produces (7 bytes per file, and an info.json of whatever length its JSON has)."""
    with tempfile.TemporaryDirectory() as tmp:
        fake_data(tmp)
        return {name: (Path(tmp) / name).stat().st_size for name in space_data.REQUIRED}


def fake_data(root, sizes=None, skip=()):
    """A small stand-in for the dataset: the required files with tiny contents (sizes given by `sizes`, default 7 bytes)."""
    root = Path(root)
    made = {}
    for name in space_data.REQUIRED:
        if name in skip:
            continue
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (sizes or {}).get(name)
        path.write_bytes(b"x" * (content if content is not None else 7))
        made[name] = path.stat().st_size
    if "processed/semantic/info.json" not in skip:  # info.json says how big the chunk file was when the index was built
        chunk = root / "processed/review_chunks.jsonl"
        size = chunk.stat().st_size if chunk.exists() else 7
        (root / "processed/semantic/info.json").write_text(json.dumps({"chunks_file": {"size": size, "sha256": "x"}}))
    return root


class VerifyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_a_complete_folder_passes(self):
        fake_data(self.dir)
        space_data.verify(self.dir)  # no exception
        self.assertEqual(space_data.missing_files(self.dir), [])

    def test_each_named_file_missing_gives_a_clear_error_that_names_it(self):
        for name in NAMED:
            with self.subTest(missing=name):
                folder = self.dir / name.replace("/", "_")
                fake_data(folder, skip=(name,))
                with self.assertRaises(space_data.DataIncomplete) as ctx:
                    space_data.verify(folder)
                message = str(ctx.exception)
                self.assertIn(name, message)
                self.assertIn("missing", message)
                self.assertIn(space_data.REPO_ID, message)

    def test_every_missing_file_is_listed(self):
        fake_data(self.dir, skip=("processed/semantic/meta.jsonl", "processed/reviews_clean_index.npz"))
        with self.assertRaises(space_data.DataIncomplete) as ctx:
            space_data.verify(self.dir)
        self.assertIn("processed/semantic/meta.jsonl", str(ctx.exception))
        self.assertIn("processed/reviews_clean_index.npz", str(ctx.exception))

    def test_an_empty_folder_names_the_files(self):
        with self.assertRaises(space_data.DataIncomplete) as ctx:
            space_data.verify(self.dir)
        for name in NAMED:
            self.assertIn(name, str(ctx.exception))

    def test_a_chunk_file_of_the_wrong_size_is_reported_before_the_app_refuses_to_load_the_index(self):
        fake_data(self.dir)
        info = self.dir / "processed/semantic/info.json"
        info.write_text(json.dumps({"chunks_file": {"size": 999, "sha256": "x"}}))
        with self.assertRaises(space_data.DataIncomplete) as ctx:
            space_data.verify(self.dir)
        self.assertIn("review_chunks.jsonl", str(ctx.exception))
        self.assertIn("999", str(ctx.exception))
        self.assertIn("byte-identical", str(ctx.exception))

    def test_optional_files_are_not_required(self):
        fake_data(self.dir)
        for name in space_data.OPTIONAL:
            self.assertNotIn(name, space_data.REQUIRED)
        space_data.verify(self.dir)

    def test_the_required_list_covers_what_the_semantic_search_opens(self):
        # src/retrieval/semantic.py: embeddings, owner, offsets, meta, info; review_chunks.jsonl; reviews_clean.jsonl and its index
        self.assertEqual(set(space_data.REQUIRED), set(NAMED) | {"processed/semantic/chunk_restaurant.npy",
                                                                  "processed/semantic/chunk_offsets.npy", "processed/semantic/info.json"})


class DownloadDecisionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_skip_when_every_file_has_its_expected_size(self):
        fake_data(self.dir)
        self.assertFalse(space_data.needs_download(self.dir, fake_sizes()))

    def test_download_when_a_file_is_missing_or_has_another_size(self):
        fake_data(self.dir)
        expected = fake_sizes()
        (self.dir / "processed/semantic/meta.jsonl").write_bytes(b"xx")
        self.assertTrue(space_data.needs_download(self.dir, expected))
        (self.dir / "processed/semantic/meta.jsonl").unlink()
        self.assertTrue(space_data.needs_download(self.dir, expected))

    def test_an_empty_folder_needs_the_download(self):
        self.assertTrue(space_data.needs_download(self.dir))

    def test_the_expected_sizes_are_those_of_the_uploaded_dataset(self):
        self.assertEqual(space_data.REQUIRED["processed/review_chunks.jsonl"], 505_962_950)
        self.assertEqual(space_data.REQUIRED["processed/reviews_clean.jsonl"], 705_310_508)


class DemoCacheShieldTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_a_downloaded_demo_cache_is_moved_aside_so_it_cannot_shadow_the_repos(self):
        (self.dir / "demo_cache").mkdir()
        (self.dir / "demo_cache" / "a.json").write_text("{}")
        aside = space_data.shield_demo_cache(self.dir)
        self.assertFalse((self.dir / "demo_cache").exists())
        self.assertTrue((aside / "a.json").exists())
        self.assertIsNone(space_data.shield_demo_cache(self.dir))  # nothing to do the second time

    def test_after_the_shield_the_app_uses_the_demo_cache_shipped_with_the_code(self):
        (self.dir / "demo_cache").mkdir()
        space_data.shield_demo_cache(self.dir)
        code = ("import sys; sys.path.insert(0, '.'); from app import demo_cache; "
                "print(demo_cache.CACHE_DIR)")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                             env={"NYC_DATA_DIR": str(self.dir), "PATH": "/usr/bin:/bin", "HOME": str(self.dir)}).stdout.strip().splitlines()[-1]
        self.assertEqual(Path(out), ROOT / "data" / "demo_cache")

    def test_without_the_shield_a_downloaded_demo_cache_would_win(self):
        (self.dir / "demo_cache").mkdir()
        code = "import sys; sys.path.insert(0, '.'); from app import demo_cache; print(demo_cache.CACHE_DIR)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                             env={"NYC_DATA_DIR": str(self.dir), "PATH": "/usr/bin:/bin", "HOME": str(self.dir)}).stdout.strip().splitlines()[-1]
        self.assertEqual(Path(out), self.dir / "demo_cache")  # why the shield exists

    def test_the_download_only_asks_for_the_processed_folder(self):
        import inspect
        self.assertIn('allow_patterns=["processed/**"]', inspect.getsource(space_data.download))


class PrepareTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "nyc_data"
        p = patch.object(space_data, "REQUIRED", fake_sizes())
        p.start()
        self.addCleanup(p.stop)
        self.lines = []

    def run_prepare(self, downloader, token=TOKEN):
        env = {"HF_TOKEN": token} if token else {}
        code = space_data.prepare(self.dir, environ=env, downloader=downloader, out=self.lines.append)
        return code, "\n".join(self.lines)

    def test_a_cold_start_downloads_then_verifies(self):
        calls = []

        def downloader(folder, token):
            calls.append((Path(folder), token))
            fake_data(folder)
        code, text = self.run_prepare(downloader)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [(self.dir, TOKEN)])  # the secret goes to the download call, not to the output
        self.assertIn("ready", text)
        self.assertNotIn(TOKEN, text)

    def test_a_warm_restart_skips_the_download(self):
        fake_data(self.dir)

        def downloader(folder, token):
            raise AssertionError("the download must be skipped")
        code, text = self.run_prepare(downloader)
        self.assertEqual(code, 0)
        self.assertIn("skipping the download", text)

    def test_a_download_that_leaves_a_file_out_exits_with_a_message_naming_it(self):
        code, text = self.run_prepare(lambda folder, token: fake_data(folder, skip=("processed/semantic/embeddings.npy",)))
        self.assertEqual(code, 1)
        self.assertIn("processed/semantic/embeddings.npy", text)
        self.assertIn("ERROR", text)

    def test_a_failed_download_exits_cleanly_and_never_prints_the_token(self):
        def downloader(folder, token):
            raise RuntimeError(f"401 Unauthorized for url with authorization {token}")
        code, text = self.run_prepare(downloader)
        self.assertEqual(code, 1)
        self.assertIn("Could not download", text)
        self.assertIn("HF_TOKEN", text)
        self.assertNotIn(TOKEN, text)

    def test_without_a_token_it_says_so_and_still_tries(self):
        seen = []
        code, text = self.run_prepare(lambda folder, token: (seen.append(token), fake_data(folder)), token="")
        self.assertEqual(code, 0)
        self.assertEqual(seen, [""])
        self.assertIn("HF_TOKEN is not set", text)

    def test_a_demo_cache_in_the_download_is_moved_aside(self):
        def downloader(folder, token):
            fake_data(folder)
            (Path(folder) / "demo_cache").mkdir()
        code, text = self.run_prepare(downloader)
        self.assertEqual(code, 0)
        self.assertFalse((self.dir / "demo_cache").exists())
        self.assertIn("demo_cache", text)

    def test_scrub(self):
        self.assertEqual(space_data.scrub(f"a {TOKEN} b", TOKEN), "a <token> b")
        self.assertEqual(space_data.scrub("no token", ""), "no token")


class ModuleIsolationTests(unittest.TestCase):
    def test_the_data_check_imports_nothing_from_the_app_or_src(self):
        code = ("import sys; sys.path.insert(0, '.'); import deploy.space_data; "
                "bad = [m for m in sys.modules if m.split('.')[0] in ('app', 'src', 'streamlit', 'torch')]; print(bad)")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        self.assertEqual(out, "[]")

    def test_the_usage_message(self):
        self.assertEqual(space_data.main(["nonsense"]), 2)


class EnsureDataTests(unittest.TestCase):
    """deploy/space_data.ensure_data: where the data is, and downloading only when it is not already here."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.default = self.root / "home" / "nyc_data"
        p = patch.object(space_data, "REQUIRED", fake_sizes())
        p.start()
        self.addCleanup(p.stop)
        self.lines = []

    def run_ensure(self, environ, downloader=None):
        def never(folder, token):
            raise AssertionError("no download expected")
        folder, code = space_data.ensure_data(environ, self.repo, default_dir=self.default, downloader=downloader or never,
                                              out=self.lines.append)
        return folder, code, "\n".join(self.lines)

    def test_nyc_data_dir_set_and_complete_means_no_download_and_it_is_used(self):
        mine = self.root / "mine"
        fake_data(mine)
        folder, code, text = self.run_ensure({"NYC_DATA_DIR": str(mine)})
        self.assertEqual((folder, code), (mine, 0))
        self.assertIn("no download", text)

    def test_a_repo_data_folder_that_already_holds_the_data_means_no_download_and_nothing_to_set(self):
        fake_data(self.repo / "data")
        folder, code, text = self.run_ensure({})
        self.assertEqual((folder, code), (None, 0))  # None: the app's default folder (<repo>/data) is the right one
        self.assertIn("no download", text)

    def test_a_local_clone_with_data_is_untouched_even_if_nyc_data_dir_is_unset_and_a_token_exists(self):
        fake_data(self.repo / "data")
        folder, code, _ = self.run_ensure({"HF_TOKEN": TOKEN})
        self.assertEqual((folder, code), (None, 0))

    def test_no_data_anywhere_downloads_into_the_writable_default_folder(self):
        calls = []

        def downloader(folder, token):
            calls.append((Path(folder), token))
            fake_data(folder)
        folder, code, text = self.run_ensure({"HF_TOKEN": TOKEN}, downloader)
        self.assertEqual((folder, code), (self.default, 0))
        self.assertEqual(calls, [(self.default, TOKEN)])
        self.assertNotIn(TOKEN, text)

    def test_nyc_data_dir_set_but_empty_downloads_into_that_folder(self):
        mine = self.root / "mine"
        calls = []
        folder, code, _ = self.run_ensure({"NYC_DATA_DIR": str(mine)}, lambda f, t: (calls.append(Path(f)), fake_data(f)))
        self.assertEqual((folder, code, calls), (mine, 0, [mine]))

    def test_a_partial_repo_data_folder_is_not_mistaken_for_the_data(self):
        fake_data(self.repo / "data", skip=("processed/semantic/embeddings.npy",))
        calls = []
        folder, code, _ = self.run_ensure({}, lambda f, t: (calls.append(Path(f)), fake_data(f)))
        self.assertEqual((folder, calls), (self.default, [self.default]))

    def test_files_that_do_not_match_the_index_are_an_error_not_a_silent_download(self):
        fake_data(self.repo / "data")
        (self.repo / "data/processed/semantic/info.json").write_text(json.dumps({"chunks_file": {"size": 999, "sha256": "x"}}))
        folder, code, text = self.run_ensure({})
        self.assertEqual((folder, code), (None, 1))
        self.assertIn("byte-identical", text)

    def test_a_failed_download_returns_the_error_and_no_folder(self):
        def downloader(folder, token):
            raise RuntimeError("boom " + token)
        folder, code, text = self.run_ensure({"HF_TOKEN": TOKEN}, downloader)
        self.assertEqual((folder, code), (None, 1))
        self.assertIn("Could not download", text)
        self.assertNotIn(TOKEN, text)


class SecretsTests(unittest.TestCase):
    def test_the_four_secrets_are_copied_when_present(self):
        env = {}
        secrets = {"HF_TOKEN": "t", "GROQ_API_KEY": "k", "GROQ_MODEL": "m", "GROQ_FALLBACK_MODEL": "f", "OTHER": "x"}
        self.assertEqual(space_data.export_secrets(secrets, env), ["HF_TOKEN", "GROQ_API_KEY", "GROQ_MODEL", "GROQ_FALLBACK_MODEL"])
        self.assertEqual(env, {"HF_TOKEN": "t", "GROQ_API_KEY": "k", "GROQ_MODEL": "m", "GROQ_FALLBACK_MODEL": "f"})  # "OTHER" is not copied

    def test_absent_blank_and_already_set_values_are_left_alone(self):
        env = {"GROQ_MODEL": "from-env"}
        copied = space_data.export_secrets({"GROQ_MODEL": "from-secrets", "HF_TOKEN": "  ", "GROQ_API_KEY": "k"}, env)
        self.assertEqual(copied, ["GROQ_API_KEY"])
        self.assertEqual(env, {"GROQ_MODEL": "from-env", "GROQ_API_KEY": "k"})

    def test_a_secrets_object_that_raises_is_treated_as_empty(self):
        class Broken:
            def __contains__(self, key):
                raise FileNotFoundError("no secrets.toml")
        env = {}
        self.assertEqual(space_data.export_secrets(Broken(), env), [])
        self.assertEqual(env, {})

    def test_the_default_data_folder_is_writable_and_falls_back_to_the_temp_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(space_data.default_data_dir(tmp), Path(tmp) / "nyc_data")
            self.assertTrue((Path(tmp) / "nyc_data").is_dir())
            blocked = Path(tmp) / "file"
            blocked.write_text("not a folder")
            fallback = space_data.default_data_dir(blocked)  # cannot create a folder under a file
            self.assertEqual(fallback.name, "nyc_data")
            self.assertNotEqual(fallback.parent, blocked)


if __name__ == "__main__":
    unittest.main()
