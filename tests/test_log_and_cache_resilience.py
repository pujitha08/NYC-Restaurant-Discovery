"""The request log and the stance label cache never break a search: switch off, unwritable folder, read-only disk."""

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import pipeline
from src.retrieval import stance_rerank as sr

OUTCOME = SimpleNamespace(request_id="r1", query="vegan", borough_choice="Both", method="semantic_capped", parsed=None,
                          summary_context=None, boroughs=None, results=[], notices=[], parse_error=None, setup_error=None,
                          timings={}, llm_calls=[], model=None, allergy_sensitive=False, cached=False, ranking_fallback=False,
                          cached_at=None)


class SwitchTests(unittest.TestCase):
    def test_default_on_and_the_off_values(self):
        self.assertTrue(pipeline.logging_enabled({}))
        for v in ("1", "true", "yes", "on", "anything"):
            self.assertTrue(pipeline.logging_enabled({"NYC_LOG_REQUESTS": v}), v)
        for v in ("0", "false", "False", "NO", "off", " 0 "):
            self.assertFalse(pipeline.logging_enabled({"NYC_LOG_REQUESTS": v}), v)


class LogEventTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_writes_when_on(self):
        path = self.dir / "sub" / "log.jsonl"
        with patch.dict(os.environ, {"NYC_LOG_REQUESTS": "1"}):
            pipeline.log_event(path, "search", OUTCOME)
        self.assertEqual(json.loads(path.read_text().splitlines()[0])["query"], "vegan")

    def test_switch_off_writes_nothing(self):
        path = self.dir / "log.jsonl"
        with patch.dict(os.environ, {"NYC_LOG_REQUESTS": "off"}):
            pipeline.log_event(path, "search", OUTCOME)
        self.assertFalse(path.exists())

    def test_unwritable_folder_is_skipped_silently(self):
        blocker = self.dir / "file"
        blocker.write_text("x")
        with patch.dict(os.environ, {"NYC_LOG_REQUESTS": "1"}):
            pipeline.log_event(blocker / "log.jsonl", "search", OUTCOME)  # parent is a file: mkdir fails
        ro = self.dir / "ro"
        ro.mkdir()
        ro.chmod(stat.S_IREAD | stat.S_IEXEC)
        self.addCleanup(ro.chmod, stat.S_IRWXU)
        with patch.dict(os.environ, {"NYC_LOG_REQUESTS": "1"}):
            pipeline.log_event(ro / "log.jsonl", "search", OUTCOME)  # read-only folder (a no-op check when run as root)

    def test_a_broken_record_does_not_raise(self):
        with patch.dict(os.environ, {"NYC_LOG_REQUESTS": "1"}):
            pipeline.log_event(self.dir / "log.jsonl", "search", object())  # not an outcome: building the record fails

    def test_the_default_path_is_used_unless_told_otherwise(self):
        with patch.object(pipeline, "LOG_PATH", self.dir / "default.jsonl"), patch.dict(os.environ, {"NYC_LOG_REQUESTS": "1"}):
            pipeline.log_event(pipeline._DEFAULT_LOG, "search", OUTCOME)
        self.assertTrue((self.dir / "default.jsonl").exists())


class LabelCacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_unwritable_cache_keeps_labels_in_memory_and_stops_writing(self):
        blocker = self.dir / "file"
        blocker.write_text("x")
        cache = sr.LabelCache(blocker / "cache.jsonl")  # parent is a file: every write fails
        cache.put("llm:x", "peanut allergy", "No peanuts here.", "mention")  # does not raise
        self.assertEqual(cache.get("llm:x", "peanut allergy", "No peanuts here."), "mention")
        self.assertIsNone(cache.path)  # gave up writing
        cache.put("llm:x", "peanut allergy", "Another sentence.", "negative")  # still fine, memory only
        self.assertEqual(cache.get("llm:x", "peanut allergy", "Another sentence."), "negative")

    def test_writable_cache_still_persists(self):
        path = self.dir / "c" / "cache.jsonl"
        cache = sr.LabelCache(path)
        cache.put("llm:x", "need", "A sentence.", "negative")
        self.assertEqual(sr.LabelCache(path).get("llm:x", "need", "A sentence."), "negative")

    def test_an_unreadable_cache_file_starts_empty(self):
        path = self.dir / "cache.jsonl"
        path.write_text("{}\n")
        with patch("builtins.open", side_effect=PermissionError("denied")):
            cache = sr.LabelCache(path)
        self.assertEqual(cache.labels, {})


if __name__ == "__main__":
    unittest.main()
