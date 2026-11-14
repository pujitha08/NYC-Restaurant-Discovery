"""Budgeted, resumable stance labeling: ledger window, allowance, blackout dates, stop and resume (fakes only)."""

import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import eval.llm_judge as judge
from eval import label_stance as ls
from src.retrieval import stance_rerank as sr
from tests.test_stance_rerank import FakeBase, LabelFake, res

MODEL = "openai/gpt-oss-20b"
T0 = datetime.datetime(2026, 11, 13, 12, 0, tzinfo=datetime.timezone.utc)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.now = T0
        self.ledger = ls.Ledger(Path(tmp.name) / "l.jsonl", clock=lambda: self.now)

    def test_rolling_24_hours_and_per_model(self):
        self.ledger.add(MODEL, 1000)
        self.ledger.add("other", 5000)
        self.now = T0 + datetime.timedelta(hours=23)
        self.ledger.add(MODEL, 200)
        self.assertEqual(self.ledger.used_24h(MODEL), 1200)
        self.now = T0 + datetime.timedelta(hours=25)  # the first entry has aged out
        self.assertEqual(self.ledger.used_24h(MODEL), 200)
        self.assertEqual(ls.Ledger(self.ledger.path.parent / "none.jsonl").used_24h(MODEL), 0)

    def test_broken_line_is_ignored(self):
        self.ledger.add(MODEL, 10)
        with open(self.ledger.path, "a") as f:
            f.write("{oops\n")
        self.assertEqual(self.ledger.used_24h(MODEL), 10)


class AllowanceTests(unittest.TestCase):
    def test_stops_at_seventy_percent_and_caps_at_one_chunk(self):
        self.assertEqual(ls.allowance(0, 200_000, 0.7, 100_000), 100_000)  # chunk cap
        self.assertEqual(ls.allowance(45_000, 200_000, 0.7, 100_000), 95_000)  # 140K - 45K
        self.assertEqual(ls.allowance(140_000, 200_000), 0)
        self.assertEqual(ls.allowance(150_000, 200_000), 0)

    def test_blackout_dates(self):
        self.assertTrue(ls.blackout(datetime.datetime(2026, 11, 17, 3, tzinfo=datetime.timezone.utc)))
        self.assertTrue(ls.blackout(datetime.datetime(2026, 11, 18, 23, tzinfo=datetime.timezone.utc)))
        self.assertFalse(ls.blackout(T0))
        self.assertFalse(ls.blackout(T0, ()))


class RunTests(unittest.TestCase):
    def setUp(self):
        for target, value in ((judge, "sleep"), (ls, "CALL_ESTIMATE")):
            p = patch.object(target, value, (lambda s: None) if value == "sleep" else 20)
            p.start()
            self.addCleanup(p.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.ledger = ls.Ledger(self.dir / "l.jsonl", clock=lambda: T0)
        self.client = LabelFake()
        self.cache_path = self.dir / "cache.jsonl"
        self.queries = [{"query_id": "Q1", "query": "peanut allergy", "borough": ""},
                        {"query_id": "Q2", "query": "celiac safe", "borough": ""}]

    def reranker(self):
        base = FakeBase([res(f"G{i}", 1 - i / 10, f"Sentence one for {i} here. Another sentence about {i} there.")
                         for i in range(4)])
        labeler = sr.LLMLabeler(client=self.client, model=MODEL, rpm=0, tpm=0, batch_size=3)
        return sr.StanceReranker(base, labeler, sr.LabelCache(self.cache_path), sentences_per_restaurant=2)

    def go(self, chunk, reranker=None, **kw):
        return ls.run(reranker or self.reranker(), self.queries, ["Q1", "Q2"], self.ledger, daily_limit=200_000,
                      chunk=chunk, out=lambda *a: None, **kw)

    def test_stops_at_the_chunk_budget_and_resumes_without_repeating(self):
        first = self.go(chunk=40)  # each fake call reports 15 tokens; estimate 20 -> two calls fit
        self.assertEqual(first["calls"], 2)
        self.assertIn("budget reached", first["stopped"])
        self.assertEqual(self.ledger.used_24h(MODEL), 30)
        second = self.go(chunk=10_000)  # a new process: the cache file carries the progress
        self.assertIsNone(second["stopped"])
        self.assertEqual(sum(second["status"].values()), 0)
        total_sentences = first["sentences"] + second["sentences"]
        self.assertEqual(total_sentences, 16)  # 4 restaurants x 2 sentences x 2 queries, each labeled once
        third = self.go(chunk=10_000)
        self.assertEqual((third["calls"], third["tokens"]), (0, 0))  # nothing left: free rerun

    def test_dev_order_is_followed(self):
        self.go(chunk=40)
        done = {q["query"] for q in self.queries if self.client.payloads and any(p["need"] == q["query"] for p in self.client.payloads)}
        self.assertEqual(done, {"peanut allergy"})  # the first query in `order` is labeled before the second

    def test_dry_run_makes_no_call(self):
        result = self.go(chunk=10_000, dry_run=True)
        self.assertEqual((result["calls"], self.client.payloads), (0, []))
        self.assertEqual(result["status"], {"Q1": 8, "Q2": 8})

    def test_no_room_left_means_no_call(self):
        self.ledger.add(MODEL, 140_000, kind="estimate")
        result = self.go(chunk=10_000)
        self.assertEqual(result["calls"], 0)
        self.assertIn("budget reached", result["stopped"])


if __name__ == "__main__":
    unittest.main()
