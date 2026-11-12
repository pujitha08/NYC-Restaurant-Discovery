"""Metric tests on tiny synthetic runs and labels with hand-computed values."""

import csv
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from eval import metrics

LOG2_3, LOG2_5 = math.log2(3), math.log2(5)
# Q1 pool: A=2, B=1, C=0, D=2, E=1  -> ideal gains [2, 2, 1, 1, 0]
IDCG_Q1 = 2 / 1 + 2 / LOG2_3 + 1 / 2 + 1 / LOG2_5  # = 4.192537...
NDCG_S1_Q1 = (2 + 1 / 2) / IDCG_Q1  # 0.596298...
NDCG_S2_Q1 = (2 + 2 / LOG2_3 + 1 / 2) / IDCG_Q1  # 0.897275...
LABELS = {("Q1", "A"): 2, ("Q1", "B"): 1, ("Q1", "C"): 0, ("Q1", "D"): 2, ("Q1", "E"): 1,
          ("Q2", "X"): 0, ("Q2", "Y"): 0,  # Q2: no relevant pair in the pool
          ("Q3", "P"): 2}
RUNS = {"s1": {"Q1": ["A", "C", "B", "Z"], "Q2": ["X"], "Q3": ["P"]},  # Z is unjudged
        "s2": {"Q1": ["D", "A", "E"], "Q2": ["Y"], "Q3": []}}
CATEGORIES = {"Q1": "allergy", "Q2": "single_diet", "Q3": "allergy"}


def by(rows, system, query):
    return next(r for r in rows if r["system"] == system and r["query_id"] == query)


class QueryMetricTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.queries = metrics.evaluate(RUNS, LABELS, CATEGORIES)

    def test_s1_q1_by_hand(self):
        r = by(self.rows, "s1", "Q1")  # gains [2, 0, 1, 0(unjudged)]
        self.assertAlmostEqual(r["P5_strict"], 1 / 5)
        self.assertAlmostEqual(r["P5_lenient"], 2 / 5)
        self.assertAlmostEqual(r["P10_strict"], 1 / 10)
        self.assertAlmostEqual(r["P10_lenient"], 2 / 10)
        self.assertAlmostEqual(r["NDCG10"], (2 + 1 / 2) / IDCG_Q1)  # 0.596298
        self.assertAlmostEqual(r["NDCG10"], 0.596298, places=5)
        self.assertAlmostEqual(r["R10_strict"], 1 / 2)
        self.assertAlmostEqual(r["R10_lenient"], 2 / 4)
        self.assertEqual((r["unjudged"], r["n_results"], r["pool_rel_strict"], r["pool_rel_lenient"]), (1, 4, 2, 4))

    def test_s2_q1_by_hand(self):
        r = by(self.rows, "s2", "Q1")  # gains [2, 2, 1]
        self.assertAlmostEqual(r["P5_strict"], 2 / 5)
        self.assertAlmostEqual(r["P10_lenient"], 3 / 10)
        self.assertAlmostEqual(r["NDCG10"], (2 + 2 / LOG2_3 + 1 / 2) / IDCG_Q1)  # 0.897276
        self.assertAlmostEqual(r["NDCG10"], 0.897275, places=5)
        self.assertEqual((r["R10_strict"], r["R10_lenient"], r["unjudged"]), (1.0, 0.75, 0))

    def test_query_with_no_relevant_results(self):
        for s in ("s1", "s2"):
            r = by(self.rows, s, "Q2")
            self.assertEqual((r["P5_lenient"], r["P10_strict"]), (0.0, 0.0))
            self.assertIsNone(r["NDCG10"])  # IDCG = 0: undefined, excluded from the mean
            self.assertIsNone(r["R10_lenient"])

    def test_empty_ranking_scores_zero(self):
        r = by(self.rows, "s2", "Q3")
        self.assertEqual((r["NDCG10"], r["R10_strict"], r["P10_strict"], r["n_results"]), (0.0, 0.0, 0.0, 0))
        self.assertEqual(by(self.rows, "s1", "Q3")["NDCG10"], 1.0)

    def test_means_skip_undefined_queries(self):
        s = metrics.summarize(self.rows)
        self.assertAlmostEqual(s["s1"]["NDCG10"], (NDCG_S1_Q1 + 1.0) / 2)  # Q2 excluded
        self.assertAlmostEqual(s["s2"]["NDCG10"], (NDCG_S2_Q1 + 0.0) / 2)
        self.assertAlmostEqual(s["s1"]["P10_lenient"], (0.2 + 0 + 0.1) / 3)  # P@k over all queries
        self.assertAlmostEqual(s["s2"]["P10_lenient"], (0.3 + 0 + 0) / 3)
        self.assertEqual((s["s1"]["unjudged"], s["s2"]["unjudged"]), (1, 0))

    def test_ranks_ties_and_cutoff(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "runs.csv"
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["query_id", "system", "rank", "gmap_id", "score"])
                w.writerows([["Q", "s", "2", "C", "0.5"], ["Q", "s", "1", "B", "0.9"], ["Q", "s", "1", "A", "0.9"]])
            self.assertEqual(metrics.read_runs(p)["s"]["Q"], ["A", "B", "C"])  # rank, then gmap_id
        ranked = [f"G{i}" for i in range(12)]
        m = metrics.query_metrics(ranked, {g: 2 for g in ranked})
        self.assertEqual((m["P10_strict"], m["NDCG10"], m["R10_strict"], m["n_results"]), (1.0, 1.0, 10 / 12, 10))

    def test_dcg(self):
        self.assertAlmostEqual(metrics.dcg([2, 1, 0, 1]), 2 + 1 / LOG2_3 + 0 + 1 / LOG2_5)


class SignificanceTests(unittest.TestCase):
    def test_randomization_matches_exact_sign_flip_p(self):
        # All five differences positive: exact two-sided p = 2 / 2^5 = 0.0625.
        self.assertAlmostEqual(metrics.randomization_p([0.1] * 5, n=20_000), 0.0625, delta=0.01)
        self.assertEqual(metrics.randomization_p([0.0, 0.0, 0.0]), 1.0)
        self.assertIsNone(metrics.randomization_p([]))
        self.assertEqual(metrics.randomization_p([0.3, -0.1, 0.2], seed=1), metrics.randomization_p([0.3, -0.1, 0.2], seed=1))

    def test_bootstrap_ci(self):
        lo, hi = metrics.bootstrap_ci([0.4, 0.4, 0.4])
        self.assertAlmostEqual(lo, 0.4)
        self.assertAlmostEqual(hi, 0.4)
        lo, hi = metrics.bootstrap_ci([0.0, 1.0] * 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertEqual(metrics.bootstrap_ci([None]), (None, None))

    def test_holm(self):
        adjusted = metrics.holm([0.01, 0.04, 0.03, None])
        for got, want in zip(adjusted[:3], [0.03, 0.06, 0.06]):
            self.assertAlmostEqual(got, want)
        self.assertIsNone(adjusted[3])

    def test_compare_is_paired_on_defined_queries(self):
        rows, _ = metrics.evaluate(RUNS, LABELS, CATEGORIES)
        c = metrics.compare(rows, "s1", "s2")
        self.assertEqual(c["n"], 2)  # Q1 and Q3; Q2 is undefined for both
        self.assertAlmostEqual(c["diff"], ((NDCG_S2_Q1 - NDCG_S1_Q1) + (0.0 - 1.0)) / 2)
        self.assertEqual((c["wins"], c["losses"], c["ties"]), (1, 1, 0))
        self.assertEqual([q for q, *_ in c["per_query"]], ["Q1", "Q3"])


class LabelInputTests(unittest.TestCase):
    def test_jsonl_and_csv(self):
        with tempfile.TemporaryDirectory() as d:
            j = Path(d) / "llm.jsonl"
            j.write_text(json.dumps({"query_id": "Q", "gmap_id": "A", "label": 1}) + "\n"
                         + json.dumps({"query_id": "Q", "gmap_id": "A", "label": 2}) + "\n"
                         + '{"query_id": "Q", "gm')  # later line wins; partial line ignored
            self.assertEqual(metrics.read_labels(j), {("Q", "A"): 2})
            c = Path(d) / "adjudicated.csv"
            c.write_text("sample_id,query_id,gmap_id,label_x,adjudicated_label\nS1,Q,A,0,2\nS2,Q,B,1,\n")
            self.assertEqual(metrics.read_labels(c), {("Q", "A"): 2})  # blank = not adjudicated, skipped
            c.write_text("query_id,gmap_id,adjudicated_label\nQ,A,3\n")
            with self.assertRaisesRegex(ValueError, "label must be 0, 1 or 2"):
                metrics.read_labels(c)


class MainTests(unittest.TestCase):
    def test_provisional_marks_every_output(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with open(d / "runs.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["query_id", "system", "rank", "gmap_id", "score"])
                for q, gids in RUNS["s1"].items():
                    for rank, g in enumerate(gids, 1):
                        w.writerow([q, "semantic_log", rank, g, 1.0])
            with open(d / "labels.jsonl", "w") as f:
                for (q, g), l in LABELS.items():
                    f.write(json.dumps({"query_id": q, "gmap_id": g, "label": l}) + "\n")
            with open(d / "queries.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["query_id", "query", "category", "borough"])
                w.writerows([[q, q, c, ""] for q, c in CATEGORIES.items()])
            out = d / "reports"
            reason = "LLM labels pending validation against human labels"
            with redirect_stdout(io.StringIO()) as stdout:
                metrics.main(["--runs", str(d / "runs.csv"), "--labels", str(d / "labels.jsonl"),
                              "--queries", str(d / "queries.csv"), "--out-dir", str(out), "--resamples", "200",
                              "--provisional", reason])
            summary = (out / "metrics_summary.md").read_text()
            self.assertTrue(summary.startswith("# Retrieval metrics (PROVISIONAL)"))
            self.assertIn(f"> **PROVISIONAL:** {reason}", summary)
            self.assertIn("P@10 lenient [95% CI]", summary)
            with open(out / "metrics_per_query.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r["status"] for r in rows}, {"PROVISIONAL"})
            self.assertIn("PROVISIONAL", stdout.getvalue())
            self.assertTrue((out / "metrics_ndcg_p10.png").exists())

    def test_complete_only_drops_queries_with_unjudged_results(self):
        rows, _ = metrics.evaluate(RUNS, LABELS, CATEGORIES)
        kept, queries, dropped = metrics.complete_only(rows)
        self.assertEqual(dropped, ["Q1"])  # s1 has unjudged Z in Q1
        self.assertEqual(queries, ["Q2", "Q3"])
        self.assertTrue(all(r["unjudged"] == 0 for r in kept))

    def test_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with open(d / "runs.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["query_id", "system", "rank", "gmap_id", "score"])
                for system in ("chunk_tfidf_log", "chunk_tfidf_capped", "doc_tfidf"):
                    src = RUNS["s1"] if system != "doc_tfidf" else RUNS["s2"]
                    order = {"Q1": ["A", "B", "C"]} if system == "chunk_tfidf_capped" else {}
                    for q, gids in {**src, **order}.items():
                        for rank, g in enumerate(gids, 1):
                            w.writerow([q, system, rank, g, 1.0 / rank])
            with open(d / "labels.jsonl", "w") as f:
                for (q, g), l in LABELS.items():
                    f.write(json.dumps({"query_id": q, "gmap_id": g, "label": l}) + "\n")
            with open(d / "queries.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["query_id", "query", "category", "borough"])
                w.writerows([[q, q, c, ""] for q, c in CATEGORIES.items()])
            out = d / "reports"
            with redirect_stdout(io.StringIO()) as stdout:
                metrics.main(["--runs", str(d / "runs.csv"), "--labels", str(d / "labels.jsonl"),
                              "--queries", str(d / "queries.csv"), "--out-dir", str(out), "--resamples", "500"])
            summary = (out / "metrics_summary.md").read_text()
            self.assertIn(metrics.GAIN_FORMULA, summary)
            self.assertIn("Warning: unjudged results", summary)  # Z in chunk_tfidf_log Q1
            self.assertIn("left out of NDCG@10 and R@10): Q2", summary)
            self.assertIn("### chunk_tfidf_log → chunk_tfidf_capped", summary)
            self.assertIn("| allergy (n=2) |", summary)
            self.assertIn("WARNING: 1 unjudged", stdout.getvalue())
            with open(out / "metrics_per_query.csv", newline="") as f:
                per_query = list(csv.DictReader(f))
            self.assertEqual(len(per_query), 3 * 3)
            q2 = next(r for r in per_query if r["query_id"] == "Q2" and r["system"] == "doc_tfidf")
            self.assertEqual(q2["NDCG10"], "")  # undefined
            png = (out / "metrics_ndcg_p10.png").read_bytes()
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
