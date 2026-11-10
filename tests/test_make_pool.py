"""Judging-pool builder tests with fake retrievers (no data, indexes, model or network)."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from eval import make_pool
from src.schemas import RetrievalResult, Snippet

INFO = {f"G{i}": {"name": f"Restaurant {i}", "borough": "Brooklyn" if i % 2 else "Manhattan"} for i in range(20)}
COL = {c: i for i, c in enumerate(make_pool.COLUMNS)}


def snip(review_id, text):
    return Snippet(text=text, review_id=review_id, time=None)


def result(gid, score, snippets=()):
    """snippets: [(review_id, text)]"""
    return RetrievalResult(gmap_id=gid, score=score, method="fake", evidence_snippets=[snip(i, t) for i, t in snippets])


class FakeDensityRetriever:
    """Chunk-style retriever: results depend on density_mode; records every call."""

    def __init__(self, by_mode):
        self.by_mode = by_mode  # density_mode -> query -> [RetrievalResult]
        self.calls = []

    def search(self, query, boroughs=None, top_k=10, include_low_evidence=False, density_mode=None):
        self.calls.append({"query": query, "boroughs": boroughs, "top_k": top_k, "density_mode": density_mode})
        return self.by_mode[density_mode].get(query, [])[:top_k]

    def info(self, gmap_id):
        return INFO[gmap_id]


class FakeDocRetriever:
    """Doc-TF-IDF-style retriever: search() takes no density_mode (passing one raises TypeError)."""

    def __init__(self, by_query):
        self.by_query = by_query
        self.calls = []

    def search(self, query, boroughs=None, top_k=10, include_low_evidence=False):
        self.calls.append({"query": query, "boroughs": boroughs, "top_k": top_k})
        return self.by_query.get(query, [])[:top_k]

    def info(self, gmap_id):
        return INFO[gmap_id]


class FakeFallback:
    def __init__(self, by_gid):
        self.by_gid, self.calls = by_gid, []

    def __call__(self, query, gmap_id):
        self.calls.append((query, gmap_id))
        return list(self.by_gid.get(gmap_id, []))


QUERIES = [{"query_id": "Q1", "query": "vegan", "borough": "Brooklyn"},
           {"query_id": "Q2", "query": "celiac", "borough": ""}]


def fixture():
    chunk = FakeDensityRetriever({
        "log": {"vegan": [result("G1", 0.9, [(101, "vegan A"), (102, "vegan B")]), result("G2", 0.8),
                          result("G3", 0.7, [(301, "C one")])],
                "celiac": [result("G5", 0.6, [(501, "celiac")]), result("G10", 0.5)]},
        "capped": {"vegan": [result("G1", 0.85, [(101, "vegan A"), (102, "vegan B")]),
                             result("G4", 0.75, [(401, "capped only")])],
                   "celiac": [result("G5", 0.6, [(501, "celiac")]), result("G10", 0.5)]},
    })
    doc = FakeDocRetriever({"vegan": [result("G2", 0.05, [(201, "doc snippet for G2")]),
                                      result("G3", 0.04, [(302, "doc C two"), (301, "C one (doc cut)")])],
                            "celiac": [result("G6", 0.1, [(601, "doc celiac")]), result("G11", 0.09)]})
    sem = FakeDensityRetriever({
        "log": {"vegan": [result("G7", 2.0, [(701, "dense G7")]),
                          result("G1", 1.9, [(102, "vegan B (dense cut)"), (103, "vegan C")])], "celiac": []},
        "capped": {"vegan": [result("G7", 1.8, [(701, "dense G7")]), result("G8", 1.7, [(801, "dense G8")])],
                   "celiac": [result("G9", 1.0, [(901, "dense G9")])]},
    })
    return {"chunk_tfidf": chunk, "doc_tfidf": doc, "semantic": sem}


class BuildPoolTests(unittest.TestCase):
    def setUp(self):
        self.retrievers = fixture()
        self.fallback = FakeFallback({"G10": [snip(1001, "fallback celiac mention")]})
        self.pool, self.runs, self.finders = make_pool.build_pool(QUERIES, self.retrievers, fallback=self.fallback)
        self.rows = {(r[COL["query_id"]], r[COL["gmap_id"]]): r for r in self.pool}

    def evidence(self, qid, gid):
        r = self.rows[(qid, gid)]
        return [c for c in r[COL["evidence_1"]:COL["label"]] if c]

    def test_one_row_per_query_restaurant_pair(self):
        self.assertEqual(len(self.pool), len(self.rows))
        self.assertEqual({g for q, g in self.rows if q == "Q1"}, {"G1", "G2", "G3", "G4", "G7", "G8"})
        self.assertEqual({g for q, g in self.rows if q == "Q2"}, {"G5", "G6", "G9", "G10", "G11"})
        for r in self.pool:
            self.assertEqual(len(r), len(make_pool.COLUMNS))
            self.assertEqual(r[COL["label"]], "")
            self.assertEqual(r[COL["query"]], {"Q1": "vegan", "Q2": "celiac"}[r[COL["query_id"]]])

    def test_pool_is_blind(self):
        system_names = {name for name, _, _ in make_pool.SYSTEMS} | set(make_pool.RETRIEVERS) | {"fake"}
        for row in self.pool:
            for cell in row:
                self.assertFalse(any(name in cell for name in system_names), cell)
        for qid in ("Q1", "Q2"):
            gids = [r[COL["gmap_id"]] for r in self.pool if r[COL["query_id"]] == qid]
            self.assertEqual(gids, sorted(gids, key=lambda g: make_pool.shuffle_key(qid, g)))
        self.assertNotIn("system", make_pool.COLUMNS)

    def test_evidence_is_the_union_across_systems_deduplicated_by_review(self):
        # 102 came from chunk TF-IDF (both variants) and semantic: 2 retrievers, so first; the
        # chunk excerpt is shown. 101 (chunk only, position 0) beats 103 (semantic, position 1).
        self.assertEqual(self.evidence("Q1", "G1"),
                         ["[review 102, n/a] vegan B", "[review 101, n/a] vegan A", "[review 103, n/a] vegan C"])
        self.assertEqual(self.rows[("Q1", "G1")][COL["evidence_review_ids"]], "102;101;103")
        self.assertEqual(self.evidence("Q1", "G3"), ["[review 301, n/a] C one", "[review 302, n/a] doc C two"])
        self.assertEqual(self.evidence("Q1", "G2"), ["[review 201, n/a] doc snippet for G2"])
        self.assertEqual(self.rows[("Q1", "G2")][COL["name"]:COL["evidence_status"]], ["Restaurant 2", "Manhattan"])

    def test_variants_of_one_retriever_count_once(self):
        hits = [("chunk_tfidf", 0, [snip(1, "a")]), ("chunk_tfidf", 1, [snip(1, "a")]),
                ("semantic", 3, [snip(2, "b")]), ("doc_tfidf", 2, [snip(2, "b doc")])]
        self.assertEqual([s.review_id for s in make_pool.pool_evidence(hits)], [2, 1])
        self.assertEqual(make_pool.pool_evidence(hits)[0].text, "b doc")  # earliest system's excerpt

    def test_evidence_is_capped(self):
        hits = [("chunk_tfidf", 0, [snip(i, f"r{i}") for i in range(5)]),
                ("semantic", 3, [snip(i, f"s{i}") for i in range(5, 10)])]
        evidence = make_pool.pool_evidence(hits)
        self.assertEqual(len(evidence), make_pool.MAX_EVIDENCE)
        self.assertEqual([s.review_id for s in evidence], [0, 5, 1, 6, 2, 7])  # by position, then system

    def test_evidence_status_and_fallback(self):
        status = {k: r[COL["evidence_status"]] for k, r in self.rows.items()}
        self.assertEqual(status[("Q1", "G1")], "retrieved")
        self.assertEqual(status[("Q2", "G10")], "fallback")
        self.assertEqual(self.evidence("Q2", "G10"), ["[review 1001, n/a] fallback celiac mention"])
        self.assertEqual(status[("Q2", "G11")], "none")
        self.assertEqual(self.evidence("Q2", "G11"), [])
        self.assertEqual(self.rows[("Q2", "G11")][COL["evidence_review_ids"]], "")
        self.assertEqual(sorted(self.fallback.calls), [("celiac", "G10"), ("celiac", "G11")])  # only empty pairs

    def test_fallback_is_limited_to_three_reviews(self):
        fb = FakeFallback({"G10": [snip(i, f"f{i}") for i in range(5)]})
        pool, _, _ = make_pool.build_pool(QUERIES[1:], fixture(), fallback=fb)
        row = next(r for r in pool if r[COL["gmap_id"]] == "G10")
        self.assertEqual(row[COL["evidence_review_ids"]], "0;1;2")

    def test_no_fallback_means_none(self):
        pool, _, _ = make_pool.build_pool(QUERIES[1:], fixture())
        row = next(r for r in pool if r[COL["gmap_id"]] == "G10")
        self.assertEqual(row[COL["evidence_status"]], "none")

    def test_runs_hold_every_systems_full_ranking(self):
        self.assertEqual(len(self.runs), (3 + 2 + 2 + 2 + 2) + (2 + 2 + 2 + 0 + 1))
        by_system = {}
        for qid, system, rank, gid, score in self.runs:
            by_system.setdefault((qid, system), []).append((rank, gid, float(score)))
        self.assertEqual(by_system[("Q1", "chunk_tfidf_log")], [(1, "G1", 0.9), (2, "G2", 0.8), (3, "G3", 0.7)])
        self.assertEqual(by_system[("Q1", "semantic_capped")], [(1, "G7", 1.8), (2, "G8", 1.7)])
        self.assertNotIn(("Q2", "semantic_log"), by_system)  # no results -> no rows
        self.assertEqual({s for _, s in by_system}, {n for n, _, _ in make_pool.SYSTEMS})

    def test_scores_round_trip_exactly(self):
        score = 0.1 + 0.2  # 0.30000000000000004
        r = {"chunk_tfidf": FakeDensityRetriever({"log": {"x": [result("G1", score)]}, "capped": {}})}
        _, runs, _ = make_pool.build_pool([{"query_id": "Q", "query": "x", "borough": ""}], r)
        self.assertEqual(float(runs[0][4]), score)

    def test_density_modes_and_borough_are_passed(self):
        self.assertEqual([c["density_mode"] for c in self.retrievers["chunk_tfidf"].calls], ["log", "capped"] * 2)
        self.assertEqual([c["density_mode"] for c in self.retrievers["semantic"].calls], ["log", "capped"] * 2)
        self.assertEqual(len(self.retrievers["doc_tfidf"].calls), 2)  # called without density_mode
        for r in self.retrievers.values():
            self.assertEqual([c["boroughs"] for c in r.calls][::len(r.calls) // 2], [["Brooklyn"], None])
            self.assertTrue(all(c["top_k"] == make_pool.TOP_K for c in r.calls))

    def test_finders_track_which_systems_found_each_pair(self):
        self.assertEqual(self.finders[("Q1", "G1")], {"chunk_tfidf_log", "chunk_tfidf_capped", "semantic_log"})
        self.assertEqual(self.finders[("Q1", "G8")], {"semantic_capped"})
        self.assertEqual(self.finders[("Q2", "G9")], {"semantic_capped"})

    def test_missing_retriever_skips_its_systems(self):
        retrievers = fixture()
        retrievers["semantic"] = None
        pool, runs, _ = make_pool.build_pool(QUERIES, retrievers)
        self.assertFalse(any(r[1].startswith("semantic") for r in runs))
        self.assertNotIn("G7", {r[COL["gmap_id"]] for r in pool})


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.queries = self.dir / "frozen.csv"
        with open(self.queries, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["query_id", "query", "category", "borough"])
            w.writeheader()
            w.writerows([{**q, "category": "x"} for q in QUERIES])

    def run_main(self, fakes=None):
        fakes = fakes or fixture()
        loaders = {name: (lambda r=r: r) for name, r in fakes.items()}
        out = io.StringIO()
        with patch.dict(make_pool.RETRIEVERS, loaders), redirect_stdout(out):
            make_pool.main(["--queries", str(self.queries), "--pool", str(self.dir / "p" / "pool.csv"),
                            "--runs", str(self.dir / "p" / "runs.csv")])
        return out.getvalue()

    def test_writes_pool_and_runs_to_given_paths(self):
        fakes = fixture()
        fakes["doc_tfidf"].evidence_for = lambda gid, query, n: [snip(1001, "fallback")] if gid == "G10" else []
        out = self.run_main(fakes)
        with open(self.dir / "p" / "pool.csv", newline="") as f:
            pool = list(csv.DictReader(f))
        with open(self.dir / "p" / "runs.csv", newline="") as f:
            runs = list(csv.reader(f))
        self.assertEqual(list(pool[0]), make_pool.COLUMNS)
        self.assertEqual(runs[0], make_pool.RUN_COLUMNS)
        self.assertEqual(len(pool), 11)
        self.assertEqual(len(runs) - 1, 18)
        self.assertEqual({r["gmap_id"]: r["evidence_status"] for r in pool}["G10"], "fallback")  # doc's evidence_for
        self.assertIn("Pool: 11 pairs; per query min 5, median 5.5, max 6", out)
        self.assertIn("Pairs found only by *_capped systems: 3", out)  # G4, G8, G9
        self.assertIn("'none': 1", out)

    def test_queries_argument_is_validated(self):
        with open(self.queries, "w") as f:
            f.write("query_id,query\nQ1,vegan\n")
        with self.assertRaisesRegex(ValueError, "missing column"):
            self.run_main()
        with open(self.queries, "w") as f:
            f.write("query_id,query,borough\nQ1,vegan,\nQ1,celiac,\n")
        with self.assertRaisesRegex(ValueError, "duplicate query_id"):
            self.run_main()

    def test_default_paths(self):
        self.assertEqual(make_pool.QUERIES_PATH.name, "queries_draft.csv")
        self.assertEqual(make_pool.POOL_PATH.name, "pool.csv")
        self.assertEqual(make_pool.RUNS_PATH.name, "runs.csv")


if __name__ == "__main__":
    unittest.main()
