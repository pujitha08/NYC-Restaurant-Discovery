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


def result(gid, score, snippets=()):
    return RetrievalResult(gmap_id=gid, score=score, method="fake",
                           evidence_snippets=[Snippet(text=t, review_id=k, time=None) for k, t in enumerate(snippets)])


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


QUERIES = [{"query_id": "Q1", "query": "vegan", "borough": "Brooklyn"},
           {"query_id": "Q2", "query": "celiac", "borough": ""}]


def fixture():
    chunk = FakeDensityRetriever({
        "log": {"vegan": [result("G1", 0.9, ["chunk vegan A"]), result("G2", 0.8, []), result("G3", 0.7, ["chunk C"])],
                "celiac": [result("G5", 0.6, ["chunk celiac"])]},
        "capped": {"vegan": [result("G1", 0.85, ["chunk vegan A"]), result("G4", 0.75, ["capped only"])],
                   "celiac": [result("G5", 0.6, ["chunk celiac"])]},
    })
    doc = FakeDocRetriever({"vegan": [result("G2", 0.05, ["doc snippet for G2"]), result("G3", 0.04, ["doc C"])],
                            "celiac": [result("G6", 0.1, ["doc celiac"])]})
    sem = FakeDensityRetriever({
        "log": {"vegan": [result("G7", 2.0, ["dense G7"]), result("G1", 1.9, ["dense G1"])], "celiac": []},
        "capped": {"vegan": [result("G7", 1.8, ["dense G7"]), result("G8", 1.7, ["dense G8"])],
                   "celiac": [result("G9", 1.0, ["dense G9"])]},
    })
    return {"chunk_tfidf": chunk, "doc_tfidf": doc, "semantic": sem}


class BuildPoolTests(unittest.TestCase):
    def setUp(self):
        self.retrievers = fixture()
        self.pool, self.runs, self.finders = make_pool.build_pool(QUERIES, self.retrievers)

    def test_one_row_per_query_restaurant_pair(self):
        pairs = [(r[0], r[1]) for r in self.pool]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual({g for q, g in pairs if q == "Q1"}, {"G1", "G2", "G3", "G4", "G7", "G8"})
        self.assertEqual({g for q, g in pairs if q == "Q2"}, {"G5", "G6", "G9"})
        self.assertTrue(all(len(r) == len(make_pool.COLUMNS) and r[-1] == "" for r in self.pool))

    def test_pool_is_blind(self):
        system_names = {name for name, _, _ in make_pool.SYSTEMS} | set(make_pool.RETRIEVERS) | {"fake"}
        for row in self.pool:
            for cell in row:
                self.assertFalse(any(name in cell for name in system_names), cell)
        for qid in ("Q1", "Q2"):
            gids = [r[1] for r in self.pool if r[0] == qid]
            self.assertEqual(gids, sorted(gids, key=lambda g: make_pool.shuffle_key(qid, g)))
        self.assertNotIn("system", make_pool.COLUMNS)

    def test_runs_hold_every_systems_full_ranking(self):
        self.assertEqual(len(self.runs), 3 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 0 + 1)
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
        chunk_modes = [c["density_mode"] for c in self.retrievers["chunk_tfidf"].calls]
        self.assertEqual(chunk_modes, ["log", "capped", "log", "capped"])
        self.assertEqual([c["density_mode"] for c in self.retrievers["semantic"].calls], ["log", "capped", "log", "capped"])
        self.assertEqual(len(self.retrievers["doc_tfidf"].calls), 2)  # called without density_mode
        for r in self.retrievers.values():
            self.assertEqual([c["boroughs"] for c in r.calls][::len(r.calls) // 2], [["Brooklyn"], None])
            self.assertTrue(all(c["top_k"] == make_pool.TOP_K for c in r.calls))

    def test_snippets_from_first_system_that_has_any(self):
        rows = {(r[0], r[1]): r for r in self.pool}
        self.assertEqual(rows[("Q1", "G1")][4], "[review 0, n/a] chunk vegan A")  # chunk beats semantic
        self.assertEqual(rows[("Q1", "G2")][4], "[review 0, n/a] doc snippet for G2")  # chunk had none
        self.assertEqual(rows[("Q1", "G4")][4], "[review 0, n/a] capped only")
        self.assertEqual(rows[("Q1", "G3")][5:7], ["", ""])  # padded to N_SNIPPETS
        self.assertEqual(rows[("Q1", "G2")][2:4], ["Restaurant 2", "Manhattan"])

    def test_finders_track_which_systems_found_each_pair(self):
        self.assertEqual(self.finders[("Q1", "G1")], {"chunk_tfidf_log", "chunk_tfidf_capped", "semantic_log"})
        self.assertEqual(self.finders[("Q1", "G8")], {"semantic_capped"})
        self.assertEqual(self.finders[("Q2", "G9")], {"semantic_capped"})

    def test_missing_retriever_skips_its_systems(self):
        retrievers = fixture()
        retrievers["semantic"] = None
        pool, runs, _ = make_pool.build_pool(QUERIES, retrievers)
        self.assertFalse(any(r[1].startswith("semantic") for r in runs))
        self.assertNotIn("G7", {r[1] for r in pool})


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

    def run_main(self, *extra):
        fakes = fixture()
        loaders = {name: (lambda r=r: r) for name, r in fakes.items()}
        out = io.StringIO()
        with patch.dict(make_pool.RETRIEVERS, loaders), redirect_stdout(out):
            make_pool.main(["--queries", str(self.queries), "--pool", str(self.dir / "p" / "pool.csv"),
                            "--runs", str(self.dir / "p" / "runs.csv"), *extra])
        return out.getvalue()

    def test_writes_pool_and_runs_to_given_paths(self):
        out = self.run_main()
        with open(self.dir / "p" / "pool.csv", newline="") as f:
            pool = list(csv.reader(f))
        with open(self.dir / "p" / "runs.csv", newline="") as f:
            runs = list(csv.reader(f))
        self.assertEqual(pool[0], make_pool.COLUMNS)
        self.assertEqual(runs[0], make_pool.RUN_COLUMNS)
        self.assertEqual(len(pool) - 1, 9)
        self.assertEqual(len(runs) - 1, 15)
        self.assertIn("Pool: 9 pairs; per query min 3, median 4.5, max 6", out)
        self.assertIn("Pairs found only by *_capped systems: 3", out)  # G4, G8, G9

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
