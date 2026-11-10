"""Stratified human-sample tests on a synthetic pool (no data or network)."""

import csv
import io
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

from eval import make_pool, make_sample

# query_id -> category; 12 queries, pairs per query below
CATEGORIES = {**{f"Q{i:02d}": "single_diet" for i in range(1, 7)}, "Q07": "mixed_group", "Q08": "mixed_group",
              "Q09": "allergy", "Q10": "allergy", "Q11": "allergy", "Q12": "diet_plus_filter"}


def synthetic(n_per_query=25, none_every=17, single_every=4):
    """(pool rows, runs rows, query rows); some pairs have no evidence, some one finder."""
    pool, runs, queries = [], [], []
    k = 0
    for qid, cat in CATEGORIES.items():
        queries.append({"query_id": qid, "query": f"query {qid}", "category": cat, "borough": ""})
        for j in range(n_per_query):
            k += 1
            gid = f"{qid}-G{j}"
            status = "none" if k % none_every == 0 else "retrieved"
            row = {c: "" for c in make_pool.COLUMNS}
            row.update(query_id=qid, query=f"query {qid}", gmap_id=gid, name=f"R {gid}", borough="Brooklyn",
                       evidence_status=status, evidence_1="" if status == "none" else f"[review {k}, 2020-01] text {k}")
            pool.append(row)
            systems = ["doc_tfidf"] if k % single_every == 0 else ["doc_tfidf", "semantic_log"]
            runs += [{"query_id": qid, "system": s, "rank": "1", "gmap_id": gid, "score": "1.0"} for s in systems]
    return pool, runs, queries


class SelectTests(unittest.TestCase):
    def setUp(self):
        pool, runs, queries = synthetic()
        self.pairs = make_sample.annotate(pool, runs, queries)

    def chosen(self, **kw):
        return [self.pairs[i] for i in make_sample.select(self.pairs, kw.pop("n", 120), **kw)]

    def test_size_uniqueness_and_determinism(self):
        a = make_sample.select(self.pairs, 120, seed=1)
        self.assertEqual(len(a), 120)
        self.assertEqual(len(set(a)), 120)
        self.assertEqual(a, make_sample.select(self.pairs, 120, seed=1))
        self.assertNotEqual(sorted(a), sorted(make_sample.select(self.pairs, 120, seed=2)))

    def test_every_query_is_covered(self):
        self.assertEqual({p["query_id"] for p in self.chosen()}, set(CATEGORIES))

    def test_quotas_are_met(self):
        s = self.chosen(min_none=10, min_allergy=40, min_single=35)
        self.assertGreaterEqual(sum(p["evidence_status"] == "none" for p in s), 10)
        self.assertGreaterEqual(sum(p["category"] == "allergy" for p in s), 40)
        self.assertGreaterEqual(sum(p["single_system"] for p in s), 35)

    def test_quota_takes_all_when_the_pool_has_fewer(self):
        n_none = sum(p["evidence_status"] == "none" for p in self.pairs)
        s = self.chosen(min_none=n_none + 50, min_allergy=0, min_single=0)
        self.assertEqual(sum(p["evidence_status"] == "none" for p in s), n_none)

    def test_quota_picks_rotate_across_queries(self):
        s = self.chosen(min_allergy=45, min_none=0, min_single=0)
        per_allergy_query = Counter(p["query_id"] for p in s if p["category"] == "allergy")
        self.assertEqual(set(per_allergy_query), {"Q09", "Q10", "Q11"})
        self.assertLessEqual(max(per_allergy_query.values()) - min(per_allergy_query.values()), 1)

    def test_remaining_slots_follow_category_shares(self):
        s = Counter(p["category"] for p in self.chosen(min_none=0, min_allergy=0, min_single=0))
        self.assertGreater(s["single_diet"], s["mixed_group"])  # 6 of 12 queries vs 2
        self.assertAlmostEqual(s["single_diet"], 60, delta=3)

    def test_sample_larger_than_pool(self):
        self.assertEqual(len(make_sample.select(self.pairs[:30], 120)), 30)


class MainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        pool, runs, queries = synthetic()
        for name, rows in (("pool.csv", pool), ("runs.csv", runs), ("queries.csv", queries)):
            with open(self.dir / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        self.out = self.dir / "labels"

    def run_main(self, *extra):
        argv = ["--pool", str(self.dir / "pool.csv"), "--runs", str(self.dir / "runs.csv"),
                "--queries", str(self.dir / "queries.csv"), "--out-dir", str(self.out), *extra]
        with redirect_stdout(io.StringIO()) as out:
            make_sample.main(argv)
        return out.getvalue()

    def read(self, name):
        with open(self.out / name, newline="") as f:
            return list(csv.DictReader(f))

    def test_judge_files_have_same_pairs_in_different_orders(self):
        self.run_main("--judges", "ana", "ben", "cai", "--n", "60")
        files = {j: self.read(f"labels_{j}.csv") for j in ("ana", "ben", "cai")}
        ids = {j: [r["sample_id"] for r in rows] for j, rows in files.items()}
        self.assertTrue(all(len(v) == 60 for v in ids.values()))
        self.assertEqual(set(ids["ana"]), set(ids["ben"]))
        self.assertEqual(set(ids["ana"]), set(ids["cai"]))
        self.assertNotEqual(ids["ana"], ids["ben"])
        self.assertNotEqual(ids["ben"], ids["cai"])
        for rows in files.values():
            self.assertEqual(list(rows[0]), make_sample.JUDGE_COLUMNS)
            self.assertTrue(all(r["label"] == "" and r["note"] == "" for r in rows))
        # same content per sample_id across judges
        by_id = {j: {r["sample_id"]: r for r in rows} for j, rows in files.items()}
        self.assertTrue(all(by_id["ana"][i] == by_id["ben"][i] for i in ids["ana"]))

    def test_judge_files_are_blind_and_key_maps_back(self):
        self.run_main("--judges", "ana", "ben", "--n", "40")
        cols = make_sample.JUDGE_COLUMNS
        self.assertNotIn("gmap_id", cols)
        self.assertFalse(any(c in cols for c in ("system", "single_system", "category", "rank", "score")))
        key = self.read("sample_key.csv")
        self.assertEqual(list(key[0]), make_sample.KEY_COLUMNS)
        self.assertEqual([r["sample_id"] for r in key], [f"S{k:03d}" for k in range(1, 41)])
        judge = {r["sample_id"]: r for r in self.read("labels_ana.csv")}
        for k in key:
            self.assertEqual(judge[k["sample_id"]]["query_id"], k["query_id"])
            self.assertEqual(judge[k["sample_id"]]["name"], f"R {k['gmap_id']}")

    def test_refuses_to_overwrite_without_force(self):
        self.run_main("--judges", "ana", "--n", "20")
        with self.assertRaises(SystemExit):
            self.run_main("--judges", "ana", "--n", "20")
        self.run_main("--judges", "ana", "--n", "20", "--force")

    def test_rejects_bad_or_duplicate_judge_names(self):
        for names in (["a b"], ["ana", "ana"], ["../x"]):
            with self.subTest(names=names), self.assertRaises(SystemExit):
                self.run_main("--judges", *names)


if __name__ == "__main__":
    unittest.main()
