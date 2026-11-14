"""Held-out tuning of the stance re-ranker: the split, the grid search and the report (synthetic features)."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from eval import tune_stance_rerank as tune
from src.retrieval.stance_rerank import STANCES, Candidate
from src.schemas import RetrievalResult


def cand(gid, base_norm, **counts):
    return Candidate(RetrievalResult(gid, base_norm, "x"), base_norm, {s: counts.get(s, 0) for s in STANCES},
                     [{"text": f"{gid} says no", "stance": "negative", "review_id": 1}] if counts.get("negative") else [])


def features(n_queries=6):
    """Each query: the warned restaurant W leads on base score; G (accommodating) and M follow."""
    return {f"Q{i:02d}": ([cand("W", 1.0, negative=3), cand("G", 0.7, accommodation=3), cand("M", 0.4, mention=3)], True)
            for i in range(1, n_queries + 1)}


def labels(qids, w=0, g=2, m=1):
    return {(q, gid): lab for q in qids for gid, lab in (("W", w), ("G", g), ("M", m))}


class SplitTests(unittest.TestCase):
    ids = [f"Q{i:02d}" for i in range(1, 36)]

    def test_20_dev_15_test_disjoint_and_complete(self):
        dev, test = tune.split_queries(self.ids)
        self.assertEqual((len(dev), len(test)), (20, 15))
        self.assertEqual(set(dev) | set(test), set(self.ids))
        self.assertFalse(set(dev) & set(test))

    def test_fixed_seed_is_deterministic_and_seed_changes_it(self):
        self.assertEqual(tune.split_queries(self.ids), tune.split_queries(list(reversed(self.ids))))
        self.assertNotEqual(tune.split_queries(self.ids)[0], tune.split_queries(self.ids, seed=1)[0])

    def test_default_seed_is_6165(self):
        self.assertEqual(tune.SEED, 6165)


class GridTests(unittest.TestCase):
    def test_zero_weights_give_the_base_order(self):
        f = features(1)
        self.assertEqual(tune.top_ids(f["Q01"][0], tune.ZERO, True), ["W", "G", "M"])

    def test_tuning_moves_the_warned_restaurant_down(self):
        f = features()
        dev = list(f)
        best, results = tune.grid_search(f, labels(dev), dev, {})
        self.assertEqual(tune.top_ids(f["Q01"][0], best, True)[0], "G")
        self.assertGreater(len(results), 100)
        base = tune.summarize_rows(tune.rows_for(f, labels(dev), dev, tune.ZERO, "b", {}))["NDCG10"]
        tuned = tune.summarize_rows(tune.rows_for(f, labels(dev), dev, best, "s", {}))["NDCG10"]
        self.assertGreater(tuned, base)

    def test_ties_prefer_smaller_weights(self):
        f = features(2)
        dev = list(f)
        flat = {(q, g): 1 for q in dev for g in "WGM"}  # no ranking can differ: every NDCG is 1
        best, _ = tune.grid_search(f, flat, dev, {})
        self.assertEqual(best, tune.ZERO)

    def test_test_queries_never_influence_the_chosen_weights(self):
        f = features(6)
        dev, test = list(f)[:4], list(f)[4:]
        a, _ = tune.grid_search(f, labels(dev + test), dev, {})
        flipped = {**labels(dev), **labels(test, w=2, g=0, m=0)}  # opposite truth on the test queries
        b, _ = tune.grid_search(f, flipped, dev, {})
        self.assertEqual(a, b)

    def test_unjudged_results_count_as_zero(self):
        f = features(1)
        rows = tune.rows_for(f, {("Q01", "G"): 2}, ["Q01"], tune.ZERO, "s", {})
        self.assertEqual(rows[0]["unjudged"], 2)  # W and M have no label


class ReportTests(unittest.TestCase):
    def test_split_table_and_report(self):
        f = features(6)
        qids = list(f)
        cats = {q: ("allergy" if i % 2 else "single_diet") for i, q in enumerate(qids)}
        lab = labels(qids)
        tbl = tune.split_table(f, lab, qids, {tune.STANCE: {"w_pos": 0.3, "w_neg": 0.9, "w_strong_neg": 0.1}}, cats, n=200)
        s = tbl[tune.STANCE]
        self.assertGreater(s["NDCG10"], tbl[tune.BASE]["NDCG10"])
        self.assertEqual(s["vs_base"]["wins"], 6)
        self.assertEqual(s["by_category"]["allergy"]["n"], 3)
        self.assertNotIn("vs_base", tbl[tune.BASE])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.md"
            dev, test = qids[:4], qids[4:]
            tuned = {"w_pos": 0.3, "w_neg": 0.9, "w_strong_neg": 0.1}
            full = {t: tune.split_table(f, lab, ids, {tune.STANCE: tuned}, cats, n=200) for t, ids in (
                ("Held-out test split", test), ("Dev split (weights were tuned here: optimistic)", dev),
                ("All 35 queries (includes the dev queries: optimistic)", qids))}
            tune.write_report(out, args=Namespace(labels=Path("eval/llm_labels.jsonl"), queries=Path("eval/queries_final.csv"),
                                                  seed=6165, top_n=30), dev=dev, test=test, tuned=tuned,
                              grid_results=[0] * 144, tbl=full, features=f, labels=lab, categories=cats,
                              info=lambda g: {"name": g}, qid_by_text={"halal chinese food": "Q01"},
                              new_pairs={"dev": 0, "test": 0, "all": 0}, labeler_id="llm:test")
            text = out.read_text()
        for needle in ("Held-out test split", "Limitations", "PROVISIONAL", "New pairs needing judge labels", "halal chinese food",
                       "Weights were tuned on the dev queries only"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
