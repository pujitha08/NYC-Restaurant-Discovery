"""Extended pool: top-K of semantic_capped minus pairs that already have labels."""

import unittest

from eval import make_extended_pool as mep
from eval import make_pool
from tests.test_make_pool import COL, FakeDensityRetriever, result


class ExtendedPoolTests(unittest.TestCase):
    def setUp(self):
        gids = [f"G{i}" for i in range(1, 6)]
        sem = FakeDensityRetriever({"capped": {"vegan": [result(g, 1 - i / 10, [(i * 100, f"text {g}")])
                                                         for i, g in enumerate(gids)]}, "log": {}})
        self.pool, self.runs, _ = make_pool.build_pool([{"query_id": "Q1", "query": "vegan", "borough": ""}],
                                                       {"semantic": sem}, systems=mep.BASE_SYSTEM, top_k=5)

    def test_only_semantic_capped_is_run_and_top_k_is_honored(self):
        self.assertEqual({r[1] for r in self.runs}, {"semantic_capped"})
        self.assertEqual(len(self.pool), 5)

    def test_existing_labels_are_excluded_and_never_modified(self):
        labeled = {("Q1", "G1"): 2, ("Q1", "G3"): 0, ("Q2", "G2"): 1}
        rows = mep.new_rows(self.pool, labeled)
        self.assertEqual({r[COL["gmap_id"]] for r in rows}, {"G2", "G4", "G5"})
        self.assertEqual(labeled, {("Q1", "G1"): 2, ("Q1", "G3"): 0, ("Q2", "G2"): 1})

    def test_rows_keep_the_pool_format(self):
        for r in mep.new_rows(self.pool, {}):
            self.assertEqual(len(r), len(make_pool.COLUMNS))
            self.assertEqual(r[COL["label"]], "")


if __name__ == "__main__":
    unittest.main()
