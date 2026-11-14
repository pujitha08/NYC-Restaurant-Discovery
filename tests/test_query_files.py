"""The frozen query set is the default everywhere; the draft is kept as a historical file."""

import csv
import unittest
from pathlib import Path

from eval import make_pool, make_sample, metrics
from eval.make_pool import read_queries

EVAL = Path(make_pool.__file__).parent
FINAL, DRAFT = EVAL / "queries_final.csv", EVAL / "queries_draft.csv"


class QueryFileTests(unittest.TestCase):
    def test_every_queries_default_is_the_frozen_file(self):
        for module in (make_pool, make_sample, metrics):
            with self.subTest(module=module.__name__):
                self.assertEqual(module.QUERIES_PATH.resolve(), FINAL.resolve())

    def test_frozen_file_validates(self):
        queries = read_queries(FINAL)
        self.assertEqual(len(queries), 35)
        self.assertEqual([q["query_id"] for q in queries], [f"Q{i:02d}" for i in range(1, 36)])
        cats = {q["category"] for q in queries}
        self.assertEqual(cats, {"single_diet", "mixed_group", "allergy", "diet_plus_filter"})

    def test_draft_is_kept_as_history(self):
        self.assertTrue(DRAFT.exists())
        with open(DRAFT, newline="", encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 30)


if __name__ == "__main__":
    unittest.main()
