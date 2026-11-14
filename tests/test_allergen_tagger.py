"""Allergen tagger on tiny synthetic data (no dataset download)."""

import tempfile
import unittest

import numpy as np

from src.classifier import allergen_tagger as at


class MappingTests(unittest.TestCase):
    def test_eu_to_fda(self):
        self.assertEqual(at.map_labels(["gluten", "crustaceans", "eggs"]), ["egg", "shellfish", "wheat"])
        self.assertEqual(at.map_labels(["celery", "mustard", "sulphites", "lupin", "molluscs"]), [])
        self.assertEqual(at.map_labels(["molluscs", "fish"]), ["fish"])
        with self.assertRaises(ValueError):
            at.map_labels(["kiwi"])
        self.assertEqual(len(at.FDA_LABELS), 9)
        self.assertEqual(set(at.HF_TO_FDA.values()), set(at.FDA_LABELS))

    def test_dedup_is_case_insensitive_and_unions_labels(self):
        texts, Y, stats = at.prepare([("Peanut Noodles", ["peanuts"]), ("peanut noodles", ["sesame"]),
                                      ("Celery Soup", ["celery"]), ("  ", ["milk"])])
        self.assertEqual(texts, ["Peanut Noodles", "Celery Soup"])
        col = {l: j for j, l in enumerate(at.FDA_LABELS)}
        self.assertEqual(Y[0, col["peanuts"]] + Y[0, col["sesame"]], 2)
        self.assertEqual(Y[1].sum(), 0)  # EU-only label -> negative row
        self.assertEqual((stats["duplicates_removed"], stats["conflicting_duplicates"], stats["no_fda_label"]), (2, 1, 1))

    def test_split_has_no_lowercased_overlap(self):
        texts, Y, _ = at.prepare([(f"Dish {i}", ["milk"] if i % 2 else []) for i in range(100)]
                                 + [(f"DISH {i}", []) for i in range(100)])
        (tr, _), (te, _) = at.split(texts, Y, test_size=0.25, seed=1)
        self.assertEqual(len(tr) + len(te), 100)
        self.assertFalse({t.lower() for t in tr} & {t.lower() for t in te})
        self.assertEqual(at.split(texts, Y, seed=1)[1][0], at.split(texts, Y, seed=1)[1][0])  # deterministic


class ModelTests(unittest.TestCase):
    def test_learns_save_load_and_tag(self):
        words = {"peanuts": ["peanut", "satay"], "milk": ["cheese", "cream"], "shellfish": ["shrimp", "crab"]}
        rows = []
        for i in range(60):
            for hf, (a, b) in (("peanuts", words["peanuts"]), ("milk", words["milk"]), ("crustaceans", words["shellfish"])):
                rows.append((f"{a} special {i}", [hf]))
                rows.append((f"house {b} plate {i}", [hf]))
            rows.append((f"green salad {i}", []))
        texts, Y, _ = at.prepare(rows)
        tagger = at.AllergenTagger().fit(texts, Y)
        self.assertEqual(tagger.tag(["peanut special 999"]), [["peanuts"]])
        self.assertEqual(tagger.tag(["shrimp special 999"]), [["shellfish"]])
        with tempfile.TemporaryDirectory() as d:
            tagger.save(d, {"note": "test"})
            again = at.AllergenTagger.load(d)
            np.testing.assert_array_equal(again.predict(texts[:20]), tagger.predict(texts[:20]))

    def test_evaluate_by_hand(self):
        Y_true = np.zeros((4, 9), dtype=int)
        Y_pred = np.zeros((4, 9), dtype=int)
        Y_true[[0, 1], 0] = 1  # milk: two positives
        Y_pred[[0, 2], 0] = 1  # one hit, one false positive
        r = at.evaluate(Y_true, Y_pred)
        self.assertEqual((r["per_label"]["milk"]["precision"], r["per_label"]["milk"]["recall"]), (0.5, 0.5))
        self.assertEqual(r["per_label"]["milk"]["support"], 2)
        self.assertAlmostEqual(r["macro_f1"], 0.5 / 9)  # other labels have no positives -> F1 0
        self.assertAlmostEqual(r["micro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
