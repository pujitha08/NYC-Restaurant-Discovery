"""Provider consistency check on synthetic labels (no network)."""

import unittest
from collections import Counter

from sklearn.metrics import cohen_kappa_score

from eval import provider_check as pc


def rec(label, provider="openrouter", source="llm", sha="h"):
    return {"label": label, "provider": provider, "source": source, "prompt_sha256": sha}


class SelectTests(unittest.TestCase):
    def test_stratified_and_only_model_labels_from_the_reference_provider(self):
        cats = ["allergy", "single_diet", "mixed_group", "diet_plus_filter"]
        rows, ref = [], {}
        for c, cat in enumerate(cats):
            for q in range(4):
                for g in range(6):
                    qid, gid = f"Q{c}{q}", f"G{g}"
                    rows.append({"sample_id": f"S{c}{q}{g}", "query_id": qid, "gmap_id": gid, "category": cat})
                    ref[(qid, gid)] = rec(1, source="rule" if g == 0 else "llm", provider="groq" if g == 1 else "openrouter")
        chosen = pc.select(rows, ref)
        self.assertEqual(Counter(r["category"] for r in chosen), Counter(pc.DEFAULT_ALLOCATION))
        self.assertTrue(all(r["gmap_id"] not in ("G0", "G1") for r in chosen))  # no rule or groq labels
        self.assertEqual(len({(r["query_id"], r["gmap_id"]) for r in chosen}), 40)
        allergy_queries = Counter(r["query_id"] for r in chosen if r["category"] == "allergy")
        self.assertEqual(len(allergy_queries), 4)  # rotated across queries
        self.assertEqual(chosen, pc.select(rows, ref))  # deterministic


class CompareTests(unittest.TestCase):
    def test_aggregates_by_hand(self):
        ref = {(f"Q", f"G{i}"): rec(l) for i, l in enumerate([0, 1, 2, 2, 0, 1])}
        chk = {(f"Q", f"G{i}"): rec(l, "groq") for i, l in enumerate([0, 2, 0, 2, 0, 1])}
        r = pc.compare(ref, chk)
        self.assertEqual(r["n"], 6)
        self.assertAlmostEqual(r["exact"], 4 / 6)
        self.assertEqual((r["differ_by_1"], r["differ_by_2"], r["check_higher"], r["check_lower"]), (1, 1, 1, 1))
        a, b = [0, 1, 2, 2, 0, 1], [0, 2, 0, 2, 0, 1]
        self.assertAlmostEqual(r["kappa_quadratic"], cohen_kappa_score(a, b, weights="quadratic"))
        self.assertEqual(r["providers"], {"reference": {"openrouter": 6}, "check": {"groq": 6}})
        self.assertTrue(r["prompt_hashes_match"])
        with self.assertRaises(ValueError):
            pc.compare(ref, {})


if __name__ == "__main__":
    unittest.main()
