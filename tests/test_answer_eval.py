"""answer_eval: sheets and scores on synthetic outputs (no pipeline, model or network)."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from eval import answer_eval as ae
from eval.stats import krippendorff_alpha


def output(version, label_for_warning):
    recs = []
    for q in ("Q20", "Q01"):
        results = []
        for rank in (1, 2, 3, 4):
            exc = [{"review_id": rank * 10, "date": "2019-05", "text": f"review text {q} {rank}"}]
            results.append({"rank": rank, "gmap_id": f"{q}-G{rank}", "name": f"Place {rank}", "borough": "Manhattan",
                            "avg_rating": 4.0, "low_evidence": rank == 2, "excerpts": exc,
                            "summary": None if rank == 3 else {"status": "mixed", "note": None, "limitations": [],
                                                               "claims": [{"label": label_for_warning, "text": f"claim {rank}",
                                                                           "excerpts": exc}]},
                            "summary_error": {"message": "rate limit"} if rank == 3 else None, "summary_skipped": None})
        recs.append({"query_id": q, "query": f"query {q}", "boroughs": None, "results": results})
    return {"version": version, "method": "semantic_capped", "records": recs}


class SheetTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        for name, (v, lab) in {"old.json": ("old", "support"), "fixed.json": ("fixed", "contradiction")}.items():
            (self.dir / name).write_text(json.dumps(output(v, lab)))

    def make(self):
        return ae.make_sheets(self.dir / "old.json", self.dir / "fixed.json", ["ana", "ben"], self.dir / "out")

    def read(self, name):
        with open(self.dir / "out" / name, newline="") as f:
            return list(csv.DictReader(f))

    def test_claim_sheet_hides_version_and_covers_top3(self):
        n, by_version = self.make()
        self.assertEqual((n, by_version), (8, {"old": 4, "fixed": 4}))  # 2 queries x 2 summarized of top 3 x 2 versions
        rows, key = self.read("claim_review.csv"), self.read("claim_key.csv")
        self.assertEqual(list(rows[0]), ae.CLAIM_COLUMNS)
        self.assertFalse(any("version" in c for c in ae.CLAIM_COLUMNS))
        self.assertTrue(all(r["supported_by_excerpt"] == "" and r["unjustified_safety_claim"] == "" for r in rows))
        self.assertEqual({k["claim_id"] for k in key}, {r["claim_id"] for r in rows})
        self.assertIn("[2019-05] review text", rows[0]["cited_excerpts"])
        self.assertNotIn("claim 4", " ".join(r["claim"] for r in rows))  # rank 4 is outside the top 3

    def test_usability_sheets(self):
        self.make()
        a, b = self.read("usability_ana.csv"), self.read("usability_ben.csv")
        self.assertEqual(list(a[0]), ae.USABILITY_COLUMNS)
        self.assertEqual(sorted(r["query_id"] for r in a), ["Q01", "Q20"])
        row = next(r for r in a if r["query_id"] == "Q20")
        self.assertIn("[low evidence]", row["result_2"])
        self.assertIn("Status: mixed", row["summary_1"])
        self.assertIn("No summary (rate limit)", row["summary_3"])
        self.assertIn("[contradiction] claim 1", row["summary_1"])  # fixed version only
        self.assertEqual({r["query_id"] for r in b}, {"Q01", "Q20"})

    def test_refuses_overwrite_and_mismatched_methods(self):
        self.make()
        with self.assertRaises(SystemExit):
            self.make()
        bad = output("fixed", "support")
        bad["method"] = "doc_tfidf"
        (self.dir / "fixed.json").write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "different retrieval methods"):
            ae.make_sheets(self.dir / "old.json", self.dir / "fixed.json", ["ana"], self.dir / "out2")


class ScoreTests(unittest.TestCase):
    def test_claim_rates_by_hand(self):
        key = [{"claim_id": f"C{i}", "version": "old" if i < 4 else "fixed"} for i in range(8)]
        review = [{"claim_id": "C0", "supported_by_excerpt": "yes", "unjustified_safety_claim": "yes"},
                  {"claim_id": "C1", "supported_by_excerpt": "no", "unjustified_safety_claim": "yes"},
                  {"claim_id": "C2", "supported_by_excerpt": "partly", "unjustified_safety_claim": "no"},
                  {"claim_id": "C3", "supported_by_excerpt": "yes", "unjustified_safety_claim": "no"},
                  {"claim_id": "C4", "supported_by_excerpt": "yes", "unjustified_safety_claim": "no"},
                  {"claim_id": "C5", "supported_by_excerpt": "Yes ", "unjustified_safety_claim": "No"},
                  {"claim_id": "C6", "supported_by_excerpt": "partly", "unjustified_safety_claim": "no"},
                  {"claim_id": "C7", "supported_by_excerpt": "", "unjustified_safety_claim": ""}]  # not reviewed yet
        s = ae.score_claims(review, key)
        self.assertEqual((s["old"]["claims"], s["old"]["safety"]), (4, 2))
        self.assertAlmostEqual(s["old"]["unsupported_rate_strict"], 1 / 4)
        self.assertAlmostEqual(s["old"]["unsupported_rate_lenient"], 2 / 4)
        self.assertEqual((s["fixed"]["claims"], s["fixed"]["safety"]), (3, 0))
        self.assertAlmostEqual(s["fixed"]["unsupported_rate_lenient"], 1 / 3)
        with self.assertRaisesRegex(ValueError, "yes/partly/no"):
            ae.score_claims([{"claim_id": "C0", "supported_by_excerpt": "maybe", "unjustified_safety_claim": "no"}], key)

    def test_usability_mean_and_alpha(self):
        sheets = {"ana": [{"query_id": "Q1", "usefulness": "5"}, {"query_id": "Q2", "usefulness": "2"},
                          {"query_id": "Q3", "usefulness": "4"}],
                  "ben": [{"query_id": "Q1", "usefulness": "4"}, {"query_id": "Q2", "usefulness": "1"},
                          {"query_id": "Q3", "usefulness": ""}]}
        u = ae.score_usability(sheets)
        self.assertEqual(u["n_queries"], 3)
        self.assertAlmostEqual(u["mean"], (5 + 2 + 4 + 4 + 1) / 5)
        self.assertEqual(u["per_rater"], {"ana": 11 / 3, "ben": 2.5})
        self.assertAlmostEqual(u["alpha_ordinal"], krippendorff_alpha([[5, 4], [2, 1], [4, None]], "ordinal"))
        with self.assertRaisesRegex(ValueError, "1-5"):
            ae.score_usability({"ana": [{"query_id": "Q1", "usefulness": "6"}]})

    def test_eval_queries_cover_every_category(self):
        rows = ae.read_queries()
        self.assertEqual(len(rows), 10)
        self.assertEqual({r["category"] for r in rows}, {"single_diet", "mixed_group", "allergy", "diet_plus_filter"})
        self.assertTrue({"Q20", "Q22"} <= {r["query_id"] for r in rows})


if __name__ == "__main__":
    unittest.main()
