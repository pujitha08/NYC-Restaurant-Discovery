"""No-evidence audit: flagged-call detection, label overrides, dropping, fingerprint math and sensitivity (synthetic data)."""

import json
import tempfile
import unittest
from pathlib import Path

from eval import no_evidence_audit as ne


def rec(q, g, label, reason="because", review_ids=(1,), source="llm", provider="groq", upstream=None):
    return {"query_id": q, "gmap_id": g, "label": label, "reason": reason, "review_ids": list(review_ids), "source": source,
            "provider": provider, "openrouter_upstream": upstream}


class DetectionTests(unittest.TestCase):
    def test_flag_needs_llm_source_no_evidence_prefix_and_no_cited_review(self):
        self.assertTrue(ne.is_no_evidence_call(rec("Q1", "A", 0, "no evidence found", [])))
        self.assertTrue(ne.is_no_evidence_call(rec("Q1", "A", 0, "No evidence found (restaurant is not a bakery)", [])))
        self.assertFalse(ne.is_no_evidence_call(rec("Q1", "A", 0, "no evidence found", [7])))  # cites a review
        self.assertFalse(ne.is_no_evidence_call(rec("Q1", "A", 0, "no evidence found", [], source="rule")))
        self.assertFalse(ne.is_no_evidence_call(rec("Q1", "A", 0, "The reviews show no evidence", [])))  # not a prefix
        self.assertFalse(ne.is_no_evidence_call(rec("Q1", "A", 2, "Great vegan menu", [3])))

    def test_final_records_last_wins_and_read_records_skips_partial_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "l.jsonl"
            path.write_text(json.dumps(rec("Q1", "A", 0)) + "\n" + json.dumps(rec("Q1", "A", 2)) + "\n{partial")
            records = ne.read_records(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(ne.final_records(records)[("Q1", "A")]["label"], 2)

    def test_census_counts_by_provider_category_and_label(self):
        finals = ne.final_records([rec("Q1", "A", 0, "no evidence found", []), rec("Q1", "B", 2),
                                   rec("Q2", "C", 0, "no evidence found", [], provider="openrouter", upstream="crusoe"),
                                   rec("Q2", "D", 0, "no evidence", [], source="rule")])
        cats = {"Q1": "allergy", "Q2": "single_diet"}
        c = ne.census(finals, cats)
        self.assertEqual((c["pairs"], c["llm_calls"], c["flagged"], c["rule"]), (4, 3, 2, 1))
        self.assertEqual(c["by_source"], {"groq": (1, 2), "openrouter/crusoe": (1, 1)})
        self.assertEqual((c["by_category"]["allergy"], c["by_category"]["single_diet"]), ((1, 2), (1, 1)))
        self.assertEqual(c["labels"], {0: 2})
        self.assertEqual(ne.cross_tab(finals, cats)[("groq", "allergy")], [1, 2])


class TransformTests(unittest.TestCase):
    def test_override_labels_does_not_modify_its_input(self):
        labels = {("Q1", "A"): 0, ("Q1", "B"): 1}
        out = ne.override_labels(labels, {("Q1", "A"): 2, ("Q1", "C"): 1})
        self.assertEqual(out, {("Q1", "A"): 2, ("Q1", "B"): 1, ("Q1", "C"): 1})
        self.assertEqual(labels, {("Q1", "A"): 0, ("Q1", "B"): 1})

    def test_drop_pairs_renumbers_without_touching_the_input(self):
        runs = {"s": {"Q1": ["A", "B", "C"], "Q2": ["A", "B"]}}
        out = ne.drop_pairs(runs, {("Q1", "B")})
        self.assertEqual(out, {"s": {"Q1": ["A", "C"], "Q2": ["A", "B"]}})
        self.assertEqual(runs["s"]["Q1"], ["A", "B", "C"])

    def test_systems_hit_counts_flagged_in_the_top_k(self):
        runs = {"s": {"Q1": ["A", "B", "C"]}, "t": {"Q1": ["C"]}}
        hit = ne.systems_hit(runs, {("Q1", "B"), ("Q1", "C")}, k=2)
        self.assertEqual(hit, {"s": (1, 2), "t": (1, 1)})


class FingerprintTests(unittest.TestCase):
    def test_ratio_is_actual_over_estimate_with_and_without_reviews(self):
        build = lambda p, ev: {"reviews": ev}
        parse = lambda p: [{"text": "x" * 80}]
        pair = {"sample_id": "S1", "flagged": True, "llm_record": {"usage": {"prompt_tokens": 100}}}
        out = ne.fingerprint([pair], "s" * 100, build, parse)[0]
        with_ev = (100 + len(json.dumps({"reviews": parse(None)}))) / 4
        without = (100 + len(json.dumps({"reviews": []}))) / 4
        self.assertAlmostEqual(out["with_sheet_evidence"], 100 / with_ev)
        self.assertAlmostEqual(out["with_no_reviews"], 100 / without)
        self.assertLess(out["with_sheet_evidence"], out["with_no_reviews"])  # more reviews -> larger estimate -> smaller ratio


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        # Two systems, two queries. s1 puts a flagged zero first; s2 puts a relevant result first.
        self.runs = {"s1": {"Q1": ["F", "A", "B"], "Q2": ["F2", "C"]}, "s2": {"Q1": ["A", "B", "F"], "Q2": ["C", "F2"]}}
        self.labels = {("Q1", "F"): 0, ("Q1", "A"): 2, ("Q1", "B"): 1, ("Q2", "F2"): 0, ("Q2", "C"): 2}
        self.cats = {"Q1": "single_diet", "Q2": "allergy"}
        self.flagged = {("Q1", "F"), ("Q2", "F2")}

    def test_variants_dropping_improves_the_system_that_had_flagged_zeros_first(self):
        s = ne.sensitivity(self.runs, self.labels, self.cats, {}, self.flagged, n=200)
        base, dropped = s["baseline (LLM labels)"]["summary"], s["flagged no-evidence pairs dropped"]["summary"]
        self.assertLess(base["s1"], base["s2"])
        self.assertAlmostEqual(dropped["s1"], dropped["s2"])  # with the zeros gone both rank the same relevant items first
        self.assertGreater(dropped["s1"], base["s1"])
        self.assertEqual(sum(s["baseline (LLM labels)"]["unjudged"].values()), 0)

    def test_human_override_changes_only_the_overridden_pairs(self):
        s = ne.sensitivity(self.runs, self.labels, self.cats, {("Q1", "A"): 0}, self.flagged, n=200)
        self.assertLess(s["human labels override LLM on the sample"]["summary"]["s2"], s["baseline (LLM labels)"]["summary"]["s2"])
        self.assertEqual(ne.ranking({"a": 0.2, "b": 0.9, "c": 0.5}), ["b", "c", "a"])

    def test_reproduction_check_against_a_published_per_query_file(self):
        rows = [{"system": "s1", "query_id": "Q1", "NDCG10": 0.5}, {"system": "s1", "query_id": "Q2", "NDCG10": None}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            path.write_text("status,query_id,system,NDCG10\nPROVISIONAL,Q1,s1,0.5001\nPROVISIONAL,Q2,s1,\n")
            diff, n = ne.reproduction_check(rows, path)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(diff, 0.0001)


if __name__ == "__main__":
    unittest.main()
