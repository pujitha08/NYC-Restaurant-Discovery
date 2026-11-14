"""Judge validation report: CSV loading, restaurant matching, statistics and the written report (synthetic data)."""

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from eval import validation_report as vr

HEADER = ["sample_id", "query_id", "query", "name", "borough", "evidence_status", "evidence_1", "evidence_2", "label", "note"]


def write_csv(path, rows, encoding="utf-8"):
    with open(path, "w", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def row(sid, qid, name, label, ev1="", note="", borough="Manhattan"):
    return [sid, qid, "q text", name, borough, "retrieved" if ev1 else "none", ev1, "", label, note]


def llm_rec(label, source="llm", provider="groq", upstream=None, reason="because"):
    return {"label": label, "source": source, "provider": provider, "openrouter_upstream": upstream, "reason": reason}


class LoaderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "h.csv"

    def test_blank_labels_are_skipped_and_counted(self):
        write_csv(self.path, [row("S1", "Q1", "A", "2"), row("S2", "Q1", "B", ""), row("S3", "Q1", "C", " 0 ")])
        rows, blank = vr.read_human(self.path)
        self.assertEqual([(r["sample_id"], r["human"]) for r in rows], [("S1", 2), ("S3", 0)])
        self.assertEqual(blank, ["S2"])

    def test_bom_and_unicode_names_are_fine(self):
        write_csv(self.path, [row("S1", "Q1", "Café China 倾城", "1")], encoding="utf-8-sig")
        self.assertEqual(vr.read_human(self.path)[0][0]["name"], "Café China 倾城")

    def test_wrong_encoding_is_a_clear_error(self):
        write_csv(self.path, [row("S1", "Q1", "Café", "1")], encoding="cp1252")
        with self.assertRaisesRegex(ValueError, "not valid UTF-8.*CSV UTF-8"):
            vr.read_human(self.path)

    def test_invalid_label_and_missing_column(self):
        write_csv(self.path, [row("S1", "Q1", "A", "3")])
        with self.assertRaisesRegex(ValueError, "S1='3'"):
            vr.read_human(self.path)
        self.path.write_text("sample_id,query_id\nS1,Q1\n")
        with self.assertRaisesRegex(ValueError, "missing column"):
            vr.read_human(self.path)


class ResolveTests(unittest.TestCase):
    meta = {"G1": {"name": "Alpha", "borough": "Manhattan"}, "G2": {"name": "Beta", "borough": "Brooklyn"},
            "G3": {"name": "Twin", "borough": "Brooklyn"}, "G4": {"name": "Twin", "borough": "Brooklyn"}}
    llm = {("Q1", "G1"): llm_rec(2), ("Q1", "G2"): llm_rec(0), ("Q1", "G3"): llm_rec(1), ("Q1", "G4"): llm_rec(1)}

    def human(self, *rows):
        return [{**dict(zip(HEADER, r)), "human": int(r[8] or 0)} for r in rows]

    def test_cited_review_ids_identify_the_restaurant(self):
        rows = self.human(row("S1", "Q1", "Twin", "1", "[review 77, 2020-05] text", borough="Brooklyn"))
        matched, unmatched = vr.resolve(rows, self.llm, self.meta, lambda ids: {77: "G4"})
        self.assertEqual((matched[0]["gmap_id"], matched[0]["method"], unmatched), ("G4", "review ids", []))

    def test_review_owner_with_a_different_name_is_rejected(self):
        rows = self.human(row("S1", "Q1", "Alpha", "1", "[review 77, 2020-05] text"))
        matched, unmatched = vr.resolve(rows, self.llm, self.meta, lambda ids: {77: "G2"})  # G2 is "Beta"
        self.assertEqual(matched[0]["method"], "name + borough")  # falls back to the unique Alpha/Manhattan

    def test_name_and_borough_fallback_for_rows_without_evidence(self):
        matched, _ = vr.resolve(self.human(row("S1", "Q1", "alpha", "2")), self.llm, self.meta, lambda ids: {})
        self.assertEqual((matched[0]["gmap_id"], matched[0]["llm"]), ("G1", 2))

    def test_ambiguous_or_unknown_or_unlabeled_rows_are_listed_not_guessed(self):
        rows = self.human(row("S1", "Q1", "Twin", "1", borough="Brooklyn"), row("S2", "Q1", "Nowhere", "1"),
                          row("S3", "Q2", "Alpha", "1"))
        matched, unmatched = vr.resolve(rows, self.llm, self.meta, lambda ids: {})
        self.assertEqual(matched, [])
        self.assertEqual([(r["sample_id"], why.split()[0]) for r, why in unmatched],
                         [("S1", "2"), ("S2", "restaurant"), ("S3", "no")])

    def test_sample_key_wins_when_present(self):
        matched, _ = vr.resolve(self.human(row("S1", "Q1", "whatever", "0")), self.llm, self.meta, lambda ids: {},
                                {"S1": "G2"})
        self.assertEqual((matched[0]["gmap_id"], matched[0]["method"]), ("G2", "sample_key"))

    def test_cited_review_ids_parsing(self):
        self.assertEqual(vr.cited_review_ids({"evidence_1": "[review 12, 2020-01] a", "evidence_2": "no id", "evidence_3": "[review 9] b"}), [12, 9])


def pair(sid, qid, h, m, note="", source="llm", provider="groq", upstream=None):
    return {"sample_id": sid, "query_id": qid, "query": "q", "name": f"R{sid}", "human": h, "llm": m, "note": note,
            "llm_record": llm_rec(m, source, provider, upstream, f"reason {sid}")}


class MojibakeTests(unittest.TestCase):
    def test_repair(self):
        self.assertEqual(vr.repair_mojibake("Kest√© Pizza & Vino"), "Kesté Pizza & Vino")
        self.assertEqual(vr.repair_mojibake("Franchia Vegan Caf√©"), "Franchia Vegan Café")
        self.assertEqual(vr.repair_mojibake("Plain Name"), "Plain Name")
        self.assertEqual(vr.repair_mojibake("Weird √ name \u2603"), "Weird √ name \u2603")  # does not fit: left alone

    def test_garbled_row_name_matches_the_clean_restaurant(self):
        meta = {"G1": {"name": "Kesté Pizza & Vino", "borough": "Manhattan"}}
        llm = {("Q1", "G1"): llm_rec(2)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.csv"
            write_csv(path, [row("S1", "Q1", "Kest√© Pizza & Vino", "2")])
            rows, _ = vr.read_human(path)
        matched, unmatched = vr.resolve(rows, llm, meta, lambda ids: {})
        self.assertEqual((len(matched), unmatched), (1, []))


class AnalyzeTests(unittest.TestCase):
    cats = {"Q1": "allergy", "Q2": "single_diet"}

    def test_perfect_agreement(self):
        pairs = [pair(f"S{i}", "Q1", i % 3, i % 3) for i in range(9)]
        r = vr.analyze(pairs, self.cats)
        self.assertEqual((r["agreement"], r["kappa"], r["kappa_quadratic"]), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(r["alpha_ordinal"], 1.0)
        self.assertEqual(r["big_disagreements"], [])

    def test_counts_confusion_and_big_disagreements_in_both_directions(self):
        pairs = [pair("S1", "Q1", 0, 2, "no cross-contact info"), pair("S2", "Q1", 2, 0, "clearly safe"),
                 pair("S3", "Q2", 1, 2), pair("S4", "Q2", 1, 1), pair("S5", "Q2", 2, 2)]
        r = vr.analyze(pairs, self.cats)
        self.assertEqual(r["n"], 5)
        self.assertAlmostEqual(r["agreement"], 0.4)
        self.assertEqual(r["confusion"], [[0, 0, 1], [0, 1, 1], [1, 0, 1]])
        self.assertEqual((r["llm_higher"], r["llm_lower"], r["one_step"]), (2, 1, 1))
        self.assertEqual([p["sample_id"] for p in r["big_disagreements"]], ["S1", "S2"])  # human 0 / LLM 2 first
        self.assertEqual(r["by_category"]["allergy"]["n"], 2)
        self.assertAlmostEqual(r["by_category"]["single_diet"]["agreement"], 2 / 3)
        self.assertEqual(r["by_category"]["mixed_group"]["n"], 0)

    def test_mean_label_by_source(self):
        pairs = [pair("S1", "Q1", 2, 2, provider="openrouter", upstream="crusoe"), pair("S2", "Q1", 0, 1, provider="groq"),
                 pair("S3", "Q2", 0, 0, source="rule")]
        by = vr.analyze(pairs, self.cats)["by_source"]
        self.assertEqual(set(by), {"openrouter/crusoe", "groq", "rule (no evidence shown)"})
        self.assertEqual((by["groq"]["mean_human"], by["groq"]["mean_llm"], by["groq"]["agreement"]), (0.0, 1.0, 0.0))

    def test_no_evidence_mismatch_is_flagged_and_excluded_in_the_sensitivity_numbers(self):
        flagged = pair("S1", "Q1", 2, 0)
        flagged["llm_record"]["reason"] = "No evidence found"
        flagged["evidence_status"] = "retrieved"
        honest_none = pair("S2", "Q1", 0, 0)
        honest_none["llm_record"]["reason"] = "no evidence found"
        honest_none["evidence_status"] = "none"  # the sheet showed nothing either: not a mismatch
        rule = pair("S3", "Q1", 0, 0, source="rule")
        rule["evidence_status"] = "retrieved"
        agree = pair("S4", "Q2", 1, 1)
        r = vr.analyze([flagged, honest_none, rule, agree], self.cats)["no_evidence_mismatch"]
        self.assertEqual((r["n"], r["ids"], r["n_without"], r["agreement_without"]), (1, ["S1"], 3, 1.0))

    def test_bootstrap_ci_is_deterministic_and_skipped_for_tiny_samples(self):
        pairs = [pair(f"S{i}", "Q1", i % 3, (i + (i % 4 == 0)) % 3) for i in range(30)]
        self.assertEqual(vr.analyze(pairs, self.cats)["ci_agreement"], vr.analyze(pairs, self.cats)["ci_agreement"])
        self.assertEqual(vr.bootstrap(lambda a, b: 1.0, pairs[:3]), (None, None))


class ReportTests(unittest.TestCase):
    def test_report_has_every_section_and_the_notes(self):
        cats = {"Q1": "allergy", "Q2": "single_diet"}
        pairs = [pair("S1", "Q1", 0, 2, "pipe | in note"), pair("S2", "Q1", 2, 0), pair("S3", "Q2", 1, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.md"
            vr.write_report(out, args=Namespace(human=Path("eval/labels/labels_x.csv"), llm=Path("eval/llm_labels.jsonl")),
                            rows=[{}] * 4, blank=["S9"], matched=[{**p, "method": "review ids"} for p in pairs],
                            unmatched=[({"sample_id": "S7", "query_id": "Q2", "name": "Nope"}, "restaurant not found")],
                            result=vr.analyze(pairs, cats), categories=cats, provider_counts={"groq": 3})
            text = out.read_text()
        for needle in ("## Agreement", "Cohen's kappa (quadratic-weighted)", "Krippendorff's alpha (ordinal)", "## Confusion matrix",
                       "## Mean label by LLM label source", "## Agreement by query category", "Human 0, LLM 2", "Human 2, LLM 0",
                       "pipe \\| in note", "Rows not matched", "## Limitations", "One rater", "single provider", "1 with a blank label skipped",
                       "Matched: 3 of 4"):
            self.assertIn(needle, text)

    def test_report_with_no_matches_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.md"
            vr.write_report(out, args=Namespace(human=Path("h.csv"), llm=Path("l.jsonl")), rows=[], blank=[], matched=[],
                            unmatched=[], result={}, categories={}, provider_counts={})
            self.assertIn("nothing to report", out.read_text())


if __name__ == "__main__":
    unittest.main()
