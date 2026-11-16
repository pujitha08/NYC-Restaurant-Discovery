"""A match that comes only from the restaurant's own name is not evidence (the Halal Guys case). No model calls."""

import json
import unittest
from types import SimpleNamespace

from app import coverage, evidence_filter, pipeline
from src.llm import evidence_summarizer
from src.schemas import ParsedQuery, RetrievalResult, Snippet

NAME = "The Halal Guys"
NAME_ONLY = "The Halal Guys are the best ever. Food is always fresh and delicious."
REAL = "Delicious halal meal, definetely going back insha Allah."
TERMS = ["halal"]


class MaskingTests(unittest.TestCase):
    def test_the_name_alone_is_not_a_mention(self):
        self.assertFalse(evidence_filter.mentions(NAME_ONLY, TERMS, NAME))
        self.assertTrue(evidence_filter.mentions_only_in_name(NAME_ONLY, TERMS, NAME))

    def test_a_real_mention_counts_even_next_to_the_name(self):
        text = "The Halal Guys serve halal chicken."
        self.assertTrue(evidence_filter.mentions(text, TERMS, NAME))
        self.assertFalse(evidence_filter.mentions_only_in_name(text, TERMS, NAME))

    def test_name_without_the_article_possessive_and_case(self):
        for text in ("Halal Guys is dry.", "i love the halal guys!", "Halal Guys's cart is open", "THE HALAL  GUYS rock"):
            self.assertFalse(evidence_filter.mentions(text, TERMS, NAME), text)

    def test_no_name_means_no_masking(self):
        self.assertTrue(evidence_filter.mentions(NAME_ONLY, TERMS, None))
        self.assertTrue(evidence_filter.mentions(NAME_ONLY, TERMS, ""))

    def test_a_name_that_is_mostly_the_term(self):
        self.assertFalse(evidence_filter.mentions("Alnoor Halal Deli is near me", TERMS, "Alnoor Halal Deli"))
        self.assertTrue(evidence_filter.mentions("Best halal place in town", TERMS, "Alnoor Halal Deli"))

    def test_highlight_skips_the_name_and_escapes(self):
        html = evidence_filter.highlight_outside_name("The Halal Guys <b>halal</b> food", TERMS, NAME, "<mark>{}</mark>")
        self.assertEqual(html, "The Halal Guys &lt;b&gt;<mark>halal</mark>&lt;/b&gt; food")


class CoverageTests(unittest.TestCase):
    def test_coverage_chip_ignores_name_only_excerpts(self):
        excerpts = [pipeline.Excerpt(1, "2020-01", NAME_ONLY), pipeline.Excerpt(2, "2020-02", REAL)]
        parsed = ParsedQuery(dietary_restrictions=["halal"])
        with_name = coverage.coverage(parsed, excerpts, NAME)
        self.assertEqual(with_name, [(coverage.FOUND, "halal: mentioned in 1 excerpt")])
        self.assertEqual(coverage.coverage(parsed, excerpts), [(coverage.FOUND, "halal: mentioned in 2 excerpts")])


def raw_summary(*claims):
    return evidence_summarizer.EvidenceSummary(
        "evidence_found", [evidence_summarizer.EvidenceClaim(t, ids, label) for t, ids, label in claims], [])


def view_with(*texts):
    return pipeline.ResultView(1, "G", NAME, "Manhattan", "", 4.0, 100, False, 1.0,
                               [pipeline.Excerpt(i, "2020-01", t) for i, t in enumerate(texts, 1)])


CONTEXT = ParsedQuery(dietary_restrictions=["halal"])


class SummaryTests(unittest.TestCase):
    def test_a_support_claim_resting_on_the_name_becomes_uncertainty_and_the_status_follows(self):
        view = view_with(NAME_ONLY)
        s = pipeline.to_summary(raw_summary(("One reviewer reported halal food.", [1], "support")), view, False, CONTEXT)
        self.assertEqual([c.label for c in s.claims], ["uncertainty"])
        self.assertEqual(s.claims[0].text, evidence_filter.NAME_ONLY_CLAIM)
        self.assertEqual(s.claims[0].excerpts[0].text, NAME_ONLY)  # the review is still there to read
        self.assertEqual(s.status, pipeline.STATUS_INSUFFICIENT)

    def test_a_real_halal_mention_keeps_its_label(self):
        view = view_with(NAME_ONLY, REAL)
        s = pipeline.to_summary(raw_summary(("One reviewer reported a halal meal.", [2], "support"),
                                            ("Another praised the place.", [1], "support")), view, False, CONTEXT)
        self.assertEqual([c.label for c in s.claims], ["support", "uncertainty"])
        self.assertEqual(s.status, pipeline.STATUS_SUPPORTED)

    def test_a_claim_citing_both_keeps_its_label(self):
        view = view_with(NAME_ONLY, REAL)
        s = pipeline.to_summary(raw_summary(("Reviewers reported halal food.", [1, 2], "support")), view, False, CONTEXT)
        self.assertEqual(s.claims[0].label, "support")

    def test_no_diet_or_allergen_means_no_filtering(self):
        view = view_with(NAME_ONLY)
        s = pipeline.to_summary(raw_summary(("Praise.", [1], "support")), view, False, ParsedQuery(preferences=["halal"]))
        self.assertEqual(s.claims[0].label, "support")

    def test_allergy_guard_still_holds(self):
        view = view_with("Peanut allergy: they were careful.")
        view.name = "Nut House"
        s = pipeline.to_summary(raw_summary(("One reviewer said they were careful.", [1], "support")), view, True,
                                ParsedQuery(allergens=["peanut"]))
        self.assertEqual(s.status, pipeline.STATUS_NEUTRAL)


class PromptTests(unittest.TestCase):
    def test_the_name_is_sent_as_data_and_the_prompt_explains_the_rule(self):
        seen = {}

        def create(**kwargs):
            seen["payload"] = json.loads(kwargs["messages"][1]["content"])
            body = {"status": "insufficient_evidence", "claims": [], "limitations": []}
            message = SimpleNamespace(content=json.dumps(body))
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="m", usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = RetrievalResult("G", 1.0, "m", [Snippet(NAME_ONLY, 1, 0)])
        evidence_summarizer.summarize_evidence(CONTEXT, result, client=client, model="openai/gpt-oss-20b",
                                               restaurant_name="  The  Halal Guys ")
        self.assertEqual(seen["payload"]["restaurant_name"], NAME)
        self.assertIn("restaurant's own name is not evidence", evidence_summarizer.SYSTEM)
        self.assertIn("Halal Guys", evidence_summarizer.SYSTEM)

    def test_without_a_name_the_payload_has_no_such_key(self):
        seen = {}

        def create(**kwargs):
            seen["payload"] = json.loads(kwargs["messages"][1]["content"])
            body = {"status": "insufficient_evidence", "claims": [], "limitations": []}
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)),
                                                            finish_reason="stop")], model="m", usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        evidence_summarizer.summarize_evidence(CONTEXT, RetrievalResult("G", 1.0, "m", [Snippet(REAL, 1, 0)]),
                                               client=client, model="openai/gpt-oss-20b")
        self.assertNotIn("restaurant_name", seen["payload"])


if __name__ == "__main__":
    unittest.main()
