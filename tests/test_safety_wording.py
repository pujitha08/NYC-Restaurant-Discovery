"""Allergy wording: the summarizer prompt forbids safety language, and a deterministic guard replaces any claim that still
uses it outside quotation marks (keeping label, citation and excerpt)."""

import unittest
from types import SimpleNamespace

from app import pipeline
from src.llm import evidence_summarizer

EXCERPT = pipeline.Excerpt(7, "2019-05", "The burger was made without nuts, they said it was safe.")


def raw_summary(*claims):
    return SimpleNamespace(status="evidence_found", limitations=["Reports are old."],
                           claims=[SimpleNamespace(text=t, label=lab, review_ids=[7]) for t, lab in claims])


def view():
    return SimpleNamespace(excerpts=[EXCERPT])


class WordListTests(unittest.TestCase):
    def test_each_forbidden_word_is_caught_in_the_apps_voice(self):
        for text in ("The restaurant is safe for celiacs.", "It was handled safely.", "A reviewer discussed safety.",
                     "A reviewer gave a guarantee.", "The staff guaranteed it.", "Reassuring them about the allergy.",
                     "The manager reassured them.", "Some reassurance was offered.", "SAFE for nut allergies", "Not safe at all."):
            with self.subTest(text=text):
                self.assertTrue(pipeline.has_safety_wording(text))

    def test_neutral_wording_and_lookalikes_pass(self):
        for text in ("One reviewer said the burger was made without nuts.", "The reviewer reported an allergic reaction.",
                     "A reviewer felt unsafe there.", "The safeguards were not described.", "A reviewer mentioned a deposit box.",
                     ""):
            with self.subTest(text=text):
                self.assertFalse(pipeline.has_safety_wording(text))

    def test_a_reviewers_own_quoted_words_are_allowed(self):
        self.assertFalse(pipeline.has_safety_wording('One reviewer wrote "it felt completely safe for my son".'))
        self.assertFalse(pipeline.has_safety_wording("One reviewer wrote “they were very reassuring” about the allergy."))

    def test_words_outside_the_quotes_still_count(self):
        self.assertTrue(pipeline.has_safety_wording('One reviewer wrote "it felt great" and called the place safe.'))
        self.assertTrue(pipeline.has_safety_wording('The reviewer said "nuts were avoided", reassuring them.'))
        self.assertTrue(pipeline.has_safety_wording('A reviewer said "fine" and "ok", and the place is safe.'))

    def test_an_unbalanced_quote_protects_nothing(self):
        self.assertTrue(pipeline.has_safety_wording('The reviewer wrote "it was safe'))


class ReplacementTests(unittest.TestCase):
    def test_guarded_claims_with_safety_wording_are_replaced_but_keep_label_and_citation(self):
        raw = raw_summary(("One reviewer reported that the restaurant is safe for celiacs.", "support"),
                          ("One reviewer said the burger was made without nuts.", "support"),
                          ("A reviewer warned it was not safe, reassuring nobody.", "contradiction"))
        s = pipeline.to_summary(raw, view(), guarded=True)
        self.assertEqual([c.text for c in s.claims], [pipeline.NEUTRAL_CLAIM, "One reviewer said the burger was made without nuts.",
                                                      pipeline.NEUTRAL_CLAIM])
        self.assertEqual([c.label for c in s.claims], ["support", "support", "contradiction"])
        for c in s.claims:
            self.assertEqual(c.excerpts, [EXCERPT])  # the citation and the excerpt it points to are kept
        self.assertEqual(pipeline.NEUTRAL_CLAIM, "A reviewer’s comment about this allergy is quoted below.")

    def test_a_claim_quoting_the_reviewers_own_safe_is_kept(self):
        raw = raw_summary(('One reviewer wrote "the burger was safe for my allergy".', "support"))
        s = pipeline.to_summary(raw, view(), guarded=True)
        self.assertEqual(s.claims[0].text, 'One reviewer wrote "the burger was safe for my allergy".')

    def test_the_status_is_unaffected_by_replacement(self):
        raw = raw_summary(("The place is safe.", "support"))
        self.assertEqual(pipeline.to_summary(raw, view(), guarded=True).status, pipeline.STATUS_NEUTRAL)  # still never "supported"

    def test_non_allergy_requests_are_not_rewritten(self):
        raw = raw_summary(("One reviewer said the place was safe and clean.", "support"))
        self.assertEqual(pipeline.to_summary(raw, view(), guarded=False).claims[0].text, "One reviewer said the place was safe and clean.")


class PromptTests(unittest.TestCase):
    def test_prompt_forbids_safety_language_for_allergy_requests_and_keeps_the_other_rules(self):
        system = " ".join(evidence_summarizer.SYSTEM.split())
        self.assertIn("For allergy requests", system)
        for word in ("safe, safety, safely, guarantee", "reassure (reassured, reassuring, reassurance)", "inside direct quotation marks"):
            self.assertIn(word, system)
        self.assertIn("the reviewer said the burger was made without nuts", system)
        for rule in ("Never claim restaurant allergen safety, certification, or absence of cross-contamination.",
                     "A reviewer warning against the restaurant is never support.",
                     "A claim's wording must match its label"):
            self.assertIn(rule, system)


if __name__ == "__main__":
    unittest.main()
