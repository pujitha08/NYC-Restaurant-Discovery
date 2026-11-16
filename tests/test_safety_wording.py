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


class AllergyPhraseTests(unittest.TestCase):
    def test_no_doubled_allergy(self):
        self.assertEqual(pipeline.allergy_phrase(["peanut allergy"]), "a peanut allergy")
        self.assertEqual(pipeline.allergy_phrase(["Peanut Allergies"]), "a Peanut allergy")
        self.assertEqual(pipeline.allergy_phrase(["peanut", "milk allergies"]), "peanut and milk allergies")
        self.assertEqual(pipeline.allergy_phrase(["egg allergy", "sesame allergy", "soy"]), "egg, sesame and soy allergies")

    def test_plain_names_and_empty_input_still_work(self):
        self.assertEqual(pipeline.allergy_phrase(["peanut"]), "a peanut allergy")
        self.assertEqual(pipeline.allergy_phrase(["egg"]), "an egg allergy")
        for empty in ([], None, [""], ["allergy"], ["allergies"]):
            self.assertEqual(pipeline.allergy_phrase(empty), "this allergy")

    def test_results_summary_line_does_not_double_allergy(self):
        from tests.test_results_summary import outcome, view
        line = pipeline.results_summary(outcome([view(pipeline.STATUS_PROBLEMS)], True, ["peanut allergy"]))
        self.assertIn("warn about a peanut allergy", line)
        self.assertNotIn("allergy allergy", line)


class MergeNeutralTests(unittest.TestCase):
    def ex(self, i):
        return pipeline.Excerpt(i, "2019-05", f"text {i}")

    def claim(self, text, label, *ids):
        return pipeline.Claim(text, label, [self.ex(i) for i in ids])

    def test_identical_neutral_lines_collapse_into_one_that_keeps_every_quote(self):
        n = pipeline.NEUTRAL_CLAIM
        merged = pipeline.merge_neutral_claims([self.claim(n, "support", 1), self.claim(n, "support", 2, 1), self.claim(n, "support", 3)])
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].text, merged[0].label), (n, "support"))
        self.assertEqual([e.review_id for e in merged[0].excerpts], [1, 2, 3])  # each review once, in order

    def test_different_labels_and_ordinary_claims_stay_separate_and_in_order(self):
        n = pipeline.NEUTRAL_CLAIM
        claims = [self.claim("A real claim.", "support", 1), self.claim(n, "support", 2), self.claim(n, "contradiction", 3),
                  self.claim(n, "support", 4), self.claim("Another claim.", "uncertainty", 5)]
        merged = pipeline.merge_neutral_claims(claims)
        self.assertEqual([(c.text, c.label) for c in merged], [("A real claim.", "support"), (n, "support"), (n, "contradiction"),
                                                              ("Another claim.", "uncertainty")])
        self.assertEqual([e.review_id for e in merged[1].excerpts], [2, 4])

    def test_the_input_claims_are_not_modified(self):
        n = pipeline.NEUTRAL_CLAIM
        a, b = self.claim(n, "support", 1), self.claim(n, "support", 2)
        pipeline.merge_neutral_claims([a, b])
        self.assertEqual([e.review_id for e in a.excerpts], [1])

    def test_the_card_shows_one_neutral_line(self):
        from app.streamlit_app import ordered_claims
        n = pipeline.NEUTRAL_CLAIM
        summary = SimpleNamespace(claims=[self.claim(n, "support", 1), self.claim(n, "support", 2), self.claim("A warning.", "contradiction", 3)])
        shown = ordered_claims(summary)
        self.assertEqual([c.text for c in shown], ["A warning.", n])
        self.assertEqual(len(shown[1].excerpts), 2)


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
