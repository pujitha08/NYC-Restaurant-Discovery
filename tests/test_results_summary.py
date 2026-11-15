"""The one-line summary under the results heading: counts only what is on the cards, hides zeros, never says 'safe'."""

import unittest
from types import SimpleNamespace

from app import pipeline
from app.highlight import html_text
from src.schemas import ParsedQuery


def view(status=None, **stance):
    summary = None if status is None else SimpleNamespace(status=status)
    return SimpleNamespace(summary=summary, stance_counts=stance or None)


def outcome(views, allergy=False, allergens=()):
    return SimpleNamespace(results=views, allergy_sensitive=allergy, parsed=ParsedQuery(allergens=list(allergens)))


S, M, P, I, N = (pipeline.STATUS_SUPPORTED, pipeline.STATUS_MIXED, pipeline.STATUS_PROBLEMS, pipeline.STATUS_INSUFFICIENT,
                 pipeline.STATUS_NEUTRAL)


class AllergySummaryTests(unittest.TestCase):
    def test_example_sentence(self):
        views = [view(P)] * 2 + [view(M)] + [view(N)] * 2 + [view(I)] * 5
        self.assertEqual(pipeline.results_summary(outcome(views, True, ["peanuts"])),
                         "3 restaurants have reviews that warn about peanuts · 2 describe accommodating it · 5 show no direct evidence")

    def test_each_restaurant_is_counted_once_and_zero_parts_are_hidden(self):
        line = pipeline.results_summary(outcome([view(N)] * 3, True, ["peanuts"]))
        self.assertEqual(line, "3 describe accommodating it")
        self.assertNotIn("warn", line)
        self.assertNotIn("no direct evidence", line)

    def test_singular_wording_and_no_named_allergen(self):
        self.assertEqual(pipeline.results_summary(outcome([view(P)], True)),
                         "1 restaurant has reviews that warn about this allergy")
        self.assertEqual(pipeline.results_summary(outcome([view(N)], True, ["milk", "egg"])), "1 describes accommodating them")

    def test_stance_counts_on_the_card_count_too(self):
        views = [view(N, negative=1, accommodation=2), view(I, accommodation=1), view(None, negative=2), view(I)]
        self.assertEqual(pipeline.results_summary(outcome(views, True, ["peanuts"])),
                         "2 restaurants have reviews that warn about peanuts · 1 describes accommodating it · 1 show no direct evidence")

    def test_unsummarized_cards_are_not_called_no_evidence(self):
        line = pipeline.results_summary(outcome([view(None)] * 2 + [view(I)], True, ["peanuts"]))
        self.assertEqual(line, "1 show no direct evidence · 2 not summarized yet")

    def test_never_says_safe_or_supported(self):
        views = [view(s) for s in (P, M, N, I, None)]
        line = pipeline.results_summary(outcome(views, True, ["peanuts"])).lower()
        for word in ("safe", "supported", "guarantee", "free of"):
            self.assertNotIn(word, line)


class OtherQuerySummaryTests(unittest.TestCase):
    def test_supported_mixed_insufficient(self):
        views = [view(S)] * 4 + [view(M)] * 2 + [view(I)] * 3 + [view(P)]
        self.assertEqual(pipeline.results_summary(outcome(views)),
                         "4 supported by reviews · 2 with mixed evidence · 1 with reviews that report problems · 3 with insufficient evidence")

    def test_zero_parts_hidden(self):
        self.assertEqual(pipeline.results_summary(outcome([view(S)] * 5 + [view(None)])), "5 supported by reviews · 1 not summarized yet")

    def test_no_results_is_empty(self):
        self.assertEqual(pipeline.results_summary(outcome([])), "")


class EscapingTests(unittest.TestCase):
    def test_allergen_text_is_escaped_when_rendered(self):
        line = pipeline.results_summary(outcome([view(P)], True, ['<img src=x onerror=alert(1)>']))
        self.assertIn("<img", line)  # the function returns plain text ...
        html = html_text(line)  # ... and the app escapes it for its HTML block
        self.assertNotIn("<img", html)
        self.assertIn("&lt;img", html)


if __name__ == "__main__":
    unittest.main()
