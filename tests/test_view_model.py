"""The results-screen logic (app/view_model.py): the mapping to columns, counts, the one quote, wording, the states."""

import re
import unittest

from app import pipeline, view_model as vm
from src.schemas import ParsedQuery


def excerpt(i, text, rating=5):
    return pipeline.Excerpt(i, "2020-05", text, rating)


def view(rank=1, name="Place", claims=None, status=None, error=None, excerpts=None, guarded=False):
    ex = {e.review_id: e for e in (excerpts or [])}
    cl = [pipeline.Claim(t, label, [ex[i] for i in ids if i in ex]) for t, label, ids in (claims or [])]
    labels = {c.label for c in cl}
    if status is None:
        status = (pipeline.STATUS_INSUFFICIENT if "support" not in labels and "contradiction" not in labels
                  else pipeline.STATUS_MIXED if {"support", "contradiction"} <= labels
                  else pipeline.STATUS_PROBLEMS if "contradiction" in labels
                  else pipeline.STATUS_NEUTRAL if guarded else pipeline.STATUS_SUPPORTED)
    summary = None if claims is None and error is None and status is None else pipeline.Summary(status, cl, [], guarded)
    v = pipeline.ResultView(rank, f"G{rank}", name, "Brooklyn", "", 4.5, 50, False, 1.0, list(excerpts or []))
    v.summary = None if claims is None else summary
    v.summary_error = error
    return v


def outcome(views, query="restaurant careful about peanut allergies", parsed=None, allergy=True, borough="Both", **kw):
    parsed = parsed or ParsedQuery(allergens=["peanut"])
    return pipeline.SearchOutcome("r", query, borough, "semantic_capped", parsed, parsed, None, views,
                                  allergy_sensitive=allergy, **kw)


E1, E2, E3 = excerpt(1, "They were careful with my peanut allergy."), excerpt(2, "I got sick after the satay."), \
    excerpt(3, "Great pasta.")
CARE = view(1, claims=[("A reviewer said they were careful.", "support", [1])], excerpts=[E1, E3], guarded=True)
PROBLEMS = view(2, claims=[("A reviewer got sick.", "contradiction", [2])], excerpts=[E2])
MIXED = view(3, claims=[("Careful.", "support", [1]), ("Sick.", "contradiction", [2])], excerpts=[E1, E2])
INSUFF = view(4, claims=[], excerpts=[E3])
UNCERTAIN = view(5, claims=[("Mentions nuts.", "uncertainty", [3])], excerpts=[E3])
FAILED = view(6, error={"category": "rate_limit", "code": "rate_limited", "message": "m"}, excerpts=[E1])
SKIPPED = view(7, excerpts=[E1])  # no summary, no error: not summarized
SKIPPED.summary_skipped = "not requested"


class MappingTests(unittest.TestCase):
    def test_the_mapping_table(self):
        self.assertEqual(vm.group_of(CARE), vm.CARE)
        self.assertEqual(vm.group_of(PROBLEMS), vm.PROBLEMS)
        self.assertEqual(vm.group_of(MIXED), vm.MIXED)
        self.assertEqual(vm.group_of(INSUFF), vm.INSUFFICIENT)
        self.assertEqual(vm.group_of(UNCERTAIN), vm.INSUFFICIENT)  # only an uncertainty claim: not enough evidence
        self.assertEqual(vm.group_of(FAILED), vm.UNAVAILABLE)
        self.assertEqual(vm.group_of(SKIPPED), vm.NOT_SUMMARIZED)

    def test_reviews_discuss_this_with_a_supporting_claim_is_care(self):
        self.assertEqual(CARE.summary.status, pipeline.STATUS_NEUTRAL)  # the allergy guard's word
        self.assertEqual(vm.group_of(CARE), vm.CARE)

    def test_insufficient_status_wins_over_any_claims(self):
        v = view(1, claims=[("x", "support", [1])], excerpts=[E1], status=pipeline.STATUS_INSUFFICIENT)
        self.assertEqual(vm.group_of(v), vm.INSUFFICIENT)

    def test_identical_neutral_lines_do_not_change_the_group(self):
        n = pipeline.NEUTRAL_CLAIM
        v = view(1, claims=[(n, "support", [1]), (n, "support", [3])], excerpts=[E1, E3], guarded=True)
        self.assertEqual(vm.group_of(v), vm.CARE)

    def test_columns_put_mixed_in_both_and_unsummarized_under_not_enough(self):
        c = vm.columns(outcome([CARE, PROBLEMS, MIXED, INSUFF, UNCERTAIN, FAILED, SKIPPED]))
        self.assertEqual([v.rank for v in c.care], [1, 3])
        self.assertEqual([v.rank for v in c.problems], [2, 3])
        self.assertEqual([v.rank for v in c.mixed], [3])
        self.assertEqual([v.rank for v in c.insufficient], [4, 5, 7])
        self.assertEqual([v.rank for v in c.unavailable], [6])
        self.assertEqual(c.checked, 7)

    def test_no_label_or_badge_says_supported_or_safe(self):
        o = outcome([CARE])
        texts = [vm.care_title(o), vm.care_subline(o), vm.problems_subline(o), vm.count_line(o), vm.empty_problems_message()]
        texts += [vm.badge_text(g, o) for g in vm.BADGES]
        for text in texts:
            self.assertNotRegex(text, r"(?i)\bsupported\b|\bsafe\b|safe to eat")
        self.assertIn("not a safety guarantee", vm.empty_problems_message())


class CountLineTests(unittest.TestCase):
    def test_allergy_wording(self):
        o = outcome([CARE, PROBLEMS, MIXED, INSUFF, SKIPPED])
        self.assertEqual(vm.count_line(o), "5 restaurants checked · 2 mention allergy care · 2 report problems · "
                                           "2 not enough evidence · 1 listed in both columns")

    def test_diet_wording(self):
        o = outcome([CARE], query="halal restaurants", parsed=ParsedQuery(dietary_restrictions=["halal"]), allergy=False)
        self.assertEqual(vm.count_line(o), "1 restaurant checked · 1 mention halal options · 0 report problems · "
                                           "0 not enough evidence")

    def test_unavailable_summaries_are_counted_apart(self):
        self.assertIn("1 summary unavailable", vm.count_line(outcome([CARE, FAILED])))


class WordingTests(unittest.TestCase):
    def test_need_phrase(self):
        self.assertEqual(vm.need_phrase(outcome([], parsed=ParsedQuery(allergens=["peanut"]))), "peanut allergies")
        self.assertEqual(vm.need_phrase(outcome([], parsed=ParsedQuery(allergens=["peanut allergy", "milk"]))),
                         "peanut and milk allergies")
        self.assertEqual(vm.need_phrase(outcome([], parsed=ParsedQuery(dietary_restrictions=["halal"]), allergy=False)),
                         "halal options")
        self.assertEqual(vm.need_phrase(outcome([], query="Pizza  Place", parsed=ParsedQuery(), allergy=False)), "pizza place")

    def test_titles_by_kind(self):
        diet = outcome([], parsed=ParsedQuery(dietary_restrictions=["kosher"]), allergy=False)
        self.assertEqual(vm.care_title(diet), "Reviews mention kosher options")
        self.assertEqual(vm.problems_subline(diet), "Reviewers describe kosher problems here. Read before you go.")
        allergy = outcome([])
        self.assertEqual(vm.care_title(allergy), "Reviews mention allergy care")
        self.assertEqual(vm.care_subline(allergy), "Worth a closer look. Confirm with the restaurant.")
        self.assertEqual(vm.problems_subline(allergy), "Reviewers describe allergy trouble here. Read before you go.")

    def test_trace(self):
        o = outcome([], timings={"parse_s": 1.2, "retrieve_s": 3.4, "summarize_s": 5.6})
        self.assertEqual(vm.trace_parts(o, 605540),
                         [("Understood", 1.2), ("Searched 605,540 passages", 3.4), ("Summarized with quotes", 5.6)])
        saved = outcome([], cached=True)
        self.assertTrue(all(s is None for _, s in vm.trace_parts(saved)))


class QuoteTests(unittest.TestCase):
    def test_the_quote_is_one_sentence_from_a_review_cited_by_the_column_label(self):
        long = excerpt(1, "Lovely room and kind staff. They were careful with my peanut allergy and checked the kitchen. Dessert was fine.")
        v = view(1, claims=[("Careful.", "support", [1])], excerpts=[long])
        q = vm.quote_for(v, outcome([v]), vm.CARE)
        self.assertEqual(q.text, "They were careful with my peanut allergy and checked the kitchen.")
        self.assertEqual((q.review_id, q.date, q.rating, q.n_reviews), (1, "2020-05", 5, 1))

    def test_each_column_of_a_mixed_restaurant_has_its_own_quote(self):
        o = outcome([MIXED])
        self.assertEqual(vm.quote_for(MIXED, o, vm.CARE).text, "They were careful with my peanut allergy.")
        self.assertEqual(vm.quote_for(MIXED, o, vm.PROBLEMS).text, "I got sick after the satay.")

    def test_n_reviews_counts_distinct_cited_reviews(self):
        e = [excerpt(i, f"Careful about peanut allergy number {i}.") for i in (1, 2, 3)]
        v = view(1, claims=[("a", "support", [1, 2]), ("b", "support", [2, 3]), ("c", "contradiction", [3])], excerpts=e)
        o = outcome([v])
        self.assertEqual(vm.quote_for(v, o, vm.CARE).n_reviews, 3)
        self.assertEqual(vm.quote_for(v, o, vm.PROBLEMS).n_reviews, 1)

    def test_pill_wording(self):
        self.assertEqual(vm.review_pill(1), "Based on 1 review")
        self.assertEqual(vm.review_pill(4), "Based on 4 reviews")

    def test_a_sentence_that_only_names_the_restaurant_loses_to_one_about_the_need(self):
        e = [excerpt(1, "The Halal Guys are the best ever."), excerpt(2, "Delicious halal meal, going back.")]
        v = view(1, name="The Halal Guys", claims=[("a", "support", [1, 2])], excerpts=e)
        o = outcome([v], query="halal restaurants", parsed=ParsedQuery(dietary_restrictions=["halal"]), allergy=False)
        self.assertEqual(vm.quote_for(v, o, vm.CARE).text, "Delicious halal meal, going back.")

    def test_long_sentences_are_cut_around_the_match(self):
        text = "word " * 120 + "peanut allergy handled well " + "word " * 120
        v = view(1, claims=[("a", "support", [1])], excerpts=[excerpt(1, text)])
        q = vm.quote_for(v, outcome([v]), vm.CARE)
        self.assertLessEqual(len(q.text), vm.MAX_QUOTE_CHARS + 4)
        self.assertIn("peanut", q.text)

    def test_fallback_quote_for_state_a_ignores_labels(self):
        q = vm.fallback_quote(FAILED, outcome([FAILED]))
        self.assertEqual(q.text, "They were careful with my peanut allergy.")
        self.assertIsNone(vm.fallback_quote(view(1, error={"category": "x", "code": "y", "message": "m"}), outcome([])))

    def test_no_text_no_quote(self):
        v = view(1, claims=[("a", "support", [1])], excerpts=[])
        self.assertIsNone(vm.quote_for(v, outcome([v]), vm.CARE))


class StateTests(unittest.TestCase):
    def test_no_evidence_when_nothing_is_care_problems_mixed_or_unavailable(self):
        self.assertTrue(vm.is_no_evidence(outcome([INSUFF, UNCERTAIN, SKIPPED])))
        self.assertFalse(vm.is_no_evidence(outcome([INSUFF, CARE])))
        self.assertFalse(vm.is_no_evidence(outcome([INSUFF, FAILED])))
        self.assertFalse(vm.is_no_evidence(outcome([])))

    def test_suggestions_offer_both_boroughs_a_broader_request_and_peanut(self):
        o = outcome([INSUFF], query="sesame allergies", parsed=ParsedQuery(allergens=["sesame"]), borough="Brooklyn")
        labels = [label for label, _, _ in vm.suggestions(o)]
        self.assertEqual(labels, ["Both boroughs", "A broader request", "Peanut"])
        both = vm.suggestions(o)[0]
        self.assertEqual((both[1], both[2]), ("sesame allergies", "Both"))

    def test_no_both_boroughs_suggestion_when_already_both(self):
        o = outcome([INSUFF], parsed=ParsedQuery(dietary_restrictions=["kosher"]), query="kosher", allergy=False)
        self.assertNotIn("Both boroughs", [label for label, _, _ in vm.suggestions(o)])


class UnsupportedTests(unittest.TestCase):
    def test_hours_prices_and_availability_are_split_from_the_dietary_part(self):
        cases = {
            "halal restaurants open late": ("open late", "halal restaurants"),
            "vegan brunch under $20 in Brooklyn": ("under $20", "vegan brunch in Brooklyn"),
            "kosher restaurant that takes reservations tonight": ("reservations tonight", "kosher restaurant"),
            "peanut allergy friendly, open until midnight": ("open until midnight", "peanut allergy friendly"),
            "open at 9pm vegan": ("open at 9pm", "vegan"),
        }
        for query, (clause, rest) in cases.items():
            got = vm.split_unsupported(query)
            self.assertEqual((got.clause, got.rest), (clause, rest), query)

    def test_a_request_with_nothing_to_split_is_left_alone(self):
        for query in ("restaurant careful about peanut allergies", "halal restaurants", "gluten-free pizza",
                      "vegan and gluten-free options for a group", "kosher restaurants", "sesame allergies in Brooklyn"):
            self.assertIsNone(vm.split_unsupported(query), query)

    def test_no_dietary_part_leaves_an_empty_rest(self):
        got = vm.split_unsupported("what time does it open")
        self.assertEqual(got.rest, "")


if __name__ == "__main__":
    unittest.main()
