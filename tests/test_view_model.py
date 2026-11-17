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


E1, E2, E3 = excerpt(1, "They were careful with my peanut allergy."), excerpt(2, "I got sick after the peanut satay."), \
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

    def test_each_restaurant_is_in_exactly_one_list_and_mixed_ones_are_in_no_column(self):
        c = vm.columns(outcome([CARE, PROBLEMS, MIXED, INSUFF, UNCERTAIN, FAILED, SKIPPED]))
        self.assertEqual([v.rank for v in c.care], [1])
        self.assertEqual([v.rank for v in c.problems], [2])
        self.assertEqual([v.rank for v in c.mixed], [3])
        self.assertEqual([v.rank for v in c.insufficient], [4, 5, 7])
        self.assertEqual([v.rank for v in c.unavailable], [6])
        self.assertEqual(c.checked, 7)
        ranks = [v.rank for lst in (c.care, c.problems, c.mixed, c.insufficient, c.unavailable) for v in lst]
        self.assertEqual(sorted(ranks), [1, 2, 3, 4, 5, 6, 7])  # nobody twice, nobody missing

    def test_no_label_or_badge_says_supported_or_safe(self):
        o = outcome([CARE])
        texts = [vm.care_title(o), vm.care_subline(o), vm.problems_subline(o), vm.count_line(o), vm.empty_problems_message()]
        texts += [vm.badge_text(g, o) for g in vm.BADGES]
        for text in texts:
            self.assertNotRegex(text, r"(?i)\bsupported\b|\bsafe\b|safe to eat")
        self.assertIn("not a safety guarantee", vm.empty_problems_message())


class EmptyColumnTests(unittest.TestCase):
    def test_empty_care_message_for_an_allergy_request(self):
        self.assertEqual(vm.empty_care_message(outcome([PROBLEMS])),
                         "No restaurant in the top results has reviews describing care for peanut allergies. "
                         "That doesn't mean none exist; reviews just don't say.")

    def test_empty_care_message_for_a_diet_request(self):
        o = outcome([PROBLEMS], query="halal restaurants", parsed=ParsedQuery(dietary_restrictions=["halal"]), allergy=False)
        self.assertEqual(vm.empty_care_message(o),
                         "No restaurant in the top results has reviews describing halal options. "
                         "That doesn't mean none exist; reviews just don't say.")

    def test_empty_care_message_without_a_parsed_need(self):
        o = outcome([PROBLEMS], query="pizza", parsed=ParsedQuery(), allergy=False)
        self.assertIn("describing what you asked for.", vm.empty_care_message(o))

    def test_the_messages_never_say_safe_or_supported(self):
        for o in (outcome([PROBLEMS]), outcome([PROBLEMS], parsed=ParsedQuery(dietary_restrictions=["kosher"]), allergy=False)):
            self.assertNotRegex(vm.empty_care_message(o), r"(?i)\bsafe\b|supported")


class CountLineTests(unittest.TestCase):
    def test_allergy_wording(self):
        o = outcome([CARE, PROBLEMS, MIXED, INSUFF, SKIPPED])
        self.assertEqual(vm.count_line(o), "5 restaurants checked · 1 mention allergy care · 1 report problems · "
                                           "1 with mixed reviews · 2 not enough evidence")

    def test_diet_wording(self):
        o = outcome([CARE], query="halal restaurants", parsed=ParsedQuery(dietary_restrictions=["halal"]), allergy=False)
        self.assertEqual(vm.count_line(o), "1 restaurant checked · 1 mention halal options · 0 are critical · "
                                           "0 not enough evidence")

    def test_unavailable_summaries_are_counted_apart_for_a_live_search_only(self):
        self.assertIn("1 summary unavailable", vm.count_line(outcome([CARE, FAILED])))
        self.assertNotIn("unavailable", vm.count_line(outcome([CARE, FAILED], cached=True)))

    def test_the_parts_add_up_to_the_restaurants_checked(self):
        o = outcome([CARE, PROBLEMS, MIXED, INSUFF, FAILED, SKIPPED])
        numbers = [int(n) for n in re.findall(r"(?:^|· )(\d+) ", vm.count_line(o))]
        self.assertEqual(numbers[0], sum(numbers[1:]))


class WordingTests(unittest.TestCase):
    def test_need_phrase(self):
        self.assertEqual(vm.need_phrase(outcome([], parsed=ParsedQuery(allergens=["peanut"]))), "peanut allergies")
        self.assertEqual(vm.need_phrase(outcome([], parsed=ParsedQuery(allergens=["peanut allergy", "milk"]))),
                         "peanut and milk allergies")
        self.assertEqual(vm.need_phrase(outcome([], parsed=ParsedQuery(dietary_restrictions=["halal"]), allergy=False)),
                         "halal options")
        self.assertEqual(vm.need_phrase(outcome([], query="Pizza  Place", parsed=ParsedQuery(), allergy=False)), "your request")  # never the raw query

    def test_titles_by_kind(self):
        diet = outcome([], parsed=ParsedQuery(dietary_restrictions=["kosher"]), allergy=False)
        self.assertEqual(vm.care_title(diet), "Reviews mention kosher options")
        self.assertEqual(vm.problems_subline(diet), "Reviewers are critical of the kosher options here. Read before you go.")
        self.assertEqual(vm.problems_column_title(diet), "Reviews are critical")
        allergy = outcome([])
        self.assertEqual(vm.care_title(allergy), "Reviews mention allergy care")
        self.assertEqual(vm.care_subline(allergy), "Worth a closer look. Confirm with the restaurant.")
        self.assertEqual(vm.problems_subline(allergy), "Reviewers describe allergy trouble here. Read before you go.")
        self.assertEqual(vm.problems_column_title(allergy), "Reviews report problems")

    def test_trace(self):
        o = outcome([], timings={"parse_s": 1.2, "retrieve_s": 3.4, "summarize_s": 5.6})
        self.assertEqual(vm.trace_parts(o, 605540),
                         [("Understood", 1.2), ("Searched 605,540 passages", 3.4), ("Summarized with quotes", 5.6)])
        saved = outcome([], cached=True)
        self.assertTrue(all(s is None for _, s in vm.trace_parts(saved)))


class QuoteTests(unittest.TestCase):
    def test_the_quote_is_the_matched_sentence_and_the_next_from_a_review_cited_by_the_column_label(self):
        long = excerpt(1, "Lovely room and kind staff. They were careful with my peanut allergy and checked the kitchen. Dessert was fine.")
        v = view(1, claims=[("Careful.", "support", [1])], excerpts=[long])
        q = vm.quote_for(v, outcome([v]), vm.CARE)
        self.assertEqual(q.text, "They were careful with my peanut allergy and checked the kitchen. … Dessert was fine.")
        self.assertEqual((q.review_id, q.date, q.rating, q.n_reviews), (1, "2020-05", 5, 1))

    def test_each_column_of_a_mixed_restaurant_has_its_own_quote(self):
        o = outcome([MIXED])
        self.assertEqual(vm.quote_for(MIXED, o, vm.CARE).text, "They were careful with my peanut allergy.")
        self.assertEqual(vm.quote_for(MIXED, o, vm.PROBLEMS).text, "I got sick after the peanut satay.")

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


class HeadingAndLabelTests(unittest.TestCase):
    def test_heading_uses_the_parsed_need_and_is_neutral_otherwise(self):
        self.assertEqual(vm.results_heading(outcome([])), "What reviewers say about peanut allergies")
        diet = outcome([], parsed=ParsedQuery(dietary_restrictions=["halal"]), allergy=False)
        self.assertEqual(vm.results_heading(diet), "What reviewers say about halal options")
        self.assertEqual(vm.results_heading(outcome([], query="pizza", parsed=ParsedQuery(), allergy=False)),
                         "What reviewers say about your request")  # no need parsed: not the raw query
        failed = outcome([])
        failed.parsed = None
        self.assertEqual(vm.results_heading(failed), "What reviewers say")  # the parser failed

    def test_source_label(self):
        self.assertEqual(vm.source_label(outcome([], cached=True)), "Saved demo result")
        self.assertEqual(vm.source_label(outcome([])), "Live search")


class LiveUnavailableTests(unittest.TestCase):
    def err(self, category):
        return {"category": category, "code": "x", "message": "m"}

    def test_a_parser_failure_is_unavailable_with_its_reason(self):
        for category, reason in (("rate_limit", "rate limit"), ("timeout", "timeout"), ("network", "no connection"),
                                 ("bad_request", "the language model didn't answer")):
            o = outcome([CARE], parse_error=self.err(category))
            self.assertEqual(vm.live_unavailable(o), reason)

    def test_every_summary_failing_is_unavailable_but_some_failing_is_not(self):
        a, b = view(1, error=self.err("rate_limit"), excerpts=[E1]), view(2, error=self.err("rate_limit"), excerpts=[E1])
        self.assertEqual(vm.live_unavailable(outcome([a, b])), "rate limit")
        self.assertIsNone(vm.live_unavailable(outcome([a, CARE])))

    def test_a_missing_model_setup_is_unavailable(self):
        self.assertEqual(vm.live_unavailable(outcome([SKIPPED], setup_error=self.err("configuration"))),
                         "the language model isn't set up")

    def test_a_saved_result_is_never_unavailable_and_a_normal_search_is_not(self):
        self.assertIsNone(vm.live_unavailable(outcome([CARE], cached=True, parse_error=self.err("rate_limit"))))
        self.assertIsNone(vm.live_unavailable(outcome([CARE, PROBLEMS])))
        self.assertIsNone(vm.live_unavailable(outcome([])))

    def test_message(self):
        self.assertEqual(vm.unavailable_message("rate limit"),
                         "Live search is unavailable right now (rate limit). Try one of the saved examples.")


class FindingTests(unittest.TestCase):
    def finding(self, claims, excerpts, allergy=True, guarded=True, **kw):
        v = view(1, claims=claims, excerpts=excerpts, guarded=guarded)
        o = outcome([v], allergy=allergy, **kw)
        return vm.finding_for(v, o, vm.CARE), vm.finding_for(v, o, vm.PROBLEMS)

    def test_the_finding_is_the_models_claim_verbatim_with_its_cited_reviews(self):
        care, problems = self.finding([("One reviewer reported that staff were careful with a peanut allergy.", "support", [1]),
                                       ("Another reviewer got sick.", "contradiction", [2])],
                                      [excerpt(1, "They were careful with my peanut allergy."), excerpt(2, "I got sick after the peanut satay.")])
        self.assertEqual(care.text, "One reviewer reported that staff were careful with a peanut allergy.")
        self.assertEqual(care.review_ids, [1])
        self.assertEqual(problems.text, "Another reviewer got sick.")

    def test_no_finding_without_a_cited_review_in_the_restaurants_evidence(self):
        v = view(1, claims=[("A finding.", "support", [99])], excerpts=[E1], guarded=True)
        self.assertEqual(v.summary.claims[0].excerpts, [])  # the id is not among the restaurant's excerpts
        self.assertIsNone(vm.finding_for(v, outcome([v]), vm.CARE))
        v.summary.claims[0].excerpts = [excerpt(99, "peanut allergy care")]  # cited, but not in v.excerpts either
        self.assertIsNone(vm.finding_for(v, outcome([v]), vm.CARE))

    def test_the_neutral_replacement_lines_are_not_findings(self):
        from app import evidence_filter
        for text in (pipeline.NEUTRAL_CLAIM, evidence_filter.NAME_ONLY_CLAIM):
            care, _ = self.finding([(text, "support", [1])], [E1])
            self.assertIsNone(care, text)

    def test_an_allergy_finding_that_asserts_safety_is_dropped_and_the_next_claim_is_used(self):
        for bad in ("The restaurant is safe for peanut allergies.", "One reviewer called it safely prepared.", "It is a guaranteed peanut-free kitchen with a guarantee.",
                    "There is no risk for peanut allergies.", "A risk-free place for nut allergies.", "The staff were reassuring about peanuts.",
                    "SAFE for peanut allergies"):
            care, _ = self.finding([(bad, "support", [1]), ("One reviewer said staff asked about peanut allergies.", "support", [1])], [E1])
            self.assertEqual(care.text, "One reviewer said staff asked about peanut allergies.", bad)
            only, _ = self.finding([(bad, "support", [1])], [E1])
            self.assertIsNone(only, bad)

    def test_the_safety_check_is_for_allergy_requests_only(self):
        care, _ = self.finding([("A safe and friendly place for vegans.", "support", [1])], [E1], allergy=False,
                               parsed=ParsedQuery(dietary_restrictions=["vegan"]))
        self.assertEqual(care.text, "A safe and friendly place for vegans.")

    def test_a_finding_only_comes_from_a_counting_claim(self):
        low = excerpt(1, "Staff said no peanuts.", 1)  # a 1-star review: the care gate drops the claim
        care, _ = self.finding([("They said no peanuts.", "support", [1])], [low])
        self.assertIsNone(care)

    def test_finding_is_not_escaped_here_so_the_page_must_escape_it(self):
        care, _ = self.finding([("<b>bold</b> peanut allergy care", "support", [1])], [E1])
        self.assertEqual(care.text, "<b>bold</b> peanut allergy care")


class NonAllergyWordingTests(unittest.TestCase):
    """Allergy requests keep "report problems" / "Warned"; diet, group and general searches say critical / Criticized."""

    def kinds(self):
        allergy = outcome([CARE, PROBLEMS, MIXED])
        diet = outcome([CARE, PROBLEMS, MIXED], query="halal restaurants", parsed=ParsedQuery(dietary_restrictions=["halal"]), allergy=False)
        general = outcome([CARE, PROBLEMS, MIXED], query="pizza", parsed=ParsedQuery(), allergy=False)
        return allergy, diet, general

    def test_the_request_kind_is_read_from_the_field_the_allergy_guard_uses(self):
        allergy, diet, general = self.kinds()
        self.assertEqual([vm.is_allergy_request(o) for o in (allergy, diet, general)], [True, False, False])
        allergy.allergy_sensitive = False
        self.assertFalse(vm.is_allergy_request(allergy))

    def test_column_title(self):
        allergy, diet, general = self.kinds()
        self.assertEqual(vm.problems_column_title(allergy), "Reviews report problems")
        self.assertEqual(vm.problems_column_title(diet), "Reviews are critical")
        self.assertEqual(vm.problems_column_title(general), "Reviews are critical")
        self.assertEqual(vm.problems_column_title(), "Reviews report problems")  # no outcome: the default stays

    def test_warned_half(self):
        allergy, diet, general = self.kinds()
        self.assertEqual(vm.warned_title(allergy), "Warned")
        self.assertEqual((vm.warned_title(diet), vm.warned_title(general)), ("Criticized", "Criticized"))
        self.assertIn("warn about them", vm.disagree_sub(allergy))
        self.assertIn("criticize them", vm.disagree_sub(diet))

    def test_count_line(self):
        allergy, diet, general = self.kinds()
        self.assertIn("1 report problems", vm.count_line(allergy))
        self.assertNotIn("are critical", vm.count_line(allergy))
        for o in (diet, general):
            self.assertIn("1 are critical", vm.count_line(o))
            self.assertNotIn("report problems", vm.count_line(o))

    def test_notes_and_badge(self):
        allergy, diet, _ = self.kinds()
        self.assertIn("report problems", vm.problems_column_note(allergy, True))
        self.assertEqual(vm.problems_column_note(diet, True), "Critical reviews only appear below, where reviewers disagree.")
        self.assertEqual(vm.problems_column_note(diet, False), "No restaurant in the top results has critical reviews. Reviews are anecdotal.")
        self.assertEqual(vm.badge_text(vm.PROBLEMS, allergy), "⚠ Reviews report problems")
        self.assertEqual(vm.badge_text(vm.PROBLEMS, diet), "⚠ Reviews are critical")
        for o in (allergy, diet):
            self.assertNotRegex(vm.problems_subline(o) + vm.problems_column_note(o, False), r"(?i)\bsafe\b|supported")
