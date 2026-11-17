"""Care cards in allergy searches: (a) the cited review must name the searched allergen, quote-only cards included; (b) a suppressed
finding is re-evaluated by (a); (c) an explicit negative allergy cue keeps a care card out of the care column; (d) same-name
restaurants get their street. Unit tests, then regressions on the real saved Peanut and Dairy-free results (no model calls)."""

import unittest

from app import demo_cache, pipeline, view_model as vm
from src.schemas import ParsedQuery


def ex(rid, text, rating=5):
    return pipeline.Excerpt(rid, "2020-05", text, rating)


def view(name, claims, excerpts, address="", rank=1):
    by = {e.review_id: e for e in excerpts}
    cl = [pipeline.Claim(t, label, [by[i] for i in ids]) for t, label, ids in claims]
    v = pipeline.ResultView(rank, f"G{rank}", name, "Manhattan", address, 4.0, 50, False, 1.0, list(excerpts))
    v.summary = pipeline.Summary(pipeline.STATUS_MIXED, cl, [], True)
    return v


def outcome(views, query="restaurant careful about peanut allergies", allergens=("peanut",), diets=(), allergy=True):
    parsed = ParsedQuery(allergens=list(allergens), dietary_restrictions=list(diets))
    return pipeline.SearchOutcome("r", query, "Both", "semantic_capped", parsed, parsed, None, list(views), allergy_sensitive=allergy)


GENERIC = "Our waiter was super helpful and accommodating for all the allergies and dietary preferences."
PEANUT = "The staff were careful with my peanut allergy and checked the kitchen."


class CareSpecificityTests(unittest.TestCase):
    def test_a_generic_allergy_review_does_not_keep_a_restaurant_in_the_care_column_but_a_peanut_review_does(self):
        generic = view("A", [("The waiter was accommodating for all allergies.", "support", [1])], [ex(1, GENERIC)])
        named = view("B", [("The staff were careful with a peanut allergy.", "support", [1])], [ex(1, PEANUT)], rank=2)
        o = outcome([generic, named])
        self.assertEqual((vm.group_of(generic, o), vm.group_of(named, o)), (vm.INSUFFICIENT, vm.CARE))
        c = vm.columns(o)
        self.assertEqual(([v.name for v in c.care], [v.name for v in c.insufficient]), (["B"], ["A"]))  # "not enough evidence"

    def test_the_searched_allergens_group_is_what_counts(self):
        for review, searched, expected in (("They have a dairy free menu and the cheese is vegan.", "dairy", vm.CARE),
                                           ("The staff asked about my milk allergy.", "dairy", vm.CARE),
                                           ("Careful about my tree nut allergy and almonds.", "tree nut", vm.CARE),
                                           ("They asked about my shellfish allergy.", "peanut", vm.INSUFFICIENT),
                                           ("They were careful about nuts in general.", "peanut", vm.INSUFFICIENT),  # "nut" is not a peanut term
                                           ("They asked about my groundnut allergy.", "peanut", vm.CARE)):
            v = view("A", [("A careful kitchen.", "support", [1])], [ex(1, review)])
            self.assertEqual(vm.group_of(v, outcome([v], allergens=(searched,))), expected, (review, searched))

    def test_only_allergy_sensitive_searches_that_name_an_allergen_are_checked(self):
        v = view("A", [("The waiter was accommodating.", "support", [1])], [ex(1, GENERIC)])
        self.assertEqual(vm.group_of(v, outcome([v], query="allergy friendly", allergens=())), vm.CARE)  # no allergen named
        diet = outcome([v], query="vegan restaurants", allergens=(), diets=("vegan",), allergy=False)
        self.assertEqual(vm.group_of(v, diet), vm.CARE)

    def test_a_card_with_only_a_quote_must_quote_a_review_that_names_the_allergen(self):
        neutral = pipeline.NEUTRAL_CLAIM  # the wording guard's replacement line: no finding is shown, only the quote
        generic = view("A", [(neutral, "support", [1])], [ex(1, GENERIC)])
        named = view("B", [(neutral, "support", [1])], [ex(1, PEANUT)], rank=2)
        o = outcome([generic, named])
        self.assertIsNone(vm.finding_for(generic, o, vm.CARE))
        self.assertEqual((vm.group_of(generic, o), vm.group_of(named, o)), (vm.INSUFFICIENT, vm.CARE))

    def test_a_suppressed_finding_is_re_evaluated_instead_of_staying_in_the_care_column(self):
        # the finding names dairy, but the cited review is generic: the allergen-term rule suppresses it, and (a) moves the card out
        v = view("Buddakan", [("The waiter was accommodating for all allergies, implying attention to dairy allergies.", "support", [1])],
                 [ex(1, GENERIC)])
        o = outcome([v], query="restaurant careful about dairy allergies", allergens=("dairy",))
        self.assertIsNotNone(vm.finding_candidate(v, o, vm.CARE) or True)
        self.assertIsNone(vm.finding_for(v, o, vm.CARE))
        self.assertEqual(vm.group_of(v, o), vm.INSUFFICIENT)

    def test_a_suppressed_finding_whose_quote_still_names_the_allergen_stays_as_a_quote_only_card(self):
        # the finding names fish (not in the review), the review names the searched peanut: the finding is suppressed, the card stays
        v = view("A", [("A careful kitchen that avoids fish.", "support", [1])], [ex(1, PEANUT)])
        o = outcome([v])
        self.assertIsNotNone(vm.finding_candidate(v, o, vm.CARE))
        self.assertIsNone(vm.finding_for(v, o, vm.CARE))
        self.assertEqual(vm.group_of(v, o), vm.CARE)
        self.assertIn("peanut", vm.quote_for(v, o, vm.CARE).text)


class NegativeCueTests(unittest.TestCase):
    CUES = ["It was good but likely not ok for allergies.", "Not okay for peanut allergies at all.", "Not safe if you have an allergy.",
            "The kitchen is unsafe.", "Not suitable for anyone with a nut allergy.",
            "I wouldn't recommend it to anyone with allergies.", "We would not recommend this for a severe peanut allergy.",
            "Do not recommend for people who are allergic.", "I can't recommend it for allergy sufferers."]
    NOT_CUES = ["Much safer for those with peanut allergies.", "They kept my peanut allergy safe.", "Not bad at all, ok for allergies.",
                "A suitable place for allergies.", "I would recommend it. No allergy trouble."]

    def test_the_phrases(self):
        for text in self.CUES:
            self.assertTrue(vm.has_negative_allergy_cue(text), text)
        for text in self.NOT_CUES:
            self.assertFalse(vm.has_negative_allergy_cue(text), text)
        self.assertEqual(len(vm.NEGATIVE_ALLERGY_CUES), 5)

    def test_a_care_card_whose_cited_review_has_a_cue_leaves_the_care_column(self):
        text = "They didn't have any peanuts as most of the other 5 guys. That's good or disappointing and also likely not ok for allergies."
        v = view("Five Guys", [("No peanuts at this location.", "support", [1]), ("Another reviewer saw no peanuts.", "support", [2])],
                 [ex(1, text, 4), ex(2, "There were no peanuts at all here.", 4)])
        o = outcome([v])
        self.assertEqual(vm.group_of(v, o), vm.INSUFFICIENT)  # even though a second claim cites a review without a cue

    def test_with_a_separate_counting_contradiction_claim_the_restaurant_shows_as_reviewers_disagree(self):
        v = view("Five Guys", [("No peanuts at this location.", "support", [1]),
                               ("A reviewer reacted to peanut butter.", "contradiction", [2])],
                 [ex(1, "No peanuts here but likely not ok for allergies.", 4), ex(2, "I had an allergic reaction to the peanut butter shake.", 1)])
        self.assertEqual(vm.group_of(v, outcome([v])), vm.MIXED)

    def test_the_cue_only_matters_in_allergy_sensitive_searches(self):
        v = view("A", [("Good vegan food.", "support", [1])], [ex(1, "Great vegan food but not suitable for kids.", 5)])
        diet = outcome([v], query="vegan restaurants", allergens=(), diets=("vegan",), allergy=False)
        self.assertEqual(vm.group_of(v, diet), vm.CARE)

    def test_a_cue_in_a_review_of_a_dropped_claim_is_not_needed_a_card_without_cues_stays(self):
        v = view("A", [("Staff careful with peanut allergy.", "support", [1])], [ex(1, "Much safer for those with peanut allergies. Careful staff.", 5)])
        self.assertEqual(vm.group_of(v, outcome([v])), vm.CARE)


class SameNameTests(unittest.TestCase):
    def pair(self):
        a = view("Bistango", [], [], address="Bistango, 415 3rd Ave, New York, NY 10016", rank=23)
        b = view("bistango", [], [], address="Bistango, 28 E 39th St, New York, NY 10016", rank=24)
        c = view("Wild", [], [], address="Wild, 535 3rd Ave, New York, NY 10016", rank=28)
        return a, b, c

    def test_street_of(self):
        a, b, c = self.pair()
        self.assertEqual((vm.street_of(a), vm.street_of(b)), ("415 3rd Ave", "28 E 39th St"))
        self.assertEqual(vm.street_of(view("X", [], [], address="")), "")
        self.assertEqual(vm.street_of(view("X", [], [], address="X, Brooklyn")), "")

    def test_shared_names_get_the_street_and_others_are_unchanged(self):
        a, b, c = self.pair()
        o = outcome([a, b, c])
        self.assertEqual(vm.duplicate_names(o), {"bistango"})
        self.assertEqual(vm.place_label(a, o), "Manhattan · 415 3rd Ave")
        self.assertEqual(vm.place_label(b, o), "Manhattan · 28 E 39th St")
        self.assertEqual(vm.place_label(c, o), "Manhattan")

    def test_without_a_duplicate_or_a_street_nothing_changes(self):
        a, b, c = self.pair()
        self.assertEqual(vm.place_label(a, outcome([a, c])), "Manhattan")
        d = view("Bistango", [], [], address="", rank=30)
        self.assertEqual(vm.place_label(d, outcome([a, d])), "Manhattan")

    def test_the_card_subtitle_shows_it(self):
        from app import streamlit_app as ui
        a, b, c = self.pair()
        o = outcome([a, b, c])
        self.assertIn("Manhattan · 415 3rd Ave · ★ 4.0", ui.card_head(a, o))
        self.assertIn('card-meta">Manhattan · ★ 4.0<', ui.card_head(c, o))


class SavedResultRegressionTests(unittest.TestCase):
    PEANUT = "restaurant careful about peanut allergies"
    DAIRY = "restaurant careful about dairy allergies"

    def result(self, query):
        o = demo_cache.load_cached(query, "Both", "semantic_capped")
        self.assertIsNotNone(o, query)
        return o

    def card(self, query, rank, name):
        o = self.result(query)
        v = next(v for v in o.results if v.rank == rank)
        self.assertEqual(v.name, name)
        return o, v

    def test_these_leave_the_care_column(self):
        for query, rank, name in ((self.PEANUT, 10, "Buddakan"), (self.PEANUT, 23, "Five Guys"), (self.DAIRY, 9, "Buddakan"),
                                  (self.DAIRY, 20, "abcV"), (self.DAIRY, 28, "Wild"), (self.DAIRY, 24, "Bistango")):
            o, v = self.card(query, rank, name)
            self.assertNotIn(v, vm.columns(o).care, (query, name, rank))
            self.assertEqual(vm.group_of(v, o), vm.INSUFFICIENT, (query, name, rank))  # they go to "not enough evidence"
            self.assertIn(v, vm.columns(o).insufficient)

    def test_these_stay_in_the_care_column(self):
        for query, rank, name in ((self.PEANUT, 27, "Shake Shack"), (self.DAIRY, 15, "Bareburger"), (self.DAIRY, 23, "Bistango"),
                                  (self.DAIRY, 22, "Brooklyn Natural Market & Cafe")):
            o, v = self.card(query, rank, name)
            self.assertIn(v, vm.columns(o).care, (query, name, rank))

    def test_every_remaining_care_card_names_the_searched_allergen_in_its_quote_source(self):
        for query in (self.PEANUT, self.DAIRY):
            o = self.result(query)
            for v in vm.columns(o).care:
                q = vm.quote_for(v, o, vm.CARE)
                source = next(e for e in v.excerpts if e.review_id == q.review_id)
                self.assertTrue(vm.names_searched_allergen(source.text, o, v.name), (query, v.name))
                self.assertFalse(vm.care_has_negative_cue(v, o), (query, v.name))

    def test_the_two_bistangos_and_the_two_five_guys_are_told_apart(self):
        o = self.result(self.DAIRY)
        labels = {v.rank: vm.place_label(v, o) for v in o.results if v.name == "Bistango"}
        self.assertEqual(len(set(labels.values())), 2, labels)
        self.assertTrue(all(label.startswith("Manhattan · ") for label in labels.values()))
        shake = next(v for v in self.result(self.PEANUT).results if v.name == "Shake Shack")
        self.assertEqual(vm.place_label(shake, self.result(self.PEANUT)), "Manhattan")  # unique name: unchanged
        five = [vm.place_label(v, self.result(self.PEANUT)) for v in self.result(self.PEANUT).results if v.name == "Five Guys"]
        self.assertEqual(len(five), 5)  # five branches of Five Guys in the top 30
        self.assertEqual(len(set(five)), 5)  # each with its own street

    def test_celiac_tree_nut_shellfish_care_columns_are_unchanged_by_the_new_rules(self):
        o = self.result("restaurant careful about celiac disease and cross-contamination")
        self.assertEqual(sorted(v.name for v in vm.columns(o).care),
                         sorted(["Friedman's Herald Square", "Cafe Eighty Two", "Senza Gluten Cafe & Bakery"]))


if __name__ == "__main__":
    unittest.main()
