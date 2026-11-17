"""(a) the quote under a finding comes from a review that finding cites; (b) allergen terms in a finding must be supported by a
cited review, group by group. Synthetic unit tests, then regression tests on the real saved results (no model calls)."""

import unittest

from app import demo_cache, pipeline, view_model as vm
from src import allergy
from src.schemas import ParsedQuery


def ex(rid, text, rating=5):
    return pipeline.Excerpt(rid, "2020-05", text, rating)


def view(name, claims, excerpts):
    by = {e.review_id: e for e in excerpts}
    cl = [pipeline.Claim(t, label, [by[i] for i in ids]) for t, label, ids in claims]
    v = pipeline.ResultView(1, "G1", name, "Manhattan", "", 4.0, 50, False, 1.0, list(excerpts))
    v.summary = pipeline.Summary(pipeline.STATUS_MIXED, cl, [], True)
    return v


def outcome(views, query="restaurant careful about peanut allergies", allergens=("peanut",), diets=(), allergy_flag=True):
    parsed = ParsedQuery(allergens=list(allergens), dietary_restrictions=list(diets))
    return pipeline.SearchOutcome("r", query, "Both", "semantic_capped", parsed, parsed, None, list(views), allergy_sensitive=allergy_flag)


class QuoteFromTheCitedReviewTests(unittest.TestCase):
    def diet(self, v):
        return outcome([v], query="vegan restaurants", allergens=(), diets=("vegan",), allergy_flag=False)

    def two_claims(self):
        # claim 1 (the finding) cites review 1, which says little; review 2 (cited only by claim 2) has the best matching sentence
        r1 = ex(1, "Lovely evening with friends. The dessert was good. We will come back.")
        r2 = ex(2, "Great vegan options for everyone and a very vegan friendly kitchen.")
        return view("Place", [("One reviewer enjoyed an evening out.", "support", [1]),
                              ("Another reviewer praised the vegan options.", "support", [2])], [r1, r2]), r1, r2

    def test_the_matched_sentence_wins_when_it_lies_in_a_cited_review(self):
        r1 = ex(1, "Staff were careful with my peanut allergy. They checked the kitchen. Dessert was good.")
        v = view("Place", [("One reviewer said staff were careful with a peanut allergy.", "support", [1])], [r1])
        o = outcome([v])
        q = vm.quote_for(v, o, vm.CARE)
        self.assertEqual(q.text, "Staff were careful with my peanut allergy. … They checked the kitchen.")
        self.assertEqual(q.review_id, 1)

    def test_with_a_finding_the_quote_comes_from_the_review_the_finding_cites_not_from_a_better_one_elsewhere(self):
        v, r1, r2 = self.two_claims()
        o = self.diet(v)
        self.assertEqual(vm.finding_for(v, o, vm.CARE).review_ids, [1])
        q = vm.quote_for(v, o, vm.CARE)
        self.assertEqual(q.review_id, 1)  # not review 2, whose sentence matches the query better
        self.assertEqual(q.text, "Lovely evening with friends. … The dessert was good.")  # nothing matched: its first two sentences

    def test_the_best_matching_sentence_inside_the_cited_review_is_used_when_there_is_one(self):
        r1 = ex(1, "Lovely evening with friends. They asked about my peanut allergy at the door. We will come back.")
        r2 = ex(2, "Peanut allergy peanut allergy: careful handling of a peanut allergy everywhere.")
        v = view("Place", [("A reviewer said staff asked about a peanut allergy.", "support", [1]),
                           ("Another reviewer praised peanut allergy handling.", "support", [2])], [r1, r2])
        q = vm.quote_for(v, outcome([v]), vm.CARE)
        self.assertEqual(q.review_id, 1)
        self.assertTrue(q.text.startswith("They asked about my peanut allergy at the door."))

    def test_the_quote_never_comes_from_a_review_the_finding_does_not_cite(self):
        for column, label in ((vm.CARE, "support"), (vm.PROBLEMS, "contradiction")):
            r1 = ex(1, "I had an allergic reaction to the peanut sauce. Awful night.", 1 if column == vm.CARE else 2)
            r2 = ex(2, "Very careful about peanut allergy, a peanut allergy reaction never happened.", 5)
            v = view("Place", [("First claim about peanut allergies.", label, [1 if column == vm.PROBLEMS else 2]),
                               ("Second claim about peanut allergies.", label, [2 if column == vm.PROBLEMS else 1])], [r1, r2])
            o = outcome([v])
            f = vm.finding_for(v, o, column)
            if f:
                self.assertIn(vm.quote_for(v, o, column).review_id, f.review_ids)

    def test_a_finding_that_cites_no_usable_review_is_not_shown_and_the_card_keeps_the_quote(self):
        r1 = ex(1, "They were careful with my peanut allergy.")
        v = view("Place", [("A finding with no review text.", "support", [1])], [r1])
        v.summary.claims[0].excerpts = [ex(1, "   ")]  # the only cited review has no text
        o = outcome([v])
        self.assertIsNone(vm.finding_for(v, o, vm.CARE))
        self.assertIsNone(vm.finding_candidate(v, o, vm.CARE))
        v.summary.claims[0].excerpts = [ex(99, "careful with my peanut allergy")]  # cited, but not in this restaurant's evidence
        self.assertIsNone(vm.finding_for(v, o, vm.CARE))

    def test_based_on_n_reviews_is_unchanged_by_the_quote_rule(self):
        v, r1, r2 = self.two_claims()
        self.assertEqual(vm.quote_for(v, self.diet(v), vm.CARE).n_reviews, 1)  # review 1 mentions no need; only review 2 counts

    def test_without_a_finding_the_quote_is_chosen_as_before(self):
        r1 = ex(1, "Lovely evening with friends. The dessert was good.")
        r2 = ex(2, "They were very careful with my peanut allergy.")
        v = view("Place", [("Spoke about friends.", "support", [1]), ("Careful claim.", "support", [2])], [r1, r2])
        for claim in v.summary.claims:
            claim.text = pipeline.NEUTRAL_CLAIM  # both are the neutral replacement line: no finding
        o = outcome([v])
        self.assertIsNone(vm.finding_for(v, o, vm.CARE))
        self.assertEqual(vm.quote_for(v, o, vm.CARE).review_id, 2)  # the best match across the cited reviews


class AllergenVocabularyTests(unittest.TestCase):
    def test_the_groups_and_their_terms(self):
        self.assertEqual(list(allergy.ALLERGEN_GROUPS), ["shellfish", "peanut", "tree nut", "sesame", "dairy", "gluten", "egg", "soy", "fish"])
        self.assertEqual(allergy.ALLERGEN_GROUPS["peanut"], ("peanut", "groundnut"))
        self.assertNotIn("nut", allergy.ALLERGEN_GROUPS["peanut"])  # the generic word is not a peanut term
        self.assertEqual(allergy.ALLERGEN_GROUPS["shellfish"][:4], ("shellfish", "seafood", "shrimp", "prawn"))
        self.assertEqual(allergy.ALLERGEN_GROUPS["dairy"], ("dairy", "milk", "lactose", "cheese"))
        self.assertEqual(allergy.ALLERGEN_GROUPS["gluten"], ("gluten", "celiac", "coeliac", "wheat"))

    def test_peanut_matches_peanut_and_groundnut_variants_but_not_the_generic_nut(self):
        pat = allergy.GROUP_PATTERNS["peanut"]
        for text in ("peanut", "Peanuts", "peanut butter", "groundnut", "groundnuts", "GROUNDNUT oil"):
            self.assertTrue(pat.search(text), text)
        for text in ("nut", "nuts", "tree nut", "almond", "walnut", "peanu"):
            self.assertFalse(pat.search(text), text)


class AllergenConsistencyTests(unittest.TestCase):
    def finding(self, finding_text, review_text, name="Place", label="contradiction"):
        """(view, the finding shown, the candidate). A neutral request (no allergy flag, no named allergen) isolates the allergen
        check (b) from the allergen gate of the columns."""
        r = ex(1, review_text, 1 if label == "contradiction" else 5)
        v = view(name, [(finding_text, label, [1])], [r])
        col = vm.PROBLEMS if label == "contradiction" else vm.CARE
        o = outcome([v], query="restaurants", allergens=(), allergy_flag=False)
        return v, vm.finding_for(v, o, col), vm.finding_candidate(v, o, col)

    def test_a_peanut_finding_needs_a_peanut_term_in_a_cited_review(self):
        said = "One reviewer warned that the restaurant ignored a peanut allergy and caused a reaction."
        for review, kept in (("We told them about the tree nut and sesame allergy and I had a reaction.", False),
                             ("I had a reaction. Nuts were in the dish.", False),  # the generic word is not a peanut term
                             ("I have a peanut allergy and had a reaction.", True),
                             ("Peanuts were in the sauce and my friend had a reaction.", True),
                             ("Groundnuts in the sauce gave me a reaction.", True),
                             ("groundnut oil gave me a reaction", True)):
            v, shown, candidate = self.finding(said, review)
            self.assertIsNotNone(candidate)  # it is a candidate either way
            self.assertEqual(shown is not None, kept, review)

    def test_a_suppressed_finding_falls_back_to_the_quote_only_card(self):
        said = "One reviewer warned about peanut allergies."
        v, shown, candidate = self.finding(said, "We told them about the tree nut and sesame allergy and I had a reaction.")
        self.assertIsNone(shown)
        q = vm.quote_for(v, outcome([v], query="restaurants", allergens=(), allergy_flag=False), vm.PROBLEMS)
        self.assertIsNotNone(q)  # the card still has a quote

    def test_every_group_the_finding_names_must_be_supported_separately(self):
        said = "One reviewer reported a reaction to shellfish and milk."
        for review, kept in (("I got a reaction from the shrimp and the milk.", True),
                             ("I got a reaction from the shrimp.", False),  # no dairy term
                             ("I got a reaction from the milk.", False),  # no shellfish term
                             ("I got a reaction from the lobster and the cheese.", True)):
            v, shown, _ = self.finding(said, review)
            self.assertEqual(shown is not None, kept, review)

    def test_a_term_of_the_same_group_is_enough_in_either_direction(self):
        for said, review in (("A reaction to shellfish.", "The shrimp gave me a reaction."), ("A reaction to shrimp.", "Shellfish gave me a reaction."),
                             ("A reaction to dairy.", "The cheese gave me a reaction."), ("A reaction to gluten.", "As a celiac I had a reaction.")):
            v, shown, _ = self.finding(said, review)
            self.assertIsNotNone(shown, (said, review))

    def test_a_term_of_another_group_does_not_count(self):
        v, shown, _ = self.finding("A reaction to shellfish.", "The fish gave me a reaction.")
        self.assertIsNone(shown)  # fish is not shellfish
        v, shown, _ = self.finding("A reaction to peanuts.", "The almonds gave me a reaction.")
        self.assertIsNone(shown)

    def test_a_finding_that_names_no_allergen_is_not_checked(self):
        v, shown, _ = self.finding("One reviewer was unhappy with the service.", "Service was slow and rude, I had a reaction.")
        self.assertIsNotNone(shown)

    def test_the_restaurants_own_name_is_neither_a_mention_nor_a_support(self):
        v, shown, _ = self.finding("A reaction to shellfish.", "Red Lobster served me a dish and I got a reaction.", name="Red Lobster")
        self.assertIsNone(shown)  # "lobster" only in the name
        v, shown, _ = self.finding("A reaction to shellfish at Boilery Seafood.", "I got a reaction to the shrimp.", name="Boilery Seafood")
        self.assertIsNotNone(shown)  # "seafood" in the finding is the name; the shrimp supports "shellfish"

    def test_the_gaps_report_the_group_and_the_finding_terms(self):
        said = "One reviewer reported a peanut allergy."
        v, _, candidate = self.finding(said, "Tree nut and sesame allergy.")
        self.assertEqual(vm.allergen_gaps(candidate, "Place"), [("peanut", ["peanut"])])

    def test_one_cited_review_with_the_term_is_enough_among_several(self):
        r1, r2 = ex(1, "Slow service and a reaction."), ex(2, "I have a peanut allergy and had a reaction.")
        v = view("Place", [("A reviewer warned about peanut allergies.", "contradiction", [1, 2])], [r1, r2])
        self.assertIsNotNone(vm.finding_for(v, outcome([v], query="restaurants", allergens=(), allergy_flag=False), vm.PROBLEMS))

    def test_diet_searches_are_checked_too(self):
        r = ex(1, "Great vegan food and lots of options.", 5)
        v = view("Place", [("A reviewer praised the gluten free options.", "support", [1])], [r])
        o = outcome([v], query="vegan and gluten-free options for a group", allergens=(), diets=("vegan", "gluten-free"), allergy_flag=False)
        self.assertIsNone(vm.finding_for(v, o, vm.CARE))  # the review never says gluten


class SavedResultRegressionTests(unittest.TestCase):
    """The real cases from the saved results (looked up by restaurant name, not by position)."""

    def card(self, query, name, column):
        o = demo_cache.load_cached(query, "Both", "semantic_capped")
        self.assertIsNotNone(o, query)
        c = vm.columns(o)
        lists = {"problems": c.problems, "care": c.care, "mixed": c.mixed}
        view_ = next(v for v in lists[column] if v.name == name)
        return o, view_

    def test_peanut_mission_finding_names_no_allergen_so_it_is_not_checked_and_keeps_its_quote(self):
        o, v = self.card("restaurant careful about peanut allergies", "Mission", "problems")
        f = vm.finding_for(v, o, vm.PROBLEMS)
        self.assertEqual(f.text, "One reviewer warned that the staff and kitchen were inattentive to food allergies and advised avoiding "
                                 "the restaurant for diners with allergies.")  # the top-30 build's wording: no peanut in it
        self.assertEqual(vm.allergen_gaps(f, v.name), [])
        self.assertIn(vm.quote_for(v, o, vm.PROBLEMS).review_id, f.review_ids)

    def test_the_original_mission_case_is_suppressed(self):
        # the first saved Peanut result said "peanut allergies" about a review that is about tree nut and sesame
        from tests.test_display_rules import MISSION
        r = ex(1576831, MISSION, 1)
        v = view("Mission", [("One reviewer warned that the restaurant was not attentive to peanut allergies and advised patrons with "
                              "such allergies to avoid the venue.", "contradiction", [1576831])], [r])
        o = outcome([v])
        cand = vm.finding_candidate(v, o, vm.PROBLEMS)
        self.assertEqual(vm.allergen_gaps(cand, "Mission"), [("peanut", ["peanut"])])
        self.assertIsNone(vm.finding_for(v, o, vm.PROBLEMS))
        self.assertIsNotNone(vm.quote_for(v, o, vm.PROBLEMS))

    def test_peanut_ladybird_warned_is_suppressed_and_the_praised_half_has_no_finding_either(self):
        o, v = self.card("restaurant careful about peanut allergies", "Ladybird", "problems")  # its praise names nut and soy, not peanut
        cand = vm.finding_candidate(v, o, vm.PROBLEMS)
        self.assertIn("peanut", cand.text.lower())
        self.assertIn("peanut", [g for g, _ in vm.allergen_gaps(cand, v.name)])
        self.assertIsNone(vm.finding_for(v, o, vm.PROBLEMS))
        self.assertIsNotNone(vm.quote_for(v, o, vm.PROBLEMS))
        self.assertIsNone(vm.finding_for(v, o, vm.CARE))

    def test_shellfish_buddakan_generic_praise_no_longer_counts_and_warned_is_kept(self):
        o, v = self.card("restaurant careful about shellfish allergies", "Buddakan", "problems")  # no longer in Reviewers disagree
        cand = vm.finding_candidate(v, o, vm.CARE)
        self.assertIsNone(cand)  # the praise (cited review says "allergies" only) fails the care specificity rule
        warned = vm.finding_for(v, o, vm.PROBLEMS)
        self.assertIn("oyster oil", warned.text)  # the oyster-oil review names shellfish
        self.assertEqual(vm.quote_for(v, o, vm.PROBLEMS).review_id, 79848)

    def test_shellfish_boilery_warned_is_kept_because_its_review_says_shrimp(self):
        o, v = self.card("restaurant careful about shellfish allergies", "Boilery Seafood and Grill", "mixed")
        warned = vm.finding_for(v, o, vm.PROBLEMS)
        self.assertIn("shrimp", warned.text)
        self.assertEqual(vm.allergen_gaps(warned, v.name), [])  # shellfish and shrimp are one group; the review says "shrimp"
        self.assertIn(vm.quote_for(v, o, vm.PROBLEMS).review_id, warned.review_ids)

    def test_peanut_pig_and_khao_keeps_its_finding(self):
        o, v = self.card("restaurant careful about peanut allergies", "Pig and Khao", "problems")
        f = vm.finding_for(v, o, vm.PROBLEMS)
        self.assertIn("severe peanut allergy sufferers should avoid the restaurant", f.text)
        self.assertEqual(vm.allergen_gaps(f, v.name), [])  # its cited reviews say peanut
        self.assertIn(vm.quote_for(v, o, vm.PROBLEMS).review_id, f.review_ids)

    def test_celiac_friedmans_herald_square_keeps_its_finding(self):
        o, v = self.card("restaurant careful about celiac disease and cross-contamination", "Friedman's Herald Square", "care")
        f = vm.finding_for(v, o, vm.CARE)
        self.assertEqual(f.text, "One reviewer reported the restaurant is careful about cross‑contamination and can cook all menu items "
                                 "with or without gluten.")
        q = vm.quote_for(v, o, vm.CARE)
        self.assertEqual(q.review_id, 1393345)  # the review the finding cites
        self.assertIn(q.review_id, f.review_ids)

    def test_in_every_saved_result_a_shown_finding_is_quoted_from_a_review_it_cites(self):
        for query, borough, method in demo_cache.DEMO_QUERIES:
            o = demo_cache.load_cached(query, borough, method)
            c = vm.columns(o)
            for col, lst in ((vm.CARE, c.care + c.mixed), (vm.PROBLEMS, c.problems + c.mixed)):
                for v in lst:
                    f = vm.finding_for(v, o, col)
                    if f:
                        self.assertIn(vm.quote_for(v, o, col).review_id, f.review_ids, (query, v.name, col))
                        self.assertEqual(vm.allergen_gaps(f, v.name), [], (query, v.name, col))


if __name__ == "__main__":
    unittest.main()
