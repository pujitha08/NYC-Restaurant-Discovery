"""Deterministic display-time rules (no model call): the care gate, the allergy problem gate, honest review counts and quotes.

The review texts are real ones from the saved results (the cases that showed the problems), shortened only where noted.
"""

import unittest

from app import pipeline, view_model as vm
from src.schemas import ParsedQuery

HERALD = ("While many restaurants have gluten free items on their menus, most fail to recognize that celiac disease isn't a "
          "fad, and cross contamination is a real danger to those of us who have it. That's not an issue here. I've been here "
          "twice, and although it's a bit pricy on first glance, it's actually a bargain in reality.")
MISSION = ("If you or anyone in your party suffers from food allergies, please avoid this restaurant. Unfortunately, the "
           "waitstaff and kitchen are not attentive to food allergies. We informed the waitstaff when ordering concerning one "
           "diner in our party with a tree nut and sesame allergy. The kitchen brought a cabbage salad with tahini, despite "
           "telling us that all of the dishes we ordered were safe for someone with a sesame allergy.")
SIZZLING = ["Pros: Big portion Friendly staff Cons: Sesame chicken was not chicken Food was clearly microwaved",
            "NOT RECOMMENDED !! Sesame chicken feels like chewing on a old tire, for they price they charge they could at least use fresh ingredients .",
            "Very disappointed. The food was very bland. The General Tso and sesame chicken was the exact same taste aside from the sesames that were thrown on top."]
RED_LOBSTER = [
    (738786, 1, "I specifically told Adam I have a seafood allergy. He told me no seafood contimation will happen. I ate the food and my face began to swell. food was contimated with seafood."),
    (739842, 1, "I would like to warn anyone with any allergies not to eat at the Red Lobster. Both of us were taken to the hospital."),
    (738174, 1, "My salmon was dry and bland. I have a family member allergic to shellfish so I can’t even take the food home because of the cross contamination."),
]
BUDDAKAN = [
    (79107, 5, "Great and friendly staff that took time to inquire about ingredients for those with allergies."),
    (79848, 3, "I have a shellfish allergy and was unaware that they used Oyster Oil in MANY non-fish dishes. I had a bad reaction and was very sick afterwards."),
    (80054, 1, "My wife is allergic to mushrooms. They told her it was safe to eat the spring rolls. She projectile vomited all night."),
]


def ex(rid, text, rating=5):
    return pipeline.Excerpt(rid, "2020-05", text, rating)


def view(name, claims, excerpts, status=None):
    by = {e.review_id: e for e in excerpts}
    cl = [pipeline.Claim(t, label, [by[i] for i in ids]) for t, label, ids in claims]
    v = pipeline.ResultView(1, "G1", name, "Manhattan", "", 4.0, 50, False, 1.0, list(excerpts))
    labels = {c.label for c in cl}
    v.summary = pipeline.Summary(status or (pipeline.STATUS_INSUFFICIENT if not labels else pipeline.STATUS_MIXED), cl, [], True)
    return v


def outcome(views, query, allergens=(), diets=(), allergy=True):
    parsed = ParsedQuery(allergens=list(allergens), dietary_restrictions=list(diets))
    return pipeline.SearchOutcome("r", query, "Both", "semantic_capped", parsed, parsed, None, list(views), allergy_sensitive=allergy)


class CareGateTests(unittest.TestCase):
    def test_a_support_claim_backed_only_by_one_and_two_star_reviews_is_ignored(self):
        v = view("Superiority Burger", [("The burger had no nuts.", "support", [1])], [ex(1, "Do not eat here. I had a severe reaction.", 1)])
        self.assertEqual(vm.group_of(v, outcome([v], "tree nut", ["tree nut"])), vm.INSUFFICIENT)
        v = view("Place", [("a", "support", [1, 2])], [ex(1, "bad", 1), ex(2, "also bad", 2)])
        self.assertEqual(vm.group_of(v, outcome([v], "x", ["peanut"])), vm.INSUFFICIENT)

    def test_one_review_of_three_stars_or_more_is_enough(self):
        v = view("Place", [("a", "support", [1, 2])], [ex(1, "bad", 1), ex(2, "careful with my peanut allergy", 3)])
        self.assertEqual(vm.group_of(v, outcome([v], "x", ["peanut"])), vm.CARE)

    def test_an_unknown_rating_never_excludes(self):
        v = view("Place", [("a", "support", [1])], [ex(1, "careful with my peanut allergy", None)])
        self.assertEqual(vm.group_of(v, outcome([v], "x", ["peanut"])), vm.CARE)

    def test_the_restaurant_falls_to_problems_mixed_or_not_enough_by_its_other_claims(self):
        cue = ex(2, "I had an allergic reaction to the nuts.", 1)
        low = ex(1, "Staff said no nuts.", 1)
        problems = view("A", [("s", "support", [1]), ("c", "contradiction", [2])], [low, cue])
        self.assertEqual(vm.group_of(problems, outcome([problems], "x", ["nut"])), vm.PROBLEMS)
        good = ex(3, "They were careful about my nut allergy.", 4)
        mixed = view("B", [("s", "support", [3]), ("s2", "support", [1]), ("c", "contradiction", [2])], [good, low, cue])
        self.assertEqual(vm.group_of(mixed, outcome([mixed], "x", ["nut"])), vm.MIXED)
        unclear = view("C", [("s", "support", [1]), ("u", "uncertainty", [3])], [low, good])
        self.assertEqual(vm.group_of(unclear, outcome([unclear], "x", ["nut"])), vm.INSUFFICIENT)

    def test_the_care_quote_comes_from_a_review_rated_three_or_more(self):
        low, good = ex(1, "Staff were careful with my peanut allergy but it was awful.", 1), ex(2, "Great peanut allergy care here.", 5)
        v = view("Place", [("s", "support", [1, 2])], [low, good])
        q = vm.quote_for(v, outcome([v], "peanut allergy", ["peanut"]), vm.CARE)
        self.assertEqual((q.review_id, q.rating), (2, 5))

    def test_the_gate_changes_the_counts(self):
        low = view("A", [("s", "support", [1])], [ex(1, "peanut allergy handled", 1)])
        good = view("B", [("s", "support", [1])], [ex(1, "peanut allergy handled", 5)])
        o = outcome([low, good], "peanut allergy", ["peanut"])
        self.assertEqual((len(vm.columns(o).care), len(vm.columns(o).insufficient)), (1, 1))


class ProblemGateTests(unittest.TestCase):
    def sizzling(self):
        excerpts = [ex(i, t, 2 if i == 0 else 1) for i, t in enumerate(SIZZLING)]
        return view("Sizzling Wok", [(f"complaint {i}", "contradiction", [i]) for i in range(3)], excerpts)

    def test_sizzling_wok_sesame_chicken_is_not_an_allergy_problem(self):
        v = self.sizzling()
        o = outcome([v], "restaurant careful about sesame allergies", ["sesame"])
        self.assertEqual(vm.group_of(v, o), vm.INSUFFICIENT)  # nothing else remains: not enough evidence
        self.assertEqual(vm.columns(o).problems, [])
        self.assertEqual(vm.columns(o).insufficient, [v])

    def test_the_same_complaints_still_count_when_the_request_is_not_an_allergy(self):
        v = self.sizzling()
        o = outcome([v], "sesame chicken", diets=["sesame"], allergy=False)
        self.assertEqual(vm.group_of(v, o), vm.PROBLEMS)

    def test_one_cue_review_is_enough_and_the_other_claims_are_ignored(self):
        excerpts = [ex(0, SIZZLING[0], 2), ex(1, "I had an allergic reaction to the sesame.", 1)]
        v = view("Place", [("a", "contradiction", [0]), ("b", "contradiction", [1])], excerpts)
        o = outcome([v], "sesame allergies", ["sesame"])
        self.assertEqual(vm.group_of(v, o), vm.PROBLEMS)
        self.assertEqual([r.review_id for r in vm.cited_reviews(v, "contradiction", o)], [1])

    def test_every_cue_word_passes_and_dish_words_do_not(self):
        for text in ("allergy", "allergic", "allergies", "an adverse reaction", "anaphylactic shock", "I used an EpiPen",
                     "an epi-pen", "broke out in hives", "my face began to swell", "swelling", "a rash", "I got sick",
                     "she was ill", "the illness lasted days", "taken to the hospital", "after an ER visit", "I'm celiac",
                     "intolerant to dairy", "lactose intolerance", "cross-contamination", "cross contamination", "glutened me",
                     "I can’t eat there", "I can't eat there", "cannot eat there"):
            self.assertTrue(vm.has_allergy_cue(text), text)
        for text in ("the sesame chicken tasted like an old tire", "I will skill up", "a billing mistake", "tasty and bland",
                     "General Tso was the same as the sesame chicken", ""):
            self.assertFalse(vm.has_allergy_cue(text), text)


SPICE = ("I would like to warn anyone with any allergies not to eat at the Red Lobster. Since my mother and I are both very "
         "allergic to any spices, we asked for no salt and pepper. Both of us were taken to the hospital.")
MISSION_2 = ("If you have a food allergy, you should seriously consider going elsewhere. Our party of four had one nut and one "
             "wheat allergy. We then found a lone piece of a wheat noodle at the bottom of our dish.")


class AllergenGateTests(unittest.TestCase):
    """A claim about an allergy request counts only if a cited review names that allergen (or a synonym)."""

    def test_red_lobster_the_spice_allergy_claim_is_ignored_for_shellfish(self):
        excerpts = [ex(738786, RED_LOBSTER[0][2], 1), ex(739842, SPICE, 1), ex(738174, RED_LOBSTER[2][2], 1)]
        v = view("Red Lobster", [("seafood allergy", "contradiction", [738786]), ("spice allergy", "contradiction", [739842]),
                                 ("cross contamination", "contradiction", [738174])], excerpts)
        o = outcome([v], "restaurant careful about shellfish allergies", ["shellfish"])
        self.assertEqual([c.text for c, _ in vm.effective_claims(v, o)], ["seafood allergy", "cross contamination"])
        self.assertEqual(vm.group_of(v, o), vm.PROBLEMS)
        self.assertEqual([r.review_id for r in vm.cited_reviews(v, "contradiction", o)], [738786, 738174])
        self.assertEqual(vm.quote_for(v, o, vm.PROBLEMS).n_reviews, 2)

    def test_the_restaurants_own_name_is_not_a_synonym(self):
        # "Red Lobster" would match "lobster"; only the spice allergy review is cited, and it names no shellfish
        v = view("Red Lobster", [("spice allergy", "contradiction", [739842])], [ex(739842, SPICE, 1)])
        o = outcome([v], "restaurant careful about shellfish allergies", ["shellfish"])
        self.assertEqual(vm.group_of(v, o), vm.INSUFFICIENT)

    def test_buddakan_the_mushroom_allergy_claim_is_ignored_for_shellfish(self):
        excerpts = [ex(i, t, r) for i, r, t in BUDDAKAN]
        v = view("Buddakan", [("staff asked", "support", [79107]), ("oyster oil", "contradiction", [79848]),
                              ("mushroom", "contradiction", [80054])], excerpts)
        o = outcome([v], "restaurant careful about shellfish allergies", ["shellfish"])
        self.assertEqual([c.text for c, _ in vm.effective_claims(v, o)], ["staff asked", "oyster oil"])
        self.assertEqual(vm.group_of(v, o), vm.MIXED)
        self.assertEqual(vm.quote_for(v, o, vm.PROBLEMS).n_reviews, 1)  # the mushroom review is no longer counted

    def test_a_care_claim_may_make_a_generic_allergy_statement_but_a_problem_claim_may_not(self):
        text = "Great and friendly staff that took time to inquire about ingredients for those with allergies."
        care = view("Place", [("staff asked", "support", [1])], [ex(1, text, 5)])
        o = outcome([care], "restaurant careful about shellfish allergies", ["shellfish"])
        self.assertEqual(vm.group_of(care, o), vm.CARE)
        problem = view("Place", [("staff did not ask", "contradiction", [1])], [ex(1, "Staff ignored my allergies and I got sick.", 1)])
        self.assertEqual(vm.group_of(problem, outcome([problem], "shellfish allergies", ["shellfish"])), vm.INSUFFICIENT)

    def test_a_care_claim_with_neither_the_allergen_nor_an_allergy_statement_is_ignored(self):
        v = view("Place", [("good food", "support", [1])], [ex(1, "Wonderful dinner, great desserts and friendly staff.", 5)])
        self.assertEqual(vm.group_of(v, outcome([v], "shellfish allergies", ["shellfish"])), vm.INSUFFICIENT)

    def test_mission_is_a_problem_for_peanut_tree_nut_and_sesame_because_its_reviews_say_nut_and_sesame(self):
        v = view("Mission", [("not attentive", "contradiction", [1]), ("waitress", "contradiction", [2])],
                 [ex(1, MISSION, 1), ex(2, MISSION_2, 1)])
        peanut = outcome([v], "restaurant careful about peanut allergies", ["peanut"])
        self.assertEqual(vm.group_of(v, peanut), vm.PROBLEMS)  # "tree nut" and "one nut" are generic nut mentions
        self.assertEqual([r.review_id for r in vm.cited_reviews(v, "contradiction", peanut)], [1, 2])
        self.assertEqual(vm.group_of(v, outcome([v], "restaurant careful about tree nut allergies", ["tree nut"])), vm.PROBLEMS)
        sesame = outcome([v], "restaurant careful about sesame allergies", ["sesame"])
        self.assertEqual(vm.group_of(v, sesame), vm.PROBLEMS)
        self.assertEqual([r.review_id for r in vm.cited_reviews(v, "contradiction", sesame)], [1])  # only review 1 says sesame/tahini

    def test_mission_stays_out_of_a_peanut_search_when_its_reviews_name_no_nut_at_all(self):
        v = view("Mission", [("wheat", "contradiction", [1])], [ex(1, "We told them about the wheat allergy and still found a noodle in the dish.", 1)])
        self.assertEqual(vm.group_of(v, outcome([v], "peanut allergies", ["peanut"])), vm.INSUFFICIENT)

    def test_superiority_burger_is_a_peanut_problem_through_the_generic_nuts_mention(self):
        text = ("I am quite allergic. The employee confidently told me that there are no nuts in the burger. After a bite I had a severe "
                "allergic reaction. His website says: CONTAINS NUTS.")
        v = view("Superiority Burger", [("no nuts", "contradiction", [1])], [ex(1, text, 1)])
        o = outcome([v], "restaurant careful about peanut allergies", ["peanut"])
        self.assertEqual(vm.group_of(v, o), vm.PROBLEMS)
        self.assertEqual(vm.quote_for(v, o, vm.PROBLEMS).n_reviews, 1)

    def test_ladybird_keeps_its_cashew_warning_for_peanut(self):
        text = ("The waitress told us with certainty that they do not use those nuts at all in any dishes for allergy reasons "
                "specifically. Unfortunately when we asked if there was a dessert she might recommend, the strawberry cheesecake "
                "that was suggested did in fact contain Cashews.")
        v = view("Ladybird", [("cashews", "contradiction", [1])], [ex(1, text, 2)])
        self.assertEqual(vm.group_of(v, outcome([v], "peanut allergies", ["peanut"])), vm.PROBLEMS)

    def test_the_generic_nut_mention_is_for_warnings_only_care_claims_stay_strict(self):
        v = view("Place", [("careful", "support", [1])], [ex(1, "Wonderful staff and the best nuts on the salad.", 5)])
        self.assertEqual(vm.group_of(v, outcome([v], "peanut allergies", ["peanut"])), vm.INSUFFICIENT)
        v2 = view("Place", [("careful", "support", [1])], [ex(1, "They were careful about my allergies and the nuts.", 5)])
        self.assertEqual(vm.group_of(v2, outcome([v2], "peanut allergies", ["peanut"])), vm.CARE)  # a generic allergy statement

    def test_the_generic_nut_mention_does_not_open_other_allergens(self):
        milos = view("Estiatorio Milos", [("wait", "contradiction", [1])],
                     [ex(1, "Though I still had to wait for a new meal, nor did they remove it off of my bill which was disappointing. So if you have allergies be aware.", 1)])
        self.assertEqual(vm.group_of(milos, outcome([milos], "shellfish allergies", ["shellfish"])), vm.INSUFFICIENT)
        buddakan = view("Buddakan", [("menu", "contradiction", [1])],
                        [ex(1, "Allergy info wasn't made clear on the menu. I now always mention this allergy wherever I go, especially for Asian restaurants.", 3)])
        self.assertEqual(vm.group_of(buddakan, outcome([buddakan], "sesame allergies", ["sesame"])), vm.INSUFFICIENT)
        nuts = view("Place", [("nuts", "contradiction", [1])], [ex(1, "There were nuts in my dish and I had a reaction.", 1)])
        self.assertEqual(vm.group_of(nuts, outcome([nuts], "sesame allergies", ["sesame"])), vm.INSUFFICIENT)
        self.assertEqual(vm.group_of(nuts, outcome([nuts], "shellfish allergies", ["shellfish"])), vm.INSUFFICIENT)

    def test_the_synonym_groups(self):
        from src.allergy import allergen_groups
        cases = {"shellfish": ["shellfish", "shrimp", "lobster", "crab", "oyster", "clam", "mussel", "scallop", "seafood", "prawns"],
                 "peanut": ["peanut", "peanuts"], "tree nut": ["tree nut", "almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut", "nuts"],
                 "sesame": ["sesame", "tahini"], "dairy": ["dairy", "milk", "lactose", "cheese"],
                 "gluten": ["gluten", "celiac", "wheat"]}
        for group, words in cases.items():
            for word in words:
                self.assertIn(group, allergen_groups(f"{word} allergy"), word)
                self.assertTrue(vm.names_the_allergen(f"I am allergic to {word}.", vm.allergen_patterns(outcome([], "x", [group]))), word)
        self.assertEqual(allergen_groups("peanut allergy"), ["peanut"])  # "nut" does not match inside "peanut"
        self.assertEqual(allergen_groups("celiac disease"), ["gluten"])
        self.assertEqual(allergen_groups("pollen"), [])

    def test_not_gated_without_a_named_allergen_or_for_other_requests(self):
        v = view("Place", [("good", "support", [1])], [ex(1, "Wonderful dinner.", 5)])
        self.assertEqual(vm.group_of(v, outcome([v], "allergy friendly", [])), vm.CARE)  # allergy wording, no allergen parsed
        diet = outcome([v], "vegan options", diets=["vegan"], allergy=False)
        self.assertEqual(vm.group_of(v, diet), vm.CARE)

    def test_an_allergen_in_no_group_is_matched_by_its_own_words(self):
        v = view("Place", [("c", "contradiction", [1]), ("d", "contradiction", [2])],
                 [ex(1, "I had a reaction to the mustard.", 1), ex(2, "I had a reaction to the sauce.", 1)])
        o = outcome([v], "mustard allergy", ["mustard"])
        self.assertEqual([r.review_id for r in vm.cited_reviews(v, "contradiction", o)], [1])


class HonestCountTests(unittest.TestCase):
    def red_lobster(self, extra=()):
        excerpts = [ex(i, t, r) for i, r, t in RED_LOBSTER] + list(extra)
        claims = [("one", "contradiction", [738786]), ("two", "contradiction", [739842]), ("three", "contradiction", [738174]),
                  ("again", "contradiction", [738786, 739842])]  # reviews cited twice
        claims += [("filler", "contradiction", [e.review_id]) for e in extra]
        return view("Red Lobster", claims, excerpts)

    def test_red_lobster_distinct_reviews_are_counted_once(self):
        v = self.red_lobster()
        o = outcome([v], "restaurant careful about shellfish allergies", ["shellfish"])
        q = vm.quote_for(v, o, vm.PROBLEMS)
        self.assertEqual(q.n_reviews, 3)  # 4 claims cite 5 times; 3 distinct reviews, each about an allergy

    def test_a_cited_review_that_mentions_neither_the_need_nor_an_allergy_cue_is_not_counted(self):
        extra = [ex(1, "Slow service and the food was cold when it arrived.", 1)]
        v = self.red_lobster(extra)
        v.summary.claims[-1].label = "contradiction"
        # an allergy request's problem gate would already drop that claim; the count must not include its review either
        o = outcome([v], "restaurant careful about shellfish allergies", ["shellfish"])
        q = vm.quote_for(v, o, vm.PROBLEMS)
        self.assertEqual(q.n_reviews, 3)
        self.assertNotIn(1, [r.review_id for r in vm.cited_reviews(v, "contradiction", o)])

    def test_buddakan_counts_each_side_apart(self):
        excerpts = [ex(i, t, r) for i, r, t in BUDDAKAN]
        v = view("Buddakan", [("staff asked", "support", [79107]), ("oyster oil", "contradiction", [79848]),
                              ("mushroom", "contradiction", [80054, 79848])], excerpts)
        o = outcome([v], "restaurant careful about shellfish allergies", ["shellfish"])
        self.assertEqual(vm.group_of(v, o), vm.MIXED)
        self.assertEqual(vm.quote_for(v, o, vm.CARE).n_reviews, 1)
        self.assertEqual(vm.quote_for(v, o, vm.PROBLEMS).n_reviews, 2)  # 79848 is cited twice but counted once

    def test_a_diet_request_counts_only_reviews_that_mention_the_diet_outside_the_name(self):
        excerpts = [ex(1, "Halal food, delicious.", 5), ex(2, "The Halal Guys are the best ever.", 5), ex(3, "Great place, friendly staff.", 5)]
        v = view("The Halal Guys", [("a", "support", [1, 2, 3])], excerpts)
        o = outcome([v], "halal restaurants", diets=["halal"], allergy=False)
        self.assertEqual(vm.quote_for(v, o, vm.CARE).n_reviews, 1)

    def test_no_parsed_need_counts_every_cited_review(self):
        v = view("Place", [("a", "support", [1, 2])], [ex(1, "Nice", 5), ex(2, "Nice too", 4)])
        o = outcome([v], "pizza", allergy=False)
        self.assertEqual(vm.quote_for(v, o, vm.CARE).n_reviews, 2)


class QuoteTests(unittest.TestCase):
    def test_friedman_herald_square_quote_includes_the_payoff_sentence(self):
        v = view("Friedman's Herald Square", [("careful", "support", [1])], [ex(1, HERALD, 5)])
        q = vm.quote_for(v, outcome([v], "restaurant careful about celiac disease", ["celiac disease"]), vm.CARE)
        self.assertEqual(q.text.split(" … ")[-1], "That's not an issue here.")
        self.assertTrue(q.text.startswith("While many restaurants have gluten free items"))
        self.assertLessEqual(len(q.text), 285)
        self.assertEqual(q.text.count(" … "), 1)

    def test_mission_problems_quote_is_a_sentence_with_an_allergy_cue_plus_the_next(self):
        v = view("Mission", [("not attentive", "contradiction", [1])], [ex(1, MISSION, 1)])
        o = outcome([v], "restaurant careful about peanut allergies", ["peanut"])
        q = vm.quote_for(v, o, vm.PROBLEMS)
        first = q.text.split(" … ")[0]
        self.assertTrue(vm.has_allergy_cue(first))
        self.assertLessEqual(len(q.text), 285)
        self.assertGreater(q.text.count(" … "), 0)  # the sentence after it

    def test_a_cue_sentence_beats_a_better_matching_sentence_without_one_for_problems_only(self):
        text = "The peanut sauce was tasty and the peanut dish too. My friend had an allergic reaction at the table."
        v = view("Place", [("a", "contradiction", [1]), ("b", "support", [1])], [ex(1, text, 4)])
        o = outcome([v], "peanut allergies", ["peanut"])
        self.assertTrue(vm.quote_for(v, o, vm.PROBLEMS).text.startswith("My friend had an allergic reaction"))
        self.assertTrue(vm.quote_for(v, o, vm.CARE).text.startswith("The peanut sauce was tasty"))

    def test_a_short_match_also_gets_the_sentence_before_it(self):
        v = view("Place", [("a", "support", [1])], [ex(1, "We went on a Friday night. Kosher! The staff were lovely and kind.", 5)])
        o = outcome([v], "kosher restaurants", diets=["kosher"], allergy=False)
        self.assertEqual(vm.quote_for(v, o, vm.CARE).text, "We went on a Friday night. … Kosher! … The staff were lovely and kind.")

    def test_a_long_match_is_not_given_a_before_sentence(self):
        text = "Before sentence here. The kosher kitchen is supervised and the staff explained every dish to us patiently. Next one."
        v = view("Place", [("a", "support", [1])], [ex(1, text, 5)])
        o = outcome([v], "kosher restaurants", diets=["kosher"], allergy=False)
        self.assertTrue(vm.quote_for(v, o, vm.CARE).text.startswith("The kosher kitchen"))

    def test_the_quote_is_capped_and_the_match_is_never_dropped(self):
        long_match = "We loved the kosher kitchen " + "and its careful staff " * 9 + "very much."
        nxt = "Then " + "more words follow here " * 12 + "end."
        v = view("Place", [("a", "support", [1])], [ex(1, f"{long_match} {nxt}", 5)])
        o = outcome([v], "kosher restaurants", diets=["kosher"], allergy=False)
        q = vm.quote_for(v, o, vm.CARE).text
        self.assertLessEqual(len(q), 285)
        self.assertTrue(q.startswith("We loved the kosher kitchen"))
        v2 = view("Place", [("a", "support", [1])], [ex(1, "We loved the kosher kitchen. " + "x " * 200, 5)])
        q2 = vm.quote_for(v2, o, vm.CARE).text
        self.assertLessEqual(len(q2), 285)
        self.assertTrue(q2.startswith("We loved the kosher kitchen. … "))
        self.assertTrue(q2.endswith(" …"))

    def test_a_match_at_the_end_of_the_review_has_nothing_after_it(self):
        v = view("Place", [("a", "support", [1])], [ex(1, "Lovely evening. Everything here is strictly kosher and certified by the board.", 5)])
        o = outcome([v], "kosher restaurants", diets=["kosher"], allergy=False)
        self.assertEqual(vm.quote_for(v, o, vm.CARE).text, "Everything here is strictly kosher and certified by the board.")

    def test_state_a_fallback_quote_uses_the_same_composition(self):
        v = view("Place", [], [ex(1, "We went on a Friday night. Kosher! The staff were lovely and kind.", 5)])
        v.summary = None
        v.summary_error = {"category": "rate_limit", "code": "x", "message": "m"}
        o = outcome([v], "kosher restaurants", diets=["kosher"], allergy=False)
        self.assertIn(" … ", vm.fallback_quote(v, o).text)


if __name__ == "__main__":
    unittest.main()
