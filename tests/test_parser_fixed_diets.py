"""Halal and kosher are always dietary restrictions, whatever the model returned (no model calls: fake client)."""

import json
import unittest
from types import SimpleNamespace

from src.llm import query_parser
from src.llm.query_parser import enforce_fixed_diets, normalize_diet, parse_query
from src.schemas import ParsedQuery


def client_returning(**fields):
    body = {"location": None, "dietary_restrictions": [], "allergens": [], "group_needs": [], "preferences": [], **fields}

    def create(**kwargs):
        message = SimpleNamespace(content=json.dumps(body))
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="m", usage=None)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def parse(text, **fields):
    return parse_query(text, client=client_returning(**fields), model="openai/gpt-oss-20b")


class FixedDietTests(unittest.TestCase):
    def test_halal_in_preferences_moves_to_dietary_restrictions(self):
        p = parse("halal restaurants", preferences=["halal restaurants"])
        self.assertEqual(p.dietary_restrictions, ["halal"])
        self.assertEqual(p.preferences, [])

    def test_kosher_missing_from_diets_is_added(self):
        p = parse("kosher restaurants", preferences=["kosher restaurants"])
        self.assertEqual(p.dietary_restrictions, ["kosher"])
        self.assertEqual(p.preferences, [])

    def test_the_model_already_right_changes_nothing(self):
        p = parse("halal restaurants", dietary_restrictions=["halal"])
        self.assertEqual((p.dietary_restrictions, p.preferences), (["halal"], []))

    def test_same_result_whichever_way_the_model_answers(self):
        answers = [dict(preferences=["halal restaurants"]), dict(dietary_restrictions=["halal"]),
                   dict(dietary_restrictions=["Halal food"], preferences=["halal"]), {}]
        results = [parse("halal restaurants", **a) for a in answers]
        self.assertEqual([(r.dietary_restrictions, r.preferences) for r in results], [(["halal"], [])] * 4)

    def test_other_words_in_a_preference_are_kept(self):
        p = parse("halal chinese food", preferences=["halal chinese food"])
        self.assertEqual((p.dietary_restrictions, p.preferences), (["halal"], ["chinese food"]))

    def test_both_terms_and_other_diets(self):
        p = parse("kosher or halal vegan options", dietary_restrictions=["Vegan"], preferences=["kosher", "halal"])
        self.assertEqual(p.dietary_restrictions, ["vegan", "halal", "kosher"])
        self.assertEqual(p.preferences, [])

    def test_group_needs_mentions_are_cleaned_too(self):
        p = parse("halal for my friend", group_needs=["halal"], preferences=[])
        self.assertEqual((p.dietary_restrictions, p.group_needs), (["halal"], []))

    def test_a_negated_mention_is_never_turned_into_a_requirement(self):
        for text in ("not halal food", "non-kosher is fine", "no halal needed"):
            p = parse(text, preferences=["anything"])
            self.assertEqual(p.dietary_restrictions, [], text)

    def test_normalization(self):
        self.assertEqual(normalize_diet(" Halal Food "), "halal")
        self.assertEqual(normalize_diet("Kosher-style"), "kosher")
        self.assertEqual(normalize_diet("Gluten Free"), "gluten-free")
        self.assertEqual(normalize_diet("Dairy free"), "dairy-free")
        self.assertEqual(normalize_diet("vegan"), "vegan")

    def test_allowed_values_and_prompt(self):
        self.assertIn("halal", query_parser.DIETARY_VALUES)
        self.assertIn("kosher", query_parser.DIETARY_VALUES)
        self.assertIn("Halal and kosher are\nalways dietary_restrictions", query_parser.SYSTEM)

    def test_enforce_is_idempotent(self):
        p = enforce_fixed_diets(ParsedQuery(preferences=["halal restaurants"]), "halal restaurants")
        again = enforce_fixed_diets(ParsedQuery(**p.__dict__), "halal restaurants")
        self.assertEqual(p, again)


if __name__ == "__main__":
    unittest.main()
