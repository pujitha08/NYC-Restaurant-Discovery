"""Challenge fixtures and offline contract tests; expected outputs are authored mocks.

Run with unittest discovery. Importing this module never loads .env or calls an API.
These fixtures are also reused for the explicitly authorized live challenge run.
"""

import copy
import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

from src.schemas import ParsedQuery, RetrievalResult, Snippet
from src.llm.query_parser import parse_query
from src.llm.evidence_summarizer import summarize_evidence


PARSER_CASES = [
    ("Q1", "I need a restaurant that can accommodate a sesame allergy.",
     ParsedQuery(allergens=["sesame"], group_needs=["sesame allergy accommodation"])),
    ("Q2", "Find a restaurant where vegans and meat eaters can dine together.",
     ParsedQuery(dietary_restrictions=["vegan"],
                 group_needs=["vegans and meat eaters dining together"],
                 preferences=["vegan and meat options"])),
    ("Q3", "I have celiac disease and need a dedicated gluten-free preparation area.",
     ParsedQuery(dietary_restrictions=["gluten-free"],
                 group_needs=["celiac disease; dedicated gluten-free preparation area"])),
    ("Q4", "Find a vegetarian restaurant that also has dairy-free options.",
     ParsedQuery(dietary_restrictions=["vegetarian", "dairy-free"],
                 preferences=["vegetarian restaurant with dairy-free options"])),
]

EVIDENCE_QUERY = PARSER_CASES[0][2]
EVIDENCE_RESULT = RetrievalResult("synthetic-sesame-challenge", 0.5, "synthetic", [
    Snippet("During my 2019 visit, staff discussed my sesame allergy and prepared a meal without sesame ingredients at my request.",
            301, 1546300800000),
    Snippet("During my 2020 visit, staff refused my request for sesame-allergy accommodation because they could not avoid cross-contact.",
            302, 1577836800000),
    Snippet("During my 2021 visit, I enjoyed the sesame noodles and their nutty flavor.",
            303, 1609459200000),
])
EXPECTED_EVIDENCE = {
    "status": "evidence_found",
    "claims": [
        {"text": "One reviewer reported staff discussing a sesame allergy and preparing a meal without sesame ingredients during a 2019 visit.",
         "review_ids": [301], "label": "support"},
        {"text": "Another reviewer reported refusal of sesame-allergy accommodation over cross-contact concerns during a 2020 visit.",
         "review_ids": [302], "label": "contradiction"},
        {"text": "A third reviewer described sesame noodles, without discussing allergy accommodation.",
         "review_ids": [303], "label": "uncertainty"},
    ],
    "limitations": ["Historical, conflicting reports do not establish current policies or allergy safety."],
}


def fake_client(payload):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(refusal=None, content=json.dumps(payload)))])
    return client


class ChallengeParserTests(unittest.TestCase):
    def check_case(self, index):
        _, text, expected = PARSER_CASES[index]
        client = fake_client(asdict(expected))
        actual = parse_query(text, client=client, model="openai/gpt-oss-20b")
        self.assertEqual(actual, expected)
        self.assertIsNone(actual.location)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(json.loads(request["messages"][1]["content"]), {"query_text": text})
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        return actual

    def test_sesame_allergy_and_accommodation_survive_contract(self):
        actual = self.check_case(0)
        self.assertEqual(actual.allergens, ["sesame"])
        self.assertEqual(actual.preferences, [])
        self.assertIn("accommodation", actual.group_needs[0])

    def test_vegan_and_meat_eater_group_without_location(self):
        actual = self.check_case(1)
        self.assertEqual(actual.dietary_restrictions, ["vegan"])
        self.assertIn("meat eaters", actual.group_needs[0])
        self.assertEqual(actual.allergens, [])

    def test_celiac_and_dedicated_preparation_survive_contract(self):
        actual = self.check_case(2)
        self.assertIn("gluten-free", actual.dietary_restrictions)
        self.assertIn("celiac", actual.group_needs[0])
        self.assertIn("dedicated", actual.group_needs[0])

    def test_vegetarian_and_dairy_free_are_not_replaced_by_vegan(self):
        actual = self.check_case(3)
        self.assertEqual(set(actual.dietary_restrictions), {"vegetarian", "dairy-free"})
        self.assertNotIn("vegan", actual.dietary_restrictions)
        self.assertEqual(actual.allergens, [])


class ChallengeEvidenceTests(unittest.TestCase):
    def test_historical_accommodation_refusal_and_unrelated_mention(self):
        before = copy.deepcopy(EVIDENCE_RESULT)
        client = fake_client(EXPECTED_EVIDENCE)
        actual = summarize_evidence(EVIDENCE_QUERY, EVIDENCE_RESULT,
                                    client=client, model="openai/gpt-oss-20b")
        self.assertEqual([c.label for c in actual.claims], ["support", "contradiction", "uncertainty"])
        self.assertEqual([c.review_ids for c in actual.claims], [[301], [302], [303]])
        self.assertTrue(all("reviewer" in c.text for c in actual.claims))
        self.assertIn("current policies", actual.limitations[0])
        request = client.chat.completions.create.call_args.kwargs
        sent = json.loads(request["messages"][1]["content"])
        self.assertEqual(sent["snippets"], [asdict(s) for s in EVIDENCE_RESULT.evidence_snippets])
        self.assertEqual(sent["query"], asdict(EVIDENCE_QUERY))
        self.assertEqual(EVIDENCE_RESULT, before)

    def test_dish_mention_alone_is_insufficient_in_authored_response(self):
        unrelated = RetrievalResult("fake", 0.1, "synthetic", [EVIDENCE_RESULT.evidence_snippets[2]])
        client = fake_client({"status": "insufficient_evidence", "claims": [],
                              "limitations": ["Sesame noodles do not describe allergy accommodation or current safety."]})
        actual = summarize_evidence(EVIDENCE_QUERY, unrelated, client=client, model="openai/gpt-oss-20b")
        self.assertEqual(actual.status, "insufficient_evidence")
        self.assertEqual(actual.claims, [])
        self.assertIn("do not establish allergen safety", actual.limitations[-1])


if __name__ == "__main__":
    unittest.main()
