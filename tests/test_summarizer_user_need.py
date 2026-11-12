"""Summarizer labels relative to the user's stated need (fix/summarizer-user-need).

Regression for reports/failure_examples.md case 9 (Q20 / Mission). Offline tests check the request
(prompt definitions, user_request in the payload, the app passing the exact words); whether the model
follows them is checked live. Fake SDK client only.
"""

import json
import unittest
from types import SimpleNamespace

from app import pipeline
from src.llm import evidence_summarizer
from src.llm.evidence_summarizer import MAX_USER_REQUEST, summarize_evidence
from src.schemas import ParsedQuery, RetrievalResult

from tests.test_allergy_guard import MISSION_ID, MISSION_SNIPPETS, Q20, MissionRetriever, parsed

MODEL = "openai/gpt-oss-20b"
# What the fixed summarizer should return for Mission: both warnings are contradictions of the need.
MISSION_FIXED = {"status": "evidence_found", "claims": [
    {"text": "One reviewer advised people with food allergies to avoid the restaurant, reporting inattentive staff.",
     "review_ids": [1576831], "label": "contradiction"},
    {"text": "Another reviewer reported no concern from staff about nut and wheat allergies and suggested going elsewhere.",
     "review_ids": [1576786], "label": "contradiction"}],
    "limitations": ["Neither review mentions peanuts specifically."]}


class Client:
    def __init__(self, summary=MISSION_FIXED, parsed_body=None):
        self.summary, self.parsed_body, self.calls = summary, parsed_body, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        name = kwargs["response_format"]["json_schema"]["name"]
        body = self.parsed_body if name == "parsed_query" else self.summary
        msg = SimpleNamespace(content=json.dumps(body), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)


def mission():
    return RetrievalResult(MISSION_ID, 2.13, "embedding", list(MISSION_SNIPPETS))


class PromptTests(unittest.TestCase):
    def test_labels_are_defined_against_the_users_need(self):
        system = evidence_summarizer.SYSTEM
        self.assertIn("relative to the USER'S NEED", system)
        self.assertIn("contradiction = the reviewer reports the opposite: a warning, a refusal", system)
        self.assertIn("advice to avoid the restaurant", system)
        self.assertIn("A reviewer warning against the restaurant is never support.", system)
        self.assertIn("mere mention of the allergen", system)
        self.assertNotIn("Label each support, contradiction, or uncertainty relative to the query.", system)


class PayloadTests(unittest.TestCase):
    def test_user_request_is_sent_as_data(self):
        client = Client()
        summary = summarize_evidence(ParsedQuery(allergens=["peanut"]), mission(), client=client, model=MODEL,
                                     user_request=f"  {Q20}  ")
        payload = json.loads(client.calls[0]["messages"][1]["content"])
        self.assertEqual(payload["user_request"], Q20)  # stripped
        self.assertEqual(payload["query"]["allergens"], ["peanut"])
        self.assertEqual([c.label for c in summary.claims], ["contradiction", "contradiction"])

    def test_without_user_request_the_payload_is_unchanged(self):
        client = Client()
        summarize_evidence(ParsedQuery(allergens=["peanut"]), mission(), client=client, model=MODEL)
        self.assertEqual(set(json.loads(client.calls[0]["messages"][1]["content"])), {"query", "snippets"})

    def test_invalid_user_request_never_requests(self):
        for bad in ("", "   ", 42, "x" * (MAX_USER_REQUEST + 1)):
            with self.subTest(bad=bad):
                client = Client()
                with self.assertRaises(ValueError):
                    summarize_evidence(ParsedQuery(), mission(), client=client, model=MODEL, user_request=bad)
                self.assertEqual(client.calls, [])


class AppTests(unittest.TestCase):
    def test_app_passes_the_exact_request_and_mission_is_not_supported(self):
        client = Client(parsed_body=parsed(allergens=["peanut"]))
        out, _ = pipeline.run_search(Q20, "Both", "semantic_capped", retriever=MissionRetriever(),
                                     client=pipeline.UsageRecorder(client), model=MODEL, log_path=None)
        summary_call = next(c for c in client.calls if c["response_format"]["json_schema"]["name"] == "evidence_summary")
        self.assertEqual(json.loads(summary_call["messages"][1]["content"])["user_request"], Q20)
        s = out.results[0].summary
        self.assertEqual([c.label for c in s.claims], ["contradiction", "contradiction"])
        self.assertNotEqual(s.status, pipeline.STATUS_SUPPORTED)
        self.assertEqual(s.status, pipeline.STATUS_PROBLEMS)  # contradictions only -> "Reviews report problems"

    def test_long_requests_are_truncated_for_the_summarizer(self):
        client = Client(parsed_body=parsed(allergens=["peanut"]))
        long_query = "careful about peanut allergies " * 30
        pipeline.run_search(long_query, "Both", "semantic_capped", retriever=MissionRetriever(),
                            client=pipeline.UsageRecorder(client), model=MODEL, log_path=None)
        call = next(c for c in client.calls if c["response_format"]["json_schema"]["name"] == "evidence_summary")
        self.assertLessEqual(len(json.loads(call["messages"][1]["content"])["user_request"]), MAX_USER_REQUEST)


if __name__ == "__main__":
    unittest.main()
