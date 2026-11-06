"""Offline tests: no SDK installation, credentials or datasets needed."""

import copy
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.schemas import ParsedQuery
from src.llm.query_parser import parse_query
from src.llm.groq_client import (LLMAPIError, LLMConfigurationError, LLMResponseError,
                               get_client, resolve_model)


def fake_client(payload):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload), refusal=None))])
    return client


class QueryParserTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(location="Brooklyn", dietary_restrictions=["vegan"],
                         allergens=["peanuts (one guest)"],
                         group_needs=["one vegan friend", "one guest with a peanut allergy"],
                         preferences=["not a vegan-only restaurant"])

    def test_mixed_needs_and_negation_contract(self):
        client = fake_client(self.data)
        before = copy.deepcopy(self.data)
        result = parse_query("Brooklyn, one vegan friend, one guest allergic to peanuts; "
                             "not a vegan-only restaurant", client=client, model="openai/gpt-oss-20b")
        self.assertIsInstance(result, ParsedQuery)
        self.assertEqual(result, ParsedQuery(**before))
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "openai/gpt-oss-20b")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        self.assertFalse(request["response_format"]["json_schema"]["schema"]["additionalProperties"])
        self.assertIn("No nut allergy", request["messages"][0]["content"])
        self.assertIn("mixed", request["messages"][0]["content"])

    def test_unspecified_fields(self):
        payload = {key: [] for key in self.data if key != "location"}
        payload["location"] = None
        self.assertEqual(parse_query("dinner", client=fake_client(payload), model="openai/gpt-oss-20b"), ParsedQuery())

    def check_dietary_fixture(self, text, expected):
        """Checks contract preservation and prompt delivery, not real model comprehension."""
        client = fake_client(expected)
        parsed = parse_query(text, client=client, model="openai/gpt-oss-20b")
        self.assertEqual(parsed, ParsedQuery(**expected))
        self.assertEqual(parsed.allergens, [])
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(json.loads(messages[1]["content"])["query_text"], text)
        self.assertIn("Do not put a dietary\nrequirement ONLY in group_needs", messages[0]["content"])
        self.assertIn("WHO needs each option", messages[0]["content"])
        self.assertIn("Never infer an allergy", messages[0]["content"])
        return parsed

    def test_mixed_vegetarian_and_meat_diners(self):
        self.check_dietary_fixture(
            "My friend is vegetarian, but I eat meat. We want dinner in Manhattan.",
            dict(location="Manhattan", dietary_restrictions=["vegetarian"], allergens=[],
                 group_needs=["mixed dietary needs: friend is vegetarian; speaker eats meat"],
                 preferences=["meat options for the speaker", "dinner"]))

    def test_vegan_speaker_and_meat_eating_friends(self):
        self.check_dietary_fixture("I'm vegan but my friends eat meat.",
            dict(location=None, dietary_restrictions=["vegan"], allergens=[],
                 group_needs=["speaker is vegan; friends eat meat"],
                 preferences=["meat options for friends"]))

    def test_multiple_requested_dietary_options(self):
        self.check_dietary_fixture("We need halal and vegetarian options.",
            dict(location=None, dietary_restrictions=["halal", "vegetarian"], allergens=[],
                 group_needs=[], preferences=[]))

    def test_gluten_free_is_not_an_inferred_allergy(self):
        self.check_dietary_fixture("Find gluten-free pizza in Brooklyn.",
            dict(location="Brooklyn", dietary_restrictions=["gluten-free"], allergens=[],
                 group_needs=[], preferences=["pizza"]))

    def test_invalid_fields(self):
        variants = [{}, {**self.data, "extra": True}, {**self.data, "location": 3},
                    {**self.data, "allergens": "nuts"}, {**self.data, "preferences": [False]},
                    {**self.data, "group_needs": [""]}, {**self.data, "location": " "}, []]
        for payload in variants:
            with self.subTest(payload=payload), self.assertRaises(LLMResponseError):
                parse_query("dinner", client=fake_client(payload), model="openai/gpt-oss-20b")

    def test_invalid_json_and_incomplete_responses(self):
        for content in ("not JSON", "```json\n{}\n```", '{"location":null,"location":null}', 'NaN', None):
            client = fake_client(self.data)
            client.chat.completions.create.return_value.choices[0].message.content = content
            with self.subTest(content=content), self.assertRaises(LLMResponseError):
                parse_query("dinner", client=client, model="openai/gpt-oss-20b")
        for reason, refusal in (("length", None), ("stop", "refused")):
            client = fake_client(self.data)
            choice = client.chat.completions.create.return_value.choices[0]
            choice.finish_reason, choice.message.refusal = reason, refusal
            with self.assertRaises(LLMResponseError):
                parse_query("dinner", client=client, model="openai/gpt-oss-20b")

    def test_blank_input_never_requests(self):
        client = Mock()
        for text in ("", " ", None, 42):
            with self.assertRaises(ValueError):
                parse_query(text, client=client, model="openai/gpt-oss-20b")
        client.chat.completions.create.assert_not_called()

    def test_api_errors_redacted(self):
        for error in (TimeoutError("private detail"), RuntimeError("private detail")):
            client = Mock()
            client.chat.completions.create.side_effect = error
            with self.assertRaises(LLMAPIError) as caught:
                parse_query("dinner", client=client, model="openai/gpt-oss-20b")
            self.assertNotIn("private detail", str(caught.exception))

    def test_configuration_and_lazy_client(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMConfigurationError):
                get_client()
            with self.assertRaises(LLMConfigurationError):
                resolve_model()
            client = fake_client(self.data)
            parse_query("dinner", client=client, model="openai/gpt-oss-20b")  # no key needed
        # Sentinel used only with a mocked SDK constructor, never a credential or network request.
        sdk = SimpleNamespace(OpenAI=Mock())
        with patch.dict(sys.modules, {"openai": sdk}), patch.dict(
                os.environ, {"GROQ_API_KEY": "offline-sentinel", "GROQ_MODEL": "configured-model"}):
            self.assertEqual(resolve_model(), "configured-model")
            self.assertEqual(resolve_model("override"), "override")
            get_client()
            kwargs = sdk.OpenAI.call_args.kwargs
            self.assertEqual(kwargs["base_url"], "https://api.groq.com/openai/v1")
            self.assertEqual(kwargs["max_retries"], 0)
            self.assertEqual(kwargs["timeout"], 30.0)
        client = Mock()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(LLMConfigurationError):
            parse_query("dinner", client=client)
        client.chat.completions.create.assert_not_called()

    def test_groqcloud_environment_configuration(self):
        sdk = SimpleNamespace(OpenAI=Mock())
        sdk.OpenAI.return_value = fake_client(self.data)
        with patch.dict(sys.modules, {"openai": sdk}), patch.dict(os.environ, {
                "GROQ_API_KEY": "offline-sentinel", "GROQ_MODEL": "openai/gpt-oss-20b"}, clear=True):
            parsed = parse_query("Brooklyn dinner")
            kwargs = sdk.OpenAI.call_args.kwargs
            self.assertEqual(kwargs["api_key"], os.environ["GROQ_API_KEY"])
            self.assertEqual(kwargs["base_url"], "https://api.groq.com/openai/v1")
            request = sdk.OpenAI.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(request["model"], "openai/gpt-oss-20b")
            self.assertTrue(request["response_format"]["json_schema"]["strict"])
            self.assertEqual(parsed, ParsedQuery(**self.data))

    def test_model_support_checked_before_request(self):
        client = fake_client(self.data)
        with self.assertRaises(LLMConfigurationError):
            parse_query("dinner", client=client, model="unsupported-model")
        client.chat.completions.create.assert_not_called()
        parse_query("dinner", client=client, model="openai/gpt-oss-safeguard-20b")
        self.assertFalse(client.chat.completions.create.call_args.kwargs[
            "response_format"]["json_schema"]["strict"])

    def test_wire_schemas_use_conservative_groq_subset(self):
        from src.llm.query_parser import QUERY_SCHEMA
        from src.llm.evidence_summarizer import SUMMARY_SCHEMA

        def check(node):
            self.assertTrue(set(node) <= {"type", "properties", "required",
                                          "additionalProperties", "items", "enum"})
            if node.get("type") == "object":
                self.assertFalse(node["additionalProperties"])
                self.assertEqual(set(node["required"]), set(node["properties"]))
                for child in node["properties"].values():
                    check(child)
            if "items" in node:
                check(node["items"])

        check(QUERY_SCHEMA)
        check(SUMMARY_SCHEMA)


if __name__ == "__main__":
    unittest.main()
