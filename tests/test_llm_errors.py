"""Offline classification using real SDK exception instances; no transport is called."""

import json
import os
import sys
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.llm.groq_client import LLMError, LLMAPIError, LLMConfigurationError, LLMResponseError, get_client
from src.llm.query_parser import parse_query
from src.llm.evidence_summarizer import summarize_evidence
from src.schemas import ParsedQuery, RetrievalResult, Snippet

try:
    import openai
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
except ImportError:
    openai = None

PRIVATE = "SIMULATED_SECRET_REVIEW_AND_TOKEN"
MODEL = "openai/gpt-oss-20b"


def response_client(content="{}", finish="stop", refusal=None):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content, refusal=refusal))])
    return client


class SafeErrorContractTests(unittest.TestCase):
    def test_builtin_timeout_is_safe_and_has_no_sensitive_context(self):
        client = Mock()
        client.chat.completions.create.side_effect = TimeoutError(PRIVATE)
        with self.assertRaises(LLMAPIError) as caught:
            parse_query("dinner", client=client, model=MODEL)
        error = caught.exception
        self.assertEqual((error.category, error.code), ("timeout", "request_timeout"))
        self.assertIsNone(error.__context__)
        self.assertIsNone(error.__cause__)
        self.assertNotIn(PRIVATE, str(error))
        self.assertIsInstance(error, RuntimeError)
        client.chat.completions.create.assert_called_once()

    def test_local_configuration_rejects_empty_placeholder_or_whitespace_key(self):
        for key in ("", "REPLACE_WITH_YOUR_GROQ_KEY", "your-groq-api-key-here", "invalid key"):
            with self.subTest(key_type="invalid local value"), patch.dict(os.environ, {"GROQ_API_KEY": key}):
                with self.assertRaises(LLMConfigurationError) as caught:
                    get_client()
                self.assertEqual(caught.exception.category, "configuration")
                self.assertEqual(caught.exception.code, "invalid_api_key_configuration")
                self.assertIsInstance(caught.exception, ValueError)

    def test_sdk_initialization_failure_does_not_leak(self):
        sdk = SimpleNamespace(OpenAI=Mock(side_effect=RuntimeError(PRIVATE)))
        with patch.dict(sys.modules, {"openai": sdk}), patch.dict(os.environ, {"GROQ_API_KEY": "offline-sentinel"}):
            with self.assertRaises(LLMConfigurationError) as caught:
                get_client()
        self.assertEqual(caught.exception.code, "client_initialization_failed")
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn(PRIVATE, str(caught.exception))

    def test_model_configuration_codes(self):
        for model, code in (("", "missing_model"), ("unsupported-model", "unsupported_model")):
            client = Mock()
            with self.assertRaises(LLMConfigurationError) as caught:
                parse_query("dinner", client=client, model=model)
            self.assertEqual(caught.exception.code, code)
            client.chat.completions.create.assert_not_called()

    def test_invalid_refused_and_incomplete_response_codes(self):
        for client, code in (
            (response_client(PRIVATE), "invalid_response"),
            (response_client("{}", refusal=PRIVATE), "refused_response"),
            (response_client("{}", finish="content_filter"), "refused_response"),
            (response_client(PRIVATE, finish="length"), "incomplete_response"),
            (response_client(json.dumps({"unexpected": PRIVATE})), "invalid_response"),
        ):
            with self.subTest(code=code), self.assertRaises(LLMResponseError) as caught:
                parse_query("dinner", client=client, model=MODEL)
            error = caught.exception
            self.assertEqual((error.category, error.code), ("response", code))
            self.assertIsNone(error.__context__)
            self.assertNotIn(PRIVATE, str(error))


@unittest.skipIf(openai is None, "Install existing openai dependency to test real SDK exception classes")
class SDKFailureTests(unittest.TestCase):
    def status_error(self, cls, status, code=None):
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(status, request=request, headers={"X-Private": PRIVATE})
        return cls(PRIVATE, response=response, body={"message": PRIVATE, "code": code})

    def assert_failure(self, exc, category, code, status=None):
        client = Mock()
        client.chat.completions.create.side_effect = exc
        # Capture formatted traceback inside the handler, before unittest clears it.
        try:
            parse_query("dinner", client=client, model=MODEL)
        except LLMAPIError as error:
            self.assertIsInstance(error, LLMError)
            self.assertEqual((error.category, error.code, error.http_status), (category, code, status))
            self.assertIsNone(error.__context__)
            self.assertIsNone(error.__cause__)
            self.assertNotIn(PRIVATE, "".join(traceback.format_exception(error)))
            self.assertNotIn(PRIVATE, repr(vars(error)))
        else:
            self.fail("Expected a safe LLMAPIError")
        client.chat.completions.create.assert_called_once()

    def test_bad_request_output_parse_failed(self):
        self.assert_failure(self.status_error(openai.BadRequestError, 400, "output_parse_failed"),
                            "bad_request", "output_parse_failed", 400)

    def test_bad_request_arbitrary_code_is_not_exposed(self):
        self.assert_failure(self.status_error(openai.BadRequestError, 400, PRIVATE),
                            "bad_request", "bad_request", 400)

    def test_rate_limit(self):
        self.assert_failure(self.status_error(openai.RateLimitError, 429), "rate_limit", "rate_limited", 429)

    def test_authentication(self):
        self.assert_failure(self.status_error(openai.AuthenticationError, 401),
                            "authentication", "authentication_failed", 401)

    def test_permission(self):
        self.assert_failure(self.status_error(openai.PermissionDeniedError, 403),
                            "permission", "permission_denied", 403)

    def test_timeout(self):
        self.assert_failure(openai.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com")),
                            "timeout", "request_timeout")

    def test_network(self):
        self.assert_failure(openai.APIConnectionError(message=PRIVATE,
                            request=httpx.Request("POST", "https://api.groq.com")), "network", "connection_failed")

    def test_other_provider_failures(self):
        for status in (404, 500, 503):
            with self.subTest(status=status):
                self.assert_failure(self.status_error(openai.APIStatusError, status),
                                    "provider", "provider_failure", status)

    def test_summarizer_uses_same_safe_rate_limit_contract(self):
        client = Mock()
        client.chat.completions.create.side_effect = self.status_error(openai.RateLimitError, 429)
        with self.assertRaises(LLMAPIError) as caught:
            summarize_evidence(ParsedQuery(), RetrievalResult("fake", 0.1, "synthetic", [Snippet("Text", 1)]),
                               client=client, model=MODEL)
        self.assertEqual(caught.exception.category, "rate_limit")
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn(PRIVATE, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
