"""Lazy GroqCloud client and strict JSON handling; importing this module is offline."""

import json
import os


# Documentation snapshot: https://console.groq.com/docs/structured-outputs (2026-11-05).
# Account access is separate. Recheck these sets when choosing a new model.
STRICT_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b",
                           "qwen/qwen3.8-27b", "meta-models/muse-spark-open"})
SCHEMA_MODELS = STRICT_MODELS | {"openai/gpt-oss-safeguard-20b"}


class LLMConfigurationError(ValueError):
    """Required local configuration is missing."""


class LLMResponseError(ValueError):
    """The response is incomplete or violates the output contract."""


class LLMAPIError(RuntimeError):
    """The SDK request failed; details are deliberately not echoed."""


def get_client():
    """Create a GroqCloud-only SDK client on demand, never at import time."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key or key == "your-groq-api-key-here":
        raise LLMConfigurationError("Set GROQ_API_KEY before making a live request.")
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMConfigurationError("Install the project's openai dependency.") from None
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", timeout=30.0, max_retries=0)


def resolve_model(model=None):
    value = model if model is not None else os.environ.get("GROQ_MODEL", "")
    if not isinstance(value, str) or not value.strip() or value.strip() == "your-groq-model-id-here":
        raise LLMConfigurationError("Set GROQ_MODEL or pass a nonempty model ID explicitly.")
    return value.strip()


def exact_object(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise LLMResponseError("JSON object has missing or unexpected fields.")


def nonempty_string(value):
    if not isinstance(value, str) or not value.strip():
        raise LLMResponseError("Expected a nonempty string.")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LLMResponseError("Duplicate JSON object key.")
        result[key] = value
    return result


def _invalid_constant(value):
    raise LLMResponseError("Nonstandard JSON numeric constant.")


def request_json(*, client, model, name, schema, system, payload):
    """Request schema-constrained JSON; injected clients must be offline fakes or GroqCloud clients.

    SDK retries are disabled for clients we create. Injected clients own their policy.
    """
    model = resolve_model(model)
    if model not in SCHEMA_MODELS:
        raise LLMConfigurationError(
            "Model is not in the documented Groq JSON Schema support list; review model support first.")
    if client is None:
        client = get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            response_format={"type": "json_schema", "json_schema": {
                "name": name, "strict": model in STRICT_MODELS, "schema": schema}},
        )
    except Exception:
        # Do not propagate provider text: it may contain headers, keys or review data.
        raise LLMAPIError("GroqCloud request failed; check configuration, access and connectivity.") from None
    try:
        if len(response.choices) != 1:
            raise LLMResponseError("Expected exactly one response choice.")
        choice = response.choices[0]
        if choice.finish_reason != "stop" or getattr(choice.message, "refusal", None):
            raise LLMResponseError("The model refused or did not complete its response.")
        content = choice.message.content
        if not isinstance(content, str):
            raise LLMResponseError("Expected JSON text.")
        return json.loads(content, object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    except (AttributeError, TypeError, ValueError, IndexError):
        raise LLMResponseError("Invalid, refused or incomplete structured response.") from None
