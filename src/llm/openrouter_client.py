"""Lazy OpenRouter client configuration (an optional provider for the LLM judge only).

Requests go through src.llm.groq_client.request_json(..., provider="openrouter"), which reuses the
strict JSON parsing, validation and typed, redacted errors. Importing this module is offline.

Configuration: OPENROUTER_API_KEY (required), OPENROUTER_MODEL (default openai/gpt-oss-120b).
No optional attribution headers (HTTP-Referer / X-Title) are sent.
"""

import os

from src.llm.groq_client import LLMConfigurationError

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
PROVIDER_NAME = "OpenRouter"
_PLACEHOLDERS = {"your-openrouter-api-key-here", "replace_with_your_openrouter_key"}


def resolve_model(model=None):
    value = model if model is not None else os.environ.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL
    if not isinstance(value, str) or not value.strip() or value.strip() == "your-openrouter-model-id-here":
        raise LLMConfigurationError("Set OPENROUTER_MODEL or pass a nonempty model ID explicitly.", code="missing_model")
    return value.strip()


def get_client():
    """An OpenAI-compatible SDK client for OpenRouter, created on demand (never at import time)."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or key.lower() in _PLACEHOLDERS or any(c.isspace() for c in key):
        raise LLMConfigurationError("Set a non-placeholder OPENROUTER_API_KEY in local configuration.",
                                    code="invalid_api_key_configuration")
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMConfigurationError("Install the project's openai dependency.") from None
    failure = None
    try:
        client = OpenAI(api_key=key, base_url=BASE_URL, timeout=30.0, max_retries=0)
    except Exception:
        failure = LLMConfigurationError("Could not initialize the OpenRouter SDK client. Check local SDK configuration.",
                                        code="client_initialization_failed")
    if failure is not None:
        raise failure from None
    return client
