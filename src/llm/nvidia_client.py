"""Lazy NVIDIA API catalog client (build.nvidia.com; an optional provider for the LLM judge only).

Requests go through src.llm.groq_client.request_json(..., provider="nvidia"), which reuses the strict JSON
parsing, validation and typed, redacted errors. Importing this module is offline.

Configuration: NVIDIA_API_KEY (required), NVIDIA_MODEL (default openai/gpt-oss-120b).
Per NVIDIA's API reference for openai/gpt-oss-120b, reasoning effort is the top-level `reasoning_effort`
field (low/medium/high, default medium). The free trial's rate limits are not published (reported as about
40 requests per minute; see the account page on build.nvidia.com).
"""

import os

from src.llm.groq_client import LLMConfigurationError

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
PROVIDER_NAME = "NVIDIA"
TIMEOUT = 60.0  # reasoning models on the shared trial endpoint can be slower than Groq
_PLACEHOLDERS = {"your-nvidia-api-key-here", "replace_with_your_nvidia_key"}


def resolve_model(model=None):
    value = model if model is not None else os.environ.get("NVIDIA_MODEL", "").strip() or DEFAULT_MODEL
    if not isinstance(value, str) or not value.strip():
        raise LLMConfigurationError("Set NVIDIA_MODEL or pass a nonempty model ID explicitly.", code="missing_model")
    return value.strip()


def get_client():
    """An OpenAI-compatible SDK client for NVIDIA's API catalog, created on demand (never at import time)."""
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key or key.lower() in _PLACEHOLDERS or any(c.isspace() for c in key):
        raise LLMConfigurationError("Set a non-placeholder NVIDIA_API_KEY in local configuration.",
                                    code="invalid_api_key_configuration")
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMConfigurationError("Install the project's openai dependency.") from None
    failure = None
    try:
        client = OpenAI(api_key=key, base_url=BASE_URL, timeout=TIMEOUT, max_retries=0)
    except Exception:
        failure = LLMConfigurationError("Could not initialize the NVIDIA SDK client. Check local SDK configuration.",
                                        code="client_initialization_failed")
    if failure is not None:
        raise failure from None
    return client
