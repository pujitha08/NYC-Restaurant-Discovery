"""Lazy GroqCloud client and strict JSON handling; importing this module is offline."""

import json
import os
import re


# Documentation snapshot: https://console.groq.com/docs/structured-outputs (2026-11-05).
# Account access is separate. Recheck these sets when choosing a new model.
STRICT_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b",
                           "qwen/qwen3.8-27b", "meta-models/muse-spark-open"})
SCHEMA_MODELS = STRICT_MODELS | {"openai/gpt-oss-safeguard-20b"}


class LLMError(Exception):
    """Safe public error metadata. Never attach a raw SDK exception or response."""

    category = "llm"
    default_code = "llm_error"

    def __init__(self, message, *, code=None, http_status=None):
        super().__init__(message)
        self.code = code or self.default_code
        self.http_status = http_status


class LLMConfigurationError(LLMError, ValueError):
    """Required local configuration is missing."""

    category = "configuration"
    default_code = "invalid_configuration"


class LLMResponseError(LLMError, ValueError):
    """The response is incomplete or violates the output contract."""

    category = "response"
    default_code = "invalid_response"


class LLMAPIError(LLMError, RuntimeError):
    """The SDK request failed; details are deliberately not echoed."""

    category = "provider"
    default_code = "provider_failure"

    def __init__(self, message, *, category="provider", code=None, http_status=None):
        super().__init__(message, code=code, http_status=http_status)
        self.category = category


def _safe_api_error(exc, provider_name="GroqCloud"):
    """Classify supported SDK exceptions without copying provider text or arbitrary codes."""
    try:
        import openai
    except ImportError:
        openai = None  # Offline fakes and demo also work without the SDK installed.

    def is_sdk(name):
        cls = getattr(openai, name, None)
        return isinstance(cls, type) and isinstance(exc, cls)

    status = getattr(exc, "status_code", None) if is_sdk("APIStatusError") else None
    status = status if type(status) is int and 100 <= status <= 599 else None
    if is_sdk("APITimeoutError") or isinstance(exc, TimeoutError):
        category, code, message = "timeout", "request_timeout", "The model request timed out. Try again manually later."
    elif is_sdk("APIConnectionError") or isinstance(exc, ConnectionError):
        category, code, message = "network", "connection_failed", f"Could not reach {provider_name}. Check connectivity."
    elif is_sdk("AuthenticationError") or status == 401:
        category, code, message = "authentication", "authentication_failed", f"{provider_name} authentication failed. Check the local API key."
    elif status == 402:  # OpenRouter: insufficient credits
        category, code, message = "billing", "payment_required", f"{provider_name} requires payment or credits for this request."
    elif is_sdk("PermissionDeniedError") or status == 403:
        category, code, message = "permission", "permission_denied", f"{provider_name} denied access. Check account and model permissions."
    elif is_sdk("RateLimitError") or status == 429:
        category, code, message = "rate_limit", "rate_limited", f"{provider_name} rate limit reached. Wait before a manual retry and check quota."
    elif is_sdk("BadRequestError") or status == 400:
        category = "bad_request"
        # The SDK exposes parsed error codes. Only this known, non-sensitive value is retained.
        if getattr(exc, "code", None) == "output_parse_failed":
            code, message = "output_parse_failed", f"{provider_name} could not produce a parsable structured response. No result is available."
        else:
            code, message = "bad_request", f"{provider_name} rejected the request. Check the model and structured-output configuration."
    else:
        category, code, message = "provider", "provider_failure", "The model request failed. No result is available."
    return LLMAPIError(message, category=category, code=code, http_status=status)


def get_client():
    """Create a GroqCloud-only SDK client on demand, never at import time."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if (not key or key.lower() in {"your-groq-api-key-here", "replace_with_your_groq_key"}
            or any(c.isspace() for c in key)):
        raise LLMConfigurationError("Set a non-placeholder GROQ_API_KEY in local configuration.",
                                    code="invalid_api_key_configuration")
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMConfigurationError("Install the project's openai dependency.") from None
    failure = None
    try:
        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", timeout=30.0, max_retries=0)
    except Exception:
        failure = LLMConfigurationError("Could not initialize the GroqCloud SDK client. Check local SDK configuration.",
                                        code="client_initialization_failed")
    if failure is not None:
        raise failure from None
    return client


def resolve_model(model=None):
    value = model if model is not None else os.environ.get("GROQ_MODEL", "")
    if not isinstance(value, str) or not value.strip() or value.strip() == "your-groq-model-id-here":
        raise LLMConfigurationError("Set GROQ_MODEL or pass a nonempty model ID explicitly.", code="missing_model")
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


REASONING_EFFORTS = ("low", "medium", "high")


def supports_reasoning_effort(model):
    """Groq documents reasoning_effort low/medium/high for the gpt-oss models."""
    return model.startswith("openai/gpt-oss-")


def _usage(response):
    """Token counts from a response, as plain ints (None when not reported); nothing else is copied."""
    usage = getattr(response, "usage", None)

    def count(obj, name):
        value = getattr(obj, name, None)
        return value if type(value) is int and value >= 0 else None

    details = getattr(usage, "completion_tokens_details", None)
    return {"prompt_tokens": count(usage, "prompt_tokens"), "completion_tokens": count(usage, "completion_tokens"),
            "reasoning_tokens": count(details, "reasoning_tokens"), "total_tokens": count(usage, "total_tokens")}


PROVIDERS = ("groq", "openrouter")
_SERVED_BY = re.compile(r"^[\w .()/-]{1,64}$")


def _served_by(response):
    """OpenRouter's top-level "provider" field (which upstream served the request), if it is a short name."""
    value = getattr(response, "provider", None)
    return value if isinstance(value, str) and _SERVED_BY.match(value) else None


def request_json(*, client, model, name, schema, system, payload, reasoning_effort=None, return_usage=False,
                 provider="groq"):
    """Request schema-constrained JSON; injected clients must be offline fakes or clients for `provider`.

    SDK retries are disabled for clients we create. Injected clients own their policy.
    reasoning_effort ("low", "medium", "high") is sent only for models that support it (gpt-oss);
    other models ignore it. With return_usage=True, returns (parsed JSON, usage dict with
    prompt/completion/reasoning/total token counts, None where the provider does not report them).

    provider="groq" (default) is GroqCloud: the documented JSON Schema allow-list applies.
    provider="openrouter" (src.llm.openrouter_client; used by the LLM judge only) always requests
    strict json_schema and sends provider={"require_parameters": true}, so OpenRouter routes only
    to upstreams that support every request parameter; reasoning effort is sent as
    reasoning={"effort": ...}. Its usage dict also has "served_by" (the upstream, if reported).
    """
    if provider not in PROVIDERS:
        raise LLMConfigurationError(f"provider must be one of {PROVIDERS}.", code="invalid_provider")
    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
        raise LLMConfigurationError(f"reasoning_effort must be one of {REASONING_EFFORTS}.",
                                    code="invalid_reasoning_effort")
    if provider == "openrouter":
        from src.llm import openrouter_client
        model = openrouter_client.resolve_model(model)
        provider_name, strict = openrouter_client.PROVIDER_NAME, True
        body = {"provider": {"require_parameters": True}}
        if reasoning_effort and supports_reasoning_effort(model):
            body["reasoning"] = {"effort": reasoning_effort}
        extra = {"extra_body": body}
        if client is None:
            client = openrouter_client.get_client()
    else:
        model = resolve_model(model)
        if model not in SCHEMA_MODELS:
            raise LLMConfigurationError(
                "Model is not in the documented Groq JSON Schema support list; review model support first.",
                code="unsupported_model")
        provider_name, strict = "GroqCloud", model in STRICT_MODELS
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort and supports_reasoning_effort(model) else {}
        if client is None:
            client = get_client()
    failure = None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            response_format={"type": "json_schema", "json_schema": {
                "name": name, "strict": strict, "schema": schema}},
            **extra,
        )
    except Exception as exc:
        failure = _safe_api_error(exc, provider_name)
    # Raise outside the handler so the public exception has no sensitive __context__.
    if failure is not None:
        raise failure from None
    try:
        if len(response.choices) != 1:
            raise LLMResponseError("Expected exactly one response choice.")
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None) or choice.finish_reason == "content_filter":
            raise LLMResponseError("The model declined to provide a structured response.", code="refused_response")
        if choice.finish_reason != "stop":
            raise LLMResponseError("The model response was incomplete.", code="incomplete_response")
        content = choice.message.content
        if not isinstance(content, str):
            raise LLMResponseError("Expected JSON text.")
        data = json.loads(content, object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
        if not return_usage:
            return data
        usage = _usage(response)
        if provider == "openrouter":
            usage["served_by"] = _served_by(response)
        return data, usage
    except LLMResponseError as exc:
        failure = exc
    except (AttributeError, TypeError, ValueError, IndexError):
        failure = LLMResponseError("Invalid structured model response.")
    raise failure from None
