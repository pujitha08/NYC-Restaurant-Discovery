# GroqCloud LLM modules

These modules use the existing shared dataclasses without changing retrieval or the app.
No API access or credits are assumed. The README's original GPT-4o-mini references
describe the earlier plan; these components target GroqCloud instead.

## Public interfaces

```python
parse_query(text, *, client=None, model=None) -> ParsedQuery
summarize_evidence(query, result, *, client=None, model=None) -> EvidenceSummary
```

Import from `src.llm.query_parser` and `src.llm.evidence_summarizer`.
The parser requires nonblank text and returns the existing `ParsedQuery` fields:
nullable location and lists of dietary_restrictions, allergens, group_needs and preferences.
Its prompt preserves negation, distinguishes allergies from preferences, and keeps
person qualifiers for mixed groups. Validation checks exact keys and types, not whether
the model correctly understood the original sentence. Ambiguities need user clarification
by the calling application.

The summarizer accepts `ParsedQuery` and `RetrievalResult`. It returns local dataclasses:
`EvidenceSummary(status, claims, limitations)` and
`EvidenceClaim(text, review_ids, label)`. Labels are `support`, `contradiction`, or
`uncertainty` relative to the query. Status is `evidence_found` or
`insufficient_evidence`, never a safety judgment. Claims are capped at five and 500
characters each. Every claim requires unique integer citations from the supplied snippets.
Empty snippets return insufficient evidence immediately, with no client or model needed.
Nonempty but irrelevant evidence can also yield insufficient evidence.

Reviews are serialized as data in the user message; the system prompt explicitly treats
embedded instructions as untrusted. Claims must be attributed to reviewers; the prompt
forbids restaurant safety guarantees. A fixed limitation accompanies every result.
These measures cannot guarantee prompt-injection resistance or factual correctness.
Citation membership and JSON schema checks **cannot prove semantic truth**, detect every
unsupported assertion, or establish that a citation actually entails its claim. Human
review of live model output is required before deployment, especially for allergy queries.

## Configuration and request behavior

The existing requirements include `openai` and `python-dotenv`; no new dependency is added.
Importing the modules does not import the SDK or contact a service. A live call creates
an OpenAI-compatible SDK client pointed explicitly to `https://api.groq.com/openai/v1`, using
`GROQ_API_KEY`, a 30-second timeout and zero automatic retries. It does not contact
OpenAI's hosted API. No key is bundled.

Set `GROQ_MODEL` to a supported model ID your GroqCloud account can access, or pass
`model=` explicitly. `.env.example` provisionally sets `openai/gpt-oss-20b`; this
does not establish account access. There is no runtime default. Missing configuration raises
`LLMConfigurationError`; invalid/incomplete/refused output raises `LLMResponseError`;
request failures raise a categorized `LLMAPIError` without raw provider details or credentials.
No fallback silently converts a failed request into success.

Checked on 2026-11-05: Groq's [structured-output documentation](https://console.groq.com/docs/structured-outputs)
lists `openai/gpt-oss-20b` as supporting strict JSON Schema outputs. Requests use
`response_format={"type": "json_schema", "json_schema": {"name": ..., "strict": ..., "schema": ...}}`.
All wire-schema object fields are required and additional properties are forbidden.
Wire schemas use basic types (including nullable location), enums, objects and arrays;
string/array length constraints remain enforced by local validation to avoid relying
on undocumented strict-mode keywords. Citation checks remain local too.
`groq_client.STRICT_MODELS` contains the documented strict-capable IDs;
`openai/gpt-oss-safeguard-20b` uses best-effort mode. Other model IDs are rejected before
client initialization until support is reviewed. Neither list guarantees account access.
The [OpenAI compatibility guide](https://console.groq.com/docs/openai) documents the
SDK endpoint configuration. No additional SDK or dependency is needed.

Environment files are not loaded automatically. For this project, only when an explicit
live test is authorized, load credentials from the local gitignored `.env` with
`dotenv.load_dotenv()` and confirm it is ignored before use. The client consumes
`GROQ_API_KEY` from that process environment. Never use credentials from chats or
screenshots, and never put them in source control or logs. Live requests must be explicitly authorized.
An injected `client` must implement `client.chat.completions.create(...)`; use an offline
fake in tests or an explicitly GroqCloud-configured SDK client in production. Injected clients
own their endpoint, timeout and retry policy; do not inject a default OpenAI-hosted client.

## Integration example (future live usage; not run or verified)

```python
from src.llm.query_parser import parse_query
from src.llm.evidence_summarizer import summarize_evidence

# Only run after provisioning GroqCloud configuration and authorizing paid API usage.
raw_query = "Brooklyn brunch for one vegan friend and a guest allergic to peanuts"
parsed = parse_query(raw_query)

# Existing retrievers accept text and optional boroughs, not ParsedQuery.
# This minimal example retains the original wording and maps only an exact supported borough.
# A teammate must define a richer location/constraint policy before integrating it.
boroughs = [parsed.location] if parsed.location in ("Manhattan", "Brooklyn") else None
results = retriever.search(raw_query, boroughs=boroughs)
summaries = [summarize_evidence(parsed, result) for result in results]
```

No production adapter is implemented here. The offline demonstration below provides
minimal wiring with a synthetic retriever. Do not blindly concatenate allergen names into search
text: doing so can lose negation or group qualifiers. Dietary fields are not enforced
as hard filters by current retrievers, and their evidence selector still uses an IDF
heuristic rather than parsed requirements. Retrieval scores are not safety probabilities.
Metadata such as restaurant name and low_evidence is separate (`retriever.info(gmap_id)`).

Retrieval-owner concern: `ChunkTfidfRetriever.search` slices
`hit[start:start + TOP_CHUNKS]` without bounding the slice to that restaurant's owner
range before `_evidence` loads review IDs. A restaurant with fewer than three matching
chunks may receive another restaurant's evidence. This is a static inspection concern,
not a reproduced integration test. `Snippet` has no restaurant ID, so this summarizer
cannot independently check ownership. The retrieval owner must confirm/fix it before
end-to-end grounding is claimed. Retrieval code is unchanged.

## Offline verification

From the repository root:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q src/llm tests
```

Tests use shared dataclasses, synthetic evidence, mocked responses and a mocked SDK
constructor. The SDK-exception regression tests use real exception objects from the
existing `openai` dependency without calling a transport; those tests skip if the SDK
is absent. The remaining offline tests and demo also work without the SDK installed.
No tests need a dataset, credentials or network access. They test mixed/negated fixture preservation, schema
validation, citation failures, contradictory reports, uncertainty, empty evidence,
API errors, lazy initialization and endpoint configuration. Fixtures are not evidence
that a real model extracts mixed needs correctly.

Not verified offline: real SDK/API interoperability, chosen-model availability and JSON
schema behavior, cost/latency, extraction accuracy, semantic grounding, robustness to
adversarial reviews, processed-data retrieval and Streamlit integration.
Team decisions remain: model/budget, canonical allergy/diet vocabulary, multi-location
and neighborhood mapping, person-specific constraints, and summary UI/error handling.

GroqCloud verification on 2026-11-05 with Python 3.11.0: all 22 unittest tests passed.
The earlier 17-test suite also passed with socket connections blocked. Compilation and
`git diff --check` passed. Tests include a mocked SDK constructor verifying GroqCloud's
endpoint and environment configuration, model support checks and wire-schema structure.
These are offline mock tests; they make no live API requests and do not establish
model comprehension. They include the mixed-diet and reviewer-attribution regressions.

## Limited live diagnostic

A later diagnostic on 2026-11-05 made exactly one request, with retries disabled,
using the existing strict schema and installed OpenAI-compatible SDK 3.46.0.
`Find vegan-friendly restaurants in Brooklyn.` returned location `Brooklyn`,
dietary_restrictions `["vegan"]`, and empty allergens, group_needs and preferences.
This matches the input and confirms this parser request path worked with
`openai/gpt-oss-20b` on GroqCloud for that request.

Subsequent live mixed-group parsing preserved Manhattan, vegetarian requirements,
person-specific needs and meat options, without inventing allergens. A synthetic
summarizer test with review IDs 101 and 202 produced third-person reviewer attribution,
valid citations, support and contradiction labels, and conflicting-evidence and
allergy-safety limitations. Only a few synthetic examples were checked; these are not
a general extraction or faithfulness evaluation.

Intermittent HTTP 400 failures were observed, including `output_parse_failed` during
summarization. A later request with the same strict schema and updated prompt succeeded.
The underlying causes remain unknown; no schema or SDK defect was confirmed. Successful
requests do not establish consistent reliability, ongoing quota or model availability.
The credential-bearing `.env.txt` file is now ignored as well as `.env`; neither belongs
in Git. No secret values are included in this diagnostic record.

## Safe error contract and UI handling

`parse_query` and `summarize_evidence` keep their existing signatures, dataclass outputs,
strict output/citation validation, and exception types. A common `LLMError` base now
allows callers to handle their errors together. `LLMConfigurationError` and
`LLMResponseError` remain subclasses of `ValueError`; `LLMAPIError` remains a subclass
of `RuntimeError`. Each has a safe `category`, `code`, and nullable `http_status`.
The API mapping uses supported OpenAI SDK exception classes when available.

| Exception/category | Codes | Suggested UI behavior |
|---|---|---|
| `LLMConfigurationError` / `configuration` | `invalid_configuration`, `invalid_api_key_configuration`, `missing_model`, `unsupported_model`, `client_initialization_failed` | Show configuration guidance; no model result exists. Never echo the supplied configuration. |
| `LLMAPIError` / `authentication` | `authentication_failed` | Ask the user to check the local key outside the UI/chat. |
| `LLMAPIError` / `permission` | `permission_denied` | Explain that account/model access must be checked. |
| `LLMAPIError` / `rate_limit` | `rate_limited` | Indicate rate/quota limits; offer a later manual retry. |
| `LLMAPIError` / `timeout` | `request_timeout` | Show a timeout message and an optional manual retry. |
| `LLMAPIError` / `network` | `connection_failed` | Suggest checking connectivity. |
| `LLMAPIError` / `bad_request` | `bad_request`, `output_parse_failed` | Show request/structured-output failure; flag for developer review. Do not display a partial summary. |
| `LLMAPIError` / `provider` | `provider_failure` | Show temporary unavailability or developer-review guidance. Do not infer whether billing or model availability is the cause. |
| `LLMResponseError` / `response` | `invalid_response`, `refused_response`, `incomplete_response` | Explain that no usable structured response is available; do not manufacture a query or claims. |

Only the known provider code `output_parse_failed` is retained, when the SDK exposes
that exact code. Arbitrary provider codes, bodies, messages, headers, review text and
request objects are never copied into public error metadata. API errors are raised
outside the original exception handler so the public exception does not chain a raw
SDK failure. Normal formatted tracebacks are checked against simulated secrets. Do not
log SDK exceptions, response bodies or traceback locals; local-variable capture can
expose request data independently of an exception's safe public fields.

```python
from src.llm.groq_client import LLMError

try:
    parsed = parse_query(raw_query)
except LLMError as error:
    ui_error = {"category": error.category, "code": error.code,
                "http_status": error.http_status, "message": str(error)}
    # Present ui_error; stop this parser attempt before retrieval.
```

Caller programming/input errors (for example blank parser text or a non-ParsedQuery
summary input) retain their existing `ValueError`/`TypeError` behavior; validate those
before calling. Errors are not successful empty results. A valid empty snippet list
still returns deterministic `insufficient_evidence` without a request. Nothing retries,
switches models, or substitutes results automatically. Billing and other provider
failures cannot always be distinguished from HTTP status alone.

## Fully offline pipeline demonstration

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe examples\llm_pipeline_demo.py
.\.venv\Scripts\python.exe examples\llm_pipeline_demo.py --scenario all
```

No `.env` is loaded. Every scenario uses fake restaurants, historical review snippets,
and scripted fake SDK responses injected into the REAL `parse_query` and
`summarize_evidence` functions. There is no live mode, index download or paid request.
The fake model does not understand arbitrary text; its canned outputs are for the
example fixtures only. The fake retriever is not a search-quality benchmark.

Expected successful behavior:

- A synthetic vegetarian/sesame-allergy query parses into the existing dataclass.
- The exact original raw query is passed to `search(query_terms, boroughs=...)`.
- Brooklyn is filtered only because it is an exact supported borough and is explicitly
  present in the raw query. Neighborhoods, unknown locations, or unsupported boroughs
  do not become borough filters. Unresolved locations are not asserted to be satisfied.
- The matching synthetic restaurant returns `RetrievalResult` with `Snippet` evidence.
- Output contains support `[reviews 101]` for a 2019 experience and contradiction
  `[reviews 102]` for a 2020 refusal, followed by uncertainty and allergy limitations.

`--scenario all` also demonstrates a parser response error before any search, no
search results without a summary request, a simulated summary timeout with a safe
message instead of fabricated claims, and a matched restaurant with no review evidence.
The last case reports `insufficient_evidence` with limitations and no claims or summary
SDK request; it is distinct from no matches and from a summary failure.
Individual scenarios are `parser-error`, `no-results`, `summary-error`, and
`insufficient-evidence`. Deliberately simulated failures are handled by the
example rather than crashing. The demo's `run_pipeline` returns a status string so
its branches are testable; it is not a new production API.

### Connecting real components later

1. Use this example's call ordering, but replace `FakeRetriever` with an explicitly
   loaded real retriever after the data/index owner provisions artifacts.
2. Replace `FakeSDKClient` only after separately authorizing live requests and configuring
   a GroqCloud client from the local ignored `.env`. The normal helper
   uses GroqCloud's endpoint with retries disabled. Do not inject an OpenAI-hosted client.
3. Preserve raw query wording, especially negation and mixed-group qualifiers. The
   existing retrievers accept text plus boroughs, not `ParsedQuery`. Do not treat diets
   or allergens as enforced database filters or retrieval scores as safety probabilities.
4. Decide and validate the location policy with the retrieval/app owner; this example
   intentionally does not infer boroughs from neighborhoods or silently satisfy unknown
   locations. Decide how parser omissions and mandatory accommodation needs are clarified.
5. Handle parser errors before search, empty results explicitly, and summary errors per
   result. Display only validated claims/citations and their limitations. Do not claim
   current restaurant policy or allergen safety from historical anecdotes.
6. Resolve the cross-restaurant chunk-evidence ownership concern described above before
   claiming grounded real retrieval integration. `Snippet` lacks a restaurant ID, so
   citation membership alone cannot establish that a review belongs to a result.

The demo verifies synthetic wiring and graceful application handling only. It does not
verify live retrieval, ranking quality, model extraction or entailment, resistance to
adversarial reviews, or Streamlit integration. The challenge report's documented parser
gaps and intermittent structured-output failures remain unresolved by these changes.

Updated offline verification: 49 tests pass with the existing SDK installed, including
21 new error-handling and demo tests beyond the 28-test challenge baseline. The demo's
five scenarios also ran with socket connections blocked. No live requests were made
for these error-handling or demonstration changes. Compile-check the example as well:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src\llm tests examples
```
