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
request failures raise a generic `LLMAPIError` without provider details or credentials.
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

No adapter is implemented here. Do not blindly concatenate allergen names into search
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

Tests use only the standard library and shared dataclasses, synthetic evidence, mocked
responses and a mocked SDK constructor. They do not need openai installed, a dataset,
credentials or network access. They test mixed/negated fixture preservation, schema
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
