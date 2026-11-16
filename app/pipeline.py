"""Search pipeline for the app: parse -> retrieve -> summarize, as plain functions (no Streamlit).

Follows examples/llm_pipeline_demo.py and the integration rules in docs/llm.md:
- The raw query text goes to the retriever (it does not take ParsedQuery); diets and allergens are
  never database filters, and retrieval scores are not safety probabilities.
- Only exact borough names become filters: the user's borough choice, or (with "Both") a borough
  the parser returned as location AND that appears verbatim in the request.
- Parser failure -> search with the raw request and a notice; summaries then use the raw request
  as context. Summarizer failure or rate limit -> show the review excerpts without a summary.
  A missing key or model -> retrieval still works; no LLM calls; a setup message.
- User-facing errors follow the category table in docs/llm.md; no configuration value, key,
  provider text or raw exception is shown or logged.

Every request (and every "Explain" summary) is appended to logs/app_requests.jsonl (git-ignored):
query, parsed result, method, results, summaries, prompt hashes, model, per-stage timings,
token usage per LLM call, and error categories/codes.
"""

import datetime
from collections import Counter
import hashlib
import json
import os
import dataclasses
import random
import re
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from types import SimpleNamespace

from eval.make_pool import INDEX_EXISTS, RETRIEVERS, SYSTEMS
from src.allergy import ALLERGEN_NAMES, ALLERGY_TERMS  # noqa: F401 (re-exported)
from app import evidence_filter
from app.coverage import need_terms
from src.config import BOROUGHS, DATA_DIR
from src.llm import evidence_summarizer, query_parser
from src.llm.evidence_summarizer import MAX_USER_REQUEST, summarize_evidence
from src.llm.groq_client import LLMConfigurationError, LLMError, _usage, resolve_model
from src.llm.query_parser import parse_query
from src.schemas import ParsedQuery, RetrievalResult

REPO = Path(__file__).resolve().parent.parent
LOG_PATH = REPO / "logs" / "app_requests.jsonl"
_DEFAULT_LOG = object()  # log_path default: LOG_PATH as set at call time (None disables logging)
ENV_PATH = REPO / ".env"

# The evaluated systems (eval/make_pool.py SYSTEMS): name -> (retriever, extra search arguments).
METHODS = {name: (retriever, kwargs) for name, retriever, kwargs in SYSTEMS}
STANCE_METHOD, STANCE_FALLBACK = "semantic_capped_stance", "semantic_capped"
# PROVISIONAL: on all 35 queries (reports/metrics_summary.md) semantic_capped has the highest NDCG@10 (0.708
# [0.631, 0.785]; doc_tfidf 0.612, semantic_log 0.616, chunk_tfidf 0.603) and P@10 lenient (0.666). Only the gain over
# semantic_log survives Holm correction (p 0.024); vs doc_tfidf the unadjusted p is 0.053 and the Holm-adjusted
# p 0.368, so it is NOT significantly better than doc_tfidf. Labels are LLM-judge labels not yet validated against
# human labels, and this default was chosen using the same 35 queries it is scored on (no held-out set).
DEFAULT_METHOD = "semantic_capped"
BOROUGH_CHOICES = ["Both", *BOROUGHS]
TOP_K = 10
# Live searches summarize only the top results, two at a time, and retry a rate-limited call with exponential backoff
# (a 10-restaurant burst hit the provider's rate limit). The rest are listed without a summary ("not summarized").
AUTO_SUMMARIES = 6
SUMMARY_WORKERS = 2
BACKOFF_SECONDS = (2.0, 4.0, 8.0)  # waits before the 2nd, 3rd and 4th attempt of a call that hit a rate limit
DATA_THROUGH = "September 2021"

STATUS_SUPPORTED, STATUS_MIXED, STATUS_INSUFFICIENT = "supported", "mixed", "insufficient evidence"
# Allergy guard: for allergy-sensitive queries "supported" is never shown (reviews cannot establish allergy
# suitability, and the summarizer's "support" can mean "on topic", e.g. two "avoid" warnings for Q20).
STATUS_NEUTRAL = "Reviews discuss this — read each claim"
# Contradiction claims and no support claim (uncertainty claims may be present): reviewers report problems.
STATUS_PROBLEMS = "Reviews report problems"
# Deterministic wording guard for allergy requests: a claim that uses safety language in the app's own voice (outside
# quotation marks) is replaced by a neutral line; its label, citation and excerpt are kept, so the quoted review still shows.
SAFETY_WORDS = re.compile(r"\b(?:safe|safely|safety|guarantee[sd]?|guaranteeing|reassur\w*)\b", re.IGNORECASE)
_QUOTED = re.compile(r'"[^"]*"|\u201c[^\u201d]*\u201d')
NEUTRAL_CLAIM = "A reviewer\u2019s comment about this allergy is quoted below."


def has_safety_wording(text):
    """True if text uses safe / safely / safety / guarantee(d) / reassur* outside direct quotation marks.

    Balanced double-quoted spans (straight or curly) are the reviewer's own words and are ignored; an unbalanced quote
    protects nothing.
    """
    return bool(SAFETY_WORDS.search(_QUOTED.sub(" ", text or "")))


ALLERGY_NOTE = ("Allergy suitability can't be confirmed from reviews. Ask the restaurant about ingredients, "
                "preparation and cross-contact before ordering.")
LLM_REASONING_EFFORT = "low"  # parser and summarizer calls (gpt-oss); fewer reasoning tokens per search

# docs/llm.md "Safe error contract and UI handling": category -> what the user is told.
ERROR_MESSAGES = {
    "configuration": "The language model is not configured (set GROQ_API_KEY and GROQ_MODEL in .env).",
    "authentication": "The language model rejected the API key. Check the key in your local .env (not here).",
    "permission": "The language model account cannot use this model. Check account and model access.",
    "rate_limit": "The language model's rate limit was reached. Wait a minute, then try again.",
    "timeout": "The language model timed out. You can try again.",
    "network": "The language model could not be reached. Check your connection.",
    "bad_request": "The language model could not return a structured answer for this request.",
    "billing": "The language model provider requires payment or credits.",
    "provider": "The language model is temporarily unavailable.",
    "response": "The language model returned no usable structured answer.",
    "labels_unavailable": "Stance labels for this query are not cached.",
}


# ---- results ------------------------------------------------------------------
@dataclass
class Excerpt:
    review_id: int
    date: str  # YYYY-MM or "date unknown"
    text: str
    rating: Optional[int] = None  # the review's star rating when known (older saved demo results have none)


@dataclass
class Claim:
    text: str
    label: str  # support / contradiction / uncertainty
    excerpts: List[Excerpt]


@dataclass
class Summary:
    status: str  # supported / mixed / STATUS_PROBLEMS / insufficient evidence / STATUS_NEUTRAL (allergy guard)
    claims: List[Claim]
    limitations: List[str]
    guarded: bool = False  # allergy-sensitive query: no "supported" status and no support check marks
    note: Optional[str] = None  # ALLERGY_NOTE when guarded


@dataclass
class ResultView:
    rank: int
    gmap_id: str
    name: str
    borough: str
    address: str
    avg_rating: Optional[float]
    num_reviews: int
    low_evidence: bool
    score: float
    excerpts: List[Excerpt]
    summary: Optional[Summary] = None
    summary_error: Optional[Dict] = None  # {"category", "code", "message"}
    summary_skipped: Optional[str] = None  # why no summary was requested
    # Stance labels of the evidence sentences (semantic_capped_stance only; None when the ranking has none):
    # counts by stance, and [{"text", "stance", "review_id"}] in evidence order.
    stance_counts: Optional[Dict[str, int]] = None
    stance_sentences: Optional[List[Dict]] = None


@dataclass
class SearchOutcome:
    request_id: str
    query: str
    borough_choice: str
    method: str
    parsed: Optional[ParsedQuery]
    summary_context: ParsedQuery  # what the summarizer is told about the need
    boroughs: Optional[List[str]]
    results: List[ResultView]
    notices: List[str] = field(default_factory=list)
    parse_error: Optional[Dict] = None
    setup_error: Optional[Dict] = None
    timings: Dict[str, float] = field(default_factory=dict)
    llm_calls: List[Dict] = field(default_factory=list)
    model: Optional[str] = None
    allergy_sensitive: bool = False
    cached: bool = False  # served from data/demo_cache (app/demo_cache.py), not a live search
    ranking_fallback: bool = False  # stance re-ranking was unavailable; the base ranking is shown instead
    cached_at: Optional[str] = None  # YYYY-MM-DD the cached demo result was saved (cached outcomes only)


# ---- LLM client ---------------------------------------------------------------
class UsageRecorder:
    """Wraps an SDK client to record token usage and latency of each call (no content, no headers)."""

    def __init__(self, client):
        self._client, self.calls = client, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        t = time.monotonic()
        name = kwargs.get("response_format", {}).get("json_schema", {}).get("name")
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception:
            self.calls.append({"call": name, "ok": False, "latency_s": round(time.monotonic() - t, 3), "usage": None})
            raise
        self.calls.append({"call": name, "ok": True, "latency_s": round(time.monotonic() - t, 3),
                           "usage": _usage(response)})
        return response

    def take(self):
        calls, self.calls = self.calls, []
        return calls


def load_env(path=ENV_PATH):
    """Load the git-ignored repo-root .env (if present) without overriding the environment."""
    if Path(path).exists():
        from dotenv import load_dotenv
        load_dotenv(path, override=False)


def make_llm_client():
    """(UsageRecorder-wrapped client, model, None) or (None, None, setup error dict)."""
    load_env()
    try:
        model = resolve_model(None)
        from src.llm.groq_client import get_client
        return UsageRecorder(get_client()), model, None
    except LLMConfigurationError as e:
        return None, None, error_info(e)


def error_info(e):
    category = getattr(e, "category", "provider")
    return {"category": category, "code": getattr(e, "code", "error"),
            "message": ERROR_MESSAGES.get(category, ERROR_MESSAGES["provider"])}


def prompt_hashes():
    h = lambda system, schema: hashlib.sha256((system + "\n" + json.dumps(schema, sort_keys=True)).encode()).hexdigest()
    return {"parser": h(query_parser.SYSTEM, query_parser.QUERY_SCHEMA),
            "summarizer": h(evidence_summarizer.SYSTEM, evidence_summarizer.SUMMARY_SCHEMA)}


# ---- pipeline stages ----------------------------------------------------------
def month(ms):
    return time.strftime("%Y-%m", time.gmtime(ms / 1000)) if ms else "date unknown"


def borough_filter(choice, parsed, raw_query):
    """(boroughs or None, notice or None). Only exact borough names ever become filters."""
    named = None
    if parsed and isinstance(parsed.location, str):
        for b in BOROUGHS:
            if parsed.location.strip().casefold() == b.casefold() and re.search(rf"\b{re.escape(b)}\b", raw_query, re.I):
                named = b
    if choice in BOROUGHS:
        notice = (f"Your request mentions {named}, but the borough filter is set to {choice}; showing {choice}."
                  if named and named != choice else None)
        return [choice], notice
    if named:
        return [named], f"Your request names {named}, so results are limited to {named}."
    return None, None


def parse(query, client, model):
    """(ParsedQuery or None, error dict or None)."""
    if client is None:
        return None, None
    try:
        return parse_query(query, client=client, model=model, reasoning_effort=LLM_REASONING_EFFORT), None
    except LLMError as e:
        return None, error_info(e)


def retrieve(query, retriever, method, boroughs, top_k=TOP_K, include_low_evidence=False, search_options=None):
    """Ranked ResultViews; the raw query text goes to the retriever."""
    _, kwargs = METHODS[method]
    results = retriever.search(query, boroughs=boroughs, top_k=top_k, include_low_evidence=include_low_evidence,
                               **kwargs, **(search_options or {}))
    views = []
    for rank, r in enumerate(results, 1):
        m = retriever.info(r.gmap_id)
        views.append(ResultView(
            rank=rank, gmap_id=r.gmap_id, name=m["name"], borough=m["borough"], address=m.get("address", ""),
            avg_rating=m.get("avg_rating"), num_reviews=m.get("num_clean_reviews", 0),
            low_evidence=bool(m.get("low_evidence")), score=float(r.score),
            excerpts=[Excerpt(s.review_id, month(s.time), s.text, getattr(s, "rating", None)) for s in r.evidence_snippets],
            stance_counts=dict(r.stance_counts) if getattr(r, "stance_counts", None) else None,
            stance_sentences=[dict(x) for x in r.sentences] if getattr(r, "sentences", None) else None))
    return views, results


def stance_bar(view):
    """The evidence bar of a result from its stance labels, or None when it has none.

    {"accommodate", "warns", "mention", "other" (not about the need), "bar" (emoji squares), "text", "warning"}.
    "warning" is the strongest warning sentence: the first negative one, in evidence order (evidence is ordered
    by how well its review matched the query). These are sentence labels from a model, not a safety verdict.
    """
    counts = view.stance_counts
    if not counts or not sum(counts.values()):
        return None
    a, w, m = (counts.get(k, 0) for k in ("accommodation", "negative", "mention"))
    warning = next((x["text"] for x in view.stance_sentences or [] if x["stance"] == "negative"), None)
    return {"accommodate": a, "warns": w, "mention": m, "other": counts.get("not_about_allergy", 0),
            "bar": "🟩" * a + "🟥" * w + "🟨" * m + "⬜" * counts.get("not_about_allergy", 0),
            "text": f"{a} accommodate / {w} warns / {m} mention", "warning": warning}


_ALLERGY_SUFFIX = re.compile(r"\s+allerg(?:y|ies)\s*$", re.IGNORECASE)


def allergy_phrase(allergens):
    """'a peanut allergy', 'peanut and milk allergies', 'peanut, milk and egg allergies'; 'this allergy' if none are named.

    A parsed name that already ends in "allergy" or "allergies" ("peanut allergy") is not given a second one.
    """
    names = [_ALLERGY_SUFFIX.sub("", a.strip()).strip() for a in allergens or [] if a and a.strip()]
    names = [n for n in names if n and n.lower() not in ("allergy", "allergies")]
    if not names:
        return "this allergy"
    if len(names) == 1:
        return f"{'an' if names[0][0].lower() in 'aeiou' else 'a'} {names[0]} allergy"
    return f"{', '.join(names[:-1])} and {names[-1]} allergies"


def merge_neutral_claims(claims):
    """Collapse claims that the wording guard replaced by the same neutral line (and that share a label) into one claim.

    The merged claim keeps every cited excerpt (once per review), so no quoted review is lost; other claims are unchanged
    and the order is kept (the merged claim sits where the first of its group was).
    """
    out, index = [], {}
    for c in claims:
        if c.text != NEUTRAL_CLAIM:
            out.append(c)
            continue
        if c.label in index:
            merged = out[index[c.label]]
            seen = {e.review_id for e in merged.excerpts}
            merged.excerpts = merged.excerpts + [e for e in c.excerpts if e.review_id not in seen]
        else:
            index[c.label] = len(out)
            out.append(Claim(c.text, c.label, list(c.excerpts)))
    return out


def results_summary(outcome):
    """One plain-text line summarizing the result cards, or "" if no card has a summary or stance counts.

    No model call: it counts the status badges and stance counts already on the cards, each restaurant once, and only
    restaurants that have a summary or stance counts ("Of the top N summarized:" names N). Restaurants without either are
    left out, never counted as evidence either way, and zero parts are hidden.
    Allergy-sensitive requests: warn (a "reports problems"/"mixed" badge, or a warning sentence in its stance counts) /
    describe accommodation (the cautious "reviews discuss this" badge, or an accommodation sentence) / no direct evidence.
    Other requests: supported / mixed / reports problems / insufficient evidence. The wording never says a restaurant is safe.
    """
    stance = lambda v, k: (v.stance_counts or {}).get(k, 0)
    views = [v for v in outcome.results if v.summary is not None or v.stance_counts]
    if not views:
        return ""
    allergens = [a.strip() for a in (getattr(outcome.parsed, "allergens", None) or []) if a and a.strip()]
    counts = Counter()
    for v in views:
        status = v.summary.status if v.summary is not None else None
        if outcome.allergy_sensitive:
            if status in (STATUS_PROBLEMS, STATUS_MIXED) or stance(v, "negative"):
                counts["warn"] += 1
            elif status == STATUS_NEUTRAL or stance(v, "accommodation"):
                counts["accommodate"] += 1
            else:
                counts["none"] += 1
        else:
            counts[{STATUS_SUPPORTED: "supported", STATUS_MIXED: "mixed", STATUS_PROBLEMS: "problems",
                    STATUS_NEUTRAL: "supported"}.get(status, "none")] += 1
    plural = lambda n, one, many: one if n == 1 else many
    parts = []
    if outcome.allergy_sensitive:
        n = counts["warn"]
        if n:
            parts.append(f"{n} {plural(n, 'restaurant has', 'restaurants have')} reviews that warn about {allergy_phrase(allergens)}")
        n = counts["accommodate"]
        if n:
            parts.append(f"{n} {plural(n, 'describes', 'describe')} accommodating {'it' if len(allergens) <= 1 else 'them'}")
        n = counts["none"]
        if n:
            parts.append(f"{n} show no direct evidence")
    else:
        for key, text in (("supported", "supported by reviews"), ("mixed", "with mixed evidence"),
                          ("problems", "with reviews that report problems"), ("none", "with insufficient evidence")):
            if counts[key]:
                parts.append(f"{counts[key]} {text}")
    return f"Of the top {len(views)} summarized: " + " · ".join(parts)


def is_allergy_sensitive(query, parsed):
    """True if parsed allergens are non-empty, parsing failed (parsed is None), or the text uses allergy wording
    or names an FDA major allergen (the parser can miss allergens, e.g. gpt-oss-20b on Q20)."""
    return (parsed is None or bool(parsed.allergens) or bool(ALLERGY_TERMS.search(query))
            or bool(ALLERGEN_NAMES.search(query)))


def display_status(summary, guarded=False):
    if summary.status == "insufficient_evidence":
        return STATUS_INSUFFICIENT
    labels = {c.label for c in summary.claims}
    if "contradiction" in labels:
        return STATUS_MIXED if "support" in labels else STATUS_PROBLEMS
    if "support" not in labels:
        return STATUS_INSUFFICIENT
    return STATUS_NEUTRAL if guarded else STATUS_SUPPORTED


def context_terms(context):
    """The words of the parsed diets and allergens (what a claim has to be about to count as evidence)."""
    needs = (context.dietary_restrictions + context.allergens) if context is not None else []
    return [t for need in needs for t in need_terms(need)]


def drop_name_only_evidence(claims, terms, name):
    """Support and contradiction claims whose cited reviews mention the need only inside the restaurant's name
    ("The Halal Guys") become a neutral uncertainty claim; the cited reviews stay, so they can still be read."""
    if not terms or not name:
        return claims
    out = []
    for c in claims:
        name_only = (c.label in ("support", "contradiction") and c.excerpts
                     and not any(evidence_filter.mentions(e.text, terms, name) for e in c.excerpts)
                     and any(evidence_filter.mentions_only_in_name(e.text, terms, name) for e in c.excerpts))
        out.append(Claim(evidence_filter.NAME_ONLY_CLAIM, "uncertainty", c.excerpts) if name_only else c)
    return out


def to_summary(raw, view, guarded=False, context=None):
    by_id = {e.review_id: e for e in view.excerpts}
    claims = [Claim(NEUTRAL_CLAIM if guarded and has_safety_wording(c.text) else c.text, c.label,
                    [by_id[i] for i in c.review_ids if i in by_id]) for c in raw.claims]
    claims = drop_name_only_evidence(claims, context_terms(context), getattr(view, "name", None))
    status_source = SimpleNamespace(status=raw.status, claims=claims)  # the status follows the final labels
    return Summary(display_status(status_source, guarded), claims, list(raw.limitations), guarded,
                   ALLERGY_NOTE if guarded else None)


# A summary call that comes back as a bad request ("could not produce a parsable structured response": the model wrote prose
# instead of the JSON) or as an unusable answer is tried once more with less evidence, which is a different prompt.
RETRY_SHORT_CATEGORIES = ("bad_request", "response")
SHORT_EXCERPTS, SHORT_WORDS = 2, 40


def sanitize_text(text):
    """The text with control and format characters turned into spaces and whitespace collapsed."""
    return " ".join("".join(" " if unicodedata.category(c)[0] == "C" else c for c in text or "").split())


def shortened(result):
    """A copy of the result with only the first SHORT_EXCERPTS excerpts, each sanitized and cut to SHORT_WORDS words."""
    snippets = []
    for snippet in result.evidence_snippets:
        words = sanitize_text(snippet.text).split()
        if not words:
            continue
        text = " ".join(words[:SHORT_WORDS]) + (" …" if len(words) > SHORT_WORDS else "")
        snippets.append(dataclasses.replace(snippet, text=text))
        if len(snippets) == SHORT_EXCERPTS:
            break
    return dataclasses.replace(result, evidence_snippets=snippets)


def _summarize_once(view, result, context, client, model, guarded, user_request):
    """One summary call: fills view.summary and returns None, or sets view.summary_error and returns it."""
    try:
        raw = summarize_evidence(context, result, client=client, model=model, reasoning_effort=LLM_REASONING_EFFORT,
                                 user_request=user_request, restaurant_name=view.name)
        view.summary = to_summary(raw, view, guarded, context)
        view.summary_error = None
        return None
    except LLMError as e:
        view.summary_error = error_info(e)
        return view.summary_error


def summarize(view, result, context, client, model, guarded=False, user_request=None):
    """Fill view.summary or view.summary_error. Returns the error dict (or None).

    A bad request or an unusable answer is retried once with shortened, sanitized evidence; if that fails too the error stays
    (the card then shows "the AI summary isn't available right now"), and the rest of the search is not affected.
    """
    if not result.evidence_snippets:
        view.summary = Summary(STATUS_INSUFFICIENT, [], ["No review excerpts were retrieved for this restaurant."],
                               guarded, ALLERGY_NOTE if guarded else None)
        return None
    error = _summarize_once(view, result, context, client, model, guarded, user_request)
    if error and error["category"] in RETRY_SHORT_CATEGORIES:
        short = shortened(result)
        if short.evidence_snippets:
            error = _summarize_once(view, short, context, client, model, guarded, user_request)
    return error


def _backoff_delay(base):
    return base * random.uniform(0.75, 1.25)  # jitter, so the two workers do not retry in lockstep


def summarize_with_backoff(view, result, context, client, model, guarded=False, user_request=None, sleep=None):
    """summarize(), retried with exponential backoff while the provider answers with a rate limit (429).

    Returns the final error dict (or None). Other errors are not retried. `sleep` is time.sleep unless given (tests).
    """
    sleep = sleep or time.sleep
    error = summarize(view, result, context, client, model, guarded, user_request)
    for base in BACKOFF_SECONDS:
        if not error or error["category"] != "rate_limit":
            break
        sleep(_backoff_delay(base))
        error = summarize(view, result, context, client, model, guarded, user_request)
    return error


def summarize_results(outcome, raw, client, auto_summaries, stop_message=None, sleep=None):
    """Summarize the top `auto_summaries` results, SUMMARY_WORKERS at a time, each call retried on a rate limit.

    A result whose call still fails keeps its summary_error (the card then offers a retry). A configuration,
    authentication or billing error stops the remaining calls: those results get summary_skipped with the message.
    Results past `auto_summaries` are marked "not requested". Returns the message of a stopping error, or None.
    """
    todo = []
    for view, result in zip(outcome.results, raw):
        if view.rank > auto_summaries:
            view.summary_skipped = "not requested"
        elif stop_message:
            view.summary_skipped = stop_message
        else:
            todo.append((view, result))
    stop = {"message": None}
    lock = threading.Lock()

    def work(pair):
        view, result = pair
        if stop["message"]:
            view.summary_skipped = stop["message"]
            return
        error = summarize_with_backoff(view, result, outcome.summary_context, client, outcome.model,
                                       outcome.allergy_sensitive, outcome.query[:MAX_USER_REQUEST], sleep)
        if error and error["category"] in ("configuration", "authentication", "billing"):
            with lock:
                stop["message"] = error["message"]

    with ThreadPoolExecutor(max_workers=SUMMARY_WORKERS) as pool:
        list(pool.map(work, todo))
    return stop["message"]


# ---- entry points -------------------------------------------------------------
def run_search(query, borough_choice, method, *, retriever, client, model, setup_error=None,
               auto_summaries=AUTO_SUMMARIES, top_k=TOP_K, include_low_evidence=False, log_path=_DEFAULT_LOG,
               prior=None, search_options=None, progress=None):
    """Parse, retrieve and summarize the top results; returns (SearchOutcome, raw results).

    prior: an earlier SearchOutcome for the same query; its parse result is reused instead of calling the parser
    (compare mode). search_options: extra keyword arguments for the retriever's search (e.g. cache_only).
    progress: optional callable(stage) told when each stage starts: "understand" (only when the parser is called),
    "search", "read" (reviews retrieved, results built) and "summarize" (only when summaries will be requested).
    """
    progress = progress or (lambda stage: None)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    query = (query or "").strip()
    if not query:
        raise ValueError("empty query")
    t0 = time.monotonic()
    outcome = SearchOutcome(request_id=uuid.uuid4().hex[:12], query=query, borough_choice=borough_choice,
                            method=method, parsed=None, summary_context=ParsedQuery(), boroughs=None, results=[],
                            setup_error=setup_error, model=model)
    if setup_error:
        outcome.notices.append(setup_error["message"] + " Searching without query parsing or summaries.")

    t = time.monotonic()
    if prior is not None:
        outcome.parsed, outcome.parse_error = prior.parsed, prior.parse_error
    else:
        if client is not None:
            progress("understand")
        outcome.parsed, outcome.parse_error = parse(query, client, model)
    outcome.timings["parse_s"] = round(time.monotonic() - t, 3)
    if outcome.parse_error:
        outcome.notices.append(f"Could not interpret the request ({outcome.parse_error['message']}) "
                               "Searching with your exact words instead.")
    # The summarizer needs the need: the parsed query, or the raw request when parsing failed.
    outcome.summary_context = outcome.parsed or ParsedQuery(preferences=[query])
    outcome.allergy_sensitive = is_allergy_sensitive(query, outcome.parsed)

    outcome.boroughs, notice = borough_filter(borough_choice, outcome.parsed, query)
    if notice:
        outcome.notices.append(notice)
    t = time.monotonic()
    progress("search")
    try:
        outcome.results, raw = retrieve(query, retriever, method, outcome.boroughs, top_k, include_low_evidence,
                                        search_options)
    except LLMError as e:  # the re-ranker labels sentences with the LLM; without it, show the base ranking
        if method != STANCE_METHOD:
            raise
        outcome.notices.append(f"Could not label the review sentences ({error_info(e)['message']}) "
                               f"Showing the {STANCE_FALLBACK} ranking instead.")
        outcome.ranking_fallback = True
        base_retriever = load_fallback(retriever)
        outcome.results, raw = retrieve(query, base_retriever, STANCE_FALLBACK, outcome.boroughs, top_k,
                                        include_low_evidence)
    outcome.timings["retrieve_s"] = round(time.monotonic() - t, 3)
    if not outcome.results:
        outcome.notices.append("No restaurants matched. Try fewer or different words, or another borough.")

    if outcome.results:
        progress("read")
    t = time.monotonic()
    if client is not None and auto_summaries > 0 and outcome.results:
        progress("summarize")
    stop = None if client is not None else (setup_error or {}).get("message", "Summaries are unavailable.")
    stop = summarize_results(outcome, raw, client, auto_summaries if auto_summaries > 0 else 0, stop) or stop
    outcome.timings["summarize_s"] = round(time.monotonic() - t, 3)
    if any(v.summary_error for v in outcome.results) or (stop and client is not None):
        outcome.notices.append("Some summaries are unavailable; the review excerpts are shown instead.")
    outcome.timings["total_s"] = round(time.monotonic() - t0, 3)
    outcome.llm_calls = client.take() if isinstance(client, UsageRecorder) else []
    log_event(log_path, "search", outcome)
    return outcome, raw


def explain(outcome, raw, gmap_id, *, client, model, log_path=_DEFAULT_LOG):
    """Summarize one more result on request ("Explain"); logs the event. Returns the ResultView."""
    index = next(i for i, v in enumerate(outcome.results) if v.gmap_id == gmap_id)
    view = outcome.results[index]
    t = time.monotonic()
    if client is None:
        view.summary_error = outcome.setup_error or error_info(LLMConfigurationError("not configured"))
    else:
        summarize_with_backoff(view, raw[index], outcome.summary_context, client, model, outcome.allergy_sensitive,
                               outcome.query[:MAX_USER_REQUEST])
        view.summary_skipped = None
    elapsed = round(time.monotonic() - t, 3)
    calls = client.take() if isinstance(client, UsageRecorder) else []
    log_event(log_path, "explain", outcome, extra={"gmap_id": gmap_id, "summarize_s": elapsed, "llm_calls": calls,
                                                     "summary": _summary_record(view)})
    return view


# ---- logging ------------------------------------------------------------------
def _summary_record(v):
    return {"gmap_id": v.gmap_id, "status": v.summary.status if v.summary else None,
            "claims": [{"text": c.text, "label": c.label, "review_ids": [e.review_id for e in c.excerpts]}
                       for c in v.summary.claims] if v.summary else [],
            "limitations": v.summary.limitations if v.summary else [],
            "error": {k: v.summary_error[k] for k in ("category", "code")} if v.summary_error else None,
            "guarded": v.summary.guarded if v.summary else None, "skipped": v.summary_skipped}


def log_record(kind, outcome, extra=None):
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "event": kind, "request_id": outcome.request_id, "query": outcome.query,
        "borough_choice": outcome.borough_choice, "boroughs": outcome.boroughs, "method": outcome.method,
        "allergy_sensitive": outcome.allergy_sensitive, "cached": outcome.cached,
        "ranking_fallback": outcome.ranking_fallback, "reasoning_effort": LLM_REASONING_EFFORT,
        "parsed": asdict(outcome.parsed) if outcome.parsed else None,
        "model": outcome.model, "prompt_sha256": prompt_hashes(),
        "errors": {"setup": _code(outcome.setup_error), "parse": _code(outcome.parse_error),
                   "summaries": {v.gmap_id: _code(v.summary_error) for v in outcome.results if v.summary_error}},
        "notices": outcome.notices,
    }
    if kind == "search":
        record.update(timings=outcome.timings, llm_calls=outcome.llm_calls,
                      results=[{"rank": v.rank, "gmap_id": v.gmap_id, "name": v.name, "score": v.score,
                                "low_evidence": v.low_evidence, "review_ids": [e.review_id for e in v.excerpts]}
                               for v in outcome.results],
                      summaries=[_summary_record(v) for v in outcome.results if v.summary or v.summary_error])
    record.update(extra or {})
    return record


def _code(err):
    return {"category": err["category"], "code": err["code"]} if err else None


def logging_enabled(environ=None):
    """Request logging is on unless NYC_LOG_REQUESTS is 0, false, no or off (a hosted Space turns it off)."""
    return (os.environ if environ is None else environ).get("NYC_LOG_REQUESTS", "1").strip().lower() not in ("0", "false", "no", "off")


def log_event(path, kind, outcome, extra=None):
    """Append one request record to the log. Never raises: with the switch off, or a folder that cannot be written
    (read-only or ephemeral disk), the record is skipped silently."""
    try:
        if path is _DEFAULT_LOG:
            path = LOG_PATH
        if path is None or not logging_enabled():
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record(kind, outcome, extra), ensure_ascii=False) + "\n")
    except Exception:  # logging must never break a search
        return


def data_stats(processed_dir=None):
    """Data scale for the sidebar, read from data/processed (missing files are simply left out).

    Keys: restaurants, reviews (clean reviews kept), chunks (semantic index), index_built (date).
    """
    base = Path(processed_dir) if processed_dir else DATA_DIR / "processed"
    stats = {}
    for name, keys in (("cleaning_stats.json", {"restaurants": "restaurants", "reviews": "kept"}),
                       ("semantic/info.json", {"chunks": "chunk_count", "index_built": "build_date"})):
        try:
            raw = json.loads((base / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for out_key, key in keys.items():
            value = raw.get(key) if isinstance(raw, dict) else None
            if isinstance(value, (int, str)) and not isinstance(value, bool):
                stats[out_key] = value[:10] if out_key == "index_built" else value
    return stats


def data_dir_missing():
    """True when the data directory (NYC_DATA_DIR or the repo's data/) is not a folder."""
    return not Path(DATA_DIR).is_dir()


def available_methods():
    """The METHODS whose index files exist, in METHODS order."""
    return [name for name, (retriever, _) in METHODS.items() if INDEX_EXISTS[retriever]()]


def load_fallback(retriever):
    """The base retriever under a re-ranker (it has a .base), or the retriever itself."""
    return getattr(retriever, "base", retriever)


def load_retriever(name):
    """A retriever by name (doc_tfidf, chunk_tfidf, semantic); None if its index is not built."""
    return RETRIEVERS[name]()
