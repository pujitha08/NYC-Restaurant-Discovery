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
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from types import SimpleNamespace

from eval.make_pool import RETRIEVERS, SYSTEMS
from src.config import BOROUGHS
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
# PROVISIONAL (2026-11-12): semantic_capped had the highest NDCG@10 (0.777) and P@10 (0.872) in
# reports/metrics_summary.md, but on LLM-judge labels not yet validated against human labels, only the 18
# fully judged queries (single diet and mixed group; no allergy or diet-plus-filter queries), and no
# difference between systems is significant (Holm-adjusted p >= 0.93). Revisit after the full pool is judged.
DEFAULT_METHOD = "semantic_capped"
BOROUGH_CHOICES = ["Both", *BOROUGHS]
TOP_K = 10
AUTO_SUMMARIES = 3  # summarize the top results automatically; the rest on request ("Explain")
DATA_THROUGH = "September 2021"

STATUS_SUPPORTED, STATUS_MIXED, STATUS_INSUFFICIENT = "supported", "mixed", "insufficient evidence"
# Allergy guard: for allergy-sensitive queries "supported" is never shown (reviews cannot establish allergy
# suitability, and the summarizer's "support" can mean "on topic", e.g. two "avoid" warnings for Q20).
STATUS_NEUTRAL = "Reviews discuss this — read each claim"
# Contradiction claims and no support claim (uncertainty claims may be present): reviewers report problems.
STATUS_PROBLEMS = "Reviews report problems"
ALLERGY_NOTE = ("Allergy suitability can't be confirmed from reviews. Ask the restaurant about ingredients, "
                "preparation and cross-contact before ordering.")
ALLERGY_TERMS = re.compile(
    r"\b(allerg\w*|celiac\w*|coeliac\w*|anaphyla\w*|epi-?pens?|cross[- ]?(?:contact|contaminat\w*)"
    r"|(?:tree[- ]?)?nut[- ]free|peanut[- ]free|gluten[- ]intoleran\w*)\b", re.IGNORECASE)
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
}


# ---- results ------------------------------------------------------------------
@dataclass
class Excerpt:
    review_id: int
    date: str  # YYYY-MM or "date unknown"
    text: str


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


def retrieve(query, retriever, method, boroughs, top_k=TOP_K, include_low_evidence=False):
    """Ranked ResultViews; the raw query text goes to the retriever."""
    _, kwargs = METHODS[method]
    results = retriever.search(query, boroughs=boroughs, top_k=top_k, include_low_evidence=include_low_evidence, **kwargs)
    views = []
    for rank, r in enumerate(results, 1):
        m = retriever.info(r.gmap_id)
        views.append(ResultView(
            rank=rank, gmap_id=r.gmap_id, name=m["name"], borough=m["borough"], address=m.get("address", ""),
            avg_rating=m.get("avg_rating"), num_reviews=m.get("num_clean_reviews", 0),
            low_evidence=bool(m.get("low_evidence")), score=float(r.score),
            excerpts=[Excerpt(s.review_id, month(s.time), s.text) for s in r.evidence_snippets]))
    return views, results


def is_allergy_sensitive(query, parsed):
    """True if parsed allergens are non-empty, parsing failed (parsed is None) or the text names allergy terms."""
    return parsed is None or bool(parsed.allergens) or bool(ALLERGY_TERMS.search(query))


def display_status(summary, guarded=False):
    if summary.status == "insufficient_evidence":
        return STATUS_INSUFFICIENT
    labels = {c.label for c in summary.claims}
    if "contradiction" in labels:
        return STATUS_MIXED if "support" in labels else STATUS_PROBLEMS
    if "support" not in labels:
        return STATUS_INSUFFICIENT
    return STATUS_NEUTRAL if guarded else STATUS_SUPPORTED


def to_summary(raw, view, guarded=False):
    by_id = {e.review_id: e for e in view.excerpts}
    claims = [Claim(c.text, c.label, [by_id[i] for i in c.review_ids if i in by_id]) for c in raw.claims]
    return Summary(display_status(raw, guarded), claims, list(raw.limitations), guarded,
                   ALLERGY_NOTE if guarded else None)


def summarize(view, result, context, client, model, guarded=False, user_request=None):
    """Fill view.summary or view.summary_error. Returns the error dict (or None)."""
    if not result.evidence_snippets:
        view.summary = Summary(STATUS_INSUFFICIENT, [], ["No review excerpts were retrieved for this restaurant."],
                               guarded, ALLERGY_NOTE if guarded else None)
        return None
    try:
        raw = summarize_evidence(context, result, client=client, model=model, reasoning_effort=LLM_REASONING_EFFORT,
                                 user_request=user_request)
        view.summary = to_summary(raw, view, guarded)
        view.summary_error = None
        return None
    except LLMError as e:
        view.summary_error = error_info(e)
        return view.summary_error


# ---- entry points -------------------------------------------------------------
def run_search(query, borough_choice, method, *, retriever, client, model, setup_error=None,
               auto_summaries=AUTO_SUMMARIES, top_k=TOP_K, include_low_evidence=False, log_path=_DEFAULT_LOG):
    """Parse, retrieve and summarize the top results; returns (SearchOutcome, raw results)."""
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
    outcome.results, raw = retrieve(query, retriever, method, outcome.boroughs, top_k, include_low_evidence)
    outcome.timings["retrieve_s"] = round(time.monotonic() - t, 3)
    if not outcome.results:
        outcome.notices.append("No restaurants matched. Try fewer or different words, or another borough.")

    t = time.monotonic()
    stop = None if client is not None else (setup_error or {}).get("message", "Summaries are unavailable.")
    for view, result in zip(outcome.results, raw):
        if view.rank > auto_summaries:
            view.summary_skipped = "not requested"
        elif stop:
            view.summary_skipped = stop
        else:
            error = summarize(view, result, outcome.summary_context, client, model, outcome.allergy_sensitive,
                              outcome.query[:MAX_USER_REQUEST])
            if error and error["category"] in ("rate_limit", "configuration", "authentication", "billing"):
                stop = error["message"]  # do not keep calling; show excerpts for the rest
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
        summarize(view, raw[index], outcome.summary_context, client, model, outcome.allergy_sensitive,
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
        "allergy_sensitive": outcome.allergy_sensitive, "reasoning_effort": LLM_REASONING_EFFORT,
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


def log_event(path, kind, outcome, extra=None):
    if path is _DEFAULT_LOG:
        path = LOG_PATH
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_record(kind, outcome, extra), ensure_ascii=False) + "\n")


def load_retriever(name):
    """A retriever by name (doc_tfidf, chunk_tfidf, semantic); None if its index is not built."""
    return RETRIEVERS[name]()
