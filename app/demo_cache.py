"""Cached demo results: a fallback for when the live language model is unavailable.

The full SearchOutcome (results, excerpts, summaries) of each DEMO.md query is saved to
data/demo_cache/*.json. They hold only public review text already shown in the app. A cached outcome is
served for an exact (query, borough, method) match, either when the "Demo mode" toggle is on or when a live
search failed with a rate-limit, timeout or network error, and is always labeled "cached demo result".

Regenerate (needs the semantic index and a working Groq key; spaces the queries out for the 8K tokens/min limit):
    python -m app.demo_cache --build
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from app import pipeline
from src.schemas import ParsedQuery

CACHE_DIR = pipeline.REPO / "data" / "demo_cache"
CACHE_LABEL = "cached demo result"
FALLBACK_CATEGORIES = ("rate_limit", "timeout", "network")
# DEMO.md: (query, borough choice, method). The fourth query is also shown with semantic_log.
DEMO_QUERIES = [
    ("vegan options for a group", "Brooklyn", "semantic_capped"),
    ("restaurant careful about peanut allergies", "Both", "semantic_capped"),
    ("safe for celiac disease, no cross contamination", "Both", "semantic_capped"),
    ("halal chinese food", "Manhattan", "semantic_capped"),
    ("halal chinese food", "Manhattan", "semantic_log"),
]
SECONDS_BETWEEN_BUILDS = 45


def normalize(query):
    return re.sub(r"\s+", " ", query or "").strip().casefold()


def cache_key(query, borough_choice, method):
    return hashlib.sha256("\n".join((normalize(query), borough_choice, method)).encode()).hexdigest()[:16]


def cache_path(query, borough_choice, method, cache_dir=None):
    return Path(cache_dir or CACHE_DIR) / f"{cache_key(query, borough_choice, method)}.json"


# ---- (de)serialization --------------------------------------------------------
def outcome_to_dict(outcome):
    """JSON-ready dict of a SearchOutcome without run-specific data (request id, timings, token usage)."""
    d = dataclasses.asdict(outcome)
    d.update(request_id="", timings={}, llm_calls=[], cached=False)
    return d


def _excerpts(items):
    return [pipeline.Excerpt(**e) for e in items]


def _summary(d):
    if d is None:
        return None
    claims = [pipeline.Claim(c["text"], c["label"], _excerpts(c["excerpts"])) for c in d["claims"]]
    return pipeline.Summary(d["status"], claims, list(d["limitations"]), d["guarded"], d["note"])


def outcome_from_dict(d):
    results = []
    for v in d["results"]:
        v = dict(v, excerpts=_excerpts(v["excerpts"]), summary=_summary(v["summary"]))
        results.append(pipeline.ResultView(**v))
    parsed = ParsedQuery(**d["parsed"]) if d["parsed"] else None
    return pipeline.SearchOutcome(**{**d, "parsed": parsed, "summary_context": ParsedQuery(**d["summary_context"]),
                                     "results": results})


def _frozen(outcome):
    """Copy for saving/serving: ranks past the summarized ones cannot be explained without the live model."""
    d = outcome_to_dict(outcome)
    for v in d["results"]:
        if v["summary"] is None and not v["summary_error"]:
            v["summary_skipped"] = f"not included in the {CACHE_LABEL}"
        v["summary_error"] = None
    d["notices"], d["parse_error"], d["setup_error"] = [], None, None
    return d


# ---- save / load --------------------------------------------------------------
def save_outcome(outcome, cache_dir=None):
    path = cache_path(outcome.query, outcome.borough_choice, outcome.method, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"cached_at": datetime.date.today().isoformat(), "outcome": _frozen(outcome)}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def load_cached(query, borough_choice, method, cache_dir=None):
    """The cached SearchOutcome for an exact (query, borough, method) match, else None."""
    path = cache_path(query, borough_choice, method, cache_dir)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        outcome = outcome_from_dict(record["outcome"])
    except (ValueError, KeyError, TypeError):
        return None  # unreadable cache: behave as if there is none
    if (normalize(outcome.query), outcome.borough_choice, outcome.method) != (normalize(query), borough_choice, method):
        return None
    outcome.request_id = pipeline.uuid.uuid4().hex[:12]
    outcome.cached = True
    outcome.notices = [f"This is a {CACHE_LABEL} (saved {record['cached_at']}), not a live search."]
    return outcome


def live_failed(outcome):
    """True if the live run hit a rate-limit, timeout or network error (parser or any summary)."""
    errors = [outcome.parse_error] + [v.summary_error for v in outcome.results]
    return any(e and e["category"] in FALLBACK_CATEGORIES for e in errors)


def search_with_fallback(query, borough_choice, method, *, demo_mode=False, cache_dir=None, **kwargs):
    """pipeline.run_search, or the cached outcome in demo mode / after a rate-limit, timeout or network failure.

    Returns (SearchOutcome, raw results); raw is empty for a cached outcome (no live Explain).
    """
    cached = load_cached(query, borough_choice, method, cache_dir) if query and query.strip() else None
    if demo_mode and cached:
        return cached, []
    outcome, raw = pipeline.run_search(query, borough_choice, method, **kwargs)
    if demo_mode:
        outcome.notices.append(f"Demo mode is on, but no {CACHE_LABEL} exists for this query; ran a live search.")
    elif cached and live_failed(outcome):
        cached.notices.insert(0, "The language model was unavailable, so a " + CACHE_LABEL + " is shown instead.")
        return cached, []
    return outcome, raw


# ---- build --------------------------------------------------------------------
def build(cache_dir=None, gap=SECONDS_BETWEEN_BUILDS):
    client, model, setup_error = pipeline.make_llm_client()
    if setup_error:
        sys.exit(f"Cannot build the demo cache: {setup_error['message']}")
    retrievers = {}
    for n, (query, borough, method) in enumerate(DEMO_QUERIES):
        name = pipeline.METHODS[method][0]
        if name not in retrievers:
            retrievers[name] = pipeline.load_retriever(name)
            if retrievers[name] is None:
                sys.exit(f"The {name} index is not built; run: python -m src.retrieval.semantic --build")
        if n:
            time.sleep(gap)
        outcome, _ = pipeline.run_search(query, borough, method, retriever=retrievers[name], client=client,
                                         model=model, log_path=None)
        summarized = [v for v in outcome.results if v.summary]
        if outcome.parse_error or any(v.summary_error for v in outcome.results) or len(summarized) < pipeline.AUTO_SUMMARIES:
            sys.exit(f"Live run for {query!r} ({method}) was incomplete (rate limit or error); wait and rerun. "
                     "Nothing was saved for it.")
        print(save_outcome(outcome, cache_dir), query, borough, method)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Demo-result cache")
    ap.add_argument("--build", action="store_true", help="run the DEMO.md queries live and save them")
    args = ap.parse_args()
    if args.build:
        build()
    else:
        ap.print_help()
