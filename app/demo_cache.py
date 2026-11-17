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

from app import examples, pipeline
from src.config import DATA_DIR
from src.schemas import ParsedQuery

# The saved demo results follow DATA_DIR (so a relocated data directory can carry them); if it has none, the copy tracked in
# the repository (<repo>/data/demo_cache) is used, which is what a hosted Space ships.
CACHE_DIR = DATA_DIR / "demo_cache" if (DATA_DIR / "demo_cache").is_dir() else pipeline.REPO / "data" / "demo_cache"
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
# The home-screen chips (app/examples.py): searched across both boroughs with the default method. A query already above
# (the peanut one) is not listed twice. Their cache files are built with `python -m app.demo_cache --build`.
DEMO_QUERIES += [(q, examples.CHIP_BOROUGH, examples.CHIP_METHOD) for q in examples.chip_queries()
                 if (q, examples.CHIP_BOROUGH, examples.CHIP_METHOD) not in DEMO_QUERIES]
# Queries whose cache files were saved before the chips existed; the rest are built when the cache is rebuilt.
SAVED_BEFORE_CHIPS = DEMO_QUERIES[:5]
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
    outcome.cached, outcome.cached_at = True, record["cached_at"]
    outcome.notices = [cached_notice(outcome.cached_at)]  # exactly one notice about the cache, with the saved date
    return outcome


def cached_notice(saved, unavailable=False):
    if unavailable:
        return f"The language model was unavailable, so a {CACHE_LABEL} (saved {saved}) is shown instead."
    return f"This is a {CACHE_LABEL} (saved {saved}), not a live search."


def live_failed(outcome):
    """True if the live run hit a rate-limit, timeout or network error: in the parser, or in every summary it tried.

    One failed summary among several is not a failure of the run: its card offers a retry (the saved result is not swapped in).
    """
    if outcome.parse_error and outcome.parse_error["category"] in FALLBACK_CATEGORIES:
        return True
    tried = [v for v in outcome.results if v.summary or v.summary_error]
    return bool(tried) and all(v.summary_error and v.summary_error["category"] in FALLBACK_CATEGORIES for v in tried)


def search_with_fallback(query, borough_choice, method, *, demo_mode=False, cache_dir=None, progress=None, **kwargs):
    """pipeline.run_search, or the cached outcome in demo mode / after a rate-limit, timeout or network failure.

    Returns (SearchOutcome, raw results); raw is empty for a cached outcome (no live Explain).
    """
    cached = load_cached(query, borough_choice, method, cache_dir) if query and query.strip() else None
    if demo_mode and cached:
        if progress:
            progress("cache")
        return cached, []
    outcome, raw = pipeline.run_search(query, borough_choice, method, progress=progress, **kwargs)
    if cached and not demo_mode and live_failed(outcome):
        cached.notices = [cached_notice(cached.cached_at, unavailable=True)]  # replaces the plain cached notice: one only
        if progress:
            progress("cache")
        return cached, []
    return outcome, raw


# ---- build --------------------------------------------------------------------
BUILD_TOKEN_BUDGET = 7000  # a build never uses more than about this many tokens in any 60 s (the provider limit is 8,000 a minute)
BUILD_RETRY_SECONDS = (20.0,)  # a build retries a per-minute 429 once, after this wait
ESTIMATED_CALL_TOKENS = 1800  # until the first summary call has shown what one costs


def call_tokens(calls):
    """Total tokens of the recorded model calls (a call without usage counts as 0)."""
    return sum((c.get("usage") or {}).get("total_tokens") or 0 for c in calls)


def paced_summaries(outcome, raw, client, *, budget=BUILD_TOKEN_BUDGET, window=60.0, estimate=ESTIMATED_CALL_TOKENS,
                    clock=time.monotonic, sleep=time.sleep):
    """Summarize every result of the outcome one after another, waiting whenever the next call could push the tokens used in
    the last `window` seconds over `budget`. Returns (tokens used, the daily-limit error or None).

    The first daily-limit 429 stops at once: nothing more is asked of the provider. Other errors stay on their result.
    """
    used, total = [], 0  # (finish time, tokens) of the calls still inside the window
    for view, result in zip(outcome.results, raw):
        while True:
            now = clock()
            used = [(t, n) for t, n in used if now - t < window]
            guess = max([n for _, n in used[-3:]] or [estimate])
            if not used or sum(n for _, n in used) + guess <= budget:
                break
            sleep(max(0.5, window - (now - used[0][0]) + 0.1))
        view.summary_skipped = None
        error = pipeline.summarize_with_backoff(view, result, outcome.summary_context, client, outcome.model,
                                                outcome.allergy_sensitive, outcome.query[:pipeline.MAX_USER_REQUEST], sleep=sleep)
        tokens = call_tokens(client.take() if isinstance(client, pipeline.UsageRecorder) else [])
        used.append((clock(), tokens))
        total += tokens
        if error and error["code"] == "daily_limit":
            return total, error
    return total, None


def build(cache_dir=None, gap=SECONDS_BETWEEN_BUILDS, only_missing=False, top_k=None, only=None, budget=BUILD_TOKEN_BUDGET,
          sleep=time.sleep):
    """Run the demo queries live and save them. Returns [(query, tokens used)].

    top_k (default None: the usual top 10 with the top 6 summarized): a larger pool, every restaurant of it summarized one after
    another under a token budget per minute (paced_summaries). only: build just these queries (matched like cache keys).
    A daily-limit 429 stops the whole build with nothing saved for the unfinished query.
    """
    client, model, setup_error = pipeline.make_llm_client()
    if setup_error:
        sys.exit(f"Cannot build the demo cache: {setup_error['message']}")
    retrievers = {}
    wanted = {normalize(q) for q in only or []}
    queries = [q for q in DEMO_QUERIES if not (only_missing and cache_path(*q, cache_dir).exists())
               and (not wanted or normalize(q[0]) in wanted)]
    if not queries:
        print("Nothing to build.")
    done = []
    for n, (query, borough, method) in enumerate(queries):
        name = pipeline.METHODS[method][0]
        if name not in retrievers:
            retrievers[name] = pipeline.load_retriever(name)
            if retrievers[name] is None:
                sys.exit(f"The {name} index is not built; run: python -m src.retrieval.semantic --build")
        if n:
            sleep(gap)
        if top_k:
            outcome, raw = pipeline.run_search(query, borough, method, retriever=retrievers[name], client=client, model=model,
                                               log_path=None, top_k=top_k, auto_summaries=0)
            tokens = call_tokens(outcome.llm_calls)  # the parser call
            if not outcome.parse_error:
                saved_backoff, pipeline.BACKOFF_SECONDS = pipeline.BACKOFF_SECONDS, BUILD_RETRY_SECONDS
                try:
                    more, error = paced_summaries(outcome, raw, client, budget=budget, sleep=sleep)
                finally:
                    pipeline.BACKOFF_SECONDS = saved_backoff
                tokens += more
                if error:
                    sys.exit(f"Daily token limit reached while building {query!r} after {tokens} tokens ({error['message']}). "
                             "Nothing was saved for it or for the queries after it.")
            complete = (not outcome.parse_error and not any(v.summary_error for v in outcome.results)
                        and all(v.summary for v in outcome.results))
        else:
            outcome, _ = pipeline.run_search(query, borough, method, retriever=retrievers[name], client=client,
                                             model=model, log_path=None)
            tokens = call_tokens(outcome.llm_calls)
            summarized = [v for v in outcome.results if v.summary]
            complete = not (outcome.parse_error or any(v.summary_error for v in outcome.results)
                            or len(summarized) < pipeline.AUTO_SUMMARIES)
        if not complete:
            sys.exit(f"Live run for {query!r} ({method}) was incomplete (rate limit or error); wait and rerun. "
                     "Nothing was saved for it.")
        print(save_outcome(outcome, cache_dir), query, borough, method, f"{len(outcome.results)} restaurants, {tokens} tokens")
        done.append((query, tokens))
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description="Demo-result cache")
    ap.add_argument("--build", action="store_true", help="run the demo queries live and save them")
    ap.add_argument("--missing", action="store_true", help="with --build: only the queries without a cache file yet")
    ap.add_argument("--only", action="append", metavar="QUERY", help="with --build: only this query (repeat for several)")
    ap.add_argument("--top-k", type=int, default=None, metavar="N",
                    help="with --build: a pool of N restaurants, all summarized under a token budget per minute (default: top 10, top 6 summarized)")
    args = ap.parse_args(argv)
    if args.build:
        build(only_missing=args.missing, top_k=args.top_k, only=args.only)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
