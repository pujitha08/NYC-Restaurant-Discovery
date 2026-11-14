"""Label the stance re-ranker's evidence sentences in resumable, token-budgeted chunks.

The stance labels (src/retrieval/stance_rerank.py) are cached on disk by (labeler, query, sentence), so a run can
stop and resume anywhere. This script is the budgeted driver for a rate-limited model:

- Order: the dev queries first (the 20/15 split of eval/tune_stance_rerank.py, seed 6165), then the test queries.
- Budget: a rolling-24h ledger of tokens used (data/processed/stance_rerank/usage_ledger.jsonl) is checked before
  every call. A run stops before a call that would take the 24h total past STOP_FRACTION (70%) of the daily
  limit, and after CHUNK_TOKENS (~100K) used in this run. Rerun later; finished sentences are not paid for twice.
- Daily limit: Groq does not report a daily token limit in its response headers, so DAILY_TOKEN_LIMIT is an
  assumption (200,000 for gpt-oss-20b on the free tier; override with --daily-limit). The ledger only knows what
  went through this script plus any "estimate" entries added by hand for other use of the same model.
- Dates in --blackout (default 2026-11-17 and 2026-11-18, demo days) are refused for the demo model.
- Provider: NVIDIA or OpenRouter if their key is set in .env (much higher limits), else Groq.

    python -m eval.label_stance [--chunk-tokens 100000] [--daily-limit 200000] [--dry-run]
"""

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path

from eval.make_pool import read_queries
from eval.tune_stance_rerank import QUERIES_PATH, split_queries
from src.config import DATA_DIR

LEDGER_PATH = DATA_DIR / "processed" / "stance_rerank" / "usage_ledger.jsonl"
DAILY_TOKEN_LIMIT = 200_000
STOP_FRACTION = 0.70
CHUNK_TOKENS = 100_000
BLACKOUT = ("2026-11-17", "2026-11-18")
CALL_ESTIMATE = 3_500  # tokens booked before a call whose size is not known yet (corrected by the usage report)


def now():
    return datetime.datetime.now(datetime.timezone.utc)


class Ledger:
    """Append-only JSONL of {ts, model, tokens, kind}; used_24h sums the entries of the last 24 hours."""

    def __init__(self, path=LEDGER_PATH, clock=now):
        self.path, self.clock = Path(path), clock

    def add(self, model, tokens, kind="label"):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": self.clock().isoformat(timespec="seconds"), "model": model, "tokens": int(tokens),
                                "kind": kind}) + "\n")

    def used_24h(self, model):
        if not self.path.exists():
            return 0
        cutoff, total = self.clock() - datetime.timedelta(hours=24), 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r["model"] == model and datetime.datetime.fromisoformat(r["ts"]) > cutoff:
                        total += r["tokens"]
                except (ValueError, KeyError):
                    continue
        return total


def allowance(used_24h, daily_limit=DAILY_TOKEN_LIMIT, stop_fraction=STOP_FRACTION, chunk=CHUNK_TOKENS):
    """Tokens this run may still use: the room left under stop_fraction of the daily limit, capped at one chunk."""
    return max(0, min(chunk, int(daily_limit * stop_fraction) - used_24h))


def blackout(today, dates=BLACKOUT):
    return today.date().isoformat() in dates


def pending(reranker, queries, order):
    """Yield (query_id, text, [uncached sentences]) in `order`, for queries that still have unlabeled sentences."""
    by_id = {q["query_id"]: q for q in queries}
    for qid in order:
        q = by_id[qid]
        boroughs = [q["borough"].strip()] if q.get("borough", "").strip() else None
        results = reranker.base.search(q["query"], boroughs=boroughs, top_k=reranker.top_n, **reranker.base_kwargs)
        picks = [reranker.pick_sentences(q["query"], r) for r in results]
        todo = list(dict.fromkeys(t for ps in picks for t, _ in ps
                                  if reranker.cache.get(reranker.labeler.id, q["query"], t) is None))
        yield qid, q["query"], todo


def run(reranker, queries, order, ledger, *, daily_limit, chunk, stop_fraction=STOP_FRACTION, dry_run=False, out=print):
    """Label until the budget is spent. Returns {"tokens", "calls", "sentences", "stopped", "status": {qid: pending}}."""
    labeler, model = reranker.labeler, reranker.labeler.model
    budget = allowance(ledger.used_24h(model), daily_limit, stop_fraction, chunk)
    out(f"{model}: used in the last 24h {ledger.used_24h(model):,}; this run may use {budget:,} tokens "
        f"(stop at {stop_fraction:.0%} of an assumed {daily_limit:,}/day, chunk {chunk:,})")
    spent, calls, sentences, stopped, status = 0, 0, 0, None, {}
    for qid, text, todo in pending(reranker, queries, order):
        status[qid] = len(todo)
        for i in range(0, len(todo), labeler.batch_size):
            batch = todo[i:i + labeler.batch_size]
            if stopped or dry_run:
                continue
            if spent + CALL_ESTIMATE > budget:
                stopped = f"budget reached after {spent:,} tokens"
                continue
            before = labeler.tokens_used
            for sentence, stance in zip(batch, labeler.label_batch(text, batch)):
                reranker.cache.put(labeler.id, text, sentence, stance)
            used = labeler.tokens_used - before
            ledger.add(model, used)
            spent, calls, sentences = spent + used, calls + 1, sentences + len(batch)
            status[qid] -= len(batch)
            out(f"  {qid}: {sentences:,} sentences labeled this run, {spent:,} tokens")
    return {"tokens": spent, "calls": calls, "sentences": sentences, "stopped": stopped, "status": status}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Budgeted, resumable stance labeling for the re-ranker")
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH)
    ap.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    ap.add_argument("--daily-limit", type=int, default=DAILY_TOKEN_LIMIT)
    ap.add_argument("--stop-fraction", type=float, default=STOP_FRACTION)
    ap.add_argument("--blackout", nargs="*", default=list(BLACKOUT), help="YYYY-MM-DD dates on which to refuse to run")
    ap.add_argument("--dry-run", action="store_true", help="only count the unlabeled sentences per query")
    args = ap.parse_args(argv)

    from eval import make_pool
    from src.retrieval.stance_rerank import LLMLabeler, StanceReranker, _load_env
    _load_env()
    import os
    real_key = lambda name: bool(os.environ.get(name, "").strip()) and not re.search(r"your-|replace_with", os.environ[name], re.I)
    if real_key("NVIDIA_API_KEY") or real_key("OPENROUTER_API_KEY"):
        sys.exit("An NVIDIA or OpenRouter key is set: this driver only knows the Groq budget. Use the provider-specific "
                 "labeler instead (not implemented here) or remove the key for this run.")
    if not args.dry_run and blackout(now(), args.blackout):
        sys.exit(f"{now().date()} is a blackout date ({', '.join(args.blackout)}): not using the demo model today.")
    semantic = make_pool.load_semantic()
    if semantic is None:
        sys.exit("The semantic index is not built; run: python -m src.retrieval.semantic --build")
    reranker = StanceReranker.from_defaults(semantic)
    if not isinstance(reranker.labeler, LLMLabeler):
        sys.exit("The trained stance classifier exists, so no LLM labeling is needed.")
    queries = read_queries(args.queries)
    dev, test = split_queries([q["query_id"] for q in queries])
    result = run(reranker, queries, dev + test, Ledger(), daily_limit=args.daily_limit, chunk=args.chunk_tokens,
                 stop_fraction=args.stop_fraction, dry_run=args.dry_run)
    done = [q for q in dev + test if result["status"].get(q) == 0]
    left = sum(n for n in result["status"].values())
    print(f"This run: {result['calls']} calls, {result['sentences']:,} sentences, {result['tokens']:,} tokens"
          + (f"; stopped: {result['stopped']}" if result["stopped"] else ""))
    print(f"Queries fully labeled so far: {len(done)} of {len(dev) + len(test)}; sentences still unlabeled "
          f"(in the queries visited): {left:,}")


if __name__ == "__main__":
    main()
