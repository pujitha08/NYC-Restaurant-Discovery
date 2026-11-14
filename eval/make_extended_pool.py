"""Judging pool for the stance re-ranker's full candidate set: the top-K of semantic_capped (default: the pre-registered depth, 20).

The re-ranker can only reorder the base top-N (pre-registered N=20, src/config.py), so every weight setting should land on judged pairs. This builds
the pool rows (same columns, evidence and blind ordering as eval/make_pool.py) for the top-K of semantic_capped on
each query, and keeps only the pairs that have no label yet in --labels (the existing labels are never touched).
Judge the result with the same prompt and settings as the existing labels:

    python -m eval.make_extended_pool --dev-only [--top-k 20] [--queries-only Q02 Q03 ...]
    python -m eval.llm_judge --pool eval/pool_extended.csv --provider groq --model openai/gpt-oss-120b --dry-run

Output (git-ignored): eval/pool_extended.csv. Judge labels are appended to the same eval/llm_labels.jsonl.
"""

import argparse
import sys
from pathlib import Path

from eval import make_pool
from eval.metrics import read_labels
from eval.tune_stance_rerank import split_queries
from src.config import STANCE_RERANK

EVAL_DIR = Path(__file__).parent
OUT_PATH = EVAL_DIR / "pool_extended.csv"
LABELS_PATH = EVAL_DIR / "llm_labels.jsonl"
BASE_SYSTEM = [s for s in make_pool.SYSTEMS if s[0] == "semantic_capped"]


def new_rows(pool_rows, labeled):
    """Pool rows whose (query_id, gmap_id) has no label yet."""
    q, g = make_pool.COLUMNS.index("query_id"), make_pool.COLUMNS.index("gmap_id")
    return [r for r in pool_rows if (r[q], r[g]) not in labeled]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pool of unlabeled top-K semantic_capped pairs")
    ap.add_argument("--queries", type=Path, default=make_pool.QUERIES_PATH)
    ap.add_argument("--labels", type=Path, default=LABELS_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--top-k", type=int, default=STANCE_RERANK["top_n"])
    ap.add_argument("--dev-only", action="store_true", help="only the 20 dev queries (split seed 6165)")
    ap.add_argument("--queries-only", nargs="*", default=None, help="only these query ids")
    args = ap.parse_args(argv)

    queries = make_pool.read_queries(args.queries)
    if args.dev_only:
        dev, _ = split_queries([q["query_id"] for q in queries])
        queries = [q for q in queries if q["query_id"] in dev]
    if args.queries_only:
        queries = [q for q in queries if q["query_id"] in set(args.queries_only)]
    semantic, doc = make_pool.load_semantic(), make_pool.load_doc_tfidf()
    if semantic is None:
        sys.exit("The semantic index is not built; run: python -m src.retrieval.semantic --build")
    fallback = lambda query, gmap_id: doc.evidence_for(gmap_id, query, make_pool.N_FALLBACK)
    pool_rows, _, _ = make_pool.build_pool(queries, {"semantic": semantic}, systems=BASE_SYSTEM, top_k=args.top_k,
                                           fallback=fallback)
    labeled = read_labels(args.labels)
    rows = new_rows(pool_rows, labeled)
    make_pool.write_csv(args.out, make_pool.COLUMNS, rows)
    status = {}
    for r in rows:
        status[r[make_pool.COLUMNS.index("evidence_status")]] = status.get(r[make_pool.COLUMNS.index("evidence_status")], 0) + 1
    print(f"{len(queries)} queries, top {args.top_k} of semantic_capped: {len(pool_rows)} pairs, "
          f"{len(pool_rows) - len(rows)} already labeled, {len(rows)} NEW (evidence status {status})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
