"""Build the judging pool for the relevance labels, plus the per-system runs for the metrics.

For every query, take the top 10 restaurants of each system, merge them (a restaurant found by
several systems appears once per query) and write:

- pool.csv: one row per (query, restaurant) with three evidence snippets and an empty `label`
  column for the judges (see RUBRIC.md). It does not say which system found a restaurant, and
  rows are ordered by a hash, not by rank, so judges are not biased by either.
- runs.csv: query_id, system, rank, gmap_id, score for every system's full top 10. The metrics
  script needs it. Judges should not open it (it would reveal the systems).

Both files are git-ignored.

Systems (SYSTEMS): doc TF-IDF; chunk TF-IDF and semantic, each with the "log" and "capped"
density modes. The semantic systems are skipped if its index has not been built
(python -m src.retrieval.semantic --build).

Run from the repo root:
    python -m eval.make_pool [--queries eval/queries_draft.csv] [--pool eval/pool.csv] [--runs eval/runs.csv]
"""

import argparse
import csv
import hashlib
import statistics
from collections import Counter
from pathlib import Path

from src.retrieval.tfidf import format_snippet

EVAL_DIR = Path(__file__).parent
QUERIES_PATH = EVAL_DIR / "queries_draft.csv"
POOL_PATH = EVAL_DIR / "pool.csv"
RUNS_PATH = EVAL_DIR / "runs.csv"
TOP_K = 10
N_SNIPPETS = 3
COLUMNS = ["query_id", "gmap_id", "name", "borough"] + [f"best_snippet_{i}" for i in range(1, N_SNIPPETS + 1)] + ["label"]
RUN_COLUMNS = ["query_id", "system", "rank", "gmap_id", "score"]
QUERY_COLUMNS = ("query_id", "query", "borough")


def load_doc_tfidf():
    from src.retrieval.tfidf import TfidfRetriever
    return TfidfRetriever.load()


def load_chunk_tfidf():
    from src.retrieval.tfidf_chunks import ChunkTfidfRetriever
    return ChunkTfidfRetriever.load()


def load_semantic():
    """The sentence-embedding retriever, or None if its index has not been built.

    A stale index (chunks/docs changed since the build) raises CacheMismatchError rather than
    being skipped, so a pool is never built from mismatched embeddings.
    """
    from src.retrieval.semantic import SemanticRetriever  # needs sentence-transformers
    if not SemanticRetriever.exists():
        print("Semantic index not built; run: python -m src.retrieval.semantic --build")
        return None
    return SemanticRetriever.load()


# Retriever name -> loader; each is loaded once and shared by its systems.
RETRIEVERS = {"chunk_tfidf": load_chunk_tfidf, "doc_tfidf": load_doc_tfidf, "semantic": load_semantic}

# (system name, retriever name, extra search() arguments). Order matters only for snippets: a
# restaurant's snippets come from the first system in this list that has any for it, so chunk
# TF-IDF (requirement-filtered excerpts) goes first and semantic (always >= 1 snippet) last.
SYSTEMS = [
    ("chunk_tfidf_log", "chunk_tfidf", {"density_mode": "log"}),
    ("chunk_tfidf_capped", "chunk_tfidf", {"density_mode": "capped"}),
    ("doc_tfidf", "doc_tfidf", {}),
    ("semantic_log", "semantic", {"density_mode": "log"}),
    ("semantic_capped", "semantic", {"density_mode": "capped"}),
]


def shuffle_key(query_id, gmap_id):
    """Stable order that does not reveal the rank in any system."""
    return hashlib.md5(f"{query_id}|{gmap_id}".encode()).hexdigest()


def read_queries(path):
    """Rows of a query CSV with query_id, query and borough columns (borough may be empty)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in QUERY_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        queries = list(reader)
    dupes = [qid for qid, n in Counter(q["query_id"] for q in queries).items() if n > 1]
    if dupes:
        raise ValueError(f"{path}: duplicate query_id(s) {', '.join(dupes)}")
    return queries


def build_pool(queries, retrievers, systems=SYSTEMS, top_k=TOP_K):
    """Run every available system on every query.

    retrievers: retriever name -> loaded retriever (systems whose retriever is missing are skipped).
    Returns (pool rows, run rows, finders) where finders maps (query_id, gmap_id) -> set of systems.
    """
    active = [(name, retrievers[rname], kwargs) for name, rname, kwargs in systems if retrievers.get(rname)]
    pool_rows, run_rows, finders = [], [], {}
    for q in queries:
        qid = q["query_id"]
        boroughs = [q["borough"].strip()] if q.get("borough", "").strip() else None
        found = {}  # gmap_id -> (name, borough, snippets); first system with snippets wins
        for name, r, kwargs in active:
            for rank, res in enumerate(r.search(q["query"], boroughs=boroughs, top_k=top_k, **kwargs), 1):
                run_rows.append([qid, name, rank, res.gmap_id, repr(float(res.score))])
                finders.setdefault((qid, res.gmap_id), set()).add(name)
                snippets = [format_snippet(s, width=None) for s in res.evidence_snippets]
                if res.gmap_id not in found or (not found[res.gmap_id][2] and snippets):
                    info = r.info(res.gmap_id)
                    found[res.gmap_id] = (info["name"], info["borough"], snippets)
        for gid in sorted(found, key=lambda g: shuffle_key(qid, g)):
            name, borough, snippets = found[gid]
            snippets = (snippets + [""] * N_SNIPPETS)[:N_SNIPPETS]
            pool_rows.append([qid, gid, name, borough, *snippets, ""])
    return pool_rows, run_rows, finders


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def summarize(queries, pool_rows, run_rows, finders, systems_used):
    per_query = Counter(r[0] for r in pool_rows)
    sizes = [per_query.get(q["query_id"], 0) for q in queries]
    print(f"{len(queries)} queries; systems used: {', '.join(systems_used) or 'none'}")
    for name, n in Counter(r[1] for r in run_rows).items():
        print(f"  {name}: {n} results")
    if sizes:
        print(f"Pool: {len(pool_rows)} pairs; per query min {min(sizes)}, median {statistics.median(sizes):g}, "
              f"max {max(sizes)}")
    single = Counter(next(iter(s)) for s in finders.values() if len(s) == 1)
    print(f"Pairs found by only one system: {sum(single.values())} " + str(dict(single)))
    capped_only = sum(1 for s in finders.values() if s and all(n.endswith("_capped") for n in s))
    print(f"Pairs found only by *_capped systems: {capped_only}")
    print(f"Rows with no snippet: {sum(1 for r in pool_rows if not r[4])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the blind judging pool and the per-system runs")
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH, help=f"query CSV (default: {QUERIES_PATH})")
    ap.add_argument("--pool", type=Path, default=POOL_PATH, help=f"pool output (default: {POOL_PATH})")
    ap.add_argument("--runs", type=Path, default=RUNS_PATH, help=f"runs output (default: {RUNS_PATH})")
    args = ap.parse_args(argv)

    queries = read_queries(args.queries)
    needed = {rname for _, rname, _ in SYSTEMS}
    retrievers = {}
    for rname, load in RETRIEVERS.items():
        if rname in needed:
            retrievers[rname] = load()
    skipped = [name for name, rname, _ in SYSTEMS if retrievers.get(rname) is None]
    for name in skipped:
        print(f"Skipping {name}: retriever not available")

    pool_rows, run_rows, finders = build_pool(queries, retrievers)
    write_csv(args.pool, COLUMNS, pool_rows)
    write_csv(args.runs, RUN_COLUMNS, run_rows)
    summarize(queries, pool_rows, run_rows, finders, [n for n, _, _ in SYSTEMS if n not in skipped])
    print(f"Wrote {args.pool} and {args.runs}")


if __name__ == "__main__":
    main()
