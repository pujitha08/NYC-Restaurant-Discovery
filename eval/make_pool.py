"""Build the judging pool for the relevance labels, plus the per-system runs for the metrics.

For every query, take the top 10 restaurants of each system, merge them (a restaurant found by
several systems appears once per query) and write:

- pool.csv: one row per (query, restaurant) with up to six evidence reviews and an empty
  `label` column for the judges (see RUBRIC.md). It does not say which system found a
  restaurant, and rows are ordered by a hash, not by rank, so judges are not biased by either.

  Evidence is pooled: the union of the snippets of every system that found the restaurant,
  one per review_id, ordered by how many retrievers (doc TF-IDF, chunk TF-IDF, semantic)
  returned that review, then by its best position in a system's snippet list. Snippet scores
  are not comparable across retrievers, so agreement is the cross-system ordering.
  If no system returned a snippet, up to three of the restaurant's own reviews containing
  the query's requirement terms are added (doc TF-IDF's evidence_for). evidence_status is
  "retrieved", "fallback" or "none" (no review shown).
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
MAX_EVIDENCE = 6  # pooled snippets per row
N_FALLBACK = 3  # reviews added when no system returned a snippet
EVIDENCE_COLUMNS = [f"evidence_{i}" for i in range(1, MAX_EVIDENCE + 1)]
COLUMNS = (["query_id", "query", "gmap_id", "name", "borough", "evidence_status", "evidence_review_ids"]
           + EVIDENCE_COLUMNS + ["label"])
EVIDENCE_RETRIEVED, EVIDENCE_FALLBACK, EVIDENCE_NONE = "retrieved", "fallback", "none"
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

# (system name, retriever name, extra search() arguments). Order is the last tie-breaker for
# pooled evidence, and decides which excerpt of a review is shown when retrievers cut it
# differently: chunk TF-IDF first, semantic (always >= 1 snippet, even off-topic) last.
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


def pool_evidence(hits, max_evidence=MAX_EVIDENCE):
    """Union of the snippets in `hits`, one per review_id, best-supported first.

    hits: [(retriever name, system index, snippets)] for one (query, restaurant) pair. A review's
    support is the number of distinct retrievers that returned it (log/capped variants of one
    retriever count once); ties go to its best position in any snippet list, then system order.
    The excerpt shown is the one from the earliest system that returned the review.
    """
    best = {}  # review_id -> {"retrievers", "pos", "sys", "snippet"}
    for rname, sys_index, snippets in sorted(hits, key=lambda h: h[1]):  # earliest system first
        for pos, s in enumerate(snippets):
            entry = best.setdefault(s.review_id, {"retrievers": set(), "pos": pos, "sys": sys_index, "snippet": s})
            entry["retrievers"].add(rname)
            entry["pos"] = min(entry["pos"], pos)
    ranked = sorted(best.values(), key=lambda e: (-len(e["retrievers"]), e["pos"], e["sys"], e["snippet"].review_id))
    return [e["snippet"] for e in ranked[:max_evidence]]


def build_pool(queries, retrievers, systems=SYSTEMS, top_k=TOP_K, fallback=None):
    """Run every available system on every query.

    retrievers: retriever name -> loaded retriever (systems whose retriever is missing are skipped).
    fallback: optional callable(query_text, gmap_id) -> [Snippet], used when no system returned
    a snippet for a pair.
    Returns (pool rows, run rows, finders) where finders maps (query_id, gmap_id) -> set of systems.
    """
    active = [(i, name, rname, retrievers[rname], kwargs)
              for i, (name, rname, kwargs) in enumerate(systems) if retrievers.get(rname)]
    pool_rows, run_rows, finders = [], [], {}
    for q in queries:
        qid = q["query_id"]
        boroughs = [q["borough"].strip()] if q.get("borough", "").strip() else None
        found = {}  # gmap_id -> {"info": (name, borough), "hits": [(retriever, system index, snippets)]}
        for i, name, rname, r, kwargs in active:
            for rank, res in enumerate(r.search(q["query"], boroughs=boroughs, top_k=top_k, **kwargs), 1):
                run_rows.append([qid, name, rank, res.gmap_id, repr(float(res.score))])
                finders.setdefault((qid, res.gmap_id), set()).add(name)
                if res.gmap_id not in found:
                    info = r.info(res.gmap_id)
                    found[res.gmap_id] = {"info": (info["name"], info["borough"]), "hits": []}
                found[res.gmap_id]["hits"].append((rname, i, res.evidence_snippets))
        for gid in sorted(found, key=lambda g: shuffle_key(qid, g)):
            name, borough = found[gid]["info"]
            evidence, status = pool_evidence(found[gid]["hits"]), EVIDENCE_RETRIEVED
            if not evidence:
                evidence = (fallback(q["query"], gid) if fallback else [])[:N_FALLBACK]
                status = EVIDENCE_FALLBACK if evidence else EVIDENCE_NONE
            texts = [format_snippet(s, width=None) for s in evidence]
            ids = ";".join(str(s.review_id) for s in evidence)
            pool_rows.append([qid, q["query"], gid, name, borough, status, ids,
                              *(texts + [""] * MAX_EVIDENCE)[:MAX_EVIDENCE], ""])
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
    status = Counter(r[COLUMNS.index("evidence_status")] for r in pool_rows)
    n_evidence = Counter(sum(1 for c in r[COLUMNS.index("evidence_1"):-1] if c) for r in pool_rows)
    print(f"Evidence status: {dict(status)}; reviews shown per row: {dict(sorted(n_evidence.items()))}")


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

    doc = retrievers.get("doc_tfidf")
    fallback = (lambda query, gmap_id: doc.evidence_for(gmap_id, query, N_FALLBACK)) if doc else None
    pool_rows, run_rows, finders = build_pool(queries, retrievers, fallback=fallback)
    write_csv(args.pool, COLUMNS, pool_rows)
    write_csv(args.runs, RUN_COLUMNS, run_rows)
    summarize(queries, pool_rows, run_rows, finders, [n for n, _, _ in SYSTEMS if n not in skipped])
    print(f"Wrote {args.pool} and {args.runs}")


if __name__ == "__main__":
    main()
