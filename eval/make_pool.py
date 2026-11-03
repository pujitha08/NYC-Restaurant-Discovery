"""Build the judging pool for the relevance labels.

For every query in queries_draft.csv, take the top 10 restaurants of each system, merge them
(a restaurant found by several systems appears once) and write eval/pool.csv with an empty
`label` column for the judges (see RUBRIC.md). The system that found a restaurant is not
recorded in the file, so judges are not biased by it.

Systems: doc TF-IDF, chunk TF-IDF, and a placeholder for the semantic retriever. A system
that is not built yet is skipped; add it to `SYSTEMS` when it exists.

Run from the repo root:
    python -m eval.make_pool
"""

import csv
import hashlib
from pathlib import Path

from src.retrieval.tfidf import TfidfRetriever, format_snippet
from src.retrieval.tfidf_chunks import ChunkTfidfRetriever

EVAL_DIR = Path(__file__).parent
QUERIES_PATH = EVAL_DIR / "queries_draft.csv"
POOL_PATH = EVAL_DIR / "pool.csv"
TOP_K = 10
N_SNIPPETS = 3
COLUMNS = ["query_id", "gmap_id", "name", "borough"] + [f"best_snippet_{i}" for i in range(1, N_SNIPPETS + 1)] + ["label"]


def load_doc_tfidf():
    return TfidfRetriever.load()


def load_chunk_tfidf():
    return ChunkTfidfRetriever.load()


def load_semantic():
    """Placeholder for the sentence-embedding retriever; return an object with
    search(query, boroughs, top_k) and info(gmap_id) once it exists."""
    return None


# Name -> loader. Snippets come from the first system that has any for a restaurant, so the
# chunk retriever (best evidence filter and excerpts) goes first.
SYSTEMS = {"chunk_tfidf": load_chunk_tfidf, "doc_tfidf": load_doc_tfidf, "semantic": load_semantic}


def shuffle_key(query_id, gmap_id):
    """Stable order that does not reveal the rank in any system."""
    return hashlib.md5(f"{query_id}|{gmap_id}".encode()).hexdigest()


def main():
    with open(QUERIES_PATH, newline="", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    retrievers = {}
    for name, load in SYSTEMS.items():
        r = load()
        if r is None:
            print(f"Skipping {name}: not built yet")
        else:
            retrievers[name] = r

    rows, per_system = [], {name: 0 for name in retrievers}
    for q in queries:
        boroughs = [q["borough"]] if q["borough"] else None
        found = {}  # gmap_id -> (name, borough, snippets)
        for name, r in retrievers.items():
            for res in r.search(q["query"], boroughs=boroughs, top_k=TOP_K):
                per_system[name] += 1
                info = r.info(res.gmap_id)
                snippets = [format_snippet(s, width=None) for s in res.evidence_snippets]
                if res.gmap_id not in found:
                    found[res.gmap_id] = (info["name"], info["borough"], snippets)
                elif not found[res.gmap_id][2] and snippets:
                    found[res.gmap_id] = (info["name"], info["borough"], snippets)
        for gid in sorted(found, key=lambda g: shuffle_key(q["query_id"], g)):
            name, borough, snippets = found[gid]
            snippets = (snippets + [""] * N_SNIPPETS)[:N_SNIPPETS]
            rows.append([q["query_id"], gid, name, borough, *snippets, ""])

    with open(POOL_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)

    print(f"{len(queries)} queries; systems used: {', '.join(retrievers)}")
    for name, n in per_system.items():
        print(f"  {name}: {n} results")
    print(f"Pool: {len(rows)} rows ({len(rows) / len(queries):.1f} per query); wrote {POOL_PATH}")
    empty = sum(1 for r in rows if not r[4])
    print(f"Rows with no snippet: {empty}")


if __name__ == "__main__":
    main()
