"""TF-IDF baseline: rank restaurant documents against a query, with review snippets as evidence.

Build the index (also done automatically on first search):
    python -m src.retrieval.tfidf --build
Search:
    python -m src.retrieval.tfidf "vegan options for a group" --borough Brooklyn
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DATA_DIR
from src.schemas import RetrievalResult

PROCESSED = DATA_DIR / "processed"
DOCS_PATH = PROCESSED / "restaurant_docs.jsonl"
REVIEWS_PATH = PROCESSED / "reviews_clean.jsonl"
INDEX_DIR = PROCESSED / "tfidf"
VECTORIZER_FILE = "vectorizer.joblib"
MATRIX_FILE = "matrix.npz"
META_FILE = "meta.jsonl"

VECTORIZER_PARAMS = dict(sublinear_tf=True, stop_words="english", ngram_range=(1, 2),
                         min_df=3, max_df=0.5, dtype=np.float32)
N_SNIPPETS = 3
METHOD = "tfidf"


def index_review_offsets(path):
    """Map gmap_id -> (byte offset of its first review, review count) in reviews_clean.jsonl.

    A restaurant's reviews are contiguous in the file (build_docs relies on the same property).
    """
    spans = {}
    with open(path, "rb") as f:
        offset = 0
        for line in f:
            gid = json.loads(line)["gmap_id"]
            if gid in spans:
                spans[gid][1] += 1
            else:
                spans[gid] = [offset, 1]
            offset += len(line)
    return spans


class TfidfRetriever:
    def __init__(self, vectorizer, matrix, meta, reviews_path=REVIEWS_PATH):
        self.vectorizer = vectorizer
        self.matrix = matrix  # (n_restaurants, n_terms), rows L2-normalized, CSR
        self.meta = meta  # one dict per matrix row
        self.reviews_path = Path(reviews_path)
        self._boroughs = np.array([m["borough"].lower() for m in meta])
        self._low_evidence = np.array([m["low_evidence"] for m in meta])

    # ---- build / save / load -------------------------------------------
    @classmethod
    def build(cls, docs_path=DOCS_PATH, reviews_path=REVIEWS_PATH):
        spans = index_review_offsets(reviews_path)
        texts, meta = [], []
        with open(docs_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                texts.append(d.pop("text"))
                offset, count = spans.get(d["gmap_id"], (0, 0))
                meta.append({**d, "review_offset": offset, "review_count": count})
        vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
        matrix = vectorizer.fit_transform(texts).tocsr()
        return cls(vectorizer, matrix, meta, reviews_path)

    def save(self, index_dir=INDEX_DIR):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, index_dir / VECTORIZER_FILE)
        sparse.save_npz(index_dir / MATRIX_FILE, self.matrix)
        with open(index_dir / META_FILE, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_dir=INDEX_DIR, reviews_path=REVIEWS_PATH):
        index_dir = Path(index_dir)
        vectorizer = joblib.load(index_dir / VECTORIZER_FILE)  # our own file; pickle is trusted
        matrix = sparse.load_npz(index_dir / MATRIX_FILE).tocsr()
        with open(index_dir / META_FILE, encoding="utf-8") as f:
            meta = [json.loads(line) for line in f]
        return cls(vectorizer, matrix, meta, reviews_path)

    @classmethod
    def exists(cls, index_dir=INDEX_DIR):
        return all((Path(index_dir) / n).exists() for n in (VECTORIZER_FILE, MATRIX_FILE, META_FILE))

    # ---- search ---------------------------------------------------------
    def search(self, query_terms, boroughs=None, top_k=10, include_low_evidence=False):
        """Rank restaurants by cosine similarity to the query; low_evidence ones are skipped by default."""
        q = self.vectorizer.transform([query_terms])
        if q.nnz == 0:  # no query term is in the vocabulary
            return []
        scores = (self.matrix @ q.T).toarray().ravel()

        allowed = np.ones(len(scores), dtype=bool)
        if boroughs:
            allowed &= np.isin(self._boroughs, [b.lower() for b in boroughs])
        if not include_low_evidence:
            allowed &= ~self._low_evidence
        scores = np.where(allowed, scores, 0.0)

        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [RetrievalResult(gmap_id=self.meta[i]["gmap_id"], score=float(scores[i]), method=METHOD,
                                evidence_snippets=self.snippets(i, q))
                for i in top if scores[i] > 0]

    def snippets(self, row, q, n=N_SNIPPETS):
        """The n cleaned reviews of restaurant `row` most similar to the query vector `q`."""
        m = self.meta[row]
        if not m["review_count"]:
            return []
        with open(self.reviews_path, "rb") as f:
            f.seek(m["review_offset"])
            texts = [json.loads(f.readline())["text"] for _ in range(m["review_count"])]
        sims = (self.vectorizer.transform(texts) @ q.T).toarray().ravel()
        best = np.argsort(-sims)[:n]
        return [texts[i] for i in best if sims[i] > 0]

    def info(self, gmap_id):
        return next(m for m in self.meta if m["gmap_id"] == gmap_id)


def print_results(retriever, results, width=240):
    for rank, r in enumerate(results, 1):
        m = retriever.info(r.gmap_id)
        print(f"{rank:>2}. {m['name']} ({m['borough']})  score={r.score:.3f}  "
              f"reviews={m['num_clean_reviews']}  {m['address']}")
        for s in r.evidence_snippets:
            print(f"      - {s[:width]}{'…' if len(s) > width else ''}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="TF-IDF restaurant search")
    ap.add_argument("query", nargs="?", help="query text, e.g. 'vegan options for a group'")
    ap.add_argument("--borough", action="append", help="restrict to a borough (repeat for several)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--include-low-evidence", action="store_true",
                    help="also rank restaurants with fewer than 5 clean reviews")
    ap.add_argument("--build", action="store_true", help="(re)build and save the index")
    args = ap.parse_args(argv)

    if args.build or not TfidfRetriever.exists():
        t = time.time()
        print("Building TF-IDF index ...", flush=True)
        retriever = TfidfRetriever.build()
        retriever.save()
        print(f"Built in {time.time() - t:.0f}s: {retriever.matrix.shape[0]:,} documents x "
              f"{retriever.matrix.shape[1]:,} terms, {retriever.matrix.nnz:,} nonzeros; saved to {INDEX_DIR}")
    else:
        retriever = TfidfRetriever.load()
    if not args.query:
        return
    results = retriever.search(args.query, boroughs=args.borough, top_k=args.top_k,
                               include_low_evidence=args.include_low_evidence)
    if not results:
        print("No matches.")
    print_results(retriever, results)


if __name__ == "__main__":
    main()
