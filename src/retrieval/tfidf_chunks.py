"""Chunk-level TF-IDF: score review chunks, then aggregate per restaurant.

Document-level TF-IDF favors small restaurants (a single mention is a large share of a short
document). Scoring fixed-size chunks removes that length bias.

Build the index (also done automatically on first search):
    python -m src.retrieval.tfidf_chunks --build
Search:
    python -m src.retrieval.tfidf_chunks "vegan options for a group" --borough Brooklyn
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.retrieval.evidence import REVIEWS_PATH, ReviewStore, best_reviews
from src.retrieval.tfidf import DOCS_PATH, PROCESSED, VECTORIZER_PARAMS, print_results
from src.schemas import RetrievalResult

CHUNKS_PATH = PROCESSED / "review_chunks.jsonl"
INDEX_DIR = PROCESSED / "tfidf_chunks"
VECTORIZER_FILE = "vectorizer.joblib"
MATRIX_FILE = "matrix.npz"
OFFSETS_FILE = "chunk_offsets.npy"  # byte offset of each chunk in review_chunks.jsonl
OWNER_FILE = "chunk_restaurant.npy"  # restaurant row (index into meta) of each chunk
META_FILE = "meta.jsonl"

METHOD = "tfidf_chunks"
TOP_CHUNKS = 3  # chunks summed per restaurant, and returned as evidence
DENSITY_THRESHOLD = 0.05  # a chunk "matches" above this cosine score
DENSITY_WEIGHT = 0.1  # weight of log(1 + number of matching chunks)
DENSITY_CAP = 0.2  # ceiling of the bonus in "capped" mode
# How the number of matching chunks adds to a restaurant's score:
#   "log"    0.1 * log(1 + n)
#   "none"   no bonus: score is the sum of the top-3 chunk scores
#   "capped" min(0.1 * log(1 + n), 0.2)
DENSITY_MODES = ("log", "none", "capped")


class ChunkTfidfRetriever:
    def __init__(self, vectorizer, matrix, offsets, owner, meta, chunks_path=CHUNKS_PATH,
                 reviews_path=REVIEWS_PATH, density_mode="log"):
        if density_mode not in DENSITY_MODES:
            raise ValueError(f"density_mode must be one of {DENSITY_MODES}, got {density_mode!r}")
        self.density_mode = density_mode
        self.vectorizer = vectorizer
        self.matrix = matrix  # (n_chunks, n_terms), rows L2-normalized, CSR
        self.offsets = offsets
        self.owner = owner
        self.meta = meta  # one dict per restaurant (row of restaurant_docs.jsonl, minus text)
        self.chunks_path = Path(chunks_path)
        self.store = ReviewStore(reviews_path)
        self._by_id = {m["gmap_id"]: i for i, m in enumerate(meta)}
        self._boroughs = np.array([m["borough"].lower() for m in meta])
        self._low_evidence = np.array([m["low_evidence"] for m in meta])

    # ---- build / save / load -------------------------------------------
    @classmethod
    def build(cls, docs_path=DOCS_PATH, chunks_path=CHUNKS_PATH):
        meta, row = [], {}
        with open(docs_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                d.pop("text")
                row[d["gmap_id"]] = len(meta)
                meta.append(d)
        texts, offsets, owner = [], [], []
        with open(chunks_path, "rb") as f:
            pos = 0
            for line in f:
                c = json.loads(line)
                offsets.append(pos)
                pos += len(line)
                texts.append(c["text"])
                owner.append(row[c["gmap_id"]])
        vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
        matrix = vectorizer.fit_transform(texts).tocsr()
        return cls(vectorizer, matrix, np.array(offsets, dtype=np.int64), np.array(owner, dtype=np.int32),
                   meta, chunks_path)

    def save(self, index_dir=INDEX_DIR):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, index_dir / VECTORIZER_FILE)
        sparse.save_npz(index_dir / MATRIX_FILE, self.matrix)
        np.save(index_dir / OFFSETS_FILE, self.offsets)
        np.save(index_dir / OWNER_FILE, self.owner)
        with open(index_dir / META_FILE, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_dir=INDEX_DIR, chunks_path=CHUNKS_PATH, reviews_path=REVIEWS_PATH, density_mode="log"):
        index_dir = Path(index_dir)
        vectorizer = joblib.load(index_dir / VECTORIZER_FILE)  # our own file; pickle is trusted
        matrix = sparse.load_npz(index_dir / MATRIX_FILE).tocsr()
        with open(index_dir / META_FILE, encoding="utf-8") as f:
            meta = [json.loads(line) for line in f]
        return cls(vectorizer, matrix, np.load(index_dir / OFFSETS_FILE), np.load(index_dir / OWNER_FILE),
                   meta, chunks_path, reviews_path, density_mode)

    @classmethod
    def exists(cls, index_dir=INDEX_DIR):
        names = (VECTORIZER_FILE, MATRIX_FILE, OFFSETS_FILE, OWNER_FILE, META_FILE)
        return all((Path(index_dir) / n).exists() for n in names)

    # ---- search ---------------------------------------------------------
    def _aggregate(self, q, density_mode):
        """Per-restaurant score = sum of top-3 chunk scores + density bonus (see DENSITY_MODES).

        Returns (scores per restaurant, chunk ids sorted by (restaurant, -score), their restaurants).
        """
        chunk_scores = (self.matrix @ q.T).toarray().ravel()
        hit = np.flatnonzero(chunk_scores > 0)
        owners, scores = self.owner[hit], chunk_scores[hit]
        order = np.lexsort((-scores, owners))  # by restaurant, then best chunk first
        hit, owners, scores = hit[order], owners[order], scores[order]

        n = len(self.meta)
        first = np.searchsorted(owners, owners, side="left")  # start of each restaurant's run
        rank_in_restaurant = np.arange(len(owners)) - first
        top = rank_in_restaurant < TOP_CHUNKS
        total = np.bincount(owners[top], weights=scores[top], minlength=n)
        matching = np.bincount(owners[scores > DENSITY_THRESHOLD], minlength=n)
        return total + self._density_bonus(matching, density_mode), hit, owners

    @staticmethod
    def _density_bonus(matching, density_mode):
        bonus = DENSITY_WEIGHT * np.log1p(matching)
        if density_mode == "none":
            return np.zeros_like(bonus)
        if density_mode == "capped":
            return np.minimum(bonus, DENSITY_CAP)
        return bonus

    def _mode(self, density_mode):
        density_mode = density_mode or self.density_mode
        if density_mode not in DENSITY_MODES:
            raise ValueError(f"density_mode must be one of {DENSITY_MODES}, got {density_mode!r}")
        return density_mode

    def restaurant_scores(self, query_terms, boroughs=None, include_low_evidence=False, density_mode=None):
        """Aggregated score of every restaurant (0 where filtered out or unmatched)."""
        q = self.vectorizer.transform([query_terms])
        if q.nnz == 0:
            return np.zeros(len(self.meta))
        return self._mask(self._aggregate(q, self._mode(density_mode))[0], boroughs, include_low_evidence)

    def _mask(self, scores, boroughs, include_low_evidence):
        allowed = np.ones(len(scores), dtype=bool)
        if boroughs:
            allowed &= np.isin(self._boroughs, [b.lower() for b in boroughs])
        if not include_low_evidence:
            allowed &= ~self._low_evidence
        return np.where(allowed, scores, 0.0)

    def search(self, query_terms, boroughs=None, top_k=10, include_low_evidence=False, density_mode=None):
        """Rank restaurants by aggregated chunk score; low_evidence ones are skipped by default.

        density_mode overrides the retriever's default for this call.
        """
        q = self.vectorizer.transform([query_terms])
        if q.nnz == 0:  # no query term is in the vocabulary
            return []
        scores, hit, owners = self._aggregate(q, self._mode(density_mode))
        scores = self._mask(scores, boroughs, include_low_evidence)

        k = min(top_k, len(scores))
        best = np.argpartition(-scores, k - 1)[:k]
        best = best[np.argsort(-scores[best])]
        results = []
        for i in best:
            if scores[i] <= 0:
                continue
            start = np.searchsorted(owners, i, side="left")  # best chunk of restaurant i comes first
            chunk_ids = hit[start:start + TOP_CHUNKS]
            results.append(RetrievalResult(gmap_id=self.meta[i]["gmap_id"], score=float(scores[i]),
                                           method=METHOD, evidence_snippets=self._evidence(chunk_ids, q)))
        return results

    def _evidence(self, chunk_ids, q):
        """Split the chunks back into their reviews and return the 3 reviews that best match the query."""
        review_ids = []
        with open(self.chunks_path, "rb") as f:
            for cid in chunk_ids:
                f.seek(self.offsets[cid])
                review_ids += json.loads(f.readline())["review_ids"]
        unique_ids = list(dict.fromkeys(review_ids))  # a long review split over chunks appears once
        return best_reviews(self.vectorizer, q, self.store.get(unique_ids))

    def info(self, gmap_id):
        return self.meta[self._by_id[gmap_id]]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chunk-level TF-IDF restaurant search")
    ap.add_argument("query", nargs="?", help="query text, e.g. 'vegan options for a group'")
    ap.add_argument("--borough", action="append", help="restrict to a borough (repeat for several)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--include-low-evidence", action="store_true",
                    help="also rank restaurants with fewer than 5 clean reviews")
    ap.add_argument("--density-mode", choices=DENSITY_MODES, default="log",
                    help="bonus for restaurants with many matching chunks (default: log)")
    ap.add_argument("--build", action="store_true", help="(re)build and save the index")
    args = ap.parse_args(argv)

    if args.build or not ChunkTfidfRetriever.exists():
        t = time.time()
        print("Building chunk TF-IDF index ...", flush=True)
        retriever = ChunkTfidfRetriever.build()
        retriever.save()
        retriever.density_mode = args.density_mode
        print(f"Built in {time.time() - t:.0f}s: {retriever.matrix.shape[0]:,} chunks x "
              f"{retriever.matrix.shape[1]:,} terms, {retriever.matrix.nnz:,} nonzeros; saved to {INDEX_DIR}")
    else:
        retriever = ChunkTfidfRetriever.load(density_mode=args.density_mode)
    if not args.query:
        return
    results = retriever.search(args.query, boroughs=args.borough, top_k=args.top_k,
                               include_low_evidence=args.include_low_evidence)
    if not results:
        print("No matches.")
    print_results(retriever, results)


if __name__ == "__main__":
    main()
