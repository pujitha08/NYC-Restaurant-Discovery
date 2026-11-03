"""Single-review evidence for retrieval results.

Both retrievers return the reviews that best match the query, each with its review_id and
time. This module only picks and loads reviews; it has no say in how restaurants are ranked.
"""

import json
from pathlib import Path

import numpy as np

from src.config import DATA_DIR
from src.schemas import Snippet

REVIEWS_PATH = DATA_DIR / "processed" / "reviews_clean.jsonl"
INDEX_PATH = DATA_DIR / "processed" / "reviews_clean_index.npz"
N_SNIPPETS = 3
_ID_PREFIX = b'{"review_id": '  # how clean_reviews.py starts every line


class ReviewStore:
    """Random access to reviews_clean.jsonl by review_id.

    The id -> byte offset index is built on first use, saved next to the data, and rebuilt
    if the reviews file changes size.
    """

    def __init__(self, path=REVIEWS_PATH, index_path=INDEX_PATH):
        self.path = Path(path)
        self.index_path = Path(index_path)
        self._ids = self._offsets = None

    def _build_index(self):
        ids, offsets, pos = [], [], 0
        with open(self.path, "rb") as f:
            for line in f:
                assert line.startswith(_ID_PREFIX), "unexpected reviews_clean.jsonl line format"
                ids.append(int(line[len(_ID_PREFIX):line.index(b",")]))
                offsets.append(pos)
                pos += len(line)
        ids, offsets = np.array(ids, dtype=np.int64), np.array(offsets, dtype=np.int64)
        assert (np.diff(ids) > 0).all(), "review_ids are not strictly increasing"
        np.savez(self.index_path, ids=ids, offsets=offsets, size=self.path.stat().st_size)
        return ids, offsets

    def _load_index(self):
        if self._ids is None:
            if self.index_path.exists():
                z = np.load(self.index_path)
                if int(z["size"]) == self.path.stat().st_size:
                    self._ids, self._offsets = z["ids"], z["offsets"]
            if self._ids is None:
                self._ids, self._offsets = self._build_index()

    def get(self, review_ids):
        """Reviews for the given ids (unknown ids are skipped), as dicts, in the order given."""
        self._load_index()
        out = []
        with open(self.path, "rb") as f:
            for rid in review_ids:
                i = np.searchsorted(self._ids, rid)
                if i < len(self._ids) and self._ids[i] == rid:
                    f.seek(self._offsets[i])
                    out.append(json.loads(f.readline()))
        return out

    def read_span(self, offset, count):
        """`count` consecutive reviews starting at byte `offset` (one restaurant's reviews)."""
        with open(self.path, "rb") as f:
            f.seek(offset)
            return [json.loads(f.readline()) for _ in range(count)]


def best_reviews(vectorizer, q, records, n=N_SNIPPETS):
    """The n reviews most similar to query vector `q` (TF-IDF cosine), as Snippets; zero-score ones are dropped."""
    if not records:
        return []
    sims = (vectorizer.transform([r["text"] for r in records]) @ q.T).toarray().ravel()
    best = np.argsort(-sims)[:n]
    return [Snippet(text=records[i]["text"], review_id=records[i]["review_id"], time=records[i]["time"])
            for i in best if sims[i] > 0]
