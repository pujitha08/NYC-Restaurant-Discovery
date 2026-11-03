"""Single-review evidence for retrieval results.

Both retrievers return the reviews that best match the query, each with its review_id and
time. This module only picks and loads reviews; it has no say in how restaurants are ranked.
"""

import json
import re
from pathlib import Path

import numpy as np

from src.config import DATA_DIR
from src.schemas import Snippet

REVIEWS_PATH = DATA_DIR / "processed" / "reviews_clean.jsonl"
INDEX_PATH = DATA_DIR / "processed" / "reviews_clean_index.npz"
N_SNIPPETS = 3
MAX_WORDS = 60  # excerpt length
REQUIREMENT_MARGIN = 1.0  # idf distance from the rarest query term
_WORD_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
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


def query_terms(vectorizer, q):
    """Unigram features of query vector `q` as {term: idf}."""
    names = vectorizer.get_feature_names_out()
    return {names[j]: float(vectorizer.idf_[j]) for j in q.indices if " " not in names[j]}


def requirement_terms(vectorizer, q, margin=REQUIREMENT_MARGIN):
    """The most specific query terms: unigrams whose idf is within `margin` of the query's rarest one.

    idf is a proxy for specificity: in "nut-free bakery", nut (7.9) is a requirement while bakery
    (5.9, a category) and free (4.3) are not; in "celiac safe" only celiac is. A margin of 1.0
    keeps terms up to e times as common as the rarest. A heuristic, to be replaced by the LLM
    query parser's allergens/dietary_restrictions.
    """
    terms = query_terms(vectorizer, q)
    if not terms:
        return []
    top = max(terms.values())
    return [t for t, idf in terms.items() if idf >= top - margin]


def _has_term(text, terms):
    """True if a word of `text` equals a term or its plural (nut ~ nuts)."""
    words = set(_WORD_RE.findall(text.lower()))
    return any(t in words or t + "s" in words or t + "es" in words for t in terms)


def excerpt(text, weights, max_words=MAX_WORDS):
    """The sentence window of at most `max_words` words around the best query match, with ellipses.

    The best sentence is the one whose words carry the most query-term weight (idf); neighbouring
    sentences are added while they fit. A best sentence longer than max_words is cut to a
    max_words window around its first query term.
    """
    if len(text.split()) <= max_words:
        return text
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    lengths = [len(s.split()) for s in sentences]

    def weight(s):
        words = set(_WORD_RE.findall(s.lower()))
        return sum(w for t, w in weights.items() if t in words or t + "s" in words or t + "es" in words)

    best = max(range(len(sentences)), key=lambda i: weight(sentences[i]))
    lo = hi = best
    total = lengths[best]
    grew = True
    while grew:
        grew = False
        for j in (lo - 1, hi + 1):  # previous sentence first, then next
            if 0 <= j < len(sentences) and total + lengths[j] <= max_words:
                lo, hi, total, grew = min(lo, j), max(hi, j), total + lengths[j], True
    words = " ".join(sentences[lo:hi + 1]).split()
    if len(words) > max_words:  # a single sentence over the limit: window around its first match
        hit = next((i for i, w in enumerate(words)
                    if any(t in _WORD_RE.findall(w.lower()) for t in weights)), 0)
        start = max(0, min(hit - max_words // 2, len(words) - max_words))
        words = words[start:start + max_words]
        return ("… " if start > 0 or lo > 0 else "") + " ".join(words) + " …"
    out = " ".join(words)
    return ("… " if lo > 0 else "") + out + (" …" if hi < len(sentences) - 1 else "")


def best_reviews(vectorizer, q, records, n=N_SNIPPETS, requirements=None, max_words=MAX_WORDS):
    """The n reviews most similar to query vector `q` (TF-IDF cosine), as excerpted Snippets.

    Reviews are ranked by whole-review similarity; zero-score ones are dropped. If `requirements`
    is given (see requirement_terms), a review is kept only if it contains at least one of those
    terms. Each kept review is trimmed with excerpt() to `max_words` words (None keeps it whole).
    review_id and time are those of the source review.
    """
    if not records:
        return []
    sims = (vectorizer.transform([r["text"] for r in records]) @ q.T).toarray().ravel()
    weights = query_terms(vectorizer, q)
    out = []
    for i in np.argsort(-sims):
        if sims[i] <= 0 or len(out) == n:
            break
        r = records[i]
        if requirements and not _has_term(r["text"], requirements):
            continue
        text = excerpt(r["text"], weights, max_words) if max_words else r["text"]
        out.append(Snippet(text=text, review_id=r["review_id"], time=r["time"]))
    return out
