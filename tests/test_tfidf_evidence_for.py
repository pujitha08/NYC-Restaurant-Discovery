"""TfidfRetriever.evidence_for: fallback evidence from a restaurant's own reviews (synthetic files)."""

import json
import tempfile
import unittest
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from src.retrieval.tfidf import TfidfRetriever, index_review_offsets

REVIEWS = [  # (gmap_id, text); a restaurant's reviews are contiguous, as in reviews_clean.jsonl
    ("A", "Great halal chicken over rice, the halal cart is the best."),
    ("A", "Long line but friendly staff."),
    ("A", "Their halal lamb is tender."),
    ("A", "Cheap and fast."),
    ("B", "Best chinese food in the area."),
    ("B", "Dumplings are amazing, chinese food done right."),
    ("C", "Halal options and chinese food on the same menu."),
] + [(f"F{k}", "The chinese food here is fine.") for k in range(9)]  # make "chinese" common, "halal" rare
RESTAURANTS = list(dict.fromkeys(g for g, _ in REVIEWS))


class EvidenceForTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "reviews_clean.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rid, (gid, text) in enumerate(REVIEWS):
                f.write(json.dumps({"review_id": rid, "gmap_id": gid, "user_id": "u", "rating": 5,
                                    "time": 1_600_000_000_000, "text": text}) + "\n")
        spans = index_review_offsets(path)
        meta, texts = [], []
        for gid in RESTAURANTS:
            offset, count = spans[gid]
            meta.append({"gmap_id": gid, "name": gid, "borough": "Manhattan", "low_evidence": False,
                         "num_clean_reviews": count, "address": "", "review_offset": offset, "review_count": count})
            texts.append(" ".join(t for g, t in REVIEWS if g == gid))
        vectorizer = TfidfVectorizer(stop_words="english")
        self.r = TfidfRetriever(vectorizer, vectorizer.fit_transform(texts).tocsr(), meta, reviews_path=path)

    def test_returns_only_this_restaurants_reviews_with_the_requirement_term(self):
        snippets = self.r.evidence_for("A", "halal chinese food")
        self.assertEqual(sorted(s.review_id for s in snippets), [0, 2])  # the two "halal" reviews of A
        self.assertTrue(all("halal" in s.text.lower() for s in snippets))

    def test_no_matching_review_gives_empty(self):
        self.assertEqual(self.r.evidence_for("B", "halal chinese food"), [])  # B never says halal

    def test_limit_unknown_restaurant_and_unknown_terms(self):
        self.assertEqual(len(self.r.evidence_for("A", "halal", n=1)), 1)
        self.assertEqual(self.r.evidence_for("Z", "halal"), [])
        self.assertEqual(self.r.evidence_for("A", "zzzz qqqq"), [])

    def test_info_uses_the_id_index(self):
        self.assertEqual(self.r.info("C")["name"], "C")


if __name__ == "__main__":
    unittest.main()
