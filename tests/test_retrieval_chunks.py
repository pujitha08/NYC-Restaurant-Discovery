"""Chunk-retriever evidence ownership and query_terms caching; synthetic in-memory data only."""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.retrieval import evidence
from src.retrieval.evidence import query_terms, requirement_terms
from src.retrieval.tfidf_chunks import TOP_CHUNKS, ChunkTfidfRetriever


def make_retriever(texts, owner, n_restaurants):
    """A ChunkTfidfRetriever over in-memory chunks; chunk/review files are never opened."""
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts).tocsr()
    meta = [{"gmap_id": f"R{k}", "borough": "Brooklyn", "low_evidence": False} for k in range(n_restaurants)]
    return ChunkTfidfRetriever(vectorizer, matrix, np.zeros(len(texts), dtype=np.int64),
                               np.array(owner, dtype=np.int32), meta, chunks_path="/nonexistent",
                               reviews_path="/nonexistent")


def evidence_chunks(retriever, query, **kwargs):
    """Run search() and return {gmap_id: chunk ids passed to _evidence} without reading files."""
    captured = []
    with patch.object(retriever, "_evidence",
                      side_effect=lambda chunk_ids, q, requirements=None: captured.append(list(chunk_ids)) or []):
        results = retriever.search(query, **kwargs)
    return {r.gmap_id: ids for r, ids in zip(results, captured)}


class ChunkEvidenceOwnershipTests(unittest.TestCase):
    def test_restaurant_with_one_match_gets_only_its_own_chunk(self):
        # R0 has one matching chunk, immediately followed by R1 with three.
        texts = ["vegan tacos", "burger and fries", "vegan bowl", "vegan salad", "vegan cake"]
        r = make_retriever(texts, owner=[0, 0, 1, 1, 1], n_restaurants=2)
        chunks = evidence_chunks(r, "vegan", top_k=2)
        self.assertEqual(set(chunks), {"R0", "R1"})
        self.assertEqual(chunks["R0"], [0])
        self.assertEqual(sorted(chunks["R1"]), [2, 3, 4])

    def test_every_result_receives_only_chunks_it_owns(self):
        # Restaurants with 1, 2, 4 and 1 matching chunks, interleaved with non-matching ones.
        texts = ["vegan a", "pizza", "vegan b", "vegan c", "steak", "vegan d", "vegan e",
                 "vegan f", "vegan g", "fries", "vegan h"]
        owner = [0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3]
        r = make_retriever(texts, owner, n_restaurants=4)
        chunks = evidence_chunks(r, "vegan", top_k=4)
        self.assertEqual(len(chunks), 4)
        for gmap_id, ids in chunks.items():
            with self.subTest(gmap_id=gmap_id):
                self.assertTrue(ids)
                self.assertLessEqual(len(ids), TOP_CHUNKS)
                self.assertEqual({f"R{owner[c]}" for c in ids}, {gmap_id})

    def test_evidence_is_the_restaurants_best_chunks_in_score_order(self):
        texts = ["vegan", "vegan vegan options", "vegan options here", "vegan and meat", "fries only"]
        r = make_retriever(texts, owner=[0, 0, 0, 0, 0], n_restaurants=1)
        q = r.vectorizer.transform(["vegan options"])
        chunk_scores = (r.matrix @ q.T).toarray().ravel()
        expected = sorted(np.flatnonzero(chunk_scores > 0), key=lambda c: -chunk_scores[c])[:TOP_CHUNKS]
        self.assertEqual(evidence_chunks(r, "vegan options", top_k=1)["R0"], [int(c) for c in expected])

    def test_borough_filter_and_ranking_unaffected_by_bound(self):
        texts = ["vegan tacos", "vegan bowl", "vegan salad", "vegan cake"]
        r = make_retriever(texts, owner=[0, 1, 1, 1], n_restaurants=2)
        r.meta[0]["borough"] = "Manhattan"
        r._boroughs = np.array([m["borough"].lower() for m in r.meta])
        self.assertEqual(list(evidence_chunks(r, "vegan", boroughs=["Brooklyn"], top_k=5)), ["R1"])


class QueryTermsCacheTests(unittest.TestCase):
    def setUp(self):
        getattr(evidence, "_FEATURE_NAMES", {}).clear()  # start each test with an empty cache
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.vectorizer.fit(["vegan options for a group", "nut free bakery", "celiac safe kitchen",
                             "peanut allergy safe", "vegan bakery in brooklyn"])

    @staticmethod
    def uncached_query_terms(vectorizer, q):
        """The pre-cache implementation, kept here as the reference."""
        names = vectorizer.get_feature_names_out()
        return {names[j]: float(vectorizer.idf_[j]) for j in q.indices if " " not in names[j]}

    def test_identical_output_to_uncached_implementation(self):
        for text in ["vegan options for a group", "nut free bakery", "celiac safe", "unknown words only", ""]:
            q = self.vectorizer.transform([text])
            with self.subTest(text=text):
                self.assertEqual(query_terms(self.vectorizer, q), self.uncached_query_terms(self.vectorizer, q))
                self.assertEqual(query_terms(self.vectorizer, q), self.uncached_query_terms(self.vectorizer, q))

    def test_requirement_terms_unchanged(self):
        q = self.vectorizer.transform(["nut free bakery"])
        terms = self.uncached_query_terms(self.vectorizer, q)
        top = max(terms.values())
        expected = [t for t, idf in terms.items() if idf >= top - evidence.REQUIREMENT_MARGIN]
        self.assertEqual(requirement_terms(self.vectorizer, q), expected)

    def test_feature_names_computed_once_per_vectorizer(self):
        q = self.vectorizer.transform(["vegan bakery"])
        with patch.object(self.vectorizer, "get_feature_names_out",
                          wraps=self.vectorizer.get_feature_names_out) as spy:
            for _ in range(11):
                query_terms(self.vectorizer, q)
        self.assertEqual(spy.call_count, 1)

    def test_cache_refreshes_after_refit(self):
        q = self.vectorizer.transform(["vegan"])
        query_terms(self.vectorizer, q)
        self.vectorizer.fit(["shellfish allergy", "sesame free", "vegan"])
        q = self.vectorizer.transform(["shellfish vegan"])
        self.assertEqual(query_terms(self.vectorizer, q), self.uncached_query_terms(self.vectorizer, q))

    def test_separate_vectorizers_do_not_share_names(self):
        other = TfidfVectorizer().fit(["zebra yak", "xylophone"])
        q1, q2 = self.vectorizer.transform(["vegan"]), other.transform(["zebra"])
        self.assertEqual(query_terms(self.vectorizer, q1), self.uncached_query_terms(self.vectorizer, q1))
        self.assertEqual(query_terms(other, q2), self.uncached_query_terms(other, q2))


if __name__ == "__main__":
    unittest.main()
