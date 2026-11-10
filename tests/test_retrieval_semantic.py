"""Semantic retriever tests: tiny synthetic chunk files and a fake embedder (no model, data or network)."""

import json
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.retrieval import semantic
from src.retrieval.semantic import (MATCH_THRESHOLD, CacheMismatchError, SemanticRetriever, excerpt_around)
from src.retrieval.tfidf_chunks import restaurant_chunks

DIM = 64
RESERVED = {w: i for i, w in enumerate(["vegan", "meat", "pizza", "celiac", "shellfish", "fish", "group"])}


class FakeEmbedder:
    """Deterministic bag-of-words embedding: reserved words get their own dimension, other words
    hash into the remaining ones. Rows are L2-normalized like the real model's."""

    model_name = semantic.MODEL_NAME

    def __init__(self):
        self.calls = 0

    def encode(self, texts, batch_size=None):
        self.calls += 1
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace(".", " ").replace(",", " ").split():
                j = RESERVED.get(w, len(RESERVED) + zlib.crc32(w.encode()) % (DIM - len(RESERVED)))
                out[i, j] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


# (gmap_id, borough, low_evidence, [chunks, each a list of review texts])
RESTAURANTS = [
    ("R0", "Brooklyn", False, [["vegan tacos"], ["burger and fries"]]),  # exactly one matching chunk
    ("R1", "Brooklyn", False, [["vegan bowl"], ["vegan salad"], ["vegan vegan cake"], ["vegan soup"]]),
    ("R2", "Manhattan", False, [["vegan pizza"], ["vegan pasta"]]),
    ("R3", "Brooklyn", True, [["vegan vegan vegan"]]),  # low_evidence
    ("R4", "Brooklyn", False, [["steak frites"], ["oysters on ice"]]),  # no match
    ("R5", "Brooklyn", False, [["vegan donuts"]]),  # R5 and R6 are exact ties
    ("R6", "Brooklyn", False, [["vegan donuts"]]),
]


def write_fixture(root):
    """Write restaurant_docs, review_chunks and reviews_clean files; return paths and review owners."""
    root = Path(root)
    docs, chunks, reviews = root / "restaurant_docs.jsonl", root / "review_chunks.jsonl", root / "reviews_clean.jsonl"
    owner_of, rid = {}, 0
    with open(docs, "w") as fd, open(chunks, "w") as fc, open(reviews, "w") as fr:
        for gid, borough, low, chunk_list in RESTAURANTS:
            texts = []
            for k, review_texts in enumerate(chunk_list):
                ids = []
                for text in review_texts:
                    fr.write(json.dumps({"review_id": rid, "gmap_id": gid, "user_id": "u", "rating": 5,
                                         "time": 1_600_000_000_000 + rid, "text": text}) + "\n")
                    owner_of[rid] = gid
                    ids.append(rid)
                    texts.append(text)
                    rid += 1
                fc.write(json.dumps({"chunk_id": f"{gid}:{k}", "gmap_id": gid, "text": " ".join(review_texts),
                                     "review_ids": ids, "times": [None] * len(ids)}) + "\n")
            fd.write(json.dumps({"gmap_id": gid, "name": f"Name {gid}", "borough": borough, "address": "addr",
                                 "categories": [], "avg_rating": 4.0, "num_clean_reviews": len(texts),
                                 "low_evidence": low, "text": "\n".join(texts)}) + "\n")
    return {"docs_path": docs, "chunks_path": chunks, "reviews_path": reviews,
            "review_index_path": root / "reviews_index.npz"}, owner_of


class SemanticFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths, self.owner_of = write_fixture(self.tmp.name)
        self.embedder = FakeEmbedder()
        self.r = SemanticRetriever.build(embedder=self.embedder, progress=False, **self.paths)

    def ids(self, results):
        return [x.gmap_id for x in results]


class SearchTests(SemanticFixture):
    def test_results_are_embedding_results_in_score_order(self):
        results = self.r.search("vegan", top_k=10)
        self.assertTrue(results)
        self.assertTrue(all(x.method == "embedding" for x in results))
        scores = [x.score for x in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(self.r.info("R1")["name"], "Name R1")
        self.assertNotIn("R4", self.ids(results))  # no chunk above MATCH_THRESHOLD

    def test_borough_filter(self):
        brooklyn = self.ids(self.r.search("vegan", boroughs=["Brooklyn"], top_k=10))
        self.assertNotIn("R2", brooklyn)
        self.assertIn("R1", brooklyn)
        self.assertEqual(self.ids(self.r.search("vegan", boroughs=["manhattan"], top_k=10)), ["R2"])
        self.assertEqual(self.r.search("vegan", boroughs=["Queens"], top_k=10), [])

    def test_low_evidence_excluded_by_default(self):
        self.assertNotIn("R3", self.ids(self.r.search("vegan", top_k=10)))
        self.assertIn("R3", self.ids(self.r.search("vegan", top_k=10, include_low_evidence=True)))

    def test_evidence_comes_only_from_the_restaurants_own_reviews(self):
        results = self.r.search("vegan", top_k=10, include_low_evidence=True)
        for x in results:
            with self.subTest(gmap_id=x.gmap_id):
                self.assertTrue(x.evidence_snippets)
                self.assertEqual({self.owner_of[s.review_id] for s in x.evidence_snippets}, {x.gmap_id})

    def test_restaurant_with_one_matching_chunk_gets_only_that_chunk(self):
        # R0 has one hit chunk and sits right before R1, which has four.
        chunk_scores = self.r.embeddings @ self.embedder.encode(["vegan"])[0]
        _, hit, owners = semantic.aggregate_chunks(chunk_scores, self.r.owner, len(self.r.meta), "log",
                                                   MATCH_THRESHOLD, MATCH_THRESHOLD)
        self.assertEqual([self.r.owner[c] for c in restaurant_chunks(hit, owners, 0)], [0])
        r0 = next(x for x in self.r.search("vegan", top_k=10) if x.gmap_id == "R0")
        self.assertEqual([s.text for s in r0.evidence_snippets], ["vegan tacos"])

    def test_evidence_chunks_are_bounded_to_the_restaurant(self):
        # Checks the slice itself (independently of the review-level gmap_id filter below).
        captured = {}
        original = self.r._evidence

        def spy(rows, chunk_ids, qvec, **kwargs):
            captured.update({int(row): [int(c) for c in ids] for row, ids in zip(rows, chunk_ids)})
            return original(rows, chunk_ids, qvec, **kwargs)

        with patch.object(self.r, "_evidence", side_effect=spy):
            self.r.search("vegan", top_k=10, include_low_evidence=True)
        self.assertEqual(captured[0], [0])  # R0: its one matching chunk, nothing from R1
        for row, ids in captured.items():
            with self.subTest(row=row):
                self.assertTrue(ids)
                self.assertTrue(all(self.r.owner[c] == row for c in ids))

    def test_foreign_review_ids_in_a_chunk_are_dropped(self):
        # Defense in depth: even a chunk record that lists another restaurant's review is filtered.
        foreign = next(rid for rid, gid in self.owner_of.items() if gid == "R1")
        lines = self.paths["chunks_path"].read_text().splitlines()
        first = json.loads(lines[0])  # R0's matching chunk
        first["review_ids"].append(foreign)
        self.r.offsets = self.r.offsets.copy()
        with open(self.paths["chunks_path"], "a") as f:
            self.r.offsets[0] = f.tell()
            f.write(json.dumps(first) + "\n")
        r0 = next(x for x in self.r.search("vegan", top_k=10) if x.gmap_id == "R0")
        self.assertNotIn(foreign, [s.review_id for s in r0.evidence_snippets])

    def test_stable_ordering_for_ties_and_repeated_searches(self):
        first = self.ids(self.r.search("vegan", top_k=10))
        self.assertLess(first.index("R5"), first.index("R6"))  # tie broken by restaurant row
        for _ in range(3):
            self.assertEqual(self.ids(self.r.search("vegan", top_k=10)), first)

    def test_top_k_and_empty_query(self):
        self.assertEqual(len(self.r.search("vegan", top_k=2)), 2)
        self.assertEqual(self.r.search("   "), [])
        self.assertEqual(self.r.search("zzz unknownword"), [])

    def test_density_modes(self):
        log = self.r.restaurant_scores("vegan", density_mode="log")
        none = self.r.restaurant_scores("vegan", density_mode="none")
        capped = self.r.restaurant_scores("vegan", density_mode="capped")
        r1 = 1
        self.assertGreater(log[r1], none[r1])  # R1 has 4 matching chunks
        self.assertLessEqual(capped[r1] - none[r1], 0.2 + 1e-9)
        with self.assertRaises(ValueError):
            self.r.search("vegan", density_mode="bogus")


class SnippetTests(SemanticFixture):
    def test_long_review_is_cut_around_its_best_sentence(self):
        sentences = ["Filler words here."] * 30 + ["The vegan menu is great."] + ["More filler."] * 30
        text = excerpt_around(sentences, 30, max_words=20)
        self.assertIn("The vegan menu is great.", text)
        self.assertLessEqual(len(text.replace("…", "").split()), 20)
        self.assertTrue(text.startswith("…") and text.endswith("…"))

    def test_snippet_fields(self):
        r1 = next(x for x in self.r.search("vegan", top_k=10) if x.gmap_id == "R1")
        self.assertLessEqual(len(r1.evidence_snippets), 3)
        for s in r1.evidence_snippets:
            self.assertIsInstance(s.review_id, int)
            self.assertIsInstance(s.time, int)
            self.assertIn("vegan", s.text)


class CacheTests(SemanticFixture):
    def save_and_load(self, **overrides):
        index_dir = Path(self.tmp.name) / "semantic"
        self.r.save(index_dir, docs_path=self.paths["docs_path"])
        kwargs = {k: v for k, v in self.paths.items()}
        kwargs.update(overrides)
        return index_dir, kwargs

    def test_round_trip_gives_identical_results(self):
        index_dir, kwargs = self.save_and_load()
        info = json.loads((index_dir / semantic.INFO_FILE).read_text())
        self.assertEqual(info["chunk_count"], sum(len(chunks) for *_, chunks in RESTAURANTS))
        self.assertEqual(info["model"], semantic.MODEL_NAME)
        self.assertEqual(np.load(index_dir / semantic.EMBEDDINGS_FILE).dtype, np.float16)
        loaded = SemanticRetriever.load(index_dir, embedder=FakeEmbedder(), **kwargs)
        a, b = self.r.search("vegan", top_k=10), loaded.search("vegan", top_k=10)
        self.assertEqual([(x.gmap_id, x.score, x.evidence_snippets) for x in a],
                         [(x.gmap_id, x.score, x.evidence_snippets) for x in b])

    def test_refuses_cache_when_chunks_file_changed(self):
        index_dir, kwargs = self.save_and_load()
        chunks = self.paths["chunks_path"]
        chunks.write_text(chunks.read_text().replace("vegan tacos", "vegan tacoz"))  # same size, new hash
        with self.assertRaisesRegex(CacheMismatchError, "review_chunks.jsonl changed"):
            SemanticRetriever.load(index_dir, embedder=FakeEmbedder(), **kwargs)

    def test_refuses_cache_when_docs_file_changed(self):
        index_dir, kwargs = self.save_and_load()
        with open(self.paths["docs_path"], "a") as f:
            f.write("\n")
        with self.assertRaisesRegex(CacheMismatchError, "restaurant_docs.jsonl changed"):
            SemanticRetriever.load(index_dir, embedder=FakeEmbedder(), **kwargs)

    def test_refuses_cache_built_with_another_model(self):
        index_dir, kwargs = self.save_and_load()
        other = FakeEmbedder()
        other.model_name = "some/other-model"
        with self.assertRaisesRegex(CacheMismatchError, "model"):
            SemanticRetriever.load(index_dir, embedder=other, **kwargs)

    def test_refuses_inconsistent_arrays(self):
        index_dir, kwargs = self.save_and_load()
        np.save(index_dir / semantic.OWNER_FILE, self.r.owner[:-1])
        with self.assertRaisesRegex(CacheMismatchError, "inconsistent"):
            SemanticRetriever.load(index_dir, embedder=FakeEmbedder(), **kwargs)

    def test_exists(self):
        self.assertFalse(SemanticRetriever.exists(Path(self.tmp.name) / "missing"))
        index_dir, _ = self.save_and_load()
        self.assertTrue(SemanticRetriever.exists(index_dir))


class MakePoolWiringTests(unittest.TestCase):
    def test_load_semantic_skips_unbuilt_index(self):
        from eval import make_pool
        with patch.object(SemanticRetriever, "exists", return_value=False):
            self.assertIsNone(make_pool.load_semantic())
        self.assertIs(make_pool.SYSTEMS["semantic"], make_pool.load_semantic)


if __name__ == "__main__":
    unittest.main()
