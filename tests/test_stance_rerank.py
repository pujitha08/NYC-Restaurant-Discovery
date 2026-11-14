"""Stance re-ranking: scoring math, label cache, LLM labeler batching/validation and the reranker (fakes only)."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import eval.llm_judge as judge
from app import pipeline
from src.llm.groq_client import LLMAPIError, LLMResponseError
from src.retrieval import stance_rerank as sr
from src.schemas import RetrievalResult, Snippet
from tests.test_app_pipeline import FakeClient, FakeRetriever

MODEL = "openai/gpt-oss-20b"


def cand(base_norm, **counts):
    return sr.Candidate(RetrievalResult("G", base_norm, "x"), base_norm, {s: counts.get(s, 0) for s in sr.STANCES}, [])


class ScoringTests(unittest.TestCase):
    def test_normalize_scores(self):
        self.assertEqual(sr.normalize_scores([3.0, 2.0, 1.0]), [1.0, 0.5, 0.0])
        self.assertEqual(sr.normalize_scores([0.7, 0.7]), [1.0, 1.0])
        self.assertEqual(sr.normalize_scores([]), [])

    def test_shares_use_all_kept_sentences(self):
        c = cand(1.0, accommodation=1, negative=1, mention=1, not_about_allergy=1)
        self.assertEqual((c.n, c.share("accommodation"), c.share("negative")), (4, 0.25, 0.25))
        self.assertEqual(cand(1.0).share("negative"), 0.0)  # no sentences: no shares, no division by zero

    def test_score_formula(self):
        c = cand(0.8, accommodation=2, negative=1, mention=1)  # shares 0.5 and 0.25
        self.assertAlmostEqual(sr.stance_score(c, 0.3, 0.6), 0.8 + 0.3 * 0.5 - 0.6 * 0.25)

    def test_strong_negative_penalty_only_for_allergy_queries_with_a_negative(self):
        c = cand(0.5, negative=1, mention=3)
        self.assertAlmostEqual(sr.stance_score(c, 0, 0, 0.2, allergy=True), 0.3)
        self.assertAlmostEqual(sr.stance_score(c, 0, 0, 0.2, allergy=False), 0.5)
        self.assertAlmostEqual(sr.stance_score(cand(0.5, mention=1), 0, 0, 0.2, allergy=True), 0.5)

    def test_zero_weights_keep_the_base_order(self):
        cands = [cand(1.0, negative=2), cand(0.6, accommodation=2), cand(0.2)]
        self.assertEqual([c.base_norm for c in sr.rescore(cands, 0, 0)], [1.0, 0.6, 0.2])

    def test_warning_sinks_and_accommodation_rises(self):
        warned, good = cand(1.0, negative=3), cand(0.8, accommodation=3)
        self.assertIs(sr.rescore([warned, good], 0.3, 0.6)[0], good)  # 0.8 + 0.3 vs 1.0 - 0.6

    def test_ties_keep_base_order(self):
        a, b = cand(0.5), cand(0.5)
        self.assertEqual(sr.rescore([a, b], 0.3, 0.6), [a, b])

    def test_config_weights_start_at_the_agreed_values(self):
        from src.config import STANCE_RERANK
        self.assertEqual((STANCE_RERANK["w_pos"], STANCE_RERANK["w_neg"], STANCE_RERANK["top_n"]), (0.3, 0.6, 20))  # depth 20 is pre-registered


class CacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "sub" / "cache.jsonl"

    def test_put_get_and_persistence(self):
        cache = sr.LabelCache(self.path)
        self.assertIsNone(cache.get("m", "peanut allergy", "No peanuts here."))
        cache.put("m", "peanut allergy", "No peanuts here.", "accommodation")
        again = sr.LabelCache(self.path)  # a new run reads it back
        self.assertEqual(again.get("m", "peanut allergy", "No peanuts here."), "accommodation")

    def test_key_is_labeler_need_and_sentence(self):
        cache = sr.LabelCache(self.path)
        cache.put("m", "Peanut  Allergy", "A sentence.", "negative")
        self.assertEqual(cache.get("m", "peanut allergy", "A  sentence."), "negative")  # whitespace and case of the need
        self.assertIsNone(cache.get("other", "peanut allergy", "A sentence."))
        self.assertIsNone(cache.get("m", "celiac", "A sentence."))
        self.assertIsNone(cache.get("m", "peanut allergy", "Another sentence."))

    def test_file_holds_hashes_not_text(self):
        cache = sr.LabelCache(self.path)
        cache.put("m", "secret need", "secret sentence", "mention")
        self.assertNotIn("secret", self.path.read_text())

    def test_unchanged_label_is_not_rewritten_and_bad_stance_is_rejected(self):
        cache = sr.LabelCache(self.path)
        cache.put("m", "n", "s", "mention")
        cache.put("m", "n", "s", "mention")
        self.assertEqual(len(self.path.read_text().splitlines()), 1)
        with self.assertRaises(ValueError):
            cache.put("m", "n", "s", "great")

    def test_partial_last_line_is_ignored(self):
        cache = sr.LabelCache(self.path)
        cache.put("m", "n", "s", "mention")
        with open(self.path, "a") as f:
            f.write('{"labeler": "m", "need"')
        self.assertEqual(sr.LabelCache(self.path).get("m", "n", "s"), "mention")

    def test_memory_only_cache(self):
        cache = sr.LabelCache(None)
        cache.put("m", "n", "s", "negative")
        self.assertEqual(cache.get("m", "n", "s"), "negative")


class LabelFake:
    """Fake Groq client: labels a sentence 'negative' if it contains 'not', 'accommodation' if 'careful', else mention."""

    def __init__(self, broken_once=False):
        self.payloads, self.broken_once = [], broken_once
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        payload = json.loads(kw["messages"][1]["content"])
        self.payloads.append(payload)
        labels = [{"id": s["id"], "stance": "negative" if " not " in f" {s['text'].lower()} " else
                   "accommodation" if "careful" in s["text"].lower() else "mention"} for s in payload["sentences"]]
        if self.broken_once and len(self.payloads) == 1:
            labels = labels[:-1]
        msg = SimpleNamespace(content=json.dumps({"labels": labels}), refusal=None)
        use = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15, completion_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=use)


class LabelerTests(unittest.TestCase):
    def setUp(self):
        p = patch.object(judge, "sleep", lambda s: None)
        p.start()
        self.addCleanup(p.stop)

    def labeler(self, client, **kw):
        return sr.LLMLabeler(client=client, model=MODEL, rpm=0, tpm=0, **kw)

    def test_validate_labels(self):
        ok = {"labels": [{"id": "S2", "stance": "mention"}, {"id": "S1", "stance": "negative"}]}
        self.assertEqual(sr.validate_labels(ok, 2), ["negative", "mention"])  # sentence order, not answer order
        for bad in ({"labels": []}, {"labels": ok["labels"][:1]}, {"x": 1},
                    {"labels": [{"id": "S1", "stance": "mention"}, {"id": "S1", "stance": "mention"}]},
                    {"labels": [{"id": "S1", "stance": "great"}, {"id": "S2", "stance": "mention"}]},
                    {"labels": [{"id": "S1", "stance": "mention"}, {"id": "S3", "stance": "mention"}]}):
            with self.assertRaises(LLMResponseError):
                sr.validate_labels(bad, 2)

    def test_many_sentences_per_call_in_batches(self):
        client = LabelFake()
        sentences = [f"Sentence number {i} was fine." for i in range(7)]
        out = self.labeler(client, batch_size=3).label("peanut allergy", sentences)
        self.assertEqual(out, ["mention"] * 7)
        self.assertEqual([len(p["sentences"]) for p in client.payloads], [3, 3, 1])
        self.assertEqual(client.payloads[0]["need"], "peanut allergy")

    def test_invalid_answer_is_retried_once(self):
        client = LabelFake(broken_once=True)
        out = self.labeler(client).label("n", ["They were careful with allergies.", "This is not celiac safe."])
        self.assertEqual(out, ["accommodation", "negative"])
        self.assertEqual(len(client.payloads), 2)

    def test_non_retryable_error_propagates(self):
        class Down(LabelFake):
            def create(self, **kw):
                raise ValueError("boom")
        with self.assertRaises(LLMAPIError):
            self.labeler(Down()).label("n", ["A sentence here."])

    def test_client_is_created_lazily(self):
        made = []
        labeler = sr.LLMLabeler(model=MODEL, rpm=0, tpm=0, client_factory=lambda: made.append(1) or LabelFake())
        self.assertEqual(made, [])
        labeler.label("n", ["Just a sentence."])
        self.assertEqual(made, [1])


class FakeEmbedder:
    """Similarity to the query = number of query words in the sentence (as a 1-d vector)."""

    def encode(self, texts):
        return np.array([[float(len(set(t.lower().split()) & {"peanut", "allergy", "allergies"}))] for t in texts])


class FakeBase:
    embedder = FakeEmbedder()

    def __init__(self, results):
        self.results, self.calls = results, []

    def search(self, query, boroughs=None, top_k=10, include_low_evidence=False, density_mode=None):
        self.calls.append({"top_k": top_k, "density_mode": density_mode})
        return self.results[:top_k]

    def info(self, gmap_id):
        return {"name": gmap_id}


def res(gid, score, *texts):
    return RetrievalResult(gid, score, "semantic", [Snippet(t, i, None) for i, t in enumerate(texts, 1)])


class RerankerTests(unittest.TestCase):
    def setUp(self):
        p = patch.object(judge, "sleep", lambda s: None)
        p.start()
        self.addCleanup(p.stop)
        self.client = LabelFake()
        self.base = FakeBase([
            res("WARN", 1.0, "This place is not safe for a peanut allergy. The pasta was fine."),
            res("GOOD", 0.9, "They were careful about my peanut allergy. Great service overall."),
            res("MEH", 0.8, "Peanut noodles were tasty.")])
        self.cache = sr.LabelCache(None)

    def reranker(self, **kw):
        labeler = sr.LLMLabeler(client=self.client, model=MODEL, rpm=0, tpm=0)
        return sr.StanceReranker(self.base, labeler, self.cache, sentences_per_restaurant=2, **kw)

    def test_reranks_by_stance_and_keeps_base_evidence(self):
        out = self.reranker().search("peanut allergy", top_k=3)
        self.assertEqual([r.gmap_id for r in out], ["GOOD", "WARN", "MEH"])  # WARN: 1.0 - 0.6*0.5 - 0.1; GOOD: 0.5 + 0.3*0.5
        self.assertEqual(out[0].evidence_snippets, self.base.results[1].evidence_snippets)
        self.assertEqual(out[0].base_score, 0.9)
        self.assertEqual(out[1].stance_counts["negative"], 1)
        self.assertEqual(out[1].sentences[0]["stance"], "negative")

    def test_candidates_come_from_the_top_n_of_the_base(self):
        self.reranker(top_n=2).search("peanut allergy", top_k=1)
        self.assertEqual(self.base.calls, [{"top_k": 2, "density_mode": "capped"}])

    def test_keeps_the_sentences_most_similar_to_the_query(self):
        picks = self.reranker().pick_sentences("peanut allergy", self.base.results[0])
        self.assertEqual([t for t, _ in picks], ["This place is not safe for a peanut allergy.", "The pasta was fine."])
        one = sr.StanceReranker(self.base, None, sentences_per_restaurant=1).pick_sentences("peanut allergy", self.base.results[0])
        self.assertEqual([t for t, _ in one], ["This place is not safe for a peanut allergy."])

    def test_second_run_uses_the_cache_and_is_identical(self):
        r = self.reranker()
        first = [x.gmap_id for x in r.search("peanut allergy", top_k=3)]
        calls = len(self.client.payloads)
        again = [x.gmap_id for x in r.search("peanut allergy", top_k=3)]
        self.assertEqual((first, len(self.client.payloads)), (again, calls))  # no new LLM call
        self.assertGreater(calls, 0)

    def test_cache_is_per_need(self):
        r = self.reranker()
        r.search("peanut allergy", top_k=3)
        calls = len(self.client.payloads)
        r.search("nut allergy", top_k=3)
        self.assertGreater(len(self.client.payloads), calls)

    def test_weight_overrides_and_zero_weights_reproduce_the_base_order(self):
        r = self.reranker()
        self.assertEqual([x.gmap_id for x in r.search("peanut allergy", top_k=3, w_pos=0, w_neg=0, w_strong_neg=0)],
                         ["WARN", "GOOD", "MEH"])

    def test_blank_query_and_labeler_failure(self):
        self.assertEqual(self.reranker().search("  "), [])

        class Fail:
            id = "x"

            def label(self, need, sentences):
                raise LLMAPIError("down", category="rate_limit")
        with self.assertRaises(LLMAPIError):  # no silent fallback: evaluations must not mix rankings
            sr.StanceReranker(self.base, Fail(), self.cache).search("peanut allergy")

    def test_classifier_labeler_ignores_the_need(self):
        class Model:
            def predict(self, X):
                return ["negative" if x[0] > 0 else "mention" for x in X]
        lab = sr.ClassifierLabeler(Model(), FakeEmbedder())
        self.assertEqual(lab.label("anything", ["peanut one", "plain"]), ["negative", "mention"])
        self.assertEqual(lab.label("anything", []), [])


class AppFallbackTests(unittest.TestCase):
    def test_labeling_failure_falls_back_to_the_base_ranking(self):
        class Broken:
            def __init__(self):
                self.base = FakeRetriever()

            def search(self, *a, **kw):
                raise LLMAPIError("limit", category="rate_limit")

            def info(self, gid):
                return self.base.info(gid)
        out, _ = pipeline.run_search("vegan options for a group", "Both", pipeline.STANCE_METHOD, retriever=Broken(),
                                     client=pipeline.UsageRecorder(FakeClient()), model=MODEL, log_path=None)
        self.assertEqual(len(out.results), 4)
        self.assertTrue(any("Showing the semantic_capped ranking" in n for n in out.notices))


if __name__ == "__main__":
    unittest.main()
