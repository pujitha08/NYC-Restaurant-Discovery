"""Compare mode: rank-change markers, fallbacks and the demo-mode (no model call) path (fakes only)."""

import tempfile
import unittest
from pathlib import Path

from app import compare, demo_cache, pipeline
from src.llm.groq_client import LLMAPIError
from src.retrieval.stance_rerank import LabelsUnavailable, StanceResult
from tests.test_app_pipeline import MODEL, FakeClient, FakeRetriever


def view(gid, rank):
    return pipeline.ResultView(rank, gid, gid, "Brooklyn", "", 4.0, 10, False, 1.0, [])


class FakeStance:
    """Stance re-ranker stand-in: the FakeRetriever's results reversed, with stance labels on each."""

    def __init__(self, fail=None):
        self.base, self.fail, self.calls = FakeRetriever(), fail, []

    def search(self, query, boroughs=None, top_k=10, include_low_evidence=False, density_mode=None, cache_only=False):
        self.calls.append({"cache_only": cache_only})
        if self.fail:
            raise self.fail
        out = self.base.search(query, boroughs, 30, include_low_evidence, density_mode)[::-1][:top_k]
        return [StanceResult(r.gmap_id, 1.0 - i / 10, "semantic_stance", r.evidence_snippets, r.score,
                             {"accommodation": 2, "mention": 1, "negative": 1, "not_about_allergy": 0},
                             [{"text": "Staff were not careful.", "stance": "negative", "review_id": 1}])
                for i, r in enumerate(out)]

    def info(self, gid):
        return self.base.info(gid)


class RankChangeTests(unittest.TestCase):
    def test_up_down_same_new_and_dropped(self):
        base = [view("A", 1), view("B", 2), view("C", 3), view("D", 4)]
        stance = [view("B", 1), view("A", 2), view("C", 3), view("E", 4)]
        changes, dropped = compare.rank_changes(base, stance)
        self.assertEqual(changes, {"B": (compare.UP, 1), "A": (compare.DOWN, 1), "C": (compare.SAME, 0),
                                   "E": (compare.NEW, 0)})
        self.assertEqual(dropped, ["D"])

    def test_multi_place_moves(self):
        changes, _ = compare.rank_changes([view("A", 1), view("B", 2), view("C", 3)], [view("C", 1), view("A", 2), view("B", 3)])
        self.assertEqual(changes["C"], (compare.UP, 2))


class StanceBarTests(unittest.TestCase):
    def labeled(self, counts, sentences=()):
        v = view("A", 1)
        v.stance_counts, v.stance_sentences = counts, list(sentences)
        return v

    def test_counts_text_bar_and_first_warning(self):
        v = self.labeled({"accommodation": 7, "mention": 4, "negative": 1, "not_about_allergy": 2},
                         [{"text": "Great.", "stance": "accommodation", "review_id": 1},
                          {"text": "Got sick here.", "stance": "negative", "review_id": 2},
                          {"text": "Also bad.", "stance": "negative", "review_id": 3}])
        bar = pipeline.stance_bar(v)
        self.assertEqual(bar["text"], "7 accommodate / 1 warns / 4 mention")
        self.assertEqual(bar["warning"], "Got sick here.")
        self.assertEqual(bar["bar"], "🟩" * 7 + "🟥" + "🟨" * 4 + "⬜" * 2)

    def test_no_warning_sentence_when_nothing_is_negative(self):
        bar = pipeline.stance_bar(self.labeled({"accommodation": 1, "negative": 0}, [{"text": "Fine.", "stance": "accommodation", "review_id": 1}]))
        self.assertIsNone(bar["warning"])

    def test_nothing_without_labels(self):
        self.assertIsNone(pipeline.stance_bar(view("A", 1)))
        self.assertIsNone(pipeline.stance_bar(self.labeled({})))
        self.assertIsNone(pipeline.stance_bar(self.labeled({"mention": 0, "negative": 0})))

    def test_cached_outcome_round_trip_keeps_stance_fields(self):
        out = pipeline.SearchOutcome("r", "q", "Both", "semantic_capped_stance", None, pipeline.ParsedQuery(), None,
                                     [self.labeled({"negative": 1}, [{"text": "No.", "stance": "negative", "review_id": 1}])])
        back = demo_cache.outcome_from_dict(demo_cache.outcome_to_dict(out))
        self.assertEqual(back.results[0].stance_sentences[0]["text"], "No.")


class RunCompareTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.client = pipeline.UsageRecorder(FakeClient(location=None))

    def run_compare(self, stance, **kw):
        return compare.run_compare("vegan options for a group", "Both", base_retriever=FakeRetriever(),
                                   stance_retriever=stance, client=self.client, model=MODEL, log_path=None,
                                   cache_dir=self.dir, **kw)

    def test_both_columns_with_changes_and_one_parse(self):
        c = self.run_compare(FakeStance())
        self.assertEqual([v.gmap_id for v in c.base.results], ["G1", "G2", "G3", "G4"])
        self.assertEqual([v.gmap_id for v in c.stance.results], ["G4", "G3", "G2", "G1"])
        self.assertEqual(c.changes["G4"], (compare.UP, 3))
        self.assertEqual(c.changes["G1"], (compare.DOWN, 3))
        self.assertEqual(self.client._client.calls, ["parsed_query"])  # parsed once, no summaries
        self.assertTrue(all(v.summary is None for v in c.base.results + c.stance.results))
        self.assertEqual(c.stance.results[0].stance_counts["negative"], 1)
        self.assertEqual(c.stance.results[0].stance_sentences[0]["stance"], "negative")

    def test_rate_limited_labeler_shows_the_base_ranking_alone(self):
        c = self.run_compare(FakeStance(fail=LLMAPIError("limit", category="rate_limit")))
        self.assertIsNone(c.stance)
        self.assertEqual(len(c.base.results), 4)
        self.assertTrue(any("stance ranking is unavailable" in n for n in c.notices))
        self.assertTrue(any("rate limit" in n for n in c.notices))

    def test_missing_stance_retriever(self):
        c = self.run_compare(None)
        self.assertIsNone(c.stance)
        self.assertTrue(any("not available" in n for n in c.notices))

    def test_demo_mode_reuses_the_cached_base_and_never_calls_the_labeler_live(self):
        out, _ = pipeline.run_search("vegan options for a group", "Both", compare.BASE_METHOD, retriever=FakeRetriever(),
                                     client=pipeline.UsageRecorder(FakeClient(location=None)), model=MODEL, log_path=None)
        demo_cache.save_outcome(out, self.dir)
        stance = FakeStance()
        c = self.run_compare(stance, demo_mode=True)
        self.assertTrue(c.base.cached)
        self.assertEqual(self.client._client.calls, [])  # no model call at all
        self.assertEqual(stance.calls, [{"cache_only": True}])
        self.assertIsNotNone(c.stance)

    def test_demo_mode_without_cached_labels_falls_back(self):
        c = self.run_compare(FakeStance(fail=LabelsUnavailable("not cached")), demo_mode=True)
        self.assertIsNone(c.stance)
        self.assertTrue(any("not cached" in n for n in c.notices))


if __name__ == "__main__":
    unittest.main()
