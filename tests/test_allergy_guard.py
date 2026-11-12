"""Allergy guard and reasoning effort in the app pipeline (regression: Q20 / Mission, see
reports/failure_examples.md case 9). Fake retriever and SDK client; no data or network."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import pipeline
from src.llm.evidence_summarizer import summarize_evidence
from src.llm.query_parser import parse_query
from src.schemas import ParsedQuery, RetrievalResult, Snippet

MODEL = "openai/gpt-oss-20b"
Q20 = "restaurant careful about peanut allergies"
MISSION_ID = "0x89c2598152ae11f1:0x778df2824b1172c1"
MISSION_SNIPPETS = [
    Snippet("If you or anyone in your party suffers from food allergies, please avoid this restaurant. Unfortunately, "
            "the waitstaff and kitchen are not attentive to food allergies. We informed the waitstaff when ordering "
            "concerning one diner in our party with a tree nut and sesame allergy. …", 1576831, 1503000000000),
    Snippet("If you have a food allergy, you should seriously consider going elsewhere. Our party of four had one nut "
            "and one wheat allergy--both of which were explicitly shared with our waitress when our order was placed. "
            "She acknowledged, and offered no concerns regarding our menu choices in light of this information. "
            "We ordered six plates to share. …", 1576786, 1500000000000),
]
# The summarizer's live answer for Mission on 2026-11-10 (labels as returned): two warnings labeled "support".
MISSION_SUMMARY = {"status": "evidence_found", "claims": [
    {"text": "One reviewer reported that the restaurant's staff were inattentive to food allergies, advising those "
             "with allergies to avoid the venue.", "review_ids": [1576831], "label": "support"},
    {"text": "Another reviewer warned that the restaurant is not suitable for diners with food allergies, noting a "
             "lack of concern from staff despite being informed of nut and wheat allergies.",
     "review_ids": [1576786], "label": "support"},
    {"text": "Both reviewers mentioned nut allergies but did not specifically reference peanuts, leaving uncertainty "
             "about the restaurant's safety for peanut‑allergic patrons.", "review_ids": [1576831, 1576786],
     "label": "uncertainty"}],
    "limitations": ["The excerpts do not mention peanuts specifically; conclusions are based on general allergy warnings."]}
SUPPORT_ONLY = {"status": "evidence_found", "claims": [
    {"text": "One reviewer reported vegan options.", "review_ids": [1576831], "label": "support"}], "limitations": []}
# Mission as labeled by the summarizer with the user's need (fix/summarizer-user-need live run): two contradictions.
MISSION_CONTRADICTIONS = {"status": "evidence_found", "claims": [
    {"text": "One reviewer reported that the restaurant staff were inattentive to food allergies, advising people "
             "with allergies to avoid the venue.", "review_ids": [1576831], "label": "contradiction"},
    {"text": "Another reviewer described a similar experience, noting that the waitress offered no reassurance after "
             "being told about a nut allergy and recommending the restaurant be avoided.",
     "review_ids": [1576786], "label": "contradiction"}], "limitations": []}
MIXED = {"status": "evidence_found", "claims": [
    {"text": "One reviewer reported accommodation.", "review_ids": [1576831], "label": "support"},
    {"text": "Another reviewer reported a reaction.", "review_ids": [1576786], "label": "contradiction"}], "limitations": []}


class MissionRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, boroughs=None, top_k=10, include_low_evidence=False, density_mode=None):
        self.calls.append(query)
        return [RetrievalResult(MISSION_ID, 2.13, "embedding", list(MISSION_SNIPPETS))]

    def info(self, gmap_id):
        return {"name": "Mission", "borough": "Manhattan", "address": "", "avg_rating": 4.0,
                "num_clean_reviews": 396, "low_evidence": False}


class Client:
    """parsed_query -> `parsed` (or invalid JSON if None); evidence_summary -> `summary`. Records kwargs."""

    def __init__(self, parsed, summary):
        self.parsed, self.summary, self.calls = parsed, summary, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        name = kwargs["response_format"]["json_schema"]["name"]
        body = (self.parsed if name == "parsed_query" else self.summary)
        content = "not JSON" if body is None else json.dumps(body)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=None),
                                                        finish_reason="stop")], usage=None)


def parsed(**kw):
    base = {"location": None, "dietary_restrictions": [], "allergens": [], "group_needs": [], "preferences": []}
    return {**base, **kw}


class SearchCase(unittest.TestCase):
    """Shared setup: run one search over the Mission fixture with scripted parser/summarizer answers."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log = Path(tmp.name) / "log.jsonl"

    def search(self, query, parsed_body, summary_body):
        self.client = Client(parsed_body, summary_body)
        out, _ = pipeline.run_search(query, "Both", "semantic_capped", retriever=MissionRetriever(),
                                     client=pipeline.UsageRecorder(self.client), model=MODEL, log_path=self.log)
        return out


class GuardTests(SearchCase):
    def test_mission_regression_is_never_supported(self):
        out = self.search(Q20, parsed(allergens=["peanut"]), MISSION_SUMMARY)  # parsed exactly as in the live run
        self.assertTrue(out.allergy_sensitive)
        s = out.results[0].summary
        self.assertNotEqual(s.status, pipeline.STATUS_SUPPORTED)
        self.assertEqual(s.status, pipeline.STATUS_NEUTRAL)
        self.assertEqual(s.status, "Reviews discuss this — read each claim")
        self.assertTrue(s.guarded)
        self.assertEqual(s.note, pipeline.ALLERGY_NOTE)
        self.assertIn("can't be confirmed from reviews", s.note)
        self.assertEqual([c.label for c in s.claims], ["support", "support", "uncertainty"])  # labels kept for the log
        self.assertEqual([e.review_id for e in s.claims[0].excerpts], [1576831])
        rec = json.loads(self.log.read_text().splitlines()[0])
        self.assertEqual((rec["allergy_sensitive"], rec["summaries"][0]["status"], rec["summaries"][0]["guarded"]),
                         (True, pipeline.STATUS_NEUTRAL, True))

    def test_guard_triggers(self):
        cases = {
            "parsed allergens": ("vegan brunch", parsed(allergens=["sesame"])),
            "raw text: allergy": ("place that handles my kid's allergy well", parsed(dietary_restrictions=["vegan"])),
            "raw text: celiac": ("safe for celiac disease", parsed(dietary_restrictions=["gluten-free"])),
            "raw text: cross contact": ("no cross-contact please", parsed()),
            "raw text: nut-free": ("nut-free bakery", parsed(preferences=["bakery"])),
            # Allergen names without allergy wording, with a parser that missed the allergen field
            "allergen name: no peanuts please": ("no peanuts please", parsed(preferences=["no peanuts"])),
            "allergen name: sesame-free": ("sesame-free dinner in Brooklyn", parsed(location="Brooklyn")),
            "allergen name: shrimp": ("somewhere without shrimp in the kitchen", parsed()),
            "allergen name: tahini": ("falafel but no tahini", parsed(preferences=["falafel"])),
            "allergen name: dairy": ("dairy-free brunch", parsed(dietary_restrictions=["dairy-free"])),
            "over-cautious: peanut noodles": ("peanut noodles", parsed(preferences=["peanut noodles"])),
            "parse failed": ("vegan brunch", None),
        }
        for name, (query, body) in cases.items():
            with self.subTest(name):
                out = self.search(query, body, SUPPORT_ONLY)
                self.assertTrue(out.allergy_sensitive)
                self.assertEqual(out.results[0].summary.status, pipeline.STATUS_NEUTRAL)

    def test_non_allergy_queries_stay_unguarded(self):
        for query in ("vegan options for a group", "halal chinese food in Manhattan", "gluten-free pizza",
                      "kosher bakery", "best donut shop", "nutritious bowls", "eggplant parm", "pescatarian seafood"):
            with self.subTest(query=query):
                self.assertFalse(pipeline.is_allergy_sensitive(query, ParsedQuery(dietary_restrictions=["vegan"])))

    def test_allergen_names_trigger_without_allergy_wording(self):
        for query in ("no peanuts please", "sesame-free", "almond croissant", "cashew chicken", "walnut salad",
                      "lobster roll", "crab cakes", "soy sauce free", "egg sandwich", "fish tacos", "milk tea",
                      "whole wheat bagels", "tree nut free cookies"):
            with self.subTest(query=query):
                self.assertTrue(pipeline.is_allergy_sensitive(query, ParsedQuery()))

    def test_non_allergy_query_keeps_supported(self):
        out = self.search("vegan options for a group", parsed(dietary_restrictions=["vegan"]), SUPPORT_ONLY)
        self.assertFalse(out.allergy_sensitive)
        s = out.results[0].summary
        self.assertEqual((s.status, s.guarded, s.note), (pipeline.STATUS_SUPPORTED, False, None))

    def test_mixed_and_insufficient_unchanged_under_guard(self):
        out = self.search(Q20, parsed(allergens=["peanut"]), MIXED)
        self.assertEqual(out.results[0].summary.status, pipeline.STATUS_MIXED)
        self.assertEqual(out.results[0].summary.note, pipeline.ALLERGY_NOTE)
        out = self.search(Q20, parsed(allergens=["peanut"]), {"status": "insufficient_evidence", "claims": [],
                                                              "limitations": ["Nothing about peanuts."]})
        self.assertEqual(out.results[0].summary.status, pipeline.STATUS_INSUFFICIENT)

    def test_status_function(self):
        s = SimpleNamespace(status="evidence_found", claims=[SimpleNamespace(label="support")])
        self.assertEqual(pipeline.display_status(s), pipeline.STATUS_SUPPORTED)
        self.assertEqual(pipeline.display_status(s, guarded=True), pipeline.STATUS_NEUTRAL)
        self.assertFalse(pipeline.is_allergy_sensitive("vegan food", ParsedQuery()))
        self.assertTrue(pipeline.is_allergy_sensitive("vegan food", None))
        self.assertTrue(pipeline.is_allergy_sensitive("Coeliac friendly", ParsedQuery()))


class ProblemsStatusTests(SearchCase):
    def test_mission_contradiction_only_reports_problems(self):
        out = self.search(Q20, parsed(allergens=["peanut"]), MISSION_CONTRADICTIONS)
        s = out.results[0].summary
        self.assertEqual(s.status, pipeline.STATUS_PROBLEMS)
        self.assertEqual(s.status, "Reviews report problems")
        self.assertEqual(s.note, pipeline.ALLERGY_NOTE)  # the guard note stays
        self.assertEqual([c.label for c in s.claims], ["contradiction", "contradiction"])

    def test_problems_without_the_guard(self):
        out = self.search("vegan options for a group", parsed(dietary_restrictions=["vegan"]), MISSION_CONTRADICTIONS)
        self.assertFalse(out.allergy_sensitive)
        self.assertEqual(out.results[0].summary.status, pipeline.STATUS_PROBLEMS)

    def test_status_rules(self):
        s = lambda *labels: SimpleNamespace(status="evidence_found", claims=[SimpleNamespace(label=l) for l in labels])
        for labels, guarded, want in [
            (("contradiction",), False, pipeline.STATUS_PROBLEMS),
            (("contradiction", "contradiction"), True, pipeline.STATUS_PROBLEMS),
            (("contradiction", "uncertainty"), False, pipeline.STATUS_PROBLEMS),  # no support claim
            (("support", "contradiction"), False, pipeline.STATUS_MIXED),
            (("support", "contradiction"), True, pipeline.STATUS_MIXED),
            (("support",), False, pipeline.STATUS_SUPPORTED),
            (("support",), True, pipeline.STATUS_NEUTRAL),
            (("uncertainty",), False, pipeline.STATUS_INSUFFICIENT),
        ]:
            with self.subTest(labels=labels, guarded=guarded):
                self.assertEqual(pipeline.display_status(s(*labels), guarded), want)


class ReasoningEffortTests(unittest.TestCase):
    def test_app_sends_low_effort_for_parser_and_summarizer(self):
        client = Client(parsed(allergens=["peanut"]), MISSION_SUMMARY)
        pipeline.run_search(Q20, "Both", "semantic_capped", retriever=MissionRetriever(),
                            client=pipeline.UsageRecorder(client), model=MODEL, log_path=None)
        self.assertEqual([c["response_format"]["json_schema"]["name"] for c in client.calls],
                         ["parsed_query", "evidence_summary"])
        self.assertEqual([c.get("reasoning_effort") for c in client.calls], ["low", "low"])
        self.assertEqual(pipeline.LLM_REASONING_EFFORT, "low")

    def test_modules_default_is_unchanged(self):
        client = Client(parsed(allergens=["peanut"]), MISSION_SUMMARY)
        parse_query(Q20, client=client, model=MODEL)
        summarize_evidence(ParsedQuery(allergens=["peanut"]), RetrievalResult(MISSION_ID, 1.0, "x", list(MISSION_SNIPPETS)),
                           client=client, model=MODEL)
        self.assertTrue(all("reasoning_effort" not in c for c in client.calls))
        parse_query(Q20, client=client, model=MODEL, reasoning_effort="low")
        self.assertEqual(client.calls[-1]["reasoning_effort"], "low")


try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, "streamlit not installed")
class GuardUiTests(unittest.TestCase):
    def run_app(self, summary):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        llm = (pipeline.UsageRecorder(Client(parsed(allergens=["peanut"]), summary)), MODEL, None)
        with patch.object(pipeline, "LOG_PATH", Path(tmp.name) / "log.jsonl"), \
                patch.object(pipeline, "load_retriever", lambda name: MissionRetriever()), \
                patch.object(pipeline, "make_llm_client", lambda: llm):
            at = AppTest.from_file(str(Path(pipeline.REPO) / "app" / "streamlit_app.py"), default_timeout=30)
            at.run()
            at.text_input[0].input(Q20)
            at.button[0].click()
            at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def test_problems_status_uses_warning_style(self):
        at = self.run_app(MISSION_CONTRADICTIONS)
        warnings = [w.value for w in at.warning]
        self.assertTrue(any("Reviews report problems" in w for w in warnings), warnings)
        self.assertNotIn("Reviews report problems", " ".join(m.value for m in at.markdown))  # not plain text
        self.assertIn("⚠ **Contradicts**", " ".join(m.value for m in at.markdown))

    def test_mission_shows_no_check_marks_or_supported(self):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        llm = (pipeline.UsageRecorder(Client(parsed(allergens=["peanut"]), MISSION_SUMMARY)), MODEL, None)
        with patch.object(pipeline, "LOG_PATH", Path(tmp.name) / "log.jsonl"), \
                patch.object(pipeline, "load_retriever", lambda name: MissionRetriever()), \
                patch.object(pipeline, "make_llm_client", lambda: llm):
            at = AppTest.from_file(str(Path(pipeline.REPO) / "app" / "streamlit_app.py"), default_timeout=30)
            at.run()
            at.text_input[0].input(Q20)
            at.button[0].click()
            at.run()
        self.assertFalse(at.exception, at.exception)
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("Reviews discuss this — read each claim", text)
        self.assertNotIn("✓", text)
        self.assertNotIn("**supported**", text)
        self.assertIn("• Reviewer report", text)
        self.assertIn(pipeline.ALLERGY_NOTE, " ".join(w.value for w in at.warning))


if __name__ == "__main__":
    unittest.main()
