"""app/pipeline.py with fake retrievers and a fake LLM client (no data, network or Streamlit)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

from app import pipeline
from src.schemas import ParsedQuery, RetrievalResult, Snippet

MODEL = "openai/gpt-oss-20b"
T2019, T2020 = 1546300800000, 1577836800000
META = {f"G{i}": {"name": f"Place {i}", "borough": "Brooklyn" if i % 2 else "Manhattan", "address": f"{i} Main St",
                  "avg_rating": 4.0 + i / 10, "num_clean_reviews": 40 + i, "low_evidence": i == 5} for i in range(1, 7)}


class FakeRetriever:
    def __init__(self, gids=("G1", "G2", "G3", "G4"), no_snippets=()):
        self.gids, self.no_snippets, self.calls = list(gids), set(no_snippets), []

    def search(self, query, boroughs=None, top_k=10, include_low_evidence=False, density_mode=None):
        self.calls.append({"query": query, "boroughs": boroughs, "top_k": top_k,
                           "include_low_evidence": include_low_evidence, "density_mode": density_mode})
        out = []
        for gid in self.gids:
            if boroughs and META[gid]["borough"] not in boroughs:
                continue
            if META[gid]["low_evidence"] and not include_low_evidence:
                continue
            base = int(gid[1:]) * 100
            snippets = [] if gid in self.no_snippets else [
                Snippet("Great vegan options, our group loved it.", base + 1, T2019),
                Snippet("They told us nothing on the menu was vegan.", base + 2, T2020)]
            out.append(RetrievalResult(gid, 1.0 / (len(out) + 1), "fake", snippets))
        return out[:top_k]

    def info(self, gmap_id):
        return META[gmap_id]


def rate_limit_error():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return openai.RateLimitError("private provider text", response=httpx.Response(429, request=request), body=None)


class FakeClient:
    """Scripted by schema name; summaries cite the reviews they were given."""

    def __init__(self, parser="ok", summary="ok", location="Brooklyn"):
        self.parser, self.summary, self.location, self.calls = parser, summary, location, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        name = kwargs["response_format"]["json_schema"]["name"]
        self.calls.append(name)
        if name == "parsed_query":
            if self.parser == "bad":
                return self.response("not JSON")
            return self.response({"location": self.location, "dietary_restrictions": ["vegan"], "allergens": [],
                                  "group_needs": ["group dinner"], "preferences": []})
        if self.summary == "rate_limit":
            raise rate_limit_error()
        ids = [s["review_id"] for s in json.loads(kwargs["messages"][1]["content"])["snippets"]]
        claims = [{"text": "One reviewer reported vegan options for a group.", "review_ids": [ids[0]], "label": "support"}]
        if self.summary == "mixed":
            claims.append({"text": "Another reviewer reported no vegan dishes.", "review_ids": [ids[1]],
                           "label": "contradiction"})
        return self.response({"status": "evidence_found", "claims": claims, "limitations": ["Reports are old."]})

    @staticmethod
    def response(content):
        content = content if isinstance(content, str) else json.dumps(content)
        msg = SimpleNamespace(content=content, refusal=None)
        use = SimpleNamespace(prompt_tokens=500, completion_tokens=50, total_tokens=550,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=20))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=use)


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log = Path(tmp.name) / "logs" / "app_requests.jsonl"

    def run_search(self, query="vegan options for a group in Brooklyn", borough="Both", method="semantic_capped",
                   retriever=None, client=None, setup_error=None, **kw):
        self.retriever = retriever or FakeRetriever()
        self.client = pipeline.UsageRecorder(client) if client is not None else None
        return pipeline.run_search(query, borough, method, retriever=self.retriever, client=self.client, model=MODEL,
                                   setup_error=setup_error, log_path=self.log, **kw)

    def logged(self):
        return [json.loads(l) for l in open(self.log)]


class SuccessTests(PipelineTestCase):
    def test_parse_retrieve_and_summarize_top_three(self):
        fake = FakeClient(summary="mixed")
        out, raw = self.run_search(client=fake)
        self.assertEqual(out.parsed.dietary_restrictions, ["vegan"])
        call = self.retriever.calls[0]
        self.assertEqual(call["query"], "vegan options for a group in Brooklyn")  # raw text, not ParsedQuery
        self.assertEqual((call["boroughs"], call["density_mode"]), (["Brooklyn"], "capped"))
        self.assertIn("limited to Brooklyn", " ".join(out.notices))
        self.assertEqual(fake.calls, ["parsed_query"] + ["evidence_summary"] * 2)  # G1, G3 in Brooklyn
        self.assertEqual([v.gmap_id for v in out.results], ["G1", "G3"])
        v = out.results[0]
        self.assertEqual((v.name, v.borough, v.avg_rating, v.num_reviews), ("Place 1", "Brooklyn", 4.1, 41))
        self.assertEqual(v.summary.status, pipeline.STATUS_MIXED)
        support, contra = v.summary.claims
        self.assertEqual((support.label, contra.label), ("support", "contradiction"))
        self.assertEqual([(e.review_id, e.date) for e in support.excerpts], [(101, "2019-01")])
        self.assertEqual(contra.excerpts[0].text, "They told us nothing on the menu was vegan.")
        self.assertEqual(set(out.timings), {"parse_s", "retrieve_s", "summarize_s", "total_s"})

    def test_supported_status_and_auto_summary_limit(self):
        out, _ = self.run_search(query="vegan options for a group", client=FakeClient(location=None))
        self.assertIsNone(out.boroughs)  # "Both" and no borough named
        self.assertEqual([v.summary.status for v in out.results[:3]], [pipeline.STATUS_SUPPORTED] * 3)
        self.assertIsNone(out.results[3].summary)
        self.assertEqual(out.results[3].summary_skipped, "not requested")

    def test_log_record(self):
        os.environ["GROQ_API_KEY"] = "sentinel-key-must-not-be-logged"
        self.addCleanup(os.environ.pop, "GROQ_API_KEY", None)
        out, _ = self.run_search(client=FakeClient())
        (rec,) = self.logged()
        self.assertEqual((rec["event"], rec["query"], rec["method"], rec["borough_choice"]),
                         ("search", out.query, "semantic_capped", "Both"))
        self.assertEqual(rec["parsed"]["dietary_restrictions"], ["vegan"])
        self.assertEqual(rec["prompt_sha256"], pipeline.prompt_hashes())
        self.assertEqual(set(rec["timings"]), {"parse_s", "retrieve_s", "summarize_s", "total_s"})
        self.assertEqual([c["call"] for c in rec["llm_calls"]], ["parsed_query", "evidence_summary", "evidence_summary"])
        self.assertEqual(rec["llm_calls"][0]["usage"]["total_tokens"], 550)
        self.assertEqual([r["gmap_id"] for r in rec["results"]], ["G1", "G3"])
        self.assertEqual(rec["summaries"][0]["claims"][0]["review_ids"], [101])
        self.assertNotIn("sentinel-key", self.log.read_text())

    def test_explain_on_request(self):
        out, raw = self.run_search(query="vegan options for a group", client=FakeClient(location=None))
        view = pipeline.explain(out, raw, "G4", client=self.client, model=MODEL, log_path=self.log)
        self.assertEqual(view.summary.status, pipeline.STATUS_SUPPORTED)
        self.assertIsNone(view.summary_skipped)
        rec = self.logged()[-1]
        self.assertEqual((rec["event"], rec["gmap_id"], rec["llm_calls"][0]["call"]), ("explain", "G4", "evidence_summary"))
        self.assertEqual(rec["summary"]["status"], pipeline.STATUS_SUPPORTED)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.run_search(query="   ", client=FakeClient())
        with self.assertRaises(ValueError):
            self.run_search(method="bm25", client=FakeClient())


class FailureTests(PipelineTestCase):
    def test_parser_failure_falls_back_to_the_raw_query(self):
        out, _ = self.run_search(query="vegan in Brooklyn", client=FakeClient(parser="bad"))
        self.assertIsNone(out.parsed)
        self.assertEqual((out.parse_error["category"], out.parse_error["code"]), ("response", "invalid_response"))
        self.assertIn("Searching with your exact words", " ".join(out.notices))
        self.assertEqual(self.retriever.calls[0]["query"], "vegan in Brooklyn")
        self.assertIsNone(self.retriever.calls[0]["boroughs"])  # no parsed location -> no inferred filter
        self.assertEqual(out.summary_context, ParsedQuery(preferences=["vegan in Brooklyn"]))
        self.assertIsNotNone(out.results[0].summary)  # summaries still run
        self.assertEqual(self.logged()[0]["errors"]["parse"], {"category": "response", "code": "invalid_response"})

    def test_empty_results(self):
        fake = FakeClient()
        out, _ = self.run_search(retriever=FakeRetriever(gids=()), client=fake)
        self.assertEqual(out.results, [])
        self.assertIn("No restaurants matched", " ".join(out.notices))
        self.assertEqual(fake.calls, ["parsed_query"])  # no summary requests

    def test_summarizer_rate_limit_stops_and_shows_excerpts(self):
        fake = FakeClient(summary="rate_limit", location=None)
        out, _ = self.run_search(query="vegan options for a group", client=fake)
        first, second = out.results[0], out.results[1]
        self.assertEqual((first.summary_error["category"], first.summary_error["code"]), ("rate_limit", "rate_limited"))
        self.assertIn("rate limit", first.summary_error["message"])
        self.assertNotIn("private provider text", json.dumps(first.summary_error))
        self.assertIsNone(second.summary)
        self.assertIn("rate limit", second.summary_skipped)  # stopped: no further calls
        self.assertEqual(fake.calls, ["parsed_query", "evidence_summary"])
        self.assertTrue(first.excerpts)  # excerpts still shown
        self.assertIn("Some summaries are unavailable", " ".join(out.notices))
        rec = self.logged()[0]
        self.assertEqual(rec["errors"]["summaries"]["G1"], {"category": "rate_limit", "code": "rate_limited"})
        self.assertFalse(rec["llm_calls"][1]["ok"])

    def test_low_evidence_and_no_snippets(self):
        fake = FakeClient(location=None)
        out, _ = self.run_search(query="vegan options", retriever=FakeRetriever(gids=("G5", "G1"), no_snippets={"G5"}),
                                 client=fake, include_low_evidence=True)
        self.assertTrue(self.retriever.calls[0]["include_low_evidence"])
        g5 = out.results[0]
        self.assertTrue(g5.low_evidence)
        self.assertEqual(g5.summary.status, pipeline.STATUS_INSUFFICIENT)  # no excerpts -> no call
        self.assertEqual(fake.calls, ["parsed_query", "evidence_summary"])  # only G1 summarized
        self.assertFalse(out.results[1].low_evidence)

    def test_missing_api_key_still_searches(self):
        setup = {"category": "configuration", "code": "invalid_api_key_configuration",
                 "message": pipeline.ERROR_MESSAGES["configuration"]}
        out, raw = self.run_search(client=None, setup_error=setup)
        self.assertIsNone(out.parsed)
        self.assertTrue(out.results)
        self.assertTrue(all(v.summary is None for v in out.results))
        self.assertIn("GROQ_API_KEY", out.results[0].summary_skipped)
        self.assertIn("without query parsing or summaries", " ".join(out.notices))
        view = pipeline.explain(out, raw, out.results[0].gmap_id, client=None, model=None, log_path=self.log)
        self.assertEqual(view.summary_error["category"], "configuration")

    def test_make_llm_client_reports_missing_key(self):
        with patch.dict(os.environ, {"GROQ_MODEL": MODEL}, clear=True), patch.object(pipeline, "load_env"):
            client, model, err = pipeline.make_llm_client()
        self.assertEqual((client, model, err["category"], err["code"]),
                         (None, None, "configuration", "invalid_api_key_configuration"))


class BoroughTests(unittest.TestCase):
    def test_rules(self):
        p = lambda loc: ParsedQuery(location=loc)
        self.assertEqual(pipeline.borough_filter("Manhattan", None, "x"), (["Manhattan"], None))
        boroughs, notice = pipeline.borough_filter("Manhattan", p("Brooklyn"), "tacos in brooklyn")
        self.assertEqual(boroughs, ["Manhattan"])
        self.assertIn("mentions Brooklyn", notice)
        self.assertEqual(pipeline.borough_filter("Both", p("Brooklyn"), "tacos in Brooklyn")[0], ["Brooklyn"])
        self.assertEqual(pipeline.borough_filter("Both", p("Brooklyn"), "tacos near me"), (None, None))  # not in text
        self.assertEqual(pipeline.borough_filter("Both", p("Williamsburg"), "tacos in Williamsburg"), (None, None))
        self.assertEqual(pipeline.borough_filter("Both", p("Queens"), "tacos in Queens"), (None, None))

    def test_status_mapping_and_methods(self):
        s = lambda status, *labels: SimpleNamespace(status=status, claims=[SimpleNamespace(label=l) for l in labels])
        self.assertEqual(pipeline.display_status(s("insufficient_evidence")), pipeline.STATUS_INSUFFICIENT)
        self.assertEqual(pipeline.display_status(s("evidence_found", "support")), pipeline.STATUS_SUPPORTED)
        self.assertEqual(pipeline.display_status(s("evidence_found", "support", "contradiction")), pipeline.STATUS_MIXED)
        self.assertEqual(pipeline.display_status(s("evidence_found", "uncertainty")), pipeline.STATUS_INSUFFICIENT)
        from eval.make_pool import SYSTEMS
        self.assertEqual(list(pipeline.METHODS), [n for n, _, _ in SYSTEMS])
        self.assertIn(pipeline.DEFAULT_METHOD, pipeline.METHODS)


if __name__ == "__main__":
    unittest.main()
