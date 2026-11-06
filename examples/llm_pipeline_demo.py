"""Offline wiring demonstration: synthetic reviews AND scripted fake model outputs.

No .env loading, API key, network, real index or paid request is used in any scenario.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BOROUGHS
from src.schemas import RetrievalResult, Snippet
from src.llm.query_parser import parse_query
from src.llm.evidence_summarizer import summarize_evidence
from src.llm.groq_client import LLMError

MODEL = "openai/gpt-oss-20b"  # Only a label passed to a fake client; no live mode exists.
RAW_QUERY = "Vegetarian dinner in Brooklyn for a friend with a sesame allergy."


class FakeRetriever:
    """Tiny in-memory fixture with the existing search signature, not an IR algorithm.

    Only boroughs and top_k affect these canned results. Diets are not database filters.
    """

    def __init__(self, empty=False, empty_evidence=False):
        self.empty = empty
        self.empty_evidence = empty_evidence
        self.calls = []
        self.meta = {
            "fake-brooklyn": {"name": "Synthetic Brooklyn Cafe", "borough": "Brooklyn"},
            "fake-manhattan": {"name": "Synthetic Manhattan Cafe", "borough": "Manhattan"},
        }

    def search(self, query_terms, boroughs=None, top_k=10, include_low_evidence=False):
        self.calls.append((query_terms, boroughs))
        if self.empty:
            return []
        out = []
        for i, (gid, info) in enumerate(self.meta.items()):
            if boroughs and info["borough"].lower() not in {b.lower() for b in boroughs}:
                continue
            base = 101 + i * 100
            out.append(RetrievalResult(gid, 0.5, "synthetic", [
                Snippet("On a 2019 visit, staff prepared my vegetarian dinner and discussed my sesame allergy.",
                        base, 1546300800000),
                Snippet("On a 2020 visit, staff refused my sesame-allergy request because of cross-contact concerns.",
                        base + 1, 1577836800000)]))
            if self.empty_evidence:
                out[-1].evidence_snippets = []
        return out[:max(0, top_k)]

    def info(self, gmap_id):
        return self.meta[gmap_id]


class FakeSDKClient:
    """Scripted responses exercise REAL module validation, not real model understanding."""

    def __init__(self, scenario="success", location="Brooklyn"):
        self.scenario = scenario
        self.location = location
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        schema_name = kwargs["response_format"]["json_schema"]["name"]
        if schema_name == "parsed_query":
            if self.scenario == "parser-error":
                return self.response("not JSON")
            output = {"location": self.location, "dietary_restrictions": ["vegetarian"],
                      "allergens": ["sesame"], "group_needs": ["friend has sesame allergy"],
                      "preferences": ["dinner"]}
        else:
            if self.scenario == "summary-error":
                raise TimeoutError("SIMULATED_PRIVATE_DETAIL must never appear in the public error")
            snippets = json.loads(kwargs["messages"][1]["content"])["snippets"]
            # Explicitly scripted for the two synthetic restaurants above.
            ids = [s["review_id"] for s in snippets]
            if ids not in ([101, 102], [201, 202]):
                raise ValueError("No scripted response for these synthetic fixtures.")
            output = {"status": "evidence_found", "claims": [
                {"text": "One reviewer reported a vegetarian meal and discussion of a sesame allergy in 2019.",
                 "review_ids": [ids[0]], "label": "support"},
                {"text": "Another reviewer reported refusal of sesame-allergy accommodation in 2020 over cross-contact concerns.",
                 "review_ids": [ids[1]], "label": "contradiction"}],
                "limitations": ["Historical reports conflict; current accommodation is unknown."]}
        return self.response(json.dumps(output))

    @staticmethod
    def response(content):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content=content, refusal=None))])


def borough_filters(location, raw_query):
    """Filter only an exact supported borough explicitly present in the raw text."""
    if not isinstance(location, str):
        return None
    for borough in BOROUGHS:
        if (location.strip().casefold() == borough.casefold()
                and re.search(r"\b" + re.escape(borough) + r"\b", raw_query, re.IGNORECASE)):
            return [borough]
    return None


def run_pipeline(raw_query, retriever, client, write=print):
    """Example application handling; failures produce no replacement query or evidence."""
    try:
        parsed = parse_query(raw_query, client=client, model=MODEL)
    except LLMError as error:
        write(f"Parser unavailable [{error.category}/{error.code}]: {error}")
        return "parser_error"
    boroughs = borough_filters(parsed.location, raw_query)
    write(f"Parsed diets: {parsed.dietary_restrictions}; allergens: {parsed.allergens}")
    write(f"Borough filter: {boroughs}; dietary requirements are not database filters.")
    # Keep the exact original string: the retriever does not accept ParsedQuery.
    results = retriever.search(raw_query, boroughs=boroughs)
    if not results:
        write("No search results. No summarizer request made.")
        return "no_results"
    failures = False
    insufficient = False
    for result in results:
        write(retriever.info(result.gmap_id)["name"])
        try:
            summary = summarize_evidence(parsed, result, client=client, model=MODEL)
        except LLMError as error:
            write(f"Summary unavailable [{error.category}/{error.code}]: {error}")
            failures = True
            continue  # Preserve other results; do not invent an explanation for this one.
        write(f"Status: {summary.status}")
        if summary.status == "insufficient_evidence":
            insufficient = True
            write("Restaurant matched, but review evidence is insufficient. No accommodation conclusion is available.")
        for claim in summary.claims:
            ids = ", ".join(str(rid) for rid in claim.review_ids)
            write(f"  {claim.label} [reviews {ids}]: {claim.text}")
        for limitation in summary.limitations:
            write(f"  Limitation: {limitation}")
    if failures:
        return "summary_error"
    return "insufficient_evidence" if insufficient else "complete"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["success", "parser-error", "no-results", "summary-error", "insufficient-evidence", "all"],
                        default="success")
    args = parser.parse_args(argv)
    print("OFFLINE DEMO: fake restaurants, reviews and model responses; no live integration is verified.")
    scenarios = ("success", "parser-error", "no-results", "summary-error", "insufficient-evidence") if args.scenario == "all" else (args.scenario,)
    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        run_pipeline(RAW_QUERY, FakeRetriever(empty=scenario == "no-results",
                     empty_evidence=scenario == "insufficient-evidence"), FakeSDKClient(scenario))


if __name__ == "__main__":
    main()
