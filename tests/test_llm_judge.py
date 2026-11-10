"""LLM judge tests with a fake SDK client and synthetic pool rows (no network, no API key)."""

import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval import llm_judge, make_pool
from src.llm.groq_client import LLMAPIError, LLMResponseError

MODEL = "openai/gpt-oss-20b"


def response(content, finish_reason="stop"):
    msg = SimpleNamespace(content=content if isinstance(content, str) else json.dumps(content), refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)])


class FakeClient:
    """OpenAI-compatible fake: returns scripted answers (dicts) or raises scripted exceptions."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return response(item)


def pool_row(qid, gid, evidence, query="vegan options"):
    row = {c: "" for c in make_pool.COLUMNS}
    row.update(query_id=qid, query=query, gmap_id=gid, name=f"R{gid}", borough="Brooklyn",
               evidence_status="retrieved" if evidence else "none",
               evidence_review_ids=";".join(str(i) for i, _ in evidence))
    for k, (rid, text) in enumerate(evidence, 1):
        row[f"evidence_{k}"] = f"[review {rid}, 2020-05] {text}"
    return row


ROWS = [pool_row("Q1", "G1", [(11, "Lots of vegan options, great for our group."), (12, "Service was slow.")]),
        pool_row("Q1", "G2", []),
        pool_row("Q2", "G3", [(31, "Not much for vegan options.")], query="vegan brunch")]
GOOD = {"label": 2, "reason": "A reviewer reports many vegan options for a group.", "review_ids": [11]}
ZERO = {"label": 0, "reason": "The only review says there are few vegan options.", "review_ids": [31]}


class JudgeTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.pool = self.dir / "pool.csv"
        with open(self.pool, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=make_pool.COLUMNS)
            w.writeheader()
            w.writerows(ROWS)
        self.out, self.errors = self.dir / "llm_labels.jsonl", self.dir / "errors.jsonl"
        self.sleeps = []
        p = patch.object(llm_judge, "sleep", side_effect=self.sleeps.append)
        p.start()
        self.addCleanup(p.stop)

    def run_judge(self, client, *extra):
        argv = ["--pool", str(self.pool), "--out", str(self.out), "--errors", str(self.errors),
                "--model", MODEL, "--rpm", "0", "--tpm", "0", *extra]
        with redirect_stdout(io.StringIO()) as out:
            llm_judge.main(argv, client=client)
        return out.getvalue()

    def records(self, path=None):
        path = path or self.out
        return [json.loads(l) for l in open(path)] if Path(path).exists() else []


class EndToEndTests(JudgeTestCase):
    def test_labels_pairs_and_logs_metadata(self):
        client = FakeClient([GOOD, ZERO])
        self.run_judge(client)
        recs = {r["gmap_id"]: r for r in self.records()}
        self.assertEqual(set(recs), {"G1", "G2", "G3"})
        self.assertEqual(len(client.calls), 2)  # G2 has no evidence: rule label, no call
        self.assertEqual((recs["G1"]["label"], recs["G1"]["review_ids"], recs["G1"]["source"]), (2, [11], "llm"))
        self.assertEqual((recs["G2"]["label"], recs["G2"]["source"], recs["G2"]["model"]), (0, "rule", None))
        system = llm_judge.system_prompt(llm_judge.RUBRIC_PATH.read_text())
        for r in recs.values():
            self.assertEqual(r["prompt_sha256"], llm_judge.prompt_hash(system))
            self.assertRegex(r["timestamp"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$")
        self.assertEqual(recs["G1"]["model"], MODEL)

    def test_request_contains_rubric_query_and_evidence_with_strict_schema(self):
        client = FakeClient([GOOD, ZERO])
        self.run_judge(client)
        call = client.calls[0]
        system, user = call["messages"][0]["content"], json.loads(call["messages"][1]["content"])
        self.assertIn("No evidence found = 0", system)  # rubric text is in the prompt
        self.assertIn("Several needs", system)
        self.assertEqual(user["query"], "vegan options")
        self.assertEqual(user["restaurant"], {"name": "RG1", "borough": "Brooklyn"})
        self.assertEqual([r["review_id"] for r in user["reviews"]], [11, 12])
        self.assertEqual(user["reviews"][0]["date"], "2020-05")
        fmt = call["response_format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertTrue(fmt["json_schema"]["strict"])
        self.assertEqual(fmt["json_schema"]["schema"], llm_judge.JUDGE_SCHEMA)
        self.assertEqual(call["model"], MODEL)

    def test_resumable_and_tolerates_a_partial_line(self):
        self.run_judge(FakeClient([GOOD, ZERO]))
        with open(self.out, "a") as f:
            f.write('{"query_id": "Q9", "gmap_')  # interrupted write
        client = FakeClient([])
        self.run_judge(client)
        self.assertEqual(client.calls, [])
        self.assertEqual(len([l for l in open(self.out) if l.strip().endswith("}")]), 3)

    def test_sample_flag_restricts_pairs(self):
        key = self.dir / "sample_key.csv"
        with open(key, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "query_id", "gmap_id", "category", "single_system", "evidence_status"])
            w.writerow(["S001", "Q2", "G3", "single_diet", 0, "retrieved"])
        client = FakeClient([ZERO])
        self.run_judge(client, "--sample", "--sample-key", str(key))
        self.assertEqual([r["gmap_id"] for r in self.records()], ["G3"])

    def test_limit(self):
        client = FakeClient([GOOD])
        self.run_judge(client, "--limit", "1")
        self.assertEqual(len(self.records()), 1)

    def test_dry_run_makes_no_call_and_writes_nothing(self):
        out = self.run_judge(FakeClient([]), "--dry-run")
        self.assertIn("Estimated tokens per call", out)
        self.assertFalse(self.out.exists())

    def test_retries_a_timeout_through_the_real_request_path(self):
        client = FakeClient([TimeoutError(), GOOD, ZERO])
        self.run_judge(client)
        g1 = next(r for r in self.records() if r["gmap_id"] == "G1")
        self.assertEqual(g1["attempts"], 2)
        self.assertEqual(len(self.sleeps), 1)


class RetryTests(JudgeTestCase):
    def judge(self, side_effect, max_retries=3):
        with patch.object(llm_judge, "request_json", side_effect=side_effect) as fake:
            rec, err = llm_judge.judge_pair(ROWS[0], client=None, model=MODEL, system="s", phash="h",
                                            max_retries=max_retries, pacer=llm_judge.Pacer(0, 0))
        return rec, err, fake.call_count

    def test_rate_limit_backs_off_then_succeeds(self):
        rl = LLMAPIError("rate", category="rate_limit", code="rate_limited", http_status=429)
        rec, err, calls = self.judge([rl, rl, GOOD])
        self.assertIsNone(err)
        self.assertEqual((rec["attempts"], calls), (3, 3))
        self.assertEqual(len(self.sleeps), 2)
        self.assertLess(self.sleeps[0], self.sleeps[1])  # exponential

    def test_gives_up_after_max_retries(self):
        rl = LLMAPIError("rate", category="rate_limit", code="rate_limited", http_status=429)
        rec, err, calls = self.judge([rl] * 3, max_retries=2)
        self.assertIsNone(rec)
        self.assertEqual((err["category"], err["code"], err["http_status"], calls), ("rate_limit", "rate_limited", 429, 3))

    def test_output_parse_failed_is_retried_but_bad_request_is_not(self):
        rec, err, calls = self.judge([LLMAPIError("x", category="bad_request", code="output_parse_failed"), GOOD])
        self.assertEqual((rec["label"], calls), (2, 2))
        rec, err, calls = self.judge([LLMAPIError("x", category="bad_request", code="bad_request")])
        self.assertEqual((rec, err["code"], calls), (None, "bad_request", 1))

    def test_invalid_answers_are_rejected(self):
        bad_answers = [
            {"label": 2, "reason": "Cites a review that was not shown.", "review_ids": [99]},
            {"label": 1, "reason": "No citation.", "review_ids": []},
            {"label": 3, "reason": "Out of range.", "review_ids": [11]},
            {"label": True, "reason": "Boolean label.", "review_ids": [11]},
            {"label": 2, "reason": "Two lines.\nSecond line.", "review_ids": [11]},
            {"label": 2, "reason": "x" * 301, "review_ids": [11]},
            {"label": 2, "reason": "Duplicate ids.", "review_ids": [11, 11]},
            {"label": 2, "reason": "Extra key.", "review_ids": [11], "confidence": 0.9},
        ]
        for answer in bad_answers:
            with self.subTest(answer=answer):
                self.sleeps.clear()
                rec, err, calls = self.judge([answer, answer])
                self.assertIsNone(rec)
                self.assertEqual((err["category"], calls), ("response", 2))  # retried once

    def test_run_stops_after_consecutive_rate_limited_pairs(self):
        rl = LLMAPIError("rate", category="rate_limit", code="rate_limited", http_status=429)
        rows = [pool_row("Q1", f"G{k}", [(k, "vegan")]) for k in range(10)]
        with open(self.pool, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=make_pool.COLUMNS)
            w.writeheader()
            w.writerows(rows)
        with patch.object(llm_judge, "request_json", side_effect=rl) as fake:
            out = self.run_judge(FakeClient([]), "--max-retries", "1", "--max-rate-limited", "2")
        self.assertIn("Stopping: 2 consecutive pairs hit rate limits", out)
        self.assertEqual(fake.call_count, 4)
        self.assertEqual(len(self.records(self.errors)), 2)
        self.assertEqual(self.records(), [])


class HelperTests(unittest.TestCase):
    def test_prompt_hash_changes_with_rubric(self):
        a, b = llm_judge.system_prompt("rule A"), llm_judge.system_prompt("rule B")
        self.assertNotEqual(llm_judge.prompt_hash(a), llm_judge.prompt_hash(b))
        self.assertEqual(llm_judge.prompt_hash(a), llm_judge.prompt_hash(llm_judge.system_prompt("rule A")))

    def test_larger_model_note(self):
        note = llm_judge.larger_model_note("openai/gpt-oss-20b")
        self.assertIn("openai/gpt-oss-120b", note)
        self.assertIsNone(llm_judge.larger_model_note("openai/gpt-oss-120b"))
        self.assertIsNone(llm_judge.larger_model_note("meta-models/muse-spark-open"))

    def test_pacer_respects_token_budget(self):
        clock, sleeps = [0.0], []

        def fake_sleep(s):
            sleeps.append(s)
            clock[0] += s

        with patch.object(llm_judge, "sleep", side_effect=fake_sleep):
            pacer = llm_judge.Pacer(rpm=100, tpm=7000, clock=lambda: clock[0])
            pacer.wait(4000)
            pacer.wait(4000)  # would exceed 7000 within the minute
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(clock[0], 60)

    def test_pacer_respects_rpm(self):
        clock = [0.0]
        with patch.object(llm_judge, "sleep", side_effect=lambda s: clock.__setitem__(0, clock[0] + s)):
            pacer = llm_judge.Pacer(rpm=2, tpm=0, clock=lambda: clock[0])
            for _ in range(3):
                pacer.wait(10)
        self.assertGreaterEqual(clock[0], 60)

    def test_parse_evidence_rejects_malformed_cells(self):
        row = pool_row("Q", "G", [(1, "ok")])
        row["evidence_2"] = "no prefix"
        with self.assertRaisesRegex(ValueError, "unparseable"):
            llm_judge.parse_evidence(row)

    def test_env_loading_only_when_git_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text("NYC_JUDGE_TEST_VAR=loaded\n")
            self.addCleanup(os.environ.pop, "NYC_JUDGE_TEST_VAR", None)
            with patch.object(llm_judge.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
                with self.assertRaises(SystemExit):
                    llm_judge.ensure_env_ignored(env)
            self.assertNotIn("NYC_JUDGE_TEST_VAR", os.environ)
            with patch.object(llm_judge.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
                self.assertTrue(llm_judge.ensure_env_ignored(env))
            self.assertEqual(os.environ["NYC_JUDGE_TEST_VAR"], "loaded")
            self.assertFalse(llm_judge.ensure_env_ignored(Path(d) / "missing.env"))


if __name__ == "__main__":
    unittest.main()
