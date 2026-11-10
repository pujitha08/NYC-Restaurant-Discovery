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
from src.llm.groq_client import LLMAPIError

MODEL = "openai/gpt-oss-20b"
RUBRIC = llm_judge.RUBRIC_PATH.read_text()


def usage(prompt=1000, completion=120, reasoning=80):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion,
                           completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning))


def response(content, use=None):
    msg = SimpleNamespace(content=content if isinstance(content, str) else json.dumps(content), refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=use or usage())


class FakeClient:
    """OpenAI-compatible fake: scripted answers (dict / str) or exceptions; a callable item gets the kwargs."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if callable(item) and not isinstance(item, type):
            item = item(kwargs)
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


def answer_batch(labels):
    """A scripted batch answer: labels maps gmap_id -> (label, review_ids); ids resolved from the request."""
    def build(kwargs):
        restaurants = json.loads(kwargs["messages"][1]["content"])["restaurants"]
        return {"labels": [{"restaurant_id": r["restaurant_id"], "label": labels[r["name"][1:]][0],
                            "reason": f"Judged {r['name']}.", "review_ids": labels[r["name"][1:]][1]}
                           for r in restaurants]}
    return build


class JudgeTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.pool = self.dir / "pool.csv"
        self.write_pool(ROWS)
        self.out, self.errors = self.dir / "llm_labels.jsonl", self.dir / "errors.jsonl"
        self.sleeps = []
        p = patch.object(llm_judge, "sleep", side_effect=self.sleeps.append)
        p.start()
        self.addCleanup(p.stop)

    def write_pool(self, rows):
        with open(self.pool, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=make_pool.COLUMNS)
            w.writeheader()
            w.writerows(rows)

    def run_judge(self, client, *extra):
        argv = ["--pool", str(self.pool), "--out", str(self.out), "--errors", str(self.errors),
                "--model", MODEL, "--rpm", "0", "--tpm", "0", *extra]
        with redirect_stdout(io.StringIO()) as out:
            llm_judge.main(argv, client=client)
        return out.getvalue()

    def records(self, path=None):
        path = path or self.out
        return [json.loads(l) for l in open(path)] if Path(path).exists() else []

    def judge(self, client=None, **kw):
        return llm_judge.Judge(client=client, model=kw.pop("model", MODEL), rubric_text=RUBRIC,
                               max_retries=kw.pop("max_retries", 3), pacer=llm_judge.Pacer(0, 0), **kw)


class EndToEndTests(JudgeTestCase):
    def test_labels_pairs_and_logs_metadata_and_usage(self):
        client = FakeClient([GOOD, ZERO])
        self.run_judge(client)
        recs = {r["gmap_id"]: r for r in self.records()}
        self.assertEqual(set(recs), {"G1", "G2", "G3"})
        self.assertEqual(len(client.calls), 2)  # G2 has no evidence: rule label, no call
        g1 = recs["G1"]
        self.assertEqual((g1["label"], g1["review_ids"], g1["source"], g1["batch_size"]), (2, [11], "llm", 1))
        self.assertEqual(g1["usage"], {"prompt_tokens": 1000, "completion_tokens": 120, "reasoning_tokens": 80,
                                       "total_tokens": 1120})
        self.assertEqual(g1["usage_per_pair"]["prompt_tokens"], 1000)
        self.assertEqual((g1["model"], g1["reasoning_effort"]), (MODEL, "low"))
        self.assertEqual((recs["G2"]["label"], recs["G2"]["source"], recs["G2"]["usage"]), (0, "rule", None))
        for r in recs.values():
            self.assertEqual(r["prompt_sha256"], llm_judge.prompt_hash(llm_judge.system_prompt(RUBRIC)))
            self.assertRegex(r["timestamp"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$")

    def test_request_contains_rubric_query_evidence_schema_and_effort(self):
        client = FakeClient([GOOD, ZERO])
        self.run_judge(client, "--reasoning-effort", "medium")
        call = client.calls[0]
        system, user = call["messages"][0]["content"], json.loads(call["messages"][1]["content"])
        self.assertIn("No evidence found = 0", system)
        self.assertNotIn("BATCH MODE", system)
        self.assertEqual(user["query"], "vegan options")
        self.assertEqual(user["restaurant"], {"name": "RG1", "borough": "Brooklyn"})
        self.assertEqual([r["review_id"] for r in user["reviews"]], [11, 12])
        fmt = call["response_format"]["json_schema"]
        self.assertEqual((fmt["strict"], fmt["schema"]), (True, llm_judge.JUDGE_SCHEMA))
        self.assertEqual(call["reasoning_effort"], "medium")

    def test_reasoning_effort_not_sent_to_models_without_it(self):
        client = FakeClient([GOOD, ZERO])
        out = self.run_judge(client, "--model", "qwen/qwen3.8-27b")
        self.assertNotIn("reasoning_effort", client.calls[0])
        self.assertIn("does not take reasoning_effort", out)
        self.assertIsNone(self.records()[-1]["reasoning_effort"])

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
        self.run_judge(FakeClient([ZERO]), "--sample", "--sample-key", str(key))
        self.assertEqual([r["gmap_id"] for r in self.records()], ["G3"])

    def test_dry_run_makes_no_call_and_writes_nothing(self):
        out = self.run_judge(FakeClient([]), "--dry-run", "--batch-size", "3")
        self.assertIn("Estimated tokens per call", out)
        self.assertFalse(self.out.exists())

    def test_retries_a_timeout_through_the_real_request_path(self):
        self.run_judge(FakeClient([TimeoutError(), GOOD, ZERO]))
        g1 = next(r for r in self.records() if r["gmap_id"] == "G1")
        self.assertEqual(g1["attempts"], 2)
        self.assertEqual(len(self.sleeps), 1)


class RetryTests(JudgeTestCase):
    def single(self, side_effect, max_retries=3):
        with patch.object(llm_judge, "request_json", side_effect=side_effect) as fake:
            rec, err = self.judge(max_retries=max_retries).single(ROWS[0])
        return rec, err, fake.call_count

    def ok(self, data):
        return data, {"prompt_tokens": 1, "completion_tokens": 1, "reasoning_tokens": None, "total_tokens": 2}

    def test_rate_limit_backs_off_then_succeeds(self):
        rl = LLMAPIError("rate", category="rate_limit", code="rate_limited", http_status=429)
        rec, err, calls = self.single([rl, rl, self.ok(GOOD)])
        self.assertIsNone(err)
        self.assertEqual((rec["attempts"], calls), (3, 3))
        self.assertEqual(len(self.sleeps), 2)
        self.assertLess(self.sleeps[0], self.sleeps[1])
        self.assertIsNone(rec["usage"]["reasoning_tokens"])

    def test_gives_up_after_max_retries(self):
        rl = LLMAPIError("rate", category="rate_limit", code="rate_limited", http_status=429)
        rec, err, calls = self.single([rl] * 3, max_retries=2)
        self.assertIsNone(rec)
        self.assertEqual((err["category"], err["code"], err["http_status"], err["attempts"], calls),
                         ("rate_limit", "rate_limited", 429, 3, 3))

    def test_output_parse_failed_is_retried_but_bad_request_is_not(self):
        rec, _, calls = self.single([LLMAPIError("x", category="bad_request", code="output_parse_failed"), self.ok(GOOD)])
        self.assertEqual((rec["label"], calls), (2, 2))
        rec, err, calls = self.single([LLMAPIError("x", category="bad_request", code="bad_request")])
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
                rec, err, calls = self.single([self.ok(answer), self.ok(answer)])
                self.assertIsNone(rec)
                self.assertEqual((err["category"], calls), ("response", 2))  # retried once

    def test_run_stops_after_consecutive_rate_limited_calls(self):
        rl = LLMAPIError("rate", category="rate_limit", code="rate_limited", http_status=429)
        self.write_pool([pool_row("Q1", f"G{k}", [(k, "vegan")]) for k in range(10)])
        with patch.object(llm_judge, "request_json", side_effect=rl) as fake:
            out = self.run_judge(FakeClient([]), "--max-retries", "1", "--max-rate-limited", "2")
        self.assertIn("Stopping: 2 consecutive calls hit rate limits", out)
        self.assertEqual(fake.call_count, 4)
        self.assertEqual(len(self.records(self.errors)), 2)
        self.assertEqual(self.records(), [])


BATCH_ROWS = [pool_row("Q1", "G1", [(11, "Great vegan menu."), (12, "Friendly staff.")]),
              pool_row("Q1", "G2", [(21, "No vegan options at all.")]),
              pool_row("Q1", "G3", [(31, "A vegan side salad."), (32, "Busy at night.")])]
BATCH_LABELS = {"G1": (2, [11]), "G2": (0, [21]), "G3": (1, [31])}


class BatchTests(JudgeTestCase):
    def test_batch_success_records_each_pair_with_shared_usage(self):
        client = FakeClient([answer_batch(BATCH_LABELS)])
        outcomes = self.judge(client).batch(BATCH_ROWS)
        self.assertEqual(len(client.calls), 1)
        recs = [r for r, e in outcomes]
        self.assertEqual([(r["gmap_id"], r["label"], r["review_ids"]) for r in recs],
                         [("G1", 2, [11]), ("G2", 0, [21]), ("G3", 1, [31])])  # in input order
        self.assertTrue(all(r["batch_size"] == 3 and r["batch_id"] == recs[0]["batch_id"] for r in recs))
        self.assertEqual(recs[0]["usage"]["total_tokens"], 1120)
        self.assertEqual(recs[0]["usage_per_pair"]["prompt_tokens"], round(1000 / 3, 1))
        self.assertEqual(recs[0]["prompt_sha256"], llm_judge.prompt_hash(llm_judge.system_prompt(RUBRIC, batch=True),
                                                                         llm_judge.BATCH_SCHEMA))
        call = client.calls[0]
        self.assertIn("BATCH MODE", call["messages"][0]["content"])
        self.assertEqual(call["response_format"]["json_schema"]["schema"], llm_judge.BATCH_SCHEMA)
        payload = json.loads(call["messages"][1]["content"])
        self.assertEqual(payload["query"], "vegan options")
        self.assertEqual([r["restaurant_id"] for r in payload["restaurants"]], ["R1", "R2", "R3"])
        by_name = {r["name"]: [v["review_id"] for v in r["reviews"]] for r in payload["restaurants"]}
        self.assertEqual(by_name, {"RG1": [11, 12], "RG2": [21], "RG3": [31, 32]})  # each its own evidence
        self.assertNotIn("gmap_id", json.dumps(payload))

    def test_restaurant_order_is_shuffled_with_a_fixed_seed(self):
        rows = [pool_row("Q1", f"G{k}", [(k, "vegan")]) for k in range(8)]
        orders = []
        for seed in (1, 1, 2):
            client = FakeClient([answer_batch({f"G{k}": (1, [k]) for k in range(8)})])
            self.judge(client, seed=seed).batch(rows)
            orders.append([r["name"] for r in json.loads(client.calls[0]["messages"][1]["content"])["restaurants"]])
        self.assertEqual(orders[0], orders[1])
        self.assertNotEqual(orders[0], orders[2])
        self.assertEqual(sorted(orders[0]), sorted(f"RG{k}" for k in range(8)))

    def malformed(self, mutate):
        """Run a batch whose answer is mutated twice (initial + retry), then single-pair fallbacks."""
        def bad(kwargs):
            data = answer_batch(BATCH_LABELS)(kwargs)
            mutate(data["labels"])
            return data
        singles = [{"label": BATCH_LABELS[g][0], "reason": "single", "review_ids": BATCH_LABELS[g][1]} for g in ("G1", "G2", "G3")]
        client = FakeClient([bad, bad, *singles])
        outcomes = self.judge(client).batch(BATCH_ROWS)
        return client, [r for r, _ in outcomes]

    def test_malformed_batches_fall_back_to_single_calls(self):
        cases = {
            "missing restaurant": lambda labels: labels.pop(),
            "duplicate restaurant": lambda labels: labels.append(dict(labels[0])),
            "unknown restaurant": lambda labels: labels[0].update(restaurant_id="R9"),
            "cites another restaurant's review": lambda labels: next(
                l for l in labels if l["review_ids"] == [11]).update(review_ids=[21]),
            "not a list": lambda labels: labels.clear() or labels.append("oops"),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                client, recs = self.malformed(mutate)
                self.assertEqual(len(client.calls), 2 + 3)  # batch, batch retry, three singles
                self.assertTrue(all(r and r["fallback"] and r["batch_size"] == 1 for r in recs))
                self.assertEqual([r["label"] for r in recs], [2, 0, 1])
                self.assertNotIn("BATCH MODE", client.calls[2]["messages"][0]["content"])

    def test_unparseable_batch_json_falls_back(self):
        singles = [{"label": BATCH_LABELS[g][0], "reason": "single", "review_ids": BATCH_LABELS[g][1]} for g in ("G1", "G2", "G3")]
        client = FakeClient(["{not json", "[]", *singles])
        recs = [r for r, _ in self.judge(client).batch(BATCH_ROWS)]
        self.assertEqual([r["label"] for r in recs], [2, 0, 1])

    def test_rate_limited_batch_does_not_fall_back(self):
        rl = LLMAPIError("rate", category="rate_limit", code="rate_limited", http_status=429)
        with patch.object(llm_judge, "request_json", side_effect=[rl, rl]) as fake:
            outcomes = self.judge(max_retries=1).batch(BATCH_ROWS)
        self.assertEqual(fake.call_count, 2)
        self.assertTrue(all(r is None and e["category"] == "rate_limit" and e["batch_size"] == 3 for r, e in outcomes))

    def test_make_batches_by_query_size_and_tokens(self):
        rows = ([pool_row("Q1", f"A{k}", [(k, "vegan " * 5)]) for k in range(5)]
                + [pool_row("Q2", f"B{k}", [(100 + k, "celiac")]) for k in range(2)])
        system = llm_judge.system_prompt(RUBRIC, batch=True)
        sizes = [[r["gmap_id"] for r in b] for b in llm_judge.make_batches(rows, 2, system, tpm=0)]
        self.assertEqual(sizes, [["A0", "A1"], ["A2", "A3"], ["A4"], ["B0", "B1"]])  # never mixes queries
        one = llm_judge.estimate_tokens(system, llm_judge.batch_payload(rows[:1], ["R1"]))
        tight = llm_judge.make_batches(rows, 8, system, tpm=one + llm_judge.OUTPUT_TOKENS_PER_EXTRA + 60)
        self.assertTrue(all(len(b) <= 2 for b in tight))
        for b in tight:
            if len(b) > 1:
                payload = llm_judge.batch_payload(b, [f"R{k}" for k in range(1, len(b) + 1)])
                self.assertLessEqual(llm_judge.estimate_tokens(system, payload, len(b)), one + 210)

    def test_main_with_batches(self):
        self.write_pool(BATCH_ROWS + [pool_row("Q1", "G4", []), pool_row("Q2", "G5", [(51, "vegan brunch")])])
        client = FakeClient([answer_batch(BATCH_LABELS), {"label": 2, "reason": "ok.", "review_ids": [51]}])
        self.run_judge(client, "--batch-size", "5")
        recs = {r["gmap_id"]: r for r in self.records()}
        self.assertEqual(len(client.calls), 2)  # one batch for Q1, one single for Q2
        self.assertEqual({g: recs[g]["batch_size"] for g in recs}, {"G1": 3, "G2": 3, "G3": 3, "G4": 0, "G5": 1})
        self.assertEqual(recs["G4"]["source"], "rule")


class HelperTests(unittest.TestCase):
    def test_prompt_hash_changes_with_rubric_and_mode(self):
        a, b = llm_judge.system_prompt("rule A"), llm_judge.system_prompt("rule B")
        self.assertNotEqual(llm_judge.prompt_hash(a), llm_judge.prompt_hash(b))
        self.assertEqual(llm_judge.prompt_hash(a), llm_judge.prompt_hash(llm_judge.system_prompt("rule A")))
        batch = llm_judge.system_prompt("rule A", batch=True)
        self.assertNotEqual(llm_judge.prompt_hash(a), llm_judge.prompt_hash(batch, llm_judge.BATCH_SCHEMA))

    def test_larger_model_note(self):
        self.assertIn("openai/gpt-oss-120b", llm_judge.larger_model_note("openai/gpt-oss-20b"))
        self.assertIsNone(llm_judge.larger_model_note("openai/gpt-oss-120b"))
        self.assertIsNone(llm_judge.larger_model_note("meta-models/muse-spark-open"))

    def test_pacer_respects_token_budget_and_uses_actual_usage(self):
        clock, sleeps = [0.0], []

        def fake_sleep(s):
            sleeps.append(s)
            clock[0] += s

        with patch.object(llm_judge, "sleep", side_effect=fake_sleep):
            pacer = llm_judge.Pacer(rpm=100, tpm=7000, clock=lambda: clock[0])
            pacer.wait(4000)
            pacer.adjust(1000)  # the call actually used 1,000 tokens
            pacer.wait(4000)  # fits now
            self.assertEqual(sleeps, [])
            pacer.wait(4000)  # 1,000 + 4,000 + 4,000 > 7,000
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
