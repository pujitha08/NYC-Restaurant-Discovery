"""Judge validation tests on synthetic adjudicated and LLM labels."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from eval import validate_judge

# sample_id, query_id, gmap_id, category, human, llm (None = no LLM label), source
DATA = [
    ("S001", "Q19", "G1", "allergy", 0, 2, "llm"),  # LLM too generous on an allergy query
    ("S002", "Q19", "G2", "allergy", 2, 2, "llm"),
    ("S003", "Q21", "G3", "allergy", 1, 0, "llm"),
    ("S004", "Q01", "G4", "single_diet", 2, 2, "llm"),
    ("S005", "Q01", "G5", "single_diet", 1, 2, "llm"),
    ("S006", "Q13", "G6", "mixed_group", 0, 0, "rule"),
    ("S007", "Q13", "G7", "mixed_group", 2, None, "llm"),  # not labeled by the LLM
]


class ValidateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.adj, self.llm, self.out = self.dir / "adjudicated.csv", self.dir / "llm.jsonl", self.dir / "dis.csv"
        with open(self.adj, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "query_id", "gmap_id", "category", "query", "name", "adjudicated_label",
                        "adjudication_note"])
            for sid, qid, gid, cat, human, _, _ in DATA:
                w.writerow([sid, qid, gid, cat, f"query {qid}", f"N{gid}", human, ""])
            w.writerow(["S008", "Q02", "G8", "single_diet", "q", "n", "", ""])  # not adjudicated yet
        with open(self.llm, "w") as f:
            for sid, qid, gid, cat, human, llm, src in DATA:
                if llm is not None:
                    f.write(json.dumps({"query_id": qid, "gmap_id": gid, "label": llm, "reason": f"reason {sid}",
                                        "review_ids": [1], "source": src}) + "\n")

    def report(self):
        adjudicated, blank = validate_judge.read_adjudicated(self.adj)
        return validate_judge.compare(adjudicated, validate_judge.read_llm(self.llm)), blank

    def test_confusion_kappa_and_direction(self):
        report, blank = self.report()
        self.assertEqual(blank, ["S008"])
        self.assertEqual(report["missing"], ["S007"])
        all_ = report["groups"]["all"]
        self.assertEqual(all_["n"], 6)
        self.assertEqual(all_["confusion"], [[1, 0, 1], [1, 0, 1], [0, 0, 2]])  # rows human, cols LLM
        self.assertAlmostEqual(all_["accuracy"], 3 / 6)
        self.assertEqual((all_["llm_higher"], all_["llm_lower"]), (2, 1))

    def test_allergy_breakout(self):
        report, _ = self.report()
        allergy, other = report["groups"]["allergy"], report["groups"]["other"]
        self.assertEqual((allergy["n"], other["n"]), (3, 3))
        self.assertEqual(allergy["confusion"], [[0, 0, 1], [1, 0, 0], [0, 0, 1]])
        self.assertEqual(allergy["llm_higher"], 1)
        self.assertEqual(report["by_source"]["rule"]["n"], 1)

    def test_disagreements_put_allergy_and_large_gaps_first(self):
        report, _ = self.report()
        self.assertEqual([p["sample_id"] for p in report["disagreements"]], ["S001", "S003", "S005"])

    def test_cli_writes_disagreements(self):
        with redirect_stdout(io.StringIO()) as out:
            validate_judge.main(["--llm", str(self.llm), "--adjudicated", str(self.adj), "--out", str(self.out)])
        self.assertIn("== allergy: n=3", out.getvalue())
        with open(self.out, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["sample_id"], "S001")
        self.assertEqual(rows[0]["llm_reason"], "reason S001")

    def test_invalid_adjudicated_label(self):
        with open(self.adj, "a", newline="") as f:
            csv.writer(f).writerow(["S009", "Q1", "G9", "allergy", "q", "n", "5", ""])
        with self.assertRaisesRegex(ValueError, "S009='5'"):
            validate_judge.read_adjudicated(self.adj)


if __name__ == "__main__":
    unittest.main()
