"""Agreement statistics and the agreement CLI (synthetic labels)."""

import csv
import io
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from eval import agreement
from eval.stats import cohen_kappa, confusion_matrix, krippendorff_alpha, raw_agreement


class StatsTests(unittest.TestCase):
    def test_krippendorff_published_example(self):
        # Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 4 observers x 12 units.
        N = None
        a = [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N]
        b = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3]
        c = [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N]
        d = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N]
        units = list(zip(a, b, c, d))
        self.assertAlmostEqual(krippendorff_alpha(units, "nominal"), 0.743, places=3)
        self.assertAlmostEqual(krippendorff_alpha(units, "ordinal"), 0.815, places=3)
        self.assertAlmostEqual(krippendorff_alpha(units, "interval"), 0.849, places=3)

    def test_alpha_edge_cases(self):
        self.assertEqual(krippendorff_alpha([[0, 0], [1, 1], [2, 2]]), 1.0)
        self.assertTrue(math.isnan(krippendorff_alpha([[1, None], [None, 2]])))  # nothing pairable
        with self.assertRaises(ValueError):
            krippendorff_alpha([[0, 1], [1, 2]], "ratio")

    def test_kappa_matches_sklearn(self):
        rng = random.Random(3)
        for _ in range(5):
            a = [rng.choice([0, 1, 2]) for _ in range(150)]
            b = [x if rng.random() < 0.7 else rng.choice([0, 1, 2]) for x in a]
            for w in (None, "linear", "quadratic"):
                self.assertAlmostEqual(cohen_kappa(a, b, w), cohen_kappa_score(a, b, weights=w), places=12)

    def test_kappa_and_agreement_basics(self):
        self.assertEqual(cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2], "quadratic"), 1.0)
        self.assertEqual(raw_agreement([0, 1, 2, 2], [0, 1, 1, 2]), 0.75)
        self.assertEqual(confusion_matrix([0, 2, 2], [0, 1, 2]), [[1, 0, 0], [0, 0, 0], [0, 1, 1]])
        with self.assertRaises(ValueError):
            cohen_kappa([0], [0, 1])
        with self.assertRaises(ValueError):
            cohen_kappa([0, 1], [1, 0], "cubic")


LABELS = {  # sample_id -> (ana, ben, cai); "" = not labeled
    "S001": (2, 2, 2), "S002": (0, 0, 0), "S003": (1, 2, 1), "S004": (0, 2, 1),
    "S005": (1, 1, 1), "S006": (2, 1, 2), "S007": (0, 0, ""), "S008": (1, 0, 0),
}


class AgreementCliTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.files = []
        for j, judge in enumerate(("ana", "ben", "cai")):
            path = self.dir / f"labels_{judge}.csv"
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["sample_id", "query_id", "query", "name", "label", "note"])
                w.writeheader()
                for sid, labels in reversed(list(LABELS.items())) if j == 1 else LABELS.items():
                    w.writerow({"sample_id": sid, "query_id": "Q" + sid[-1], "query": "q", "name": f"N{sid}",
                                "label": labels[j], "note": "unsure" if (judge == "ben" and sid == "S004") else ""})
            self.files.append(path)
        with open(self.dir / "sample_key.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "query_id", "gmap_id", "category", "single_system", "evidence_status"])
            for sid in LABELS:
                w.writerow([sid, "Q" + sid[-1], f"G{sid}", "allergy" if sid in ("S003", "S004") else "single_diet", 0, "retrieved"])

    def test_report_numbers(self):
        report = agreement.analyze({agreement.judge_name(p): agreement.read_judge(p) for p in self.files})
        ana = [LABELS[s][0] for s in LABELS]
        ben = [LABELS[s][1] for s in LABELS]
        pair = report["pairwise"][("ana", "ben")]
        self.assertEqual(pair["n"], 8)
        self.assertAlmostEqual(pair["kappa_quadratic"], cohen_kappa_score(ana, ben, weights="quadratic"))
        self.assertEqual(report["pairwise"][("ana", "cai")]["n"], 7)  # cai skipped S007
        self.assertEqual(report["per_judge"]["cai"]["missing"], 1)
        units = [[v if v != "" else None for v in LABELS[s]] for s in sorted(LABELS)]
        self.assertAlmostEqual(report["alpha_ordinal"], krippendorff_alpha(units, "ordinal"))
        self.assertEqual(report["n_complete"], 7)
        self.assertAlmostEqual(report["unanimous"], 3 / 7)
        dis = [r["sample_id"] for r in report["disagreements"]]
        self.assertEqual(dis[0], "S004")  # spread 2 first
        self.assertEqual(set(dis), {"S003", "S004", "S006", "S008"})

    def test_cli_writes_disagreements_and_template(self):
        with redirect_stdout(io.StringIO()) as out:
            agreement.main([str(p) for p in self.files])
        self.assertIn("Krippendorff's alpha", out.getvalue())
        with open(self.dir / "disagreements.csv", newline="") as f:
            dis = list(csv.DictReader(f))
        self.assertEqual(dis[0]["sample_id"], "S004")
        self.assertEqual(dis[0]["note_ben"], "unsure")
        with open(self.dir / "adjudicated_template.csv", newline="") as f:
            tpl = {r["sample_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(len(tpl), 8)
        self.assertEqual(tpl["S001"]["adjudicated_label"], "2")  # unanimous: pre-filled
        self.assertEqual(tpl["S003"]["adjudicated_label"], "")  # disagreement: blank
        self.assertEqual(tpl["S003"]["majority"], "1")
        self.assertEqual(tpl["S004"]["majority"], "")  # 0/2/1: no majority
        self.assertEqual(tpl["S007"]["adjudicated_label"], "")  # not labeled by all
        self.assertEqual(tpl["S003"]["gmap_id"], "GS003")
        self.assertEqual(tpl["S003"]["category"], "allergy")

    def test_invalid_label_is_reported(self):
        with open(self.files[0], "a", newline="") as f:
            csv.writer(f).writerow(["S009", "Q9", "q", "N", "3", ""])
        with self.assertRaisesRegex(ValueError, "S009='3'"):
            agreement.read_judge(self.files[0])


if __name__ == "__main__":
    unittest.main()
