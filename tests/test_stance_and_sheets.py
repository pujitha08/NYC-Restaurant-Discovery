"""Sentence sheets and the stance classifier on synthetic data (fake embedder and tagger; no model or data)."""

import csv
import io
import json
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from eval import make_sentence_sheets as ms
from src.classifier import stance
from src.classifier.allergen_tagger import FDA_LABELS


class FakeEmbedder:
    model_name = "fake"

    def encode(self, texts):
        out = np.zeros((len(texts), 32), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % 32] += 1
        return out


class FakeTagger:
    """Tags 'peanut' -> peanuts and 'shrimp' -> shellfish."""

    def tag(self, texts):
        return [[l for l, w in (("peanuts", "peanut"), ("shellfish", "shrimp")) if w in t.lower()] for t in texts]

    def predict(self, texts):
        return np.array([[int(l in tags) for l in FDA_LABELS] for tags in self.tag(texts)], dtype=np.int8)


TEMPLATES = {
    "accommodation": "staff were great with my peanut allergy and checked the kitchen {i}",
    "mention": "the peanut noodles were tasty and huge portion {i}",
    "negative": "they refused my peanut allergy request and i got sick {i}",
    "not_about_allergy": "nice vegan brunch spot with friendly staff {i}",
}


def write_sheet(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ms.COLUMNS)
        w.writeheader()
        w.writerows(rows)


class StanceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        shared = [("S1", TEMPLATES["accommodation"].format(i=0)), ("S2", TEMPLATES["negative"].format(i=0)),
                  ("S3", TEMPLATES["mention"].format(i=0))]
        own = {j: [(f"{j}{k}", TEMPLATES[s].format(i=k)) for k, s in enumerate(stance.STANCES * 4)]
               for j in ("ana", "ben", "cai")}
        shared_labels = {"ana": ["accommodation", "negative", "mention"], "ben": ["accommodation", "negative", "negative"],
                         "cai": ["accommodation", "mention", "mention"]}  # S1 unanimous; S2 negative 2-1; S3 mention 2-1
        self.paths = []
        for j in ("ana", "ben", "cai"):
            rows = [{"sentence_id": sid, "sentence": s, "allergens_mentioned": "peanuts", "stance": st, "note": ""}
                    for (sid, s), st in zip(shared, shared_labels[j])]
            for (sid, s), st in zip(own[j], stance.STANCES * 4):
                rows.append({"sentence_id": sid, "sentence": s,
                             "allergens_mentioned": "" if st == "not_about_allergy" else "peanuts", "stance": st, "note": ""})
            rows.append({"sentence_id": f"{j}-blank", "sentence": "unlabeled row", "allergens_mentioned": "",
                         "stance": "", "note": ""})
            path = self.dir / f"sentences_{j}.csv"
            write_sheet(path, rows)
            self.paths.append(path)

    def test_end_to_end(self):
        r = stance.run(self.paths, embedder=FakeEmbedder(), tagger=FakeTagger(), menu_macro_f1=0.8, out_dir=self.dir / "out")
        self.assertEqual(r["labeled_sentences"], 3 + 3 * 16)  # blank rows skipped
        self.assertEqual(r["agreement"]["n_shared"], 3)
        self.assertAlmostEqual(r["agreement"]["unanimous"], 1 / 3)
        self.assertEqual(r["ties_dropped"], 0)
        self.assertEqual(set(r["stance_cv"]["per_class"]), set(stance.STANCES))
        self.assertGreaterEqual(r["stance_cv"]["macro_f1"], 0.0)
        self.assertEqual(r["stance_cv"]["folds"], 5)
        peanuts = r["tagger_on_reviews"]["per_label"]["peanuts"]
        self.assertEqual(peanuts["recall"], 1.0)  # every gold peanut sentence says "peanut"
        self.assertAlmostEqual(r["menu_to_review_drop"], 0.8 - r["tagger_on_reviews"]["macro_f1"])
        self.assertTrue((self.dir / "out" / "stance.joblib").exists())
        saved = json.loads((self.dir / "out" / "report.json").read_text())
        self.assertEqual(saved["gold_sentences"], r["gold_sentences"])
        with redirect_stdout(io.StringIO()) as out:
            stance.print_report(r)
        self.assertIn("macro-F1", out.getvalue())

    def test_majority_and_ties(self):
        items = {"X": {"sentence": "s", "labels": {"a": ("negative", ["peanuts"]), "b": ("mention", ["peanuts", "soy"])}},
                 "Y": {"sentence": "t", "labels": {"a": ("negative", ["soy"]), "b": ("negative", []), "c": ("mention", ["soy"])}}}
        ids, _, stances, Y, ties = stance.gold(items)
        self.assertEqual((ids, stances, ties), (["Y"], ["negative"], 1))  # X is a 1-1 tie
        self.assertEqual([FDA_LABELS[j] for j in np.flatnonzero(Y[0])], ["soy"])  # 2 of 3 judges

    def test_invalid_values(self):
        path = self.dir / "sentences_bad.csv"
        write_sheet(path, [{"sentence_id": "Z", "sentence": "x", "allergens_mentioned": "", "stance": "positive", "note": ""}])
        with self.assertRaisesRegex(ValueError, "stance must be one of"):
            stance.read_sheets([path])
        write_sheet(path, [{"sentence_id": "Z", "sentence": "x", "allergens_mentioned": "kiwi", "stance": "mention", "note": ""}])
        with self.assertRaisesRegex(ValueError, "unknown allergen"):
            stance.read_sheets([path])


class SheetTests(unittest.TestCase):
    def records(self):
        texts = []
        for i in range(40):
            texts += [f"My kid has a peanut allergy and they were not careful at all, visit {i}.",
                      f"The peanut sauce on the noodles was delicious tonight, visit {i}.",
                      f"Great vegan options for our whole group at dinner, visit {i}.",
                      f"They gave me shrimp even though I said shellfish allergy, visit {i}.",
                      f"Short one {i}."]
        return [{"text": t, "review_id": k, "gmap_id": f"G{k % 7}"} for k, t in enumerate(texts)]

    def test_strata_assignment_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            with redirect_stdout(io.StringIO()):
                key, files, seen = ms.main(["--judges", "ana", "ben", "cai", "--out-dir", d], tagger=FakeTagger(),
                                           records=self.records())
            self.assertGreater(seen["peanuts+context"], 0)
            self.assertGreater(seen["peanuts-plain"], 0)
            self.assertGreater(seen["allergy+negation"], 0)
            self.assertEqual(seen["sesame-plain"], 0)
            ids = [k["sentence_id"] for k in key]
            self.assertEqual(len(ids), len(set(ids)))
            shared = {k["sentence_id"] for k in key if k["shared"] == 1}
            for j in ("ana", "ben", "cai"):
                with open(Path(d) / f"sentences_{j}.csv", newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(list(rows[0]), ms.COLUMNS)
                self.assertTrue(shared <= {r["sentence_id"] for r in rows})
                self.assertTrue(all(r["stance"] == "" and r["note"] == "" for r in rows))
                self.assertTrue(all(ms.MIN_WORDS <= len(r["sentence"].split()) <= ms.MAX_WORDS for r in rows))
                peanut = [r for r in rows if "peanut" in r["sentence"].lower()]
                self.assertTrue(peanut and all("peanuts" in r["allergens_mentioned"] for r in peanut))
            own = [k for k in key if k["shared"] == 0]
            self.assertEqual({k["assigned_to"] for k in own} <= {"ana", "ben", "cai"}, True)
            with self.assertRaises(SystemExit):  # refuses to overwrite
                ms.main(["--judges", "ana", "--out-dir", d], tagger=FakeTagger(), records=self.records())

    def test_negation_stratum_requires_negation(self):
        inc, _, _ = ms.COMPILED["allergy+negation"]
        self.assertTrue(all(p.search("They said no, my peanut allergy was not a concern") for p in inc))
        self.assertFalse(all(p.search("They handled my peanut allergy perfectly") for p in inc))


if __name__ == "__main__":
    unittest.main()
