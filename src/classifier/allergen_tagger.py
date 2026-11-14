"""Allergen-mention tagger: which FDA major allergens a short text mentions (multi-label).

Trained on the HuggingFace adisyonist/14-allergens-v2 menu-item dataset (MIT; 294,500 rows, 48
languages), mapped onto the FDA 9: gluten -> wheat (approximate: gluten also covers barley/rye/oats),
crustaceans -> shellfish, eggs -> egg; the EU-only labels (celery, mustard, sulphites, lupin,
molluscs) are dropped, and rows left with no FDA label stay as negatives. Texts are deduplicated
case-insensitively (label sets of duplicates are unioned) and split by lowercased text, so no
text appears in both train and test.

Model: character n-gram TF-IDF (char_wb, 2-5; language-agnostic, no download) and one-vs-rest
logistic regression with balanced class weights.

Scope: it tags *mentions* of allergens (a dish contains peanuts), not safety or accommodation, and
it is trained on menu items, so it is expected to do worse on review sentences (measured by
src/classifier/stance.py once review sentences are labeled).

Train, evaluate on the held-out menu split and save to data/processed/allergen_tagger/:
    HF_DATASETS_OFFLINE=1 python -m src.classifier.allergen_tagger --train
Tag text:
    python -m src.classifier.allergen_tagger "peanut noodles with sesame"
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from src.config import DATA_DIR

DATASET = "adisyonist/14-allergens-v2"
FDA_LABELS = ["milk", "egg", "fish", "shellfish", "tree_nuts", "wheat", "peanuts", "soy", "sesame"]
HF_TO_FDA = {"milk": "milk", "eggs": "egg", "fish": "fish", "crustaceans": "shellfish", "tree_nuts": "tree_nuts",
             "gluten": "wheat", "peanuts": "peanuts", "soy": "soy", "sesame": "sesame"}
DROPPED = {"celery", "mustard", "sulphites", "lupin", "molluscs"}
MODEL_DIR = DATA_DIR / "processed" / "allergen_tagger"
SEED = 6165
TEST_SIZE = 0.2
VECTORIZER = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300_000,
                  lowercase=True, dtype=np.float32)


def map_labels(hf_labels):
    """HF EU-14 labels -> sorted FDA labels (EU-only labels dropped). Unknown labels raise."""
    out = set()
    for label in hf_labels:
        if label in HF_TO_FDA:
            out.add(HF_TO_FDA[label])
        elif label not in DROPPED:
            raise ValueError(f"unknown allergen label {label!r}")
    return sorted(out)


def prepare(rows):
    """[(text, hf_labels)] -> (texts, Y, stats). Case-insensitive dedup; duplicate label sets are unioned."""
    merged, first_text, conflicts = {}, {}, 0
    for text, labels in rows:
        key = text.strip().lower()
        if not key:
            continue
        fda = set(map_labels(labels))
        if key in merged:
            conflicts += merged[key] != fda
            merged[key] |= fda
        else:
            merged[key], first_text[key] = fda, text.strip()
    keys = list(merged)
    Y = np.array([[label in merged[k] for label in FDA_LABELS] for k in keys], dtype=np.int8)
    stats = {"input_rows": len(rows), "unique_texts": len(keys), "duplicates_removed": len(rows) - len(keys),
             "conflicting_duplicates": conflicts, "no_fda_label": int((Y.sum(axis=1) == 0).sum())}
    return [first_text[k] for k in keys], Y, stats


def split(texts, Y, test_size=TEST_SIZE, seed=SEED):
    """Random split of unique lowercased texts: no lowercased text in both parts."""
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    n_test = int(round(len(idx) * test_size))
    test, train = sorted(idx[:n_test]), sorted(idx[n_test:])
    pick = lambda ids: ([texts[i] for i in ids], Y[ids])
    return pick(train), pick(test)


class AllergenTagger:
    def __init__(self, vectorizer=None, model=None, threshold=0.5):
        self.vectorizer, self.model, self.threshold = vectorizer, model, threshold

    def fit(self, texts, Y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        self.vectorizer = TfidfVectorizer(**VECTORIZER)
        X = self.vectorizer.fit_transform(texts)
        # One label at a time (n_jobs=1). Process workers get the large sparse matrix as a read-only
        # memmap, which liblinear rejects ("WRITEBACKIFCOPY base is read-only"); thread workers
        # crashed (segfault). Sequential fitting is slower but reliable.
        self.model = OneVsRestClassifier(LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000),
                                         n_jobs=1).fit(X, Y)
        return self

    def predict_proba(self, texts):
        return self.model.predict_proba(self.vectorizer.transform(list(texts)))

    def predict(self, texts):
        """Binary matrix (n x 9) at self.threshold."""
        return (self.predict_proba(texts) >= self.threshold).astype(np.int8)

    def tag(self, texts):
        """FDA label lists per text."""
        return [[FDA_LABELS[j] for j in np.flatnonzero(row)] for row in self.predict(texts)]

    def save(self, path=MODEL_DIR, meta=None):
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model, "threshold": self.threshold,
                     "labels": FDA_LABELS}, path / "tagger.joblib")
        (path / "meta.json").write_text(json.dumps(meta or {}, indent=2))

    @classmethod
    def load(cls, path=MODEL_DIR):
        import joblib
        blob = joblib.load(Path(path) / "tagger.joblib")  # our own file; pickle is trusted
        if blob["labels"] != FDA_LABELS:
            raise ValueError("saved tagger has a different label set")
        return cls(blob["vectorizer"], blob["model"], blob["threshold"])


def evaluate(Y_true, Y_pred, labels=FDA_LABELS):
    """Per-label precision/recall/F1/support, macro and micro averages."""
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    per = {}
    for j, label in enumerate(labels):
        t, p = Y_true[:, j].astype(bool), Y_pred[:, j].astype(bool)
        tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[label] = {"precision": prec, "recall": rec, "f1": f1, "support": int(t.sum())}
    tp = int((Y_true.astype(bool) & Y_pred.astype(bool)).sum())
    fp = int((~Y_true.astype(bool) & Y_pred.astype(bool)).sum())
    fn = int((Y_true.astype(bool) & ~Y_pred.astype(bool)).sum())
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    return {"per_label": per, "macro_f1": float(np.mean([v["f1"] for v in per.values()])),
            "micro_f1": 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0}


def load_hf_rows():
    from datasets import load_dataset
    d = load_dataset(DATASET)["train"]
    return list(zip(d["text"], d["allergens"]))


def print_report(report, title):
    print(f"\n{title}\n{'label':10} {'P':>6} {'R':>6} {'F1':>6} {'support':>8}")
    for label, m in report["per_label"].items():
        print(f"{label:10} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['support']:8,}")
    print(f"macro-F1 {report['macro_f1']:.3f}  micro-F1 {report['micro_f1']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="FDA-9 allergen-mention tagger (char n-gram TF-IDF + logistic regression)")
    ap.add_argument("texts", nargs="*", help="texts to tag with the saved model")
    ap.add_argument("--train", action="store_true", help="train on the HF dataset, evaluate, and save")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = ap.parse_args(argv)
    if args.train:
        t0 = time.time()
        texts, Y, stats = prepare(load_hf_rows())
        (tr_x, tr_y), (te_x, te_y) = split(texts, Y)
        print(f"Data: {stats}; train {len(tr_x):,}, test {len(te_x):,} (split by lowercased text)", flush=True)
        t = time.time()
        tagger = AllergenTagger().fit(tr_x, tr_y)
        fit_s = time.time() - t
        report = evaluate(te_y, tagger.predict(te_x))
        print_report(report, "Held-out menu test set")
        meta = {"dataset": DATASET, "labels": FDA_LABELS, "mapping": HF_TO_FDA, "dropped": sorted(DROPPED),
                "split": {"test_size": TEST_SIZE, "seed": SEED, "by": "lowercased text"}, "data": stats,
                "train_size": len(tr_x), "test_size_n": len(te_x), "vectorizer": {k: str(v) for k, v in VECTORIZER.items()},
                "model": "OneVsRest LogisticRegression(class_weight=balanced, liblinear)", "threshold": 0.5,
                "fit_seconds": round(fit_s, 1), "test_report": report,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tagger.save(args.model_dir, meta)
        print(f"\nSaved to {args.model_dir} (fit {fit_s:.0f}s, total {time.time() - t0:.0f}s)")
    if args.texts:
        tagger = AllergenTagger.load(args.model_dir)
        for text, labels in zip(args.texts, tagger.tag(args.texts)):
            print(f"{labels}  {text}")
    if not args.train and not args.texts:
        ap.print_help(sys.stderr)


if __name__ == "__main__":
    main()
