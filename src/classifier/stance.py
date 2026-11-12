"""Stance classifier for review sentences, plus the allergen tagger's accuracy on reviews.

Input: the labeled sentence sheets from eval/make_sentence_sheets.py (sentences_<judge>.csv), with
stance in {accommodation, mention, negative, not_about_allergy} and allergens_mentioned (FDA labels,
";"-separated; pre-filled by the tagger and corrected by the judge).

1. Agreement on the shared sentences: raw agreement and Krippendorff's alpha (nominal) for stance.
2. Gold labels: one per sentence. Shared sentences take the majority stance (ties are dropped and
   counted) and, per allergen, the majority of judges; others take their single judge's labels.
3. Stance classifier: logistic regression (balanced class weights) on all-MiniLM-L6-v2 sentence
   embeddings, stratified k-fold cross-validation (k = min(5, smallest class)); per-class
   precision/recall/F1 and macro-F1 from out-of-fold predictions. Classes with fewer than 2 gold
   sentences are left out and reported. A final model is fit on all gold sentences and saved.
4. Allergen tagger on the same sentences vs the gold allergens: per-label P/R/F1 and macro-F1,
   compared with its held-out menu macro-F1 (the menu -> review drop).

Run from the repo root (tomorrow, once the sheets are labeled):
    HF_HUB_OFFLINE=1 python -m src.classifier.stance eval/labels/sentences_kang.csv \\
        eval/labels/sentences_laasya.csv eval/labels/sentences_pujitha.csv
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.classifier.allergen_tagger import FDA_LABELS, MODEL_DIR as TAGGER_DIR, evaluate as evaluate_tags
from src.config import DATA_DIR
from eval.stats import krippendorff_alpha, raw_agreement

STANCES = ["accommodation", "mention", "negative", "not_about_allergy"]
OUT_DIR = DATA_DIR / "processed" / "stance"
SEED = 6165


def judge_name(path):
    stem = Path(path).stem
    return stem[len("sentences_"):] if stem.startswith("sentences_") else stem


def parse_allergens(cell, where):
    labels = [x.strip() for x in (cell or "").replace(",", ";").split(";") if x.strip()]
    bad = [x for x in labels if x not in FDA_LABELS]
    if bad:
        raise ValueError(f"{where}: unknown allergen label(s) {bad}; use {FDA_LABELS}")
    return sorted(set(labels))


def read_sheets(paths):
    """{sentence_id: {"sentence": str, "labels": {judge: (stance, allergens)}}}; unlabeled rows are skipped."""
    items = defaultdict(lambda: {"sentence": None, "labels": {}})
    for path in paths:
        judge = judge_name(path)
        with open(path, newline="", encoding="utf-8") as f:
            for n, row in enumerate(csv.DictReader(f), 2):
                stance = (row.get("stance") or "").strip().lower()
                if not stance:
                    continue
                if stance not in STANCES:
                    raise ValueError(f"{path}:{n}: stance must be one of {STANCES}, got {stance!r}")
                item = items[row["sentence_id"]]
                item["sentence"] = row["sentence"]
                item["labels"][judge] = (stance, parse_allergens(row.get("allergens_mentioned"), f"{path}:{n}"))
    return dict(items)


def agreement(items):
    """Stance agreement on sentences labeled by 2+ judges."""
    multi = {sid: it for sid, it in items.items() if len(it["labels"]) >= 2}
    judges = sorted({j for it in multi.values() for j in it["labels"]})
    units = [[STANCES.index(it["labels"][j][0]) if j in it["labels"] else None for j in judges] for it in multi.values()]
    pairs = [(a, b) for it in multi.values() for a, b in zip(list(it["labels"].values()), list(it["labels"].values())[1:])]
    return {"n_shared": len(multi), "judges": judges,
            "alpha_nominal": krippendorff_alpha(units, "nominal") if units else float("nan"),
            "unanimous": sum(len({s for s, _ in it["labels"].values()}) == 1 for it in multi.values()) / len(multi)
            if multi else float("nan"),
            "pairwise_raw": raw_agreement([a[0] for a, _ in pairs], [b[0] for _, b in pairs]) if pairs else float("nan")}


def gold(items):
    """(sentence_ids, sentences, stances, allergen matrix, n_ties_dropped)."""
    ids, texts, stances, rows, ties = [], [], [], [], 0
    for sid in sorted(items):
        labels = list(items[sid]["labels"].values())
        counts = Counter(s for s, _ in labels).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            ties += 1
            continue
        allergens = {a for a in FDA_LABELS if sum(a in al for _, al in labels) * 2 > len(labels)}
        ids.append(sid)
        texts.append(items[sid]["sentence"])
        stances.append(counts[0][0])
        rows.append([int(a in allergens) for a in FDA_LABELS])
    return ids, texts, stances, np.array(rows, dtype=np.int8).reshape(-1, len(FDA_LABELS)), ties


def cross_validate(X, y, seed=SEED):
    """Out-of-fold stance predictions and a per-class report; drops classes with < 2 examples."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    counts = Counter(y)
    keep = [i for i, c in enumerate(y) if counts[c] >= 2]
    dropped = sorted(c for c, n in counts.items() if n < 2)
    Xk, yk = X[keep], [y[i] for i in keep]
    k = min(5, min(Counter(yk).values()))
    if len(set(yk)) < 2 or k < 2:
        raise ValueError(f"not enough labeled sentences per class to cross-validate: {dict(counts)}")
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    pred = cross_val_predict(clf, Xk, yk, cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=seed))
    labels = [s for s in STANCES if s in set(yk)]
    report = classification_report(yk, pred, labels=labels, output_dict=True, zero_division=0)
    return {"folds": k, "n": len(yk), "dropped_classes": dropped,
            "per_class": {c: {m: report[c][m] for m in ("precision", "recall", "f1-score", "support")} for c in labels},
            "macro_f1": report["macro avg"]["f1-score"], "accuracy": report["accuracy"]}


def run(paths, embedder=None, tagger=None, menu_macro_f1=None, out_dir=OUT_DIR, seed=SEED):
    items = read_sheets(paths)
    agree = agreement(items)
    ids, texts, stances, Y, ties = gold(items)
    if embedder is None:
        from src.retrieval.semantic import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()
    X = embedder.encode(texts)
    cv = cross_validate(X, stances, seed)
    from sklearn.linear_model import LogisticRegression
    final = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X, stances)
    if tagger is None:
        from src.classifier.allergen_tagger import AllergenTagger
        tagger = AllergenTagger.load()
    if menu_macro_f1 is None:
        meta = TAGGER_DIR / "meta.json"
        menu_macro_f1 = json.loads(meta.read_text())["test_report"]["macro_f1"] if meta.exists() else None
    tag_report = evaluate_tags(Y, tagger.predict(texts))
    result = {"labeled_sentences": len(items), "gold_sentences": len(ids), "ties_dropped": ties,
              "stance_distribution": dict(Counter(stances)), "agreement": agree, "stance_cv": cv,
              "tagger_on_reviews": tag_report, "tagger_menu_macro_f1": menu_macro_f1,
              "menu_to_review_drop": (menu_macro_f1 - tag_report["macro_f1"]) if menu_macro_f1 is not None else None}
    if out_dir is not None:
        import joblib
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": final, "classes": list(final.classes_), "embedder": getattr(embedder, "model_name", None)},
                    out_dir / "stance.joblib")
        (out_dir / "report.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def print_report(r):
    a = r["agreement"]
    print(f"{r['labeled_sentences']} labeled sentences -> {r['gold_sentences']} gold ({r['ties_dropped']} ties dropped)")
    print(f"Stance agreement on {a['n_shared']} shared sentences: alpha (nominal) {a['alpha_nominal']:.3f}, "
          f"unanimous {a['unanimous']:.2f}")
    cv = r["stance_cv"]
    print(f"\nStance classifier ({cv['folds']}-fold CV, n={cv['n']}; dropped classes: {cv['dropped_classes'] or 'none'})")
    for c, m in cv["per_class"].items():
        print(f"  {c:18} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1-score']:.3f}  n {int(m['support'])}")
    print(f"  macro-F1 {cv['macro_f1']:.3f}  accuracy {cv['accuracy']:.3f}")
    t = r["tagger_on_reviews"]
    print(f"\nAllergen tagger on review sentences: macro-F1 {t['macro_f1']:.3f} (menu test: "
          f"{r['tagger_menu_macro_f1'] if r['tagger_menu_macro_f1'] is None else round(r['tagger_menu_macro_f1'], 3)}; "
          f"drop {r['menu_to_review_drop'] if r['menu_to_review_drop'] is None else round(r['menu_to_review_drop'], 3)})")
    for label, m in t["per_label"].items():
        print(f"  {label:10} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  n {m['support']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stance classifier and tagger-on-reviews evaluation")
    ap.add_argument("sheets", nargs="+", type=Path, help="labeled sentences_<judge>.csv files")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    missing = [p for p in args.sheets if not p.exists()]
    if missing:
        sys.exit(f"Missing: {missing}")
    print_report(run(args.sheets, out_dir=args.out_dir))
    print(f"\nSaved model and report to {args.out_dir}")


if __name__ == "__main__":
    main()
