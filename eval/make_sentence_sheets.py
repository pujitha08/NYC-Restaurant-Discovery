"""Review-sentence labeling sheets for the stance classifier and the allergen tagger's review accuracy.

Samples ~200 sentences from data/processed/reviews_clean.jsonl that mention allergy or diet terms,
stratified across the FDA allergens (half with allergy-context words such as "allergy", "-free",
"safe", half without, so plain dish mentions are included), diets, and allergy sentences with a
negation. Fixed seed; sentences are deduplicated case-insensitively and kept to 6-60 words.

Writes (default eval/labels/):
  sentences_<judge>.csv  per judge: SHARED sentences labeled by everyone (for agreement) plus their
                         own PER_JUDGE sentences, in their own shuffled order. Columns: sentence_id,
                         sentence, allergens_mentioned (pre-filled by the allergen tagger as a
                         suggestion; ";"-separated FDA labels), stance (empty), note (empty).
  sentences_key.csv      sentence_id, stratum, review_id, gmap_id, shared, assigned_to (not for judges)
Existing judge files are never overwritten unless --force is given.

Run from the repo root (needs the saved tagger, python -m src.classifier.allergen_tagger --train):
    python -m eval.make_sentence_sheets --judges kang laasya pujitha
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

from src.config import DATA_DIR

REVIEWS_PATH = DATA_DIR / "processed" / "reviews_clean.jsonl"
OUT_DIR = Path(__file__).parent / "labels"
SEED = 6165
SHARED, PER_JUDGE = 20, 60
MIN_WORDS, MAX_WORDS = 6, 60
COLUMNS = ["sentence_id", "sentence", "allergens_mentioned", "stance", "note"]
KEY_COLUMNS = ["sentence_id", "stratum", "review_id", "gmap_id", "shared", "assigned_to"]
STANCES = ["accommodation", "mention", "negative", "not_about_allergy"]

ALLERGENS = {
    "milk": r"\b(?:milk|dairy|lactose|cheese|butter|cream)\b",
    "egg": r"\beggs?\b",
    "fish": r"\bfish\b",
    "shellfish": r"\b(?:shellfish|shrimps?|crabs?|lobsters?|prawns?|crustaceans?)\b",
    "tree_nuts": r"\b(?:tree[- ]?nuts?|almonds?|walnuts?|cashews?|pecans?|pistachios?|hazelnuts?|nuts?)\b",
    "wheat": r"\b(?:wheat|gluten|celiac|coeliac)\b",
    "peanuts": r"\bpeanuts?\b",
    "soy": r"\b(?:soy|soya|tofu)\b",
    "sesame": r"\b(?:sesame|tahini)\b",
}
DIETS = {"vegan": r"\bvegans?\b", "vegetarian": r"\bvegetarians?\b", "halal": r"\bhalal\b",
         "kosher": r"\bkosher\b", "gluten_free": r"\bgluten[- ]?free\b"}
CONTEXT = r"\b(?:allerg\w*|intoleran\w*|anaphyla\w*|epi-?pens?|reactions?|cross[- ]?(?:contam\w*|contact)|\w+[- ]free|safe(?:ly)?)\b"
NEGATION = r"\b(?:not|no|never|\w+n't|refused|unable|without)\b"
# stratum -> (patterns that must all match, patterns that must not match, number of sentences)
STRATA = {**{f"{a}+context": ([p, CONTEXT], [], 7) for a, p in ALLERGENS.items()},
          **{f"{a}-plain": ([p], [CONTEXT], 7) for a, p in ALLERGENS.items()},
          **{f"diet:{d}": ([p], [], 6 if d != "vegan" else 8) for d, p in DIETS.items()},
          "allergy+negation": ([CONTEXT, NEGATION, r"\b(?:allerg\w*|celiac|coeliac|intoleran\w*|cross)"], [], 42)}
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
COMPILED = {s: ([re.compile(p, re.I) for p in inc], [re.compile(p, re.I) for p in exc], n)
            for s, (inc, exc, n) in STRATA.items()}
PREFILTER = re.compile("|".join(list(ALLERGENS.values()) + list(DIETS.values()) + [CONTEXT]), re.I)


def sentences(text):
    for s in SENTENCE_RE.split(text):
        s = s.strip()
        if MIN_WORDS <= len(s.split()) <= MAX_WORDS:
            yield s


def scan(records, seed=SEED, oversample=4):
    """Reservoir-sample candidate sentences per stratum from review records (dicts with text, review_id, gmap_id)."""
    rng = random.Random(seed)
    pools = {s: [] for s in COMPILED}
    seen = {s: 0 for s in COMPILED}
    for r in records:
        if not PREFILTER.search(r["text"]):
            continue
        for sent in sentences(r["text"]):
            for stratum, (inc, exc, n) in COMPILED.items():
                if all(p.search(sent) for p in inc) and not any(p.search(sent) for p in exc):
                    seen[stratum] += 1
                    k = n * oversample
                    item = (sent, r["review_id"], r["gmap_id"])
                    if len(pools[stratum]) < k:
                        pools[stratum].append(item)
                    else:
                        j = rng.randrange(seen[stratum])
                        if j < k:
                            pools[stratum][j] = item
    return pools, seen


def select(pools, seed=SEED):
    """Up to n unique sentences per stratum (case-insensitive dedup across strata), in stratum order."""
    rng, taken, out = random.Random(seed), set(), []
    for stratum, (_, _, n) in COMPILED.items():
        candidates = list(pools.get(stratum, []))
        rng.shuffle(candidates)
        got = 0
        for sent, rid, gid in candidates:
            if got == n:
                break
            if sent.lower() in taken:
                continue
            taken.add(sent.lower())
            out.append({"sentence": sent, "stratum": stratum, "review_id": rid, "gmap_id": gid})
            got += 1
    return out


def assign(items, judges, shared=SHARED, per_judge=PER_JUDGE, seed=SEED):
    """Give ids, mark `shared` sentences for everyone, split the rest per judge. Returns (key rows, files)."""
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    for k, item in enumerate(items, 1):
        item["sentence_id"] = f"T{k:03d}"
    shared_items = items[:shared]
    rest = items[shared:shared + per_judge * len(judges)]
    files = {j: list(shared_items) for j in judges}
    key = [{**i, "shared": 1, "assigned_to": "all"} for i in shared_items]
    for idx, item in enumerate(rest):
        judge = judges[idx % len(judges)]
        files[judge].append(item)
        key.append({**item, "shared": 0, "assigned_to": judge})
    for j in judges:
        random.Random(f"{seed}:{j}").shuffle(files[j])
    return key, files


def iter_reviews(path=REVIEWS_PATH):
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            yield {"text": r["text"], "review_id": r["review_id"], "gmap_id": r["gmap_id"]}


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None, tagger=None, records=None):
    ap = argparse.ArgumentParser(description="Stratified review-sentence labeling sheets")
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--reviews", type=Path, default=REVIEWS_PATH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    paths = {j: args.out_dir / f"sentences_{j}.csv" for j in args.judges}
    if any(p.exists() for p in paths.values()) and not args.force:
        sys.exit("Refusing to overwrite sentence sheets (they may contain labels). Use --force.")
    if tagger is None:
        from src.classifier.allergen_tagger import AllergenTagger
        tagger = AllergenTagger.load()
    pools, seen = scan(records if records is not None else iter_reviews(args.reviews), args.seed)
    items = select(pools, args.seed)
    key, files = assign(items, args.judges, seed=args.seed)
    suggestions = dict(zip((i["sentence_id"] for i in key), tagger.tag([i["sentence"] for i in key])))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for judge, rows in files.items():
        write_csv(paths[judge], COLUMNS, [{"sentence_id": r["sentence_id"], "sentence": r["sentence"],
                                           "allergens_mentioned": ";".join(suggestions[r["sentence_id"]]),
                                           "stance": "", "note": ""} for r in rows])
    write_csv(args.out_dir / "sentences_key.csv", KEY_COLUMNS, sorted(key, key=lambda r: r["sentence_id"]))
    short = {s: n for s, (_, _, n) in COMPILED.items() if sum(1 for i in items if i["stratum"] == s) < n}
    print(f"{len(items)} sentences ({len(key)} assigned): {SHARED} shared, "
          + ", ".join(f"{j} {len(files[j])} rows" for j in args.judges))
    if short:
        print("Strata with fewer sentences than planned:", {s: (sum(1 for i in items if i['stratum'] == s), n)
                                                             for s, n in short.items()})
    print("Wrote " + ", ".join(str(p) for p in paths.values()) + f" and {args.out_dir / 'sentences_key.csv'}")
    return key, files, seen


if __name__ == "__main__":
    main()
