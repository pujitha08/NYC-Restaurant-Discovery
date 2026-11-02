"""Count dietary-term mentions in the filtered reviews.

Run from the repo root (after filter_reviews and build_final):
    python -m src.data.dietary_terms
"""

import csv
import json
import random
import re
from collections import Counter

from tqdm import tqdm

from src.config import DATA_DIR

REVIEWS_PATH = DATA_DIR / "processed" / "reviews_filtered.jsonl"
RESTAURANTS_PATH = DATA_DIR / "processed" / "restaurants_final.jsonl"
OUT_PATH = DATA_DIR / "processed" / "dietary_term_counts.csv"

TOP_N = 5
EXAMPLES_PER_GROUP = 5
SEED = 0

# Term groups: case-insensitive, word-bounded. Groups may overlap (a review can
# match several), so per-group counts do not sum to a total.
TERM_GROUPS = {
    "vegan": r"\bvegans?\b",
    "vegetarian": r"\bvegetarians?\b",
    "gluten-free": r"\bgluten[- ]?free\b|\bceliacs?\b|\bcoeliacs?\b",
    "halal": r"\bhalal\b",
    "kosher": r"\bkosher\b",
    "dairy-free/lactose": r"\bdairy[- ]?free\b|\blactose\b",
    "nut allergy": (
        r"\b(?:peanut|nut|tree[- ]nut)s?[- ]allerg\w*"
        r"|\btree[- ]nuts?\b"
        r"|\ballergic to (?:peanuts?|(?:tree[- ]?)?nuts?)\b"
    ),
    "shellfish allergy": r"\bshellfish[- ]allerg\w*|\ballergic to shellfish\b",
    "allergic to (any)": r"\ballergic to\b",
    "allergy (general)": r"\ballerg(?:y|ies|ic|en|ens)\b",
    "cross-contamination": r"\bcross[- ](?:contamination|contact|contaminat\w*)\b",
}
PATTERNS = {g: re.compile(p, re.IGNORECASE) for g, p in TERM_GROUPS.items()}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentence_with_match(text, pattern):
    """Return the sentence of `text` containing the first match of `pattern`."""
    m = pattern.search(text)
    start = 0
    for s in SENTENCE_SPLIT.split(text):
        idx = text.find(s, start)
        if idx <= m.start() < idx + len(s):
            return s.strip()
        start = idx + len(s)
    return text[:200]


def main():
    names = {}
    with open(RESTAURANTS_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            names[r["gmap_id"]] = f'{r["name"]} ({r["borough"]})'

    rng = random.Random(SEED)
    review_counts = Counter()
    by_restaurant = {g: Counter() for g in PATTERNS}
    examples = {g: [] for g in PATTERNS}  # reservoir sample of sentences
    total = 0

    with open(REVIEWS_PATH, encoding="utf-8") as f:
        for line in tqdm(f, total=2_759_578, unit="rev", mininterval=5):
            rec = json.loads(line)
            text = rec["text"]
            total += 1
            for g, pat in PATTERNS.items():
                if pat.search(text):
                    review_counts[g] += 1
                    by_restaurant[g][rec["gmap_id"]] += 1
                    n = review_counts[g]
                    if len(examples[g]) < EXAMPLES_PER_GROUP:
                        examples[g].append(sentence_with_match(text, pat))
                    else:
                        j = rng.randrange(n)
                        if j < EXAMPLES_PER_GROUP:
                            examples[g][j] = sentence_with_match(text, pat)

    n_rest = len(names)
    rows = []
    for g in PATTERNS:
        top = by_restaurant[g].most_common(TOP_N)
        rows.append({
            "term_group": g,
            "reviews": review_counts[g],
            "pct_reviews": round(100 * review_counts[g] / total, 3),
            "restaurants": len(by_restaurant[g]),
            "pct_restaurants": round(100 * len(by_restaurant[g]) / n_rest, 2),
            "top_restaurants": "; ".join(f"{names.get(i, i)}: {c}" for i, c in top),
        })
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nScanned {total:,} reviews across {n_rest:,} restaurants\n")
    print(f"{'term group':22s} {'reviews':>9s} {'%rev':>7s} {'restaurants':>12s} {'%rest':>7s}")
    for r in rows:
        print(f"{r['term_group']:22s} {r['reviews']:9,d} {r['pct_reviews']:6.2f}% "
              f"{r['restaurants']:12,d} {r['pct_restaurants']:6.1f}%")
    for r in rows:
        print(f"\n=== {r['term_group']} ===")
        print(f"Top {TOP_N}: {r['top_restaurants']}")
        for s in examples[r["term_group"]]:
            print(f"  - {s[:300]}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
