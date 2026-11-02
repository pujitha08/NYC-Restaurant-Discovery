"""Add a low_evidence flag to each restaurant and write restaurants_final.jsonl.

Run from the repo root (after filter_meta and filter_reviews):
    python -m src.data.build_final
"""

import json
import statistics
from collections import Counter

from src.config import BOROUGHS, DATA_DIR

RESTAURANTS_PATH = DATA_DIR / "processed" / "restaurants_meta.jsonl"
REVIEWS_PATH = DATA_DIR / "processed" / "reviews_filtered.jsonl"
OUT_PATH = DATA_DIR / "processed" / "restaurants_final.jsonl"

# Restaurants with fewer text reviews than this are flagged low_evidence.
MIN_TEXT_REVIEWS = 5


def count_text_reviews(path):
    counts = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            counts[json.loads(line)["gmap_id"]] += 1
    return counts


def main():
    counts = count_text_reviews(REVIEWS_PATH)
    n = low = 0
    by_borough = Counter()
    with open(RESTAURANTS_PATH, encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            rec["num_text_reviews"] = counts.get(rec["gmap_id"], 0)
            rec["low_evidence"] = rec["num_text_reviews"] < MIN_TEXT_REVIEWS
            n += 1
            low += rec["low_evidence"]
            by_borough[(rec["borough"], rec["low_evidence"])] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Restaurants: {n:,}; low_evidence (<{MIN_TEXT_REVIEWS} text reviews): {low:,}; "
          f"adequate: {n - low:,}")
    for b in BOROUGHS:
        print(f"  {b}: low={by_borough[(b, True)]:,}  adequate={by_borough[(b, False)]:,}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
