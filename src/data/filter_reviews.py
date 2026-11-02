"""Stream the Google Local New York reviews and keep text reviews for in-scope restaurants.

Run from the repo root:
    python -m src.data.filter_reviews
"""

import gzip
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.config import DATA_DIR

RAW_REVIEWS_PATH = DATA_DIR / "raw" / "review-New_York.json.gz"
RESTAURANTS_PATH = DATA_DIR / "processed" / "restaurants_meta.jsonl"
OUT_PATH = DATA_DIR / "processed" / "reviews_filtered.jsonl"

KEEP_FIELDS = ("gmap_id", "user_id", "rating", "time", "text")
# Review count stated on the dataset page; only used to size the progress bar.
EXPECTED_REVIEWS = 33_459_761
MIN_TEXT_REVIEWS_THRESHOLD = 5


def load_restaurant_ids(path):
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["gmap_id"] for line in f}


def main(src=RAW_REVIEWS_PATH, restaurants=RESTAURANTS_PATH, out=OUT_PATH):
    ids = load_restaurant_ids(restaurants)
    print(f"Loaded {len(ids):,} restaurant ids", flush=True)

    scanned = kept = skipped_empty = total_words = 0
    per_restaurant = Counter()

    with gzip.open(src, "rt", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=EXPECTED_REVIEWS, unit="rev", mininterval=5, file=sys.stdout):
            scanned += 1
            rec = json.loads(line)
            if rec.get("gmap_id") not in ids:
                continue
            text = rec.get("text")
            if not text or not text.strip():
                skipped_empty += 1
                continue
            kept += 1
            total_words += len(text.split())
            per_restaurant[rec["gmap_id"]] += 1
            fout.write(json.dumps({k: rec.get(k) for k in KEEP_FIELDS}, ensure_ascii=False) + "\n")

    counts = list(per_restaurant.values())
    print("\n== Results ==")
    print(f"Reviews scanned: {scanned:,}")
    print(f"Reviews for in-scope restaurants: {kept + skipped_empty:,}")
    print(f"Skipped (null/empty text, in-scope restaurants only): {skipped_empty:,}")
    print(f"Reviews kept: {kept:,}")
    print(f"Restaurants with >=1 text review: {len(counts):,} of {len(ids):,} "
          f"({len(ids) - len(counts):,} have none)")
    if counts:
        few = sum(c < MIN_TEXT_REVIEWS_THRESHOLD for c in counts)
        print(f"Text reviews per restaurant (among those with >=1): "
              f"min={min(counts)}  median={statistics.median(counts)}  max={max(counts)}")
        print(f"Restaurants with <{MIN_TEXT_REVIEWS_THRESHOLD} text reviews: {few:,} "
              f"({few + len(ids) - len(counts):,} if restaurants with none are included)")
        print(f"Average review length: {total_words / kept:.1f} words")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:4]))
