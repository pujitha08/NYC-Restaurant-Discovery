"""Explore the Google Local New York business metadata (streamed, low memory).

Run from the repo root:
    python -m src.data.explore_meta [path/to/meta-New_York.json.gz]
"""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.config import BOROUGHS, DATA_DIR
from src.data.filter_meta import (
    CITY_BOROUGH, RESTAURANT_EXACT, RESTAURANT_EXCLUDE, ZIP_RE, classify_borough,
    matched_categories, stream_records,
)

DEFAULT_PATH = DATA_DIR / "raw" / "meta-New_York.json.gz"

def describe(values):
    return f"min={min(values)}  median={statistics.median(values)}  max={max(values)}  (n={len(values)})"


def main(path):
    fields = Counter()
    sample = None
    total = 0
    matched_cats = Counter()
    rest_total = 0
    rule_counts = Counter()
    borough_counts = Counter()
    reviews_by_borough = defaultdict(list)
    coords_by_borough = defaultdict(lambda: ([], []))  # lats, lons
    rest_with_coords = 0
    zip_city_disagree = 0

    for rec in stream_records(path):
        total += 1
        for k in rec:
            fields[k] += 1
        hits = matched_categories(rec.get("category"))
        if not hits:
            continue
        rest_total += 1
        matched_cats.update(hits)
        if sample is None:
            sample = rec
        if rec.get("latitude") is not None and rec.get("longitude") is not None:
            rest_with_coords += 1

        borough, rule = classify_borough(rec.get("address"))
        rule_counts[rule] += 1
        borough_counts[borough] += 1
        if rule == "zip":
            parts = [p.strip() for p in rec["address"].split(",")]
            city_b = CITY_BOROUGH.get(parts[-2].lower()) if len(parts) >= 3 else None
            if city_b and city_b != borough:
                zip_city_disagree += 1
        if borough in BOROUGHS:
            reviews_by_borough[borough].append(rec.get("num_of_reviews") or 0)
            if rec.get("latitude") is not None:
                coords_by_borough[borough][0].append(rec["latitude"])
                coords_by_borough[borough][1].append(rec["longitude"])

    print(f"File: {path}")
    print(f"Total businesses: {total:,}\n")

    print("== Fields (count of records containing each) ==")
    for k, n in fields.most_common():
        print(f"  {k:18s} {n:,}")

    print("\n== Sample restaurant record (long fields truncated) ==")
    short = {k: (v[:3] + ["..."] if isinstance(v, list) and len(v) > 3 else v) for k, v in sample.items()}
    print(json.dumps(short, indent=2, ensure_ascii=False)[:2500])

    print(f"\n== Restaurants: {rest_total:,} of {total:,} ({rest_total / total:.1%}) ==")
    print("Matching rule: category contains 'restaurant' (excluding "
          f"{list(RESTAURANT_EXCLUDE)}) or is one of {sorted(RESTAURANT_EXACT)}")
    print(f"Category values matched ({len(matched_cats)} distinct):")
    for c, n in matched_cats.most_common():
        print(f"  {n:7,}  {c}")

    print("\n== Borough identification ==")
    print(f"Restaurants with lat/lon: {rest_with_coords:,} of {rest_total:,}")
    print("Rules: (1) ZIP in address ('NY 11201'): 100-102=Manhattan, 112=Brooklyn, "
          "104=Bronx, 103=Staten Island, 111/113/114/116=Queens;")
    print("       (2) fallback city token before ', NY': Brooklyn/Bronx/Staten Island/New York(=Manhattan).")
    print(f"Rule used: {dict(rule_counts)}")
    print(f"ZIP vs city-token disagreements: {zip_city_disagree:,}")
    print("Restaurants per classified borough (all):")
    for b, n in borough_counts.most_common():
        print(f"  {n:7,}  {b}")

    print(f"\n== Scope: config.BOROUGHS = {BOROUGHS} ==")
    for b in BOROUGHS:
        lats, lons = coords_by_borough[b]
        print(f"{b}: {len(reviews_by_borough[b]):,} restaurants")
        if reviews_by_borough[b]:
            print(f"  num_of_reviews: {describe(reviews_by_borough[b])}")
        if lats:
            print(f"  lat range {min(lats):.3f}..{max(lats):.3f} (median {statistics.median(lats):.3f}); "
                  f"lon range {min(lons):.3f}..{max(lons):.3f} (median {statistics.median(lons):.3f})")
    in_scope = [n for b in BOROUGHS for n in reviews_by_borough[b]]
    if in_scope:
        print(f"All in-scope: {len(in_scope):,} restaurants; num_of_reviews: {describe(in_scope)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH)
