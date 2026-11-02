"""Filter the Google Local New York metadata down to in-scope, open restaurants.

Run from the repo root:
    python -m src.data.filter_meta
"""

import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.config import BOROUGHS, DATA_DIR

RAW_META_PATH = DATA_DIR / "raw" / "meta-New_York.json.gz"
OUT_PATH = DATA_DIR / "processed" / "restaurants_meta.jsonl"

# --- Restaurant definition (single place to edit) ---------------------------
# A category counts as a restaurant if it contains "restaurant" (unless it
# matches an exclusion) or is one of the explicit food-venue names below.
RESTAURANT_SUBSTRING = "restaurant"
RESTAURANT_EXACT = {
    "Cafe", "Bar & grill", "Diner", "Bistro", "Pizzeria", "Deli",
    "Sandwich shop", "Bagel shop", "Noodle shop", "Ramen shop", "Taqueria",
    "Steak house", "Brasserie", "Gastropub", "Buffet", "Food court",
}
RESTAURANT_EXCLUDE = ("supply", "equipment", "furniture", "consultant", "association")

# --- Closed-business exclusion ----------------------------------------------
# The `state` field holds either an hours status ("Open ⋅ Closes 6PM",
# "Closed ⋅ Opens 9AM") or a lasting status. Only permanent closure is excluded;
# "Temporarily closed" and missing state are kept.
EXCLUDED_STATES = {"Permanently closed"}

# --- Borough classification --------------------------------------------------
# ZIP prefix -> borough. Manhattan 100xx-102xx, Staten Island 103xx,
# Bronx 104xx, Brooklyn 112xx, Queens 111xx/113xx/114xx/116xx.
ZIP_PREFIX_BOROUGH = {
    "100": "Manhattan", "101": "Manhattan", "102": "Manhattan",
    "103": "Staten Island", "104": "Bronx", "112": "Brooklyn",
    "111": "Queens", "113": "Queens", "114": "Queens", "116": "Queens",
}
# City token (second-to-last address part) -> borough. Only unambiguous names;
# Queens neighborhoods (Astoria, Flushing, ...) are left to the ZIP rule.
CITY_BOROUGH = {
    "brooklyn": "Brooklyn", "bronx": "Bronx", "staten island": "Staten Island",
    "new york": "Manhattan",
}
ZIP_RE = re.compile(r"\bNY\s+(\d{5})\b")


def stream_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def matched_categories(categories):
    """Return the category values that make a business a restaurant (empty -> not one)."""
    hits = []
    for c in categories or []:
        low = c.lower()
        if c in RESTAURANT_EXACT or (
            RESTAURANT_SUBSTRING in low and not any(x in low for x in RESTAURANT_EXCLUDE)
        ):
            hits.append(c)
    return hits


def is_restaurant(categories):
    return bool(matched_categories(categories))


def is_permanently_closed(state):
    """True if `state` starts with an excluded lasting status (e.g. 'Permanently closed')."""
    if not state:
        return False
    head = re.split(r"\s*⋅", state)[0].strip()
    return head in EXCLUDED_STATES


def classify_borough(address):
    """First-pass borough from address: ZIP prefix, then city token. Returns (borough, rule)."""
    if not address:
        return None, "no_address"
    m = ZIP_RE.search(address)
    if m:
        b = ZIP_PREFIX_BOROUGH.get(m.group(1)[:3])
        if b:
            return b, "zip"
    parts = [p.strip() for p in address.split(",")]
    if len(parts) >= 3:
        b = CITY_BOROUGH.get(parts[-2].lower())
        if b:
            return b, "city"
    return None, "unmatched"


def to_output_record(rec, borough):
    return {
        "gmap_id": rec["gmap_id"],
        "name": rec["name"],
        "address": rec["address"],
        "borough": borough,
        "categories": rec.get("category") or [],
        "avg_rating": rec.get("avg_rating"),
        "num_of_reviews": rec.get("num_of_reviews") or 0,
        "latitude": rec.get("latitude"),
        "longitude": rec.get("longitude"),
    }


def preference_key(rec):
    """Sort key for choosing among duplicate records: open first, then most reviews."""
    return (not is_permanently_closed(rec.get("state")), rec.get("num_of_reviews") or 0)


def main(src=RAW_META_PATH, out=OUT_PATH):
    out.parent.mkdir(parents=True, exist_ok=True)
    total = restaurants = in_scope = duplicates_dropped = 0
    best = {}  # gmap_id -> (record, borough); in-scope restaurants only

    for rec in stream_records(src):
        total += 1
        if not is_restaurant(rec.get("category")):
            continue
        restaurants += 1
        borough, _ = classify_borough(rec.get("address"))
        if borough not in BOROUGHS:
            continue
        in_scope += 1
        prev = best.get(rec["gmap_id"])
        if prev is None:
            best[rec["gmap_id"]] = (rec, borough)
        else:
            duplicates_dropped += 1
            if preference_key(rec) > preference_key(prev[0]):
                best[rec["gmap_id"]] = (rec, borough)

    closed_removed = 0
    closed_by_borough = Counter()
    kept_by_borough = Counter()
    with open(out, "w", encoding="utf-8") as f:
        for rec, borough in best.values():
            if is_permanently_closed(rec.get("state")):
                closed_removed += 1
                closed_by_borough[borough] += 1
                continue
            kept_by_borough[borough] += 1
            f.write(json.dumps(to_output_record(rec, borough), ensure_ascii=False) + "\n")

    print(f"Scanned {total:,} businesses; {restaurants:,} restaurants statewide")
    print(f"In scope ({', '.join(BOROUGHS)}): {in_scope:,} records, "
          f"{duplicates_dropped:,} duplicate gmap_ids dropped -> {len(best):,} unique")
    print(f"Removed as permanently closed: {closed_removed:,} "
          f"({closed_removed / len(best):.1%} of unique in-scope) {dict(closed_by_borough)}")
    print(f"Final: {sum(kept_by_borough.values()):,} restaurants {dict(kept_by_borough)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))
