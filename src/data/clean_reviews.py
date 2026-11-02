"""Clean the filtered reviews and recompute restaurant evidence counts.

Run from the repo root (after filter_reviews and build_final):
    python -m src.data.clean_reviews

Two passes over reviews_filtered.jsonl: the first counts how often each cleaned
text occurs statewide (to detect template reviews), the second writes the output.
Each removed review is counted at the first stage that removes it.
"""

import json
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.config import DATA_DIR

PROCESSED = DATA_DIR / "processed"
IN_PATH = PROCESSED / "reviews_filtered.jsonl"
OUT_PATH = PROCESSED / "reviews_clean.jsonl"
RESTAURANTS_IN = PROCESSED / "restaurants_final.jsonl"
RESTAURANTS_OUT = PROCESSED / "restaurants_clean.jsonl"
STATS_PATH = PROCESSED / "cleaning_stats.json"

TRANSLATED_MARKER = "(Translated by Google)"
ORIGINAL_MARKER = "(Original)"
MIN_WORDS = 5
TEMPLATE_MIN_COPIES = 20  # a cleaned text this common statewide is a template review
MIN_CLEAN_REVIEWS = 5  # restaurants with fewer are low_evidence
PLACEHOLDER_DATE = (1990, 12, 31)  # source uses this date as a null
EXPECTED_REVIEWS = 2_759_578  # only sizes the progress bar


def clean_text(text):
    """Keep the translated text of a Google-translated review, then normalize whitespace."""
    if TRANSLATED_MARKER in text:
        text = text.replace(TRANSLATED_MARKER, "", 1).split(ORIGINAL_MARKER, 1)[0]
    return " ".join(text.split())


def is_placeholder_date(t):
    g = time.gmtime(t / 1000)
    return (g.tm_year, g.tm_mon, g.tm_mday) == PLACEHOLDER_DATE


def dedup_key(text):
    return hash(text.lower())


def iter_reviews(desc):
    with open(IN_PATH, encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, total=EXPECTED_REVIEWS, unit="rev", mininterval=5, desc=desc)):
            yield idx, json.loads(line)


def prepare(r):
    """Cleaned text for a review that passes the null filter, else None."""
    if r["rating"] is None or r["user_id"] is None:
        return None
    return clean_text(r["text"])


def main():
    # Pass 1: statewide counts of cleaned texts, over reviews that reach the template stage.
    seen_rows = set()
    text_counts = Counter()
    for _, r in iter_reviews("pass 1"):
        row = hash((r["gmap_id"], r["user_id"], r["time"], r["text"]))
        if row in seen_rows:
            continue
        seen_rows.add(row)
        text = prepare(r)
        if text is None or len(text.split()) < MIN_WORDS:
            continue
        text_counts[dedup_key(text)] += 1
    seen_rows.clear()

    # Pass 2: apply every stage and write.
    stats = Counter()
    seen_rows = set()
    seen_in_restaurant = set()
    per_restaurant = Counter()
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for idx, r in iter_reviews("pass 2"):
            stats["input"] += 1
            row = hash((r["gmap_id"], r["user_id"], r["time"], r["text"]))
            if row in seen_rows:
                stats["removed_exact_duplicate_row"] += 1
                continue
            seen_rows.add(row)
            text = prepare(r)
            if text is None:
                stats["removed_null_rating_or_user"] += 1
                continue
            if TRANSLATED_MARKER in r["text"]:
                stats["translated_stripped"] += 1  # informational; short ones are removed below
            if len(text.split()) < MIN_WORDS:
                stats["removed_under_min_words"] += 1
                continue
            key = dedup_key(text)
            if text_counts[key] >= TEMPLATE_MIN_COPIES:
                stats["removed_template_text"] += 1
                continue
            per_key = hash((r["gmap_id"], key))
            if per_key in seen_in_restaurant:
                stats["removed_duplicate_in_restaurant"] += 1
                continue
            seen_in_restaurant.add(per_key)

            t = r["time"]
            if is_placeholder_date(t):
                t = None
                stats["placeholder_date_nulled"] += 1
            out.write(json.dumps({"review_id": idx, "gmap_id": r["gmap_id"], "user_id": r["user_id"],
                                  "rating": r["rating"], "time": t, "text": text}, ensure_ascii=False) + "\n")
            per_restaurant[r["gmap_id"]] += 1
            stats["kept"] += 1

    n_templates = sum(1 for c in text_counts.values() if c >= TEMPLATE_MIN_COPIES)

    low = zero = 0
    old_low = 0
    with open(RESTAURANTS_IN, encoding="utf-8") as fin, open(RESTAURANTS_OUT, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            n = per_restaurant.get(rec["gmap_id"], 0)
            old_low += rec["low_evidence"]
            rec["num_clean_reviews"] = n
            rec["low_evidence"] = n < MIN_CLEAN_REVIEWS
            low += rec["low_evidence"]
            zero += n == 0
            stats["restaurants"] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats["restaurants_low_evidence"] = low
    stats["restaurants_zero_reviews"] = zero
    stats["restaurants_low_evidence_before"] = old_low
    stats["distinct_template_texts"] = n_templates
    STATS_PATH.write_text(json.dumps(dict(stats), indent=2))

    print("\n== Cleaning results ==")
    print(f"Input reviews: {stats['input']:,}")
    for k, label in [
        ("removed_exact_duplicate_row", "exact duplicate rows (same restaurant, user, time, text)"),
        ("removed_null_rating_or_user", "null rating or user_id"),
        ("removed_under_min_words", f"under {MIN_WORDS} words after cleaning"),
        ("removed_template_text", f"template texts ({TEMPLATE_MIN_COPIES}+ copies statewide; "
                                  f"{n_templates:,} distinct texts)"),
        ("removed_duplicate_in_restaurant", "duplicate text within a restaurant"),
    ]:
        print(f"  - {stats[k]:>9,}  {label}")
    print(f"Kept: {stats['kept']:,} ({100 * stats['kept'] / stats['input']:.1f}% of input)")
    print(f"Informational: {stats['translated_stripped']:,} translated reviews had the original text "
          f"stripped (before later stages); {stats['placeholder_date_nulled']:,} kept reviews had their "
          f"{PLACEHOLDER_DATE[0]}-12-31 placeholder date set to null")
    print(f"Restaurants: {stats['restaurants']:,}; with < {MIN_CLEAN_REVIEWS} clean reviews: {low:,} "
          f"(was {old_low:,}); with zero: {zero:,}")
    print(f"Wrote {OUT_PATH}\nWrote {RESTAURANTS_OUT}\nWrote {STATS_PATH}")


if __name__ == "__main__":
    main()
