"""Exploratory data analysis of the processed restaurants and reviews.

Writes PNG figures plus EDA_SUMMARY.md to reports/eda/. Reviews are streamed and
aggregated as they are read. Fields not kept in the processed files (price,
hours, description, MISC) are read from the raw metadata for the final ids.

Run from the repo root (after the data pipeline):
    python -m src.data.eda
"""

import csv
import gzip
import json
import re
import statistics
import time
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from src.config import BOROUGHS, PROJECT_ROOT
from src.data.dietary_terms import PATTERNS
from src.data.filter_meta import RAW_META_PATH, stream_records

PROCESSED = PROJECT_ROOT / "data" / "processed"
RESTAURANTS_PATH = PROCESSED / "restaurants_final.jsonl"
REVIEWS_PATH = PROCESSED / "reviews_filtered.jsonl"
OUT_DIR = PROJECT_ROOT / "reports" / "eda"
EXPECTED_REVIEWS = 2_759_578  # only sizes the progress bar

SHORT_REVIEW_WORDS = 5
LONG_REVIEW_WORDS = 200
EMBED_LIMITS = (256, 512)
TOKENS_PER_WORD = 1.3  # rough English estimate for subword tokenizers
MIN_EVIDENCE = 5  # restaurants with fewer text reviews are low_evidence
TOP_CATEGORIES = 30
HEATMAP_CATEGORIES = 15

# MISC values related to diet. "strict" ones map to a diet we can cross-check
# against review mentions; "adjacent" ones are related but not a diet.
MISC_STRICT = {"Vegetarian options": "vegetarian", "Halal food": "halal"}
MISC_DIET_RE = re.compile(
    r"vegetarian|vegan|halal|kosher|gluten|allerg|dairy|lactose|\bnuts?\b|organic|healthy", re.I
)
CORE_GROUPS = ["vegan", "vegetarian", "gluten-free", "halal", "kosher",
               "dairy-free/lactose", "nut allergy", "shellfish allergy"]
HEATMAP_GROUPS = ["vegan", "vegetarian", "gluten-free", "halal", "kosher", "any core diet"]

URL_RE = re.compile(r"https?://|www\.", re.I)
NONASCII_LETTER_RE = re.compile(r"(?![A-Za-z])[^\W\d_]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
NON_ENGLISH_SHARE = 0.30  # a review is "mostly non-ASCII" above this share of letters

# Chart style: one categorical hue for single series, ink tokens for text.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#eef4fc", "#2a78d6", "#123d73"])


# ---------------------------------------------------------------- helpers
def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def new_fig(w=7.5, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def hbar(labels, values, title, name, xlabel, color=BLUE, w=7.5, fmt="{:,}"):
    fig, ax = new_fig(w, max(2.6, 0.27 * len(labels) + 1.2))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=color, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK2, fontsize=9)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8, color=INK2)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlim(0, max(values) * 1.15)
    style(ax, "x")
    ax.grid(axis="y", visible=False)
    save(fig, name)


def pct_from_counter(counter, qs):
    """Percentiles from a {value: count} Counter."""
    total = sum(counter.values())
    keys = sorted(counter)
    out, cum, i = {}, 0, 0
    targets = sorted(qs)
    for k in keys:
        cum += counter[k]
        while i < len(targets) and cum >= targets[i] / 100 * total:
            out[targets[i]] = k
            i += 1
    return out


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


# ---------------------------------------------------------------- pass A
def load_restaurants():
    with open(RESTAURANTS_PATH, encoding="utf-8") as f:
        return {r["gmap_id"]: r for r in map(json.loads, f)}


def read_raw_extras(ids):
    """price / hours / description / MISC for the final restaurants (first copy wins)."""
    extras = {}
    for rec in stream_records(RAW_META_PATH):
        gid = rec["gmap_id"]
        if gid in ids and gid not in extras:
            extras[gid] = {
                "price": rec.get("price"),
                "has_hours": bool(rec.get("hours")),
                "has_desc": bool(rec.get("description")),
                "misc": rec.get("MISC") if isinstance(rec.get("MISC"), dict) else None,
            }
    return extras


# ---------------------------------------------------------------- pass B
class ReviewStats:
    def __init__(self):
        self.n = 0
        self.ratings = Counter()
        self.null_rating = self.null_user = 0
        self.word_lens = Counter()
        self.years = Counter()
        self.tmin, self.tmax = None, None
        self.short_texts = Counter()  # exact text, only reviews < SHORT_REVIEW_WORDS words
        self.long_hashes = Counter()  # hash of text, reviews >= SHORT_REVIEW_WORDS words
        self.urls = self.translated = self.mostly_nonascii = 0
        self.users = Counter()  # hash(user_id) -> reviews
        self.rest_words = Counter()
        self.rest_reviews = Counter()
        self.rest_groups = defaultdict(Counter)  # gmap_id -> group -> reviews mentioning


def stream_reviews():
    s = ReviewStats()
    with open(REVIEWS_PATH, encoding="utf-8") as f:
        for line in tqdm(f, total=EXPECTED_REVIEWS, unit="rev", mininterval=5):
            r = json.loads(line)
            text = r["text"]
            s.n += 1
            if r["rating"] is None:
                s.null_rating += 1
            else:
                s.ratings[r["rating"]] += 1
            w = len(text.split())
            s.word_lens[w] += 1
            t = r["time"]
            s.years[time.gmtime(t / 1000).tm_year] += 1
            s.tmin = t if s.tmin is None else min(s.tmin, t)
            s.tmax = t if s.tmax is None else max(s.tmax, t)
            if w < SHORT_REVIEW_WORDS:
                s.short_texts[text] += 1
            else:
                s.long_hashes[hash(text)] += 1
            if URL_RE.search(text):
                s.urls += 1
            if "(Translated by Google)" in text:
                s.translated += 1
            non = len(NONASCII_LETTER_RE.findall(text))
            if non and non / (non + len(ASCII_LETTER_RE.findall(text))) > NON_ENGLISH_SHARE:
                s.mostly_nonascii += 1
            if r["user_id"] is None:
                s.null_user += 1
            else:
                s.users[hash(r["user_id"])] += 1
            gid = r["gmap_id"]
            s.rest_words[gid] += w
            s.rest_reviews[gid] += 1
            for g, pat in PATTERNS.items():
                if pat.search(text):
                    s.rest_groups[gid][g] += 1
    return s


def second_pass(stats, n_dups=15, n_users=10):
    """Recover texts for the most repeated long reviews and detail the top users."""
    top_hashes = {h for h, _ in stats.long_hashes.most_common(n_dups)}
    top_users = {h for h, _ in stats.users.most_common(n_users)}
    dup_text = {}
    users = {h: {"user_id": None, "n": 0, "rests": set(), "rating": 0, "rated": 0, "words": 0, "texts": Counter()}
             for h in top_users}
    with open(REVIEWS_PATH, encoding="utf-8") as f:
        for line in tqdm(f, total=EXPECTED_REVIEWS, unit="rev", mininterval=5, desc="pass 2"):
            r = json.loads(line)
            text = r["text"]
            if top_hashes:
                h = hash(text)
                if h in top_hashes:
                    dup_text.setdefault(h, text)
            uh = hash(r["user_id"]) if r["user_id"] is not None else None
            if uh in users:
                u = users[uh]
                u["user_id"] = r["user_id"]
                u["n"] += 1
                u["rests"].add(r["gmap_id"])
                if r["rating"] is not None:
                    u["rating"] += r["rating"]
                    u["rated"] += 1
                u["words"] += len(text.split())
                u["texts"][text] += 1
    return dup_text, users


# ---------------------------------------------------------------- analysis
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rests = load_restaurants()
    ids = set(rests)
    print(f"{len(rests):,} restaurants; reading raw metadata extras ...", flush=True)
    extras = read_raw_extras(ids)
    print("streaming reviews ...", flush=True)
    st = stream_reviews()
    dup_text, top_users = second_pass(st)

    md = ["# EDA summary\n",
          f"Scope: {', '.join(BOROUGHS)}. {len(rests):,} restaurants (`restaurants_final.jsonl`), "
          f"{st.n:,} text reviews (`reviews_filtered.jsonl`). Generated by `python -m src.data.eda`.\n"]

    # ---- Restaurants -----------------------------------------------------
    md.append("## Restaurants\n")
    by_b = Counter(r["borough"] for r in rests.values())
    fig, ax = new_fig(5.5, 3.2)
    xs = list(by_b)
    ax.bar(xs, [by_b[b] for b in xs], color=[BLUE, ORANGE][: len(xs)], width=0.5)
    for i, b in enumerate(xs):
        ax.text(i, by_b[b], f"{by_b[b]:,}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_title("Restaurants per borough", loc="left", fontsize=11)
    ax.set_ylim(0, max(by_b.values()) * 1.12)
    style(ax)
    save(fig, "restaurants_borough.png")
    md.append("### Borough counts\n" + md_table(["Borough", "Restaurants", "Share"],
              [[b, f"{by_b[b]:,}", pct(by_b[b], len(rests))] for b in by_b]))

    cat_counts = Counter(c for r in rests.values() for c in r["categories"])
    top = cat_counts.most_common(TOP_CATEGORIES)
    hbar([c for c, _ in top], [n for _, n in top], f"Top {TOP_CATEGORIES} categories (a restaurant can have several)",
         "top_categories.png", "Restaurants", w=8)
    md.append(f"### Top {TOP_CATEGORIES} categories\nDistinct categories: {len(cat_counts):,}. "
              f"Restaurants have a mean of {statistics.mean(len(r['categories']) for r in rests.values()):.1f} categories each.\n\n"
              + md_table(["Category", "Restaurants"], [[c, f"{n:,}"] for c, n in top]))

    ratings = [r["avg_rating"] for r in rests.values() if r["avg_rating"] is not None]
    nrev = [r["num_of_reviews"] for r in rests.values()]
    ntxt = [r["num_text_reviews"] for r in rests.values()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    axes[0].hist(ratings, bins=np.arange(1, 5.26, 0.25), color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    axes[0].set_title("avg_rating per restaurant", loc="left", fontsize=11)
    axes[0].set_xlabel("Average rating")
    axes[0].set_ylabel("Restaurants")
    bins = np.logspace(0, np.log10(max(nrev) + 1), 40)
    axes[1].hist(nrev, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1, label="num_of_reviews (all)")
    axes[1].hist([max(n, 1) for n in ntxt], bins=bins, color=ORANGE, edgecolor=SURFACE, linewidth=1,
                 alpha=0.85, label="text reviews kept")
    axes[1].set_xscale("log")
    axes[1].set_title("Reviews per restaurant", loc="left", fontsize=11)
    axes[1].set_xlabel("Reviews (log scale)")
    axes[1].legend(frameon=False, fontsize=8)
    for a in axes:
        style(a)
    save(fig, "restaurant_distributions.png")
    q = lambda xs, ps: {p: float(np.percentile(xs, p)) for p in ps}
    rq, nq, tq = q(ratings, (5, 25, 50, 75, 95)), q(nrev, (5, 25, 50, 75, 95)), q(ntxt, (5, 25, 50, 75, 95))
    at_cap = sum(n >= 9998 for n in nrev)
    md.append("### Rating and review-count distributions\n" + md_table(
        ["Measure", "p5", "p25", "median", "p75", "p95", "max"],
        [["avg_rating", *[f"{rq[p]:.1f}" for p in (5, 25, 50, 75, 95)], f"{max(ratings):.1f}"],
         ["num_of_reviews (all reviews incl. rating-only)", *[f"{nq[p]:.0f}" for p in (5, 25, 50, 75, 95)], f"{max(nrev):,}"],
         ["text reviews kept", *[f"{tq[p]:.0f}" for p in (5, 25, 50, 75, 95)], f"{max(ntxt):,}"]])
        + f"\nRestaurants with avg_rating missing: {len(rests) - len(ratings):,}. "
          f"Restaurants with num_of_reviews >= 9,998 (looks like a source cap): {at_cap:,}. "
          f"Restaurants with no text reviews: {sum(n == 0 for n in ntxt):,}; low_evidence (<{MIN_EVIDENCE}): "
          f"{sum(r['low_evidence'] for r in rests.values()):,}.\n")

    # coverage of price / hours / description / MISC
    n = len(rests)
    have_price = sum(1 for e in extras.values() if e["price"])
    have_hours = sum(e["has_hours"] for e in extras.values())
    have_desc = sum(e["has_desc"] for e in extras.values())
    have_misc = sum(1 for e in extras.values() if e["misc"])
    cov = [("avg_rating", len(ratings)), ("hours", have_hours), ("MISC attributes", have_misc),
           ("price", have_price), ("description", have_desc)]
    fig, ax = new_fig(6.5, 3.2)
    ys = np.arange(len(cov))[::-1]
    ax.barh(ys, [100 * c / n for _, c in cov], color=BLUE, height=0.55)
    ax.set_yticks(ys)
    ax.set_yticklabels([k for k, _ in cov], color=INK2)
    for y, (_, c) in zip(ys, cov):
        ax.text(100 * c / n, y, f" {100 * c / n:.1f}%", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of restaurants with the field populated")
    ax.set_title("Field coverage", loc="left", fontsize=11)
    style(ax, "x")
    ax.grid(axis="y", visible=False)
    save(fig, "field_coverage.png")
    price_counts = Counter(e["price"] for e in extras.values() if e["price"])
    md.append("### Field coverage\n" + md_table(["Field", "Restaurants", "Coverage"],
              [[k, f"{c:,}", pct(c, n)] for k, c in cov])
              + f"\nPrice values: {dict(price_counts.most_common())}.\n")

    # MISC attributes
    misc_counts, key_counts = Counter(), Counter()
    for e in extras.values():
        if e["misc"]:
            for k, v in e["misc"].items():
                key_counts[k] += 1
                for x in (v if isinstance(v, list) else [v]):
                    misc_counts[(k, str(x))] += 1
    with open(OUT_DIR / "misc_attributes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value", "restaurants", "pct_restaurants", "dietary_related"])
        for (k, v), c in misc_counts.most_common():
            w.writerow([k, v, c, round(100 * c / n, 2), bool(MISC_DIET_RE.search(v))])
    diet_attrs = [(k, v, c) for (k, v), c in misc_counts.most_common() if MISC_DIET_RE.search(v)]
    strict_ids = {g: {i for i, e in extras.items() if e["misc"] and v_in(e["misc"], attr)}
                  for attr, g in MISC_STRICT.items()}
    any_strict = set().union(*strict_ids.values())
    adj_attr = [a for _, a, _ in diet_attrs if a not in MISC_STRICT]
    any_broad = {i for i, e in extras.items() if e["misc"] and any(
        MISC_DIET_RE.search(str(x)) for v in e["misc"].values() for x in (v if isinstance(v, list) else [v]))}
    fig, ax = new_fig(6.5, 3)
    labels = [f"{v}" for _, v, _ in diet_attrs]
    vals = [c for _, _, c in diet_attrs]
    ys = np.arange(len(labels))[::-1]
    ax.barh(ys, vals, color=BLUE, height=0.55)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, color=INK2)
    for y, v in zip(ys, vals):
        ax.text(v, y, f" {v:,}", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_title("Diet-related MISC attributes", loc="left", fontsize=11)
    ax.set_xlabel("Restaurants")
    style(ax, "x")
    ax.grid(axis="y", visible=False)
    save(fig, "misc_dietary.png")
    md.append("### MISC attributes\n"
              f"{have_misc:,} restaurants ({pct(have_misc, n)}) have a MISC dict. {len(misc_counts):,} distinct key/value "
              "pairs are listed with counts in `misc_attributes.csv` (every pair, with a `dietary_related` flag). "
              "Keys by coverage:\n\n"
              + md_table(["MISC key", "Restaurants"], [[k, f"{c:,}"] for k, c in key_counts.most_common()])
              + "\nTop 25 key/value pairs:\n\n"
              + md_table(["Key", "Value", "Restaurants"], [[k, v, f"{c:,}"] for (k, v), c in misc_counts.most_common(25)])
              + "\nDiet-related pairs (matched on vegetarian, vegan, halal, kosher, gluten, allergen, dairy, lactose, nut, "
                "organic, healthy):\n\n"
              + md_table(["Key", "Value", "Restaurants", "Class"],
                         [[k, v, f"{c:,}", "diet" if v in MISC_STRICT else "diet-adjacent"] for k, v, c in diet_attrs])
              + f"\n**No MISC attribute exists for vegan, kosher, gluten-free, dairy-free or any allergen.** "
                f"Restaurants with a vegetarian or halal attribute: {len(any_strict):,} ({pct(len(any_strict), n)}). "
                f"Including diet-adjacent organic/healthy attributes: {len(any_broad):,} ({pct(len(any_broad), n)}).\n")

    # ---- Review-mention rates, cross-check ------------------------------
    def mentions(gid, g):
        return st.rest_groups[gid][g] if gid in st.rest_groups else 0

    def any_core(gid):
        return sum(st.rest_groups[gid][g] for g in CORE_GROUPS) if gid in st.rest_groups else 0

    adequate = [i for i, r in rests.items() if not r["low_evidence"]]
    xrows = []
    for attr, g in MISC_STRICT.items():
        has = strict_ids[g] & set(adequate)
        no = set(adequate) - has
        m1 = {i for i in adequate if mentions(i, g) >= 1}
        m3 = {i for i in adequate if mentions(i, g) >= 3}
        xrows.append([f"{attr} ↔ {g}", f"{len(has):,}",
                      pct(sum(mentions(i, g) >= 1 for i in has), len(has)),
                      pct(sum(mentions(i, g) >= 3 for i in has), len(has)),
                      pct(sum(mentions(i, g) >= 1 for i in no), len(no)),
                      f"{len(m1):,}", pct(len(m1 & has), len(m1)),
                      f"{len(m3):,}", pct(len(m3 & has), len(m3))])
    md.append("## Cross-check: MISC dietary attributes vs review mentions\n"
              f"Restricted to the {len(adequate):,} restaurants with at least {MIN_EVIDENCE} text reviews. "
              "Only vegetarian and halal can be checked, since those are the only diet attributes in MISC.\n\n"
              + md_table(["Attribute ↔ review regex", "With attribute", "…≥1 mention", "…≥3 mentions",
                          "Without attribute: ≥1 mention", "Restaurants ≥1 mention", "…of which have attribute",
                          "Restaurants ≥3 mentions", "…of which have attribute"], xrows))

    # dietary mention rate by category and borough
    def rate_row(members):
        row, total_rev = {}, sum(st.rest_reviews[i] for i in members)
        for g in HEATMAP_GROUPS:
            if g == "any core diet":
                hit = sum(any_core(i) >= 1 for i in members)
            else:
                hit = sum(mentions(i, g) >= 1 for i in members)
            row[g] = 100 * hit / len(members) if members else 0
        row["reviews_with_core_mention_per_1000"] = (
            1000 * sum(any_core(i) for i in members) / total_rev if total_rev else 0)
        return row

    top_cats = [c for c, _ in cat_counts.most_common(HEATMAP_CATEGORIES)]
    cat_rows = {c: rate_row([i for i in adequate if c in rests[i]["categories"]]) for c in top_cats}
    bor_rows = {b: rate_row([i for i in adequate if rests[i]["borough"] == b]) for b in BOROUGHS}
    with open(OUT_DIR / "dietary_rate_by_group.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment_type", "segment", "restaurants"] + HEATMAP_GROUPS + ["core_mention_reviews_per_1000"])
        for typ, rows, keyf in (("category", cat_rows, lambda c: sum(1 for i in adequate if c in rests[i]["categories"])),
                                ("borough", bor_rows, lambda b: sum(1 for i in adequate if rests[i]["borough"] == b))):
            for k, r in rows.items():
                w.writerow([typ, k, keyf(k)] + [round(r[g], 2) for g in HEATMAP_GROUPS]
                           + [round(r["reviews_with_core_mention_per_1000"], 2)])
    mat = np.array([[cat_rows[c][g] for g in HEATMAP_GROUPS] for c in top_cats])
    fig, ax = new_fig(8, 6)
    ax.imshow(mat, cmap=SEQ, aspect="auto", vmin=0, vmax=mat.max())
    ax.set_xticks(range(len(HEATMAP_GROUPS)))
    ax.set_xticklabels(HEATMAP_GROUPS, color=INK2, fontsize=9)
    ax.set_yticks(range(len(top_cats)))
    ax.set_yticklabels(top_cats, color=INK2, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.55 * mat.max() else INK)
    ax.set_title("% of restaurants with ≥1 review mention, by category", loc="left", fontsize=11)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "dietary_rate_by_category.png")
    hdr = ["Segment", "Restaurants"] + [f"{g} %" for g in HEATMAP_GROUPS] + ["core-mention reviews / 1,000"]
    md.append("## Dietary-mention rate by category and borough\n"
              f"% of restaurants (≥{MIN_EVIDENCE} text reviews) with at least one review mentioning the diet. "
              "\"Any core diet\" = vegan, vegetarian, gluten-free, halal, kosher, dairy-free/lactose, nut allergy or shellfish allergy "
              "(excludes the noisy generic allergy groups). A restaurant counts under every category it has.\n\n"
              + md_table(hdr, [[c, f"{sum(1 for i in adequate if c in rests[i]['categories']):,}"]
                               + [f"{cat_rows[c][g]:.1f}" for g in HEATMAP_GROUPS]
                               + [f"{cat_rows[c]['reviews_with_core_mention_per_1000']:.1f}"] for c in top_cats])
              + "\nBy borough:\n\n"
              + md_table(hdr, [[b, f"{sum(1 for i in adequate if rests[i]['borough'] == b):,}"]
                               + [f"{bor_rows[b][g]:.1f}" for g in HEATMAP_GROUPS]
                               + [f"{bor_rows[b]['reviews_with_core_mention_per_1000']:.1f}"] for b in BOROUGHS]))

    # ---- Reviews ----------------------------------------------------------
    md.append("## Reviews\n")
    rating_total = sum(st.ratings.values())
    fig, ax = new_fig(5.5, 3.4)
    rs = sorted(st.ratings)
    ax.bar([str(r) for r in rs], [st.ratings[r] for r in rs], color=BLUE, width=0.6)
    for i, r in enumerate(rs):
        ax.text(i, st.ratings[r], f"{100 * st.ratings[r] / rating_total:.0f}%", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Reviews")
    ax.set_title("Review rating distribution", loc="left", fontsize=11)
    ax.set_ylim(0, max(st.ratings.values()) * 1.12)
    style(ax)
    save(fig, "review_ratings.png")
    md.append("### Rating distribution\n" + md_table(["Rating", "Reviews", "Share"],
              [[r, f"{st.ratings[r]:,}", pct(st.ratings[r], rating_total)] for r in rs])
              + f"\nMean rating {sum(r * c for r, c in st.ratings.items()) / rating_total:.2f}. "
                f"Reviews with a null rating (and null user_id): {st.null_rating:,}.\n")

    wp = pct_from_counter(st.word_lens, [5, 25, 50, 75, 95, 99])
    under = sum(c for w_, c in st.word_lens.items() if w_ < SHORT_REVIEW_WORDS)
    over = sum(c for w_, c in st.word_lens.items() if w_ > LONG_REVIEW_WORDS)
    mean_w = sum(w_ * c for w_, c in st.word_lens.items()) / st.n
    fig, ax = new_fig(7, 3.6)
    cl = 150
    xs = np.arange(0, cl + 1)
    ys_ = [st.word_lens.get(i, 0) for i in xs[:-1]] + [sum(c for w_, c in st.word_lens.items() if w_ >= cl)]
    ax.bar(xs, ys_, color=BLUE, width=1.0)
    ax.set_yscale("log")
    ax.axvline(wp[50], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(wp[50] + 1.5, ax.get_ylim()[1] * 0.6, f"median {wp[50]}", fontsize=9, color=INK2)
    ax.set_xlabel(f"Words per review (last bar: {cl}+)")
    ax.set_ylabel("Reviews (log scale)")
    ax.set_title("Review length", loc="left", fontsize=11)
    style(ax)
    save(fig, "review_length.png")
    md.append("### Review length (words)\n" + md_table(["p5", "p25", "median", "p75", "p95", "p99", "mean", "max"],
              [[wp[5], wp[25], wp[50], wp[75], wp[95], wp[99], f"{mean_w:.1f}", f"{max(st.word_lens):,}"]])
              + f"\nUnder {SHORT_REVIEW_WORDS} words: {under:,} ({pct(under, st.n)}). "
                f"Over {LONG_REVIEW_WORDS} words: {over:,} ({pct(over, st.n)}).\n")

    yrs = sorted(st.years)
    fig, ax = new_fig(7, 3.4)
    ax.bar([str(y) for y in yrs], [st.years[y] for y in yrs], color=BLUE, width=0.65)
    ax.set_ylabel("Reviews")
    ax.set_title("Reviews per year", loc="left", fontsize=11)
    ax.tick_params(axis="x", labelrotation=45)
    style(ax)
    save(fig, "reviews_per_year.png")
    fmt = lambda t: time.strftime("%Y-%m-%d", time.gmtime(t / 1000))
    md.append("### Date range\n" + f"{fmt(st.tmin)} to {fmt(st.tmax)} (UTC).\n\n"
              + md_table(["Year", "Reviews", "Share"], [[y, f"{st.years[y]:,}", pct(st.years[y], st.n)] for y in yrs]))

    # duplicates
    short_dup_reviews = sum(c for c in st.short_texts.values() if c > 1)
    long_dup_texts = sum(1 for c in st.long_hashes.values() if c > 1)
    long_dup_reviews = sum(c for c in st.long_hashes.values() if c > 1)
    long_extra = sum(c - 1 for c in st.long_hashes.values() if c > 1)
    top_long = [(dup_text.get(h, "?"), c) for h, c in st.long_hashes.most_common(15)]
    top_short = st.short_texts.most_common(10)
    trunc = lambda t: (t[:110] + "…") if len(t) > 110 else t
    md.append("### Duplicates, URLs and language\n"
              f"- Exact-duplicate texts of {SHORT_REVIEW_WORDS}+ words: {long_dup_texts:,} distinct texts repeated; "
              f"{long_dup_reviews:,} reviews are involved ({pct(long_dup_reviews, st.n)}), "
              f"{long_extra:,} of them redundant copies.\n"
              f"- Short reviews (<{SHORT_REVIEW_WORDS} words, e.g. \"Good\") are expected to repeat: "
              f"{short_dup_reviews:,} reviews share their exact text with another.\n"
              f"- Reviews containing a URL: {st.urls:,} ({pct(st.urls, st.n)}).\n"
              f"- Reviews containing \"(Translated by Google)\": {st.translated:,} ({pct(st.translated, st.n)}). "
              "These hold an English machine translation, so they are English-searchable but the original text may sit alongside.\n"
              f"- Approximate non-English share: {st.mostly_nonascii:,} reviews ({pct(st.mostly_nonascii, st.n)}) have more than "
              f"{int(NON_ENGLISH_SHARE * 100)}% of their letters outside ASCII A-Z. This undercounts non-English text in Latin "
              "script (Spanish, French, Italian), so treat it as a floor.\n\n"
              "Most repeated long reviews:\n\n"
              + md_table(["Copies", "Text"], [[c, trunc(t).replace("|", "/").replace("\n", " ")] for t, c in top_long])
              + "\nMost repeated short reviews:\n\n"
              + md_table(["Copies", "Text"], [[c, trunc(t).replace("|", "/").replace("\n", " ")] for t, c in top_short]))

    # users
    per_user = Counter(st.users.values())
    buckets = [("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11-50", 11, 50), ("51-100", 51, 100), ("101+", 101, 10**9)]
    brow = [[lab, f"{sum(c for k, c in per_user.items() if lo <= k <= hi):,}"] for lab, lo, hi in buckets]
    fig, ax = new_fig(6.5, 3.4)
    vals = [sum(c for k, c in per_user.items() if lo <= k <= hi) for _, lo, hi in buckets]
    ax.bar([b[0] for b in buckets], vals, color=BLUE, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("Reviews written by the user (in scope)")
    ax.set_ylabel("Users (log scale)")
    ax.set_title("Reviews per user", loc="left", fontsize=11)
    style(ax)
    save(fig, "reviews_per_user.png")
    n_users = len(st.users)
    md.append("### Users\n"
              f"{n_users:,} distinct users; mean {st.n / n_users:.2f} reviews per user, "
              f"max {max(st.users.values()):,}; {pct(per_user[1], n_users)} wrote exactly one review.\n\n"
              + md_table(["Reviews by user", "Users"], brow)
              + "\nTen most prolific users (possible spam if reviews are repetitive or one-note):\n\n"
              + md_table(["User id", "Reviews", "Distinct restaurants", "Mean rating", "Mean words", "Most repeated own text (copies)"],
                         [[u["user_id"], f"{u['n']:,}", f"{len(u['rests']):,}", f"{u['rating'] / max(u['rated'], 1):.2f}",
                           f"{u['words'] / u['n']:.1f}", f"{trunc(u['texts'].most_common(1)[0][0]).replace('|', '/')} ({u['texts'].most_common(1)[0][1]})"]
                          for u in sorted(top_users.values(), key=lambda u: -u["n"])]))

    # ---- Per-restaurant documents ----------------------------------------
    words = np.array([st.rest_words.get(i, 0) for i in rests])
    with_text = words[words > 0]
    wq = {p: float(np.percentile(with_text, p)) for p in (5, 25, 50, 75, 95, 99)}
    fig, ax = new_fig(7.5, 3.8)
    bins = np.logspace(0, np.log10(with_text.max() + 1), 50)
    ax.hist(with_text, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.set_xscale("log")
    for lim in EMBED_LIMITS:
        ax.axvline(lim, color=ORANGE, linewidth=1.5)
        ax.text(lim * 1.05, ax.get_ylim()[1] * 0.92, f"{lim} words", fontsize=9, color=INK2)
    ax.set_xlabel("Total words per restaurant if all text reviews are concatenated (log scale)")
    ax.set_ylabel("Restaurants")
    ax.set_title("Per-restaurant document length", loc="left", fontsize=11)
    style(ax)
    save(fig, "words_per_restaurant.png")
    share = lambda lim: int((with_text > lim).sum())
    md.append("## Per-restaurant documents\n"
              f"Concatenating all text reviews for each of the {len(with_text):,} restaurants with text "
              f"(words, whitespace-split):\n\n"
              + md_table(["p5", "p25", "median", "p75", "p95", "p99", "max"],
                         [[f"{wq[p]:,.0f}" for p in (5, 25, 50, 75, 95, 99)] + [f"{with_text.max():,}"]])
              + "\n" + md_table(["Embedding window", "Restaurants over (words)", "Share",
                                 f"Restaurants over (≈{TOKENS_PER_WORD}× tokens/word)", "Share "],
                                [[f"{lim}", f"{share(lim):,}", pct(share(lim), len(with_text)),
                                  f"{share(int(lim / TOKENS_PER_WORD)):,}", pct(share(int(lim / TOKENS_PER_WORD)), len(with_text))]
                                 for lim in EMBED_LIMITS])
              + "\nThe right-hand columns convert the 256/512 limits from tokens to words at an assumed "
                f"{TOKENS_PER_WORD} tokens per word, which is closer to what the model sees.\n")

    # ---- Data quality ----------------------------------------------------
    n_low = sum(r["low_evidence"] for r in rests.values())
    user_ratio = min(len(u["rests"]) / u["n"] for u in top_users.values())
    user_maxrep = max(u["texts"].most_common(1)[0][1] for u in top_users.values())
    won = sum(c for k, c in price_counts.items() if k.startswith("₩"))
    pre2005 = sum(c for y, c in st.years.items() if y < 2005)
    md.append("## Data quality issues and decisions needed\n")
    md.append(build_quality(locals()))
    (OUT_DIR / "EDA_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote", OUT_DIR / "EDA_SUMMARY.md")


def v_in(misc, value):
    """True if `value` appears under any key of a MISC dict."""
    return any(value in (v if isinstance(v, list) else [v]) for v in misc.values())


def build_quality(L):
    st, rests = L["st"], L["rests"]
    n = len(rests)
    items = [
        f"**Document length vs embedding limits.** A concatenated restaurant document has a median of "
        f"{L['wq'][50]:,.0f} words; {pct(L['share'](256), len(L['with_text']))} of restaurants exceed 256 words and "
        f"{pct(L['share'](512), len(L['with_text']))} exceed 512. One embedding per restaurant would truncate most of them. "
        "Decide: embed individual reviews (or chunks) and aggregate scores per restaurant, or select a subset of reviews "
        "per restaurant for the document.",
        f"**Short reviews.** {pct(L['under'], st.n)} of reviews have fewer than {SHORT_REVIEW_WORDS} words and the median is "
        f"{L['wp'][50]} words. Decide whether to drop reviews below a minimum length for the index, or keep them for TF-IDF only.",
        f"**Duplicate review texts.** {L['long_extra']:,} redundant copies of reviews with {SHORT_REVIEW_WORDS}+ words "
        f"({pct(L['long_extra'], st.n)}). Decide whether to de-duplicate exact repeats before indexing, otherwise they "
        "inflate term frequencies and evidence lists.",
        f"**URLs, translations and non-English text.** {st.urls:,} reviews contain URLs, {st.translated:,} are "
        f"\"(Translated by Google)\" blends, and at least {pct(st.mostly_nonascii, st.n)} are mostly non-ASCII. Decide whether to "
        "strip URLs and the translation markers, and whether to filter or separately handle non-English reviews "
        "(English-only dietary regexes miss them). Translated blends carry the original foreign-language text after "
        "\"(Original)\", so they are partly non-English even though the ASCII heuristic misses them.",
        f"**Null rating / user_id.** {st.null_rating:,} reviews have a null rating and {st.null_user:,} a null user_id "
        "(the same records). They are excluded from the rating and user statistics. Decide whether to drop them "
        "from the corpus.",
        f"**Prolific users.** The most prolific user wrote {max(st.users.values()):,} in-scope reviews. Among the top 10, "
        f"distinct restaurants are at least {L['user_ratio']:.0%} of reviews and no user repeats the same text more than "
        f"{L['user_maxrep']} times, so copy-paste spam is not evident; they look like heavy contributors. Decide whether "
        "to cap reviews per user per restaurant anyway, or leave users unweighted.",
        f"**MISC dietary attributes are almost absent.** Only vegetarian ({len(L['strict_ids']['vegetarian']):,}) and "
        f"halal ({len(L['strict_ids']['halal']):,}) attributes exist; none for vegan, kosher, gluten-free, dairy or allergens. "
        "Review text is therefore the primary evidence for most diets. Decide how much to trust the structured attributes "
        "versus mentions where they disagree (see the cross-check table).",
        f"**Sparse allergen evidence.** {L['n_low']:,} restaurants ({pct(L['n_low'], n)}) are low_evidence, and nut/shellfish "
        "allergy mentions exist for about 1% of restaurants. Decide how the app should respond when there is no evidence "
        "(say so explicitly, never imply safety).",
        f"**Review-count cap and rating-only reviews.** {L['at_cap']:,} restaurants have num_of_reviews of 9,998 or more, which "
        "looks like a source cap. `num_of_reviews` counts rating-only reviews, so it overstates the text we have; use "
        "`num_text_reviews` for evidence-based decisions.",
        f"**Field coverage.** Price is present for {pct(L['have_price'], n)} of restaurants and descriptions for "
        f"{pct(L['have_desc'], n)}, so they can't be used as hard filters. {L['won']:,} restaurants carry a won-denominated price (₩) instead of $, "
        "so price symbols are not comparable across the set. Decide whether price becomes a soft preference only and "
        "how to treat ₩ values.",
        "**Borough labels are heuristic.** Boroughs come from ZIP code, then city name. A handful of records are known to be "
        "wrong (e.g. a Brooklyn address with a Manhattan ZIP, Liberty and Ellis Island cafes). Decide whether to add a "
        "latitude/longitude sanity check, or accept the error rate.",
        f"**Timestamps.** {L['pre2005']:,} reviews predate 2005, of which {st.years[1990]:,} are dated exactly "
        "1990-12-31, which is a placeholder, not a real date. Real volume starts around 2016 and peaks in 2018-2019, "
        "and restaurant status (open/closed) is a 2021 snapshot. Decide how to treat the placeholder dates, and whether "
        "to weight recent reviews.",
        f"**Category scope.** Restaurants are matched on any category, so secondary tags such as Bar "
        f"({L['cat_counts']['Bar']:,}), Coffee shop ({L['cat_counts']['Coffee shop']:,}), Bakery "
        f"({L['cat_counts']['Bakery']:,}) and Caterer ({L['cat_counts']['Caterer']:,}) are in scope. Decide whether "
        "to keep them or restrict to restaurants whose primary category is food service.",
    ]
    return "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1)) + "\n"


if __name__ == "__main__":
    main()
