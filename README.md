# NYC Restaurant Discovery

A Text Mining / Information Retrieval course project. Users describe their
dietary needs in natural language (e.g. "nut-free brunch in Brooklyn for a
group with a vegan friend") and get ranked NYC restaurant recommendations
backed by evidence from real reviews.

## Pipeline

1. **Query parser** – GPT-4o-mini turns the free-text query into a structured `ParsedQuery`.
2. **Retrieval** – compare a TF-IDF baseline against sentence-transformer embeddings.
3. **Evidence summarizer** – GPT-4o-mini summarizes the supporting review snippets for each result.

An allergen classifier, trained on the HuggingFace `14-allergens-v2` dataset,
supports allergen-aware matching.

**Data:** Google Local Reviews 2021 (McAuley Lab). Scope starts with Manhattan
and Brooklyn; edit `BOROUGHS` in [src/config.py](src/config.py) to expand.

## Team lanes

| Lane | Owns |
|------|------|
| Data and Retrieval | `src/data`, `src/retrieval` – loading/filtering reviews, building restaurant docs, TF-IDF and embedding retrieval |
| Classification and Embeddings | `src/classifier`, `eval` – allergen classifier, embedding experiments, evaluation |
| LLM Integration and App | `src/llm`, `app` – query parser, evidence summarizer, Streamlit UI |

Shared dataclasses live in [src/schemas.py](src/schemas.py); agree on changes there before editing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then add your OPENAI_API_KEY
```

Place downloaded datasets in `data/` (git-ignored).

Run the app from the repo root:

```bash
streamlit run app/streamlit_app.py
```

## Data pipeline

Source: [Google Local Reviews 2021](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/) (McAuley Lab), New York state. Run from the repo root, in this order:

```bash
B=https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal
mkdir -p data/raw

# 1. Business metadata (58 MB)
curl -L --fail -o data/raw/meta-New_York.json.gz $B/meta-New_York.json.gz

# 2. (optional) Explore metadata: fields, restaurant categories, borough counts
python -m src.data.explore_meta

# 3. Filter to open Manhattan/Brooklyn restaurants (de-duplicated)
python -m src.data.filter_meta

# 4. Complete reviews (2.47 GB); -C - resumes an interrupted download
curl -L --fail -C - -o data/raw/review-New_York.json.gz $B/review-New_York.json.gz
gunzip -t data/raw/review-New_York.json.gz   # verify

# 5. Keep text reviews for those restaurants (streams 33.5M reviews; takes minutes)
python -m src.data.filter_reviews

# 6. Add the low_evidence flag
python -m src.data.build_final

# 7. Dietary-term frequency check
python -m src.data.dietary_terms
```

Review download URL: https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/review-New_York.json.gz

All scripts stream their input line by line. The borough scope comes from `BOROUGHS` in `src/config.py`; restaurant categories, the permanently-closed exclusion and the borough classifier (ZIP prefix, then city name) live in `src/data/filter_meta.py`.

### Outputs (in `data/processed/`, git-ignored)

| File | Contents | Expected size |
|------|----------|---------------|
| `restaurants_meta.jsonl` | `gmap_id`, `name`, `address`, `borough`, `categories` (list), `avg_rating`, `num_of_reviews`, `latitude`, `longitude` | 15,447 restaurants (9,335 Manhattan, 6,112 Brooklyn) |
| `reviews_filtered.jsonl` | `gmap_id`, `user_id`, `rating`, `time` (Unix ms), `text` | 2,759,578 text reviews |
| `restaurants_final.jsonl` | all `restaurants_meta` fields plus `num_text_reviews` (int) and `low_evidence` (bool, true if fewer than 5 text reviews) | 15,447 restaurants, 633 `low_evidence` |
| `dietary_term_counts.csv` | `term_group`, `reviews`, `pct_reviews`, `restaurants`, `pct_restaurants`, `top_restaurants` | 11 term groups |

Filtering notes: "restaurant" means a category containing "restaurant" (excluding supply/equipment-type categories) or one of Cafe, Deli, Diner, Bistro, etc. Permanently closed restaurants (3,992 in scope) are dropped; temporarily closed ones are kept. 44 duplicate `gmap_id` records were collapsed. Reviews with null or empty text (2,261,223 for in-scope restaurants) are dropped. 25 restaurants have no text reviews.

## Week 1 findings

Share of the 2,759,578 text reviews, and of the 15,447 restaurants, mentioning each term group (case-insensitive, word-bounded):

| Term group | Reviews | Restaurants | % of restaurants |
|---|---|---|---|
| vegan | 23,633 | 3,888 | 25.2% |
| vegetarian | 16,299 | 4,681 | 30.3% |
| gluten-free (incl. celiac) | 7,300 | 2,320 | 15.0% |
| kosher | 5,867 | 671 | 4.3% |
| halal | 3,597 | 802 | 5.2% |
| allergy (general) | 3,114 | 2,058 | 13.3% |
| dairy-free / lactose | 692 | 548 | 3.5% |
| allergic to (any) | 664 | 590 | 3.8% |
| cross-contamination | 260 | 227 | 1.5% |
| nut allergy | 203 | 177 | 1.1% |
| shellfish allergy | 90 | 84 | 0.5% |

Caveats:
- **Allergen mentions are sparse.** Nut and shellfish allergy appear for only about 1% of restaurants, so review text alone cannot cover allergen queries; the classifier lane will need other signals.
- **Vegetarian and kosher are noisy.** "Vegetarian friendly" or one vegetarian dish is not a dietary guarantee, and the top kosher hits are largely kosher-certified places.
- **"Allergic to (any)" and "allergy (general)" are imprecise**: many hits are non-major allergens (garlic, strawberries, spice) or unspecific.
- **A mention does not imply safety.** Matches include complaints ("misinformation... led two intolerants to be sick"), so evidence must be read in context, not counted.
- Reviews average 26.7 words; 608 restaurants have fewer than 5 text reviews (plus 25 with none).

## Layout

```
src/        config, schemas, and pipeline modules (data, retrieval, classifier, llm)
app/        Streamlit app
eval/       evaluation scripts
notebooks/  exploration
data/       local datasets (git-ignored)
```
