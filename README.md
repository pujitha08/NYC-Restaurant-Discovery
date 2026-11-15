# NYC Restaurant Discovery

A Text Mining / Information Retrieval course project. Users describe their
dietary needs in natural language (e.g. "nut-free brunch in Brooklyn for a
group with a vegan friend") and get ranked NYC restaurant recommendations
backed by evidence from real reviews.

**Scope.** Built for dietary needs and food allergies: diets, allergens and mixed groups. It does not check opening
hours, prices or availability; those words in a request guide the search but are not verified from reviews. Reviews are
Google reviews through September 2021, so results are anecdotal and are not safety guarantees. For allergies, always
confirm with the restaurant.

## Pipeline

1. **Query parser** – a Groq-hosted LLM (`GROQ_MODEL`, default `openai/gpt-oss-20b`) turns the free-text query into a structured `ParsedQuery` (`src/llm/query_parser.py`).
2. **Retrieval** – TF-IDF baselines (`doc_tfidf`, `chunk_tfidf_*`) and sentence-transformer retrieval over review chunks (`semantic_log`, `semantic_capped`; all-MiniLM-L6-v2). `semantic_capped` is the app default.
3. **Evidence summarizer** – the same Groq model summarizes the retrieved review excerpts of the top results as support / contradiction / uncertainty claims, each citing a review (`src/llm/evidence_summarizer.py`). Allergy-sensitive queries never get a "Supported" status.

4. **Stance re-ranker (optional, experimental)** – `semantic_capped_stance` re-orders the top 20 of `semantic_capped` by
   what individual review sentences say about the need (accommodation / mention / negative / not about it). It is
   selectable in the app (Advanced settings) and in Compare rankings, but **it is not the default**, and its benefit over
   `semantic_capped` has not been established: the held-out test evaluation is still pending (see "Stance re-ranker").

**Classifier lane:** `src/classifier/allergen_tagger.py` is a character n-gram TF-IDF + one-vs-rest logistic regression
model that tags which FDA major allergens a text mentions; it is trained on the HuggingFace `14-allergens-v2` menu-item
dataset, so it tags mentions, not safety. `src/classifier/stance.py` trains a logistic regression on
all-MiniLM-L6-v2 sentence embeddings to label review sentences as `accommodation`, `mention`, `negative` or
`not_about_allergy`, from hand-labeled sentence sheets (`eval/make_sentence_sheets.py`).

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

cp .env.example .env   # then set GROQ_API_KEY (and GROQ_MODEL=openai/gpt-oss-20b)
# Optional, LLM judge only: OPENROUTER_API_KEY / NVIDIA_API_KEY (see .env.example)
```

Place downloaded datasets in `data/` (git-ignored). Without a Groq key the app still searches, without parsing or summaries.

## How to run

From the repo root, after setup:

```bash
# 1. Build the data (see "Data pipeline" below for the downloads); ends with restaurant docs and review chunks
python -m src.data.build_docs

# 2. Build the semantic index (slow; writes data/processed/semantic/)
python -m src.retrieval.semantic --build

# 3. Evaluation: build the judging pool and per-system runs, label pairs with the LLM judge, compute metrics
python -m eval.make_pool
python -m eval.llm_judge --batch-size 5          # resumable; needs a judge provider key
python -m eval.metrics --labels eval/llm_labels.jsonl

# 4. Run the app
streamlit run app/streamlit_app.py

# Tests
python -m pytest -q
```

The reported metrics use provisional LLM-judge labels that have not yet been validated against human labels (see
[reports/metrics_summary.md](reports/metrics_summary.md)). [DEMO.md](DEMO.md) has the demo script.

## The app

`streamlit run app/streamlit_app.py`. A search shows:

- **Parsed request chips** (diets, allergens, borough) and an expander, "How your request was understood", that lists only
  the fields that were parsed. If no diet, allergen or group need was detected, a notice says results are ranked by general
  review match and are not checked against a diet or allergen (the results are still shown).
- **A results summary line** under the "N restaurants" heading, for example "Of the top 3 summarized: 1 restaurant has
  reviews that warn about a peanut allergy · 2 show no direct evidence". It is computed from the status badges and stance
  counts already on the cards (no model call), counts only restaurants that have a summary or stance counts, hides zero
  parts and never says a restaurant is safe. A muted line notes that preferences such as opening hours or cuisine are not
  verified from reviews.
- **Result cards** with an evidence status badge (Allergy requests never show "Supported"), the top claim and its quote,
  "Show all evidence", and a **coverage row**: one chip per parsed diet, allergen and group need saying whether the shown
  excerpts mention it ("vegan: mentioned in 3 excerpts" / "halal: not mentioned in shown excerpts"). It uses the same term
  matcher as the highlighting, counts mentions (not verdicts) and never implies safety. Parsed preferences are listed
  separately as "Not verified from reviews: ...". Cards without a summary say "Summary on request" (click Explain).
- **Demo mode** (checkbox): answers the four example queries from `data/demo_cache/*.json` (public review text already shown
  in the app; no keys) without calling the language model. Cached results are labeled "cached demo result" with the save
  date; the status line just says "Done". Cached summaries come from one earlier live run and predate the latest summarizer
  prompt wording fix; they have not been regenerated.
- **Compare rankings** (checkbox): `semantic_capped` and `semantic_capped_stance` side by side with rank-change marks and a
  sentence-stance bar per restaurant. With Demo mode on it only uses stance labels already saved on disk.
- **Advanced settings** (collapsed): retrieval method (default `semantic_capped`) and "include restaurants with fewer than
  5 reviews".

## Stance re-ranker

`src/retrieval/stance_rerank.py` re-orders the top 20 of `semantic_capped`:
`score = base_norm + w_pos*accommodation_share - w_neg*negative_share - w_strong_neg*(strong negative, allergy queries)`.
Sentence stance labels come from a labeler chosen by the explicit setting `STANCE_LABELER` in `src/config.py`:
`"llm"` (default: Groq `openai/gpt-oss-20b`, labels cached on disk in `data/processed/stance_rerank/`) or `"classifier"`
(the logistic regression of `src/classifier/stance.py`). Whether the classifier's model file exists never changes the choice.

Evaluation design (fixed before looking at test results):
- Candidate depth is pre-registered at 20 (`STANCE_RERANK["top_n"]`), used for tuning and for the test.
- The 35 evaluation queries are split 20 dev / 15 test by seed 6165. Weights were tuned on the dev queries only.
- **Frozen weights:** [reports/stance_rerank_frozen.json](reports/stance_rerank_frozen.json) records the weights chosen
  on dev (w_pos 0.9, w_neg 0.0, w_strong_neg 0.2), the rule used (best dev NDCG@10 with w_pos at most 0.9), the weight
  grids and the dev scores. These are dev numbers on the queries the weights were tuned on, so they are optimistic.
  `STANCE_RERANK` in `src/config.py` still holds the original starting weights; the frozen ones are not adopted as the app
  setting.
- **Pending:** the held-out evaluation on the 15 test queries (judging the test candidate pairs and running the evaluation
  once with the frozen weights). Until it is done, nothing here claims that the stance re-ranker beats `semantic_capped`.

```bash
python -m eval.label_stance [--dev-only]    # budgeted, resumable sentence labeling (Groq, ledger in data/processed/stance_rerank/)
python -m eval.tune_stance_rerank --labels eval/llm_labels.jsonl --dev-only --cache-only
```

## Reports

| File (reports/) | What it is |
|---|---|
| metrics_summary.md, metrics_per_query.csv, metrics_ndcg_p10.png, retrieval_comparison.json | retrieval metrics per system (provisional LLM-judge labels) |
| judge_validation.md | LLM judge vs human raters: agreement, kappa, alpha; human-vs-human section |
| no_evidence_audit.md | audit of "no evidence found" judge labels, incl. the extended top-20 dev pool |
| allergen_tagger.md | allergen tagger held-out menu results and limitations |
| sentence_check.md | labeled review sentences: rater agreement, stance classifier CV, tagger on reviews |
| stance_rerank_dev_grid1.md / _grid2.md (+ `_weights.json`) | dev-only tuner runs (grid 1, and grid 2 with `w_pos` widened once) |
| stance_rerank_frozen.json | frozen dev weights, the rule, grids and dev scores |
| failure_examples.md | before/after failure cases |
| eda/ | exploratory data analysis |

## Privacy and what not to commit

Never commit `.env` or any API key, the rater label files (`eval/labels/*`, except the two labeling guides), LLM judge labels
(`eval/llm_labels*.jsonl`), the judging pools (`eval/pool_extended.csv`, `eval/pool_test.csv`), request logs (`logs/`) or
`data/` (except `data/demo_cache/`). The raters must not see each other's files or the LLM labels while labeling. `.gitignore`
enforces this; use named `git add` paths. No test-split result or label should influence the frozen weights.

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

# 8. (optional) EDA figures and summary -> reports/eda/
python -m src.data.eda

# 9. Clean reviews and recompute low_evidence (see "Cleaning decisions")
python -m src.data.clean_reviews

# 10. Build per-restaurant documents and review chunks
#     (downloads the all-MiniLM-L6-v2 tokenizer via huggingface_hub, installed with sentence-transformers)
python -m src.data.build_docs
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

| `reviews_clean.jsonl` | `review_id` (line number in `reviews_filtered.jsonl`), `gmap_id`, `user_id`, `rating`, `time` (null for placeholder dates), `text` (cleaned) | 2,156,429 reviews |
| `restaurants_clean.jsonl` | `restaurants_final` fields with `num_clean_reviews` added and `low_evidence` recomputed on cleaned reviews | 15,447 restaurants, 826 `low_evidence` |
| `restaurant_docs.jsonl` | `RestaurantDoc` in `src/schemas.py`: `gmap_id`, `name`, `borough`, `address`, `categories`, `avg_rating`, `num_clean_reviews`, `low_evidence`, `text` (all clean reviews, newline-joined) | 15,447 docs (68 with empty text) |
| `review_chunks.jsonl` | `Chunk` in `src/schemas.py`: `chunk_id`, `gmap_id`, `text`, `review_ids`, `times` | 605,540 chunks |

Filtering notes: "restaurant" means a category containing "restaurant" (excluding supply/equipment-type categories) or one of Cafe, Deli, Diner, Bistro, etc. Permanently closed restaurants (3,992 in scope) are dropped; temporarily closed ones are kept. 44 duplicate `gmap_id` records were collapsed. Reviews with null or empty text (2,261,223 for in-scope restaurants) are dropped. 25 restaurants have no text reviews.

## Cleaning decisions

`clean_reviews.py` turns the 2,759,578 filtered reviews into 2,156,429 clean ones (78.1%). Each review is counted at the first stage that removes it:

| Stage | Rule | Removed |
|---|---|---|
| Exact duplicate rows | same `gmap_id`, `user_id`, `time` and text (copies from the source) | 65,158 |
| Null fields | `rating` or `user_id` is null | 1,995 |
| Too short | under 5 words after cleaning | 528,708 |
| Template reviews | cleaned text (case-insensitive) occurs 20+ times statewide; all copies dropped (116 distinct texts) | 4,754 |
| Duplicate in restaurant | same cleaned text of 5+ words in one restaurant; first kept | 2,534 |

Text cleaning, applied before the length check:
- Google-translated reviews keep only the translation. The "(Translated by Google)" marker and everything from "(Original)" onward are dropped (236,007 reviews were affected; most of them are then removed as too short).
- Whitespace is collapsed to single spaces.
- The placeholder date 1990-12-31 is set to null (232 kept reviews), since it is not a real review date.

Cross-restaurant copies below 20 occurrences are kept: the same text can be a legitimate review of two different places.

Restaurants: 826 now have fewer than 5 clean reviews and are `low_evidence` (633 before cleaning), including 68 with none. Those 68 get an empty-text doc but no chunks.

Chunks hold up to 5 consecutive reviews of one restaurant, closed once they reach about 150 words. Every chunk is at most 256 tokens under the all-MiniLM-L6-v2 tokenizer (special tokens included); a review longer than that is split into several chunks. Median chunk: 115 words, 147 tokens.

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
app/          Streamlit app (streamlit_app.py), its pipeline (pipeline.py: parse -> retrieve -> summarize), demo_cache.py,
              compare.py, chips.py, coverage.py, highlight.py
src/
  config.py, schemas.py   settings and shared dataclasses
  data/                   filtering, cleaning, EDA, restaurant docs and review chunks
  retrieval/              tfidf.py, tfidf_chunks.py, semantic.py, evidence.py (snippet selection), stance_rerank.py
  classifier/             allergen_tagger.py (FDA allergen mentions), stance.py (review-sentence stance)
  llm/                    groq_client.py, nvidia_client.py, openrouter_client.py, query_parser.py, evidence_summarizer.py
eval/         queries, judging pool, LLM judge, metrics, significance tests, agreement and validation scripts,
              stance labeling and tuning (label_stance.py, tune_stance_rerank.py), audits, answer_eval.py
              (eval/labels/ holds rater material and is git-ignored except the labeling guides)
reports/      metrics, validation, audits, stance re-ranker runs and frozen weights, EDA, failure examples (see "Reports")
docs/         LLM integration notes
examples/     LLM pipeline demo script
tests/        pytest suite
notebooks/    exploration
data/         local datasets (git-ignored)
```
