# Stance-aware re-ranking (PROVISIONAL)

> **PROVISIONAL:** relevance labels are LLM-judge labels, not yet validated against human labels; stance labels come from `llm:openai/gpt-oss-20b` and are not validated either.

- Labels: `eval/llm_labels.jsonl` (1113 labeled pairs); queries: `eval/queries_final.csv`
- **DEV-ONLY RUN:** 20 of 20 dev queries had cached stance labels (single_diet 6, mixed_group 6, allergy 6, diet_plus_filter 2); 0 dev queries still unlabeled. The test queries were not labeled, scored or used.
- Weights were tuned on the dev queries only (grid of 216 settings, w_pos axis widened with [0.9, 1.2, 1.8], objective: mean dev NDCG@10). Chosen: w_pos 1.8, w_neg 0.4, w_strong_neg 0.0. Starting weights (untuned): {'w_pos': 0.3, 'w_neg': 0.6, 'w_strong_neg': 0.1}.
- Re-ranks the top 20 of semantic_capped; per restaurant, its evidence sentences are labeled accommodation / mention / negative / not_about_allergy toward the query (`src/retrieval/stance_rerank.py`).
- Unjudged top-10 results count as 0 (not relevant), for every system.
- Labels by source: openrouter/crusoe 559, groq 473, rule 81

## Pre-registered choices

- **Candidate depth = 20** (the top 20 of semantic_capped are re-ranked), fixed in `src/config.py` on 2026-11-13 before any test-split stance result existed, and used unchanged for tuning on the dev queries and for the held-out test evaluation. It was chosen to bound judging cost, not tuned on any split.
- Dev/test split: 20/15 by seed 6165, fixed before tuning. Weights are tuned on dev only; the test queries are judged and scored once, after the weights are frozen.
- Test-split pairs outside the existing labels are judged only after the weights are frozen.

## Dev split (weights were tuned here: optimistic) (n=20)

| System | NDCG@10 [95% CI] | P@5 lenient [95% CI] | P@10 lenient [95% CI] | NDCG@10 vs semantic_capped [95% CI] | p (sign-flip) | better / worse / tie | Unjudged |
|---|---|---|---|---|---|---|---|
| semantic_capped | 0.598 [0.477, 0.716] | 0.660 [0.500, 0.810] | 0.650 [0.500, 0.800] | — | — | — | 0 |
| semantic_capped_stance | 0.695 [0.575, 0.802] | 0.800 [0.670, 0.910] | 0.710 [0.570, 0.845] | +0.097 [0.021, 0.178] | 0.030 | 12 / 3 / 5 | 0 |
| stance_initial_weights | 0.651 [0.536, 0.760] | 0.730 [0.600, 0.860] | 0.700 [0.565, 0.830] | +0.053 [0.009, 0.101] | 0.038 | 12 / 1 / 7 | 0 |

NDCG@10 by category (tuned weights):

| Category | n | semantic_capped | stance (tuned) | Δ NDCG@10 | p (sign-flip) |
|---|---|---|---|---|---|
| single_diet | 6 | 0.754 | 0.799 | +0.045 | 0.251 |
| mixed_group | 6 | 0.622 | 0.662 | +0.040 | 0.191 |
| allergy | 6 | 0.393 | 0.685 | +0.291 | 0.062 |
| diet_plus_filter | 2 | 0.671 | 0.512 | -0.159 | 1.000 |

## Before / after examples

Tuned weights; top 5 on the full query (all-35 view).

### peanut allergy (`Q20`: "restaurant careful about peanut allergies")

| # | Before: semantic_capped | label | After: stance re-rank | label |
|---|---|---|---|---|
| 1 | Pig and Khao (+0 / −3) | 1 | Senza Gluten (+3 / −0) | 1 |
| 2 | Mission (+0 / −3) | 0 | Virgil's Real BBQ - Times Square (+3 / −0) | 2 |
| 3 | Superiority Burger (+0 / −0) | 1 | Buddakan (+2 / −1) | 1 |
| 4 | Ladybird (+1 / −0) | 2 | Pig and Khao (+0 / −3) | 1 |
| 5 | Five Guys (+0 / −0) | 0 | Ladybird (+1 / −0) | 2 |

A sentence labeled negative in the old top 5: "DO NOT GO IF YOU HAVE A SEVERE PEANUT ALLERGY."

(+ accommodation / − negative sentences among those labeled; label = LLM-judge relevance 0/1/2, ? = unjudged)

### celiac safe (`Q22`: "safe for celiac disease, no cross contamination")

| # | Before: semantic_capped | label | After: stance re-rank | label |
|---|---|---|---|---|
| 1 | Senza Gluten (+1 / −0) | 2 | Senza Gluten Cafe & Bakery (+4 / −0) | 2 |
| 2 | Friedman's Herald Square (+3 / −1) | 2 | Cafe Eighty Two (+4 / −0) | 2 |
| 3 | Lilli and Loo (+2 / −1) | 1 | Friedman's Herald Square (+3 / −1) | 2 |
| 4 | Noglu (+3 / −0) | 2 | Noglu (+3 / −0) | 2 |
| 5 | Senza Gluten Cafe & Bakery (+4 / −0) | 2 | Bistango (+4 / −0) | 2 |

A sentence labeled negative in the old top 5: "While many restaurants have gluten free items on their menus, most fail to recognize that celiac disease isn't a fad, and cross contamination is a real danger to those of us who have it."

(+ accommodation / − negative sentences among those labeled; label = LLM-judge relevance 0/1/2, ? = unjudged)

## New pairs needing judge labels

The tuned re-ranker's top-10 contains 0 (query, restaurant) pairs with no label (0 on test queries, 0 on dev). Run `python -m eval.make_pool` and then `python -m eval.llm_judge` (resumable; labels only new pairs), then rerun this script.

## Limitations

- **Same 35 queries for development and the final claim.** Only the dev/test split protects the test numbers, and 15 queries is a small test set: CIs are wide and per-category cells (4 to 11 queries) are too small for firm conclusions.
- **The labels added for this evaluation come from a single provider** (Groq, `gpt-oss-120b`, same prompt and settings as before), whereas the original labels came from OpenRouter/Crusoe, Groq and a no-evidence rule; provider-to-provider differences in labeling were not measured, so the new top-20 pairs and the old pairs may be labeled slightly differently.
- **Labels are LLM-judge labels**, not validated against humans, and the stance labeler is the same family of model; their errors can be correlated, which flatters the re-ranker.
- **Unjudged results count as 0**, which penalizes any setting that surfaces restaurants outside the judged pool, and biases tuning toward the existing pool. The pool grows only after the new pairs above are labeled.
- **Evidence is thin:** at most 3 reviews per restaurant from retrieval, 4 sentences kept; a restaurant with no allergy sentence gets no stance signal either way.
- **Candidate set is fixed:** only the base top-20 are re-ranked, so a restaurant the embedding missed cannot be recovered.
- **Stance labels are not allergy-safety claims.** An accommodation sentence is one reviewer's statement; the app's allergy guard and note stay in force.
- Weights are tuned for NDCG@10 on this label scheme; they may not transfer to other queries or boroughs.
