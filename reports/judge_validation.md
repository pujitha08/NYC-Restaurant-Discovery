# Judge validation: human labels vs the LLM judge

- Human labels: `eval/labels/labels_pujitha.csv` (120 rows; 1 with a blank label skipped; 119 labeled). Labeled blind to the LLM labels, by one rater.
- LLM labels: `eval/llm_labels.jsonl` (1074 labeled pairs, by source: openrouter/crusoe 559, groq 434, rule (no evidence shown) 81)
- **Matched: 119 of 119 labeled rows** (by review ids: 106, name + borough: 13); 0 could not be matched and are excluded.
- Relevance scale 0/1/2 (eval/RUBRIC.md). Bootstrap 95% CIs resample items (seed 6165).

## Agreement

| Statistic | Value | 95% CI |
|---|---|---|
| n matched | 119 | |
| Raw agreement | 0.748 | [0.672, 0.824] |
| Cohen's kappa (unweighted) | 0.618 | |
| Cohen's kappa (quadratic-weighted) | 0.735 | [0.617, 0.834] |
| Krippendorff's alpha (ordinal) | 0.734 | |
| Mean label: human / LLM | 1.092 / 0.958 | |
| LLM above / below human | 10 / 20 | |
| Off by one step / by two | 24 / 6 | |

## Confusion matrix (rows: human, columns: LLM)

| human \ LLM | 0 | 1 | 2 | total |
|---|---|---|---|---|
| 0 | 35 | 2 | 0 | 37 |
| 1 | 9 | 17 | 8 | 34 |
| 2 | 6 | 5 | 37 | 48 |

## Data-quality flag: the judge reported no evidence where you saw some

29 matched pairs have an LLM call (not the no-evidence rule) whose reason says "no evidence found" and cites no review, although your sheet showed retrieved or fallback evidence for them (mean human label 0.552; samples S072, S109, S017, S037, S104, S085, S019, S068, S077, S049, S088, S118, S116, S084, S117, S064, S105, S074, S078, S101, S087, S095, S030, S070, S075, S076, S114, S071, S056). The judge may have been shown different (or empty) evidence than the sheet; this should be checked against the pool the judge actually read before trusting those labels. Sensitivity analysis without these pairs: n = 90, raw agreement 0.789, quadratic kappa 0.809 (with them: 0.748 and 0.735).

## Mean label by LLM label source

| Source | n | Mean human | Mean LLM | Agreement |
|---|---|---|---|---|
| openrouter/crusoe | 106 | 1.226 | 1.075 | 0.717 |
| rule (no evidence shown) | 13 | 0.000 | 0.000 | 1.000 |

## Agreement by query category

| Category | n | Agreement | Kappa (quadratic) | Mean human | Mean LLM | LLM above / below |
|---|---|---|---|---|---|---|
| single_diet | 43 | 0.814 | 0.828 | 1.349 | 1.372 | 5 / 3 |
| mixed_group | 30 | 0.633 | 0.589 | 1.133 | 1.033 | 4 / 7 |
| allergy | 34 | 0.765 | 0.649 | 0.735 | 0.412 | 0 / 8 |
| diet_plus_filter | 12 | 0.750 | 0.534 | 1.083 | 0.833 | 1 / 2 |

Cells with few queries are noisy; read the category rows as descriptions of this sample, not estimates.

## Largest disagreements (human and LLM two steps apart)

### Human 2, LLM 0 (the LLM is far stricter): 6

| Sample | Query | Restaurant | Source | Your note | LLM reason |
|---|---|---|---|---|---|
| S006 | Q03 gluten-free options (single_diet) | Gluten Free Food Brooklyn | openrouter/crusoe | — | Most recent review states the restaurant is not wholly gluten‑free, contradicting the query. |
| S072 | Q19 which restaurants are safe if I'm allergic to peanuts (allergy) | Lan Cafe | openrouter/crusoe | The evidence suggests the restraurant is not peanut free | no evidence found |
| S084 | Q23 nut-free bakery, no peanuts or tree nuts (allergy) | Texas Chicken & Burgers | openrouter/crusoe | — | no evidence found (restaurant is not a bakery, fails category filter) |
| S087 | Q24 bakery safe for tree nut allergy (allergy) | Bengal Tiger | openrouter/crusoe | — | no evidence found |
| S095 | Q27 vegan brunch (diet_plus_filter) | Bklyn Wild | openrouter/crusoe | — | no evidence found |
| S105 | Q30 kosher bakery (diet_plus_filter) | Fruits-A-Plenty Inc | openrouter/crusoe | — | no evidence found |

## Limitations

- **One rater.** No human-human agreement can be computed, so these numbers measure agreement with one person's reading of the rubric, not with a consensus; that person's own consistency is unknown.
- **n = 119** matched pairs. The CIs above are wide; per-category cells (4 to 11 queries) and the confusion-matrix cells are small, and the kappa values depend heavily on a few pairs.
- **A sample, not the pool.** The rater labeled a stratified sample of 119 of the 1,074 judged pairs (about 11%; drawn by `eval/make_sample.py`), so errors in the unsampled pairs are unmeasured.
- **The sample predates the new labels.** It was drawn from the earlier judging pool; the pairs added later for the stance re-ranker (top-20 candidates, a single provider: Groq, `gpt-oss-120b`) are not validated here, and provider-specific behaviour is only visible through the 'by source' table above.
- **Same judge model throughout.** All LLM labels use one model family and one prompt, so shared blind spots (for example, treating a mention of the topic as support) would not show up as disagreement between LLM sources.
- **The rater saw the evidence shown to the judge**, not the full review text of each restaurant; both are limited to what retrieval returned, so agreement says nothing about whether the shown reviews were representative.
- Matching rows to restaurants relies on the review ids cited in the evidence; rows without evidence are matched by name and borough only when that is unambiguous.
