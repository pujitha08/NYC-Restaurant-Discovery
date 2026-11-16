# Judge validation: human labels vs the LLM judge

- Human labels: `eval/labels/labels_pujitha.csv` (120 rows; 0 with a blank label skipped; 120 labeled). Labeled blind to the LLM labels, by two raters (second rater: see below).
- LLM labels: `eval/llm_labels.jsonl` (1224 labeled pairs, by source: groq 584, openrouter/crusoe 559, rule (no evidence shown) 81)
- **Matched: 120 of 120 labeled rows** (by review ids: 107, name + borough: 13); 0 could not be matched and are excluded.
- Relevance scale 0/1/2 (eval/RUBRIC.md). Bootstrap 95% CIs resample items (seed 6165).

## Agreement

| Statistic | Value | 95% CI |
|---|---|---|
| n matched | 120 | |
| Raw agreement | 0.750 | [0.667, 0.825] |
| Cohen's kappa (unweighted) | 0.620 | |
| Cohen's kappa (quadratic-weighted) | 0.737 | [0.620, 0.834] |
| Krippendorff's alpha (ordinal) | 0.736 | |
| Mean label: human / LLM | 1.100 / 0.967 | |
| LLM above / below human | 10 / 20 | |
| Off by one step / by two | 24 / 6 | |

## Confusion matrix (rows: human, columns: LLM)

| human \ LLM | 0 | 1 | 2 | total |
|---|---|---|---|---|
| 0 | 35 | 2 | 0 | 37 |
| 1 | 9 | 17 | 8 | 34 |
| 2 | 6 | 5 | 38 | 49 |

## Data-quality flag: the judge reported no evidence where you saw some

29 matched pairs have an LLM call (not the no-evidence rule) whose reason says "no evidence found" and cites no review, although your sheet showed retrieved or fallback evidence for them (mean human label 0.552; samples S072, S109, S017, S037, S104, S085, S019, S068, S077, S049, S088, S118, S116, S084, S117, S064, S105, S074, S078, S101, S087, S095, S030, S070, S075, S076, S114, S071, S056). The judge may have been shown different (or empty) evidence than the sheet; this should be checked against the pool the judge actually read before trusting those labels. Sensitivity analysis without these pairs: n = 91, raw agreement 0.791, quadratic kappa 0.811 (with them: 0.750 and 0.737).

## Mean label by LLM label source

| Source | n | Mean human | Mean LLM | Agreement |
|---|---|---|---|---|
| openrouter/crusoe | 107 | 1.234 | 1.084 | 0.720 |
| rule (no evidence shown) | 13 | 0.000 | 0.000 | 1.000 |

## Agreement by query category

| Category | n | Agreement | Kappa (quadratic) | Mean human | Mean LLM | LLM above / below |
|---|---|---|---|---|---|---|
| single_diet | 43 | 0.814 | 0.828 | 1.349 | 1.372 | 5 / 3 |
| mixed_group | 30 | 0.633 | 0.589 | 1.133 | 1.033 | 4 / 7 |
| allergy | 34 | 0.765 | 0.649 | 0.735 | 0.412 | 0 / 8 |
| diet_plus_filter | 13 | 0.769 | 0.578 | 1.154 | 0.923 | 1 / 2 |

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

## Human vs human: Pujitha and Kang

**Correction (2026-11-15).** An earlier version of this report used a `labels_kang.csv` that was not the rater's human labeling (the wrong file was sent). Every number computed from it is withdrawn, including human-vs-human quadratic kappa 0.788 / 0.790 and Kang vs LLM quadratic kappa 0.899. The sections below use the rater's actual file; Pujitha vs LLM was never affected.

- Second rater: `eval/labels/labels_kang.csv` (120 rows; 0 blank). Per the project workflow neither rater was shown the LLM labels (not verifiable from the files; see Limitations).
- **Shared labeled pairs: 120** (labeled by Pujitha only: 0; by Kang only: 0). Same sample sheet; matched on `sample_id`.

| Statistic | Value | 95% CI |
|---|---|---|
| Exact agreement | 0.750 | [0.667, 0.825] |
| Cohen's kappa (unweighted) | 0.614 | |
| Cohen's kappa (quadratic-weighted) | 0.789 | [0.690, 0.869] |
| Krippendorff's alpha (ordinal) | 0.783 | |
| Mean label: Pujitha / Kang | 1.100 / 1.108 | |
| Off by one step / by two | 27 / 3 | |

Confusion matrix (rows: Pujitha, columns: Kang):

| Pujitha \ Kang | 0 | 1 | 2 | total |
|---|---|---|---|---|
| 0 | 34 | 2 | 1 | 37 |
| 1 | 8 | 13 | 13 | 34 |
| 2 | 2 | 4 | 43 | 49 |

### Disagreements: 30 of 120

| Sample | Query | Restaurant | Pujitha | Kang | Pujitha's note | Kang's note |
|---|---|---|---|---|---|---|
| S072 | Q19 which restaurants are safe if I'm allergic to peanuts | Lan Cafe | 2 | 0 | The evidence suggests the restraurant is not peanut free | menu doesn't list out peanut as an ingredient |
| S084 | Q23 nut-free bakery, no peanuts or tree nuts | Texas Chicken & Burgers | 2 | 0 | — | no mentions of being nut free |
| S109 | Q32 dinner in manhattan where my vegan girlfriend and i can both eat, i eat meat | 99 Favor Taste | 0 | 2 | — | — |
| S006 | Q03 gluten-free options | Gluten Free Food Brooklyn | 2 | 1 | — | conflicting reviews on gluten options |
| S011 | Q05 halal restaurant | Rocca Cafe & Lounge | 1 | 2 | doesn’t confirm the availability of halal food | — |
| S018 | Q06 restaurants that serve halal meat | Milk And Honey Cafe | 1 | 0 | mixed reviws about halal food's availability | doesn't serve halal |
| S025 | Q08 vegetarian friendly menu | Pure Thai Cookhouse | 0 | 1 | — | conflicting reviews on vegetarian food |
| S038 | Q10 good for lactose intolerant people | Exotic Roti Express | 1 | 2 | This is specific to the chef but not to the restraurant altogether | — |
| S039 | Q11 pescatarian seafood restaurant | Randazzo's Clam Bar | 1 | 2 | it doesn't say anything about pescatarian but provides evidence for seafood options | — |
| S040 | Q11 pescatarian seafood restaurant | Oyster | 1 | 2 | the evidence only suggests the availability of seafood options | — |
| S045 | Q13 vegan but my friends eat meat | Franchia Vegan Café | 1 | 2 | the evidence doesn’t provide any clarity about vegan options | — |
| S046 | Q13 vegan but my friends eat meat | Spicy Moon Vegan Szechuan | 1 | 2 | the evidence says the restaurant is pure vegan and doesn't have any meat options | — |
| S048 | Q13 vegan but my friends eat meat | SESTINA | 1 | 2 | only vegan and doesn't have options for people who eat meat | — |
| S049 | Q13 vegan but my friends eat meat | Donghai | 1 | 0 | doesn’t provide any details about vegan options availability | review doesn't really say anything |
| S054 | Q14 group dinner with a vegetarian, a halal eater and a gluten-free guest | by CHLOE. - West Village | 1 | 2 | the evidence is not clear about any halal options | — |
| S055 | Q15 place where vegans and meat lovers can both eat | P.S. Kitchen | 1 | 2 | doesn't provide any details about meat options being available | — |
| S064 | Q18 date night, one of us is vegetarian and the other loves steak | K.Rico Steakhouse | 1 | 0 | doesn’t give any information about vegetarian options | not for vegetarians |
| S067 | Q18 date night, one of us is vegetarian and the other loves steak | Boucherie West Village | 0 | 1 | — | no mentions of vegetarian options |
| S076 | Q20 restaurant careful about peanut allergies | Five Guys | 1 | 0 | mixed reviws about being careful about peanut allegies | not safe for nut allergiers |
| S082 | Q22 safe for celiac disease, no cross contamination | Risotteria Melotti NYC | 2 | 1 | — | one review has celiac and bad experience |
| S087 | Q24 bakery safe for tree nut allergy | Bengal Tiger | 2 | 1 | — | conflicting reviews on allergies |
| S090 | Q25 places safe for someone with a shellfish allergy | KazuNori: The Original Hand Roll Bar | 1 | 0 | mixed reviews about being accommodating for selfish allergy | not safe for shellfish allergies |
| S093 | Q26 dedicated gluten-free fryer | Friedman's Hell's Kitchen | 1 | 2 | evidence suggests gluten-free options being available but doesn’t confirm anything about dedicated fryer | — |
| S096 | Q27 vegan brunch | Hamilton's | 1 | 2 | doesn’t provide any clear evidence for vegan options but gives details about brunch | — |
| S098 | Q27 vegan brunch | Meadowsweet | 1 | 2 | the evidence doesn’t clearly support the availability of vegan options | — |
| S106 | Q30 kosher bakery | Sesame - Boro Park | 1 | 2 | only one evidence mentions kosher other are very generic | — |
| S107 | Q31 nut-free brunch in brooklyn, one of our friends is vegan | Toad Style | 1 | 0 | the evidence doesn't give any details about nut-free dishes being available | no mentions of being nut free |
| S108 | Q32 dinner in manhattan where my vegan girlfriend and i can both eat, i eat meat | Peacefood Uptown | 2 | 1 | — | no mention of meat options |
| S112 | Q33 i have a real dairy allergy not just lactose intolerance, who takes that seriously | Bareburger | 1 | 0 | the evidence is not specific against dairy allergy | no mention of dairy allergies |
| S117 | Q34 celiac friendly italian place in brooklyn with a separate prep area | Saraghina | 1 | 0 | the evidence provides details of italian food but nothing about celiac or separate food prep area | no mentions of being celiac friendly |

## Second rater vs the LLM: Kang, computed the same way as Pujitha

Matched 120 of 120 labeled rows (unmatched: 0).

| Statistic | Pujitha vs LLM | Kang vs LLM |
|---|---|---|
| n matched | 120 | 120 |
| Raw agreement | 0.750 [0.667, 0.825] | 0.833 [0.767, 0.900] |
| Cohen's kappa (unweighted) | 0.620 | 0.737 |
| Cohen's kappa (quadratic) | 0.737 [0.620, 0.834] | 0.853 [0.773, 0.920] |
| Krippendorff's alpha (ordinal) | 0.736 | 0.852 |
| Mean label: human / LLM | 1.100 / 0.967 | 1.108 / 0.967 |
| LLM above / below human | 10 / 20 | 3 / 17 |
| Off by one step / by two | 24 / 6 | 17 / 3 |
| Agreement without the 'no evidence' flagged pairs | 0.791 (n=91) | 0.846 (n=91) |

Kang vs LLM confusion matrix (rows: Kang, columns: LLM):

| Kang \ LLM | 0 | 1 | 2 | total |
|---|---|---|---|---|
| 0 | 42 | 2 | 0 | 44 |
| 1 | 5 | 13 | 1 | 19 |
| 2 | 3 | 9 | 45 | 57 |

Agreement with the LLM by query category (Pujitha / Kang):

| Category | n | Agreement | Kappa (quadratic) |
|---|---|---|---|
| single_diet | 43 / 43 | 0.814 / 0.907 | 0.828 / 0.938 |
| mixed_group | 30 / 30 | 0.633 / 0.733 | 0.589 / 0.661 |
| allergy | 34 / 34 | 0.765 / 0.882 | 0.649 / 0.902 |
| diet_plus_filter | 13 / 13 | 0.769 / 0.692 | 0.578 / 0.596 |

## Limitations

- **Two raters.** Human-human agreement is computed above on the shared sample_ids, but the two raters are not a consensus: kappa and alpha measure how often they read the rubric alike, and both are small-sample estimates.
- **n = 120** matched pairs. The CIs above are wide; per-category cells (4 to 11 queries) and the confusion-matrix cells are small, and the kappa values depend heavily on a few pairs.
- **A sample, not the pool.** The rater labeled a stratified sample of 120 of the 1,074 pairs judged when the sample was drawn (about 11%; drawn by `eval/make_sample.py`), so errors in the unsampled pairs are unmeasured. The label file now holds 1,224 pairs because pairs were added later.
- **The sample predates the new labels.** It was drawn from the earlier judging pool; the pairs added later for the stance re-ranker (top-20 candidates, a single provider: Groq, `gpt-oss-120b`) are not validated here, and provider-specific behaviour is only visible through the 'by source' table above.
- **Same judge model throughout.** All LLM labels use one model family and one prompt, so shared blind spots (for example, treating a mention of the topic as support) would not show up as disagreement between LLM sources.
- **Blindness is not verifiable.** Neither rater was shown the LLM labels per the workflow, but the files cannot prove it. The second rater agrees with the LLM more than with the first rater, so a shared reading of the rubric and contamination cannot be told apart here; confirm with the rater before leaning on that gap.
- **The rater saw the evidence shown to the judge**, not the full review text of each restaurant; both are limited to what retrieval returned, so agreement says nothing about whether the shown reviews were representative.
- Matching rows to restaurants relies on the review ids cited in the evidence; rows without evidence are matched by name and borough only when that is unambiguous.
