# Audit: LLM judge labels with reason "no evidence found" and no cited review

No API calls were made and no label was modified. Baseline labels are the first 964 records of `eval/llm_labels.jsonl` (the ones behind `reports/metrics_summary.md`).

## Summary

- **Cause (best supported):** the judge saw the evidence. 29 of 29 flagged sampled calls have a prompt size matching the sheet's evidence (not an empty prompt). The wording comes from the rubric, which prescribes "no evidence found" for label 0 when the shown reviews are off-topic; the judge also applied the rubric's filter rule (e.g. 'bakery', 'brunch') more strictly than the human did.
- **Size:** 274 of 883 model calls in the original 964 labels (31%), all labeled 0.
- **Effect:** see section 4; semantic_capped vs doc_tfidf NDCG@10 difference: baseline: +0.097 (p 0.053, Holm 0.368); human labels override LLM on the sample: +0.076 (p 0.145, Holm 1.000); flagged no-evidence pairs dropped: +0.104 (p 0.046, Holm 0.323).

## 1. How many labels

Of 964 labeled pairs, 883 are model calls and 81 are the no-evidence rule (no reviews on the sheet, labeled 0 without a call). **274 model calls (31%) answered "no evidence ..." with no cited review**; all have label 0 (counts: {0: 274}).

| Provider | Flagged | Model calls | Share |
|---|---|---|---|
| groq | 174 | 324 | 54% |
| openrouter/crusoe | 100 | 559 | 18% |

| Query category | Flagged | Model calls | Share |
|---|---|---|---|
| single_diet | 40 | 331 | 12% |
| mixed_group | 58 | 229 | 25% |
| allergy | 138 | 234 | 59% |
| diet_plus_filter | 38 | 89 | 43% |

Flagged results in each system's top 10 (all 35 queries):

| System | Flagged / top-10 results | Share |
|---|---|---|
| chunk_tfidf_log | 84 / 350 | 24% |
| chunk_tfidf_capped | 83 / 350 | 24% |
| doc_tfidf | 74 / 350 | 21% |
| semantic_log | 122 / 350 | 35% |
| semantic_capped | 101 / 350 | 29% |

## 2. Did the judge see the evidence on your sheet?

Local files: `eval/pool.csv` is the **old-format file (Nov 3, columns `best_snippet_1..3`, no `evidence_status`, 30 draft queries)**, so it is not the pool the judge read, and the judging pool itself (gitignored) is not on this machine. What can be checked instead is the token count the API reported for each call.

| Calls | n | actual / estimate with the sheet's evidence (mean, min-max) | actual / estimate with no reviews (mean) |
|---|---|---|---|
| flagged | 29 | 1.135 (1.112-1.165) | 1.302 |
| not flagged | 77 | 1.134 (1.105-1.167) | 1.318 |

For all 29 of the 29 flagged calls the reported prompt size is closer to the prompt rebuilt with the sheet's evidence than to the same prompt with no reviews, and the ratio is the same as for calls that did cite reviews. A prompt without the reviews would be about 12-14% smaller than the one reported. So **the judge was shown evidence of the size on your sheet for every flagged call**; it did not receive empty evidence.

Sheet provenance: all 119/119 of the sheet's query texts equal `eval/queries_final.csv` (the frozen set), the sheet uses the `evidence_status` / `evidence_1..6` layout of `eval/make_pool.py` after commit 4bf191e (2026-11-10), and the sheet's 'none' rows (13) are exactly the rule-labeled calls (13 rule), while no row with evidence is rule-labeled (0). The judge ran 2026-11-12 20:14 UTC to 2026-11-13. `eval/make_sample.py` reads `--pool eval/pool.csv` (default); the exact pool file it read is not recorded in the sample, so its date and checksum could not be determined.

## 3. The 29 flagged pairs on your sheet

29 sampled pairs; your label vs the LLM's: agree 18, one step apart 6, two steps apart 5 (your labels: {2: 5, 1: 6, 0: 18}).

| Sample | Query (category) | Restaurant | Human | LLM | Your note |
|---|---|---|---|---|---|
| S072 | which restaurants are safe if I'm allergic to peanuts (allergy) | Lan Cafe | 2 | 0 | The evidence suggests the restraurant is not peanut free |
| S084 | nut-free bakery, no peanuts or tree nuts (allergy) | Texas Chicken & Burgers | 2 | 0 | — |
| S087 | bakery safe for tree nut allergy (allergy) | Bengal Tiger | 2 | 0 | — |
| S095 | vegan brunch (diet_plus_filter) | Bklyn Wild | 2 | 0 | — |
| S105 | kosher bakery (diet_plus_filter) | Fruits-A-Plenty Inc | 2 | 0 | — |
| S049 | vegan but my friends eat meat (mixed_group) | Donghai | 1 | 0 | doesn’t provide any details about vegan options availability |
| S056 | place where vegans and meat lovers can both eat (mixed_group) | NY Pizza Suprema | 1 | 0 | doesn’t talk anything about optios for vegans |
| S064 | date night, one of us is vegetarian and the other loves steak (mixed_group) | K.Rico Steakhouse | 1 | 0 | doesn’t give any information about vegetarian options |
| S068 | date night, one of us is vegetarian and the other loves steak (mixed_group) | Yummy Taco | 1 | 0 | the steak options are given but theb vegetarian options are not available as per evidence |
| S076 | restaurant careful about peanut allergies (allergy) | Five Guys | 1 | 0 | mixed reviws about being careful about peanut allegies |
| S117 | celiac friendly italian place in brooklyn with a separate prep area (allergy) | Saraghina | 1 | 0 | the evidence provides details of italian food but nothing about celiac or separate food prep area |
| S017 | restaurants that serve halal meat (single_diet) | Keens Steakhouse | 0 | 0 | — |
| S019 | restaurants that serve halal meat (single_diet) | Peter Luger Steak House | 0 | 0 | — |
| S030 | vegetarian friendly menu (single_diet) | Otto Enoteca Pizzeria | 0 | 0 | — |
| S037 | good for lactose intolerant people (single_diet) | Greecologies | 0 | 0 | doesn't provide any evidence about providing lactose intolerant options |
| S070 | which restaurants are safe if I'm allergic to peanuts (allergy) | Chipotle Mexican Grill | 0 | 0 | — |
| S071 | which restaurants are safe if I'm allergic to peanuts (allergy) | Five Guys | 0 | 0 | — |
| S074 | restaurant careful about peanut allergies (allergy) | Emerald Fortune | 0 | 0 | — |
| S075 | restaurant careful about peanut allergies (allergy) | China Long | 0 | 0 | — |
| S077 | restaurant careful about peanut allergies (allergy) | Katz's Delicatessen | 0 | 0 | — |
| S078 | restaurant careful about peanut allergies (allergy) | Zoob Zib Thai Authentic Noodle Bar | 0 | 0 | — |
| S085 | nut-free bakery, no peanuts or tree nuts (allergy) | Nutella Cafe | 0 | 0 | — |
| S088 | bakery safe for tree nut allergy (allergy) | Der Krung Thai Cuisine | 0 | 0 | — |
| S101 | gluten-free pizza (diet_plus_filter) | Little Caesars Pizza | 0 | 0 | — |
| S104 | kosher bakery (diet_plus_filter) | Safir Bakery & Cafe | 0 | 0 | — |
| S109 | dinner in manhattan where my vegan girlfriend and i can both eat, i eat meat (mixed_group) | 99 Favor Taste | 0 | 0 | — |
| S114 | i have a real dairy allergy not just lactose intolerance, who takes that seriously (allergy) | Lady's | 0 | 0 | — |
| S116 | celiac friendly italian place in brooklyn with a separate prep area (allergy) | al di la Trattoria | 0 | 0 | — |
| S118 | my son has a sesame allergy, any places in brooklyn that are careful about it? (allergy) | China Long | 0 | 0 | — |

## 4. Sensitivity of NDCG@10

Rankings: rebuilt from the local indexes; against the published `reports/metrics_per_query.csv` the baseline per-query NDCG@10 differs by at most 0.0000 over 175 (system, query) values (the baseline reproduces the published numbers).

Human override changes the label of 30 of 119 sampled pairs. Dropping removes 274 flagged pairs from the rankings (renumbering the ranks below them); since all flagged labels are 0, this changes NDCG only by moving later results up. Treating them as unjudged instead would leave NDCG unchanged (a missing label counts as 0).

| System | Baseline | Human override | Δ | Flagged dropped | Δ |
|---|---|---|---|---|---|
| doc_tfidf | 0.612 | 0.623 | +0.011 | 0.648 | +0.037 |
| chunk_tfidf_log | 0.603 | 0.601 | -0.002 | 0.648 | +0.045 |
| chunk_tfidf_capped | 0.603 | 0.600 | -0.002 | 0.649 | +0.046 |
| semantic_log | 0.616 | 0.613 | -0.004 | 0.679 | +0.063 |
| semantic_capped | 0.708 | 0.698 | -0.010 | 0.752 | +0.044 |

System order by NDCG@10 (best first):

- baseline (LLM labels): semantic_capped > semantic_log > doc_tfidf > chunk_tfidf_log > chunk_tfidf_capped
- human labels override LLM on the sample: semantic_capped > doc_tfidf > semantic_log > chunk_tfidf_log > chunk_tfidf_capped
- flagged no-evidence pairs dropped: semantic_capped > semantic_log > chunk_tfidf_capped > doc_tfidf > chunk_tfidf_log

`semantic_capped` vs `doc_tfidf` (paired sign-flip on per-query NDCG@10, 10,000 resamples; Holm over the 10 system pairs):

| Variant | semantic_capped | doc_tfidf | Difference [95% CI] | p | p (Holm) | better / worse / tie |
|---|---|---|---|---|---|---|
| baseline (LLM labels) | 0.708 | 0.612 | +0.097 [0.004, 0.191] | 0.053 | 0.368 | 20 / 13 / 2 |
| human labels override LLM on the sample | 0.698 | 0.623 | +0.076 [-0.022, 0.176] | 0.145 | 1.000 | 18 / 15 / 2 |
| flagged no-evidence pairs dropped | 0.752 | 0.648 | +0.104 [0.012, 0.204] | 0.046 | 0.323 | 19 / 14 / 2 |

Unjudged top-10 results per variant: baseline (LLM labels): 0; human labels override LLM on the sample: 0; flagged no-evidence pairs dropped: 0.

## 5. What could not be determined

- **The pool file the judge read** (and the one `eval/make_sample.py` read) is gitignored and not on this machine; the local `eval/pool.csv` is a different, older file. Whether the two were literally the same file, and their dates and checksums, cannot be established here. The token fingerprint shows only that the evidence the judge saw had the size of the sheet's evidence for each sampled call; it cannot show that the texts were identical.
- **The 245 flagged calls outside your sample** were not compared with any sheet; the conclusion for them rests on the same prompt-size behaviour of the sampled ones, and on their using the same pipeline and prompt.
- **Why the judge chose 0 for the five two-step disagreements** cannot be read from the labels (the reasons are the stock phrase). In the sample they involve a second requirement in the query (bakery, brunch) or an allergy query where the reviews do not describe the restaurant's handling; S072 carries your note that the evidence suggests the restaurant is not peanut free, which supports the judge's 0 and may be a labeling slip.
- **Rankings are rebuilt locally** (index build 2026-11-14) rather than read from the original runs.csv; they reproduce the published per-query NDCG@10.
