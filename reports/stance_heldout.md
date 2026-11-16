# Held-out evaluation of the frozen stance re-ranker (single pre-registered run)

> **This is the single pre-registered held-out run.** It was run once, with the weights frozen on the dev queries before any test result existed. Nothing was tuned, re-run or adjusted afterwards. Labels are LLM-judge labels (not validated per pair against humans), so read the result as provisional.

- Frozen weights: w_pos 0.9, w_neg 0.0, w_strong_neg 0.2 (`reports/stance_rerank_frozen.json`, commit `2b1e13f`; the file is byte-identical to that commit). Rule: Best mean dev NDCG@10 among grid-2 settings with w_pos <= 0.9 (ties to the smaller total weight, as in eval/tune_stance_rerank.grid_search). The cap was chosen by hand because the unrestricted grid-2 optimum (w_pos 1.8, w_neg 0.4, w_strong_neg 0.0) sat on the widened grid's upper edge and the dev NDCG@10 curve was nearly flat for w_pos >= 0.9 (0.691 at 0.9 vs 0.695 at 1.8).
- Candidate depth: top 20 of `semantic_capped` (pre-registered), so both systems rank the same 20 restaurants per query and differ only in the order.
- Test queries (15, seed 6165): Q01, Q04, Q05, Q08, Q09, Q11, Q14, Q21, Q23, Q24, Q26, Q27, Q29, Q32, Q35. Dev queries (not used here): 20.
- Checks before scoring: frozen file unchanged since the commit: yes; **unjudged top-20 candidates: 0 of 300**. Unjudged results in the top 10 would count as 0; there are none.
- Metrics: NDCG@10, lenient P@5 and P@10 (as in `eval/metrics.py`; relevance 0/1/2, lenient counts 1 and 2 as relevant); 95% percentile bootstrap over queries (10,000 resamples, seed 6165); paired difference with a two-sided sign-flip test.
- Stance labels: `llm:openai/gpt-oss-20b` (read from the on-disk cache; no API call in this run). Relevance labels of the 300 candidate pairs by source: groq 197, openrouter/crusoe 98, rule (no evidence shown) 5.

## Result on the 15 test queries

| System | NDCG@10 [95% CI] | P@5 lenient [95% CI] | P@10 lenient [95% CI] | NDCG@10 vs semantic_capped [95% CI] | p (sign-flip) | better / worse / tie | Unjudged |
|---|---|---|---|---|---|---|---|
| semantic_capped | 0.780 [0.661, 0.882] | 0.720 [0.547, 0.880] | 0.687 [0.493, 0.867] | — | — | — | 0 |
| semantic_capped_stance | 0.787 [0.666, 0.890] | 0.733 [0.573, 0.880] | 0.700 [0.507, 0.873] | +0.007 [-0.033, 0.050] | 0.743 | 5 / 4 / 6 | 0 |

**Paired NDCG@10 difference (stance minus semantic_capped): +0.007 [-0.033, 0.050], sign-flip p = 0.743; 5 queries better, 4 worse, 6 tied.** The difference is not significant at the 0.05 level: this run does not show that the stance re-ranker beats `semantic_capped`, and the interval includes zero.

## By category (NDCG@10)

| Category | n | semantic_capped | stance (frozen) | Δ NDCG@10 | p (sign-flip) |
|---|---|---|---|---|---|
| single_diet | 6 | 0.859 | 0.869 | +0.010 | 0.633 |
| mixed_group | 2 | 0.915 | 0.913 | -0.002 | 1.000 |
| allergy | 5 | 0.566 | 0.566 | -0.001 | 1.000 |
| diet_plus_filter | 2 | 0.942 | 0.968 | +0.025 | 1.000 |

Category cells have 2 to 6 queries each: descriptions of this test set, not estimates.

## Per query

| Query | Category | semantic_capped NDCG@10 | stance NDCG@10 | Δ | Result | P@5 base → stance |
|---|---|---|---|---|---|---|
| Q01 "vegan food" | single_diet | 1.000 | 1.000 | +0.000 | tie | 1.0 → 1.0 |
| Q04 "somewhere I can eat with no gluten at all" | single_diet | 0.796 | 0.835 | +0.039 | win | 1.0 → 1.0 |
| Q05 "halal restaurant" | single_diet | 0.794 | 0.717 | -0.077 | loss | 0.8 → 0.8 |
| Q08 "vegetarian friendly menu" | single_diet | 1.000 | 1.000 | +0.000 | tie | 1.0 → 1.0 |
| Q09 "dairy-free options" | single_diet | 0.734 | 0.817 | +0.083 | win | 0.8 → 0.8 |
| Q11 "pescatarian seafood restaurant" | single_diet | 0.830 | 0.844 | +0.014 | win | 0.8 → 0.8 |
| Q14 "group dinner with a vegetarian, a halal eater and a gluten-f" | mixed_group | 1.000 | 1.000 | +0.000 | tie | 1.0 → 1.0 |
| Q21 "celiac safe" | allergy | 0.965 | 0.965 | +0.000 | tie | 1.0 → 1.0 |
| Q23 "nut-free bakery, no peanuts or tree nuts" | allergy | 0.613 | 0.613 | +0.000 | tie | 0.2 → 0.2 |
| Q24 "bakery safe for tree nut allergy" | allergy | 0.533 | 0.482 | -0.051 | loss | 0.4 → 0.4 |
| Q26 "dedicated gluten-free fryer" | allergy | 0.449 | 0.273 | -0.176 | loss | 0.4 → 0.4 |
| Q27 "vegan brunch" | diet_plus_filter | 0.884 | 0.935 | +0.051 | win | 1.0 → 1.0 |
| Q29 "halal chinese food" | diet_plus_filter | 1.000 | 1.000 | +0.000 | tie | 0.2 → 0.2 |
| Q32 "dinner in manhattan where my vegan girlfriend and i can both" | mixed_group | 0.830 | 0.827 | -0.003 | loss | 1.0 → 1.0 |
| Q35 "my son has a sesame allergy, any places in brooklyn that are" | allergy | 0.271 | 0.494 | +0.224 | win | 0.2 → 0.4 |

## Limitations

- **Small test set.** 15 queries: intervals are wide, and a non-significant result is not evidence of no effect.
- **Judge labels, one family of model.** Relevance is by an LLM judge; stance labels come from the same model family, so errors can be correlated, which may favor the re-ranker. The judge agrees with a human rater about 0.75 of the time on a 120-pair sample (`reports/judge_validation.md`).
- **Mixed label sources.** The 111 test pairs added for this run were judged by one provider (Groq `gpt-oss-120b`); the other 189 came from the earlier judging. Many of the new labels are 0 with the stock reason "no evidence found" (see `reports/no_evidence_audit.md` for the dev equivalent), which are strict readings of the need rather than missing evidence.
- **w_neg = 0.** The frozen setting ignores the share of negative sentences except for the strong-negative term, which applies to allergy queries only.
- **Weights chosen by hand within a rule.** The `w_pos <= 0.9` cap was a judgment on the dev split; the dev numbers are optimistic.
- **No default was changed.** The app default remains `semantic_capped`; `STANCE_RERANK` in `src/config.py` still holds the starting weights.
