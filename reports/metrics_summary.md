# Retrieval metrics (PROVISIONAL)

> **PROVISIONAL:** LLM judge labels (gpt-oss-120b, OpenRouter/Crusoe) not yet validated against human labels; only the 18 fully judged queries (Q01-Q18: single diet and mixed group) are scored, because OpenRouter credits ran out mid-run

- Runs: `/Users/kang/Documents/GitHub/NYC-Restaurant-Discovery/eval/runs.csv`; labels: `eval/llm_labels.jsonl` (640 labeled pairs); queries: `eval/queries_final.csv`
- 18 queries, 5 systems, top 10 per system; macro averages over queries
- Relevance: strict = label 2, lenient = label ≥ 1
- NDCG: gain = label (0, 1, 2); DCG@10 = sum_{i=1..10} gain_i / log2(i + 1); NDCG@10 = DCG@10 / IDCG@10; IDCG from the best ordering of the query's judged pool
- Significance: paired randomization (sign-flip, 10,000 resamples) on per-query NDCG@10, Holm-adjusted across the 10 system pairs; percentile bootstrap 95% CIs; seed 6165

Every top-10 result is judged (0 unjudged).

> **Partial: 17 queries left out** because some top-10 results are not judged yet (--complete-only): Q19, Q20, Q21, Q22, Q23, Q24, Q25, Q26, Q27, Q28, Q29, Q30, Q31, Q32, Q33, Q34, Q35. Category tables cover only the remaining queries.

## By system

| System | P@5 strict | P@5 lenient | P@10 strict | P@10 lenient [95% CI] | NDCG@10 [95% CI] | R@10 strict | R@10 lenient | Unjudged |
|---|---|---|---|---|---|---|---|---|
| doc_tfidf | 0.600 | 0.844 | 0.606 | 0.867 [0.767, 0.944] | 0.726 [0.623, 0.817] | 0.388 | 0.378 | 0 |
| chunk_tfidf_log | 0.622 | 0.789 | 0.628 | 0.800 [0.683, 0.906] | 0.737 [0.617, 0.850] | 0.382 | 0.341 | 0 |
| chunk_tfidf_capped | 0.600 | 0.767 | 0.606 | 0.778 [0.661, 0.883] | 0.723 [0.600, 0.837] | 0.370 | 0.330 | 0 |
| semantic_log | 0.556 | 0.811 | 0.550 | 0.806 [0.689, 0.911] | 0.717 [0.596, 0.827] | 0.367 | 0.351 | 0 |
| semantic_capped | 0.589 | 0.833 | 0.600 | 0.872 [0.767, 0.961] | 0.777 [0.669, 0.872] | 0.383 | 0.379 | 0 |

## NDCG@10 by query category

| System | single_diet (n=12) | mixed_group (n=6) | allergy (n=0) | diet_plus_filter (n=0) |
|---|---|---|---|---|
| doc_tfidf | 0.762 | 0.653 | n/a | n/a |
| chunk_tfidf_log | 0.848 | 0.515 | n/a | n/a |
| chunk_tfidf_capped | 0.811 | 0.548 | n/a | n/a |
| semantic_log | 0.747 | 0.655 | n/a | n/a |
| semantic_capped | 0.807 | 0.719 | n/a | n/a |

## P@10 lenient by query category

| System | single_diet (n=12) | mixed_group (n=6) | allergy (n=0) | diet_plus_filter (n=0) |
|---|---|---|---|---|
| doc_tfidf | 0.875 | 0.850 | n/a | n/a |
| chunk_tfidf_log | 0.883 | 0.633 | n/a | n/a |
| chunk_tfidf_capped | 0.850 | 0.633 | n/a | n/a |
| semantic_log | 0.792 | 0.833 | n/a | n/a |
| semantic_capped | 0.842 | 0.933 | n/a | n/a |

## Paired comparisons (NDCG@10)

Difference = B − A, over queries where both are defined.

| A | B | n | mean A | mean B | B − A [95% CI] | p | p (Holm) | B better / worse / tie |
|---|---|---|---|---|---|---|---|---|
| doc_tfidf | chunk_tfidf_log | 18 | 0.726 | 0.737 | 0.012 [-0.104, 0.124] | 0.844 | 1.000 | 10 / 6 / 2 |
| doc_tfidf | chunk_tfidf_capped | 18 | 0.726 | 0.723 | -0.002 [-0.119, 0.109] | 0.972 | 1.000 | 10 / 7 / 1 |
| doc_tfidf | semantic_log | 18 | 0.726 | 0.717 | -0.009 [-0.171, 0.159] | 0.919 | 1.000 | 8 / 10 / 0 |
| doc_tfidf | semantic_capped | 18 | 0.726 | 0.777 | 0.052 [-0.076, 0.184] | 0.471 | 1.000 | 8 / 9 / 1 |
| chunk_tfidf_log | chunk_tfidf_capped | 18 | 0.737 | 0.723 | -0.014 [-0.041, 0.012] | 0.360 | 1.000 | 3 / 9 / 6 |
| chunk_tfidf_log | semantic_log | 18 | 0.737 | 0.717 | -0.021 [-0.166, 0.106] | 0.782 | 1.000 | 9 / 8 / 1 |
| chunk_tfidf_log | semantic_capped | 18 | 0.737 | 0.777 | 0.040 [-0.096, 0.154] | 0.579 | 1.000 | 10 / 6 / 2 |
| chunk_tfidf_capped | semantic_log | 18 | 0.723 | 0.717 | -0.007 [-0.157, 0.128] | 0.928 | 1.000 | 8 / 9 / 1 |
| chunk_tfidf_capped | semantic_capped | 18 | 0.723 | 0.777 | 0.054 [-0.088, 0.175] | 0.463 | 1.000 | 10 / 6 / 2 |
| semantic_log | semantic_capped | 18 | 0.717 | 0.777 | 0.061 [0.003, 0.131] | 0.093 | 0.927 | 10 / 4 / 4 |

## Density mode: log vs capped

### chunk_tfidf_log → chunk_tfidf_capped

NDCG@10 0.737 → 0.723; difference -0.014 [-0.041, 0.012], p = 0.360 (unadjusted); capped better on 3, worse on 9, tied on 6 queries.

| Query | Category | log | capped | Δ |
|---|---|---|---|---|
| Q11 | single_diet | 0.572 | 0.418 | -0.154 |
| Q05 | single_diet | 1.000 | 0.905 | -0.095 |
| Q06 | single_diet | 0.732 | 0.640 | -0.092 |
| Q17 | mixed_group | 0.682 | 0.771 | +0.090 |
| Q08 | single_diet | 0.897 | 0.808 | -0.089 |
| Q13 | mixed_group | 0.842 | 0.927 | +0.085 |
| Q14 | mixed_group | 0.567 | 0.596 | +0.029 |
| Q09 | single_diet | 0.812 | 0.800 | -0.012 |
| Q15 | mixed_group | 0.237 | 0.229 | -0.008 |
| Q16 | mixed_group | 0.614 | 0.611 | -0.003 |
| Q03 | single_diet | 0.963 | 0.961 | -0.003 |
| Q04 | single_diet | 0.784 | 0.782 | -0.001 |
| Q01 | single_diet | 1.000 | 1.000 | +0.000 |
| Q02 | single_diet | 1.000 | 1.000 | +0.000 |
| Q07 | single_diet | 1.000 | 1.000 | +0.000 |
| Q10 | single_diet | 0.470 | 0.470 | +0.000 |
| Q12 | single_diet | 0.953 | 0.953 | +0.000 |
| Q18 | mixed_group | 0.152 | 0.152 | +0.000 |

### semantic_log → semantic_capped

NDCG@10 0.717 → 0.777; difference 0.061 [0.003, 0.131], p = 0.093 (unadjusted); capped better on 10, worse on 4, tied on 4 queries.

| Query | Category | log | capped | Δ |
|---|---|---|---|---|
| Q02 | single_diet | 0.438 | 0.876 | +0.438 |
| Q06 | single_diet | 0.576 | 0.953 | +0.377 |
| Q18 | mixed_group | 0.293 | 0.462 | +0.170 |
| Q11 | single_diet | 0.963 | 0.830 | -0.134 |
| Q09 | single_diet | 0.834 | 0.734 | -0.101 |
| Q05 | single_diet | 0.708 | 0.794 | +0.085 |
| Q14 | mixed_group | 0.922 | 1.000 | +0.078 |
| Q17 | mixed_group | 0.567 | 0.637 | +0.069 |
| Q16 | mixed_group | 0.598 | 0.666 | +0.068 |
| Q13 | mixed_group | 0.882 | 0.828 | -0.053 |
| Q15 | mixed_group | 0.670 | 0.720 | +0.050 |
| Q07 | single_diet | 0.967 | 1.000 | +0.033 |
| Q04 | single_diet | 0.778 | 0.796 | +0.018 |
| Q12 | single_diet | 0.164 | 0.159 | -0.006 |
| Q01 | single_diet | 1.000 | 1.000 | +0.000 |
| Q03 | single_diet | 1.000 | 1.000 | +0.000 |
| Q08 | single_diet | 1.000 | 1.000 | +0.000 |
| Q10 | single_diet | 0.539 | 0.539 | +0.000 |

Small query sets are noisy: treat differences whose CI includes 0 as unresolved.
