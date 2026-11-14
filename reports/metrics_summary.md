# Retrieval metrics (PROVISIONAL)

> **PROVISIONAL:** LLM labels not yet validated against human labels; pairs judged on Groq and OpenRouter/Crusoe, same model gpt-oss-120b

- Runs: `/Users/kang/Documents/GitHub/NYC-Restaurant-Discovery/eval/runs.csv`; labels: `eval/llm_labels.jsonl` (964 labeled pairs); queries: `eval/queries_final.csv`
- Labels by source: openrouter/crusoe 559, groq 324, rule 81 (rule = no evidence shown, labeled 0 without a model call)
- 35 queries, 5 systems, top 10 per system; macro averages over queries
- Relevance: strict = label 2, lenient = label ≥ 1
- NDCG: gain = label (0, 1, 2); DCG@10 = sum_{i=1..10} gain_i / log2(i + 1); NDCG@10 = DCG@10 / IDCG@10; IDCG from the best ordering of the query's judged pool
- Significance: paired randomization (sign-flip, 10,000 resamples) on per-query NDCG@10, Holm-adjusted across the 10 system pairs; percentile bootstrap 95% CIs; seed 6165

Every top-10 result is judged (0 unjudged).

## By system

| System | P@5 strict | P@5 lenient | P@10 strict | P@10 lenient [95% CI] | NDCG@10 [95% CI] | R@10 strict | R@10 lenient | Unjudged |
|---|---|---|---|---|---|---|---|---|
| doc_tfidf | 0.446 | 0.629 | 0.429 | 0.617 [0.497, 0.734] | 0.612 [0.508, 0.706] | 0.474 | 0.420 | 0 |
| chunk_tfidf_log | 0.446 | 0.589 | 0.437 | 0.577 [0.454, 0.700] | 0.603 [0.495, 0.710] | 0.425 | 0.381 | 0 |
| chunk_tfidf_capped | 0.440 | 0.583 | 0.437 | 0.571 [0.451, 0.691] | 0.603 [0.496, 0.708] | 0.438 | 0.387 | 0 |
| semantic_log | 0.423 | 0.657 | 0.394 | 0.611 [0.491, 0.734] | 0.616 [0.524, 0.709] | 0.354 | 0.401 | 0 |
| semantic_capped | 0.446 | 0.686 | 0.426 | 0.666 [0.546, 0.783] | 0.708 [0.631, 0.785] | 0.431 | 0.481 | 0 |

## NDCG@10 by query category

| System | single_diet (n=12) | mixed_group (n=8) | allergy (n=11) | diet_plus_filter (n=4) |
|---|---|---|---|---|
| doc_tfidf | 0.762 | 0.608 | 0.363 | 0.853 |
| chunk_tfidf_log | 0.848 | 0.460 | 0.388 | 0.746 |
| chunk_tfidf_capped | 0.811 | 0.485 | 0.392 | 0.793 |
| semantic_log | 0.747 | 0.665 | 0.471 | 0.525 |
| semantic_capped | 0.807 | 0.709 | 0.565 | 0.807 |

## P@10 lenient by query category

| System | single_diet (n=12) | mixed_group (n=8) | allergy (n=11) | diet_plus_filter (n=4) |
|---|---|---|---|---|
| doc_tfidf | 0.875 | 0.750 | 0.245 | 0.600 |
| chunk_tfidf_log | 0.883 | 0.550 | 0.282 | 0.525 |
| chunk_tfidf_capped | 0.850 | 0.550 | 0.291 | 0.550 |
| semantic_log | 0.792 | 0.812 | 0.300 | 0.525 |
| semantic_capped | 0.842 | 0.887 | 0.336 | 0.600 |

## Paired comparisons (NDCG@10)

Difference = B − A, over queries where both are defined.

| A | B | n | mean A | mean B | B − A [95% CI] | p | p (Holm) | B better / worse / tie |
|---|---|---|---|---|---|---|---|---|
| doc_tfidf | chunk_tfidf_log | 35 | 0.612 | 0.603 | -0.009 [-0.094, 0.073] | 0.843 | 1.000 | 18 / 14 / 3 |
| doc_tfidf | chunk_tfidf_capped | 35 | 0.612 | 0.603 | -0.009 [-0.088, 0.069] | 0.834 | 1.000 | 18 / 15 / 2 |
| doc_tfidf | semantic_log | 35 | 0.612 | 0.616 | 0.005 [-0.119, 0.126] | 0.945 | 1.000 | 20 / 15 / 0 |
| doc_tfidf | semantic_capped | 35 | 0.612 | 0.708 | 0.097 [0.004, 0.191] | 0.053 | 0.368 | 20 / 13 / 2 |
| chunk_tfidf_log | chunk_tfidf_capped | 35 | 0.603 | 0.603 | -0.000 [-0.018, 0.018] | 0.984 | 1.000 | 9 / 12 / 14 |
| chunk_tfidf_log | semantic_log | 35 | 0.603 | 0.616 | 0.013 [-0.096, 0.112] | 0.812 | 1.000 | 22 / 12 / 1 |
| chunk_tfidf_log | semantic_capped | 35 | 0.603 | 0.708 | 0.105 [0.019, 0.185] | 0.018 | 0.165 | 23 / 8 / 4 |
| chunk_tfidf_capped | semantic_log | 35 | 0.603 | 0.616 | 0.013 [-0.096, 0.113] | 0.811 | 1.000 | 20 / 14 / 1 |
| chunk_tfidf_capped | semantic_capped | 35 | 0.603 | 0.708 | 0.105 [0.018, 0.187] | 0.021 | 0.166 | 23 / 8 / 4 |
| semantic_log | semantic_capped | 35 | 0.616 | 0.708 | 0.092 [0.032, 0.166] | 0.002 | 0.024 | 20 / 6 / 9 |

## Density mode: log vs capped

### chunk_tfidf_log → chunk_tfidf_capped

NDCG@10 0.603 → 0.603; difference -0.000 [-0.018, 0.018], p = 0.984 (unadjusted); capped better on 9, worse on 12, tied on 14 queries.

| Query | Category | log | capped | Δ |
|---|---|---|---|---|
| Q11 | single_diet | 0.572 | 0.418 | -0.154 |
| Q24 | allergy | 0.269 | 0.392 | +0.123 |
| Q30 | diet_plus_filter | 0.181 | 0.304 | +0.123 |
| Q05 | single_diet | 1.000 | 0.905 | -0.095 |
| Q06 | single_diet | 0.732 | 0.640 | -0.092 |
| Q17 | mixed_group | 0.682 | 0.771 | +0.090 |
| Q08 | single_diet | 0.897 | 0.808 | -0.089 |
| Q13 | mixed_group | 0.842 | 0.927 | +0.085 |
| Q26 | allergy | 0.436 | 0.358 | -0.078 |
| Q28 | diet_plus_filter | 0.961 | 1.000 | +0.039 |
| Q32 | mixed_group | 0.276 | 0.311 | +0.035 |
| Q14 | mixed_group | 0.567 | 0.596 | +0.029 |
| Q27 | diet_plus_filter | 0.842 | 0.867 | +0.025 |
| Q31 | mixed_group | 0.310 | 0.285 | -0.025 |
| Q21 | allergy | 0.926 | 0.945 | +0.019 |
| Q22 | allergy | 0.890 | 0.874 | -0.016 |
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
| Q19 | allergy | 0.141 | 0.141 | +0.000 |
| Q20 | allergy | 0.109 | 0.109 | +0.000 |
| Q23 | allergy | 0.613 | 0.613 | +0.000 |
| Q25 | allergy | 0.312 | 0.312 | +0.000 |
| Q29 | diet_plus_filter | 1.000 | 1.000 | +0.000 |
| Q33 | allergy | 0.274 | 0.274 | +0.000 |
| Q34 | allergy | 0.000 | 0.000 | +0.000 |
| Q35 | allergy | 0.296 | 0.296 | +0.000 |

### semantic_log → semantic_capped

NDCG@10 0.616 → 0.708; difference 0.092 [0.032, 0.166], p = 0.002 (unadjusted); capped better on 20, worse on 6, tied on 9 queries.

| Query | Category | log | capped | Δ |
|---|---|---|---|---|
| Q29 | diet_plus_filter | 0.000 | 1.000 | +1.000 |
| Q02 | single_diet | 0.438 | 0.876 | +0.438 |
| Q06 | single_diet | 0.576 | 0.953 | +0.377 |
| Q20 | allergy | 0.402 | 0.744 | +0.342 |
| Q23 | allergy | 0.307 | 0.613 | +0.307 |
| Q34 | allergy | 0.246 | 0.503 | +0.257 |
| Q18 | mixed_group | 0.293 | 0.462 | +0.170 |
| Q11 | single_diet | 0.963 | 0.830 | -0.134 |
| Q30 | diet_plus_filter | 0.280 | 0.398 | +0.118 |
| Q09 | single_diet | 0.834 | 0.734 | -0.101 |
| Q19 | allergy | 0.296 | 0.391 | +0.095 |
| Q05 | single_diet | 0.708 | 0.794 | +0.085 |
| Q14 | mixed_group | 0.922 | 1.000 | +0.078 |
| Q28 | diet_plus_filter | 0.869 | 0.945 | +0.076 |
| Q17 | mixed_group | 0.567 | 0.637 | +0.069 |
| Q27 | diet_plus_filter | 0.953 | 0.884 | -0.068 |
| Q16 | mixed_group | 0.598 | 0.666 | +0.068 |
| Q13 | mixed_group | 0.882 | 0.828 | -0.053 |
| Q15 | mixed_group | 0.670 | 0.720 | +0.050 |
| Q32 | mixed_group | 0.864 | 0.830 | -0.035 |
| Q07 | single_diet | 0.967 | 1.000 | +0.033 |
| Q26 | allergy | 0.470 | 0.489 | +0.019 |
| Q04 | single_diet | 0.778 | 0.796 | +0.018 |
| Q24 | allergy | 0.518 | 0.533 | +0.015 |
| Q12 | single_diet | 0.164 | 0.159 | -0.006 |
| Q31 | mixed_group | 0.526 | 0.529 | +0.003 |
| Q01 | single_diet | 1.000 | 1.000 | +0.000 |
| Q03 | single_diet | 1.000 | 1.000 | +0.000 |
| Q08 | single_diet | 1.000 | 1.000 | +0.000 |
| Q10 | single_diet | 0.539 | 0.539 | +0.000 |
| Q21 | allergy | 0.965 | 0.965 | +0.000 |
| Q22 | allergy | 0.820 | 0.820 | +0.000 |
| Q25 | allergy | 0.339 | 0.339 | +0.000 |
| Q33 | allergy | 0.364 | 0.364 | +0.000 |
| Q35 | allergy | 0.453 | 0.453 | +0.000 |

Small query sets are noisy: treat differences whose CI includes 0 as unresolved.
