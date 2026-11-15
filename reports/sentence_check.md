# Sentence check: labeled review sentences

Results of `python -m src.classifier.stance` on the 200 hand-labeled review sentences. Every number below is read from `data/processed/stance/report.json`.

## Data

- 200 labeled sentences, all kept as gold (0 ties dropped).
- Stance distribution: mention 106, negative 42, not_about_allergy 28, accommodation 24.
- Three labelers (Kang, Laasya, Pujitha). 20 sentences were labeled by all three; the rest by one labeler each.
- Gold stance for shared sentences is the majority label. Allergen gold is the per-allergen majority.

## Labeler agreement (20 shared sentences)

| Measure | Value |
|---|---|
| Krippendorff's alpha (nominal) | 0.382 |
| Unanimous (all three agree) | 0.45 (9 of 20) |
| Mean pairwise raw agreement | 0.725 |

Alpha is low, and only 9 of 20 sentences got the same stance from all three labelers.

## Stance classifier

Logistic regression (balanced class weights) on all-MiniLM-L6-v2 sentence embeddings. 5-fold stratified cross-validation, n = 200, out-of-fold predictions.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| accommodation | 0.389 | 0.292 | 0.333 | 24 |
| mention | 0.711 | 0.509 | 0.593 | 106 |
| negative | 0.588 | 0.714 | 0.645 | 42 |
| not_about_allergy | 0.291 | 0.571 | 0.386 | 28 |

Macro-F1 **0.489**, accuracy **0.535**. The majority class (mention) is 53% of the data, so accuracy 0.535 is no better than always predicting "mention"; macro-F1 is the more informative figure.

## Allergen tagger on review sentences

The tagger (trained on menu items) was run on the same 200 sentences and scored against the gold allergen labels.

Macro-F1 **0.516** on reviews vs **0.684** on held-out menus, a drop of 0.169. Micro-F1 on reviews is 0.462.

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| milk | 0.477 | 0.656 | 0.553 | 32 |
| egg | 0.362 | 1.000 | 0.531 | 17 |
| fish | 0.412 | 0.933 | 0.571 | 15 |
| shellfish | 0.667 | 0.609 | 0.636 | 23 |
| tree_nuts | 0.571 | 0.381 | 0.457 | 21 |
| wheat | 0.165 | 0.447 | 0.241 | 38 |
| peanuts | 1.000 | 0.632 | 0.774 | 19 |
| soy | 0.250 | 0.625 | 0.357 | 16 |
| sesame | 0.361 | 0.929 | 0.520 | 14 |

Recall is high for egg, fish and sesame but precision is low, so the tagger over-tags on review text. Wheat is the weakest label. Peanuts shows precision 1.0 here, but on 19 positive sentences and with recall 0.63; this contrasts with its low menu precision (0.387) and should not be read as an improvement.

## Limitations

- **Small agreement sample.** 20 shared sentences is a small sample, so alpha 0.382 and unanimity 0.45 are imprecise estimates.
- **Different pre-fills.** The sheets were pre-filled differently: the allergen tagger for Kang and Laasya, a keyword list for Pujitha. Pre-fills can anchor labelers, so agreement and the tagger's scores on reviews may be affected, and the tagger evaluation is partly circular for the two labelers who started from its output.
- **Ambiguous guide boundary.** The labeling guide's boundary between `mention` and `not_about_allergy` was ambiguous, which likely contributes to the low agreement and to the weak not_about_allergy precision (0.291).
- **Few accommodation examples.** Accommodation has only 24 examples, so its per-class scores (F1 0.333) are unstable.
- **Small, single-split evaluation.** With 200 sentences, the cross-validation figures carry wide uncertainty and are not confidence intervals.
- **Not used in the live ranking.** Neither the stance classifier nor the allergen tagger feeds retrieval, re-ranking or the app. This report evaluates them only.
