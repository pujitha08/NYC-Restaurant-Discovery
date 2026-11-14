# Allergen-mention tagger

`src/classifier/allergen_tagger.py` tags which FDA major allergens a short text mentions (multi-label). It tags *mentions*,
not safety or accommodation, and it is **not used in the live ranking** (see Limitations).

## Held-out menu results

Held-out test split (55,001 menu items), decision threshold 0.5.

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| milk | 0.802 | 0.786 | 0.794 | 22,664 |
| egg | 0.655 | 0.779 | 0.712 | 13,469 |
| fish | 0.667 | 0.821 | 0.736 | 8,631 |
| shellfish | 0.662 | 0.849 | 0.744 | 4,600 |
| tree_nuts | 0.526 | 0.788 | 0.631 | 4,569 |
| wheat | 0.827 | 0.790 | 0.808 | 26,046 |
| peanuts | 0.387 | 0.766 | 0.514 | 1,490 |
| soy | 0.579 | 0.804 | 0.673 | 7,921 |
| sesame | 0.431 | 0.754 | 0.548 | 3,728 |
| **macro F1** | | | 0.684 | |
| **micro F1** | | | 0.738 | |

Source: `data/processed/allergen_tagger/meta.json` (`test_report`), written by
`HF_DATASETS_OFFLINE=1 python -m src.classifier.allergen_tagger --train` (trained 2026-11-14T17:31:58Z).

## Data preparation

- Dataset: `adisyonist/14-allergens-v2` (menu items, 294,500 rows). Texts are deduplicated case-insensitively
  (label sets of duplicates are unioned; 19,495 duplicates removed, 0 with conflicting labels),
  leaving **275,005 unique texts**.
- **Split: 220,004 train / 55,001 test**, a seeded random split of the unique *lowercased* texts
  (test fraction 0.2), so no lowercased text appears in both parts.
- Labels are mapped onto the **FDA 9** (milk, egg, fish, shellfish, tree_nuts, wheat, peanuts, soy, sesame): eggs → egg,
  crustaceans → shellfish, and **gluten → wheat, which is approximate** (gluten also covers barley, rye and oats). The EU-only
  labels (celery, lupin, molluscs, mustard, sulphites) are dropped. Rows left with no FDA label (54,626) stay as negatives.

## Model

Character n-gram TF-IDF (`char_wb`, 2–5-grams, `min_df` 3, sublinear tf, at most 300,000 features) and a one-vs-rest
logistic regression (liblinear) with balanced class weights. Seed **6165** fixes the train/test split; the model fit itself
sets no random state.

## Limitations

- **Trained on menu items, not reviews.** Review sentences are longer and written differently, so accuracy on reviews is expected to be
  lower; `src/classifier/stance.py` measures this once review sentences are hand-labeled.
- **Peanuts and sesame have low precision** (peanuts 0.387, sesame 0.431, against recalls of
  0.766 and 0.754): the balanced weights buy recall at the cost of many false alarms, and the peanut class is small
  (1,490 test items). Tree nuts is also low (0.526).
- **Not used in the live ranking.** Neither retrieval, the stance re-ranker nor the app imports it. It only pre-fills the allergen column of
  the sentence-labeling sheets and is evaluated on review sentences by `stance.py`.
