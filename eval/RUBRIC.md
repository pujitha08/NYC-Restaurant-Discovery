# Relevance rubric

Each row of `pool.csv` is one restaurant for one query. Fill the `label` column with 0, 1 or 2.
Judge the restaurant for the **need in the query**, using only the reviews shown in the row (up to six,
in `evidence_1` to `evidence_6`). Do not look the restaurant up elsewhere: the LLM judge sees exactly
the same reviews, and its labels are validated against yours. Judge the evidence about the need, not
how good the restaurant is overall.

| Label | Meaning |
|---|---|
| 2 | Strongly relevant: reviews clearly show the restaurant meets the need. |
| 1 | Partly relevant: it probably meets the need, or meets only part of it. |
| 0 | Not relevant: no evidence for the need, or evidence against it. |

## General rules
- **No evidence found = 0.** If none of the shown reviews says anything about the need, label 0 and write
  "no evidence found" in the note. This includes rows with `evidence_status` = `none` (no reviews shown) and
  rows whose reviews are all off-topic (for example, only about service or price). A review that names the
  cuisine or a dish without saying anything about the diet or allergy ("best Chinese food" for a halal query,
  "great lobster roll" for a shellfish-allergy query) is not evidence.
- **Several needs** (mixed group, diet plus filter): 2 only if all parts are covered. One part covered = 1.
- **Borough or category filters** in the query (`borough`, "bakery", "pizza"): a restaurant that fails the filter is 0.
- **One passing mention of a dish** (a vegan side, one gluten-free pasta) is 1 at most. A menu section, a dedicated
  option list, or several reviewers saying so is 2.
- **Noisy words** do not count: "kosher salt", "kosher-style", "vegan" used as a joke, "nut" in "Cro-nut" or "donut".
- **Mixed evidence** (some reviews for, some against): label by the most recent and most specific evidence; when
  that is a tie, 1.
- Do not use outside knowledge of the restaurant; label from what the reviews say.

## Allergy and cross-contact queries (peanut, tree nut, shellfish, celiac, fryer, cross contamination)
A mention is not safety. Be strict:
- **Negative or unrelated mention = 0.** Examples: "not celiac safe", "got sick", "has peanuts, cross contamination",
  "fried in peanut oil", "peanut butter noodles", "I have a shellfish allergy" with no statement about the restaurant.
- **Mention of accommodation = 2.** Examples: "staff asked about my allergy and changed gloves", "totally nut free",
  "separate fryer", "dedicated gluten-free kitchen", "celiac safe", "99% of the menu is safe for me".
- **Vague positives = 1.** Examples: "gluten-free options", "they have a nut-free dessert" with no statement about handling
  or cross contact.
- **Both an accommodation and a warning** (for example one reviewer says safe, another says got sick): 1, and
  put the reason in your own notes. If the warning is specific and recent, 0.
- A dish that merely contains the allergen (peanut noodles for "peanut allergy") never counts as evidence.
