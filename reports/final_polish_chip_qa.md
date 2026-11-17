# Final polish: chip QA (saved results only)

Each chip was clicked in the real app (AppTest) with the retriever and the model replaced by fakes that fail the test if called, so every result comes from `data/demo_cache` and **no model call was made** (`model_calls` = 0 for all 11). Counts come from `app/view_model.py`; the rendered page contains the same number of care, problems and disagree cards (checked in the same run).

| Chip | Care column | Problems column | Reviewers disagree | Not enough evidence | Summary unavailable | Restaurants shown twice | Quote-only halves | Findings on page | Safety claim in a finding |
|---|---|---|---|---|---|---|---|---|---|
| Peanut | 0 | 3 | 1 | 6 | 0 | 0 | 1 | 4 | 0 |
| Tree nut | 0 | 1 | 1 | 8 | 0 | 0 | 1 | 2 | 0 |
| Shellfish | 0 | 3 | 2 | 5 | 0 | 0 | 0 | 7 | 0 |
| Sesame | 1 | 1 | 0 | 8 | 0 | 0 | 0 | 2 | 0 |
| Dairy-free | 0 | 2 | 0 | 8 | 0 | 0 | 0 | 2 | 0 |
| Celiac | 3 | 0 | 3 | 4 | 0 | 0 | 1 | 8 | 0 |
| Gluten-free | 6 | 0 | 0 | 4 | 0 | 0 | 0 | 6 | 0 |
| Vegetarian | 6 | 0 | 0 | 4 | 0 | 0 | 0 | 6 | 0 |
| Halal | 5 | 0 | 1 | 4 | 0 | 0 | 0 | 7 | 0 |
| Kosher | no saved result | – | – | – | – | – | – | – | – |
| Vegan + gluten-free group | 4 | 0 | 2 | 4 | 0 | 0 | 0 | 8 | 0 |

**Kosher** has no saved result (the build hit the Groq daily token limit). Clicking the chip shows "No saved result for this setting" with a "Run it live" button (present: True) and the example chips; it makes no model call (model calls: 0).

## Notes

- "Restaurants shown twice" is counted by Google map id across the care, problems, disagree, not-enough and unavailable lists: 0 for every chip. The Halal result has two entries named "The Halal Guys"; they are two different branches (different ids), not a duplicate.
- "Reviewers disagree" restaurants (each shown once, with a Praised and a Warned half): Peanut: Ladybird; Tree nut: Ladybird; Shellfish: Buddakan, Boilery Seafood and Grill; Celiac: Senza Gluten, Friedman's Hell's Kitchen, Lilli and Loo; Halal: HALAL Xpress Asian Fusion; Vegan + gluten-free group: Champs Diner, by CHLOE. - SoHo.
- Quote-only halves have no usable finding: the model's only claim for that side was the neutral replacement line the wording guard writes (for example Ladybird's praised half in Peanut). Their quote and review count are still shown.
- Safety claims in findings: none on any page. The finding guard (`finding_is_shown_safely`, `app/view_model.py`) checks the whole sentence for safe/safely/safety/guarantee/reassure*/no risk/risk-free for allergy requests. Raw model claims with such words in the saved results: 1 (in the Vegan + gluten-free result, a diet request, where the check does not apply; that sentence was not the one shown).
- Reviewer quotes may contain the word "safe" (Ladybird: "…outlined what was safe to eat"). That is a reviewer's own words inside a quote, not the app's wording.

## Count line per chip

| Chip | Count line |
|---|---|
| Peanut | 10 restaurants checked · 0 mention allergy care · 3 report problems · 1 with mixed reviews · 6 not enough evidence |
| Tree nut | 10 restaurants checked · 0 mention allergy care · 1 report problems · 1 with mixed reviews · 8 not enough evidence |
| Shellfish | 10 restaurants checked · 0 mention allergy care · 3 report problems · 2 with mixed reviews · 5 not enough evidence |
| Sesame | 10 restaurants checked · 1 mention allergy care · 1 report problems · 8 not enough evidence |
| Dairy-free | 10 restaurants checked · 0 mention allergy care · 2 report problems · 8 not enough evidence |
| Celiac | 10 restaurants checked · 3 mention allergy care · 0 report problems · 3 with mixed reviews · 4 not enough evidence |
| Gluten-free | 10 restaurants checked · 6 mention gluten-free options · 0 report problems · 4 not enough evidence |
| Vegetarian | 10 restaurants checked · 6 mention vegetarian options · 0 report problems · 4 not enough evidence |
| Halal | 10 restaurants checked · 5 mention halal options · 0 report problems · 1 with mixed reviews · 4 not enough evidence |
| Vegan + gluten-free group | 10 restaurants checked · 4 mention vegan and gluten-free options · 0 report problems · 2 with mixed reviews · 4 not enough evidence |
