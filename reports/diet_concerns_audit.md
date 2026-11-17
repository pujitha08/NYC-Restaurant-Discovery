# Concerns audit: non-allergy chips (raw cached claims vs what the display rules did)

Counts come from the raw cached summaries (`data/demo_cache/*.json`, read with `json.load`), before any display gate. Labels are the ones stored in the cache; a claim that the name filter turned into the neutral "only mentions the restaurant by name" line at build time (`app/pipeline.py:455` `drop_name_only_evidence`, applied in `to_summary` `:469`) is stored as `uncertainty`, its original label is not kept. What each display rule does for a diet, group or general search (not an allergy request): the care gate `view_model.py:40` drops a SUPPORT claim whose cited reviews are all rated 1-2 stars; the allergy cue gate `view_model.py:46` and the allergen gate `view_model.py:83` return True without checking when `outcome.allergy_sensitive` is false, so they never drop a contradiction claim here; `effective_claims` (`view_model.py:98`) applies them; `group_of` (`view_model.py:124`) turns the counting labels into care / problems / mixed / insufficient and returns insufficient first if the summary status is insufficient (`view_model.py:135`); `columns` (`view_model.py:157`) puts each restaurant in exactly one list.

## Gluten-free — “gluten-free restaurants” (allergy_sensitive in the cache: False)

| Rank | Restaurant | support | contradiction | uncertainty | where it ended up |
|---|---|---|---|---|---|
| 1 | Senza Gluten | 3 | 0 | 0 | care column |
| 2 | Friedman's Hell's Kitchen | 3 | 0 | 0 | care column |
| 3 | Little Beet Table | 1 | 0 | 0 | care column |
| 4 | Friedman's Herald Square | 3 | 0 | 0 | care column |
| 5 | Ceci | 3 | 0 | 0 | care column |
| 6 | Bistango | 3 | 0 | 0 | care column |
| 7 | Wild | – | – | – | not enough evidence (no summary: not summarized) |
| 8 | Pappardella | – | – | – | not enough evidence (no summary: not summarized) |
| 9 | Wild Park Slope | – | – | – | not enough evidence (no summary: not summarized) |
| 10 | Noglu | – | – | – | not enough evidence (no summary: not summarized) |

Raw claim totals: support 16, contradiction 0, uncertainty 0. Restaurants with at least one contradiction claim: **0**.
Claims the build-time filters had already replaced (original label not stored): name-only line 0 []; safety-wording neutral line 0.

## Vegetarian — “vegetarian restaurants” (allergy_sensitive in the cache: False)

| Rank | Restaurant | support | contradiction | uncertainty | where it ended up |
|---|---|---|---|---|---|
| 1 | Red Bamboo | 3 | 0 | 0 | care column |
| 2 | Dirt Candy | 3 | 0 | 0 | care column |
| 3 | Jungle Cafe | 3 | 0 | 0 | care column |
| 4 | Nix | 3 | 0 | 0 | care column |
| 5 | by CHLOE. - West Village | 3 | 0 | 0 | care column |
| 6 | by CHLOE. - Flatiron | 4 | 0 | 0 | care column |
| 7 | Village Natural | – | – | – | not enough evidence (no summary: not summarized) |
| 8 | The Butcher's Daughter | – | – | – | not enough evidence (no summary: not summarized) |
| 9 | Blossom on University | – | – | – | not enough evidence (no summary: not summarized) |
| 10 | Hangawi | – | – | – | not enough evidence (no summary: not summarized) |

Raw claim totals: support 19, contradiction 0, uncertainty 0. Restaurants with at least one contradiction claim: **0**.
Claims the build-time filters had already replaced (original label not stored): name-only line 0 []; safety-wording neutral line 0.

## Halal — “halal restaurants” (allergy_sensitive in the cache: False)

| Rank | Restaurant | support | contradiction | uncertainty | where it ended up |
|---|---|---|---|---|---|
| 1 | Tacis Beyti | 1 | 0 | 0 | care column |
| 2 | The Halal Guys | 3 | 0 | 0 | care column |
| 3 | The Halal Guys | 1 | 0 | 2 | care column |
| 4 | HALAL Xpress Asian Fusion | 2 | 1 | 0 | Reviewers disagree |
| 5 | Momo's Mediterranean Grill | 3 | 0 | 0 | care column |
| 6 | Affy's Premium Grill | 3 | 0 | 0 | care column |
| 7 | The Halal Guys | – | – | – | not enough evidence (no summary: not summarized) |
| 8 | Milk And Honey Cafe | – | – | – | not enough evidence (no summary: not summarized) |
| 9 | Alnoor Halal Deli | – | – | – | not enough evidence (no summary: not summarized) |
| 10 | Manhattan Halal Restaurant | – | – | – | not enough evidence (no summary: not summarized) |

Raw claim totals: support 13, contradiction 1, uncertainty 2. Restaurants with at least one contradiction claim: **1**.
Claims the build-time filters had already replaced (original label not stored): name-only line 2 [('The Halal Guys', 'uncertainty'), ('The Halal Guys', 'uncertainty')]; safety-wording neutral line 0.

### HALAL Xpress Asian Fusion: moved to Reviewers disagree (it also has a counting support claim)
- raw: 1 contradiction claim(s); counting after the gates: 1; dropped: 0
- contradiction claim: “A reviewer warned that the restaurant is not halal.” (cites reviews [1634761])
- finding shown on the card: yes

## Kosher — “kosher restaurants” (allergy_sensitive in the cache: False)

| Rank | Restaurant | support | contradiction | uncertainty | where it ended up |
|---|---|---|---|---|---|
| 1 | Kasbah Grill | 3 | 0 | 0 | care column |
| 2 | The Prime Grill | 3 | 0 | 0 | care column |
| 3 | My Most Favorite Food | 3 | 0 | 0 | care column |
| 4 | Jerusalem Cafe | 3 | 0 | 0 | care column |
| 5 | Reserve Cut | 1 | 0 | 1 | care column |
| 6 | Glatt A La Carte | 3 | 0 | 0 | care column |
| 7 | Mr. Broadway | – | – | – | not enough evidence (no summary: not summarized) |
| 8 | Mike's Bistro | – | – | – | not enough evidence (no summary: not summarized) |
| 9 | Le Marais | – | – | – | not enough evidence (no summary: not summarized) |
| 10 | Tiberias | – | – | – | not enough evidence (no summary: not summarized) |

Raw claim totals: support 16, contradiction 0, uncertainty 1. Restaurants with at least one contradiction claim: **0**.
Claims the build-time filters had already replaced (original label not stored): name-only line 0 []; safety-wording neutral line 0.

## Vegan + gluten-free group — “vegan and gluten-free options for a group” (allergy_sensitive in the cache: False)

| Rank | Restaurant | support | contradiction | uncertainty | where it ended up |
|---|---|---|---|---|---|
| 1 | by CHLOE. - West Village | 3 | 0 | 0 | care column |
| 2 | Little Beet Table | 3 | 0 | 0 | care column |
| 3 | Friedman's Herald Square | 3 | 0 | 0 | care column |
| 4 | Champs Diner | 1 | 1 | 1 | Reviewers disagree |
| 5 | by CHLOE. - SoHo | 2 | 1 | 0 | Reviewers disagree |
| 6 | Sun In Bloom | 3 | 0 | 0 | care column |
| 7 | Perelandra Natural Food Center | – | – | – | not enough evidence (no summary: not summarized) |
| 8 | Peacefood Uptown | – | – | – | not enough evidence (no summary: not summarized) |
| 9 | S'MAC East Village | – | – | – | not enough evidence (no summary: not summarized) |
| 10 | Clementine Bakery | – | – | – | not enough evidence (no summary: not summarized) |

Raw claim totals: support 15, contradiction 2, uncertainty 1. Restaurants with at least one contradiction claim: **2**.
Claims the build-time filters had already replaced (original label not stored): name-only line 0 []; safety-wording neutral line 0.

### Champs Diner: moved to Reviewers disagree (it also has a counting support claim)
- raw: 1 contradiction claim(s); counting after the gates: 1; dropped: 0
- contradiction claim: “Another reviewer warned that the gluten‑free options were not suitable for a person with celiac disease.” (cites reviews [431374])
- finding shown on the card: yes

### by CHLOE. - SoHo: moved to Reviewers disagree (it also has a counting support claim)
- raw: 1 contradiction claim(s); counting after the gates: 1; dropped: 0
- contradiction claim: “One reviewer warned that despite the restaurant being vegan and gluten free, they experienced difficulty with food allergies and were disappointed with the accommodation.” (cites reviews [1056877])
- finding shown on the card: yes

## Summary

| Chip | Restaurants with a contradiction claim | kept in problems column | moved to Reviewers disagree | contradiction claims dropped |
|---|---|---|---|---|
| Gluten-free | 0 | 0 | 0 | 0 |
| Vegetarian | 0 | 0 | 0 | 0 |
| Halal | 1 | 0 | 1 | 0 |
| Kosher | 0 | 0 | 0 | 0 |
| Vegan + gluten-free group | 2 | 0 | 2 | 0 |
