# Audit: LLM judge labels with reason "no evidence found" and no cited review

No API calls were made and no label was modified. Baseline labels are the first 964 records of `eval/llm_labels.jsonl` (the ones behind `reports/metrics_summary.md`).

## Summary

- **Cause (best supported):** the judge saw the evidence. 29 of 29 flagged sampled calls have a prompt size matching the sheet's evidence (not an empty prompt). The wording comes from the rubric, which prescribes "no evidence found" for label 0 when the shown reviews are off-topic; the five large disagreements plausibly reflect the judge applying the rubric's category-filter rule (e.g. 'bakery', 'brunch') or its strict allergy rule more strictly than the human did (a hypothesis: the reasons do not say).
- **Size:** 274 of 883 model calls in the original 964 labels (31%), all labeled 0.
- **Effect:** see section 4; semantic_capped vs doc_tfidf NDCG@10 difference: baseline: +0.097 (p 0.053, Holm 0.368); human labels override LLM on the sample: +0.076 (p 0.145, Holm 1.000); flagged no-evidence pairs dropped: +0.104 (p 0.046, Holm 0.323).

## 1. How many labels

Of 964 labeled pairs, 883 are model calls and 81 are the no-evidence rule (no reviews on the sheet, labeled 0 without a call). **274 model calls (31%) answered "no evidence ..." with no cited review**; all have label 0 (counts: {0: 274}).

| Provider | Flagged | Model calls | Share |
|---|---|---|---|
| groq | 174 | 324 | 54% |
| openrouter/crusoe | 100 | 559 | 18% |

Provider and category are confounded (each provider judged a different mix of queries), so the provider gap above is also shown within each category (flagged / model calls):

| Category | groq | openrouter/crusoe |
|---|---|---|
| single_diet | — | 40 / 331 |
| mixed_group | 19 / 44 | 39 / 185 |
| allergy | 121 / 202 | 17 / 32 |
| diet_plus_filter | 34 / 78 | 4 / 11 |

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

Sheet provenance: 119 of 119 of the sheet's query texts equal `eval/queries_final.csv` (the frozen set), the sheet uses the `evidence_status` / `evidence_1..6` layout of `eval/make_pool.py` after commit 4bf191e (2026-11-10), and the sheet's 'none' rows (13) are exactly the rule-labeled calls (13 rule), while no row with evidence is rule-labeled (0). The judge ran 2026-11-12 20:14 UTC to 2026-11-13. `eval/make_sample.py` reads `--pool eval/pool.csv` (default); the exact pool file it read is not recorded in the sample, so its date and checksum could not be determined.

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

Rankings: from eval/runs.csv; against the published `reports/metrics_per_query.csv` the baseline per-query NDCG@10 differs by at most 0.0000 over 175 (system, query) values (the baseline reproduces the published numbers).

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

## Extended pool, top-20 dev labels

Pairs: the 149 pairs in `eval/pool_extended.csv` (top-20 of semantic_capped on the 20 dev queries, no label before this work), judged in two batches: 110 earlier (before the 120b daily quota ran out) and 39 later (the 39 judged most recently). All by Groq `openai/gpt-oss-120b`. "Flagged" means a model call whose reason starts "no evidence" and cites no review. No API calls were made and no label was modified.

**Summary.** 22 of the 39 newest labels (22/39) are flagged, against 51 of 149 for the whole extended pool and 31% (274 of 883) in the original 964 labels. Every flagged row had excerpts (none was an empty prompt) and the judge's prompt size tracks the excerpt size like the unflagged calls, so the judge saw the evidence; the words "no evidence found" are the rubric's stock phrase for label 0, not a report of empty evidence. The higher rate is mostly a matter of which queries were judged last, not of weaker candidates: the extended pool is entirely rank 11-20 (the top 10 were already labeled), where the flagged share over all dev queries is 36% against 28% at rank 1-10 (see 3), while the newest batch is concentrated in Q30, Q31, Q25, which hold 16 of the flagged labels. Where the reason can be checked against the excerpts it often does not describe them (see 2), so treat the label 0 as a strict reading of the need, not as 'nothing relevant was retrieved'.

### 1. Did the pool rows have evidence?

Of the 39 newest labels, **22 are flagged**. 

| Pair | Evidence status | Excerpts shown | Evidence characters | Label | Rank |
|---|---|---|---|---|---|
| Q25 Docks Oyster Bar | retrieved | 3 | 459 | 0 | 14 |
| Q25 Hot n Juicy Crawfish | retrieved | 3 | 802 | 0 | 15 |
| Q25 Bed-Stuy Fish Fry | retrieved | 3 | 435 | 0 | 17 |
| Q25 Red Lobster | retrieved | 3 | 600 | 0 | 20 |
| Q30 Olympia Pita | retrieved | 3 | 292 | 0 | 12 |
| Q30 Upside Craft | retrieved | 3 | 538 | 0 | 13 |
| Q30 Sunflower Cafe | retrieved | 3 | 748 | 0 | 20 |
| Q30 Cafe Renaissance | retrieved | 3 | 446 | 0 | 19 |
| Q30 Kingston Bake Shop | retrieved | 3 | 212 | 0 | 14 |
| Q30 Mozzarella | retrieved | 3 | 277 | 0 | 16 |
| Q30 Grill On Lee | retrieved | 3 | 255 | 0 | 17 |
| Q31 Simply Sweet | retrieved | 3 | 767 | 0 | 11 |
| Q31 Egg | retrieved | 3 | 747 | 0 | 17 |
| Q31 Dao Palate | retrieved | 3 | 949 | 0 | 12 |
| Q31 Buntopia | retrieved | 3 | 997 | 0 | 14 |
| Q31 Natural Blend Vegan Cafe & Juice Bar | retrieved | 3 | 773 | 0 | 16 |
| Q33 Zucker's Bagels & Smoked Fish | retrieved | 1 | 280 | 0 | 12 |
| Q33 Franchia Vegan Café | retrieved | 1 | 272 | 0 | 11 |
| Q33 Brooklyn Natural Market & Cafe | retrieved | 1 | 357 | 0 | 13 |
| Q34 Gargiulo's | retrieved | 3 | 322 | 0 | 18 |
| Q34 Monty Q's | retrieved | 3 | 615 | 0 | 20 |
| Q34 Brooklyn Roots NYC | retrieved | 3 | 814 | 0 | 11 |

Evidence status of the flagged rows: retrieved; rows with an empty evidence set: 0 of 22. All 149 pool rows have status `retrieved` with at least one excerpt, so none was an empty prompt.

### 2. Is the judge's reason consistent with the excerpts shown?

Five of the 22 flagged pairs, drawn with seed 6165. Excerpts are as on the pool row (truncated to 400 characters here).

**Q30 kosher bakery / Olympia Pita**: label 0; judge's reason: "no evidence found"

> [review 2657687, 2018-06] Great Kosher food made fresh at the restaurant and super authentic. RECOMMENDED
> [review 2657644, 2018-10] The best place to get a good meal always fresh ! most kosher stores don’t have good food these days
> [review 2657934, 2015-12] Standard kosher middle eastern fare.

**Q31 nut-free brunch in brooklyn, one of our friends is vegan / Buntopia**: label 0; judge's reason: "no evidence found"

> [review 2219055, 2018-02] Yummy food alert! I randomly found this place on Yelp while searching for vegan sushi and holy cow this was the best meal I've had in 2018. Definitely a must try! It's an omni Asian fusion restaurant in Bedstuy with a half vegan menu, seems to be owned and operated by a family, very affordable, and lovely environment. …
> [review 2219064, 2019-11] I am vegetarian and I live right next door, so I love this place! This place has very healthy and tasty Japanese food that's not super expensive, and the best part is the large number of options available for vegetarians (yay!). Plus it has a very bright feel and ambience, and the people are very nice. …
> [review 2219054, 2019-09] Yum!!! I’ve never had vegan sushi - it was delicious! Basically, I read a bunch of reviews to figure out what to try, and ended up ordering the popular things: mango tango; happy bull-da; and a watermelon shake. (I think the shake has lots of strawberries in it as well - seed city.) Eat here!

**Q31 nut-free brunch in brooklyn, one of our friends is vegan / Natural Blend Vegan Cafe & Juice Bar**: label 0; judge's reason: "no evidence found"

> [review 888126, 2020-03] If you dont know now you know ... Promise you wont dissappointed best vegan mac and chesse in all of america ... The best and healthest food in brooklyn without compromising taste in tradional food that that has always been a cultural staple in bed stuy
> [review 888201, 2019-06] One of the best foods I've had a vegan restaurants...and the prices are great
> [review 888197, 2018-07] This quaint vegan restaurant in the bed stuy neighborhood if Brooklyn is extremely delicious and posses a semi-robust menu of various plant based dishes and beverages. The staff is friendly and aim to answer any/all inquiries regarding customizing dishes or specific ingredients. The outdoor seating has a eye capturing aethestic with distinctive plants and flowers. …

**Q33 i have a real dairy allergy not just lactose intolerance, who takes that seriously / Zucker's Bagels & Smoked Fish**: label 0; judge's reason: "no evidence found"

> [review 61190, 2020-02] … Twice. They don’t seem to be serious or care about allergies. They person who did it found it funny and even though I called the supervisor he really didn’t care at all either. People with allergies or that simply don’t eat dairy: be aware of this place.

**Q33 i have a real dairy allergy not just lactose intolerance, who takes that seriously / Brooklyn Natural Market & Cafe**: label 0; judge's reason: "no evidence found"

> [review 456007, 2021-03] … I have been dying trying to avoid these restaurants where everything contains dairy. If it’s not cheese it’s an egg or milk batter and my skin just cannot. I have an allergic reaction to ALL things containing the smallest amount of dairy. It’s annoying ordering burgers saying no cheese or Mayo and still getting either or both. …

Reading of the five (the reasons are all the bare phrase, so this is an inference from the excerpts): every pair had topical excerpts, so "no evidence found" does not describe what was shown. A 0 looks defensible for the two Q33 pairs (a warning about dismissed allergies, and an allergic reviewer's complaint) and plausibly for Olympia Pita under the category-filter rule (kosher food, no sign of a bakery). It is more debatable for the two Q31 pairs, which show strong vegan evidence but nothing on nut-free or brunch: whether that is a 0 or a 1 depends on how strictly the rubric's combined-need rule is applied. Whether a 0 is justified is a labeling question this audit cannot settle without a human label.

**Did the judge see the excerpts?** Judge prompt size against evidence size across the pool (prompt tokens = 1238 + 0.242 x evidence characters, n = 149): the fixed part of the prompt is about 1238 tokens, so a call with no evidence would use about that many. Mean prompt tokens: flagged 1372 (mean evidence 560 characters), not flagged 1380 (583 characters). Mean residual from the fit: flagged -1, not flagged +1 tokens. Residuals near zero for both mean the flagged calls were given their excerpts like the others.

### 3. Share by retrieval rank (semantic_capped)

| Group | Rank 1-10 | Rank 11-20 |
|---|---|---|
| Extended pool (149 pairs) | 0 / 0 | 51 / 149 (34%) |
|   earlier batch of the extended pool | 0 / 0 | 29 / 110 (26%) |
|   newest 39 | 0 / 0 | 22 / 39 (56%) |
| All top-20 dev pairs (model calls only) | 56 / 200 (28%) | 71 / 197 (36%) |

Pairs labeled by the no-evidence rule (no excerpts at all) are excluded from these shares.

### 4. Share per query

| Query | Category | Extended pool pairs | Flagged | Share | Newest-39 pairs | Newest flagged |
|---|---|---|---|---|---|---|
| Q25 | allergy | 7 | 7 | 100% | 4 | 4 |
| Q30 | diet_plus_filter | 7 | 7 | 100% | 7 | 7 |
| Q12 | single_diet | 10 | 9 | 90% | 0 | 0 |
| Q31 | mixed_group | 8 | 5 | 62% | 8 | 5 |
| Q10 | single_diet | 10 | 6 | 60% | 0 | 0 |
| Q34 | allergy | 5 | 3 | 60% | 5 | 3 |
| Q02 | single_diet | 6 | 2 | 33% | 0 | 0 |
| Q33 | allergy | 9 | 3 | 33% | 9 | 3 |
| Q19 | allergy | 7 | 2 | 29% | 0 | 0 |
| Q20 | allergy | 7 | 2 | 29% | 0 | 0 |
| Q18 | mixed_group | 8 | 2 | 25% | 0 | 0 |
| Q22 | allergy | 9 | 2 | 22% | 0 | 0 |
| Q17 | mixed_group | 7 | 1 | 14% | 0 | 0 |
| Q03 | single_diet | 6 | 0 | 0% | 0 | 0 |
| Q06 | single_diet | 9 | 0 | 0% | 0 | 0 |
| Q07 | single_diet | 6 | 0 | 0% | 0 | 0 |
| Q13 | mixed_group | 7 | 0 | 0% | 0 | 0 |
| Q15 | mixed_group | 6 | 0 | 0% | 0 | 0 |
| Q16 | mixed_group | 9 | 0 | 0% | 0 | 0 |
| Q28 | diet_plus_filter | 6 | 0 | 0% | 6 | 0 |

The three queries with most flagged newest labels (Q30, Q31, Q25) account for 16 of 22.
