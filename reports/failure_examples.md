# Retrieval failure examples (smoke queries, before judging)

Found while comparing doc TF-IDF (`src/retrieval/tfidf.py`), chunk TF-IDF (`src/retrieval/tfidf_chunks.py`, density `log`)
and semantic retrieval (`src/retrieval/semantic.py`, all-MiniLM-L6-v2, density `log`) on five smoke queries,
top 10 each. Raw results: [`retrieval_comparison.json`](retrieval_comparison.json), produced by
`python -m eval.compare_systems`. Rubric references are to [`eval/RUBRIC.md`](../eval/RUBRIC.md).

These are qualitative examples from a single run of five queries, not measured error rates. Snippets are quoted exactly as the
system returned them (excerpts are cut to about 60 words, `…` marks a cut). Review ids are `review_id` in
`reviews_clean.jsonl`; dates are the review month.

| Query id | Query | Borough filter |
|---|---|---|
| VEG | vegan options for a group | Brooklyn |
| Q20 | restaurant careful about peanut allergies | — |
| Q22 | safe for celiac disease, no cross contamination | — |
| Q25 | shellfish allergy | — |
| Q29 | halal chinese food | Manhattan |

## Summary

| # | Case | System | Kind of failure |
|---|---|---|---|
| 1 | VEG: TACO ZONE at #1 on a negative review; tiny restaurants | doc TF-IDF | Polarity-blind term match; length bias |
| 2 | Q25: "fish allergy" accommodation with shellfish substitutes | chunk TF-IDF (also doc TF-IDF) | Wrong allergen; evidence of the opposite |
| 3 | Q29: halal dropped, generic Chinese restaurants | semantic | Rare constraint lost; density bonus |
| 4 | Q20: "DO NOT GO" peanut warnings at #1 and #2 | semantic | Polarity-blind semantic match |
| 5 | Q25: shellfish dishes instead of allergy accommodation | semantic | Allergen conflated with dish |
| 6 | Q22: Lilli and Loo, conflicting evidence differing by system | all three | Mixed evidence; snippet choice drives the label |
| 7 | Q29: ranked on "chinese food", no snippet | chunk and doc TF-IDF | Ranking and evidence filter disagree |
| 8 | Semantic always returns at least one snippet, even off-topic | semantic | Off-topic evidence shown to judges |

---

## 1. Doc TF-IDF: #1 result backed by a negative review; length bias

**Query:** VEG "vegan options for a group" (Brooklyn). **System:** doc TF-IDF. **Rank:** 1.

**TACO ZONE** (5 clean reviews), score 0.066. Its first snippet:

> [review 55800, 2021-05] Not much for vegan options, beans/veg as a substitute does not cut it.

Its other two snippets are positive and older:

> [review 55802, 2020-10] Delicious. Best chicken burrito I’ve ever had and the vegan chili was delicious

> [review 55797, 2021-04] Went here for the vegan food and loved it! Got the tofu chorizo burrito. It was humongous and yummy.

**Why it is a failure.** TF-IDF counts "vegan" and "options" without regard to polarity. The highest-scoring evidence says
the opposite of the need. The restaurant reaches #1 because its document is tiny: five short reviews in which "vegan"
is a large share of the text. Across the five queries, the median top-10 restaurant has far fewer reviews under doc TF-IDF:

| Median clean reviews of the top 10 | VEG | Q20 | Q22 | Q25 | Q29 |
|---|---|---|---|---|---|
| doc TF-IDF | 23 | 24 | 174 | 111 | 19 |
| chunk TF-IDF | 407 | 605 | 236 | 711 | 405 |
| semantic | 442 | 605 | 236 | 711 | 794 |

For VEG, doc TF-IDF's top 10 have 5, 6, 23, 35, 22, 16, 27, 62, 81 and 7 reviews. Chunk TF-IDF's have 134–1,230.
This is the length bias that chunk TF-IDF was written to remove.

**Rubric.** *Mixed evidence*: "label by the most recent and most specific evidence". The most recent review (2021-05)
is the negative one, so this is 0, or 1 at most. *Several needs*: "for a group" is not covered at all, so it cannot be 2.

## 2. Q25 chunk TF-IDF: "fish allergy" snippet for a shellfish query

**Query:** Q25 "shellfish allergy". **System:** chunk TF-IDF, rank 8 (doc TF-IDF returns the same review at rank 10).

**Tempura Matsui** (77 reviews), score 0.424:

> [review 1176631, 2018-12] We had the tempura set lunch and the chef’s special set lunch, both as an omakase. They were very accommodating with my fish allergy and replaced it with vegetable and shellfish options. The food was delicate but delicious and visually beautiful. The restaurant had the perfect ambiance, a balance of elegant but comfortable. Service was impeccable.

**Why it is a failure.** The review matches both query words ("shellfish", "allergy"), but the allergy is to fish, and
the accommodation was to serve *shellfish* instead. For a shellfish-allergic diner, this is evidence that the kitchen
uses shellfish as a substitute, not that it is careful about shellfish. (Before commit 44cda93, this result also showed
another restaurant's review; it now has 2 snippets, both its own.)

**Rubric.** *Allergy queries*: "Negative or unrelated mention = 0"; a fish-allergy accommodation is unrelated to the
shellfish need. *A dish that merely contains the allergen never counts as evidence.* Label: 0.

## 3. Q29 semantic: halal dropped

**Query:** Q29 "halal chinese food" (Manhattan). **System:** semantic (density `log`). **Ranks:** 1–10.

The top 10 are generic Chinese restaurants. **None of the 30 snippets mentions halal.** First snippet at each rank:

| Rank | Restaurant (reviews) | First snippet |
|---|---|---|
| 1 | Café China 倾城 (895) | "Best Chinese food I've ever had and the first time I've liked Chinese food as much in the US as at restaurants in China. …" [2112190, 2019-05] |
| 2 | Wo Hop (1,237) | "Can honestly say its the best chinese food I have ever eaten! We have the chicken fried rice, pork fried Dumplings and General Tso Chicken. …" [1548403, 2018-12] |
| 3 | Joe's Shanghai (1,553) | "Reasonably priced delicious Chinese food with compromised service. The Xiao long bao (soup dumplings) is amazing here. …" [2238815, 2020-03] |
| 4 | The Original Buddha Bodai Kosher Vegetarian Restaurant (576) | "The best Chinese food I have ever had, and it is kosher! Halleluyah." [72979, 2016-11] |
| 5 | Congee Village (547) | "… Beijing style pork chops and two types of porridge (seafood and scalloped) were tried. …" [2115254, 2018-12] |

Ranks 6–10: Joe's Shanghai (second location), Hop Kee, Xi'an Famous Foods, The Cottage, Great NY Noodletown. Several
snippets name pork dishes ("pork fried Dumplings", "Beijing style pork chops", "salt and pepper porkchop").

**Why it is a failure.** The query's rare constraint (halal) is drowned out by its common topic ("chinese food"). That is
typical of a dense model: "chinese food" dominates the query embedding. The density bonus makes it worse. The query
passes 6.25% of all chunks above `MATCH_THRESHOLD` (0.40), so large restaurants collect many matching chunks. In `log`
mode, the bonus is 20% of the top-10 scores. Halal Kitchen, the only restaurant that both TF-IDF systems rank #1, moves as follows:

| Density mode | Halal Kitchen's semantic rank | Median reviews of top 10 |
|---|---|---|
| `log` (default) | 12 | 715 |
| `none` | **1** | 461 |
| `capped` | **1** | 461 |

**Rubric.** *Several needs*: "2 only if all parts are covered. One part covered = 1." A Chinese restaurant with no halal
evidence is 1 at most, and should be 0 if the reviews show pork as the main offering. *Noisy words*: "kosher" (rank 4)
is not halal. *Do not use outside knowledge*: a restaurant cannot be judged halal from its cuisine or name.

## 4. Q20 semantic: "DO NOT GO" peanut warnings at ranks 1–2

**Query:** Q20 "restaurant careful about peanut allergies". **System:** semantic. **Ranks:** 1 and 2.

**#1 Pig and Khao** (492 reviews), score 2.146:

> [review 2117639, 2020-03] DO NOT GO IF YOU HAVE A SEVERE PEANUT ALLERGY. i have a severe allergy to peanuts, so I called the restaurant the night before my friends and I had brunch to ask about how Pig&Khao fares with people who allergies to peanuts. The manager said people come in often who have allergies and it won’t be a problem. …

> [review 2117626, 2020-03] … I end up having something that clearly had either touched or been cooked near peanuts / peanut oil and I go into an allergic reaction. Thank god im alive but I will never go to this restaurant again and you should not either if you or your family/friends have severe allergies.

**#2 Mission** (396 reviews), score 2.133:

> [review 1576831, 2017-08] If you or anyone in your party suffers from food allergies, please avoid this restaurant. Unfortunately, the waitstaff and kitchen are not attentive to food allergies. We informed the waitstaff when ordering concerning one diner in our party with a tree nut and sesame allergy. …

> [review 1576786, 2017-07] If you have a food allergy, you should seriously consider going elsewhere. Our party of four had one nut and one wheat allergy--both of which were explicitly shared with our waitress when our order was placed. …

**Why it is a failure.** These reviews are about exactly the topic of the query (peanut or nut allergy, whether staff are
careful), so their embeddings are very close to it, but they report the opposite. Semantic similarity does not encode
polarity. This is the most dangerous failure class for an allergy product: the top results are explicit warnings.
Chunk TF-IDF also ranks Pig and Khao (#5) and Five Guys ("Not for anyone with peanut allergies"), so the problem is not
unique to semantic retrieval, but semantic puts the warnings first.

**Rubric.** *Allergy queries*: "Negative or unrelated mention = 0" (example: "got sick"). Pig and Khao has a claimed
accommodation ("it won’t be a problem") followed by a reaction. *Both an accommodation and a warning*: "If the warning is
specific and recent, 0." Mission: "not attentive to food allergies" is 0.

## 5. Q25 semantic: shellfish dishes instead of allergy accommodation

**Query:** Q25 "shellfish allergy". **System:** semantic. **Ranks:** 3–5 (rank 2, Estiatorio Milos, is similar).

**#3 Hot n Juicy Crawfish** (387 reviews):

> [review 1145639, 2019-05] … Got a combo, oysters, and blue crabs. The food was pretty good, but they messed up the spice level on the crabs. …

**#4 Oyster** (1,310 reviews):

> [review 1316591, 2019-12] Great fresh shellfish... Just don't order the fish because the one I ordered was dry and not very flavourful

**#5 Lobster Place** (711 reviews):

> [review 2090258, 2017-01] Lots of choice from lobster to fish to shell fish

**Why it is a failure.** The model treats "shellfish allergy" as being about shellfish, and returns places that serve it.
Of semantic's 29 Q25 snippets, 16 name a shellfish dish without any allergy word (chunk TF-IDF: 12 of 23; doc TF-IDF:
1 of 12). It also blurs seafood categories ("Great fresh shellfish… don't order the fish"). Its #1 (Red Lobster) is a
more useful near-miss: "I specifically told Adam I have a seafood allergy", which generalizes shellfish to seafood.
Doc TF-IDF is the most literal here: 10 of its 12 snippets contain "shellfish allergy" or "allergic to shellfish".

**Rubric.** *A dish that merely contains the allergen never counts as evidence.* *Negative or unrelated mention = 0.*
All three examples: 0.

## 6. Q22: Lilli and Loo with conflicting evidence across systems

**Query:** Q22 "safe for celiac disease, no cross contamination". **Systems:** semantic #3, chunk TF-IDF #3,
doc TF-IDF #2. All three retrieve the restaurant, but **each shows a different set of its reviews**.

The restaurant's relevant reviews, in date order:

| Review | Date | Text (as returned) | Shown by |
|---|---|---|---|
| 2070140 | 2017-03 | "Great gluten free options! I have Celiac disease and am very sensitive to cross contamination. Lilli & Loo's staff has always been very knowledgeable about preventing cross contamination and I've never gotten sick. …" | chunk, doc |
| 2070193 | 2018-01 | "I have celiac disease and unfortunately this place is not careful anymoreI ordered an egg drop soup explain to the waitress that I had celiac disease they gave me a totally different soup … I would have been deathly ill …" | all three |
| 2070173 | 2018-04 | "Solid Asian fusion with a nice gluten free menu. I've had Celiac's disease for over ten years and it is wonderful to have access to do many old favorites …" | semantic |
| 2070194 | 2018-06 | "Tasty food and good for celiac/gluten free" | semantic |
| 2070149 | 2019-10 | "The food is flavored and good for the price I had to give it 3 stars because of cross-contamination The water glass was dirty Other than that it’s 👍🏼" | chunk, doc |

**Why it matters.** This is not a retrieval error: the restaurant has real, conflicting evidence. But the label a judge
gives depends on which three snippets they see. Semantic shows one specific warning and two vague positives. The TF-IDF
systems show the strongest accommodation, the warning, and the most recent cross-contamination complaint. The pool uses
the first system's snippets (chunk TF-IDF), so judges will see the TF-IDF set.

**Rubric.** *Both an accommodation and a warning*: 1, "and put the reason in your own notes. If the warning is specific
and recent, 0." *Mixed evidence*: "label by the most recent and most specific evidence". The 2018-01 warning is specific
(wrong soup served to a celiac diner). The most recent review (2019-10) mentions cross-contamination but is vague. Likely
label: 0 or 1; judges should check the Google Maps reviews, as the rubric allows.

## 7. Q29 TF-IDF: ranked on "chinese food", with no snippet

**Query:** Q29 "halal chinese food" (Manhattan). **Systems:** chunk TF-IDF and doc TF-IDF.

| System | Ranks with no snippet | Restaurants |
|---|---|---|
| chunk TF-IDF | 2, 3, 4, 5, 8, 10 | Wo Hop, OLLIE'S SICHUAN, Wo Hop Next Door, Wok 88, Café China, Han Dynasty |
| doc TF-IDF | 3, 5, 6, 8, 9, 10 | The China Star, Ho Lee, Empire, Guo's Garden, Silver Palace, Crispy Bamboo Village |

**Why it is a failure.** Ranking and evidence use different rules. These restaurants are ranked on chunks matching
"chinese" and "food". But snippet selection keeps only reviews containing a *requirement* term (`requirement_terms()` in
`src/retrieval/evidence.py`: the query's rarest terms by idf, which here is only "halal"). So 6 of each system's top 10
have no evidence for the one constraint that matters. The empty snippet is, in effect, an honest signal ("no halal
mention"), but the restaurant still ranks above places with halal evidence (chunk TF-IDF ranks The Halal Guys at #6 and #9).

**Rubric.** "0 = Not relevant: no evidence for the need". Judges should still check the reviews on Google Maps ("when
they are not enough"), but a restaurant with no halal mention in its reviews is 0 or 1 under *Several needs*.

## 8. Semantic always returns at least one snippet, even off-topic

**System:** semantic, all queries. By design (`SNIPPET_THRESHOLD` in `src/retrieval/semantic.py`), the best-matching
review of every result is returned, even if its similarity is below 0.30. Further snippets need at least 0.30.

Examples:
- Q29: all 30 snippets, including at least one per result, contain no mention of halal (case 3).
- VEG #1 Champs Diner, first snippet: "First time eating vegan food and although the food was good the service wasn't. We
  just stumbled along this place and thought we should try it out. I had the Mexican bean salad …" [431224, 2016-08].
  It is about vegan food, but mostly about service and says nothing about groups or options.

**Why it matters.** TF-IDF returns *no* snippet when nothing matches the requirement terms (case 7), which tells the
judge "no evidence". Semantic always returns something, which can look like evidence when it is not. For restaurants found
only by semantic, these are the snippets judges will see.

**Rubric.** "Judge the evidence about the need, not how good the restaurant is overall." An off-topic snippet is not
evidence; judges should treat it like an empty snippet and check the reviews.

---

## Crude regex counts (supporting, not findings)

`eval/compare_systems.py` counts keyword patterns in the 10 results' snippets. The patterns are crude. For example,
"cross contamination" is counted as negative, which inflates the Q22 TF-IDF numbers. Group words miss phrases like
"non-vegans". Use these only as pointers to the qualitative cases above.

| Count (snippets) | VEG sem / chunk / doc | Q20 | Q22 | Q25 |
|---|---|---|---|---|
| Group words | 2 / 1 / 1 | 8 / 7 / 3 | 3 / 3 / 1 | 2 / 3 / 4 |
| Negative patterns | 1 / 1 / 2 | 3 / 3 / 0 | 3 / 12 / 9 | 7 / 1 / 0 |
| Shellfish dish, no allergy word | — | — | — | 16 / 12 / 1 |
| "shellfish allergy" phrasing | — | — | — | 1 / 6 / 10 |
| Total snippets | 30 / 30 / 27 | 27 / 28 / 16 | 29 / 28 / 24 | 29 / 23 / 12 |

Top-10 overlap between systems:

| | VEG | Q20 | Q22 | Q25 | Q29 |
|---|---|---|---|---|---|
| semantic ∩ chunk TF-IDF | 4 | 3 | 8 | 4 | 2 |
| semantic ∩ doc TF-IDF | 0 | 0 | 6 | 0 | 0 |
| chunk ∩ doc TF-IDF | 0 | 1 | 7 | 3 | 2 |
| Union of the three top-10s | 26 | 26 | 15 | 23 | 26 |

Snippet ownership: 0 foreign snippets for every system and query (after commit 44cda93).
