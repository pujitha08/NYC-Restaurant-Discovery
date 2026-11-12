# Sentence labeling guide

Open your `sentences_<your name>.csv` (Google Sheets or Excel; save it back as CSV UTF-8, same file name) and label
each review sentence on its own, without discussing it with the others. In **`stance`** write one of:
**`accommodation`** (the restaurant handled an allergy or diet need well: "they made my dish without peanuts",
"dedicated gluten-free fryer"), **`negative`** (a warning, refusal, failure or reaction: "they couldn't
guarantee no cross-contact", "I got sick"), **`mention`** (the allergen, ingredient or diet is only mentioned,
with nothing about handling it: "the peanut noodles were great", "lots of vegan options"), or
**`not_about_allergy`** (the matched word isn't about food allergies or diets at all: "kosher salt", "nuts about
this place"). **`allergens_mentioned`** is pre-filled by a model as a suggestion: correct it to the FDA allergens
the sentence actually mentions, separated by `;` (milk, egg, fish, shellfish, tree_nuts, wheat, peanuts, soy,
sesame; gluten and celiac count as wheat; leave it empty if none). Add a short `note` when unsure. Some
sentences appear in everyone's file on purpose, to measure agreement. It takes about 45 minutes; return the file
by Friday 10 AM.
