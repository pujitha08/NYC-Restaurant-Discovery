# Demo guide (integration/demo)

## Launch

```bash
cd NYC-Restaurant-Discovery
git checkout integration/demo
source ~/tf-env/bin/activate        # or your venv with requirements.txt installed
streamlit run app/streamlit_app.py  # opens http://localhost:8501
```

Needs `data/processed/` (with the `semantic/` index) and a `.env` with `GROQ_API_KEY` and `GROQ_MODEL`.
Without a key the app still searches, without parsing or summaries.

**The demo runs on `openai/gpt-oss-20b`** (`GROQ_MODEL=openai/gpt-oss-20b` in `.env`) to keep the
`gpt-oss-120b` daily quota for the LLM judge, which passes `--model openai/gpt-oss-120b` explicitly and
ignores `GROQ_MODEL`. Groq limits are per model, so the two don't compete.

## Warm up first (2 minutes before the demo)

The first search loads the embedding index and model (about 5 s extra). Run one throwaway search, for example
"vegan food", before presenting and keep the tab open. Leave about **30 seconds between searches**: each search
uses about 3.5-4K Groq tokens (parse plus 3 summaries) and Groq allows 8K tokens per minute.

## Reading the results

The first screen is a title, a one-line value proposition, the search box with the four example chips right under it,
borough pills, **Compare rankings** and **Demo mode** toggles, and a collapsed **Advanced settings** expander (retrieval
method, "fewer than 5 reviews"). A slim note says reviews are historical and anecdotal; the full-strength warning
appears only above allergy results.

After a search, a staged progress box ("Understanding your request", "Searching 605K review passages", "Reading
reviews", "Summarizing") collapses to one line, then chips show the parsed request (diets, allergens, borough, result
count). Results are cards: the title row is "1. Champs Diner — Brooklyn · ★ 4.5 · ✓ Supported" (the badge is the
evidence status: ✓ Supported, ◐ Mixed evidence, ⚠ Reviews report problems, ○ Insufficient evidence, ℹ Reviews discuss
this — read each claim, or "… Not summarized yet"). The top card starts open. Inside a card: rating, review count and
address; ONE status row (the verdict plus, for allergy requests, the allergy reminder); the top claim as a block with a
colored left border and a plain label (Warning / Positive / Mention) and its best quote with the review date; and
**Show all evidence** for the other claims and every retrieved excerpt. Warnings are ordered first, so the claim shown by
default is never milder than a hidden one. Allergy requests never show "Supported" or "Positive": a supporting claim there
is only a "Mention". Results 4-10 say "Not summarized yet"; open one and click **Explain**. **Expand all** / **Collapse
all** sit above the list.

## Four demo queries

Use the default method (`semantic_capped`) unless noted.

1. **"vegan options for a group"**, borough **Brooklyn**. Expected: vegan-friendly Brooklyn places (Champs Diner,
   The VSPOT, Modern Love or similar) marked "✓ Supported" in the list. Click Champs Diner to show a summary such
   as "non-vegan friends love it" with its cited review. Shows the parsed query (diet vegan, group need) in
   "How your request was understood".
2. **The before/after story: "restaurant careful about peanut allergies"**, borough **Both**.
   - *Before* (reports/failure_examples.md, case 9): Mission was shown as **"supported"** with ✓ marks, although
     both reviews say to avoid it if you have food allergies.
   - *Now*: Mission's row shows **"⚠ Reviews report problems"**; no row shows ✓. Click Mission to show the
     warnings: the "Reviews report problems" warning box, the reviews marked "⚠ Contradicts", plus the note
     that allergy suitability can't be confirmed from reviews. Pig and Khao shows **"◐ Mixed evidence"** or
     **"⚠ Reviews report problems"** depending on the model run (a manager's reassurance, then a reaction). No allergy query ever shows "supported". (Checked live on gpt-oss-20b: Mission and Pig and Khao
     both "Reviews report problems". The 20b parser can miss the allergen field, but the guard still applies
     because the request text mentions allergies.)
3. **"safe for celiac disease, no cross contamination"**, borough **Both**. Expected: Senza Gluten and
   Friedman's near the top with the neutral badge "ℹ Reviews discuss this — read each claim" (click one to read
   each claim); Lilli and Loo shows **"◐ Mixed evidence"**: click it to show one 2018 report of a celiac
   diner given the wrong soup alongside positive reviews, which is the honest answer.
4. **"halal chinese food"**, borough **Manhattan**, then switch the method to `semantic_log` and search again.
   Expected: with `semantic_capped`, Halal Kitchen is at or near the top; with `semantic_log`, generic Chinese
   restaurants crowd it out (the density bonus favors big restaurants). This is why the default is capped.

Point out the caveat at the top (reviews end September 2021; not safety guarantees) and that rankings reflect review
relevance, not safety.

## Page features

- **Try an example**: four buttons (the queries below) fill the box and borough and run the search.
- **Compare rankings** (tick in the form): `semantic_capped` and `semantic_capped_stance` side by side, with
  ▲/▼/new marks on the stance column and a "🟩 accommodate / 🟥 warns / 🟨 mention" bar per restaurant. The stance
  column labels review sentences with the model on first use for a query (about 5 Groq calls, about a minute);
  with **Demo mode** on it uses only labels already saved on disk and otherwise shows the base ranking alone.
  No summaries in this view.
- **Evidence bar**: results ranked by `semantic_capped_stance` show the same bar and the strongest warning sentence.
- Query terms are highlighted in the review excerpts; the sidebar explains the stages, the data size and the badges.

## Cached demo results (offline / rate-limit fallback)

The four queries above (and `halal chinese food` with `semantic_log`) are saved in `data/demo_cache/*.json`
(public review text already shown in the app; no keys). Tick **Demo mode** in the search form to answer an exact
match (same query text, borough and method) from the cache without calling the language model. The app also
falls back to the cache by itself when a live search hits a rate-limit, timeout or network error and the query
matches. Cached results are always labeled **"cached demo result"**; results past the top 3 have no Explain.
Other queries run live. The cached summaries come from one earlier live run, so wording and a few statuses can
differ from the descriptions above (in the saved run Pig and Khao is "Reviews report problems" and Mission
"Insufficient evidence").

Regenerate (needs the semantic index and a working Groq key; takes about 4 minutes):
`python -m app.demo_cache --build`.

## If something goes wrong

- **"The language model's rate limit was reached"** or "Some summaries are unavailable": the results and review
  excerpts still show. Wait about 60 seconds, then click **Explain** on a result (or search again). Don't click
  repeatedly; each click uses tokens.
- **Slow first search**: that is the warm-up; it is fast afterwards.
- **No summaries at all / setup message**: check `.env` (key and model) and restart Streamlit.
- **Fallback**: tick **Demo mode** (see above), or `reports/failure_examples.md` (case 9) and `reports/metrics_summary.md` (provisional metrics)
  tell the before/after and results story without the live app.
