# Demo guide

## Launch

```bash
cd NYC-Restaurant-Discovery
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

The home screen has the headline, the search box, a scope line, a "Where" selector, two rows of chips (Allergies, Diets and
groups) and a three-card strip. A chip fills the box and runs the search; chips are answered from the saved results when one
exists (instant, no model call). **Evaluation tools** (header toggle) hold Compare rankings, the retrieval method (default
`semantic_capped`), "fewer than 5 reviews" and Demo mode. The stance re-ranker is an optional experiment: its held-out
evaluation showed no significant gain (NDCG@10 0.780 vs 0.787, +0.007, p = 0.74), so don't present it as better; Compare
rankings is for showing how it re-orders results, not for claiming an improvement.

After a search, a staged progress box collapses to "Done", then: "We understood:" chips with a Change link, the heading "What
reviewers say about <need>", a count line, the real timings, and two columns, **Reviews mention allergy care** and **Reviews
report problems**. Each card shows ONE highlighted quote, a "Based on N reviews" pill and "See the evidence". A restaurant
with both kinds of claim is in both columns ("Reviewers disagree"). Allergy requests never show "Supported" or "safe". If a
summary fails, the card offers "Try the summary again". Say "reviews mention", never "safe".

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

- **Chips**: one click fills the box and runs the search (the queries below are chip queries or saved demo queries).
- **Compare rankings** (tick in the form): `semantic_capped` and `semantic_capped_stance` side by side, with
  ▲/▼/new marks on the stance column and a "🟩 accommodate / 🟥 warns / 🟨 mention" bar per restaurant. The stance
  column labels review sentences with the model on first use for a query (about 5 Groq calls, about a minute);
  with **Demo mode** on it uses only labels already saved on disk and otherwise shows the base ranking alone.
  No summaries in this view.
- **Evidence bar**: results ranked by `semantic_capped_stance` show the same bar and the strongest warning sentence.
  The stance labeler is chosen by `STANCE_LABELER` in `src/config.py` (`"llm"` by default); see the README's "Stance
  re-ranker" section. The weights chosen on the dev queries are in `reports/stance_rerank_frozen.json`; they are dev
  numbers. The held-out run (`reports/stance_heldout.md`) found no significant gain, so the default stays `semantic_capped`.
- Query terms are highlighted in the quotes and review excerpts (never inside the restaurant's own name).

## Cached demo results (offline / rate-limit fallback)

The four queries above (and `halal chinese food` with `semantic_log`) are saved in `data/demo_cache/*.json`
(public review text already shown in the app; no keys). Tick **Demo mode** in the search form to answer an exact
match (same query text, borough and method) from the cache without calling the language model. The app also
falls back to the cache by itself when a live search hits a rate-limit, timeout or network error and the query
matches. Cached results are always labeled **"cached demo result"**; results past the top 3 have no Explain.
Other queries run live. In a cached result the status line just says "Done" and a grey notice says it is a saved result
with its date. The four `semantic_capped` entries were regenerated live on 2026-11-16 with the fixed summarizer wording: a claim's
wording matches its label, a warning leads over relayed staff reassurance (Pig and Khao now opens with the warning), and for
allergy requests claims avoid safety language in the app's own voice; a claim that still used it is replaced by "A reviewer's
comment about this allergy is quoted below." with its quoted review kept (this happens on some Celiac safe cards). The
`semantic_log` halal entry is still the older 2026-11-13 run. Cached results have summaries only for the top 3, so the
results summary line counts three cards. Wording and a few statuses can differ from one live run to the next, so the
descriptions in "Four demo queries" are indicative, not exact.

Regenerate (needs the semantic index and a working Groq key; takes about 4 minutes):
`python -m app.demo_cache --build`.

## If something goes wrong

- **"The language model's rate limit was reached"** or "Some summaries are unavailable": the results and review
  excerpts still show. Wait about 60 seconds, then click **Explain** on a result (or search again). Don't click
  repeatedly; each click uses tokens.
- **Slow first search**: that is the warm-up; it is fast afterwards.
- **No summaries at all / setup message**: check `.env` (key and model) and restart Streamlit.
- **Fallback**: tick **Demo mode** (see above), or `reports/failure_examples.md` (case 9) and `reports/metrics_summary.md` (provisional metrics)
  tell the before/after and results story without the live app. The other reports live in `reports/` (listed in the README).

## Privacy

Never commit `.env` or any key, `eval/labels/*` (rater files), `eval/llm_labels*.jsonl` (judge labels) or the judging pools;
`.gitignore` covers them. Only `data/demo_cache/` is tracked under `data/`.
