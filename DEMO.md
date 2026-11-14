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

Results are a compact list: each one is a collapsed row such as
"1. Champs Diner — Brooklyn · ★ 4.5 · ✓ Supported". The badge is the evidence status (✓ Supported, ◐ Mixed
evidence, ⚠ Reviews report problems, ○ Insufficient evidence, ℹ Reviews discuss this — read each claim, or
"… Not summarized yet"), followed by "Low evidence" when it applies. Allergy queries never show ✓ or "Supported".
Click a row to open it: address, review count, the summary with each claim and its cited excerpt and date,
limitations, the allergy note and all retrieved excerpts. **Expand all** / **Collapse all** sit above the list.
Results 4-10 say "Not summarized yet"; open one and click **Explain** (it stays open). A new search starts
collapsed.

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

## If something goes wrong

- **"The language model's rate limit was reached"** or "Some summaries are unavailable": the results and review
  excerpts still show. Wait about 60 seconds, then click **Explain** on a result (or search again). Don't click
  repeatedly; each click uses tokens.
- **Slow first search**: that is the warm-up; it is fast afterwards.
- **No summaries at all / setup message**: check `.env` (key and model) and restart Streamlit.
- **Fallback**: `reports/failure_examples.md` (case 9) and `reports/metrics_summary.md` (provisional metrics)
  tell the before/after and results story without the live app.
