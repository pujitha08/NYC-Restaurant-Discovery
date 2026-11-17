# Final polish: Phase 0 audit (read-only)

Branch `redesign/ui`, written before any edit of this phase. Every statement cites `file:line` of the code as it was at that point.
Anything that would need a Groq call to confirm is marked UNVERIFIED.

## A. Saved results: JSON shape and usable findings

Shape (`data/demo_cache/4382a730d527e800.json`, Peanut, read with `json.load`; the same shape in all 14 files):
`{"cached_at", "outcome": {...}}`; `outcome` keys: request_id, query, borough_choice, method, parsed, summary_context, boroughs,
results, notices, parse_error, setup_error, timings, llm_calls, model, allergy_sensitive, cached, ranking_fallback, cached_at
(written by `app/demo_cache.py:67-81`, `outcome_to_dict`/`_frozen`).
Per restaurant (`results[]`): rank, gmap_id, name, borough, address, avg_rating, num_reviews, low_evidence, score,
**excerpts** (list of `{review_id, date, text, rating}`), **summary**, summary_error, summary_skipped, stance_counts, stance_sentences.
`summary` = `{status, claims, limitations, guarded, note}` (`app/pipeline.py:119-125`, `Summary`):

1. plain-language finding sentence: `summary.claims[].text` (a one-sentence model claim, e.g. "One reviewer warned that ...").
2. its label: `summary.claims[].label` in `support` / `contradiction` / `uncertainty`.
3. cited reviews: `summary.claims[].excerpts[]` (full excerpt objects with `review_id`, `text`, `rating`); the model's raw ids are
   resolved to the restaurant's own excerpts in `to_summary` (`app/pipeline.py:427-434`, `by_id` lookup), so a cited review
   that is not in that restaurant's evidence is dropped there.
4. an overall per-restaurant summary text: **there is none.** `summary.status` is a badge word ("Reviews report problems",
   "mixed", ...), `limitations` is the fixed disclaimer plus model limitations, `note` is the fixed ALLERGY_NOTE
   (`app/pipeline.py:94`). The claims are the only prose.

Counts over ALL 14 cached files (script over `data/demo_cache/*.json`): 72 restaurants have a summary; **56** have at least one
support/contradiction claim that cites a review present in their own evidence; **16** do not (only uncertainty claims or an
"insufficient evidence" summary). Of the 147 claims in total, 0 have an empty `excerpts` list (every cited id resolved).
Caveats found: in guarded (allergy) results a claim that used safety wording is already replaced by the neutral line
`NEUTRAL_CLAIM` (`app/pipeline.py:82`, applied at `:429`), and a name-only claim by `NAME_ONLY_CLAIM`
(`app/evidence_filter.py:12`); those are not findings and must not be shown as one.

## B. Card rendering today

- Card HTML: `card_html` `app/streamlit_app.py:310-320` (name, borough, stars, ONE quote, "Based on N reviews" pill, tag/also-note
  for mixed). Shown by `show_card` `:323-329` with the "See the evidence" expander (`evidence_html` `:275-307`, which is where the
  model's claim sentences appear, under "AI summary of the reviews below", `:285-291`). The card itself shows NO finding sentence.
- Quote: `vm.quote_for` `app/view_model.py:376-`; count `n_reviews` from `about_the_need` (same file).
- Care vs problem vs mixed: `vm.group_of` `app/view_model.py:124-` (reads `effective_claims` `:98-`, i.e. after the care gate, the allergy
  problem gate, the allergen gate); `vm.columns` `:157-169` puts MIXED in **both** `care` and `problems` (`:161-164`);
  `show_columns` `:357-372` draws each list, so a mixed restaurant is drawn twice (tag at `:312`, note at `:318`).
- State A cards are drawn by `show_unavailable_card` `:332-345`, called from `show_results` `:426-427` **before** the columns.

## C. Cache lookup

- Key: `cache_key` `app/demo_cache.py:52-53` = sha256 of `normalize(query)`, `borough_choice`, `method` (joined by newlines), first 16 hex
  chars. `normalize` `:48-49` collapses whitespace and casefolds. So the key **includes the method and the borough** and the normalized
  query. `load_cached` `:109-125` also re-checks all three after reading.
- Directory: `CACHE_DIR` `:28`.
- Chips: `examples.CHIP_BOROUGH, CHIP_METHOD = "Both", "semantic_capped"` (`app/examples.py:20`); chip clicks pass `from_chip=True`
  (`app/streamlit_app.py:155-156`), `run_pending` calls `search_with_fallback(..., demo_mode=demo_mode or from_chip)` (`:499-505`).
- A typed query equal to a chip's text hits the cache only when Demo mode is on (`demo_cache.py:150-153`) and borough is "Both" and
  method is `semantic_capped`; with Demo mode off it runs live (`:154-159`), and uses the cache only as a fallback when the live run
  failed (`live_failed` `:133-141`, `:155-159`).
- Miss with Demo mode ON: falls through to `pipeline.run_search` live, silently (`demo_cache.py:154`). Miss with Demo mode OFF: live
  too. Nothing tells the user.

## D. Failure paths

- Parser: `pipeline.parse` `app/pipeline.py:258-266` returns `(None, error_info)` on any `LLMError` (429, bad_request, ...), no retry.
  `run_search` then appends the notice at `:579`, sets `summary_context = ParsedQuery(preferences=[query])` `:582`, marks the request
  allergy-sensitive (`is_allergy_sensitive` `:389`, `parsed is None` -> True, called at `:583`), and **keeps going**: retrieval and
  summaries still run.
- "Request not interpreted: searched your exact words": produced in `app/chips.py:38` (`request_chips`, `parsed is None`), shown by
  `understood_html` `app/streamlit_app.py:244-248`. After it, columns render as usual (`show_results` `:413-432`).
- Every summary fails (e.g. the same 429): all views get `summary_error`; `run_search` appends "Some summaries are unavailable" (`:615`),
  hidden in the UI (`HIDDEN_NOTICES` `streamlit_app.py:39`, filter `:240`); `vm.columns` puts every view in `unavailable`, so the page
  shows the count line "N restaurants checked · 0 mention ... · 0 report problems · 0 not enough evidence · N summary unavailable"
  (`view_model.py:243-255`), empty-column messages, and N State A cards: the "half result" the brief wants removed.
- Some fail: State A cards above the columns (`:426-431`), the rest normal.
- A saved result is swapped in only if the parser failed with rate_limit/timeout/network or every summary did (`live_failed`
  `demo_cache.py:133-141`) AND a saved entry exists for that exact key (`:155-159`).

## E. Defaults

- Evaluation tools: toggle `st.toggle("Evaluation tools", key="eval_toggle")` `streamlit_app.py:123`, default False
  (`sync_evaluation_tools` `:78-87` reads `ss.get("eval_toggle", False)`); expander created collapsed `:171`. The four tools are **rendered
  inside the collapsed expander on every run** (`:172-180`), so they are in the page even when closed.
- Compare rankings default False (`:172`); method: `ensure_method` `:97-100` + selectbox key `method` `:176`; fewer-than-5 default False `:177`;
  Demo mode default False `:178`.
- "Where": `where_selector` `:144-147`, `setdefault("borough", "Both")` then `st.pills(..., key="borough", persist_state="session")`; it is
  drawn on the home screen only (`:533-535`). A stale/invalid value in `session_state["borough"]` is not validated anywhere
  (`run_pending` `:471` only does `ss.get("borough") or "Both"`), and `persist_state="session"` keeps a past choice.
- Results heading: `show_results` `:417`, text from `vm.need_phrase` `view_model.py:195-205`, which falls back to the raw query
  (`:205`) when no allergen or diet was parsed; size by `.results-title` in `app/styles.py`.
- Count line: `vm.count_line` `view_model.py:243-255`, drawn at `streamlit_app.py:422`.
- "Change": `show_understood` `streamlit_app.py:251-254` (`st.button("Change", type="tertiary")` beside the chips in a horizontal container).

## F. Groq client

- Per-call timeout 30 s, SDK retries off: `OpenAI(..., timeout=30.0, max_retries=0)` `src/llm/groq_client.py:102`.
- App-level: a rate-limited summary call is retried with 2/4/8 s backoff (`BACKOFF_SECONDS` `app/pipeline.py:69`, `summarize_with_backoff`
  `:497-`); a bad_request/unusable answer is retried once with shortened evidence (`RETRY_SHORT_CATEGORIES` `:439`, `_summarize_once` `:462`);
  two workers (`SUMMARY_WORKERS` `:68`). The parser has no retry (`:258-266`). Worst case for a failing search is therefore well over 15 s.
- Model fallback: **none exists.** `resolve_model` `groq_client.py:111-116` reads only `GROQ_MODEL`; a search for fallback/`GROQ_FALLBACK`
  in `app/` and `src/` finds nothing.

## G. Proposal alignment (before this phase)

| # | Proposal item | Status | Evidence |
|---|---|---|---|
| 1 | NL query -> LLM parser (location, diets, allergens, group needs, preferences) | PRESENT | `src/llm/query_parser.py` (`LIST_FIELDS`, schema), called at `app/pipeline.py:576` |
| 2 | Retrieval with BOTH TF-IDF and semantic embeddings | PARTIAL | all six systems exist (`eval/make_pool.py:126-133`, `app/pipeline.py:55`) but the app default is semantic only, TF-IDF is reachable only from the method dropdown in Evaluation tools (`streamlit_app.py:176`) and only if its index files exist (`pipeline.py:723`) |
| 3 | LLM summary explaining in plain language why a restaurant may or may not suit, plus excerpts | PARTIAL | the model's claim sentences exist but appear only inside "See the evidence" (`streamlit_app.py:285-291`); the card shows only a quote (`:310-320`) |
| 4 | Allergy queries report past accommodation evidence, never medical safety | PRESENT | `SAFETY_WORDS`/`has_safety_wording` `pipeline.py:80-90`, `ALLERGY_NOTE` `:94`, gates `view_model.py:60-122`, footer `streamlit_app.py:31-32` |
| 5 | Examples where the system fails or retrieves conflicting reviews | PARTIAL | `reports/failure_examples.md` exists; in the app conflicts show as the "Reviewers disagree" tag (`streamlit_app.py:312`), drawn twice (see B) |
| 6 | Borough selection Manhattan/Brooklyn | PARTIAL | `where_selector` `streamlit_app.py:144-147`, home screen only (`:533-535`); not stale-proof (E) |
| 7 | Historical-data disclaimer | PRESENT | pill `SCOPE_PILL` `streamlit_app.py:30`, footer `FOOTER` `:31-32`, drawn at `:554-` |
