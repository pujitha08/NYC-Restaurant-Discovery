"""NYC Restaurant Discovery: a thin Streamlit UI over app/pipeline.py.

Find restaurants whose reviews show they take your food allergy seriously, and see the evidence. Reviews are evidence, not a
verdict: an allergy result never says "Supported" or "safe". What each card shows is decided in app/view_model.py (no Streamlit
there); this file draws it.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""

import hashlib
import re
import sys
from pathlib import Path

import streamlit as st

# Streamlit puts app/ on sys.path, not the repo root, so add it for `app`, `src` and `eval` imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import chips, compare, demo_cache, evidence_filter, examples, pipeline  # noqa: E402
from app import view_model as vm  # noqa: E402
from app.highlight import MARK, highlight_terms, html_text, md, squash  # noqa: E402,F401 (md is used by the tests)
from app.styles import CSS, ICON_GUARD, PALETTE, TEXT_ELEMENTS  # noqa: E402,F401 (the CSS tests read these)
from src.config import STANCE_RERANK  # noqa: E402

HEADLINE = "Find restaurants whose reviews show they take your food allergy seriously."
PLACEHOLDER = "e.g. restaurant careful about peanut allergies"
SCOPE_LINE = "Built for dietary needs and food allergies. It doesn't check opening hours, prices or availability."
SCOPE_PILL = "Manhattan + Brooklyn · Google reviews to Sept 2021"
FOOTER = ("Reviews are historical and anecdotal, not safety guarantees. "
          "For allergies, always confirm with the restaurant before you order.")
STEPS = [("1 · UNDERSTAND", "A language model turns your request into allergens, diets and a place."),
         ("2 · RETRIEVE", "Semantic search finds review passages that match your meaning, not just keywords."),
         ("3 · SUMMARIZE", "The model summarizes what reviewers said and quotes the review behind every claim.")]
NO_NEED_NOTICE = ("No dietary or allergy need was detected, so these results are ranked by general review match "
                  "and aren't checked against a diet or allergen.")
SUMMARY_UNAVAILABLE = "The AI summary isn't available right now. Here are the most relevant reviews we found."
HIDDEN_NOTICES = ("Some summaries are unavailable",)  # each card says it itself (State A)
BOROUGH_LABEL = {"Both": "Both boroughs"}


@st.cache_resource(show_spinner="Loading the search index…")
def get_retriever(name):
    return pipeline.load_retriever(name)


@st.cache_resource
def get_llm():
    return pipeline.make_llm_client()  # (client, model, setup_error); loads .env


@st.cache_data
def get_data_stats():
    return pipeline.data_stats()


# ---- state helpers ---------------------------------------------------------------------------------------------------
RESULT_KEYS = ("last_search", "last_compare", "last_unsupported", "last_miss")


def request_search(query=None, borough=None, from_chip=False, split=True, force_live=False):
    """Callback for chips, suggestions and "Search the dietary part": fill the form and ask main() to run the search."""
    ss = st.session_state
    if query is not None:
        ss["query"] = query
    if borough is not None:
        ss["borough"] = borough
    ss["pending_search"] = {"from_chip": from_chip, "split": split, "force_live": force_live}


def go_home():
    """"Change": back to the home screen with the request still in the box."""
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)


def valid_borough():
    """The borough choice from session state, reset to Both boroughs when it is missing or not one of the choices (stale state)."""
    ss = st.session_state
    if ss.get("borough") not in pipeline.BOROUGH_CHOICES:
        ss["borough"] = pipeline.BOROUGH_CHOICES[0]
    return ss["borough"]


def default_method(methods):
    """semantic_capped when it is available; otherwise a semantic method, never the first TF-IDF baseline by accident."""
    if pipeline.DEFAULT_METHOD in methods:
        return pipeline.DEFAULT_METHOD
    return next((m for m in methods if m.startswith("semantic")), methods[0] if methods else pipeline.DEFAULT_METHOD)


def ensure_method(methods):
    """A new session starts on the default; a stored choice that is no longer on offer falls back to it too."""
    if st.session_state.get("method") not in methods:
        st.session_state["method"] = default_method(methods)


# ---- small pieces ----------------------------------------------------------------------------------------------------
def card_key(prefix, outcome, view, column=""):
    """A widget key for one card: short and free of the ":" in Google map ids."""
    return f"{prefix}_{column}_{outcome.request_id}_{hashlib.md5(view.gmap_id.encode()).hexdigest()[:10]}"


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def stars(rating):
    return f"★ {rating:.1f}" if isinstance(rating, (int, float)) else "no rating"


def show_header():
    left, right = st.columns([1, 1.6], vertical_alignment="center")
    left.markdown('<div class="wordmark">NYC Restaurant Discovery</div>', unsafe_allow_html=True)
    with right:
        with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap="small"):
            st.markdown(f'<span class="pill">{html_text(SCOPE_PILL)}</span>', unsafe_allow_html=True)


def subline(stats):
    passages, restaurants = stats.get("chunks"), stats.get("restaurants")
    what = (f"We read {passages:,} review passages from {restaurants:,} restaurants"
            if isinstance(passages, int) and isinstance(restaurants, int)
            else "We read the review passages of Manhattan and Brooklyn restaurants")
    return (f"Ask in plain English. {what} and show what reviewers actually said, "
            "so you know where to look and where to think twice.")


def search_form(compact):
    """The search box and button (Enter submits); returns True when the form was submitted."""
    with st.form("search", border=False):
        box, button = st.columns([6, 1], vertical_alignment="bottom")
        box.text_input("What are you looking for?", key="query", placeholder=PLACEHOLDER, label_visibility="collapsed")
        submitted = button.form_submit_button("Search", type="primary", use_container_width=True)
    return submitted


def where_selector():
    valid_borough()
    st.pills("Where", pipeline.BOROUGH_CHOICES, selection_mode="single", key="borough", required=True,
             format_func=lambda b: BOROUGH_LABEL.get(b, b), persist_state="session")  # kept while the results screen shows


def show_chips():
    for label, row in examples.CHIP_ROWS:
        st.markdown(f'<div class="chip-label">{html_text(label)}</div>', unsafe_allow_html=True)
        with st.container(horizontal=True, gap="small"):
            for chip_label, query in row:
                st.button(chip_label, key=f"chip_{slug(chip_label)}", help=query, on_click=request_search,
                          kwargs={"query": query, "from_chip": True}, width="content")


def show_steps():
    cards = "".join(f'<div class="step"><div class="step-n">{html_text(n)}</div><div class="step-t">{html_text(t)}</div></div>'
                    for n, t in STEPS)
    st.markdown(f'<div class="steps">{cards}</div>', unsafe_allow_html=True)


def show_footer():
    st.markdown(f'<div class="footer">{html_text(FOOTER)}</div>', unsafe_allow_html=True)


def evaluation_tools(methods):
    """The "Evaluation tools" expander under the header: Compare rankings, the retrieval method, low-evidence restaurants and
    Demo mode. Closed on a fresh session (its open state is session state, so nothing stale survives a fresh load).

    Returns ((compare, method, include_low, demo), slot): slot is a container inside the expander for the comparison output.
    """
    with st.expander("Evaluation tools", key="eval_tools", on_change="rerun"):
        compare_mode = st.checkbox("Compare rankings", key="compare_mode",
                                   help="Show semantic_capped and semantic_capped_stance side by side for this query "
                                        "(the method selector is ignored; no summaries).")
        ensure_method(methods)
        method = st.selectbox("Retrieval method", methods or [pipeline.DEFAULT_METHOD], key="method")
        include_low = st.checkbox("Include restaurants with fewer than 5 reviews", key="include_low")
        demo_mode = st.checkbox("Demo mode", key="demo_mode",
                                help="Answer from the saved results only (no language model call). A setting with no saved "
                                     "result says so and offers to run it live.")
        slot = st.container()
    return (compare_mode, method, include_low, demo_mode), slot


# ---- staged progress -------------------------------------------------------------------------------------------------
def stage_text():
    """Progress stage -> the line shown while it runs (the passage count comes from the built index when known)."""
    n = get_data_stats().get("chunks")
    return {"understand": "Understanding your request",
            "search": f"Searching {n:,} review passages" if isinstance(n, int) else "Searching review passages",
            "read": "Reading reviews", "summarize": "Summarizing with quotes",
            "rerank": "Re-ranking by what the reviews say", "cache": "Loading the saved result"}


def run_staged(work, final_label, with_detail=False):
    """Run work(progress) inside an st.status that lists the stages as they happen, then collapses to one line.

    work receives progress(stage) (and detail(done, total), the "Reading reviews: 3 of 6" line, when with_detail); errors turn the status red and are re-raised (run_search itself turns LLM errors
    and rate limits into per-card states, so those finish normally). final_label(result) is the closing line.
    """
    text = stage_text()
    with st.status("Starting…", expanded=True) as status:
        seen = []

        def progress(stage):
            if seen:
                st.write(f"{text[seen[-1]]}: done")
            seen.append(stage)
            status.update(label=f"{text[stage]}…")

        def detail(done, total):
            status.update(label=f"Reading reviews: {done} of {total}…")

        try:
            result = work(progress, detail) if with_detail else work(progress)
        except Exception:
            status.update(label="The search failed", state="error", expanded=True)
            raise
        if seen:
            st.write(f"{text[seen[-1]]}: done")
        status.update(label=final_label(result), state="complete", expanded=False)
    return result


def search_final_label(result):
    outcome = result[0]
    if outcome.cached:
        return "Done"
    if any(v.summary_error for v in outcome.results) or outcome.parse_error or outcome.ranking_fallback:
        return "Done (some steps were unavailable)"
    return "Done"


# ---- the results screen ----------------------------------------------------------------------------------------------
def quote_html(excerpt, terms, name):
    """A review excerpt as a readable quote; matched words highlighted (not inside the restaurant's name), all escaped."""
    return f'<blockquote class="quote">{evidence_filter.highlight_outside_name(squash(excerpt.text), terms, name, MARK)}</blockquote>'


def show_notices(outcome):
    for notice in outcome.notices:
        if outcome.cached and notice == outcome.notices[0]:
            continue  # the saved result is named in the "How this was found" line
        if not notice.startswith(HIDDEN_NOTICES):
            st.caption(md(notice))


def understood_html(outcome):
    kinds = {chips.DIET: "chip-diet", chips.ALLERGEN: "chip-allergen"}
    body = "".join(f'<span class="chip {kinds.get(kind, "")}">{chips.ICON[kind]} {html_text(text)}</span>'
                   for kind, text in chips.request_chips(outcome.parsed, outcome.boroughs))
    return f'<div class="understood"><span class="understood-label">We understood:</span>{body}</div>'


def show_understood(outcome):
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(understood_html(outcome), unsafe_allow_html=True)
        st.button("Change", key="change", type="tertiary", on_click=go_home)


def trace_html(outcome):
    if outcome.cached:
        return (f'<div class="trace">How this was found: a saved result from {html_text(outcome.cached_at or "an earlier run")} '
                "(no live search was run)</div>")
    passages = get_data_stats().get("chunks")
    steps = [f"{i} {label}" + (f" · {seconds:.1f} s" if isinstance(seconds, (int, float)) else "")
             for i, (label, seconds) in enumerate(vm.trace_parts(outcome, passages), 1)]
    return f'<div class="trace">How this was found: {html_text(" → ".join(steps))}</div>'


def has_dietary_need(parsed):
    return parsed is not None and bool(parsed.dietary_restrictions or parsed.allergens or parsed.group_needs)


CLAIM_TAG = {"support": "✓ Mentions", "contradiction": "⚠ Warns", "uncertainty": "• Unclear"}
BADGE_CLASS = {vm.CARE: "badge-care", vm.PROBLEMS: "badge-problems", vm.MIXED: "badge-mixed"}


def evidence_html(view, outcome, group):
    """The evidence view of one restaurant: badge, the AI summary, the numbered reviews, one caveat, how it was found."""
    terms = highlight_terms(outcome.query, outcome.parsed)
    parts = [f'<span class="badge {BADGE_CLASS.get(group, "")}">{html_text(vm.badge_text(group, outcome))}</span>']
    if view.summary and view.summary.claims:
        parts.append('<div class="ev-label">AI summary of the reviews below</div>')
        for claim in pipeline.merge_neutral_claims(view.summary.claims):
            parts.append(f'<div class="claim-line"><span class="claim-tag">{CLAIM_TAG.get(claim.label, "•")}</span> '
                         f'{html_text(squash(claim.text))}</div>')
    bar = pipeline.stance_bar(view)
    if bar:
        parts.append(f'<div class="review-meta">Review sentences: {html_text(bar["text"])}</div>')
    parts.append('<div class="ev-label">Reviews</div>')
    for i, e in enumerate(view.excerpts, 1):
        meta = f"{i}. {html_text(e.date)}" + (f" · ★ {e.rating}" if getattr(e, "rating", None) else "")
        parts.append(f'<div class="review"><div class="review-meta">{meta}</div>{quote_html(e, terms, view.name)}</div>')
    if not view.excerpts:
        parts.append('<div class="review-meta">No review excerpts were retrieved for this restaurant.</div>')
    caveat = ("Reviews are anecdotal. Confirm with the restaurant before you order." if outcome.allergy_sensitive
              else "Reviews are anecdotal and may be out of date.")
    parts.append(f'<div class="caveat">{html_text(caveat)}</div>')
    p = outcome.parsed
    found = [f"Method: {outcome.method}", f"Rank {view.rank} of {len(outcome.results)}", f"Score {view.score:.3f}"]
    if p is not None:
        for label, values in (("Diets", p.dietary_restrictions), ("Allergens", p.allergens), ("Group needs", p.group_needs),
                              ("Preferences", p.preferences)):
            if values:
                found.append(f"{label}: {', '.join(values)}")
    else:
        found.append("The request could not be interpreted, so the search used your exact words.")
    items = "".join(f"<li>{html_text(x)}</li>" for x in found)
    parts.append(f'<details class="found"><summary>How this restaurant was found</summary><ul>{items}</ul></details>')
    return "".join(parts)


def quote_block(view, outcome, quote):
    if not quote:
        return '<div class="card-meta">No quote available.</div>'
    terms = highlight_terms(outcome.query, outcome.parsed)
    return (f'<blockquote class="quote">“{evidence_filter.highlight_outside_name(squash(quote.text), terms, view.name, MARK)}”'
            "</blockquote>")


def half_html(view, outcome, column):
    """What one column says about a restaurant: the model's finding (bold, verbatim, when it passes the guard), its label, the
    verbatim quote and the number of reviews behind it. With no usable finding it is the quote alone."""
    finding = vm.finding_for(view, outcome, column)
    quote = vm.quote_for(view, outcome, column)
    lead = (f'<p class="finding">{html_text(finding.text)}</p><div class="ai-label">AI summary of the reviews below</div>'
            if finding else "")
    n = quote.n_reviews if quote else 0
    pill = f'<span class="pill-n {"many" if n >= 2 else "one"}">{html_text(vm.review_pill(n))}</span>' if n else ""
    return lead + quote_block(view, outcome, quote) + pill


def card_head(view, outcome=None):
    return (f'<p class="card-name">{html_text(squash(view.name))}</p>'
            f'<div class="card-meta">{html_text(vm.place_label(view, outcome))} · {html_text(stars(view.avg_rating))}</div>')


def card_html(view, outcome, column):
    css = "care-card" if column == vm.CARE else "problems-card"
    return f'<div class="{css}">{card_head(view, outcome)}{half_html(view, outcome, column)}</div>'


def mixed_html(view, outcome):
    """One card for a restaurant with both kinds of review: a "Praised" half and a "Warned" half."""
    return (f'<div class="mixed-card">{card_head(view, outcome)}<div class="halves">'
            f'<div class="half praised"><div class="half-title">✓ Praised</div>{half_html(view, outcome, vm.CARE)}</div>'
            f'<div class="half warned"><div class="half-title">⚠ {html_text(vm.warned_title(outcome))}</div>{half_html(view, outcome, vm.PROBLEMS)}</div>'
            "</div></div>")


def show_card(outcome, view, column):
    group = vm.group_of(view, outcome)
    with st.container(key=card_key("card", outcome, view, column)):
        st.markdown(card_html(view, outcome, column), unsafe_allow_html=True)
        with st.expander("See the evidence", key=card_key("evidence", outcome, view, column)):
            st.markdown(evidence_html(view, outcome, group), unsafe_allow_html=True)


def show_mixed_card(outcome, view):
    with st.container(key=card_key("card", outcome, view, "mixed")):
        st.markdown(mixed_html(view, outcome), unsafe_allow_html=True)
        with st.expander("See the evidence", key=card_key("evidence", outcome, view, "mixed")):
            st.markdown(evidence_html(view, outcome, vm.MIXED), unsafe_allow_html=True)


def show_unavailable_card(outcome, raw, view, client, model):
    """State A: no care/problem label; the best-matching review sentence and a retry."""
    quote = vm.fallback_quote(view, outcome)
    terms = highlight_terms(outcome.query, outcome.parsed)
    with st.container(key=card_key("card_unavailable", outcome, view)):
        body = (f'<blockquote class="quote">“{evidence_filter.highlight_outside_name(squash(quote.text), terms, view.name, MARK)}”</blockquote>'
                if quote else "")
        st.markdown(f'<p class="card-name">{html_text(squash(view.name))}</p>'
                    f'<div class="card-meta">{html_text(vm.place_label(view, outcome))} · {html_text(stars(view.avg_rating))}</div>'
                    f'<p class="unavailable">{html_text(SUMMARY_UNAVAILABLE)}</p>{body}', unsafe_allow_html=True)
        if raw and st.button("Try the summary again", key=card_key("retry", outcome, view)):
            with st.spinner("Summarizing the reviews…"):
                pipeline.explain(outcome, raw, view.gmap_id, client=client, model=model)
            st.rerun()


def column_head(outcome, column):
    care = column == vm.CARE
    title, sub = (vm.care_title(outcome), vm.care_subline(outcome)) if care else (vm.problems_column_title(outcome), vm.problems_subline(outcome))
    icon = "✓" if care else "⚠"
    css = "care" if care else "problems"
    st.markdown(f'<div class="col-head {css}"><div class="t">{icon} {html_text(title)}</div><div class="s">{html_text(sub)}</div></div>',
                unsafe_allow_html=True)


def show_care_column(outcome, c, has_mixed):
    column_head(outcome, vm.CARE)
    for view in c.care:
        show_card(outcome, view, vm.CARE)
    if not c.care:
        st.markdown(f'<div class="empty-col">{html_text(vm.care_column_note(outcome, has_mixed))}</div>', unsafe_allow_html=True)


def show_concerns_column(outcome, c, has_mixed):
    column_head(outcome, vm.PROBLEMS)
    for view in c.problems:
        show_card(outcome, view, vm.PROBLEMS)
    if not c.problems:
        st.markdown(f'<div class="empty-col">{html_text(vm.problems_column_note(outcome, has_mixed))}</div>', unsafe_allow_html=True)


def show_columns(outcome, c):
    """Two columns, praise and concerns. For a diet, group or general search with no concerns column to show, the praise column
    takes the full width under one line saying so; an allergy search always keeps both columns, empty or not."""
    has_mixed = bool(c.mixed)
    if not vm.is_allergy_request(outcome) and not c.problems:
        st.markdown(f'<div class="helper-line">{html_text(vm.no_concerns_note(has_mixed))}</div>', unsafe_allow_html=True)
        show_care_column(outcome, c, has_mixed)
        return
    left, right = st.columns(2)
    with left:
        show_care_column(outcome, c, has_mixed)
    with right:
        show_concerns_column(outcome, c, has_mixed)


def show_mixed_section(outcome, c):
    """Restaurants with both praise and warnings: once, full width, below the columns."""
    if not c.mixed:
        return
    st.markdown('<div class="section-title">Reviewers disagree</div>'
                f'<div class="section-sub">{html_text(vm.disagree_sub(outcome))}</div>',
                unsafe_allow_html=True)
    for view in c.mixed:
        show_mixed_card(outcome, view)


def show_unavailable_section(outcome, raw, c, client, model):
    """Restaurants whose summary failed: below the columns, never above them."""
    if not c.unavailable:
        return
    st.markdown('<div class="section-title">Summary unavailable</div>', unsafe_allow_html=True)
    for view in c.unavailable:
        show_unavailable_card(outcome, raw, view, client, model)


def show_more(outcome, c, need):
    n = len(c.insufficient)
    if not n:
        return
    label = f"{n} more restaurant{' matched' if n == 1 else 's matched'}, but reviews don't say enough about {need}"
    with st.expander(label, key=f"more_{outcome.request_id}"):
        lines = "".join(
            f'<div class="more-line"><b>{html_text(squash(v.name))}</b> <span>{html_text(vm.place_label(v, outcome))} · {html_text(stars(v.avg_rating))}'
            f'{" · not summarized" if vm.group_of(v, outcome) == vm.NOT_SUMMARIZED else ""}</span></div>' for v in c.insufficient)
        st.markdown(lines, unsafe_allow_html=True)


def show_no_evidence(outcome, need):
    """State B: nothing in the top results says anything about the need."""
    st.markdown(f'<div class="state-title">Reviews don’t say much about {html_text(need)} here yet</div>'
                f'<div class="state-body">We found restaurants that match your area, but almost none of their reviews mention '
                f'{html_text(need)}. That is not a sign they are safe or unsafe. It means we have no evidence either way.</div>',
                unsafe_allow_html=True)
    options = vm.suggestions(outcome)
    with st.container(horizontal=True, gap="small"):
        for i, (label, query, borough) in enumerate(options):
            st.button(label, key=f"chip_suggest_{i}", on_click=request_search,
                      kwargs={"query": query, "borough": borough, "from_chip": False, "split": False}, width="content")


def show_unsupported(state):
    """State C: the request asks for something reviews cannot answer (hours, prices, availability)."""
    query, borough, unsupported = state
    rest = f"Here is what we can answer from reviews: {html_text(unsupported.rest)}" if unsupported.rest else \
        "Try describing a diet or an allergy, for example “restaurant careful about peanut allergies”."
    st.markdown('<div class="state-title">This app finds dietary and allergy information, not hours or prices</div>'
                f'<div class="state-body">We understood “{html_text(unsupported.clause)}” but can’t check it. {rest}</div>',
                unsafe_allow_html=True)
    if unsupported.rest:
        st.button("Search the dietary part", key="search_dietary_part", type="primary", on_click=request_search,
                  kwargs={"query": unsupported.rest, "borough": borough, "from_chip": False, "split": False})


def show_cache_miss(state):
    """Demo mode or a chip with no saved result for this exact request, area and method: say so; going live is the user's call."""
    query, borough, method, live_possible = state
    st.markdown('<div class="state-title">No saved result for this setting</div>'
                f'<div class="state-body">There is no saved result for “{html_text(query)}” with {html_text(BOROUGH_LABEL.get(borough, borough))} '
                f'and the {html_text(method)} method. Saved results match the request, the area and the method exactly.</div>',
                unsafe_allow_html=True)
    if live_possible:
        st.button("Run it live", key="run_live", type="primary", on_click=request_search,
                  kwargs={"query": query, "borough": borough, "from_chip": False, "split": True, "force_live": True})
    else:
        st.caption("Live search isn't available right now. Try one of the saved examples.")
    show_chips()


def show_live_unavailable(reason, wait_seconds=None):
    """The parser, the model setup or every summary failed: one message and the saved examples, no half results."""
    st.markdown(f'<div class="state-title">{html_text(vm.unavailable_message(reason, wait_seconds))}</div>', unsafe_allow_html=True)
    show_chips()


def show_results(outcome, raw, client, model):
    show_notices(outcome)
    reason = vm.live_unavailable(outcome)
    if reason:
        show_live_unavailable(reason, vm.live_wait_seconds(outcome))
        return
    show_understood(outcome)
    st.markdown(f'<h1 class="results-title">{html_text(vm.results_heading(outcome))}</h1>', unsafe_allow_html=True)
    if outcome.parsed is not None and not has_dietary_need(outcome.parsed) and not outcome.allergy_sensitive:
        st.caption(NO_NEED_NOTICE)
    if not outcome.results:
        return
    tag = "saved" if outcome.cached else "live"
    st.markdown(f'<div class="count-line"><span class="source-tag {tag}">{html_text(vm.source_label(outcome))}</span> '
                f'{html_text(vm.count_line(outcome))}</div>'
                '<div class="helper-line">Stars show how much people liked the food. The columns below show what reviewers said '
                'about your need. Sorted by how strongly reviews match it.</div>' + trace_html(outcome), unsafe_allow_html=True)
    need = vm.need_phrase(outcome)
    c = vm.columns(outcome)
    if vm.is_no_evidence(outcome):
        show_no_evidence(outcome, need)
    else:
        show_columns(outcome, c)
        show_mixed_section(outcome, c)
    show_unavailable_section(outcome, raw, c, client, model)
    show_more(outcome, c, need)


# ---- compare (evaluation tools) --------------------------------------------------------------------------------------
CHANGE_MARK = {compare.UP: ":green[▲ {n}]", compare.DOWN: ":red[▼ {n}]", compare.SAME: ":gray[＝]",
               compare.NEW: ":blue[● new]"}


def compare_row(view, change=None):
    mark = CHANGE_MARK[change[0]].format(n=change[1]) + " " if change else ""
    bar = pipeline.stance_bar(view)
    return f"{mark}**{view.rank}.** {md(view.name)} — {view.borough} · {stars(view.avg_rating)}" + (f"  \n{bar['bar']} {bar['text']}" if bar else "")


def show_compare(cmp):
    for notice in cmp.notices:
        if not (cmp.base.cached and notice == cmp.notices[0]):
            st.caption(md(notice))
    left, right = st.columns(2)
    left.markdown(f"**{compare.BASE_METHOD}**")
    for view in cmp.base.results:
        left.markdown(compare_row(view))
    if cmp.stance is None:
        return
    right.markdown(f"**{compare.STANCE_METHOD}**  \n:gray[▲ up · ▼ down · new = not in the left top 10]")
    for view in cmp.stance.results:
        right.markdown(compare_row(view, cmp.changes[view.gmap_id]))
    if cmp.dropped:
        names = {v.gmap_id: v.name for v in cmp.base.results}
        right.caption("Dropped out of the top 10: " + "; ".join(md(names[g]) for g in cmp.dropped))
    st.caption(f"Same query, same candidates: the right column re-ranks the left column's top {STANCE_RERANK['top_n']} by what review "
               "sentences say about the need. Neither is a safety ranking.")


# ---- running a search ------------------------------------------------------------------------------------------------
def run_pending(pending, tools, client, model, setup_error):
    """Run the requested search (one per request) and store what the screen shows in session state."""
    compare_mode, method, include_low, demo_mode = tools
    ss = st.session_state
    query, borough = (ss.get("query") or "").strip(), valid_borough()
    if not query:
        st.warning("Type what you are looking for first.")
        return
    for key in RESULT_KEYS:
        ss.pop(key, None)
    unsupported = vm.split_unsupported(query) if pending.get("split", True) else None
    if unsupported:  # nothing to look up: no search and no model call
        ss["last_unsupported"] = (query, borough, unsupported)
        return
    # A chip or Demo mode answers from the saved results only. A setting with none says so and offers to run it live.
    wants_saved = (demo_mode or pending.get("from_chip", False)) and not pending.get("force_live", False)
    if wants_saved and not compare_mode and demo_cache.load_cached(query, borough, method) is None:
        ss["last_miss"] = (query, borough, method, client is not None)
        return
    retriever_name, _ = pipeline.METHODS[method]
    retriever = get_retriever(retriever_name)
    if retriever is None:
        st.error(f"The {retriever_name} index is not built yet. Run `python -m src.retrieval.semantic --build` and restart the app.")
        return
    if compare_mode:
        base_retriever = get_retriever(pipeline.METHODS[compare.BASE_METHOD][0])
        if base_retriever is None:
            st.error("The semantic index is not built yet. Run `python -m src.retrieval.semantic --build` and restart the app.")
            return
        stance_retriever = get_retriever(pipeline.METHODS[compare.STANCE_METHOD][0])
        ss["last_compare"] = run_staged(
            lambda progress: compare.run_compare(
                query, borough, base_retriever=base_retriever, stance_retriever=stance_retriever, client=client,
                model=model, setup_error=setup_error, include_low_evidence=include_low, demo_mode=demo_mode,
                progress=progress),
            lambda c: "Done" if c.base.cached or c.stance is not None else "Done (the stance ranking was unavailable)")
        return
    outcome, raw = run_staged(
        lambda progress, detail: demo_cache.search_with_fallback(
            query, borough, method, demo_mode=wants_saved, retriever=retriever, client=client, model=model,
            setup_error=setup_error, include_low_evidence=include_low, progress=progress, on_summary_progress=detail),
        search_final_label, with_detail=True)
    ss["last_search"] = (outcome, raw)


def main():
    st.set_page_config(page_title="NYC Restaurant Discovery", page_icon="🍽️", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    if pipeline.data_dir_missing():
        st.error(f"The data folder was not found: `{pipeline.DATA_DIR}`. Download the search data there, or point "
                 "`NYC_DATA_DIR` at the folder that holds `processed/`, then restart the app.")
        return
    ss = st.session_state
    valid_borough()
    show_header()
    tools, eval_slot = evaluation_tools(pipeline.available_methods())  # a method whose index files are missing is not offered
    if not tools[0]:
        ss.pop("last_compare", None)  # a comparison only shows while Compare rankings is ticked, inside Evaluation tools

    client, model, setup_error = get_llm()
    if setup_error:
        st.error(f"{setup_error['message']} Search still works, without query parsing or summaries. "
                 "To enable them: copy `.env.example` to `.env`, set `GROQ_API_KEY` and `GROQ_MODEL`, "
                 "then restart the app.")

    pending = ss.pop("pending_search", None)
    on_results = pending is not None or any(k in ss for k in RESULT_KEYS)
    if not on_results:  # the home screen
        st.markdown(f'<h1 class="hero">{html_text(HEADLINE)}</h1>', unsafe_allow_html=True)
        st.markdown(f'<p class="subline">{html_text(subline(get_data_stats()))}</p>', unsafe_allow_html=True)
    submitted = search_form(compact=on_results)
    if not on_results:
        st.markdown(f'<div class="scope-line">{html_text(SCOPE_LINE)}</div>', unsafe_allow_html=True)
        where_selector()
        show_chips()
    if submitted and pending is None:
        request_search(from_chip=False)
        st.rerun()  # draw the results layout from the start of the next run

    if pending is not None:
        run_pending(pending, tools, client, model, setup_error)
    if "last_compare" in ss and tools[0]:
        with eval_slot:
            comparison = ss["last_compare"]
            show_understood(comparison.base)
            st.markdown('<div class="section-title">Compare rankings</div>', unsafe_allow_html=True)
            show_compare(comparison)
    elif "last_search" in ss:
        outcome, raw = ss["last_search"]
        show_results(outcome, raw, client, model)
    elif "last_unsupported" in ss:
        show_unsupported(ss["last_unsupported"])
    elif "last_miss" in ss:
        show_cache_miss(ss["last_miss"])
    if not on_results:
        show_steps()
    show_footer()


if __name__ == "__main__":  # streamlit runs the script as __main__; importing it (e.g. in tests) renders nothing
    main()
