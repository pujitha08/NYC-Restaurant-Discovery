"""NYC Restaurant Discovery: a thin Streamlit UI over app/pipeline.py.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit puts app/ on sys.path, not the repo root, so add it for `app`, `src` and `eval` imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import chips, compare, demo_cache, pipeline  # noqa: E402
from src.config import STANCE_RERANK  # noqa: E402
from app.highlight import highlight_html, highlight_terms, html_text, md  # noqa: E402,F401 (md is used by the tests)

TAGLINE = "Ask in plain English. Get NYC restaurants backed by review evidence."
# The slim note is always shown; the full-strength caveat only above allergy results.
SLIM_NOTE = f"Reviews are historical (Google, through {pipeline.DATA_THROUGH}) and anecdotal, not safety guarantees."
CAVEAT = (f"Reviews are historical (Google reviews through {pipeline.DATA_THROUGH}) and are not safety "
          "guarantees. **For allergies, always confirm with the restaurant before you order.**")
# Page styling: system fonts only (no web fonts, no external requests), a narrower reading column inside the wide layout,
# and pill-shaped example buttons. Static text; nothing from users or reviews goes into it.
CSS = """
<style>
html, body, [class*="st-"], button, input, textarea { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
  "Helvetica Neue", Arial, "Noto Sans", sans-serif; }
.block-container { max-width: 1100px; padding-top: 2rem; }
.tagline { font-size: 1.15rem; color: #1C1917; margin: -0.5rem 0 0.25rem 0; }
[class*="st-key-example_"] button { border-radius: 999px; padding: 0.15rem 0.9rem; border: 1px solid #C2410C;
  color: #9A3412; background: #FFFFFF; font-weight: 600; }
[class*="st-key-example_"] button:hover { background: #FAF3EA; border-color: #9A3412; }
.claim { border-left: 6px solid #475569; background: #F8FAFC; padding: 0.55rem 0.9rem; margin: 0.6rem 0;
  border-radius: 0 8px 8px 0; }
.claim-warning { border-left-color: #B91C1C; background: #FEF2F2; }
.claim-positive { border-left-color: #15803D; background: #F0FDF4; }
.claim-label { font-weight: 700; font-size: 0.85rem; letter-spacing: 0.02em; text-transform: uppercase; color: #334155; }
.claim-warning .claim-label { color: #991B1B; }
.claim-positive .claim-label { color: #14532D; }
.claim-text { color: #1C1917; font-size: 1rem; margin: 0.15rem 0 0.35rem 0; }
.quote { margin: 0.35rem 0 0.1rem 0; padding: 0.1rem 0 0.1rem 0.75rem; border-left: 2px solid #A8A29E;
  color: #1C1917; font-size: 0.95rem; }
.quote-date { display: block; color: #44403C; font-size: 0.8rem; margin-top: 0.1rem; }
</style>
"""


# One-click examples: the four DEMO.md queries (query and borough from app/demo_cache.py), with a short button label.
EXAMPLE_LABELS = ["Vegan group", "Peanut allergy", "Celiac safe", "Halal Chinese"]
EXAMPLES = [(label, query, borough) for label, (query, borough, _) in zip(EXAMPLE_LABELS, demo_cache.DEMO_QUERIES)]


def pick_example(query, borough):
    """Button callback: fill the form and ask main() to run the search on the next run."""
    st.session_state.update(query=query, borough=borough, run_example=True)


@st.cache_resource(show_spinner="Loading the search index…")
def get_retriever(name):
    return pipeline.load_retriever(name)


@st.cache_resource
def get_llm():
    return pipeline.make_llm_client()  # (client, model, setup_error); loads .env


# Short status badges for the collapsed one-line header of each result.
STATUS_BADGE = {
    pipeline.STATUS_SUPPORTED: "✓ Supported",
    pipeline.STATUS_MIXED: "◐ Mixed evidence",
    pipeline.STATUS_PROBLEMS: f"⚠ {pipeline.STATUS_PROBLEMS}",
    pipeline.STATUS_INSUFFICIENT: "○ Insufficient evidence",
    pipeline.STATUS_NEUTRAL: f"ℹ {pipeline.STATUS_NEUTRAL}",
}
BADGE_LEGEND = [
    (STATUS_BADGE[pipeline.STATUS_SUPPORTED], "the reviews shown back up your request (never shown for allergy requests)"),
    (STATUS_BADGE[pipeline.STATUS_MIXED], "some reviews support your request and others contradict it"),
    (STATUS_BADGE[pipeline.STATUS_PROBLEMS], "reviewers warn about this need; read the warnings"),
    (STATUS_BADGE[pipeline.STATUS_INSUFFICIENT], "the reviews shown don't establish it either way"),
    (STATUS_BADGE[pipeline.STATUS_NEUTRAL], "used for allergy requests: reviews mention the topic, so read each claim"),
    ("… Not summarized yet", "open the result and click Explain"),
    ("Low evidence", "fewer than 5 usable reviews for this restaurant"),
]
PIPELINE_STAGES = [
    ("Understand", "a language model turns your request into diets, allergens and group needs."),
    ("Retrieve", "your words are matched to review passages by meaning (sentence embeddings); restaurants are "
                 "ranked by their best matching passages."),
    ("Re-rank (optional)", "the *stance* method reads the evidence sentences and moves restaurants whose "
                           "reviews warn down and those that accommodate up."),
    ("Summarize", "the model summarizes what the retrieved reviews say, citing each review."),
]
ANECDOTE_NOTE = ("Review text is anecdotal, and allergy suitability cannot be confirmed from reviews. "
                 "Always ask the restaurant.")
BADGE_NOT_SUMMARIZED = "… Not summarized yet"
BADGE_UNAVAILABLE = "Summary unavailable"
PENDING_OPEN = "pending_open"  # expander key to open on the next run (set after Explain, before st.rerun)


def expander_key(outcome, view):
    """Session-state key of a result's expander. The request id is part of it, so a new search starts collapsed."""
    return f"open_{outcome.request_id}_{view.gmap_id}"


def status_badge(view):
    s = view.summary
    if s is None:
        return BADGE_UNAVAILABLE if view.summary_error else BADGE_NOT_SUMMARIZED
    if s.guarded and s.status == pipeline.STATUS_SUPPORTED:  # defensive: the guard never shows "supported"
        return STATUS_BADGE[pipeline.STATUS_NEUTRAL]
    return STATUS_BADGE.get(s.status, s.status)


def result_header(view):
    rating = f"★ {view.avg_rating:.1f}" if isinstance(view.avg_rating, (int, float)) else "no rating"
    parts = [f"{view.rank}. {md(view.name)} — {view.borough}", rating, status_badge(view)]
    if view.low_evidence:
        parts.append("Low evidence")
    return " · ".join(parts)


def show_stance(view):
    """Evidence bar and the strongest warning sentence, if the ranking labeled this result's sentences."""
    bar = pipeline.stance_bar(view)
    if bar is None:
        return
    st.markdown(f"**Review sentences:** {bar['bar']} {bar['text']}")
    st.caption("🟩 accommodate · 🟥 warns · 🟨 mention · ⬜ not about this. Labels come from a model reading single "
               "sentences; they are not a safety verdict.")
    if bar["warning"]:
        st.warning(f"**Strongest warning:** “{md(bar['warning'])}”")


# Claim labels: icon + plain word + color (never color alone). Allergy-sensitive queries never show "Positive":
# a supporting claim there is only a "Mention", so the guard holds in the card as well as in the status row.
CLAIM_STYLE = {"contradiction": ("warning", "⚠", "Warning"), "support": ("positive", "✓", "Positive"),
               "uncertainty": ("mention", "•", "Mention")}
GUARDED_CLAIM_STYLE = {**CLAIM_STYLE, "support": ("mention", "ℹ", "Mention")}
SHOW_ALL = "Show all evidence"


def quote_html(excerpt, terms):
    """A review excerpt as a dark, readable quote with its date; query terms highlighted, everything escaped first."""
    return (f'<blockquote class="quote">{highlight_html(excerpt.text, terms)}'
            f'<span class="quote-date">Review from {html_text(excerpt.date)}</span></blockquote>')


def claim_html(claim, terms, guarded, max_quotes=None):
    css, icon, word = (GUARDED_CLAIM_STYLE if guarded else CLAIM_STYLE).get(claim.label, CLAIM_STYLE["uncertainty"])
    quotes = "".join(quote_html(e, terms) for e in (claim.excerpts[:max_quotes] if max_quotes else claim.excerpts))
    return (f'<div class="claim claim-{css}"><div class="claim-label">{icon} {word}</div>'
            f'<div class="claim-text">{html_text(claim.text)}</div>{quotes}</div>')


def ordered_claims(summary):
    """Warnings first (stable otherwise), so the claim shown by default is never milder than one that is hidden."""
    return sorted(summary.claims, key=lambda c: c.label != "contradiction")


def show_status_row(view):
    """ONE compact row: the verdict badge plus, for allergy-sensitive queries, the allergy reminder."""
    s = view.summary
    text = f"**{status_badge(view)}**" + (f" · {s.note}" if s.note else "")
    if s.guarded or s.status in (pipeline.STATUS_PROBLEMS, pipeline.STATUS_MIXED):
        st.warning(text)  # allergy-sensitive results always get the cautious style
    elif s.status == pipeline.STATUS_SUPPORTED:
        st.success(text + " · not a safety guarantee")
    else:
        st.info(text)


def show_more_evidence(view, outcome, terms, shown_claims):
    """Everything not shown by default, under a "Show all evidence" expander (inside the card)."""
    s = view.summary
    claims = ordered_claims(s) if s else []
    top_extra = claims[0].excerpts[1:] if claims else []
    others = claims[1:]
    n_more = len(others) + (1 if top_extra else 0) + (1 if view.excerpts else 0) + (1 if s and s.limitations else 0)
    if not n_more:
        return
    label = f"{SHOW_ALL} ({max(len(others), 0)} more claim{'' if len(others) == 1 else 's'} · {len(view.excerpts)} excerpts)"
    with st.expander(label, key=f"all_{outcome.request_id}_{view.gmap_id}"):
        guarded = bool(s and s.guarded)
        if top_extra:
            st.markdown("".join(quote_html(e, terms) for e in top_extra), unsafe_allow_html=True)
        for c in others:
            st.markdown(claim_html(c, terms, guarded), unsafe_allow_html=True)
        for limitation in (s.limitations if s else []):
            st.caption(md(limitation))
        if view.excerpts:
            st.markdown(f"**All retrieved excerpts ({len(view.excerpts)})**")
            st.markdown("".join(quote_html(e, terms) for e in view.excerpts), unsafe_allow_html=True)


def show_summary(view, outcome, terms=()):
    s = view.summary
    show_status_row(view)
    claims = ordered_claims(s)
    if claims:  # the top claim and its best quote; the rest is under "Show all evidence"
        st.markdown(claim_html(claims[0], terms, s.guarded, max_quotes=1), unsafe_allow_html=True)
    show_more_evidence(view, outcome, terms, claims[:1])


def show_excerpts(view, title="Review excerpts", terms=()):
    if not view.excerpts:
        st.caption("No review excerpts were retrieved for this restaurant.")
        return
    st.markdown(f"**{title} ({len(view.excerpts)})**")
    st.markdown("".join(quote_html(e, terms) for e in view.excerpts), unsafe_allow_html=True)


def show_result(view, outcome, raw, client, model):
    key = expander_key(outcome, view)
    terms = highlight_terms(outcome.query, outcome.parsed)
    if key not in st.session_state:
        st.session_state[key] = view.rank == 1  # the top card starts open; Expand/Collapse all and Explain change it
    # Stateful expander (Streamlit >= 1.5x): its open state lives in st.session_state[key], so it survives reruns
    # and Explain / Expand all / Collapse all can set it. on_change="rerun" keeps that state in sync. The restaurant
    # name appears once, in this label; the card body shows rating, review count and address only.
    with st.expander(result_header(view), key=key, on_change="rerun"):
        rating = f"★ {view.avg_rating:.1f}" if isinstance(view.avg_rating, (int, float)) else "no rating"
        st.caption(f"{rating} · {view.num_reviews:,} reviews · {md(view.address)}")
        if view.low_evidence:
            st.caption("⚠ Low evidence: fewer than 5 usable reviews.")
        show_stance(view)
        if view.summary:
            show_summary(view, outcome, terms)
        elif view.summary_error:
            st.warning(f"Summary unavailable: {view.summary_error['message']}")
            show_excerpts(view, terms=terms)
        else:
            if view.summary_skipped and view.summary_skipped != "not requested":
                st.caption(f"No summary: {view.summary_skipped}")
            elif st.button("Explain", key=f"explain_{outcome.request_id}_{view.gmap_id}",
                           help="Summarize what the reviews say about your request"):
                with st.spinner("Summarizing the reviews…"):
                    pipeline.explain(outcome, raw, view.gmap_id, client=client, model=model)
                st.session_state[PENDING_OPEN] = key  # this expander already exists in this run; open it next run
                st.rerun()
            show_excerpts(view, terms=terms)


CHANGE_MARK = {compare.UP: ":green[▲ {n}]", compare.DOWN: ":red[▼ {n}]", compare.SAME: ":gray[＝]",
               compare.NEW: ":blue[● new]"}


def compare_row(view, change=None):
    rating = f"★ {view.avg_rating:.1f}" if isinstance(view.avg_rating, (int, float)) else "no rating"
    mark = CHANGE_MARK[change[0]].format(n=change[1]) + " " if change else ""
    bar = pipeline.stance_bar(view)
    return f"{mark}**{view.rank}.** {md(view.name)} — {view.borough} · {rating}" + (f"  \n{bar['bar']} {bar['text']}" if bar else "")


def show_compare(cmp):
    for notice in cmp.notices:
        st.info(notice)
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


def set_all_expanded(keys, value):
    for key in keys:
        st.session_state[key] = value


def show_request_chips(outcome):
    st.markdown(chips.CSS + chips.chips_html(chips.request_chips(outcome.parsed, outcome.boroughs, len(outcome.results))),
                unsafe_allow_html=True)


def show_parsed(outcome):
    with st.expander("How your request was understood"):
        p = outcome.parsed
        if p is None:
            st.markdown("The request could not be interpreted, so the search used your exact words.")
        else:
            rows = [("Diets", p.dietary_restrictions), ("Allergens", p.allergens), ("Group needs", p.group_needs),
                    ("Preferences (cuisine, dish, venue)", p.preferences)]
            for label, values in rows:
                st.markdown(f"**{label}:** {md(', '.join(values)) if values else '—'}")
        where = " and ".join(outcome.boroughs) if outcome.boroughs else "Manhattan and Brooklyn"
        st.markdown(f"**Borough filter:** {where}")
        st.caption("Diets and allergens guide the summaries; they are not hard filters on the results.")


@st.cache_data
def get_data_stats():
    return pipeline.data_stats()


def show_sidebar():
    with st.sidebar:
        st.header("How it works")
        for n, (name, text) in enumerate(PIPELINE_STAGES, 1):
            st.markdown(f"**{n}. {name}:** {text}")
        stats = get_data_stats()
        parts = []
        if "restaurants" in stats:
            parts.append(f"{stats['restaurants']:,} restaurants in Manhattan and Brooklyn")
        if "reviews" in stats:
            parts.append(f"{stats['reviews']:,} cleaned Google reviews through {pipeline.DATA_THROUGH}")
        if "chunks" in stats:
            parts.append(f"{stats['chunks']:,} review passages in the search index")
        if parts:
            st.markdown("**The data:** " + "; ".join(parts) + ".")
        st.subheader("Reading the badges")
        for badge, meaning in BADGE_LEGEND:
            st.markdown(f"**{badge}**: {meaning}")
        st.markdown("**🟩🟥🟨⬜ bar**: review sentences that accommodate the need, warn about it, mention it, "
                    "or aren't about it.")
        st.info(ANECDOTE_NOTE)


def main():
    st.set_page_config(page_title="NYC Restaurant Discovery", page_icon="🍽️", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("NYC Restaurant Discovery")
    st.markdown(f'<p class="tagline">{TAGLINE}</p>', unsafe_allow_html=True)
    st.caption(SLIM_NOTE)
    show_sidebar()

    client, model, setup_error = get_llm()
    if setup_error:
        st.error(f"{setup_error['message']} Search still works, without query parsing or summaries. "
                 "To enable them: copy `.env.example` to `.env`, set `GROQ_API_KEY` and `GROQ_MODEL`, "
                 "then restart the app.")

    methods = list(pipeline.METHODS)
    with st.form("search", border=False):  # just the search box and button: Enter submits
        box, button = st.columns([6, 1], vertical_alignment="bottom")
        query = box.text_input("What are you looking for?", key="query",
                               placeholder="e.g. nut-free brunch in Brooklyn for a group with a vegan friend")
        submitted = button.form_submit_button("Search", type="primary", use_container_width=True)
    for col, (label, example, example_borough) in zip(st.columns(len(EXAMPLES)), EXAMPLES):  # chips under the box
        col.button(label, key=f"example_{label}", help=f"{example} ({example_borough})", use_container_width=True,
                   on_click=pick_example, args=(example, example_borough))
    st.session_state.setdefault("borough", pipeline.BOROUGH_CHOICES[0])
    borough = st.pills("Borough", pipeline.BOROUGH_CHOICES, selection_mode="single", key="borough") \
        or pipeline.BOROUGH_CHOICES[0]  # deselecting the pill means "Both"
    left, right = st.columns(2)
    compare_mode = left.checkbox("Compare rankings", key="compare_mode",
                                 help="Show semantic_capped and semantic_capped_stance side by side for this query "
                                      "(the method selector is ignored; no summaries).")
    demo_mode = right.checkbox("Demo mode (serve saved results for the demo queries)",
                               help="For the presentation: an exact demo query is answered from data/demo_cache "
                                    "without calling the language model. Other queries run live.")
    with st.expander("Advanced settings"):  # collapsed by default
        method = st.selectbox("Retrieval method", methods, index=methods.index(pipeline.DEFAULT_METHOD))
        include_low = st.checkbox("Include restaurants with fewer than 5 reviews")

    if submitted or st.session_state.pop("run_example", False):
        if not query.strip():
            st.warning("Type what you are looking for first.")
        else:
            retriever_name, _ = pipeline.METHODS[method]
            retriever = get_retriever(retriever_name)
            if retriever is None:
                st.error(f"The {retriever_name} index is not built yet. "
                         "Run `python -m src.retrieval.semantic --build` and restart the app.")
            elif compare_mode and (base_retriever := get_retriever(pipeline.METHODS[compare.BASE_METHOD][0])) is None:
                st.error("The semantic index is not built yet. "
                         "Run `python -m src.retrieval.semantic --build` and restart the app.")
            elif compare_mode:
                with st.spinner("Searching and re-ranking (labeling review sentences can take a minute)…"):
                    comparison = compare.run_compare(
                        query, borough, base_retriever=base_retriever,
                        stance_retriever=get_retriever(pipeline.METHODS[compare.STANCE_METHOD][0]), client=client,
                        model=model, setup_error=setup_error, include_low_evidence=include_low, demo_mode=demo_mode)
                st.session_state["last_compare"] = comparison
                st.session_state.pop("last_search", None)
            else:
                with st.spinner("Searching and summarizing the top reviews…"):
                    outcome, raw = demo_cache.search_with_fallback(
                        query, borough, method, demo_mode=demo_mode, retriever=retriever, client=client, model=model,
                        setup_error=setup_error, include_low_evidence=include_low)
                st.session_state["last_search"] = (outcome, raw)
                st.session_state.pop("last_compare", None)

    if "last_compare" in st.session_state:
        comparison = st.session_state["last_compare"]
        if comparison.base.allergy_sensitive:
            st.warning(CAVEAT)
        show_request_chips(comparison.base)
        show_parsed(comparison.base)
        st.subheader("Compare rankings")
        show_compare(comparison)

    elif "last_search" in st.session_state:
        outcome, raw = st.session_state["last_search"]
        if outcome.allergy_sensitive:
            st.warning(CAVEAT)  # full strength only where the need is an allergy
        for notice in outcome.notices:
            st.info(notice)
        show_request_chips(outcome)
        show_parsed(outcome)
        if outcome.results:
            st.subheader(f"{len(outcome.results)} restaurants")
            st.caption(f"Top {pipeline.AUTO_SUMMARIES} are summarized automatically; open a result and use Explain "
                       "for the others. Ranking reflects review relevance, not safety.")
        t = outcome.timings
        if outcome.cached:
            st.caption(f"Method {outcome.method} · {demo_cache.CACHE_LABEL}")
        else:
            st.caption(f"Method {outcome.method} · parse {t.get('parse_s', 0):.1f}s · retrieve {t.get('retrieve_s', 0):.1f}s"
                       f" · summarize {t.get('summarize_s', 0):.1f}s · total {t.get('total_s', 0):.1f}s")
        if outcome.results:
            keys = [expander_key(outcome, v) for v in outcome.results]
            pending = st.session_state.pop(PENDING_OPEN, None)
            if pending in keys:
                st.session_state[pending] = True  # set before the expander is created in this run
            left, right, _ = st.columns([1, 1, 4])
            left.button("Expand all", key="expand_all", on_click=set_all_expanded, args=(keys, True))
            right.button("Collapse all", key="collapse_all", on_click=set_all_expanded, args=(keys, False))
        for view in outcome.results:
            show_result(view, outcome, raw, client, model)


if __name__ == "__main__":  # streamlit runs the script as __main__; importing it (e.g. in tests) renders nothing
    main()
