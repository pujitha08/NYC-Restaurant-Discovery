"""NYC Restaurant Discovery: a thin Streamlit UI over app/pipeline.py.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit puts app/ on sys.path, not the repo root, so add it for `app`, `src` and `eval` imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import compare, demo_cache, pipeline  # noqa: E402
from app.highlight import highlight, highlight_terms, md  # noqa: E402,F401 (md is used by the tests)

CAVEAT = (f"Reviews are historical (Google reviews through {pipeline.DATA_THROUGH}) and are not safety "
          "guarantees. **For allergies, always confirm with the restaurant before you order.**")
STATUS_TEXT = {
    pipeline.STATUS_SUPPORTED: "Evidence: **supported** by reviewers (not a safety guarantee)",
    pipeline.STATUS_MIXED: "Evidence: **mixed** ⚠ some reviews contradict others",
    pipeline.STATUS_INSUFFICIENT: "Evidence: **insufficient**: the reviews shown don't establish this",
    pipeline.STATUS_NEUTRAL: f"Evidence: **{pipeline.STATUS_NEUTRAL}**",
    pipeline.STATUS_PROBLEMS: f"⚠ Evidence: **{pipeline.STATUS_PROBLEMS}**: reviewers warn about this need",
}
CLAIM_MARK = {"support": "✓ Supports", "contradiction": "⚠ **Contradicts**", "uncertainty": "? Unclear"}
# Allergy-sensitive queries: no check marks; a "support" claim is shown as a neutral reviewer report.
GUARDED_CLAIM_MARK = {**CLAIM_MARK, "support": "• Reviewer report"}


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


def quote(excerpt, terms):
    """A review excerpt as a block quote, with the query's requirement terms highlighted (text is escaped first)."""
    st.markdown(f"> {highlight(excerpt.text, terms)}\n>\n> *Review from {excerpt.date}*", unsafe_allow_html=True)


def show_excerpts(view, title="Review excerpts", terms=()):
    if not view.excerpts:
        st.caption("No review excerpts were retrieved for this restaurant.")
        return
    st.markdown(f"**{title} ({len(view.excerpts)})**")  # plain heading: expanders can't be nested
    for e in view.excerpts:
        quote(e, terms)


def show_summary(view, terms=()):
    s = view.summary
    if s.status == pipeline.STATUS_PROBLEMS:
        st.warning(STATUS_TEXT[s.status])  # warning style: reviewers report problems
    else:
        st.markdown(STATUS_TEXT[s.status])
    if s.note:
        st.warning(s.note)
    marks = GUARDED_CLAIM_MARK if s.guarded else CLAIM_MARK
    for c in s.claims:
        st.markdown(f"{marks.get(c.label, c.label)}: {md(c.text)}")
        for e in c.excerpts:
            quote(e, terms)
    for limitation in s.limitations:
        st.caption(md(limitation))


def show_result(view, outcome, raw, client, model):
    key = expander_key(outcome, view)
    terms = highlight_terms(outcome.query, outcome.parsed)
    # Stateful expander (Streamlit >= 1.5x): its open state lives in st.session_state[key], so it survives reruns
    # and Explain / Expand all / Collapse all can set it. on_change="rerun" keeps that state in sync.
    with st.expander(result_header(view), key=key, on_change="rerun"):
        st.caption(f"{view.num_reviews:,} reviews · {md(view.address)}")
        if view.low_evidence:
            st.caption("⚠ Low evidence: fewer than 5 usable reviews.")
        show_stance(view)
        if view.summary:
            show_summary(view, terms)
            show_excerpts(view, "All retrieved excerpts", terms)
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
    st.caption("Same query, same candidates: the right column re-ranks the left column's top 30 by what review "
               "sentences say about the need. Neither is a safety ranking.")


def set_all_expanded(keys, value):
    for key in keys:
        st.session_state[key] = value


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


def main():
    st.set_page_config(page_title="NYC Restaurant Discovery", page_icon="🍽️")
    st.title("NYC Restaurant Discovery")
    st.warning(CAVEAT)

    client, model, setup_error = get_llm()
    if setup_error:
        st.error(f"{setup_error['message']} Search still works, without query parsing or summaries. "
                 "To enable them: copy `.env.example` to `.env`, set `GROQ_API_KEY` and `GROQ_MODEL`, "
                 "then restart the app.")

    methods = list(pipeline.METHODS)
    st.caption("Try an example:")
    for col, (label, example, example_borough) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        col.button(label, key=f"example_{label}", help=f"{example} ({example_borough})", use_container_width=True,
                   on_click=pick_example, args=(example, example_borough))
    with st.form("search"):
        query = st.text_input("What are you looking for?", key="query",
                              placeholder="e.g. nut-free brunch in Brooklyn for a group with a vegan friend")
        borough = st.radio("Borough", pipeline.BOROUGH_CHOICES, horizontal=True, key="borough")
        method = st.selectbox("Retrieval method", methods, index=methods.index(pipeline.DEFAULT_METHOD))
        include_low = st.checkbox("Include restaurants with fewer than 5 reviews")
        compare_mode = st.checkbox("Compare rankings", key="compare_mode",
                                   help="Show semantic_capped and semantic_capped_stance side by side for this query "
                                        "(the method selector is ignored; no summaries).")
        demo_mode = st.checkbox("Demo mode (serve saved results for the demo queries)",
                                help="For the presentation: an exact demo query is answered from data/demo_cache "
                                     "without calling the language model. Other queries run live.")
        submitted = st.form_submit_button("Search")

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
        if comparison.base.cached:
            st.warning(f"📦 **{demo_cache.CACHE_LABEL.capitalize()}** (left column): saved from an earlier live run.")
        show_parsed(comparison.base)
        st.subheader("Compare rankings")
        show_compare(comparison)

    elif "last_search" in st.session_state:
        outcome, raw = st.session_state["last_search"]
        if outcome.cached:
            st.warning(f"📦 **{demo_cache.CACHE_LABEL.capitalize()}**: saved from an earlier live run, not a live search.")
        for notice in outcome.notices:
            st.info(notice)
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
