"""Streamlit smoke tests with streamlit.testing.v1.AppTest (fake retriever and LLM; no data or network)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ImportError:  # streamlit is optional for the rest of the suite
    AppTest = None

from app import demo_cache, pipeline
from tests.test_app_pipeline import FakeClient, FakeRetriever
from tests.test_compare import FakeStance

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@unittest.skipIf(AppTest is None, "streamlit not installed")
class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log_path = Path(tmp.name) / "app_requests.jsonl"
        log = patch.object(pipeline, "LOG_PATH", self.log_path)
        log.start()
        self.addCleanup(log.stop)
        self.addCleanup(st.cache_resource.clear)
        st.cache_data.clear()  # the sidebar's data stats are cached
        self.addCleanup(st.cache_data.clear)

    def start(self, client=FakeClient(location=None), setup_error=None, retrievers=None):
        llm = (pipeline.UsageRecorder(client), "openai/gpt-oss-20b", None) if client else (None, None, setup_error)
        load = lambda name: (retrievers or {}).get(name) or FakeRetriever()
        for target, value in (("load_retriever", load), ("make_llm_client", lambda: llm)):
            p = patch.object(pipeline, target, value)
            p.start()
            self.addCleanup(p.stop)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    @staticmethod
    def button(at, label):
        return next(b for b in at.button if b.label == label)

    def search(self, at, query="vegan options for a group"):
        at.text_input[0].input(query)
        self.button(at, "Search").click()
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def test_initial_page(self):
        at = self.start()
        self.assertEqual(at.title[0].value, "NYC Restaurant Discovery")
        self.assertIn("Ask in plain English. Get NYC restaurants backed by review evidence.", " ".join(m.value for m in at.markdown))
        self.assertEqual(len(at.warning), 0)  # only a slim note on the first screen, no full-strength warning
        self.assertTrue(any("through September 2021" in c.value and "not safety guarantees" in c.value for c in at.caption))
        self.assertEqual(at.selectbox[0].value, pipeline.DEFAULT_METHOD)
        self.assertEqual(at.pills[0].options, ["Both", "Manhattan", "Brooklyn"])
        self.assertEqual(at.pills[0].value, "Both")
        self.assertEqual(len(at.error), 0)

    @staticmethod
    def results(at):
        """Result expanders (labels start with the rank), in order."""
        return [e for e in at.expander if e.label[:1].isdigit()]

    @staticmethod
    def text(block):
        return " ".join(m.value for m in block.markdown)

    def test_top_card_is_open_and_the_rest_collapsed_with_a_header(self):
        at = self.search(self.start())
        results = self.results(at)
        self.assertEqual(len(results), 4)
        self.assertEqual([e.proto.expanded for e in results], [True, False, False, False])
        self.assertRegex(results[0].label, r"^1\. Place 1 — (Brooklyn|Manhattan) · (★ \d\.\d|no rating) · ✓ Supported")
        self.assertIn("Not summarized yet", results[3].label)  # rank 4: not summarized automatically
        self.assertNotIn("✓", results[3].label)
        self.assertEqual(at.subheader[0].value, "4 restaurants")
        self.assertTrue(any(c.value.startswith("Method ") for c in at.caption))  # timings above the list

    def test_guarded_allergy_header_has_no_check_mark(self):
        at = self.search(self.start(), "peanut allergy friendly dinner")
        for e in self.results(at):
            self.assertNotIn("✓", e.label)
            self.assertNotIn("Supported", e.label)
        self.assertIn(f"ℹ {pipeline.STATUS_NEUTRAL}", self.results(at)[0].label)
        first = self.results(at)[0]
        self.assertEqual(len(first.warning), 1)  # badge and allergy reminder share ONE row
        self.assertIn(f"ℹ {pipeline.STATUS_NEUTRAL}", first.warning[0].value)
        self.assertIn(pipeline.ALLERGY_NOTE, first.warning[0].value)
        self.assertNotIn('class="claim claim-positive"', self.text(first))  # an allergy request never shows a "Positive" claim
        self.assertIn("ℹ Mention", self.text(first))

    def test_card_shows_one_status_row_the_top_claim_and_its_best_quote(self):
        at = self.search(self.start())
        first = self.results(at)[0]
        self.assertTrue(first.proto.expanded)
        self.assertFalse(any(e.proto.expanded for e in self.results(at)[1:]))
        self.assertEqual(len(first.success), 1)  # ONE status row (no separate banner and note)
        self.assertIn("✓ Supported", first.success[0].value)
        body = self.text(first)
        self.assertEqual(body.count('class="claim '), 1)  # only the top claim by default
        self.assertIn('claim-positive"><div class="claim-label">✓ Positive</div>', body)
        self.assertIn("One reviewer reported vegan options for a group.", body)
        self.assertIn("Review from 2019-01", body)
        self.assertNotIn("Place 1", body)  # the restaurant name is in the card label only, not repeated in the body
        self.assertRegex(first.caption[0].value, r"^★ \d\.\d · \d+ reviews · 1 Main St")

    def test_claim_html_escapes_model_and_review_text_and_labels_by_icon_text_and_color(self):
        from app.streamlit_app import claim_html, ordered_claims, quote_html
        evil = pipeline.Excerpt(1, "2020-05", '<img src=x onerror=alert(1)> peanut & "nuts"')
        claim = pipeline.Claim("<b>bold</b> claim & more", "contradiction", [evil])
        html = claim_html(claim, ["peanut"], guarded=True)
        self.assertNotIn("<img", html)
        self.assertNotIn("<b>", html)
        self.assertIn("&lt;b&gt;bold&lt;/b&gt; claim &amp; more", html)
        self.assertIn('class="claim claim-warning"', html)
        self.assertIn("⚠ Warning", html)
        self.assertIn(">peanut</mark>", html)
        self.assertIn("Review from 2020-05", html)
        self.assertNotIn("\n\n", html)  # one HTML block, so Markdown does not cut it
        self.assertIn("blockquote", quote_html(evil, []))

    def test_claim_labels_and_the_allergy_guarded_mapping(self):
        from app.streamlit_app import claim_html, ordered_claims
        support, doubt = pipeline.Claim("s", "support", []), pipeline.Claim("u", "uncertainty", [])
        self.assertIn("✓ Positive", claim_html(support, [], guarded=False))
        self.assertIn("claim-positive", claim_html(support, [], guarded=False))
        guarded = claim_html(support, [], guarded=True)  # allergy request: a supporting claim is only a mention
        self.assertIn("ℹ Mention", guarded)
        self.assertNotIn("Positive", guarded)
        self.assertNotIn("✓", guarded)
        self.assertIn("• Mention", claim_html(doubt, [], guarded=False))
        summary = pipeline.Summary("mixed", [support, pipeline.Claim("w", "contradiction", []), doubt], [])
        self.assertEqual([c.label for c in ordered_claims(summary)], ["contradiction", "support", "uncertainty"])

    def test_rendered_quote_keeps_the_spaces_around_highlighted_terms(self):
        from app.streamlit_app import quote_html
        excerpt = pipeline.Excerpt(1, "2020-05", "I have a severe allergy to peanuts and it is not a problem")
        html = quote_html(excerpt, ["allergy", "peanuts"])
        self.assertRegex(html, r"severe <mark[^>]*>allergy</mark> to <mark[^>]*>peanuts</mark> and it is not a problem")
        card = pipeline.Claim("Staff took my  peanut\nallergy seriously", "support", [excerpt])
        from app.streamlit_app import claim_html
        self.assertIn("Staff took my peanut allergy seriously", claim_html(card, ["peanut"], guarded=True))

    def test_show_all_evidence_expander_holds_the_remaining_evidence(self):
        at = self.search(self.start(FakeClient(location=None, summary="mixed")))
        first = self.results(at)[0]
        inner = next(e for e in first.expander if e.label.startswith("Show all evidence"))
        self.assertFalse(inner.proto.expanded)
        self.assertIn("(1 more claim · 2 excerpts)", inner.label)
        # mixed evidence: the warning claim is the top claim and the positive one is under "Show all evidence"
        self.assertEqual(self.text(first).count('class="claim '), 2)  # top claim outside + the other inside the expander
        top_markdown = [m.value for m in first.markdown if 'class="claim ' in m.value]
        self.assertIn("claim-warning", top_markdown[0])
        self.assertIn("Show all evidence", " ".join(e.label for e in first.expander))
        self.assertIn("All retrieved excerpts (2)", self.text(inner))
        self.assertIn("claim-positive", self.text(inner))

    def test_explain_keeps_result_open(self):
        at = self.search(self.start())
        explain = [b for b in at.button if b.label == "Explain"]
        self.assertEqual(len(explain), 1)  # rank 4 only; top 3 summarized automatically
        explain[0].click()
        at.run()
        self.assertFalse(at.exception, at.exception)
        results = self.results(at)
        self.assertEqual([e.proto.expanded for e in results], [True, False, False, True])
        self.assertNotIn("Not summarized yet", results[3].label)
        self.assertIn("One reviewer reported", self.text(results[3]))
        self.assertEqual(len([b for b in at.button if b.label == "Explain"]), 0)
        at.run()  # an unrelated rerun keeps it open
        self.assertTrue(self.results(at)[3].proto.expanded)
        events = [line.split('"event": "')[1].split('"')[0] for line in self.log_path.read_text().splitlines()]
        self.assertEqual(events, ["search", "explain"])  # both logged to the temp file

    def test_expand_all_collapse_all_and_new_search_resets(self):
        at = self.search(self.start())
        next(b for b in at.button if b.label == "Expand all").click()
        at.run()
        self.assertEqual([e.proto.expanded for e in self.results(at)], [True] * 4)
        at.run()
        self.assertEqual([e.proto.expanded for e in self.results(at)], [True] * 4)  # survives reruns
        next(b for b in at.button if b.label == "Collapse all").click()
        at.run()
        self.assertEqual([e.proto.expanded for e in self.results(at)], [False] * 4)
        next(b for b in at.button if b.label == "Expand all").click()
        at.run()
        self.search(at, "vegan brunch")  # a new search starts with only the top card open
        self.assertEqual([e.proto.expanded for e in self.results(at)], [True, False, False, False])

    def test_demo_mode_serves_a_labeled_cached_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcome, _ = pipeline.run_search("vegan options for a group", "Both", "semantic_capped",
                                             retriever=FakeRetriever(), client=pipeline.UsageRecorder(FakeClient()),
                                             model="openai/gpt-oss-20b", log_path=None)
            demo_cache.save_outcome(outcome, tmp)
            real = demo_cache.CACHE_DIR
            with patch.object(demo_cache, "CACHE_DIR", Path(tmp)):
                client = FakeClient()
                at = self.start(client)
                next(c for c in at.checkbox if c.label.startswith("Demo mode")).check()
                self.search(at)
                self.assertEqual(client.calls, [])  # no model call
                notices = [i.value for i in at.info if "cached demo result" in i.value]
                self.assertEqual(len(notices), 1)  # one notice, with the saved date
                self.assertRegex(notices[0], r"saved \d{4}-\d{2}-\d{2}")
                self.assertFalse(any("cached demo result" in w.value.lower() for w in at.warning))
                self.assertTrue(any(c.value.endswith("cached demo result") for c in at.caption))
                self.assertEqual(len([b for b in at.button if b.label == "Explain"]), 0)

    def test_example_buttons_fill_the_form_and_run_the_search(self):
        client = FakeClient(location=None)
        at = self.start(client)
        labels = [b.label for b in at.button if b.label != "Search"]
        self.assertEqual(labels[:4], ["Vegan group", "Peanut allergy", "Celiac safe", "Halal Chinese"])
        self.button(at, "Peanut allergy").click()
        at.run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.text_input[0].value, "restaurant careful about peanut allergies")
        self.assertEqual(at.pills[0].value, "Both")
        self.assertEqual(len(self.results(at)), 4)  # the search ran without pressing Search
        self.assertEqual(client.calls[0], "parsed_query")
        self.button(at, "Halal Chinese").click()
        at.run()
        self.assertEqual((at.text_input[0].value, at.pills[0].value), ("halal chinese food", "Manhattan"))

    def test_compare_mode_shows_two_columns_with_rank_arrows(self):
        client = FakeClient(location=None)
        at = self.start(client, retrievers={"semantic_stance": FakeStance()})
        at.checkbox(key="compare_mode").check()
        self.search(at)
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(self.results(at)), 0)  # no result expanders: two plain columns
        left, right = at.columns[-2], at.columns[-1]
        self.assertIn("semantic_capped", self.text(left))
        self.assertIn("semantic_capped_stance", self.text(right))
        self.assertIn(":green[▲ 3] **1.** Place 4", self.text(right))
        self.assertIn(":red[▼ 3] **4.** Place 1", self.text(right))
        self.assertNotIn("▲", self.text(left))
        self.assertEqual(client.calls, ["parsed_query"])  # one parse, no summaries

    def test_compare_mode_falls_back_to_the_base_ranking_with_a_notice(self):
        from src.llm.groq_client import LLMAPIError
        at = self.start(retrievers={"semantic_stance": FakeStance(fail=LLMAPIError("x", category="rate_limit"))})
        at.checkbox(key="compare_mode").check()
        self.search(at)
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(any("stance ranking is unavailable" in i.value for i in at.info))
        self.assertIn("Place 1", self.text(at.columns[-1]) + self.text(at.columns[-2]))

    def test_normal_search_clears_a_previous_comparison(self):
        at = self.start(retrievers={"semantic_stance": FakeStance()})
        at.checkbox(key="compare_mode").check()
        self.search(at)
        self.assertIn("last_compare", at.session_state)
        at.checkbox(key="compare_mode").uncheck()
        self.search(at)
        self.assertNotIn("last_compare", at.session_state)
        self.assertEqual(len(self.results(at)), 4)

    def test_evidence_bar_in_an_opened_stance_result_keeps_the_allergy_guard(self):
        at = self.start(FakeClient(location=None), retrievers={"semantic_stance": FakeStance()})
        at.selectbox[0].select("semantic_capped_stance")
        self.search(at, "peanut allergy friendly dinner")
        self.assertFalse(at.exception, at.exception)
        outcome, _ = at.session_state["last_search"]
        for v in outcome.results:
            at.session_state[f"open_{outcome.request_id}_{v.gmap_id}"] = True
        at.run()
        first = self.results(at)[0]
        body = self.text(first)
        self.assertIn("2 accommodate / 1 warns / 1 mention", body)
        self.assertIn("🟩🟩🟥🟨", body)
        self.assertTrue(any("Strongest warning" in w.value and "Staff were not careful." in w.value for w in first.warning))
        self.assertIn(pipeline.ALLERGY_NOTE, " ".join(w.value for w in first.warning))  # allergy note unchanged
        for e in self.results(at):
            self.assertNotIn("Supported", e.label)
            self.assertNotIn("✓", e.label)

    def test_excerpts_highlight_query_terms_and_escape_review_html(self):
        class Evil(FakeRetriever):
            def search(self, *a, **kw):
                out = super().search(*a, **kw)
                for r in out:
                    r.evidence_snippets[0].text = "<script>alert(1)</script> Great vegan options & more"
                return out
        at = self.start(retrievers={"semantic": Evil(), "semantic_stance": Evil()})
        at = self.search(at)
        outcome, _ = at.session_state["last_search"]
        at.session_state[f"open_{outcome.request_id}_{outcome.results[0].gmap_id}"] = True
        at.run()
        body = self.text(self.results(at)[0])
        self.assertIn('<mark style=', body)
        self.assertRegex(body, r">vegan</mark>")
        self.assertIn("&lt;script&gt;", body)
        self.assertNotIn("<script", body)

    def test_no_evidence_bar_without_stance_labels(self):
        at = self.search(self.start())
        outcome, _ = at.session_state["last_search"]
        at.session_state[f"open_{outcome.request_id}_{outcome.results[0].gmap_id}"] = True
        at.run()
        self.assertNotIn("accommodate", self.text(self.results(at)[0]))
        self.assertFalse(any("Strongest warning" in w.value for w in at.warning))

    def test_sidebar_explains_stages_data_badges_and_the_anecdote_note(self):
        with patch.object(pipeline, "data_stats", lambda: {"restaurants": 15447, "reviews": 2156429, "chunks": 605540}):
            at = self.start()
        side = " ".join(m.value for m in at.sidebar.markdown) + " " + " ".join(i.value for i in at.sidebar.info)
        for needle in ("1. Understand", "2. Retrieve", "3. Re-rank (optional)", "4. Summarize", "15,447 restaurants",
                       "2,156,429 cleaned Google reviews", "605,540 review passages", "✓ Supported", "◐ Mixed evidence",
                       "⚠ Reviews report problems", "○ Insufficient evidence", "Low evidence",
                       "Review text is anecdotal", "allergy suitability cannot be confirmed from reviews"):
            self.assertIn(needle, side)

    def test_sidebar_without_data_files_still_renders(self):
        with patch.object(pipeline, "data_stats", lambda: {}):
            at = self.start()
        side = " ".join(m.value for m in at.sidebar.markdown)
        self.assertIn("How it works", " ".join(h.value for h in at.sidebar.header))
        self.assertNotIn("The data:", side)

    def test_staged_progress_replaces_the_spinner(self):
        with patch.object(pipeline, "data_stats", lambda: {"chunks": 605540}):
            at = self.search(self.start(FakeClient(location=None)))
        self.assertEqual(len(at.status), 1)
        status = at.status[0]
        self.assertEqual((status.label, status.state), ("Done", "complete"))
        lines = self.text(status)
        for stage in ("Understanding your request: done", "Searching 606K review passages: done", "Reading reviews: done",
                      "Summarizing: done"):
            self.assertIn(stage, lines)
        self.assertLess(lines.index("Understanding"), lines.index("Searching"))
        self.assertLess(lines.index("Reading reviews"), lines.index("Summarizing"))

    def test_rate_limit_finishes_with_a_notice_and_the_results(self):
        at = self.search(self.start(FakeClient(location=None, summary="rate_limit")))
        self.assertEqual(at.status[0].state, "complete")
        self.assertIn("some steps were unavailable", at.status[0].label)
        self.assertTrue(any("Some summaries are unavailable" in i.value for i in at.info))
        self.assertEqual(len(self.results(at)), 4)  # excerpts still shown

    def test_unexpected_error_turns_the_status_red_and_is_not_swallowed(self):
        class Boom(FakeRetriever):
            def search(self, *a, **kw):
                raise RuntimeError("index exploded")
        at = self.start(retrievers={"semantic": Boom()})
        at.text_input[0].input("vegan food")
        self.button(at, "Search").click()
        at.run()
        self.assertEqual(at.status[0].state, "error")
        self.assertTrue(at.exception)

    def test_compare_mode_shows_the_rerank_stage(self):
        at = self.start(FakeClient(location=None), retrievers={"semantic_stance": FakeStance()})
        at.checkbox(key="compare_mode").check()
        self.search(at)
        self.assertEqual(at.status[0].state, "complete")
        self.assertIn("Re-ranking by what the reviews say: done", self.text(at.status[0]))

    def test_demo_mode_shows_the_cache_stage_and_one_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = pipeline.run_search("vegan options for a group", "Both", "semantic_capped", retriever=FakeRetriever(),
                                         client=pipeline.UsageRecorder(FakeClient()), model="openai/gpt-oss-20b", log_path=None)
            demo_cache.save_outcome(out, tmp)
            with patch.object(demo_cache, "CACHE_DIR", Path(tmp)):
                at = self.start(FakeClient())
                next(c for c in at.checkbox if c.label.startswith("Demo mode")).check()
                self.search(at)
        self.assertEqual(at.status[0].label, "Showing a saved demo result")
        self.assertIn("Loading the saved demo result: done", self.text(at.status[0]))
        self.assertEqual(len([i for i in at.info if "cached demo result" in i.value]), 1)

    def test_full_caveat_only_above_allergy_results(self):
        plain = self.search(self.start(), "vegan options for a group")
        self.assertEqual([w.value for w in plain.warning], [])
        allergy = self.search(self.start(), "peanut allergy friendly dinner")
        caveat = [w.value for w in allergy.warning if "always confirm with the restaurant" in w.value]
        self.assertEqual(len(caveat), 1)
        self.assertIn("through September 2021", caveat[0])

    def test_search_box_then_example_chips_then_borough_pills(self):
        at = self.start()
        labels = [b.label for b in at.button]
        self.assertEqual(labels[:5], ["Search", "Vegan group", "Peanut allergy", "Celiac safe", "Halal Chinese"])
        self.assertEqual(at.pills[0].label, "Borough")

    def test_request_chips_appear_above_the_results(self):
        at = self.search(self.start(), "vegan options for a group")
        chip_block = next(m.value for m in at.markdown if 'class="chips"' in m.value)
        self.assertIn("chip-diet", chip_block)
        self.assertIn("vegan", chip_block)
        self.assertIn("Manhattan + Brooklyn", chip_block)
        self.assertIn("4 restaurants", chip_block)

    def test_chips_show_a_named_borough_scope(self):
        at = self.search(self.start(FakeClient(location="Brooklyn")), "vegan options for a group in Brooklyn")
        block = next(m.value for m in at.markdown if 'class="chips"' in m.value)
        self.assertIn("📍 Brooklyn</span>", block)
        self.assertNotIn("Manhattan + Brooklyn", block)

    def test_borough_pill_filters_and_deselecting_means_both(self):
        retriever = FakeRetriever()
        at = self.start(retrievers={"semantic": retriever})
        at.pills[0].set_value("Brooklyn")
        self.search(at)
        self.assertEqual(retriever.calls[-1]["boroughs"], ["Brooklyn"])
        at.pills[0].set_value(None)
        self.search(at)
        self.assertIsNone(retriever.calls[-1]["boroughs"])

    def test_advanced_settings_expander_holds_method_and_low_evidence(self):
        at = self.start()
        advanced = next(e for e in at.expander if e.label == "Advanced settings")
        self.assertFalse(advanced.proto.expanded)
        self.assertEqual([w.label for w in advanced.selectbox], ["Retrieval method"])
        self.assertEqual([w.label for w in advanced.checkbox], ["Include restaurants with fewer than 5 reviews"])
        self.assertEqual((advanced.selectbox[0].value, advanced.checkbox[0].value), (pipeline.DEFAULT_METHOD, False))
        main_checkboxes = [c.label for c in at.checkbox if c not in advanced.checkbox]
        self.assertEqual(main_checkboxes[0], "Compare rankings")
        self.assertTrue(main_checkboxes[1].startswith("Demo mode"))
        self.assertEqual([r for r in self.results(at)], [])  # the expander is not mistaken for a result

    def test_settings_inside_advanced_still_apply(self):
        retriever = FakeRetriever(gids=("G1", "G2", "G3", "G4", "G5", "G6"))
        at = self.start(retrievers={"semantic": retriever})
        next(c for c in at.checkbox if c.label.startswith("Include restaurants")).check()
        self.search(at)
        self.assertTrue(retriever.calls[-1]["include_low_evidence"])

    def test_examples_are_the_demo_queries(self):
        from app.streamlit_app import EXAMPLES
        self.assertEqual([(q, b) for _, q, b in EXAMPLES], [(q, b) for q, b, _ in demo_cache.DEMO_QUERIES[:4]])

    def test_missing_key_shows_setup_message_and_still_searches(self):
        setup = {"category": "configuration", "code": "invalid_api_key_configuration",
                 "message": pipeline.ERROR_MESSAGES["configuration"]}
        at = self.search(self.start(client=None, setup_error=setup))
        self.assertIn("GROQ_API_KEY", at.error[0].value)
        self.assertIn("Search still works", at.error[0].value)
        self.assertTrue(self.results(at)[0].label.startswith("1. Place 1"))
        self.assertIn("Not summarized yet", self.results(at)[0].label)

    def test_empty_query_and_markdown_escaping(self):
        at = self.start()
        self.button(at, "Search").click()
        at.run()
        self.assertIn("Type what you are looking for", " ".join(w.value for w in at.warning))
        from app.streamlit_app import md  # noqa: imported after AppTest so st is configured
        self.assertEqual(md("$3 *happy* hour"), r"\$3 \*happy\* hour")


if __name__ == "__main__":
    unittest.main()
