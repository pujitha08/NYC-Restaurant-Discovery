"""The redesigned page with streamlit.testing.v1.AppTest (fake retriever and fake LLM client: no data, network or Groq)."""

import html as html_module
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ImportError:  # streamlit is optional for the rest of the suite
    AppTest = None

from app import demo_cache, examples, pipeline
from tests.test_app_pipeline import FakeClient, FakeRetriever
from tests.test_compare import FakeStance

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
STATS = {"restaurants": 15447, "reviews": 2156429, "chunks": 605540, "index_built": "2026-11-14"}


class NoEvidenceClient(FakeClient):
    """Summaries find nothing (insufficient evidence)."""

    def create(self, **kwargs):
        if kwargs["response_format"]["json_schema"]["name"] == "evidence_summary":
            self.calls.append("evidence_summary")
            return self.response({"status": "insufficient_evidence", "claims": [], "limitations": []})
        return super().create(**kwargs)


class ProblemsClient(FakeClient):
    """Every summary reports a problem (a contradiction, no support)."""

    def create(self, **kwargs):
        if kwargs["response_format"]["json_schema"]["name"] == "evidence_summary":
            self.calls.append("evidence_summary")
            ids = [s["review_id"] for s in json.loads(kwargs["messages"][1]["content"])["snippets"]]
            return self.response({"status": "evidence_found", "limitations": [], "claims": [
                {"text": "One reviewer reported no vegan dishes.", "review_ids": [ids[1]], "label": "contradiction"}]})
        return super().create(**kwargs)


class ProblemsAllergy(ProblemsClient):
    """Every summary reports a problem, and the parser finds a peanut allergy."""

    def create(self, **kwargs):
        res = super().create(**kwargs)
        if kwargs["response_format"]["json_schema"]["name"] == "parsed_query":
            body = json.loads(res.choices[0].message.content)
            body.update(allergens=["peanut"])
            res.choices[0].message.content = json.dumps(body)
        return res


class CueRetriever(FakeRetriever):
    """Reviews with an allergy cue word (so a contradicting claim passes the allergy problem gate)."""

    def search(self, *a, **kw):
        out = super().search(*a, **kw)
        for r in out:
            for snippet in r.evidence_snippets:
                snippet.text = "I had an allergic reaction to peanuts after eating here."
        return out


class PeanutRetriever(FakeRetriever):
    """Reviews that talk about peanut allergies (so claims pass the allergen gate)."""

    def search(self, *a, **kw):
        out = super().search(*a, **kw)
        for r in out:
            r.evidence_snippets[0].text = "Great peanut allergy care, our group was looked after."
            r.evidence_snippets[1].text = "They told us nothing on the menu was peanut free."
        return out


class FailsPlaceOne(FakeClient):
    """Summaries of Place 1 (reviews 101/102) are rate limited until `recovered` is set; the others are summarized normally."""

    recovered = False

    def create(self, **kwargs):
        if kwargs["response_format"]["json_schema"]["name"] == "evidence_summary" and not self.recovered:
            ids = [x["review_id"] for x in json.loads(kwargs["messages"][1]["content"])["snippets"]]
            if 101 in ids:
                from tests.test_app_pipeline import rate_limit_error
                raise rate_limit_error()
        return super().create(**kwargs)


class ParsedClient(FakeClient):
    """A parser that returns the given fields (e.g. allergens), a summarizer as in FakeClient."""

    def __init__(self, **fields):
        super().__init__(location=None)
        self.fields = fields

    def create(self, **kwargs):
        res = super().create(**kwargs)
        if kwargs["response_format"]["json_schema"]["name"] == "parsed_query":
            body = json.loads(res.choices[0].message.content)
            body.update(self.fields)
            res.choices[0].message.content = json.dumps(body)
        return res


@unittest.skipIf(AppTest is None, "streamlit not installed")
class PageTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        st.cache_data.clear()
        self.addCleanup(st.cache_resource.clear)
        self.addCleanup(st.cache_data.clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache_dir = Path(tmp.name) / "demo_cache"
        for obj, name, value in ((pipeline, "LOG_PATH", Path(tmp.name) / "app_requests.jsonl"),
                                 (demo_cache, "CACHE_DIR", self.cache_dir),
                                 (pipeline, "data_stats", lambda *a, **k: dict(STATS)),
                                 (pipeline, "available_methods", lambda: list(pipeline.METHODS)),
                                 (pipeline, "data_dir_missing", lambda: False)):
            p = patch.object(obj, name, value)
            p.start()
            self.addCleanup(p.stop)

    def start(self, client=None, setup_error=None, retrievers=None, run=True):
        st.cache_resource.clear()  # the page caches its LLM client and retrievers
        client = client if client is not None else FakeClient(location=None)
        llm = (pipeline.UsageRecorder(client), "openai/gpt-oss-20b", None) if not setup_error else (None, None, setup_error)
        self.client = client
        load = lambda name: (retrievers or {}).get(name) or FakeRetriever()
        for target, value in (("load_retriever", load), ("make_llm_client", lambda: llm)):
            p = patch.object(pipeline, target, value)
            p.start()
            self.addCleanup(p.stop)
        at = AppTest.from_file(APP, default_timeout=30)
        if run:
            at.run()
            self.assertFalse(at.exception, at.exception)
        return at

    def search(self, at, query="vegan options for a group"):
        at.text_input[0].input(query)
        self.button(at, "Search").click()
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def click(self, at, label):
        self.button(at, label).click()
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    @staticmethod
    def button(at, label):
        return next(b for b in at.button if b.label == label)

    @staticmethod
    def html(at):
        return " ".join(m.value for m in at.markdown)

    @staticmethod
    def plain(at):
        return html_module.unescape(re.sub(r"<[^>]+>", " ", " ".join(m.value for m in at.markdown)))

    def open_tools(self, at):
        """The tools are always in the page (inside the collapsed expander); nothing to open for AppTest."""
        self.assertEqual([e.label for e in at.expander if e.label == "Evaluation tools"], ["Evaluation tools"])
        return at

    @staticmethod
    def method_box(at):
        return next(s for s in at.selectbox if s.label == "Retrieval method")

    # ---- home screen -------------------------------------------------------------------------------------------
    def test_home_screen_content(self):
        at = self.start()
        html, plain = self.html(at), self.plain(at)
        self.assertIn("NYC Restaurant Discovery", plain)
        self.assertIn("Manhattan + Brooklyn · Google reviews to Sept 2021", plain)
        self.assertIn("Find restaurants whose reviews show they take your food allergy seriously.", plain)
        self.assertIn("Ask in plain English. We read 605,540 review passages from 15,447 restaurants and show what "
                      "reviewers actually said, so you know where to look and where to think twice.", plain)
        self.assertEqual(at.text_input[0].placeholder, "e.g. restaurant careful about peanut allergies")
        self.assertIn("Built for dietary needs and food allergies. It doesn&#x27;t check opening hours, prices or availability.", html)
        self.assertEqual(html.count("Reviews are historical and anecdotal, not safety guarantees. "
                                    "For allergies, always confirm with the restaurant before you order."), 1)

    def test_home_chips_in_two_labeled_rows_and_no_halal_chinese(self):
        at = self.start()
        html = self.html(at)
        self.assertLess(html.index("Allergies"), html.index("Diets and groups"))
        labels = [b.label for b in at.button if b.label != "Search"]
        self.assertEqual(labels, ["Peanut", "Tree nut", "Shellfish", "Sesame", "Dairy-free", "Celiac", "Gluten-free",
                                  "Vegetarian", "Halal", "Kosher", "Vegan + gluten-free group"])
        self.assertNotIn("Halal Chinese", labels)
        self.assertEqual([label for label, _ in examples.ALLERGY_CHIPS],
                         ["Peanut", "Tree nut", "Shellfish", "Sesame", "Dairy-free", "Celiac"])

    def test_where_selector_defaults_to_both_boroughs(self):
        at = self.start()
        pills = at.pills[0]
        self.assertEqual((pills.label, list(pills.options), pills.value), ("Where", ["Both boroughs", "Manhattan", "Brooklyn"], "Both"))

    def test_three_step_strip_only_before_the_first_search(self):
        at = self.start()
        plain = self.plain(at)
        for text in ("UNDERSTAND", "RETRIEVE", "SUMMARIZE", "quotes the review behind every claim"):
            self.assertIn(text, plain)
        self.assertNotIn("UNDERSTAND", self.plain(self.search(at)))

    def test_no_sidebar_legend_or_old_disclaimers(self):
        at = self.search(self.start())
        self.assertEqual(len(at.sidebar.markdown), 0)
        text = self.plain(at) + " ".join(c.value for c in at.caption)
        for old in ("Reading the badges", "Of the top", "Ranking reflects relevance, not safety", "Summary on request",
                    "Supported", "Preferences like opening hours"):
            self.assertNotIn(old, text)

    # ---- evaluation tools ---------------------------------------------------------------------------------------
    def test_a_fresh_session_shows_the_collapsed_expander_and_no_comparison_output(self):
        at = self.start()
        panel = next(e for e in at.expander if e.label == "Evaluation tools")
        self.assertFalse(panel.proto.expanded)  # collapsed by default
        self.assertEqual(len(at.toggle), 0)  # there is no header toggle any more
        self.assertFalse(next(c for c in at.checkbox if c.label == "Compare rankings").value)
        text = self.html(at) + " ".join(c.value for c in at.caption)
        for word in ("semantic_capped_stance", "Same query, same candidates", "Compare rankings</div>"):
            self.assertNotIn(word, text)  # no comparison output
        self.assertNotIn("last_compare", at.session_state)
        # the expander is the first thing under the header, above the headline
        self.assertLess(self.html(at).index("NYC Restaurant Discovery"), self.html(at).index("Find restaurants whose reviews show"))

    def test_the_expander_holds_exactly_compare_method_low_evidence_and_demo_mode(self):
        at = self.start()
        panel = next(e for e in at.expander if e.label == "Evaluation tools")
        self.assertEqual([c.label for c in panel.checkbox], ["Compare rankings", "Include restaurants with fewer than 5 reviews", "Demo mode"])
        self.assertEqual([b.label for b in panel.selectbox], ["Retrieval method"])
        self.assertEqual(len(at.checkbox), 3)  # every checkbox on the page is one of them
        self.assertEqual(len(at.selectbox), 1)
        self.assertTrue(all(not c.value for c in at.checkbox))  # all off
        self.assertIn("semantic_capped and semantic_capped_stance", next(c for c in panel.checkbox if c.label == "Compare rankings").help)
        self.assertIn("saved results only", next(c for c in panel.checkbox if c.label == "Demo mode").help)

    def test_ticking_compare_shows_the_comparison_only_inside_the_tools_and_unticking_removes_it(self):
        at = self.start(retrievers={"semantic_stance": FakeStance()})
        next(c for c in at.checkbox if c.label == "Compare rankings").check().run()
        self.search(at)
        self.assertIn("last_compare", at.session_state)
        next(c for c in at.checkbox if c.label == "Compare rankings").uncheck().run()
        self.assertNotIn("last_compare", at.session_state)
        self.assertNotIn("semantic_capped_stance", self.html(at))

    def test_nothing_stale_survives_a_fresh_load(self):
        first = self.start()
        self.method_box(first).select("chunk_tfidf_capped").run()
        next(c for c in first.checkbox if c.label == "Demo mode").check().run()
        next(c for c in first.checkbox if c.label == "Compare rankings").check().run()
        self.assertEqual(self.method_box(first).value, "chunk_tfidf_capped")
        fresh = self.start()  # a new session: a fresh page load
        self.assertEqual(self.method_box(fresh).value, "semantic_capped")
        self.assertTrue(all(not c.value for c in fresh.checkbox))
        self.assertFalse(next(e for e in fresh.expander if e.label == "Evaluation tools").proto.expanded)

    def test_retrieval_method_defaults_to_semantic_capped_in_every_new_session(self):
        for _ in range(3):
            at = self.open_tools(self.start())
            self.assertEqual(self.method_box(at).value, "semantic_capped")
            self.assertEqual(list(self.method_box(at).options)[1], "chunk_tfidf_capped")  # the option that once showed

    def test_a_stored_choice_that_is_no_longer_offered_falls_back_to_the_default_not_to_the_option_order(self):
        at = self.open_tools(self.start())
        at.session_state["method"] = "chunk_tfidf_capped"
        with patch.object(pipeline, "available_methods", lambda: ["semantic_log", "semantic_capped", "semantic_capped_stance"]):
            at.run()
        self.assertEqual(self.method_box(at).value, "semantic_capped")

    def test_without_semantic_capped_a_semantic_method_is_preferred_over_a_tfidf_baseline(self):
        with patch.object(pipeline, "available_methods", lambda: ["chunk_tfidf_log", "chunk_tfidf_capped", "doc_tfidf", "semantic_log"]):
            at = self.open_tools(self.start())
        self.assertEqual(self.method_box(at).value, "semantic_log")

    def test_the_chosen_method_is_used_for_the_search(self):
        retriever = FakeRetriever()
        at = self.open_tools(self.start(retrievers={"semantic": retriever}))
        self.method_box(at).select("semantic_log").run()
        self.search(at)
        self.assertEqual(retriever.calls[-1]["density_mode"], "log")

    def test_low_evidence_checkbox_applies(self):
        retriever = FakeRetriever()
        at = self.open_tools(self.start(retrievers={"semantic": retriever}))
        next(c for c in at.checkbox if c.label.startswith("Include restaurants")).check().run()
        self.search(at)
        self.assertTrue(retriever.calls[-1]["include_low_evidence"])

    # ---- searching ----------------------------------------------------------------------------------------------
    def test_a_search_replaces_the_home_content_with_the_results_screen(self):
        at = self.search(self.start())
        plain = self.plain(at)
        self.assertIn("We understood:", plain)
        self.assertIn("What reviewers say about vegan options", plain)
        self.assertNotIn("Find restaurants whose reviews show", plain)
        self.assertEqual(at.pills, [])  # the Where selector belongs to the home screen
        self.assertEqual(at.status[0].label, "Done")

    def test_understood_chips_name_the_diets_allergens_and_borough_and_change_goes_home(self):
        at = self.search(self.start(ParsedClient(allergens=["peanut"], dietary_restrictions=["vegan"])))
        html = self.html(at)
        self.assertIn("chip chip-diet\">🥗 vegan", html)
        self.assertIn("chip chip-allergen\">⚠ peanut", html)
        self.assertIn("📍 Manhattan + Brooklyn", html)
        self.click(at, "Change")
        self.assertIn("Find restaurants whose reviews show", self.plain(at))
        self.assertEqual(at.text_input[0].value, "vegan options for a group")  # the request is still in the box

    def test_understood_chip_text_is_escaped(self):
        at = self.search(self.start(ParsedClient(allergens=["pea<i>nut"])))
        html = self.html(at)
        self.assertNotIn("pea<i>nut", html)
        self.assertIn("pea&lt;i&gt;nut", html)

    def test_count_line_helper_line_and_trace_with_the_real_timings(self):
        at = self.search(self.start())
        plain = self.plain(at)
        self.assertRegex(plain, r"4 restaurants checked · 4 mention vegan options · 0 raise concerns · 0 not enough evidence")
        self.assertIn("Stars show how much people liked the food. The columns below show what reviewers said about your need. "
                      "Sorted by how strongly reviews match it.", plain)
        self.assertRegex(plain, r"How this was found: 1 Understood · \d+\.\d s → 2 Searched 605,540 passages · \d+\.\d s → "
                                r"3 Summarized with quotes · \d+\.\d s")

    def test_two_columns_with_icons_and_sublines_for_allergy_and_diet_requests(self):
        at = self.search(self.start(ParsedClient(allergens=["peanut"]), retrievers={"semantic": PeanutRetriever()}), "restaurant careful about peanut allergies")
        html = self.html(at)
        self.assertIn('<div class="t">✓ Reviews mention allergy care</div><div class="s">Worth a closer look. Confirm with the restaurant.</div>', html)
        self.assertIn('<div class="t">⚠ Reviews report problems</div><div class="s">Reviewers describe allergy trouble here. Read before you go.</div>', html)
        self.assertIn("What reviewers say about peanut allergies", html)
        diet = self.search(self.start(), "vegan options")
        self.assertIn("✓ Reviews mention vegan options", self.html(diet))

    def test_card_has_name_borough_rating_one_highlighted_quote_review_pill_and_evidence_expander(self):
        at = self.search(self.start())
        html = self.html(at)
        card = next(m.value for m in at.markdown if 'class="care-card"' in m.value)
        self.assertIn('<p class="card-name">Place 1</p>', card)
        self.assertRegex(card, r'card-meta">(Brooklyn|Manhattan) · ★ \d\.\d<')
        self.assertEqual(card.count("<blockquote"), 1)
        self.assertIn("<mark", card)  # matched words highlighted
        self.assertRegex(card, r'pill-n one">Based on 1 review<')
        self.assertEqual([e.label for e in at.expander].count("See the evidence"), 4)

    def test_card_shows_the_finding_first_then_the_quote_then_the_count(self):
        at = self.search(self.start())
        card = next(m.value for m in at.markdown if 'class="care-card"' in m.value)
        self.assertIn('<p class="finding">One reviewer reported vegan options for a group.</p>', card)  # verbatim, bold by CSS
        self.assertIn('<div class="ai-label">AI summary of the reviews below</div>', card)
        self.assertLess(card.index('class="finding"'), card.index("<blockquote"))
        self.assertLess(card.index("<blockquote"), card.index("Based on 1 review"))
        self.assertEqual([e.label for e in at.expander].count("See the evidence"), 4)

    def test_an_allergy_finding_that_asserts_safety_falls_back_to_the_quote_only_card(self):
        class SaysSafe(ParsedClient):
            def create(self, **kwargs):
                res = super().create(**kwargs)
                if kwargs["response_format"]["json_schema"]["name"] == "evidence_summary":
                    body = json.loads(res.choices[0].message.content)
                    body["claims"][0]["text"] = "One reviewer called the kitchen safe for peanut allergies with no risk."
                    res.choices[0].message.content = json.dumps(body)
                return res
        at = self.search(self.start(SaysSafe(allergens=["peanut"]), retrievers={"semantic": PeanutRetriever()}),
                         "restaurant careful about peanut allergies")
        card = next(m.value for m in at.markdown if 'class="care-card"' in m.value)
        self.assertNotIn('class="finding"', card)
        self.assertNotIn("safe for peanut", card)
        self.assertIn("<blockquote", card)  # the quote card is still there

    def test_the_heading_uses_the_parsed_need_not_the_raw_query(self):
        at = self.search(self.start(ParsedClient(allergens=["peanut"]), retrievers={"semantic": PeanutRetriever()}),
                         "I would really love somewhere that is careful about peanut allergies please")
        html = self.html(at)
        self.assertIn('<h1 class="results-title">What reviewers say about peanut allergies</h1>', html)
        self.assertNotIn("really love", html)

    def test_the_source_label_says_live_search_for_a_live_result(self):
        at = self.search(self.start())
        self.assertIn('<span class="source-tag live">Live search</span>', self.html(at))

    def test_a_stale_borough_in_session_state_is_reset_to_both_boroughs(self):
        at = self.start()
        at.session_state["borough"] = "Queens"
        at.run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.pills[0].value, "Both")
        retriever = FakeRetriever()
        at2 = self.start(retrievers={"semantic": retriever}, run=False)
        at2.session_state["borough"] = "Staten Island"
        at2.run()
        self.search(at2)
        self.assertIsNone(retriever.calls[-1]["boroughs"])  # searched both boroughs, not a bogus filter

    def test_the_change_link_goes_back_to_the_home_screen(self):
        at = self.search(self.start())
        self.assertIn("Change", [b.label for b in at.button])

    def test_the_evidence_view(self):
        at = self.search(self.start())
        body = next(m.value for m in at.markdown if "badge badge-care" in m.value)
        self.assertIn("AI summary of the reviews below", body)
        self.assertIn("✓ Reviews mention vegan options", body)
        self.assertIn("One reviewer reported vegan options for a group.", body)
        self.assertIn("1. 2019-", body)
        self.assertIn("2. 2020-", body)
        self.assertIn("Reviews are anecdotal and may be out of date.", body)
        self.assertEqual(body.count('class="caveat"'), 1)
        self.assertIn('<details class="found"><summary>How this restaurant was found</summary>', body)
        for fact in ("Method: semantic_capped", "Rank 1 of 4", "Diets: vegan"):
            self.assertIn(fact, body)

    def test_allergy_evidence_view_has_the_confirm_with_the_restaurant_caveat(self):
        at = self.search(self.start(ParsedClient(allergens=["peanut"]), retrievers={"semantic": PeanutRetriever()}),
                         "restaurant careful about peanut allergies")
        body = next(m.value for m in at.markdown if "badge badge-care" in m.value)
        self.assertIn("Confirm with the restaurant before you order.", body)

    def test_the_allergy_guard_never_says_supported_or_safe(self):
        at = self.search(self.start(ParsedClient(allergens=["peanut"]), retrievers={"semantic": PeanutRetriever()}), "restaurant careful about peanut allergies")
        text = self.html(at) + " ".join(c.value for c in at.caption)
        self.assertNotRegex(text, r"(?i)supported")
        self.assertNotRegex(text, r"(?i)\bsafe\b")
        self.assertNotIn("claim-positive", text)

    def test_empty_problems_column_message(self):
        at = self.search(self.start(ParsedClient(allergens=["peanut"]), retrievers={"semantic": PeanutRetriever()}), "restaurant careful about peanut allergies")
        self.assertIn("No restaurant in the top results has reviews reporting allergy problems. That is not a safety guarantee.", self.html(at))

    def test_empty_care_column_has_the_symmetric_message_for_allergy_and_diet_requests(self):
        allergy = self.search(self.start(ProblemsAllergy(location=None), retrievers={"semantic": CueRetriever()}),
                              "restaurant careful about peanut allergies")
        html = self.html(allergy)
        self.assertIn("No restaurant in the top results has reviews describing care for peanut allergies. "
                      "That doesn&#x27;t mean none exist; reviews just don&#x27;t say.", html)
        self.assertNotIn("has reviews reporting allergy problems", html)  # the problems column is full
        diet = self.search(self.start(ProblemsClient(location=None)), "vegan options")
        self.assertIn("No restaurant in the top results has reviews describing vegan options. "
                      "That doesn&#x27;t mean none exist; reviews just don&#x27;t say.", self.html(diet))

    def test_a_problems_only_restaurant_is_in_the_right_column_only(self):
        at = self.search(self.start(ProblemsClient(location=None)))
        html = self.html(at)
        self.assertEqual(html.count('class="problems-card"'), 4)
        self.assertEqual(html.count('class="care-card"'), 0)
        self.assertIn("4 restaurants checked · 0 mention vegan options · 4 raise concerns", self.plain(at))
        self.assertIn("⚠ Reviews raise concerns</div>", html)  # a diet search
        self.assertNotIn("Reviews report problems", html)

    def test_a_mixed_restaurant_appears_once_in_the_reviewers_disagree_section_with_a_praised_and_a_warned_half(self):
        at = self.search(self.start(FakeClient(summary="mixed", location=None)))
        html = self.html(at)
        self.assertEqual(html.count('class="mixed-card"'), 4)  # every restaurant is mixed, each shown once
        self.assertEqual(html.count('class="care-card"') + html.count('class="problems-card"'), 0)  # in no column
        self.assertIn('<div class="section-title">Reviewers disagree</div>', html)
        self.assertEqual(html.count('class="half praised"'), 4)
        self.assertEqual(html.count('class="half warned"'), 4)
        self.assertEqual(html.count('<p class="card-name">Place 1</p>'), 1)  # once, not once per column
        card = next(m.value for m in at.markdown if 'class="mixed-card"' in m.value)
        praised, warned = card.split('class="half warned"')
        self.assertIn("✓ Praised", praised)
        self.assertIn("Great", praised)
        self.assertIn("⚠ Raised concerns", warned)  # a diet search: not "Warned"
        self.assertNotIn("Warned", html)
        self.assertIn("nothing on the menu was", warned)
        self.assertIn("others raise concerns about them", html)
        self.assertIn("4 with mixed reviews", self.plain(at))
        self.assertLess(html.index("col-head care"), html.index("Reviewers disagree</div>"))  # below the praise column
        self.assertIn("Reviews that mention care only appear below", html)

    def test_an_allergy_result_keeps_the_warned_wording(self):
        class MixedPeanut(ParsedClient):
            def __init__(self):
                super().__init__(allergens=["peanut"])
                self.summary = "mixed"
        at = self.search(self.start(MixedPeanut(), retrievers={"semantic": CueRetriever()}), "restaurant careful about peanut allergies")
        html = self.html(at)
        self.assertEqual(html.count('class="mixed-card"'), 4)
        self.assertIn("⚠ Warned", html)
        self.assertNotIn("Raised concerns", html)
        self.assertIn("others warn about them", html)
        self.assertIn("Reviews report problems", html)
        self.assertNotIn("Reviews raise concerns", html)
        self.assertIn("4 with mixed reviews", self.plain(at))

    def test_an_allergy_problems_result_keeps_report_problems_in_the_column_and_the_count_line(self):
        at = self.search(self.start(ProblemsAllergy(location=None), retrievers={"semantic": CueRetriever()}),
                         "restaurant careful about peanut allergies")
        html, plain = self.html(at), self.plain(at)
        self.assertIn("✓ Reviews mention allergy care</div>", html)
        self.assertIn("⚠ Reviews report problems</div>", html)
        self.assertIn("4 report problems", plain)
        self.assertNotIn("raise concerns", plain)

    def column_blocks(self, at):
        """How many st.columns blocks the page has (the header and the search box each use two)."""
        return len(at.columns)

    def test_a_diet_result_with_no_concerns_renders_one_full_width_column_and_the_note(self):
        with_concerns = self.search(self.start(ProblemsClient(location=None)), "vegan options")
        two = self.column_blocks(with_concerns)
        at = self.search(self.start(FakeClient(location=None)), "vegan options")  # every summary is praise: no concerns
        html = self.html(at)
        self.assertEqual(self.column_blocks(at), two - 2)  # no st.columns(2) for the results
        self.assertIn("No concerns found in the top results. Reviews are anecdotal.", self.plain(at))
        self.assertNotIn("col-head problems", html)  # no concerns column
        self.assertNotIn("Reviews raise concerns", html)
        self.assertNotIn('class="empty-col"', html)  # and no empty box
        self.assertIn("col-head care", html)
        self.assertEqual(html.count('class="care-card"'), 4)
        self.assertLess(html.index("No concerns found"), html.index("col-head care"))  # the line is above the column

    def test_a_diet_result_with_concerns_keeps_the_two_column_layout(self):
        baseline = self.column_blocks(self.search(self.start(FakeClient(location=None)), "vegan options")) + 2
        at = self.search(self.start(ProblemsClient(location=None)), "vegan options")
        html = self.html(at)
        self.assertEqual(self.column_blocks(at), baseline)
        self.assertIn("col-head care", html)
        self.assertIn("col-head problems", html)
        self.assertNotIn("No concerns found", self.plain(at))
        self.assertEqual(html.count('class="problems-card"'), 4)

    def test_a_diet_result_with_concerns_only_in_the_disagree_section_does_not_claim_there_are_none(self):
        at = self.search(self.start(FakeClient(summary="mixed", location=None)), "vegan options")
        plain = self.plain(at)
        self.assertNotIn("No concerns found in the top results", plain)
        self.assertIn("Concerns only appear in the section below, where reviewers disagree. Reviews are anecdotal.", plain)
        self.assertNotIn("col-head problems", self.html(at))

    def test_an_allergy_result_keeps_both_columns_even_when_one_is_empty(self):
        two = self.column_blocks(self.search(self.start(ProblemsClient(location=None)), "vegan options"))  # a layout with both columns
        # no concerns (care claims only): the concerns column is still drawn, with its message
        care_only = self.search(self.start(ParsedClient(allergens=["peanut"]), retrievers={"semantic": PeanutRetriever()}),
                                "restaurant careful about peanut allergies")
        html = self.html(care_only)
        self.assertEqual(self.column_blocks(care_only), two)
        self.assertIn("col-head problems", html)
        self.assertIn('class="empty-col">No restaurant in the top results has reviews reporting allergy problems. That is not a safety guarantee.', html)
        self.assertNotIn("No concerns found", self.plain(care_only))
        # no care (problems only): the care column is still drawn, with its message
        problems_only = self.search(self.start(ProblemsAllergy(location=None), retrievers={"semantic": CueRetriever()}),
                                    "restaurant careful about peanut allergies")
        html = self.html(problems_only).replace("&#x27;", "'")
        self.assertEqual(self.column_blocks(problems_only), two)
        self.assertIn("col-head care", html)
        self.assertIn('class="empty-col">No restaurant in the top results has reviews describing care for peanut allergies', html)

    def test_no_restaurant_appears_twice_anywhere_on_the_page(self):
        for summary in ("ok", "mixed"):
            at = self.search(self.start(FakeClient(summary=summary, location=None)))
            names = re.findall(r'<p class="card-name">([^<]+)</p>', self.html(at))
            self.assertEqual(len(names), len(set(names)), names)

    def test_restaurants_without_a_summary_are_listed_under_not_enough_evidence_as_not_summarized(self):
        retriever = FakeRetriever(gids=("G1", "G2", "G3", "G4", "G6"))
        for i in range(7, 13):
            pipeline_meta = {"name": f"Place {i}", "borough": "Brooklyn", "address": "", "avg_rating": 4.0,
                             "num_clean_reviews": 50, "low_evidence": False}
            from tests.test_app_pipeline import META
            META[f"G{i}"] = pipeline_meta
        retriever = FakeRetriever(gids=tuple(f"G{i}" for i in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12)))
        at = self.search(self.start(retrievers={"semantic": retriever}))
        more = next(e for e in at.expander if "more restaurants matched, but reviews don't say enough about" in e.label)
        self.assertEqual(more.label, "4 more restaurants matched, but reviews don't say enough about vegan options")
        text = " ".join(m.value for m in more.markdown)
        self.assertEqual(text.count("not summarized"), 4)
        self.assertNotIn("claim", text)
        self.assertIn("6 mention vegan options", self.plain(at))  # the live top 6 were summarized

    def test_no_evidence_state_b(self):
        at = self.search(self.start(NoEvidenceClient(location=None)), "vegan options")
        plain = self.plain(at)
        self.assertIn("Reviews don’t say much about vegan options here yet", plain)
        self.assertIn("almost none of their reviews mention vegan options. That is not a sign they are safe or unsafe. "
                      "It means we have no evidence either way.", plain)
        self.assertNotIn("✓ Reviews mention", plain)
        labels = [b.label for b in at.button]
        self.assertIn("A broader request", labels)
        self.assertIn("Peanut", labels)
        self.assertEqual(len(at.expander) and [e.label for e in at.expander if "more restaurants matched" in e.label][0][:2], "4 ")

    def test_state_b_suggestion_runs_a_new_search(self):
        at = self.search(self.start(NoEvidenceClient(location=None)), "vegan options")
        self.click(at, "A broader request")
        self.assertEqual(at.text_input[0].value, "vegan restaurants")

    def test_summary_unavailable_state_a_is_below_the_columns_and_retry_works(self):
        client = FailsPlaceOne(location=None)
        at = self.search(self.start(client))
        html = self.html(at)
        self.assertEqual(html.count("The AI summary isn&#x27;t available right now. Here are the most relevant reviews we found."), 1)
        self.assertEqual(html.count('class="care-card"'), 3)
        self.assertEqual([b.label for b in at.button].count("Try the summary again"), 1)
        self.assertIn("1 summary unavailable", self.plain(at))
        self.assertIn('<div class="section-title">Summary unavailable</div>', html)
        self.assertGreater(html.index("Summary unavailable</div>"), html.rindex('class="care-card"'))  # below the columns
        self.assertGreater(html.index("Summary unavailable</div>"), html.index("col-head care"))
        self.assertEqual(len(at.error), 0)
        client.recovered = True  # the provider recovered
        self.click(at, "Try the summary again")
        self.assertEqual(self.html(at).count('class="care-card"'), 4)
        self.assertEqual([b.label for b in at.button].count("Try the summary again"), 0)

    def test_when_every_summary_fails_the_page_says_live_search_is_unavailable_and_offers_the_saved_examples(self):
        client = FakeClient(summary="rate_limit", location=None)
        at = self.search(self.start(client))
        html, plain = self.html(at), self.plain(at)
        self.assertIn("Live search is unavailable right now (rate limit). Try one of the saved examples.", plain)
        for hidden in ("mention vegan options", "restaurants checked", 'class="care-card"', "Summary unavailable", "Try the summary again"):
            self.assertNotIn(hidden, html)
        self.assertIn("Peanut", [b.label for b in at.button])  # the example chips
        self.assertEqual(len(at.error), 0)

    def test_when_the_parser_fails_the_page_says_so_instead_of_half_results(self):
        at = self.search(self.start(FakeClient(parser="bad", location=None)))
        plain, html = self.plain(at), self.html(at)
        self.assertIn("Live search is unavailable right now (the language model didn't answer). Try one of the saved examples.", plain)
        self.assertNotIn("Request not interpreted", html)
        self.assertNotIn("restaurants checked", html)
        self.assertNotIn("0 mention", html)
        self.assertIn("Kosher", [b.label for b in at.button])

    def test_request_about_hours_is_state_c_with_no_search_and_no_model_call(self):
        retriever = FakeRetriever()
        at = self.start(retrievers={"semantic": retriever})
        self.search(at, "halal restaurants open late")
        plain = self.plain(at)
        self.assertIn("This app finds dietary and allergy information, not hours or prices", plain)
        self.assertIn("We understood “open late” but can’t check it. Here is what we can answer from reviews: halal restaurants", plain)
        self.assertEqual((self.client.calls, retriever.calls), ([], []))
        self.click(at, "Search the dietary part")
        self.assertEqual(at.text_input[0].value, "halal restaurants")
        self.assertIn("What reviewers say about", self.plain(at))
        self.assertEqual(retriever.calls[-1]["query"], "halal restaurants")

    def test_state_c_without_a_dietary_part_has_no_button(self):
        at = self.search(self.start(), "what time does it open")
        self.assertIn("Try describing a diet or an allergy", self.plain(at))
        self.assertNotIn("Search the dietary part", [b.label for b in at.button])

    def test_a_missing_key_shows_the_setup_message_and_no_half_results(self):
        err = {"category": "configuration", "code": "x", "message": "The language model is not configured."}
        at = self.search(self.start(setup_error=err))
        self.assertIn("not configured", at.error[0].value)
        self.assertIn("Live search is unavailable right now (the language model isn't set up).", self.plain(at))
        self.assertNotIn("restaurants checked", self.html(at))

    def test_empty_query_warns(self):
        at = self.start()
        self.button(at, "Search").click()
        at.run()
        self.assertEqual(at.warning[0].value, "Type what you are looking for first.")

    def test_restaurant_names_and_review_text_are_escaped(self):
        from tests.test_app_pipeline import META
        META["G1"] = {**META["G1"], "name": "<script>x</script> Cafe"}
        self.addCleanup(META.__setitem__, "G1", {**META["G1"], "name": "Place 1"})
        at = self.search(self.start())
        html = self.html(at)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;x&lt;/script&gt; Cafe", html)

    # ---- chips and demo mode ---------------------------------------------------------------------------------------
    def test_a_chip_with_no_saved_result_says_so_and_does_not_go_live_by_itself(self):
        at = self.start()
        self.click(at, "Peanut")
        self.assertEqual(at.text_input[0].value, "restaurant careful about peanut allergies")
        plain = self.plain(at)
        self.assertIn("No saved result for this setting", plain)
        self.assertIn("Both boroughs and the semantic_capped method", plain)
        self.assertEqual(self.client.calls, [])  # nothing was sent to the model
        self.assertIn("Run it live", [b.label for b in at.button])
        self.assertNotIn("We understood:", plain)

    def saved_peanut(self):
        outcome, _ = pipeline.run_search("restaurant careful about peanut allergies", "Both", "semantic_capped", retriever=FakeRetriever(),
                                         client=pipeline.UsageRecorder(FakeClient()), model="m", log_path=None)
        demo_cache.save_outcome(outcome, self.cache_dir)

    def test_a_method_with_no_saved_result_shows_the_miss_message_not_a_silent_live_run(self):
        self.saved_peanut()
        at = self.start()
        self.method_box(at).select("semantic_log").run()
        self.click(at, "Peanut")
        self.assertIn("No saved result for this setting", self.plain(at))
        self.assertIn("semantic_log method", self.plain(at))
        self.assertEqual(self.client.calls, [])
        self.assertIn("Run it live", [b.label for b in at.button])

    def test_a_borough_with_no_saved_result_shows_the_miss_message_not_a_silent_live_run(self):
        self.saved_peanut()
        at = self.start()
        at.pills[0].set_value("Manhattan").run()
        self.click(at, "Peanut")
        self.assertIn("No saved result for this setting", self.plain(at))
        self.assertIn("Manhattan", self.plain(at))
        self.assertEqual(self.client.calls, [])

    def test_a_saved_pool_of_thirty_lists_the_too_little_evidence_restaurants_in_one_collapsed_expander(self):
        from tests.test_demo_build import retriever as many
        outcome, _ = pipeline.run_search("restaurant careful about peanut allergies", "Both", "semantic_capped", retriever=many(30),
                                         client=pipeline.UsageRecorder(FakeClient()), model="m", log_path=None, top_k=30, auto_summaries=3)
        self.assertEqual(len(outcome.results), 30)
        demo_cache.save_outcome(outcome, self.cache_dir)
        at = self.start()
        self.click(at, "Peanut")
        html = self.html(at)
        more = [e for e in at.expander if "matched, but reviews don't say enough about" in e.label]
        self.assertEqual(len(more), 1)  # one expander, not one line per restaurant on the page
        inside = " ".join(m.value for m in more[0].markdown)
        self.assertEqual(inside.count('class="more-line"'), int(more[0].label.split()[0]))
        outside = html.replace(inside, "")
        self.assertEqual(outside.count('class="more-line"'), 0)
        self.assertIn("30 restaurants checked", self.plain(at))
        self.assertFalse(more[0].proto.expanded)  # collapsed

    def test_the_saved_setting_itself_is_served_from_the_cache(self):
        self.saved_peanut()
        at = self.start()
        self.click(at, "Peanut")
        self.assertNotIn("No saved result", self.plain(at))
        self.assertIn("a saved result from", self.plain(at))
        self.assertEqual(self.client.calls, [])

    def test_run_it_live_runs_the_search(self):
        at = self.start()
        self.click(at, "Peanut")
        self.click(at, "Run it live")
        self.assertEqual(self.client.calls[0], "parsed_query")
        self.assertIn("We understood:", self.plain(at))
        self.assertNotIn("No saved result", self.plain(at))

    def test_without_a_model_the_miss_offers_no_live_button(self):
        err = {"category": "configuration", "code": "x", "message": "The language model is not configured."}
        at = self.start(setup_error=err)
        self.click(at, "Kosher")
        self.assertIn("No saved result for this setting", self.plain(at))
        self.assertNotIn("Run it live", [b.label for b in at.button])
        self.assertIn("Live search isn't available right now.", " ".join(c.value for c in at.caption))

    def test_a_chip_is_served_from_the_saved_result_without_a_model_call(self):
        query = "restaurant careful about peanut allergies"
        outcome, _ = pipeline.run_search(query, "Both", "semantic_capped", retriever=FakeRetriever(),
                                         client=pipeline.UsageRecorder(FakeClient()), model="m", log_path=None)
        demo_cache.save_outcome(outcome, self.cache_dir)
        client = FakeClient(location=None)
        at = self.start(client)
        self.click(at, "Peanut")
        self.assertEqual(client.calls, [])
        self.assertIn("a saved result from", self.plain(at))
        self.assertIn("(no live search was run)", self.plain(at))

    def test_demo_mode_without_a_saved_result_says_so_instead_of_silently_going_live(self):
        at = self.open_tools(self.start())
        next(c for c in at.checkbox if c.label == "Demo mode").check().run()
        self.search(at, "tacos for a vegan")
        self.assertEqual(self.client.calls, [])
        self.assertIn("No saved result for this setting", self.plain(at))
        self.assertNotIn("notice-cache", self.html(at))
        self.click(at, "Run it live")
        self.assertEqual(self.client.calls[0], "parsed_query")  # the user chose to go live
        self.assertEqual(len(at.info), 0)

    def test_demo_mode_serves_a_saved_result(self):
        outcome, _ = pipeline.run_search("vegan options", "Both", "semantic_capped", retriever=FakeRetriever(),
                                         client=pipeline.UsageRecorder(FakeClient()), model="m", log_path=None)
        demo_cache.save_outcome(outcome, self.cache_dir)
        client = FakeClient(location=None)
        at = self.open_tools(self.start(client))
        next(c for c in at.checkbox if c.label == "Demo mode").check().run()
        self.search(at, "vegan options")
        self.assertEqual(client.calls, [])
        self.assertIn("a saved result from", self.plain(at))
        self.assertIn("Saved demo result", self.plain(at))

    # ---- compare ---------------------------------------------------------------------------------------------------
    def test_compare_mode_shows_two_rankings_inside_the_evaluation_tools_section(self):
        at = self.open_tools(self.start(retrievers={"semantic_stance": FakeStance()}))
        next(c for c in at.checkbox if c.label == "Compare rankings").check().run()
        self.search(at)
        panel = next(e for e in at.expander if e.label == "Evaluation tools")
        inside = " ".join(m.value for m in panel.markdown)
        self.assertIn("Compare rankings", inside)
        self.assertIn("semantic_capped_stance", inside)
        outside = " ".join(m.value for m in at.main.markdown if m.value not in inside)
        self.assertNotIn("semantic_capped_stance", outside)


if __name__ == "__main__":
    unittest.main()
