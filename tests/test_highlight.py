"""Highlighting query terms in untrusted review text: escaping comes before any markup is added."""

import re
import unittest

from app.highlight import MARK, escape, highlight, highlight_html, highlight_terms, html_text, md, term_pattern
from src.schemas import ParsedQuery


def marks(html_text):
    return re.findall(r"<mark[^>]*>(.*?)</mark>", html_text)


class TermTests(unittest.TestCase):
    def test_terms_from_query_drop_generic_words(self):
        self.assertEqual(highlight_terms("restaurant careful about peanut allergies"), ["careful", "peanut", "allergies"])
        self.assertEqual(highlight_terms("halal chinese food"), ["halal", "chinese"])

    def test_parsed_terms_are_added_once(self):
        parsed = ParsedQuery(dietary_restrictions=["vegan", "gluten-free"], allergens=["peanuts"], preferences=["pizza"])
        terms = highlight_terms("vegan pizza", parsed)
        self.assertEqual(terms.count("vegan"), 1)
        self.assertIn("gluten", terms)
        self.assertIn("peanuts", terms)

    def test_term_cap_and_empty(self):
        self.assertLessEqual(len(highlight_terms(" ".join(f"word{c}" for c in "abcdefghijklmnopqrstuvwxyz"))), 12)
        self.assertEqual(highlight_terms(""), [])
        self.assertIsNone(term_pattern([]))
        self.assertIsNone(term_pattern(["of", "a"]))


class HighlightTests(unittest.TestCase):
    def test_matches_are_wrapped_and_stems_match_variants(self):
        out = highlight("They were careful about my peanut Allergy and allergies.", ["careful", "peanuts", "allergies"])
        self.assertEqual(marks(out), ["careful", "peanut", "Allergy", "allergies"])

    def test_short_terms_match_whole_words_only(self):
        out = highlight("Nut-free menu, nuts, and nutritious bowls", ["nut"])
        self.assertEqual(marks(out), ["Nut", "nuts"])

    def test_html_in_the_review_is_escaped_not_rendered(self):
        text = '<script>alert(1)</script> <img src=x onerror=alert(2)> peanut <b>bold</b> & "q"'
        out = highlight(text, ["peanut"])
        self.assertNotIn("<script", out)
        self.assertNotIn("<img", out)
        self.assertNotIn("<b>", out)
        self.assertIn("&lt;script&gt;", out)
        self.assertIn("&amp;", out)
        self.assertEqual(marks(out), ["peanut"])
        self.assertEqual(out.count("<"), out.count("<mark") + out.count("</mark>"))  # the only tags are ours

    def test_term_inside_a_tag_is_highlighted_as_text_only(self):
        out = highlight("<peanut>", ["peanut"])
        self.assertTrue(out.startswith("&lt;"))
        self.assertEqual(marks(out), ["peanut"])
        self.assertNotIn("<peanut", out)

    def test_markdown_specials_are_escaped_around_and_inside_marks(self):
        out = highlight("$3 *happy* hour with vegan_ food", ["vegan"])
        self.assertIn(r"\$3 \*happy\* hour", out)
        self.assertNotIn("*happy*", out)
        self.assertEqual(highlight("[link](http://x) #1", []), r"\[link\](http://x) \#1")

    def test_entity_lookalikes_in_the_review_stay_literal(self):
        out = highlight("&lt;b&gt; fish", ["fish"])
        self.assertTrue(out.startswith("&amp;lt;b&amp;gt;"))  # shown as the typed text, not decoded

    def test_whitespace_collapsed_so_a_line_cannot_start_a_block(self):
        self.assertEqual(highlight("great\n\n# Heading\n- item", []), r"great \# Heading - item")

    def test_no_terms_or_empty_text(self):
        self.assertEqual(highlight("plain text", []), "plain text")
        self.assertEqual(highlight("", ["vegan"]), "")
        self.assertEqual(highlight(None, ["vegan"]), "")

    def test_mark_template_has_no_user_data(self):
        self.assertEqual(MARK.count("{}"), 1)
        self.assertEqual(md("a_b"), r"a\_b")
        self.assertEqual(escape("a<b"), r"a&lt;b")


class HtmlVariantTests(unittest.TestCase):
    """highlight_html is for raw HTML blocks: entities only (no Markdown escapes), the same <mark> markup."""

    def test_matches_are_marked_and_everything_else_is_entity_escaped(self):
        out = highlight_html('<script>alert("x")</script> Great Peanut & more', ["peanut"])
        self.assertNotIn("<script", out)
        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;", out)
        self.assertEqual(marks(out), ["Peanut"])
        self.assertIn("&amp; more", out)
        self.assertEqual(out.count("<"), out.count("<mark") + out.count("</mark>"))

    def test_no_markdown_backslashes_inside_a_raw_html_block(self):
        out = highlight_html("$3 *happy* hour_1 [x] #1", ["happy"])
        self.assertNotIn("\\", out)
        self.assertIn("$3 *", out)

    def test_whitespace_collapsed_and_none_safe(self):
        self.assertEqual(highlight_html("a\n\n b", []), "a b")
        self.assertEqual(highlight_html(None, ["x"]), "")
        self.assertEqual(html_text('<b>"&'), "&lt;b&gt;&quot;&amp;")


if __name__ == "__main__":
    unittest.main()
