"""Coverage row: which parsed needs the shown excerpts mention (same matcher as the highlighting), wording, escaping."""

import unittest

from app import coverage
from app.pipeline import Excerpt
from src.schemas import ParsedQuery


def ex(*texts):
    return [Excerpt(i, "2020-01", t) for i, t in enumerate(texts, 1)]


class MatcherTests(unittest.TestCase):
    def test_terms_drop_stopwords_like_the_highlighting(self):
        self.assertEqual(coverage.need_terms("gluten-free"), ["gluten"])
        self.assertEqual(coverage.need_terms("vegan and meat lovers"), ["vegan", "meat", "lovers"])
        self.assertEqual(coverage.need_terms("group dinner"), [])  # nothing matchable

    def test_counts_excerpts_by_whole_word_or_prefix(self):
        e = ex("Great VEGAN options", "vegans welcome", "nothing here", "peanuts everywhere")
        self.assertEqual(coverage.count_excerpts("vegan", e), 2)
        self.assertEqual(coverage.count_excerpts("peanut", e), 1)
        self.assertEqual(coverage.count_excerpts("nut", e), 0)  # a short term is a whole word, not a prefix of "peanuts"
        self.assertIsNone(coverage.count_excerpts("group dinner", e))

    def test_a_multiword_need_needs_all_its_words_in_one_excerpt(self):
        e = ex("vegan menu only", "lots of meat lovers here", "vegan friends and meat lovers both happy")
        self.assertEqual(coverage.count_excerpts("vegan and meat lovers", e), 1)


class WordingTests(unittest.TestCase):
    parsed = ParsedQuery(dietary_restrictions=["vegetarian", "halal"], allergens=["peanut"], group_needs=["kids"])
    excerpts = ex("A vegetarian dish and a vegetarian plate", "vegetarian friendly; kids menu", "peanut sauce on the side")

    def test_one_wording_for_every_need_type(self):
        self.assertEqual(coverage.coverage(self.parsed, self.excerpts),
                         [(coverage.FOUND, "vegetarian: mentioned in 2 excerpts"),
                          (coverage.NONE, "halal: not mentioned in shown excerpts"),
                          (coverage.FOUND, "peanut: mentioned in 1 excerpt"),
                          (coverage.FOUND, "kids: mentioned in 1 excerpt")])

    def test_same_wording_for_a_not_found_allergen(self):
        self.assertEqual(coverage.coverage(ParsedQuery(allergens=["sesame"]), self.excerpts),
                         [(coverage.NONE, "sesame: not mentioned in shown excerpts")])

    def test_no_check_mark_and_no_safety_wording_anywhere(self):
        for text in (t for _, t in coverage.coverage(self.parsed, self.excerpts)):
            self.assertRegex(text, r"^.+: (mentioned in \d+ excerpts?|not mentioned in shown excerpts)$")
            for word in ("✓", "✔", "safe", "free", "guarantee", "not found"):
                self.assertNotIn(word, text.lower())

    def test_nothing_parsed_gives_no_row(self):
        self.assertEqual(coverage.coverage(None, self.excerpts), [])
        self.assertEqual(coverage.coverage(ParsedQuery(preferences=["ramen"]), self.excerpts), [])
        self.assertEqual(coverage.coverage_html([]), "")

    def test_preferences_are_listed_not_checked(self):
        p = ParsedQuery(preferences=["open late", "ramen"], dietary_restrictions=["vegan"])
        self.assertEqual(coverage.preferences_line(p), "Not verified from reviews: open late, ramen")
        self.assertEqual(coverage.preferences_line(ParsedQuery()), "")
        self.assertEqual(coverage.preferences_line(None), "")
        self.assertEqual([t for _, t in coverage.coverage(p, ex("open late ramen vegan"))], ["vegan: mentioned in 1 excerpt"])


class EscapingTests(unittest.TestCase):
    def test_html_is_escaped_and_single_block(self):
        parsed = ParsedQuery(dietary_restrictions=['<script>alert("x")</script> vegan'])
        html = coverage.coverage_html(coverage.coverage(parsed, ex("vegan script alert")))
        self.assertNotIn("<script", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("\n\n", html)


if __name__ == "__main__":
    unittest.main()
