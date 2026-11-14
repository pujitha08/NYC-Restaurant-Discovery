"""Request chips: contents, scope wording and HTML escaping."""

import unittest

from app import chips
from src.schemas import ParsedQuery


class ChipTests(unittest.TestCase):
    def test_diets_allergens_borough_and_count(self):
        parsed = ParsedQuery(dietary_restrictions=["vegan", "gluten-free"], allergens=["peanut"], preferences=["brunch"])
        self.assertEqual(chips.request_chips(parsed, ["Brooklyn"], 10),
                         [(chips.DIET, "vegan"), (chips.DIET, "gluten-free"), (chips.ALLERGEN, "peanut"),
                          (chips.PLACE, "Brooklyn"), (chips.COUNT, "10 restaurants")])

    def test_no_borough_filter_means_both_and_singular_count(self):
        out = chips.request_chips(ParsedQuery(), None, 1)
        self.assertEqual(out, [(chips.PLACE, "Manhattan + Brooklyn"), (chips.COUNT, "1 restaurant")])
        self.assertEqual(chips.request_chips(ParsedQuery(), ["Manhattan", "Brooklyn"], 0)[0], (chips.PLACE, "Manhattan + Brooklyn"))

    def test_unparsed_request_says_so(self):
        out = chips.request_chips(None, None, 3)
        self.assertEqual(out[0][0], chips.NOTE)
        self.assertIn("exact words", out[0][1])

    def test_long_chip_text_is_shortened(self):
        out = chips.request_chips(ParsedQuery(dietary_restrictions=["x" * 100]), None, 1)
        self.assertEqual(len(out[0][1]), chips.MAX_CHIP)
        self.assertTrue(out[0][1].endswith("…"))

    def test_html_is_escaped_and_single_block(self):
        html = chips.chips_html([(chips.DIET, '<script>alert("x")</script> & co'), (chips.COUNT, "2 restaurants")])
        self.assertNotIn("<script", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&amp; co", html)
        self.assertNotIn("\n\n", html)
        self.assertEqual(html.count('<span class="chip '), 2)


if __name__ == "__main__":
    unittest.main()
