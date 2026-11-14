"""Request chips: contents, scope wording and HTML escaping."""

import unittest

from app import chips
from src.schemas import ParsedQuery


class ChipTests(unittest.TestCase):
    def test_diets_allergens_and_borough_only(self):
        parsed = ParsedQuery(dietary_restrictions=["vegan", "gluten-free"], allergens=["peanut"], preferences=["brunch"])
        self.assertEqual(chips.request_chips(parsed, ["Brooklyn"]),
                         [(chips.DIET, "vegan"), (chips.DIET, "gluten-free"), (chips.ALLERGEN, "peanut"),
                          (chips.PLACE, "Brooklyn")])

    def test_there_is_no_result_count_chip(self):
        # The count is the heading above the list; repeating it as a chip was redundant.
        self.assertFalse(hasattr(chips, "COUNT"))
        self.assertNotIn("restaurant", " ".join(t for _, t in chips.request_chips(ParsedQuery(), None)))
        self.assertNotIn("chip-count", chips.CSS)

    def test_no_borough_filter_means_both(self):
        self.assertEqual(chips.request_chips(ParsedQuery(), None), [(chips.PLACE, "Manhattan + Brooklyn")])
        self.assertEqual(chips.request_chips(ParsedQuery(), ["Manhattan", "Brooklyn"]), [(chips.PLACE, "Manhattan + Brooklyn")])

    def test_unparsed_request_says_so(self):
        out = chips.request_chips(None, None)
        self.assertEqual(out[0][0], chips.NOTE)
        self.assertIn("exact words", out[0][1])

    def test_long_chip_text_is_shortened(self):
        out = chips.request_chips(ParsedQuery(dietary_restrictions=["x" * 100]), None)
        self.assertEqual(len(out[0][1]), chips.MAX_CHIP)
        self.assertTrue(out[0][1].endswith("…"))

    def test_html_is_escaped_and_single_block(self):
        html = chips.chips_html([(chips.DIET, '<script>alert("x")</script> & co'), (chips.PLACE, "Brooklyn")])
        self.assertNotIn("<script", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&amp; co", html)
        self.assertNotIn("\n\n", html)
        self.assertEqual(html.count('<span class="chip '), 2)


if __name__ == "__main__":
    unittest.main()
