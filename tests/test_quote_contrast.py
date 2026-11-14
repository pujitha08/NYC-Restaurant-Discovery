"""Review quotes must be dark and readable (WCAG AA, at least 4.5:1) on the red, green and grey claim backgrounds.

The colors are read from the CSS the app injects, so the test follows the real stylesheet. It also accounts for opacity:
Streamlit gives every blockquote opacity 0.6, which faded the dark quote text to grey (about 4:1 on the red background)
even though its declared color was fine, so the quote rules must reset opacity explicitly.
"""

import re
import unittest

try:
    from app import highlight
    from app import streamlit_app as ui
except ImportError:  # streamlit is optional for the rest of the suite
    ui = None

from tests.test_streamlit_config import contrast

WHITE = "#FFFFFF"


def rgb(hex_color):
    return tuple(int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


def blend(fg, bg, opacity):
    """The color a foreground of the given opacity shows on a background (what the eye sees)."""
    r = [round(opacity * f + (1 - opacity) * b) for f, b in zip(rgb(fg), rgb(bg))]
    return "#%02X%02X%02X" % tuple(r)


def rules(css):
    """{selector: {property: value}} for every plain rule (comments removed, !important kept in the value)."""
    css = re.sub(r"/\*.*?\*/", "", re.sub(r"</?style>", "", css), flags=re.S)
    out = {}
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        props = {k.strip(): v.strip() for k, v in (d.split(":", 1) for d in body.split(";") if ":" in d)}
        for selector in selectors.split(","):
            out.setdefault(selector.strip(), {}).update(props)
    return out


def value(props, name):
    return re.sub(r"\s*!important", "", props[name]).strip()


@unittest.skipIf(ui is None, "streamlit not installed")
class QuoteContrastTests(unittest.TestCase):
    css = rules(ui.CSS if ui else "")
    backgrounds = {}

    @classmethod
    def setUpClass(cls):
        base = value(cls.css[".claim"], "background")
        cls.backgrounds = {"mention (grey)": base, "warning (red)": value(cls.css[".claim-warning"], "background"),
                           "positive (green)": value(cls.css[".claim-positive"], "background"), "page (white)": WHITE}

    def test_quote_text_is_dark_and_readable_on_every_background(self):
        quote = self.css[".stApp blockquote.quote"]
        text = value(quote, "color")
        for name, bg in self.backgrounds.items():
            self.assertGreaterEqual(contrast(text, bg), 4.5, f"quote text on {name}")
            self.assertGreaterEqual(contrast(text, bg), 7, f"quote text should be near-black on {name}")

    def test_date_is_a_slightly_lighter_dark_grey_and_still_readable(self):
        quote, date = self.css[".stApp blockquote.quote"], self.css[".stApp blockquote.quote .quote-date"]
        text, grey = value(quote, "color"), value(date, "color")
        self.assertNotEqual(text.upper(), grey.upper())
        self.assertGreater(sum(rgb(grey)), sum(rgb(text)))  # lighter than the quote text ...
        self.assertLess(max(rgb(grey)) - min(rgb(grey)), 12)  # ... but still a neutral grey
        for name, bg in self.backgrounds.items():
            self.assertGreaterEqual(contrast(grey, bg), 4.5, f"date on {name}")

    def test_opacity_is_reset_so_the_declared_colors_are_what_the_eye_sees(self):
        for selector in (".stApp blockquote.quote", ".stApp blockquote.quote .quote-date"):
            self.assertEqual(self.css[selector]["opacity"].replace(" ", ""), "1!important", selector)
            self.assertTrue(self.css[selector]["color"].endswith("!important"), selector)  # beats Streamlit's blockquote rule

    def test_the_old_opacity_would_have_failed(self):
        # Negative control: Streamlit's default opacity 0.6 on the red claim background.
        text = value(self.css[".stApp blockquote.quote"], "color")
        seen = blend(text, self.backgrounds["warning (red)"], 0.6)
        self.assertLess(contrast(seen, self.backgrounds["warning (red)"]), 4.5)

    def test_claim_text_and_labels_are_readable_on_their_own_backgrounds(self):
        self.assertGreaterEqual(contrast(value(self.css[".claim-text"], "color"), self.backgrounds["warning (red)"]), 7)
        pairs = [(".claim-label", "mention (grey)"), (".claim-warning .claim-label", "warning (red)"),
                 (".claim-positive .claim-label", "positive (green)")]
        for selector, bg in pairs:
            self.assertGreaterEqual(contrast(value(self.css[selector], "color"), self.backgrounds[bg]), 4.5, selector)

    def test_highlight_mark_is_readable(self):
        bg = re.search(r"background:(#[0-9a-fA-F]{6})", highlight.MARK).group(1)
        fg = re.search(r"color:(#[0-9a-fA-F]{6})", highlight.MARK).group(1)
        self.assertGreaterEqual(contrast(fg, bg), 4.5)

    def test_helpers(self):
        self.assertEqual(blend("#000000", "#FFFFFF", 0.5), "#808080")
        self.assertEqual(rules("<style>/* c { x: y; } */ .a, .b { color: #fff !important; }</style>"),
                         {".a": {"color": "#fff !important"}, ".b": {"color": "#fff !important"}})


if __name__ == "__main__":
    unittest.main()
