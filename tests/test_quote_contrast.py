"""Review quotes must be dark and readable (WCAG AA, at least 4.5:1) on the red, green and grey claim backgrounds.

The colors are read from the CSS the app injects, so the test follows the real stylesheet. It also accounts for opacity:
Streamlit gives every blockquote opacity 0.6, which faded the dark quote text to grey (about 4:1 on the red background)
even though its declared color was fine, so the quote rules must reset opacity explicitly.
"""

import inspect
import re
import unittest

try:
    from app import highlight, styles
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
class PaletteContrastTests(unittest.TestCase):
    """Every text/background pair of the redesign is at least 4.5:1, read from the CSS the app injects."""

    css = rules(ui.CSS if ui else "")
    P = styles.PALETTE if ui else {}

    def test_the_palette_is_the_one_asked_for(self):
        expected = {"bg": "#FBF7F2", "ink": "#1F1B16", "muted": "#5E554B", "accent": "#B4461E", "care_fg": "#17695F",
                    "care_bg": "#E4F1EE", "problems_fg": "#9B3412", "problems_bg": "#FBE9E1", "disagree_fg": "#5A3B00",
                    "disagree_bg": "#FFF0C2"}
        for key, color in expected.items():
            self.assertEqual(self.P[key].upper(), color, key)

    def test_every_pair_is_at_least_4_5_to_1(self):
        P = self.P
        pairs = [("ink", "bg"), ("ink", "card"), ("muted", "bg"), ("muted", "card"), ("muted", "neutral_bg"),
                 ("on_accent", "accent"), ("accent", "card"), ("accent", "bg"), ("accent", "problems_bg"),
                 ("care_fg", "care_bg"), ("problems_fg", "problems_bg"), ("disagree_fg", "disagree_bg"),
                 ("neutral_fg", "neutral_bg"), ("mark_fg", "mark_bg"), ("ink", "mark_bg"), ("problems_fg", "card"),
                 ("care_fg", "card")]
        for fg, bg in pairs:
            self.assertGreaterEqual(contrast(P[fg], P[bg]), 4.5, f"{fg} on {bg}")

    def test_the_css_uses_the_palette_for_the_columns_tag_and_primary_button(self):
        P = self.P
        self.assertEqual((value(self.css[".col-head.care"], "background"), value(self.css[".col-head.care"], "color")),
                         (P["care_bg"], P["care_fg"]))
        self.assertEqual((value(self.css[".col-head.problems"], "background"), value(self.css[".col-head.problems"], "color")),
                         (P["problems_bg"], P["problems_fg"]))
        self.assertEqual((value(self.css[".tag-disagree"], "background"), value(self.css[".tag-disagree"], "color")),
                         (P["disagree_bg"], P["disagree_fg"]))
        self.assertEqual(value(self.css[".stApp"], "background"), P["bg"])
        button = self.css['.stButton > button[kind="primary"]']
        self.assertEqual((value(button, "background"), value(button, "color")), (P["accent"], P["on_accent"]))

    def test_columns_are_told_apart_by_icon_wording_and_line_style_not_hue_alone(self):
        # The two given text colors are about equally light (WCAG luminance 0.11 and 0.09), so hue is not what separates
        # the columns: each header has its own icon and words, and the quote line differs in style (solid vs double).
        self.assertNotEqual(self.P["care_fg"], self.P["problems_fg"])
        source = inspect.getsource(ui.column_head)
        self.assertIn("✓", source)
        self.assertIn("⚠", source)
        self.assertIn("solid", self.css[".care-card blockquote.quote"]["border-left"])
        self.assertIn("double", self.css[".problems-card blockquote.quote"]["border-left"])

    def test_buttons_are_at_least_44_px_tall(self):
        merged = {}
        for selector, props in self.css.items():
            if "button" in selector and "min-height" in props:
                merged[selector] = props["min-height"]
        self.assertTrue(merged)
        for selector, height in merged.items():
            self.assertGreaterEqual(int(height.replace("px", "")), 44, selector)
        self.assertEqual(styles.MIN_TAP, "44px")

    def test_quote_text_is_dark_and_opacity_is_reset(self):
        quote = self.css[".stApp blockquote.quote"]
        self.assertEqual(quote["opacity"].replace(" ", ""), "1!important")
        self.assertTrue(quote["color"].endswith("!important"))  # beats Streamlit's blockquote rule (opacity 0.6)
        text = value(quote, "color")
        for bg in ("bg", "card", "care_bg", "problems_bg"):
            self.assertGreaterEqual(contrast(text, self.P[bg]), 7, bg)

    def test_the_old_opacity_would_have_failed(self):
        # Negative control: Streamlit's default opacity 0.6 on the clay background.
        seen = blend(self.P["ink"], self.P["problems_bg"], 0.6)
        self.assertLess(contrast(seen, self.P["problems_bg"]), 7)

    def test_the_page_works_at_phone_width(self):
        self.assertIn("@media (max-width: 640px)", ui.CSS)
        self.assertRegex(ui.CSS, r"\.block-container \{ max-width: 1100px; padding: 1rem 1rem")  # a 16px side gutter

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
