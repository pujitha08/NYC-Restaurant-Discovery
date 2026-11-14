"""The page CSS must never reach Streamlit's icon elements.

Streamlit draws icons (sidebar toggle, expander arrows, ...) as text in the "Material Symbols" icon font. A font-family rule that
also matches those spans shows the ligature names as raw words ("keyboard_arrow_right"). These tests read the CSS the app injects
and check that every font-family rule is an explicit text-element rule that carries the icon guards.
"""

import re
import unittest

try:
    from app import chips
    from app import streamlit_app as ui
except ImportError:  # streamlit is optional for the rest of the suite
    ui = None

ICON_GUARDS = [':not([data-testid="stIconMaterial"])', ":not(.material-symbols-rounded)", ":not(.material-icons)",
               ':not([class*="material" i])']
ALLOWED_TAGS = set(ui.TEXT_ELEMENTS) if ui else set()
ALLOWED_CONTAINERS = {'[data-testid="stMarkdownContainer"]', '[data-testid="stCaptionContainer"]'}
FORBIDDEN_IN_POSITIVE_PART = ("span", "svg", "div", "stIconMaterial", "material", "[class*=", "*", "i.", "#")


def split_selectors(selector_list):
    """Split on commas that are not inside parentheses or brackets."""
    parts, depth, current = [], 0, ""
    for ch in selector_list:
        depth += ch in "([" and 1 or ch in ")]" and -1 or 0
        if ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    return [p for p in parts + [current.strip()] if p]


def font_rules(css):
    """[(selector, rule body)] for every rule that sets font-family."""
    css = re.sub(r"</?style>", "", css)
    out = []
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if "font-family" in body:
            out += [(s, body) for s in split_selectors(selectors.strip())]
    return out


def positive_part(selector):
    """The selector without its :not(...) guards."""
    previous = None
    while previous != selector:
        previous, selector = selector, re.sub(r":not\((?:[^()\[\]]|\[[^\]]*\])*\)", "", selector)
    return selector.strip()


@unittest.skipIf(ui is None, "streamlit not installed")
class CssDoesNotTargetIconsTests(unittest.TestCase):
    css = (ui.CSS if ui else "") + (chips.CSS if ui else "")

    def test_there_is_a_font_rule_and_it_uses_the_system_font(self):
        rules = font_rules(self.css)
        self.assertGreater(len(rules), 5)
        for _, body in rules:
            self.assertIn("-apple-system", body)
            self.assertNotIn("http", body)

    def test_no_universal_or_class_substring_selector_in_any_font_rule(self):
        for selector, _ in font_rules(self.css):
            positive = positive_part(selector)
            self.assertNotRegex(positive, r"(^|[\s>+~,(])\*($|[\s>+~,.#:\[)=])", selector)  # no blanket `*`
            self.assertNotIn("[class*=", positive, selector)  # the old `[class*="st-"]` is what reached the icon spans
            self.assertNotEqual(selector.strip(), "button")  # nothing bare that could wrap icon spans without guards

    def test_every_selector_is_html_body_or_an_allow_listed_text_element_with_all_icon_guards(self):
        for selector, _ in font_rules(self.css):
            if selector in ("html", "body"):
                continue
            for guard in ICON_GUARDS:
                self.assertIn(guard, selector, f"{guard} missing from {selector[:60]}")
            positive = positive_part(selector)
            self.assertTrue(positive.startswith(".stApp "), positive)
            target = positive[len(".stApp "):]
            self.assertTrue(target in ALLOWED_TAGS or target in ALLOWED_CONTAINERS, f"unexpected target {target!r}")

    def test_icon_elements_are_never_a_positive_target(self):
        for selector, _ in font_rules(self.css):
            positive = positive_part(selector)
            if positive in ("html", "body"):
                continue
            target = positive[len(".stApp "):]
            for forbidden in ("span", "svg", "div", "stIconMaterial", "material", "[class*=", "*", "#"):
                self.assertNotIn(forbidden, target, f"{forbidden!r} in {selector[:70]}")

    def test_icon_nodes_are_not_matched_by_the_allow_list(self):
        # The Material icon spans are <span data-testid="stIconMaterial" class="... material-symbols-rounded">: a span is
        # not an allow-listed text tag, and the guards exclude those attributes even if a text tag ever carried them.
        self.assertNotIn("span", ALLOWED_TAGS)
        self.assertNotIn("i", ALLOWED_TAGS)
        for guard in ICON_GUARDS:
            self.assertIn(guard, ui.ICON_GUARD)

    def test_no_external_requests_in_the_css(self):
        for needle in ("@import", "url(", "@font-face", "http://", "https://"):
            self.assertNotIn(needle, self.css)

    def test_helper_functions(self):
        self.assertEqual(split_selectors('a:not([x="1,2"]), b'), ['a:not([x="1,2"])', "b"])
        self.assertEqual(positive_part('.stApp p:not([data-testid="stIconMaterial"]):not(.m)'), ".stApp p")
        old_rule = 'html, body, [class*="st-"], button { font-family: x; }'
        self.assertIn('[class*="st-"]', [s for s, _ in font_rules(old_rule)])  # the old rule would be caught by the checks above


if __name__ == "__main__":
    unittest.main()
