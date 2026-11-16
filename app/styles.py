"""The page styling: one CSS block, injected with st.markdown. Static text; nothing from users or reviews goes into it.

Colors differ in lightness and carry an icon next to them, never hue alone; every text/background pair is at least 4.5:1
(tests/test_quote_contrast.py reads the pairs from PALETTE). System fonts only (no web fonts, no external requests).

The system font is applied to an explicit list of TEXT elements, never to a blanket or class-substring selector:
Streamlit draws its icons (expander arrows, ...) as text in the "Material Symbols" icon font, so any rule that also reaches
those spans turns the icons into raw words like "arrow_right". Icon elements are not in the list and every selector in it
also carries ICON_GUARD, so they stay excluded even if a text tag ever wraps one (tests/test_streamlit_css.py).
"""

PALETTE = {
    "bg": "#FBF7F2", "card": "#FFFFFF", "line": "#E4D9CC", "ink": "#1F1B16", "muted": "#5E554B",
    "accent": "#B4461E", "on_accent": "#FFFFFF",
    "care_fg": "#17695F", "care_bg": "#E4F1EE",
    "problems_fg": "#9B3412", "problems_bg": "#FBE9E1",
    "disagree_fg": "#5A3B00", "disagree_bg": "#FFF0C2",
    "neutral_fg": "#4A433A", "neutral_bg": "#ECE6DD",
    "mark_fg": "#1B1B1B", "mark_bg": "#FFE08A",
}
P = PALETTE

SYSTEM_FONT = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif')
ICON_GUARD = (':not([data-testid="stIconMaterial"]):not(.material-symbols-rounded):not(.material-icons)'
              ':not([class*="material" i])')
TEXT_ELEMENTS = ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "label", "a", "button", "input", "textarea", "select",
                 "summary", "blockquote", "th", "td", "caption")
TEXT_CONTAINERS = ('[data-testid="stMarkdownContainer"]', '[data-testid="stCaptionContainer"]')
FONT_SELECTORS = (["html", "body"] + [f".stApp {tag}{ICON_GUARD}" for tag in TEXT_ELEMENTS]
                  + [f".stApp {container}{ICON_GUARD}" for container in TEXT_CONTAINERS])

MIN_TAP = "44px"  # buttons are at least this tall

CSS = """
<style>
""" + ",\n".join(FONT_SELECTORS) + " { font-family: " + SYSTEM_FONT + "; }" + f"""
.stApp {{ background: {P['bg']}; color: {P['ink']}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1100px; padding: 1rem 1rem 3rem 1rem; }}
.wordmark {{ font-weight: 700; font-size: 1.1rem; color: {P['ink']}; margin: 0.35rem 0; }}
.pill {{ display: inline-block; background: {P['card']}; border: 1px solid {P['line']}; color: {P['muted']};
  border-radius: 999px; padding: 0.2rem 0.8rem; font-size: 0.85rem; line-height: 1.5; }}
.hero {{ font-size: clamp(1.7rem, 5vw, 2.7rem); line-height: 1.15; font-weight: 700; color: {P['ink']}; margin: 1.2rem 0 0.5rem; max-width: 52rem; }}
.subline {{ color: {P['muted']}; font-size: 1.05rem; line-height: 1.5; max-width: 46rem; margin: 0 0 0.9rem; }}
.scope-line {{ color: {P['muted']}; font-size: 0.9rem; margin: 0.25rem 0 0.8rem; }}
.chip-label {{ color: {P['muted']}; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; margin: 0.4rem 0 0.1rem; }}
.stButton > button, [data-testid="stFormSubmitButton"] > button {{ min-height: {MIN_TAP}; border-radius: 10px; }}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {{
  background: {P['accent']}; border-color: {P['accent']}; color: {P['on_accent']}; font-weight: 700; }}
[class*="st-key-chip_"] button {{ border-radius: 999px; padding: 0.2rem 1rem; border: 1px solid {P['accent']};
  color: {P['accent']}; background: {P['card']}; font-weight: 600; }}
[class*="st-key-chip_"] button:hover {{ background: {P['problems_bg']}; border-color: {P['problems_fg']}; }}
.steps {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1.4rem 0 0.5rem; }}
.step {{ flex: 1 1 14rem; background: {P['card']}; border: 1px solid {P['line']}; border-radius: 12px; padding: 0.8rem 1rem; }}
.step-n {{ color: {P['accent']}; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.06em; }}
.step-t {{ color: {P['muted']}; font-size: 0.95rem; line-height: 1.45; margin-top: 0.2rem; }}
.understood {{ display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; margin: 0.3rem 0; }}
.understood-label {{ color: {P['muted']}; font-size: 0.9rem; font-weight: 600; }}
.chip {{ display: inline-block; border-radius: 999px; padding: 0.15rem 0.75rem; font-size: 0.9rem; font-weight: 600;
  background: {P['neutral_bg']}; color: {P['neutral_fg']}; border: 1px solid transparent; line-height: 1.5; }}
.chip-allergen {{ background: {P['problems_bg']}; color: {P['problems_fg']}; }}
.chip-diet {{ background: {P['care_bg']}; color: {P['care_fg']}; }}
.results-title {{ font-size: clamp(1.4rem, 4vw, 2rem); line-height: 1.2; font-weight: 700; color: {P['ink']}; margin: 0.8rem 0 0.3rem; }}
.count-line {{ color: {P['ink']}; font-size: 1rem; font-weight: 600; margin: 0.2rem 0; }}
.helper-line, .trace {{ color: {P['muted']}; font-size: 0.92rem; margin: 0.15rem 0; }}
.col-head {{ border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 0.6rem; }}
.col-head .t {{ font-weight: 700; font-size: 1.1rem; }}
.col-head .s {{ font-size: 0.92rem; margin-top: 0.1rem; }}
.col-head.care {{ background: {P['care_bg']}; color: {P['care_fg']}; }}
.col-head.problems {{ background: {P['problems_bg']}; color: {P['problems_fg']}; }}
[class*="st-key-card_"] {{ background: {P['card']}; border: 1px solid {P['line']}; border-radius: 12px; padding: 0.9rem 1rem; }}
.card-name {{ font-size: 1.25rem; font-weight: 700; color: {P['ink']}; line-height: 1.25; margin: 0; }}
.card-meta {{ color: {P['muted']}; font-size: 0.9rem; margin: 0.1rem 0 0.3rem; }}
.stApp blockquote.quote {{ margin: 0.5rem 0 0.4rem 0; padding: 0.1rem 0 0.1rem 0.8rem; border-left: 3px solid {P['line']};
  color: {P['ink']} !important; opacity: 1 !important; font-size: 1rem; line-height: 1.5; }}
.care-card blockquote.quote {{ border-left: 3px solid {P['care_fg']}; }}
.problems-card blockquote.quote {{ border-left: 6px double {P['problems_fg']}; }}  /* a different line style, not only a different hue */
.pill-n {{ display: inline-block; border-radius: 999px; padding: 0.1rem 0.7rem; font-size: 0.85rem; font-weight: 600;
  background: {P['neutral_bg']}; color: {P['neutral_fg']}; }}
.care-card .pill-n.many {{ background: {P['care_bg']}; color: {P['care_fg']}; }}
.problems-card .pill-n.many {{ background: {P['problems_bg']}; color: {P['problems_fg']}; }}
.tag-disagree {{ display: inline-block; background: {P['disagree_bg']}; color: {P['disagree_fg']}; border-radius: 6px;
  padding: 0.1rem 0.5rem; font-size: 0.8rem; font-weight: 700; margin-left: 0.4rem; }}
.also-note {{ color: {P['muted']}; font-size: 0.85rem; margin-top: 0.3rem; }}
.badge {{ display: inline-block; border-radius: 8px; padding: 0.2rem 0.65rem; font-weight: 700; font-size: 0.9rem;
  background: {P['neutral_bg']}; color: {P['neutral_fg']}; }}
.badge-care {{ background: {P['care_bg']}; color: {P['care_fg']}; }}
.badge-problems {{ background: {P['problems_bg']}; color: {P['problems_fg']}; }}
.badge-mixed {{ background: {P['disagree_bg']}; color: {P['disagree_fg']}; }}
.ev-label {{ color: {P['muted']}; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; margin-top: 0.6rem; }}
.claim-line {{ margin: 0.25rem 0; color: {P['ink']}; }}
.claim-tag {{ font-weight: 700; font-size: 0.8rem; color: {P['muted']}; }}
.review {{ margin: 0.5rem 0; color: {P['ink']}; }}
.review-meta {{ color: {P['muted']}; font-size: 0.85rem; }}
.caveat {{ color: {P['muted']}; font-size: 0.88rem; margin-top: 0.6rem; }}
details.found {{ color: {P['muted']}; font-size: 0.88rem; margin-top: 0.5rem; }}
details.found summary {{ cursor: pointer; font-weight: 600; }}
.state-title {{ font-size: clamp(1.25rem, 3.6vw, 1.7rem); font-weight: 700; color: {P['ink']}; margin: 1rem 0 0.3rem; }}
.state-body {{ color: {P['muted']}; font-size: 1rem; line-height: 1.5; max-width: 46rem; margin-bottom: 0.6rem; }}
.unavailable {{ color: {P['ink']}; margin: 0.4rem 0; }}
.empty-col {{ background: {P['card']}; border: 1px dashed {P['line']}; border-radius: 12px; padding: 0.8rem 1rem; color: {P['muted']}; }}
.footer {{ color: {P['muted']}; font-size: 0.85rem; border-top: 1px solid {P['line']}; margin-top: 2rem; padding-top: 0.8rem; }}
.more-line {{ margin: 0.2rem 0; color: {P['ink']}; }}
.more-line span {{ color: {P['muted']}; }}
@media (max-width: 640px) {{ .block-container {{ padding: 0.75rem 1rem 2.5rem 1rem; }} .hero {{ margin-top: 0.6rem; }} }}
</style>
"""
