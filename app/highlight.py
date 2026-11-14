"""Safe highlighting of the query's requirement terms in review text for st.markdown(unsafe_allow_html=True).

Review text is untrusted. It is split into plain and matched segments on the RAW text; every segment is
HTML-escaped and then Markdown-escaped, and only then is the <mark> tag added around matched segments, so no
markup from a review can reach the page and the only tags present are the ones added here.
"""

import html
import re
from typing import Iterable, Optional

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|$~])")
_WORD = re.compile(r"[A-Za-z]+")
MAX_TERMS = 12
STEM_LEN = 6  # longer terms match by prefix, so "allergies" finds "allergy" and "allergic"
MIN_PREFIX_LEN = 5  # shorter terms must match whole words (a "nut" prefix would match "nutritious")
MARK = '<mark style="background:#ffe08a;color:#1b1b1b;padding:0 2px;border-radius:2px">{}</mark>'
STOPWORDS = frozenset("""
the and for with that this from have has are was were not but you your our can will any all some into about than
who what where when how out too very just like want need find looking look near around place places spot spots
restaurant restaurants food foods eat eating dinner lunch brunch breakfast options option good best great nice
someone somewhere something anything friend friends group groups free
""".split())


def md(text):
    """Escape untrusted review/model text for st.markdown (e.g. '$3' would otherwise start LaTeX)."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def escape(text):
    """Plain text -> Markdown/HTML-safe text (entities for & < >, backslashes before Markdown specials)."""
    return md(html.escape(text, quote=False))


def highlight_terms(query, parsed=None) -> list:
    """Requirement terms: the words of the query plus the parsed diets, allergens and preferences."""
    sources = [query or ""]
    if parsed is not None:
        sources += [*parsed.dietary_restrictions, *parsed.allergens, *parsed.preferences]
    terms = []
    for text in sources:
        for word in _WORD.findall(text.lower()):
            if len(word) >= 3 and word not in STOPWORDS and word not in terms:
                terms.append(word)
    return terms[:MAX_TERMS]


def term_pattern(terms: Iterable[str]) -> Optional[re.Pattern]:
    prefixes, whole = set(), set()
    for t in terms:
        t = "".join(_WORD.findall(t)).lower()
        if len(t) >= MIN_PREFIX_LEN:
            prefixes.add(re.escape(t[:STEM_LEN]))
        elif len(t) >= 3:
            whole.add(re.escape(t))
    parts = [rf"{p}\w*" for p in sorted(prefixes, key=len, reverse=True)] + [rf"{w}s?" for w in sorted(whole)]
    return re.compile(rf"\b(?:{'|'.join(parts)})\b", re.IGNORECASE) if parts else None


def highlight(text, terms) -> str:
    """Markdown/HTML-safe text with the terms wrapped in <mark>. Whitespace is collapsed to single spaces."""
    text = " ".join((text or "").split())
    pattern = term_pattern(terms)
    if pattern is None:
        return escape(text)
    out, pos = [], 0
    for m in pattern.finditer(text):
        out += [escape(text[pos:m.start()]), MARK.format(escape(m.group()))]
        pos = m.end()
    out.append(escape(text[pos:]))
    return "".join(out)


def html_text(text) -> str:
    """Plain text -> HTML-safe text for a raw HTML block (entities only: a raw block is not Markdown-processed)."""
    return html.escape(" ".join((text or "").split()), quote=True)


def highlight_html(text, terms) -> str:
    """Like highlight(), for use inside a raw HTML block: every segment is HTML-escaped, matches get the <mark> tag."""
    text = " ".join((text or "").split())
    pattern = term_pattern(terms)
    if pattern is None:
        return html_text(text)
    out, pos = [], 0
    for m in pattern.finditer(text):
        out += [html_text(text[pos:m.start()]), MARK.format(html_text(m.group()))]
        pos = m.end()
    out.append(html_text(text[pos:]))
    return "".join(out)
