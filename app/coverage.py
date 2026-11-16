"""Per-card coverage row: for each parsed diet, allergen and group need, whether the shown excerpts mention it.

The match uses the same term matcher as the highlighting (app/highlight.py): a need's words, minus the stopwords, are
matched as whole words or by prefix, and an excerpt counts when every one of those words occurs in it (so a multi-word
group need is not "found" just because one of its words is). Needs with no matchable word are left out rather than
reported either way. Only the excerpts shown on the card are searched, so the row says nothing about the restaurant
beyond them. One wording for every need type (diet, allergen, group need): "<need>: mentioned in N excerpt(s)" or
"<need>: not mentioned in shown excerpts". It counts mentions, not verdicts (an excerpt that says a place has no vegan
dishes still mentions "vegan"), and it never implies safety or uses a check mark. Parsed preferences (cuisine,
dish, venue, hours) are not checked at all; they are listed as "Not verified from reviews".

Text comes from the language model and the user, so every string is HTML-escaped; the markup around it is static.
"""

import html
import re

from app.chips import short
from app.evidence_filter import mask_name
from app.highlight import STOPWORDS, _WORD, MIN_PREFIX_LEN, term_pattern

FOUND, NONE = "found", "none"


def need_terms(need):
    """The words of a need that the matcher uses (lowercase, at least 3 letters, not a stopword)."""
    terms = []
    for word in _WORD.findall((need or "").lower()):
        if len(word) >= 3 and word not in STOPWORDS and word not in terms:
            terms.append(word)
    return terms


def count_excerpts(need, excerpts, name=None):
    """How many of the excerpts mention the need (every term of it), or None if the need has no matchable word.

    A match inside the restaurant's own name ("The Halal Guys") does not count."""
    terms = need_terms(need)
    if not terms:
        return None
    patterns = [term_pattern([t]) for t in terms]
    return sum(all(p.search(mask_name(e.text, name)) for p in patterns) for e in excerpts)


def coverage(parsed, excerpts, name=None):
    """[(state, text)] for each parsed diet, allergen and group need: state is FOUND or NONE. [] when nothing is parsed."""
    if parsed is None:
        return []
    out = []
    for need in parsed.dietary_restrictions + parsed.allergens + parsed.group_needs:
        n = count_excerpts(need, excerpts, name)
        if n is None:
            continue
        if n == 0:
            out.append((NONE, f"{short(need)}: not mentioned in shown excerpts"))
        else:
            out.append((FOUND, f"{short(need)}: mentioned in {n} {'excerpt' if n == 1 else 'excerpts'}"))
    return out


def preferences_line(parsed):
    """The muted 'Not verified from reviews: …' text (plain text, to be escaped by the caller), or ''."""
    prefs = [short(p) for p in (parsed.preferences if parsed is not None else []) if p and p.strip()]
    return f"Not verified from reviews: {', '.join(prefs)}" if prefs else ""


def coverage_html(items):
    """One HTML block (no blank lines) of chips, every text escaped, or '' for no items."""
    if not items:
        return ""
    body = "".join(f'<span class="chip chip-{state}">{html.escape(text, quote=True)}</span>' for state, text in items)
    return f'<div class="chips coverage">{body}</div>'
