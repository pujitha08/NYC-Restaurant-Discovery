"""Per-card coverage row: for each parsed diet, allergen and group need, whether the shown excerpts mention it.

The match uses the same term matcher as the highlighting (app/highlight.py): a need's words, minus the stopwords, are
matched as whole words or by prefix, and an excerpt counts when every one of those words occurs in it (so a multi-word
group need is not "found" just because one of its words is). Needs with no matchable word are left out rather than
reported either way. Only the excerpts shown on the card are searched, so the row says nothing about the restaurant
beyond them, and its wording never implies safety: allergens (and every need on allergy-sensitive requests) read
"mentioned in N excerpts" / "not mentioned in shown excerpts", never with a check mark. Parsed preferences (cuisine,
dish, venue, hours) are not checked at all; they are listed as "Not verified from reviews".

Text comes from the language model and the user, so every string is HTML-escaped; the markup around it is static.
"""

import html
import re

from app.chips import short
from app.highlight import STOPWORDS, _WORD, MIN_PREFIX_LEN, term_pattern

FOUND, NONE = "found", "none"


def need_terms(need):
    """The words of a need that the matcher uses (lowercase, at least 3 letters, not a stopword)."""
    terms = []
    for word in _WORD.findall((need or "").lower()):
        if len(word) >= 3 and word not in STOPWORDS and word not in terms:
            terms.append(word)
    return terms


def count_excerpts(need, excerpts):
    """How many of the excerpts mention the need (every term of it), or None if the need has no matchable word."""
    terms = need_terms(need)
    if not terms:
        return None
    patterns = [term_pattern([t]) for t in terms]
    return sum(all(p.search(e.text) for p in patterns) for e in excerpts)


def coverage(parsed, excerpts, allergy_sensitive=False):
    """[(state, text)] for each parsed diet, allergen and group need: state is FOUND or NONE. [] when nothing is parsed."""
    if parsed is None:
        return []
    needs = [(n, False) for n in parsed.dietary_restrictions + parsed.group_needs] + [(n, True) for n in parsed.allergens]
    out = []
    for need, is_allergen in needs:
        n = count_excerpts(need, excerpts)
        if n is None:
            continue
        label = short(need)
        careful = is_allergen or allergy_sensitive
        if n == 0:
            out.append((NONE, f"{label}: not mentioned in shown excerpts" if careful else f"{label}: not found in these excerpts"))
        else:
            word = "excerpt" if n == 1 else "excerpts"
            out.append((FOUND, f"{label}: mentioned in {n} {word}" if careful else f"{label} ✓ {n} {word}"))
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
