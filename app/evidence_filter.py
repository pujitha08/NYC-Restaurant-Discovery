"""A match that comes only from the restaurant's own name is not evidence ("The Halal Guys" says nothing about halal food).

Review text is masked with the restaurant's name before the need's words are looked for. Used for the claim labels
(pipeline.to_summary), the quote shown on a card, the highlighting and the coverage chips. It never touches retrieval or ranking.
"""

import re

from app.highlight import html_text, term_pattern

NAME_ONLY_CLAIM = "This review only mentions the restaurant by name; it does not say anything about the request."


def name_pattern(name):
    """Regex for the restaurant's name (and the name without a leading "The"), whitespace-flexible, optional possessive; or None."""
    words = (name or "").split()
    variants = []
    for ws in (words, words[1:] if words and words[0].lower() == "the" else []):
        tokens = [re.escape(w.strip(".,;:!?\"'")) for w in ws]
        tokens = [t for t in tokens if t]
        if tokens:
            variants.append(r"\s+".join(tokens))
    if not variants:
        return None
    body = "|".join(sorted(set(variants), key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{body})(?:['’]s)?(?!\w)", re.IGNORECASE)


def name_spans(text, name):
    pattern = name_pattern(name)
    return [m.span() for m in pattern.finditer(text or "")] if pattern else []


def mask_name(text, name):
    """The text with every occurrence of the restaurant's name replaced by a space."""
    pattern = name_pattern(name)
    return pattern.sub(" ", text or "") if pattern else (text or "")


def mentions(text, terms, name=None):
    """True if a term occurs in the text outside the restaurant's name."""
    pattern = term_pattern(terms)
    return bool(pattern and pattern.search(mask_name(text, name)))


def mentions_only_in_name(text, terms, name):
    """True if a term occurs in the text, but only inside the restaurant's name."""
    pattern = term_pattern(terms)
    return bool(pattern and pattern.search(text or "") and not pattern.search(mask_name(text, name)))


def highlight_outside_name(text, terms, name, mark):
    """HTML-safe text with the terms wrapped by mark.format(...), except inside the restaurant's name.

    Every segment is HTML-escaped first; `text` should already be whitespace-squashed.
    """
    text = text or ""
    pattern = term_pattern(terms)
    if pattern is None:
        return html_text(text)
    spans = name_spans(text, name)
    out, pos = [], 0
    for m in pattern.finditer(text):
        if any(a <= m.start() < b for a, b in spans):
            continue
        out += [html_text(text[pos:m.start()]), mark.format(html_text(m.group()))]
        pos = m.end()
    out.append(html_text(text[pos:]))
    return "".join(out)
