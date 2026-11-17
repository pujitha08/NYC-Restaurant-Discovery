"""What the results screen shows, as plain functions (no Streamlit): which column a restaurant belongs to, the counts, the
one quote on a card, the wording for the need, and the "this app can't check that" split of a request.

The grouping reads the claim labels the summarizer produced (support / contradiction / uncertainty), not the badge text.
The badge for an allergy request says "Reviews discuss this" where a supporting claim exists (the allergy guard never
shows "supported"), so reading the labels is what puts those restaurants under "mention care".
Nothing here says a restaurant is safe: reviews are evidence, not a verdict.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from app import pipeline
from app.coverage import need_terms
from app.evidence_filter import NAME_ONLY_CLAIM, mentions
from app.highlight import highlight_terms, squash
from src.allergy import GENERIC_ALLERGY, GROUP_PATTERNS, NUT_GENERIC, NUT_GROUPS, allergen_groups, is_allergy_text
from src.retrieval.evidence import _SENTENCE_RE

CARE, PROBLEMS, MIXED, INSUFFICIENT = "care", "problems", "mixed", "insufficient"
UNAVAILABLE = "unavailable"  # the summary call failed: no label either way (State A)
NOT_SUMMARIZED = "not_summarized"  # past the summarized top results: counted under "not enough evidence"
LABEL_OF = {CARE: "support", PROBLEMS: "contradiction"}
MAX_QUOTE_CHARS = 280


# ---- the mapping --------------------------------------------------------------------------------------------------
# Display-time rules (no model call): a claim only counts when its cited reviews back it up.
MIN_CARE_STARS = 3  # a supporting claim needs at least one cited review rated this high
ALLERGY_CUE = re.compile(
    r"\b(?:allerg\w*|reactions?|anaphyl\w*|epi-?pens?|hives|swell\w*|rash(?:es)?|sick|ill(?:ness(?:es)?)?|hospital\w*|"
    r"ER visits?|celiac|coeliac|intoleran\w*|cross[- ]?contam\w*|glutened|can[\u2019']?t eat|cannot eat)\b", re.IGNORECASE)


def has_allergy_cue(text):
    return bool(ALLERGY_CUE.search(text or ""))


def care_claim_counts(claim):
    """A supporting claim counts unless every cited review that has a star rating is rated 1-2 (an unknown rating never excludes)."""
    rated = [e.rating for e in claim.excerpts if getattr(e, "rating", None) is not None]
    return not rated or max(rated) >= MIN_CARE_STARS


def problem_claim_counts(claim, outcome):
    """For an allergy request, a contradicting claim counts only if a cited review text has an allergy cue word
    (so "the sesame chicken tasted like an old tire" is not an allergy problem). Other requests are not gated."""
    if outcome is None or not outcome.allergy_sensitive:
        return True
    return any(has_allergy_cue(e.text) for e in claim.excerpts)


def allergen_patterns(outcome):
    """Regexes for the allergens the request names: each group's synonyms, or (for an allergen in no group) its own words.
    [] when the request names no allergen, which turns the allergen gate off."""
    p = outcome.parsed if outcome is not None else None
    patterns = []
    for allergen in (p.allergens if p else []):
        groups = allergen_groups(allergen)
        if groups:
            patterns += [GROUP_PATTERNS[g] for g in groups]
        else:
            words = need_terms(allergen)
            if words:
                patterns.append(re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\w*\b", re.IGNORECASE))
    return patterns


def names_the_allergen(text, patterns, name=None, generic=False):
    """True if the review text mentions the allergen or one of its synonyms outside the restaurant's name ("Red Lobster" is
    not a lobster mention), or, when `generic`, makes a generic allergy statement ("allergy", "allergies", "allergic")."""
    from app.evidence_filter import mask_name
    masked = mask_name(text, name)
    return any(pat.search(masked) for pat in patterns) or (generic and bool(GENERIC_ALLERGY.search(masked)))


def names_a_nut_group(outcome):
    p = outcome.parsed
    return any(g in NUT_GROUPS for allergen in (p.allergens if p else []) for g in allergen_groups(allergen))


def allergen_claim_counts(claim, outcome, name, generic):
    """For an allergy request that names an allergen: a claim counts only if a cited review mentions that allergen or a
    synonym. The gate is asymmetric. Care claims (generic=True) are strict, and may also make a generic allergy statement.
    Problem claims (generic=False) are warnings, so for a peanut or tree nut request they also count a generic "nut"/"nuts"
    mention. Other requests are not gated."""
    if outcome is None or not outcome.allergy_sensitive:
        return True
    patterns = allergen_patterns(outcome)
    if not patterns:
        return True
    if not generic and names_a_nut_group(outcome):
        patterns = patterns + [NUT_GENERIC]
    return any(names_the_allergen(e.text, patterns, name, generic) for e in claim.excerpts)


def effective_claims(view, outcome=None):
    """[(claim, label)] of the summary's claims that count: supporting claims that pass the care gate and, for allergy
    requests, contradicting claims that pass the problem gate. Uncertainty claims are kept as they are."""
    if view.summary is None:
        return []
    out = []
    for claim in pipeline.merge_neutral_claims(view.summary.claims):
        if claim.label == "support" and not (care_claim_counts(claim)
                                             and allergen_claim_counts(claim, outcome, view.name, generic=True)):
            continue
        if claim.label == "contradiction" and not (problem_claim_counts(claim, outcome)
                                                   and allergen_claim_counts(claim, outcome, view.name, generic=False)):
            continue
        out.append((claim, claim.label))
    return out


def labels_of(summary, view=None, outcome=None):
    """The set of claim labels that count (all of them when no view is given)."""
    if summary is None:
        return set()
    if view is None:
        return {c.label for c in pipeline.merge_neutral_claims(summary.claims)}
    return {label for _, label in effective_claims(view, outcome)}


def group_of(view, outcome=None):
    """CARE / PROBLEMS / MIXED / INSUFFICIENT / UNAVAILABLE / NOT_SUMMARIZED.

    care = at least one counting supporting claim and no counting contradicting claim; problems = at least one counting
    contradicting claim and no supporting one; mixed = both; insufficient = the summarizer found no evidence ("Insufficient
    evidence"), or nothing counts (only uncertainty claims, or every claim failed its gate). A summary whose status is "Reviews
    discuss this" (the allergy guard's word for a supporting claim) is care: it has a supporting claim.
    """
    s = view.summary
    if s is None:
        return UNAVAILABLE if view.summary_error else NOT_SUMMARIZED
    if s.status == pipeline.STATUS_INSUFFICIENT:
        return INSUFFICIENT
    labels = labels_of(s, view, outcome)
    if "support" in labels and "contradiction" in labels:
        return MIXED
    if "support" in labels:
        return CARE
    if "contradiction" in labels:
        return PROBLEMS
    return INSUFFICIENT


@dataclass
class Columns:
    care: List  # restaurants whose counting claims are all supporting (care column), in rank order
    problems: List  # restaurants whose counting claims are all contradicting (problems column)
    mixed: List  # restaurants with both: shown once, in the "Reviewers disagree" section, never in a column
    insufficient: List  # "not enough evidence": insufficient + not summarized
    unavailable: List  # the summary failed (State A)
    checked: int


def columns(outcome):
    """Each restaurant lands in exactly one list: care, problems, mixed, insufficient or unavailable."""
    out = Columns([], [], [], [], [], len(outcome.results))
    for v in outcome.results:
        g = group_of(v, outcome)
        if g == CARE:
            out.care.append(v)
        elif g == PROBLEMS:
            out.problems.append(v)
        elif g == MIXED:
            out.mixed.append(v)
        elif g in (INSUFFICIENT, NOT_SUMMARIZED):
            out.insufficient.append(v)
        elif g == UNAVAILABLE:
            out.unavailable.append(v)
    return out


# ---- wording for the need -----------------------------------------------------------------------------------------
_ALLERGY_SUFFIX = re.compile(r"\s+allerg(?:y|ies)\s*$", re.IGNORECASE)


def _join(items):
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def need_kind(outcome):
    """"allergy", "diet" or "other"."""
    if outcome.allergy_sensitive:
        return "allergy"
    p = outcome.parsed
    return "diet" if p is not None and p.dietary_restrictions else "other"


def diet_text(outcome):
    p = outcome.parsed
    return _join([d for d in p.dietary_restrictions if d.strip()]) if p is not None and p.dietary_restrictions else ""


def need_phrase(outcome):
    """The need in plain words for the heading: "peanut allergies", "halal", "vegan and gluten-free options"."""
    p = outcome.parsed
    allergens = [_ALLERGY_SUFFIX.sub("", a.strip()).strip() for a in (p.allergens if p else []) if a and a.strip()]
    allergens = [a for a in allergens if a and a.lower() not in ("allergy", "allergies")]
    if allergens:
        return f"{_join(allergens)} allergies"
    diet = diet_text(outcome)
    if diet:
        return f"{diet} options"
    return "your request"


def care_title(outcome):
    kind = need_kind(outcome)
    if kind == "allergy":
        return "Reviews mention allergy care"
    return f"Reviews mention {diet_text(outcome)} options" if kind == "diet" else "Reviews match your request"


def problems_column_title():
    return "Reviews report problems"


def care_subline(outcome):
    return "Worth a closer look. Confirm with the restaurant." if need_kind(outcome) == "allergy" \
        else "Worth a closer look. Reviews are anecdotal."


def problems_subline(outcome):
    kind = need_kind(outcome)
    what = "allergy trouble" if kind == "allergy" else f"{diet_text(outcome)} problems" if kind == "diet" else "problems"
    return f"Reviewers describe {what} here. Read before you go."


def empty_problems_message():
    return ("No restaurant in the top results has reviews reporting allergy problems. "
            "That is not a safety guarantee.")


def empty_care_message(outcome):
    """Shown in place of the care column when no restaurant belongs there (the mirror of empty_problems_message)."""
    kind = need_kind(outcome)
    what = f"care for {need_phrase(outcome)}" if kind == "allergy" else \
        f"{diet_text(outcome)} options" if kind == "diet" else "what you asked for"
    return f"No restaurant in the top results has reviews describing {what}. That doesn't mean none exist; reviews just don't say."


def care_column_note(outcome, has_mixed):
    """Text for an empty care column: the "none" message, or a pointer to the mixed section when the care reviews are there."""
    return "Reviews that mention care only appear below, where reviewers disagree." if has_mixed else empty_care_message(outcome)


def problems_column_note(outcome, has_mixed):
    if has_mixed:
        return "Reviews that report problems only appear below, where reviewers disagree."
    return empty_problems_message() if outcome.allergy_sensitive else \
        "No restaurant in the top results has reviews reporting problems. Reviews are anecdotal."


def count_line(outcome):
    """"10 restaurants checked · 3 mention allergy care · 2 report problems · 1 with mixed reviews · 4 not enough evidence".

    The parts add up to the restaurants checked (each restaurant is in one). A summary-unavailable part appears only for a live
    search, never for a saved result.
    """
    c = columns(outcome)
    kind = need_kind(outcome)
    mention = "allergy care" if kind == "allergy" else f"{diet_text(outcome)} options" if kind == "diet" else "your request"
    n = c.checked
    parts = [f"{n} restaurant{'' if n == 1 else 's'} checked", f"{len(c.care)} mention {mention}",
             f"{len(c.problems)} report problems"]
    if c.mixed:
        parts.append(f"{len(c.mixed)} with mixed reviews")
    parts.append(f"{len(c.insufficient)} not enough evidence")
    if c.unavailable and not outcome.cached:
        parts.append(f"{len(c.unavailable)} summary unavailable")
    return " · ".join(parts)


def results_heading(outcome):
    """The heading above the results: the parsed need in plain words; neutral when the request was not parsed or named no need."""
    if outcome.parsed is None:
        return "What reviewers say"
    need = need_phrase(outcome)
    return "What reviewers say about your request" if need == "your request" else f"What reviewers say about {need}"


def source_label(outcome):
    return "Saved demo result" if outcome.cached else "Live search"


def _reason(error):
    if error.get("code") == "daily_limit":
        return "daily limit"
    return {"rate_limit": "rate limit", "timeout": "timeout", "network": "no connection"}.get(
        error["category"], "the language model didn't answer")


def live_unavailable(outcome):
    """Why a live search produced nothing to show (a short phrase), or None when it did.

    A parser failure, a model that is not set up, or every summary failing means there is nothing honest to put in the columns:
    the page then says live search is unavailable instead of counting failed steps as "0 mention care". A saved result never is.
    """
    if outcome.cached:
        return None
    if outcome.parse_error:
        return _reason(outcome.parse_error)
    if outcome.setup_error:
        return "the language model isn't set up"
    daily = next((v.summary_error for v in outcome.results if v.summary_error and v.summary_error.get("code") == "daily_limit"), None)
    if daily:  # the daily limit stops the whole search: nothing after it was tried, so nothing honest is left to show
        return _reason(daily)
    tried = [v for v in outcome.results if v.summary or v.summary_error]
    if tried and all(v.summary_error for v in tried):
        return _reason(tried[0].summary_error)
    return None


def live_wait_seconds(outcome):
    """Seconds the provider asked us to wait (from the parser error or the first summary error that has one), or None."""
    errors = [outcome.parse_error] + [v.summary_error for v in outcome.results]
    return next((e["retry_seconds"] for e in errors if e and e.get("retry_seconds")), None)


def unavailable_message(reason, wait_seconds=None):
    text = f"Live search is unavailable right now ({reason}). Try one of the saved examples."
    if reason == "daily limit":
        text += " " + pipeline.daily_limit_message(wait_seconds)
    return text


def trace_parts(outcome, passages=None):
    """[(step label, seconds or None)] for "How this was found"; a saved demo result has no timings."""
    t = outcome.timings or {}
    searched = f"Searched {passages:,} passages" if isinstance(passages, int) else "Searched review passages"
    steps = [("Understood", t.get("parse_s")), (searched, t.get("retrieve_s")), ("Summarized with quotes", t.get("summarize_s"))]
    return [(label, seconds if not outcome.cached else None) for label, seconds in steps]


# ---- the quote on a card ------------------------------------------------------------------------------------------
@dataclass
class Quote:
    text: str
    review_id: int
    date: str
    rating: Optional[int]
    n_reviews: int  # distinct reviews behind the claim(s) this quote comes from


def sentences(text):
    out = []
    for part in _SENTENCE_RE.split(text or ""):
        part = squash(part).strip("… ").strip()
        if part:
            out.append(part)
    return out


def _window(sentence, terms, limit=MAX_QUOTE_CHARS):
    """The sentence, cut to about `limit` characters around its first match when it is longer."""
    if len(sentence) <= limit:
        return sentence
    m = re.search("|".join(re.escape(t) for t in terms), sentence, re.IGNORECASE) if terms else None
    start = max(0, min((m.start() if m else 0) - limit // 3, len(sentence) - limit))
    cut = sentence[start:start + limit].strip()
    return ("… " if start else "") + cut + (" …" if start + limit < len(sentence) else "")


def _score(sentence, name, terms, needs, prefer_cue=False):
    """(allergy cue when preferred, need-word hits outside the restaurant's name, other query-word hits, shorter is better)."""
    need_hits = sum(mentions(sentence, [t], name) for t in needs)
    other_hits = sum(mentions(sentence, [t], name) for t in terms if t not in needs)
    return (int(prefer_cue and has_allergy_cue(sentence)), need_hits, other_hits, -len(sentence))


def best_match(excerpts, name, terms, needs, prefer_cue=False):
    """(sentences of the review, index of the matched one, excerpt): the sentence with the most need-word hits outside the
    restaurant's name, then other matched words. With prefer_cue, a sentence with an allergy cue word wins first."""
    best = None
    for e in excerpts:
        parts = sentences(e.text)
        for i, sentence in enumerate(parts):
            key = _score(sentence, name, terms, needs, prefer_cue)
            if best is None or key > best[0]:
                best = (key, parts, i, e)
    return (best[1], best[2], best[3]) if best else (None, None, None)


MIN_MATCH_WORDS = 8  # a shorter matched sentence also gets the one before it
QUOTE_JOINER = " … "


def compose_quote(parts, i, terms, limit=MAX_QUOTE_CHARS):
    """The matched sentence plus the one after it (and the one before it when the match is under MIN_MATCH_WORDS words),
    joined with ellipses, at most about `limit` characters: the neighbours are dropped before the match is ever cut."""
    match = parts[i]
    before = parts[i - 1] if i > 0 and len(match.split()) < MIN_MATCH_WORDS else None
    after = parts[i + 1] if i + 1 < len(parts) else None
    for chosen in ([before, match, after], [match, after]):
        text = QUOTE_JOINER.join(x for x in chosen if x)
        if len(text) <= limit:
            return text
    if after and len(match) <= limit:  # the match fits but the neighbour does not: keep the match and a shortened neighbour
        room = limit - len(match) - len(QUOTE_JOINER)
        if room >= 40:
            words = after.split()
            cut = ""
            for w in words:
                if len(cut) + len(w) + 1 > room - 2:
                    break
                cut = f"{cut} {w}".strip()
            if cut:
                return match + QUOTE_JOINER + cut + " …"
    return match if len(match) <= limit else _window(match, terms, limit)


def need_words(outcome):
    """The words of the parsed diets and allergens (what a quote should be about)."""
    p = outcome.parsed
    return [t for need in ((p.dietary_restrictions + p.allergens) if p else []) for t in need_terms(need)]


def about_the_need(text, outcome, name=None, warning=False):
    """True if the review text mentions the need (a diet or allergen word, not in the restaurant's name) or, for an allergy
    request, has an allergy cue word. With no parsed need and no allergy request, every cited review counts."""
    needs = need_words(outcome)
    if mentions(text, needs, name) if needs else False:
        return True
    patterns = allergen_patterns(outcome) + ([NUT_GENERIC] if warning and names_a_nut_group(outcome) else [])
    if patterns and names_the_allergen(text, patterns, name):
        return True
    if outcome.allergy_sensitive:
        return has_allergy_cue(text)
    return not needs


def cited_reviews(view, label, outcome=None):
    """The reviews cited by the counting claims with the label, each once (by review id), in claim order."""
    seen, out = set(), []
    for claim, claim_label in effective_claims(view, outcome):
        if claim_label != label:
            continue
        for e in claim.excerpts:
            if e.review_id not in seen:
                seen.add(e.review_id)
                out.append(e)
    return out


def quote_for(view, outcome, group):
    """The one quote for a card in the column `group` (CARE or PROBLEMS), from the reviews cited by the counting claims of that
    column's label (care: reviews rated 3 stars or more). Problems cards in an allergy request prefer a sentence with an allergy
    cue word. "Based on N reviews" counts the distinct cited reviews that mention the need or an allergy cue.
    Falls back to the best sentence of any retrieved review; None if there is no text."""
    terms = highlight_terms(outcome.query, outcome.parsed)
    needs = need_words(outcome)
    cited = cited_reviews(view, LABEL_OF[group], outcome)
    pool = [e for e in cited if group != CARE or getattr(e, "rating", None) is None or e.rating >= MIN_CARE_STARS] or cited
    prefer_cue = group == PROBLEMS and outcome.allergy_sensitive
    parts, i, excerpt = best_match(pool, view.name, terms, needs, prefer_cue)
    if parts is None:
        parts, i, excerpt = best_match(view.excerpts, view.name, terms, needs, prefer_cue)
    if parts is None:
        return None
    n = len({e.review_id for e in cited if about_the_need(e.text, outcome, view.name, warning=group == PROBLEMS)})
    return Quote(compose_quote(parts, i, terms), excerpt.review_id, excerpt.date, getattr(excerpt, "rating", None), n)


def fallback_quote(view, outcome):
    """State A: the best-matching review sentence among the retrieved reviews (no care/problem label)."""
    terms = highlight_terms(outcome.query, outcome.parsed)
    parts, i, excerpt = best_match(view.excerpts, view.name, terms, need_words(outcome), outcome.allergy_sensitive)
    if parts is None:
        return None
    return Quote(compose_quote(parts, i, terms), excerpt.review_id, excerpt.date, getattr(excerpt, "rating", None), 0)


def review_pill(n):
    return f"Based on {n} review{'' if n == 1 else 's'}"


# ---- the finding on a card ---------------------------------------------------------------------------------------
UNSAFE_FINDING = re.compile(r"\b(?:no risk|risk[- ]free|without (?:any )?risk|zero risk|no danger|completely safe)\b", re.IGNORECASE)


@dataclass
class Finding:
    text: str  # the model's claim, verbatim
    review_ids: List[int]  # the cited reviews that are in this restaurant's evidence


def finding_is_shown_safely(text, outcome):
    """For an allergy request the finding must not assert safety (safe, safely, safety, guarantee, reassure*, no risk, ...).
    Looks at the whole sentence, quoted words included. Other requests are not checked."""
    if not outcome.allergy_sensitive:
        return True
    return not (pipeline.SAFETY_WORDS.search(text) or UNSAFE_FINDING.search(text))


def finding_for(view, outcome, group):
    """The model's plain-language finding for a card in the column `group` (CARE or PROBLEMS), verbatim, or None.

    It is the first counting claim of that column's label that (a) is a real finding (not the neutral replacement line the
    wording guard or the name filter writes), (b) cites at least one review that is in this restaurant's evidence, and (c) for an
    allergy request does not assert safety. With no such claim the card shows the quote alone.
    """
    evidence_ids = {e.review_id for e in view.excerpts}
    for claim, label in effective_claims(view, outcome):
        if label != LABEL_OF[group]:
            continue
        text = squash(claim.text)
        if not text or text in (squash(pipeline.NEUTRAL_CLAIM), squash(NAME_ONLY_CLAIM)):
            continue
        ids = [e.review_id for e in claim.excerpts if e.review_id in evidence_ids]
        if ids and finding_is_shown_safely(text, outcome):
            return Finding(text, ids)
    return None


# ---- the evidence view --------------------------------------------------------------------------------------------
BADGES = {CARE: ("✓", "Reviews mention it"), PROBLEMS: ("⚠", "Reviews report problems"),
          MIXED: ("◐", "Reviewers disagree"), INSUFFICIENT: ("○", "Not enough evidence"),
          UNAVAILABLE: ("…", "Summary unavailable"), NOT_SUMMARIZED: ("○", "Not summarized")}


def badge_text(group, outcome):
    icon, text = BADGES[group]
    if group == CARE:
        text = {"allergy": "Reviews mention allergy care", "diet": f"Reviews mention {diet_text(outcome)} options"}.get(
            need_kind(outcome), "Reviews match your request")
    return f"{icon} {text}"


# ---- the states ---------------------------------------------------------------------------------------------------
def is_no_evidence(outcome):
    """State B: results exist, but none has care, problems, mixed or an unavailable summary: nothing to show either way."""
    if not outcome.results:
        return False
    return all(group_of(v, outcome) in (INSUFFICIENT, NOT_SUMMARIZED) for v in outcome.results)


def suggestions(outcome):
    """[(chip label, query, borough)] for State B: widen the area, ask more broadly, or try the demo allergy."""
    out = []
    if outcome.borough_choice != pipeline.BOROUGH_CHOICES[0]:
        out.append(("Both boroughs", outcome.query, pipeline.BOROUGH_CHOICES[0]))
    p = outcome.parsed
    if p is not None and p.allergens:
        broad = f"restaurants with {_join([a for a in p.allergens])}-free options"
    elif p is not None and p.dietary_restrictions:
        broad = f"{diet_text(outcome)} restaurants"
    else:
        broad = "restaurants with dietary options"
    if broad.casefold() != outcome.query.strip().casefold():
        out.append(("A broader request", broad, outcome.borough_choice))
    peanut = "restaurant careful about peanut allergies"
    if peanut.casefold() != outcome.query.strip().casefold():
        out.append(("Peanut", peanut, outcome.borough_choice))
    return out


# What the app cannot check (reviews say nothing reliable about it): opening hours, prices, availability.
_TIME = r"(?:\d{1,2}(?::\d\d)?\s?(?:am|pm)|midnight|noon|\d{1,2}\b)"
UNSUPPORTED = [re.compile(p, re.IGNORECASE) for p in (
    rf"\b(?:open|opens|opening|closes?|closing)\b(?:\s+(?:late|early|now|today|tonight|all night|24 hours|24/7|"
    rf"(?:until|till|after|before|at)\s+{_TIME}|on\s+(?:sun|mon|tues|wednes|thurs|fri|satur)days?))?",
    r"\b(?:late[- ]night|24[/ -]?7|24 hours|all night|after midnight|hours of operation|opening hours)\b",
    r"\b(?:cheap(?:est)?|inexpensive|affordable|budget|expensive|pricey|prices?|priced)\b"
    r"|\b(?:under|less than|below|around)\s+\$?\d+\b|\$\d+|\${2,}",
    r"\b(?:reservations?|reserve|booking|book a table|walk-?ins?|wait times?|availability|available (?:tonight|today|now))\b",
    r"\b(?:tonight|right now|tomorrow|this (?:weekend|evening|afternoon))\b",
)]
_DIET_WORDS = re.compile(r"\b(?:vegan|vegetarian|halal|kosher|gluten|dairy|celiac|coeliac|pescatarian|plant[- ]based|allerg\w*)\b",
                         re.IGNORECASE)
_CONNECTOR_WORDS = frozenset("""and or but that which who is are its it's thats that's please also with for in at a an the
takes take accepts accept offers offer has have""".split())


def _trim(text):
    """The text without connector words and punctuation at both ends ("and", "that is", ",")."""
    words = text.split()
    word = lambda w: re.sub(r"\W+", "", w.lower().replace("'", ""))
    while words and (word(words[0]) in _CONNECTOR_WORDS or not word(words[0])):
        words.pop(0)
    while words and (word(words[-1]) in _CONNECTOR_WORDS or not word(words[-1])):
        words.pop()
    return " ".join(words).strip(" ,.;:-")


@dataclass
class Unsupported:
    clause: str  # what was understood but cannot be checked, e.g. "open late"
    rest: str  # the dietary part of the request, "" if nothing is left


def split_unsupported(query):
    """None if the request asks nothing about hours, prices or availability; otherwise the unsupported clause(s) and the
    rest of the request (the dietary part).

    A minimal rule, not a parser: the request is searched for hour, price and availability phrases (UNSUPPORTED); the
    matched text, up to the next comma, is the clause that cannot be checked, and the request without it is the part we can answer.
    """
    text = squash(query)
    spans = []
    for pattern in UNSUPPORTED:
        for m in pattern.finditer(text):
            spans.append(m.span())
    if not spans:
        return None
    spans.sort()
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    clause = "; ".join(text[a:b].strip(" ,.;") for a, b in merged)
    rest, pos = [], 0
    for a, b in merged:
        rest.append(text[pos:a])
        pos = b
    rest.append(text[pos:])
    remainder = squash(" ".join(rest))
    remainder = _trim(remainder)
    has_need = bool(_DIET_WORDS.search(remainder)) or is_allergy_text(remainder)
    return Unsupported(clause, remainder if has_need else "")
