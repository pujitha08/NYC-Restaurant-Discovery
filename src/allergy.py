"""Allergy wording shared by the app's allergy guard and the stance re-ranker."""

import re

ALLERGY_TERMS = re.compile(
    r"\b(allerg\w*|celiac\w*|coeliac\w*|anaphyla\w*|epi-?pens?|cross[- ]?(?:contact|contaminat\w*)"
    r"|(?:tree[- ]?)?nut[- ]free|peanut[- ]free|gluten[- ]intoleran\w*)\b", re.IGNORECASE)
# FDA major allergens and common synonyms. A request naming one is guarded even without allergy wording
# ("no peanuts please", "sesame-free"); this is deliberately over-cautious ("peanut noodles" is guarded too).
# "gluten-free" alone is treated as a diet; celiac wording is caught by ALLERGY_TERMS.
ALLERGEN_NAMES = re.compile(
    r"\b(peanuts?|tree[- ]?nuts?|almonds?|cashews?|walnuts?|pecans?|pistachios?|hazelnuts?|macadamias?"
    r"|shellfish|shrimps?|crabs?|lobsters?|prawns?|crawfish|sesame|tahini|soy|soya|soybeans?"
    r"|eggs?|fish|milk|dairy|wheat)\b", re.IGNORECASE)


def is_allergy_text(query):
    """True if the text uses allergy wording or names an FDA major allergen."""
    return bool(ALLERGY_TERMS.search(query or "") or ALLERGEN_NAMES.search(query or ""))
