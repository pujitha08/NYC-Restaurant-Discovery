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


# Allergen groups: the words that name the allergen or stand for it in a review. app/coverage.py has no such lists (it only
# matches the words of the parsed need), so they live here, next to ALLERGEN_NAMES, which they extend. A parsed allergen
# belongs to every group one of whose terms occurs in it ("celiac disease" -> gluten, "tree nut" -> tree nut).
ALLERGEN_GROUPS = {
    "shellfish": ("shellfish", "seafood", "shrimp", "prawn", "lobster", "crab", "crawfish", "crayfish", "oyster", "clam",
                  "mussel", "scallop"),
    "peanut": ("peanut", "groundnut"),
    "tree nut": ("tree nut", "nut", "almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut", "macadamia"),
    "sesame": ("sesame", "tahini"),
    "dairy": ("dairy", "milk", "lactose", "cheese"),
    "gluten": ("gluten", "celiac", "coeliac", "wheat"),
    "egg": ("egg",),
    "soy": ("soy", "soya", "soybean"),
    "fish": ("fish",),
}


def _term_regex(term):
    """Whole-word regex for a term (a space may be a hyphen too), optionally plural."""
    return r"\b" + r"[\s-]+".join(re.escape(w) for w in term.split()) + r"(?:e?s)?\b"


GROUP_PATTERNS = {name: re.compile("|".join(_term_regex(t) for t in terms), re.IGNORECASE)
                  for name, terms in ALLERGEN_GROUPS.items()}
GENERIC_ALLERGY = re.compile(r"\ballerg\w*\b", re.IGNORECASE)  # allergy, allergies, allergic


def allergen_groups(allergen):
    """The groups a parsed allergen belongs to, e.g. "peanut allergy" -> ["peanut"]; [] if it names none of them."""
    return [name for name, pattern in GROUP_PATTERNS.items() if pattern.search(allergen or "")]
# A warning about "nuts" in general matters to a peanut or a tree nut allergy, so problem claims (not care claims) also accept it.
NUT_GENERIC = re.compile(r"\bnuts?\b", re.IGNORECASE)
NUT_GROUPS = ("peanut", "tree nut")
