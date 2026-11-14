"""The parsed request as chips above the results: diets, allergens, borough and the result count.

Chip text comes from the language model and the user, so it is HTML-escaped here; the markup around it is static.
"""

import html

MAX_CHIP = 40

DIET, ALLERGEN, PLACE, COUNT, NOTE = "diet", "allergen", "place", "count", "note"
ICON = {DIET: "🥗", ALLERGEN: "⚠", PLACE: "📍", COUNT: "🍽", NOTE: "ℹ"}
CSS = """
<style>
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.25rem 0 0.75rem 0; }
.chip { display: inline-block; border-radius: 999px; padding: 0.15rem 0.75rem; font-size: 0.9rem; font-weight: 600;
  border: 1px solid transparent; line-height: 1.5; }
.chip-diet { background: #E8F3E9; color: #14532D; border-color: #A7D3AD; }
.chip-allergen { background: #FDECEA; color: #7F1D1D; border-color: #F2B8B3; }
.chip-place { background: #FAF3EA; color: #7C2D12; border-color: #E8CDB0; }
.chip-count { background: #1C1917; color: #FFFFFF; }
.chip-note { background: #F3F4F6; color: #1F2937; border-color: #D1D5DB; }
</style>
"""


def short(text):
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_CHIP else text[:MAX_CHIP - 1] + "…"


def request_chips(parsed, boroughs, count):
    """[(kind, text)]: each diet and allergen, the borough scope and the number of results."""
    chips = []
    if parsed is None:
        chips.append((NOTE, "Request not interpreted: searched your exact words"))
    else:
        chips += [(DIET, short(d)) for d in parsed.dietary_restrictions]
        chips += [(ALLERGEN, short(a)) for a in parsed.allergens]
    chips.append((PLACE, " + ".join(boroughs) if boroughs else "Manhattan + Brooklyn"))
    chips.append((COUNT, f"{count} restaurant{'' if count == 1 else 's'}"))
    return chips


def chips_html(chips):
    """One HTML block (no blank lines) with every chip's text escaped."""
    body = "".join(f'<span class="chip chip-{kind}">{ICON[kind]} {html.escape(text, quote=True)}</span>' for kind, text in chips)
    return f'<div class="chips">{body}</div>'
