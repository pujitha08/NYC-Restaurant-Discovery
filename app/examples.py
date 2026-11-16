"""The one-click example chips on the home screen, in two labeled rows. Each chip fills the search box with a natural
sentence and runs it; every query here is also in app/demo_cache.DEMO_QUERIES, so a built demo cache answers it instantly."""

ALLERGY_CHIPS = [
    ("Peanut", "restaurant careful about peanut allergies"),
    ("Tree nut", "restaurant careful about tree nut allergies"),
    ("Shellfish", "restaurant careful about shellfish allergies"),
    ("Sesame", "restaurant careful about sesame allergies"),
    ("Dairy-free", "restaurant careful about dairy allergies"),
    ("Celiac", "restaurant careful about celiac disease and cross-contamination"),
]
DIET_CHIPS = [
    ("Gluten-free", "gluten-free restaurants"),
    ("Vegetarian", "vegetarian restaurants"),
    ("Halal", "halal restaurants"),
    ("Kosher", "kosher restaurants"),
    ("Vegan + gluten-free group", "vegan and gluten-free options for a group"),
]
CHIP_ROWS = [("Allergies", ALLERGY_CHIPS), ("Diets and groups", DIET_CHIPS)]
CHIP_BOROUGH, CHIP_METHOD = "Both", "semantic_capped"


def chip_queries():
    return [query for _, row in CHIP_ROWS for _, query in row]
