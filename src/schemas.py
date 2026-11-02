"""Shared dataclasses passed between pipeline stages."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParsedQuery:
    """Structured form of a natural-language query, produced by the LLM parser."""

    location: Optional[str] = None
    dietary_restrictions: List[str] = field(default_factory=list)
    allergens: List[str] = field(default_factory=list)
    group_needs: List[str] = field(default_factory=list)
    preferences: List[str] = field(default_factory=list)


@dataclass
class RestaurantDoc:
    """A restaurant with its aggregated review text."""

    restaurant_id: str
    name: str
    borough: str
    address: str
    categories: List[str] = field(default_factory=list)
    avg_rating: Optional[float] = None
    num_reviews: int = 0
    reviews: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """A single ranked hit from a retrieval method."""

    restaurant_id: str
    score: float
    method: str  # e.g. "tfidf" or "embedding"
    evidence_snippets: List[str] = field(default_factory=list)
