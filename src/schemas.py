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
    """A restaurant with its cleaned review text (one record of restaurant_docs.jsonl).

    gmap_id is the Google Local business id.
    """

    gmap_id: str
    name: str
    borough: str
    address: str
    categories: List[str] = field(default_factory=list)
    avg_rating: Optional[float] = None
    num_clean_reviews: int = 0
    low_evidence: bool = False  # fewer than 5 clean reviews
    text: str = ""  # all clean reviews concatenated
    reviews: List[str] = field(default_factory=list)  # optional; not stored in restaurant_docs.jsonl


@dataclass
class Chunk:
    """A retrieval unit: up to 5 consecutive reviews of one restaurant (one record of review_chunks.jsonl)."""

    chunk_id: str
    gmap_id: str
    text: str
    review_ids: List[int] = field(default_factory=list)
    times: List[Optional[int]] = field(default_factory=list)  # Unix ms; None for placeholder dates


@dataclass
class RetrievalResult:
    """A single ranked hit from a retrieval method."""

    gmap_id: str
    score: float
    method: str  # e.g. "tfidf" or "embedding"
    evidence_snippets: List[str] = field(default_factory=list)
