"""Compare mode: semantic_capped and semantic_capped_stance for the same query, with rank-change markers.

The query is parsed once (the stance run reuses the base run's parse). Neither column is summarized (no
summary calls); open a ranking on its own for summaries. In demo mode the base column comes from the cached demo
outcome when there is one, and the stance column uses only labels already in the on-disk cache (no model call).
If the stance column cannot be produced (rate limit, no cached labels, not configured) the base ranking is
shown alone with a notice.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app import demo_cache, pipeline

BASE_METHOD, STANCE_METHOD = "semantic_capped", pipeline.STANCE_METHOD
UP, DOWN, SAME, NEW = "up", "down", "same", "new"


@dataclass
class Comparison:
    base: pipeline.SearchOutcome
    stance: Optional[pipeline.SearchOutcome]  # None when the stance ranking is unavailable
    changes: Dict[str, tuple] = field(default_factory=dict)  # gmap_id -> (UP|DOWN|SAME|NEW, places moved)
    dropped: List[str] = field(default_factory=list)  # gmap_ids in the base top 10 but not the stance top 10
    notices: List[str] = field(default_factory=list)


def rank_changes(base_views, stance_views):
    """({gmap_id: (kind, places)} for the stance column, [gmap_ids dropped from the base top 10]).

    kind is UP/DOWN with the number of places moved, SAME, or NEW (not in the base top 10).
    """
    base_rank = {v.gmap_id: v.rank for v in base_views}
    changes = {}
    for v in stance_views:
        old = base_rank.get(v.gmap_id)
        changes[v.gmap_id] = (NEW, 0) if old is None else (UP, old - v.rank) if v.rank < old else \
            (DOWN, v.rank - old) if v.rank > old else (SAME, 0)
    shown = {v.gmap_id for v in stance_views}
    return changes, [v.gmap_id for v in base_views if v.gmap_id not in shown]


def run_compare(query, borough_choice, *, base_retriever, stance_retriever, client, model, setup_error=None,
                include_low_evidence=False, demo_mode=False, cache_dir=None, **log):
    """Both rankings for one query as a Comparison; `log` (e.g. log_path) goes to pipeline.run_search."""
    common = dict(client=client, model=model, setup_error=setup_error, include_low_evidence=include_low_evidence,
                  auto_summaries=0, **log)
    base, _ = demo_cache.search_with_fallback(query, borough_choice, BASE_METHOD, demo_mode=demo_mode,
                                              cache_dir=cache_dir, retriever=base_retriever, **common)
    comparison = Comparison(base, None, notices=list(base.notices))
    if stance_retriever is None:
        comparison.notices.append("The stance re-ranker is not available, so only semantic_capped is shown.")
        return comparison
    stance, _ = pipeline.run_search(query, borough_choice, STANCE_METHOD, retriever=stance_retriever, prior=base,
                                    search_options={"cache_only": True} if demo_mode else None, **common)
    if stance.ranking_fallback or not stance.results:
        why = next((n for n in stance.notices if n.startswith("Could not label")), "")
        comparison.notices.append("The stance ranking is unavailable, so only semantic_capped is shown. " + why)
        return comparison
    comparison.stance = stance
    comparison.changes, comparison.dropped = rank_changes(base.results, stance.results)
    return comparison
