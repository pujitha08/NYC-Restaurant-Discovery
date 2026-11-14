"""Central project configuration.

Change BOROUGHS here to widen or narrow the geographic scope of the project.
"""

from pathlib import Path

# Boroughs included in the pipeline. To expand, add e.g. "Queens", "Bronx",
# "Staten Island".
BOROUGHS = ["Manhattan", "Brooklyn"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Stance-aware re-ranking (src/retrieval/stance_rerank.py): re-rank the top `top_n` restaurants of
# semantic_capped as  base_norm + w_pos * accommodation_share - w_neg * negative_share
# (minus w_strong_neg if an allergy query has any negative sentence). Weights are tuned on the dev split only
# (eval/tune_stance_rerank.py); the initial values are w_pos=0.3, w_neg=0.6.
# PRE-REGISTERED (2026-11-13, before any test-split stance result was computed or looked at): the candidate depth is
# top_n = 20 for BOTH weight tuning (dev queries) and the final held-out test evaluation. It was fixed for judging
# cost (the judge pool is the top-20 of semantic_capped) and is not tuned; eval/tune_stance_rerank.py has no depth flag.
STANCE_RERANK = {"top_n": 20, "sentences_per_restaurant": 4, "w_pos": 0.3, "w_neg": 0.6, "w_strong_neg": 0.1}
