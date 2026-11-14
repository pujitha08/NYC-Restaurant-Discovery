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
STANCE_RERANK = {"top_n": 30, "sentences_per_restaurant": 4, "w_pos": 0.3, "w_neg": 0.6, "w_strong_neg": 0.1}
