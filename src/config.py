"""Central project configuration.

Change BOROUGHS here to widen or narrow the geographic scope of the project.
"""

import os
from pathlib import Path

# Boroughs included in the pipeline. To expand, add e.g. "Queens", "Bronx",
# "Staten Island".
BOROUGHS = ["Manhattan", "Brooklyn"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_data_dir(environ=None):
    """The data directory: $NYC_DATA_DIR if it is set and not blank (e.g. a downloaded cache dir on a hosted Space),
    else <repo>/data. Every module that reads or writes under data/ builds its paths from DATA_DIR."""
    raw = (os.environ if environ is None else environ).get("NYC_DATA_DIR", "").strip()
    return Path(raw).expanduser() if raw else PROJECT_ROOT / "data"


DATA_DIR = resolve_data_dir()

# Stance-aware re-ranking (src/retrieval/stance_rerank.py): re-rank the top `top_n` restaurants of
# semantic_capped as  base_norm + w_pos * accommodation_share - w_neg * negative_share
# (minus w_strong_neg if an allergy query has any negative sentence). Weights are tuned on the dev split only
# (eval/tune_stance_rerank.py); the initial values are w_pos=0.3, w_neg=0.6.
# PRE-REGISTERED (2026-11-13, before any test-split stance result was computed or looked at): the candidate depth is
# top_n = 20 for BOTH weight tuning (dev queries) and the final held-out test evaluation. It was fixed for judging
# cost (the judge pool is the top-20 of semantic_capped) and is not tuned; eval/tune_stance_rerank.py has no depth flag.
# Which stance labeler the re-ranker, the tuner, the labeling script and the app use: "llm" (Groq, labels cached on disk) or
# "classifier" (the model trained by src/classifier/stance.py). Whether data/processed/stance/stance.joblib exists never
# changes this: an evaluation must not switch labelers because a file appeared.
STANCE_LABELER = "llm"
STANCE_RERANK = {"top_n": 20, "sentences_per_restaurant": 4, "w_pos": 0.3, "w_neg": 0.6, "w_strong_neg": 0.1}
