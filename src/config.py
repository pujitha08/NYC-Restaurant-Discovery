"""Central project configuration.

Change BOROUGHS here to widen or narrow the geographic scope of the project.
"""

from pathlib import Path

# Boroughs included in the pipeline. To expand, add e.g. "Queens", "Bronx",
# "Staten Island".
BOROUGHS = ["Manhattan", "Brooklyn"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
