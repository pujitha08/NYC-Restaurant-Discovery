"""Main file for Streamlit Community Cloud (set it as the app's "Main file path"): fetch the search data once, then run the app.

Locally nothing changes: run `streamlit run app/streamlit_app.py` as before; this file is not needed there.

Order matters. The environment defaults and the secrets are set first; st.set_page_config (the first Streamlit call, made once, here)
comes next so the page can say that the data is loading; the data is fetched and NYC_DATA_DIR is exported before app/ or src/ is imported
(src/config.py reads it at import); only then is the app drawn, with configure_page=False because the page is already configured.
Nothing here prints or logs a secret.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before torch is imported: avoids a crash of the CPU build on some hosts
os.environ.setdefault("NYC_LOG_REQUESTS", "0")  # a hosted app keeps no request log

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.page_config import PAGE_CONFIG  # noqa: E402  (imports nothing from src/)
from deploy import space_data  # noqa: E402  (imports nothing from app/ or src/)

st.set_page_config(**PAGE_CONFIG)  # the first Streamlit call, made once; the app is told not to repeat it

try:
    space_data.export_secrets(st.secrets, os.environ)  # HF_TOKEN, GROQ_API_KEY, GROQ_MODEL, GROQ_FALLBACK_MODEL, if present
except Exception:  # no secrets configured: the environment may still hold them
    pass


@st.cache_resource(show_spinner="Loading data, first start takes a few minutes ...")
def prepare_data():
    """Once per server process: make sure the search data is available and point NYC_DATA_DIR at it. A failure raises, so it is not
    cached and the next page load tries again."""
    lines = []
    folder, code = space_data.ensure_data(os.environ, ROOT, out=lines.append)
    if code:
        raise RuntimeError("\n".join(lines))
    if folder is not None:
        os.environ["NYC_DATA_DIR"] = str(folder)  # before app/ and src/ are imported below
    return str(folder) if folder is not None else None


try:
    prepare_data()
except Exception as e:
    token = os.environ.get("HF_TOKEN", "")
    st.error("The search data is not available, so the app cannot start.\n\n" + space_data.scrub(e, token)
             + "\n\nCheck the Space secrets (HF_TOKEN must be able to read the dataset) and reload the page.")
    st.stop()

from app import streamlit_app  # noqa: E402  (first import of app/ and src/: NYC_DATA_DIR is final)

streamlit_app.main(configure_page=False)
