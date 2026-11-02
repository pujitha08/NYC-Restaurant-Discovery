"""Placeholder Streamlit app. Run with: streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit puts app/ on sys.path, not the repo root, so add it for `src` imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BOROUGHS

st.set_page_config(page_title="NYC Restaurant Discovery", page_icon="🍽️")

st.title("NYC Restaurant Discovery")
st.caption(f"Currently covering: {', '.join(BOROUGHS)}")

query = st.text_input("Describe what you're looking for")

if st.button("Search") and query:
    st.info("Search is not implemented yet.")
