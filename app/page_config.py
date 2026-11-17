"""The page configuration of the app, in a module that imports nothing from src/.

deploy/streamlit_cloud_app.py applies it itself, as the first Streamlit call, so it can show a "loading data" message before the app
starts; app/streamlit_app.py applies it when it is run directly (the usual way). Importing this module must not read NYC_DATA_DIR,
which src/config.py does at import, so it stays free of src imports.
"""

PAGE_CONFIG = dict(page_title="NYC Restaurant Discovery", page_icon="🍽️", layout="wide", initial_sidebar_state="collapsed")
