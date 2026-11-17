# Deploying on Streamlit Community Cloud

The app runs from `deploy/streamlit_cloud_app.py`. It fetches the search data (about 1.7 GB, a private Hugging Face dataset) on the first
start, then draws the normal app. Locally nothing changes: `streamlit run app/streamlit_app.py` works as before and needs no secrets.

## Before you start
- The private dataset `Pujitha8199/nyc-restaurant-data` holds a top-level `processed/` folder (`review_chunks.jsonl`, `reviews_clean.jsonl`,
  `reviews_clean_index.npz`, `semantic/{embeddings.npy, meta.jsonl, chunk_restaurant.npy, chunk_offsets.npy, info.json}`,
  optional `cleaning_stats.json` and `stance_rerank/label_cache.jsonl`). The files must be byte-identical to the ones the index was built from.
- A Hugging Face **read** token that can read that dataset (Settings > Access Tokens). A fine-grained token limited to the one dataset is enough.
- A Groq API key.
- This branch merged to `main` (or any branch you choose in step 2).

## Steps on share.streamlit.io
1. **Create app**: "Create app" > "Deploy a public app from GitHub".
2. **Repository**: the GitHub repository of this project. **Branch**: `main` (after the merge).
3. **Main file path**: `deploy/streamlit_cloud_app.py`.
4. **Advanced settings > Python version**: **3.11**. (It is the version `requirements.txt` was resolved and tested with; scikit-learn is pinned
   to 1.6.1 because the TF-IDF indexes are pickles built with it. Do not leave this on a newer default.)
5. **Advanced settings > Secrets**: paste the TOML below with your real values.
6. **Deploy**. The first start installs the packages, then shows "Loading data, first start takes a few minutes ..." while the dataset downloads.

### Secrets (TOML)
```toml
HF_TOKEN = "PLACEHOLDER_huggingface_read_token"
GROQ_API_KEY = "PLACEHOLDER_groq_api_key"
GROQ_MODEL = "PLACEHOLDER_model_id"                   # for example the model you tested with
GROQ_FALLBACK_MODEL = "PLACEHOLDER_optional_second_model_id"   # optional: leave out to have no fallback
```
`GROQ_FALLBACK_MODEL` is tried once after a rate-limit, timeout, network or provider error of the main model, and must support structured output
(see `src/llm/groq_client.py`); a daily token limit is never retried. Never put real values in the repository: `.env` and `.streamlit/secrets.toml`
are git-ignored. The app copies these four secrets into the environment and never prints or logs them.

## What the entry file does
- Sets `OMP_NUM_THREADS=1` and `NYC_LOG_REQUESTS=0` as defaults (before torch is imported; no request log on a hosted app).
- Copies the four secrets into the environment, calls `st.set_page_config` (the first Streamlit call, made once), then prepares the data once per
  server process (`deploy/space_data.py`): it skips the download if `NYC_DATA_DIR` already holds the files or `<repo>/data/processed` does, otherwise
  downloads into `~/nyc_data` (or the temp folder if the home folder is not writable) and sets `NYC_DATA_DIR` **before** `app/` and `src/` are imported.
- If the data cannot be fetched or is incomplete it shows an error naming the missing file (never the token) and stops; the next page load tries again.
- The saved demo results (`data/demo_cache/`) come from the repository, and the downloaded folder never shadows them.

## Sleep and re-download
These are the platform's rules as I know them from its documentation, not something tested here; check the Community Cloud docs for the current numbers.
- An app with no visitors for about 12 hours is put to sleep; the next visitor wakes it, which re-runs the start (installs are cached, the app restarts).
- The container's disk is not persistent. After a sleep or restart the dataset is downloaded again (a few minutes) and the search index is loaded again
  on the first typed query. While the app stays awake the data and the loaded index stay in place.
- The chips (saved results) do **not** load the search index, so they answer immediately even right after a wake-up, once the data is in place.

## Memory (measured on the real data, CPU torch, `OMP_NUM_THREADS=1`; limit 2.7 GB)
| Situation | Resident memory |
|---|---|
| Home page rendered | 233 MiB |
| Chip answered from its saved result (the index is not loaded) | 234 MiB |
| Search index load (a typed query's first step) | peaks at 1,264 MiB (float16 file and float32 copy alive together), then about 0.9 GiB |
| Typed query: index + embedding model + retrieval + evidence for the top 10 | **peak 1,751 MiB** (1.75 GiB, `/usr/bin/time -l`) |
| Streamlit server overhead on top | about 0.2 GB (measured earlier on the first screen) |

So a typed query needs about 2 GB in total, roughly 0.7 GB under the 2.7 GB limit. Compare rankings loads a second retriever object over the same
index files (the saved stance labels), so expect somewhat more; that was not measured.

## Risks to check on the first deploy
- **CPU torch.** `requirements.txt` starts with `--extra-index-url https://download.pytorch.org/whl/cpu`. pip then prefers the `+cpu` wheel, but if the
  platform installs with `uv` (default "first index wins") it may take the much larger CUDA build of torch from PyPI. If the build log shows `nvidia-*`
  packages or the install is far above 1 GB, change the torch line to `torch==<version>+cpu ; sys_platform == "linux"` with a second line
  `torch ; sys_platform != "linux"` (use the version the log shows). This could not be tested here.
- **Disk.** The download is about 1.7 GB; check that the platform's disk allows it.
- **Cold start time.** Install, download and index load add up to a few minutes; the first visitor waits.
