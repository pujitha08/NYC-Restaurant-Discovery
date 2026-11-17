"""Fetch and check the search data for a hosted deployment (Streamlit Community Cloud), before the app starts.

`prepare(folder)` downloads the private dataset into the folder (skipping the download when the files are already there with the
expected sizes), checks that every file the semantic search needs exists, and says what is wrong if not. `ensure_data` adds the
"already have data" rules of deploy/streamlit_cloud_app.py. It imports nothing from app/ or src/ (they read NYC_DATA_DIR when they are
first imported, so the variable must be set by then), and it never prints the access token. `python -m deploy.space_data prepare
<folder>` runs it from a shell.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ID = "Pujitha8199/nyc-restaurant-data"
REPO_TYPE = "dataset"
# What the semantic search cannot start without: relative path -> expected size in bytes (the files of the dataset as uploaded).
REQUIRED = {
    "processed/review_chunks.jsonl": 505_962_950,
    "processed/reviews_clean.jsonl": 705_310_508,
    "processed/reviews_clean_index.npz": 34_503_618,
    "processed/semantic/embeddings.npy": 465_054_848,
    "processed/semantic/meta.jsonl": 4_706_757,
    "processed/semantic/chunk_restaurant.npy": 2_422_288,
    "processed/semantic/chunk_offsets.npy": 4_844_448,
    "processed/semantic/info.json": 519,
}
# Used when present but not needed to start: the home page numbers, and the saved stance labels (the app appends to them, so their
# size is not checked).
OPTIONAL = ("processed/cleaning_stats.json", "processed/stance_rerank/label_cache.jsonl")


class DataIncomplete(Exception):
    """The data folder lacks a file the app needs (the message names it)."""


def missing_files(data_dir):
    return [name for name in REQUIRED if not (Path(data_dir) / name).is_file()]


def verify(data_dir):
    """Raise DataIncomplete naming every missing required file, or a chunk file that is not the one the index was built from."""
    data_dir = Path(data_dir)
    missing = missing_files(data_dir)
    if missing:
        raise DataIncomplete(f"The search data is incomplete: missing {', '.join(missing)} in {data_dir}. "
                             f"Check that the dataset {REPO_ID} holds them under processed/ and that the download finished.")
    info = json.loads((data_dir / "processed/semantic/info.json").read_text(encoding="utf-8"))
    expected = (info.get("chunks_file") or {}).get("size")
    actual = (data_dir / "processed/review_chunks.jsonl").stat().st_size
    if expected is not None and expected != actual:
        raise DataIncomplete(f"processed/review_chunks.jsonl is {actual:,} bytes but the semantic index was built from a file of "
                             f"{expected:,} bytes: the upload is not byte-identical (or the download is truncated). "
                             "The app would refuse to load the index (its sha256 check).")


def needs_download(data_dir, expected=None):
    """True unless every required file is there with its expected size (a warm restart then skips the download)."""
    data_dir = Path(data_dir)
    for name, size in (expected or REQUIRED).items():
        path = data_dir / name
        if not path.is_file() or path.stat().st_size != size:
            return True
    return False


def shield_demo_cache(data_dir):
    """The app uses <data dir>/demo_cache instead of the saved results in the repo when that folder exists, so a downloaded folder of
    that name is moved aside. Returns the new name, or None."""
    folder = Path(data_dir) / "demo_cache"
    if not folder.exists():
        return None
    aside = Path(data_dir) / "demo_cache.from_dataset"
    if aside.exists():
        shutil.rmtree(aside, ignore_errors=True)
    folder.rename(aside)
    return aside


def scrub(text, token):
    return str(text).replace(token, "<token>") if token else str(text)


def download(data_dir, token):
    """snapshot_download the dataset's processed/ folder into data_dir (only that folder: no demo_cache, nothing else)."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=REPO_ID, repo_type=REPO_TYPE, local_dir=str(data_dir), allow_patterns=["processed/**"],
                      token=token or None)


def prepare(data_dir, environ=None, downloader=download, out=print):
    """Download if needed, verify, shield the repo's demo_cache. Returns 0 when the app can start, else 1 (the reason is printed)."""
    environ = os.environ if environ is None else environ
    token = (environ.get("HF_TOKEN") or "").strip()  # a Space secret; never printed
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if needs_download(data_dir):
        out(f"Downloading the search data from {REPO_ID} into {data_dir} (about 1.7 GB; this takes a few minutes) ...")
        if not token:
            out("Note: HF_TOKEN is not set. A private dataset needs it as a Space secret.")
        try:
            downloader(data_dir, token)
        except Exception as e:  # the message may quote a URL, never the token; scrub it anyway
            out(f"Could not download {REPO_ID}: {type(e).__name__}: {scrub(e, token)[:400]}")
            out("Check that HF_TOKEN (a Space secret) can read the dataset and that the dataset name is right.")
            return 1
    else:
        out(f"The search data is already in {data_dir}; skipping the download.")
    try:
        verify(data_dir)
    except DataIncomplete as e:
        out(f"ERROR: {e}")
        return 1
    aside = shield_demo_cache(data_dir)
    if aside:
        out(f"Moved the dataset's demo_cache aside ({aside.name}): the app uses the saved results shipped with the code.")
    out(f"The search data is ready: NYC_DATA_DIR={data_dir}")
    return 0


SECRET_KEYS = ("HF_TOKEN", "GROQ_API_KEY", "GROQ_MODEL", "GROQ_FALLBACK_MODEL")


def export_secrets(secrets, environ, keys=SECRET_KEYS):
    """Copy the deployment's secrets into the environment (where the Groq client and the download read them) when they are present
    and the variable is not already set. Returns the names copied; values are never printed."""
    copied = []
    for key in keys:
        try:
            value = secrets[key] if key in secrets else None
        except Exception:  # no secrets file / not readable: nothing to copy
            value = None
        if value is not None and str(value).strip() and not (environ.get(key) or "").strip():
            environ[key] = str(value).strip()
            copied.append(key)
    return copied


def default_data_dir(home=None):
    """A writable folder for the download: ~/nyc_data, or the system temp folder if the home folder is not writable."""
    for base in (Path(home) if home else Path.home(), Path(tempfile.gettempdir())):
        folder = base / "nyc_data"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".write_test"
            probe.write_text("x")
            probe.unlink()
            return folder
        except OSError:
            continue
    return Path(tempfile.gettempdir()) / "nyc_data"


def ensure_data(environ, repo_root, default_dir=None, downloader=download, out=print):
    """Decide where the data is and download it only if it is not already here. Returns (folder to set as NYC_DATA_DIR or None, exit code).

    1. NYC_DATA_DIR is set and that folder already holds the files: use it, no download.
    2. NYC_DATA_DIR is not set and <repo>/data already holds the files (a local clone with the data): return None (the app's default
       folder is right), no download.
    3. Otherwise download into NYC_DATA_DIR if it is set, else into default_dir (a writable folder), and check the result.
    Files present but not matching what the index was built from are an error (not a silent re-download over someone's data).
    """
    explicit = (environ.get("NYC_DATA_DIR") or "").strip()
    local = Path(repo_root) / "data"
    folder, export = (Path(explicit), Path(explicit)) if explicit else (local, None)  # export None: the app's default folder is right
    if not missing_files(folder):
        try:
            verify(folder)
        except DataIncomplete as e:
            out(f"ERROR: {e}")
            return None, 1
        out(f"The search data is already in {folder}; no download.")
        return export, 0
    target = Path(explicit) if explicit else Path(default_dir or default_data_dir())
    code = prepare(target, environ, downloader, out)
    return (target if code == 0 else None), code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] != "prepare":
        print("usage: python -m deploy.space_data prepare <data folder>", file=sys.stderr)
        return 2
    return prepare(argv[1])


if __name__ == "__main__":
    sys.exit(main())
