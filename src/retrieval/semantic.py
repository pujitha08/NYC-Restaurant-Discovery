"""Semantic retrieval: embed review chunks with all-MiniLM-L6-v2, then aggregate per restaurant.

A drop-in alternative to ChunkTfidfRetriever: same search()/info() interface and the same
top-3-chunks-plus-density aggregation (shared code in tfidf_chunks), with cosine similarity of
normalized sentence embeddings instead of TF-IDF.

Build the index (about 18 min on Apple-silicon MPS for 605,540 chunks; not done automatically):
    python -m src.retrieval.semantic --build
Search:
    python -m src.retrieval.semantic "vegan options for a group" --borough Brooklyn
"""

import argparse
import datetime
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.retrieval.evidence import _SENTENCE_RE, MAX_WORDS, N_SNIPPETS, REVIEWS_PATH, ReviewStore
from src.retrieval.tfidf import DOCS_PATH, PROCESSED, print_results
from src.retrieval.tfidf_chunks import (CHUNKS_PATH, DENSITY_MODES, aggregate_chunks,
                                        restaurant_chunks)
from src.schemas import RetrievalResult, Snippet

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = PROCESSED / "semantic"
EMBEDDINGS_FILE = "embeddings.npy"  # (n_chunks, dim) float16, rows L2-normalized
OWNER_FILE = "chunk_restaurant.npy"  # restaurant row (index into meta) of each chunk
OFFSETS_FILE = "chunk_offsets.npy"  # byte offset of each chunk in review_chunks.jsonl
META_FILE = "meta.jsonl"  # one dict per restaurant (row of restaurant_docs.jsonl, minus text)
INFO_FILE = "info.json"  # model, counts, build date and input-file signatures

METHOD = "embedding"
# A chunk "matches" the query (counts as a hit: eligible for the top-3 sum, the density bonus
# and evidence) when its cosine similarity is above this value. TF-IDF scores are 0 for chunks
# without a query term; embedding cosines are almost never 0, so a floor is needed. On a 5,000
# chunk sample, 0.40 is about the 99th percentile for "vegan options for a group" and leaves
# hundreds of matching chunks corpus-wide even for the sparsest test query ("shellfish allergy",
# whose 99.9th percentile is 0.37). It is query-dependent: generic queries ("halal chinese food",
# median 0.25) pass ~6% of chunks, so the density bonus favors large restaurants more for them.
MATCH_THRESHOLD = 0.40
# Review snippets: the best-matching review is always returned; further ones only above this.
SNIPPET_THRESHOLD = 0.30
BUILD_BATCH = 4096  # chunks per progress-bar step (the model batches internally)
ENCODE_BATCH = 128


class CacheMismatchError(RuntimeError):
    """The saved embeddings do not belong to the current chunks/docs files or model."""


def default_device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


class SentenceTransformerEmbedder:
    """Lazily loaded sentence-transformers model; encode() returns L2-normalized float32 rows."""

    def __init__(self, model_name=MODEL_NAME, device=None):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self.device = self.device or default_device()
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self):
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=ENCODE_BATCH):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def file_signature(path, block=1 << 23):
    """{"size", "sha256"} of a file, read in 8 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return {"size": Path(path).stat().st_size, "sha256": h.hexdigest()}


def peak_rss_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9  # bytes on macOS, KB on Linux


def excerpt_around(sentences, best, max_words=MAX_WORDS):
    """Neighbouring sentences around sentences[best], up to max_words words, with ellipses."""
    lengths = [len(s.split()) for s in sentences]
    lo = hi = best
    total = lengths[best]
    grew = True
    while grew:
        grew = False
        for j in (lo - 1, hi + 1):  # previous sentence first, then next
            if 0 <= j < len(sentences) and total + lengths[j] <= max_words:
                lo, hi, total, grew = min(lo, j), max(hi, j), total + lengths[j], True
    words = " ".join(sentences[lo:hi + 1]).split()
    if len(words) > max_words:  # one sentence over the limit: keep its start
        return ("… " if lo > 0 else "") + " ".join(words[:max_words]) + " …"
    return ("… " if lo > 0 else "") + " ".join(words) + (" …" if hi < len(sentences) - 1 else "")


class SemanticRetriever:
    def __init__(self, embeddings, owner, offsets, meta, embedder=None, chunks_path=CHUNKS_PATH,
                 reviews_path=REVIEWS_PATH, review_index_path=None, density_mode="log"):
        if density_mode not in DENSITY_MODES:
            raise ValueError(f"density_mode must be one of {DENSITY_MODES}, got {density_mode!r}")
        self.density_mode = density_mode
        self.embeddings = embeddings  # (n_chunks, dim) float32, rows L2-normalized
        self.owner = owner
        self.offsets = offsets
        self.meta = meta
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.chunks_path = Path(chunks_path)
        reviews_path = Path(reviews_path)
        if review_index_path is None and reviews_path != Path(REVIEWS_PATH):
            review_index_path = reviews_path.with_name(reviews_path.stem + "_index.npz")
        self.store = ReviewStore(reviews_path, review_index_path) if review_index_path else ReviewStore(reviews_path)
        self._by_id = {m["gmap_id"]: i for i, m in enumerate(meta)}
        self._boroughs = np.array([m["borough"].lower() for m in meta])
        self._low_evidence = np.array([m["low_evidence"] for m in meta], dtype=bool)

    # ---- build / save / load -------------------------------------------
    @classmethod
    def build(cls, docs_path=DOCS_PATH, chunks_path=CHUNKS_PATH, embedder=None, reviews_path=REVIEWS_PATH,
              review_index_path=None, progress=True):
        embedder = embedder or SentenceTransformerEmbedder()
        meta, row = [], {}
        with open(docs_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                d.pop("text")
                row[d["gmap_id"]] = len(meta)
                meta.append(d)
        texts, offsets, owner = [], [], []
        with open(chunks_path, "rb") as f:
            pos = 0
            for line in f:
                c = json.loads(line)
                offsets.append(pos)
                pos += len(line)
                texts.append(c["text"])
                owner.append(row[c["gmap_id"]])
        emb = None
        steps = range(0, len(texts), BUILD_BATCH)
        for start in tqdm(steps, unit="batch", disable=not progress, file=sys.stdout,
                          desc=f"Embedding {len(texts):,} chunks ({BUILD_BATCH}/batch)"):
            part = embedder.encode(texts[start:start + BUILD_BATCH])
            if emb is None:
                emb = np.empty((len(texts), part.shape[1]), dtype=np.float16)
            emb[start:start + len(part)] = part
        if emb is None:
            emb = np.zeros((0, 1), dtype=np.float16)
        retriever = cls(emb.astype(np.float32), np.array(owner, dtype=np.int32), np.array(offsets, dtype=np.int64),
                        meta, embedder, chunks_path, reviews_path, review_index_path)
        retriever._emb16 = emb  # what save() writes; search uses the same float16-rounded values
        return retriever

    def save(self, index_dir=INDEX_DIR, docs_path=DOCS_PATH, extra_info=None):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        emb16 = getattr(self, "_emb16", None)
        np.save(index_dir / EMBEDDINGS_FILE, emb16 if emb16 is not None else self.embeddings.astype(np.float16))
        np.save(index_dir / OWNER_FILE, self.owner)
        np.save(index_dir / OFFSETS_FILE, self.offsets)
        with open(index_dir / META_FILE, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        info = {"model": self.embedder.model_name, "dim": int(self.embeddings.shape[1]),
                "chunk_count": int(len(self.owner)), "restaurant_count": len(self.meta),
                "build_date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "chunks_file": file_signature(self.chunks_path), "docs_file": file_signature(docs_path),
                "match_threshold": MATCH_THRESHOLD, **(extra_info or {})}
        (index_dir / INFO_FILE).write_text(json.dumps(info, indent=2))
        return info

    @classmethod
    def exists(cls, index_dir=INDEX_DIR):
        names = (EMBEDDINGS_FILE, OWNER_FILE, OFFSETS_FILE, META_FILE, INFO_FILE)
        return all((Path(index_dir) / n).exists() for n in names)

    @classmethod
    def load(cls, index_dir=INDEX_DIR, chunks_path=CHUNKS_PATH, docs_path=DOCS_PATH, reviews_path=REVIEWS_PATH,
             review_index_path=None, embedder=None, density_mode="log"):
        """Load a saved index; raise CacheMismatchError if it does not match the inputs or model."""
        index_dir = Path(index_dir)
        embedder = embedder or SentenceTransformerEmbedder()
        info = json.loads((index_dir / INFO_FILE).read_text())
        problems = []
        if info.get("model") != embedder.model_name:
            problems.append(f"model {info.get('model')!r} != {embedder.model_name!r}")
        for key, path in (("chunks_file", chunks_path), ("docs_file", docs_path)):
            if info.get(key) != file_signature(path):
                problems.append(f"{Path(path).name} changed since the build ({key} size/sha256 differ)")
        if problems:
            raise CacheMismatchError(f"Semantic index in {index_dir} is stale: " + "; ".join(problems)
                                     + ". Rebuild with: python -m src.retrieval.semantic --build")
        emb = np.load(index_dir / EMBEDDINGS_FILE)
        owner, offsets = np.load(index_dir / OWNER_FILE), np.load(index_dir / OFFSETS_FILE)
        with open(index_dir / META_FILE, encoding="utf-8") as f:
            meta = [json.loads(line) for line in f]
        if not (len(emb) == len(owner) == len(offsets) == info["chunk_count"]) or len(meta) != info["restaurant_count"]:
            raise CacheMismatchError(f"Semantic index in {index_dir} is inconsistent (array lengths differ "
                                     "from info.json). Rebuild with: python -m src.retrieval.semantic --build")
        return cls(emb.astype(np.float32), owner, offsets, meta, embedder, chunks_path, reviews_path,
                   review_index_path, density_mode)

    # ---- search ---------------------------------------------------------
    def _mode(self, density_mode):
        density_mode = density_mode or self.density_mode
        if density_mode not in DENSITY_MODES:
            raise ValueError(f"density_mode must be one of {DENSITY_MODES}, got {density_mode!r}")
        return density_mode

    def _allowed(self, boroughs, include_low_evidence):
        allowed = np.ones(len(self.meta), dtype=bool)
        if boroughs:
            allowed &= np.isin(self._boroughs, [b.lower() for b in boroughs])
        if not include_low_evidence:
            allowed &= ~self._low_evidence
        return allowed

    def _scores(self, qvec, boroughs, include_low_evidence, density_mode):
        chunk_scores = self.embeddings @ qvec  # cosine: rows and query are L2-normalized
        chunk_scores = np.where(self._allowed(boroughs, include_low_evidence)[self.owner], chunk_scores, -1.0)
        return aggregate_chunks(chunk_scores, self.owner, len(self.meta), density_mode,
                                hit_threshold=MATCH_THRESHOLD, density_threshold=MATCH_THRESHOLD)

    def restaurant_scores(self, query_terms, boroughs=None, include_low_evidence=False, density_mode=None):
        """Aggregated score of every restaurant (0 where filtered out or without matching chunks)."""
        qvec = self.embedder.encode([query_terms])[0]
        return self._scores(qvec, boroughs, include_low_evidence, self._mode(density_mode))[0]

    def search(self, query_terms, boroughs=None, top_k=10, include_low_evidence=False, density_mode=None):
        """Rank restaurants by aggregated chunk similarity; low_evidence ones are skipped by default.

        Ties are broken by restaurant row, so the order is deterministic.
        """
        if not isinstance(query_terms, str) or not query_terms.strip():
            return []
        qvec = self.embedder.encode([query_terms])[0]
        scores, hit, owners = self._scores(qvec, boroughs, include_low_evidence, self._mode(density_mode))
        rows = np.flatnonzero(scores > 0)
        rows = rows[np.lexsort((rows, -scores[rows]))][:top_k]
        chunk_ids = [restaurant_chunks(hit, owners, i) for i in rows]
        evidence = self._evidence(rows, chunk_ids, qvec)
        return [RetrievalResult(gmap_id=self.meta[i]["gmap_id"], score=float(scores[i]), method=METHOD,
                                evidence_snippets=ev) for i, ev in zip(rows, evidence)]

    def _review_records(self, row, chunk_ids, f):
        """Reviews of the given chunks (each once, chunk order), restricted to restaurant `row`."""
        review_ids = []
        for cid in chunk_ids:
            f.seek(self.offsets[cid])
            review_ids += json.loads(f.readline())["review_ids"]
        gmap_id = self.meta[row]["gmap_id"]
        unique_ids = list(dict.fromkeys(review_ids))  # a long review split over chunks appears once
        return [r for r in self.store.get(unique_ids) if r["gmap_id"] == gmap_id]

    def _evidence(self, rows, chunk_ids, qvec, n=N_SNIPPETS, max_words=MAX_WORDS):
        """Per result, the n reviews (from its own top chunks) most similar to the query, as Snippets.

        All candidate reviews of all results are embedded in one batch; long winners are then cut
        to the max_words window around their most similar sentence (a second, smaller batch).
        """
        with open(self.chunks_path, "rb") as f:
            candidates = [self._review_records(row, ids, f) for row, ids in zip(rows, chunk_ids)]
        flat = [r for recs in candidates for r in recs]
        sims = self.embedder.encode([r["text"] for r in flat]) @ qvec if flat else np.zeros(0)

        chosen, pos = [], 0
        for recs in candidates:
            s = sims[pos:pos + len(recs)]
            pos += len(recs)
            order = sorted(range(len(recs)), key=lambda j: (-s[j], recs[j]["review_id"]))
            keep = [j for rank, j in enumerate(order[:n]) if rank == 0 or s[j] >= SNIPPET_THRESHOLD]
            chosen.append([recs[j] for j in keep])

        long_reviews = [r for picks in chosen for r in picks if max_words and len(r["text"].split()) > max_words]
        split = {id(r): [s for s in _SENTENCE_RE.split(r["text"]) if s.strip()] for r in long_reviews}
        sentences = [s for r in long_reviews for s in split[id(r)]]
        sent_sims = self.embedder.encode(sentences) @ qvec if sentences else np.zeros(0)
        best_sentence, pos = {}, 0
        for r in long_reviews:
            k = len(split[id(r)])
            best_sentence[id(r)] = int(np.argmax(sent_sims[pos:pos + k]))
            pos += k

        out = []
        for picks in chosen:
            out.append([Snippet(text=excerpt_around(split[id(r)], best_sentence[id(r)], max_words)
                                if id(r) in best_sentence else r["text"],
                                review_id=r["review_id"], time=r["time"]) for r in picks])
        return out

    def info(self, gmap_id):
        return self.meta[self._by_id[gmap_id]]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Semantic (embedding) restaurant search over review chunks")
    ap.add_argument("query", nargs="?", help="query text, e.g. 'vegan options for a group'")
    ap.add_argument("--borough", action="append", help="restrict to a borough (repeat for several)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--include-low-evidence", action="store_true",
                    help="also rank restaurants with fewer than 5 clean reviews")
    ap.add_argument("--density-mode", choices=DENSITY_MODES, default="log",
                    help="bonus for restaurants with many matching chunks (default: log)")
    ap.add_argument("--build", action="store_true", help="(re)build and save the embedding index (slow)")
    args = ap.parse_args(argv)

    if args.build:
        t = time.time()
        embedder = SentenceTransformerEmbedder()
        print(f"Building semantic index with {MODEL_NAME} on {embedder.model.device} ...", flush=True)
        retriever = SemanticRetriever.build(embedder=embedder)
        build_s = time.time() - t
        info = retriever.save(extra_info={"build_seconds": round(build_s, 1), "device": str(embedder.model.device),
                                          "peak_rss_gb": round(peak_rss_gb(), 2)})
        retriever.density_mode = args.density_mode
        size_mb = sum(p.stat().st_size for p in INDEX_DIR.iterdir()) / 1e6
        print(f"Built in {build_s:.0f}s ({info['chunk_count'] / build_s:,.0f} chunks/s): {info['chunk_count']:,} "
              f"chunks x {info['dim']} dims; {size_mb:,.0f} MB in {INDEX_DIR}; peak RSS {peak_rss_gb():.1f} GB")
    elif not SemanticRetriever.exists():
        sys.exit(f"No semantic index in {INDEX_DIR}. Build it first (about 18 min on MPS): "
                 "python -m src.retrieval.semantic --build")
    else:
        t = time.time()
        try:
            retriever = SemanticRetriever.load(density_mode=args.density_mode)
        except CacheMismatchError as e:
            sys.exit(str(e))
        retriever.embedder.encode(["warm up"])  # load the model before timing the query
        print(f"Loaded index and model in {time.time() - t:.1f}s")
    if not args.query:
        return
    t = time.time()
    results = retriever.search(args.query, boroughs=args.borough, top_k=args.top_k,
                               include_low_evidence=args.include_low_evidence)
    query_s = time.time() - t
    if not results:
        print("No matches.")
    print_results(retriever, results)
    print(f"Query time: {query_s:.2f}s")


if __name__ == "__main__":
    main()
