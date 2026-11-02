"""Build restaurant documents and review chunks from the cleaned reviews.

Run from the repo root (after clean_reviews):
    python -m src.data.build_docs

Writes restaurant_docs.jsonl (RestaurantDoc, one per restaurant) and
review_chunks.jsonl (Chunk). Output records use `gmap_id`.

Needs the all-MiniLM-L6-v2 tokenizer (fetched with huggingface_hub, which
sentence-transformers installs) to guarantee the token limit.
"""

import dataclasses
import json
import re
from collections import Counter

import numpy as np
from tqdm import tqdm

from src.config import DATA_DIR
from src.schemas import Chunk, RestaurantDoc

PROCESSED = DATA_DIR / "processed"
REVIEWS_PATH = PROCESSED / "reviews_clean.jsonl"
RESTAURANTS_PATH = PROCESSED / "restaurants_clean.jsonl"
DOCS_PATH = PROCESSED / "restaurant_docs.jsonl"
CHUNKS_PATH = PROCESSED / "review_chunks.jsonl"

# A chunk holds up to MAX_REVIEWS consecutive reviews and is closed once it reaches SOFT_WORDS words.
MAX_REVIEWS = 5
SOFT_WORDS = 150
# Hard limit, special tokens included: the max_seq_length of all-MiniLM-L6-v2.
MAX_TOKENS = 256
TOKENIZER_REPO = "sentence-transformers/all-MiniLM-L6-v2"

# Packing uses a cheap token estimate (every word, punctuation mark and non-ASCII character is
# at least one token, plus one per SUBWORD_CHARS characters of a long word); the real tokenizer
# then checks every chunk. A group over the limit is split by fit_chunk() until its pieces fit.
# That happens for single reviews longer than a chunk, and for the few groups the estimate
# under-counts (digit strings, Korean, dense German/Spanish text).
EST_BUDGET = MAX_TOKENS - 2
SUBWORD_CHARS = 5
PIECE_RE = re.compile(r"[A-Za-z0-9]+|\S")


def load_tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    return tok


def est_tokens(text):
    n = 0
    for m in PIECE_RE.finditer(text):
        piece = m.group()
        n += 1 + len(piece) // SUBWORD_CHARS if piece[0].isalnum() and piece.isascii() else 1
    return n


def join(parts):
    return " ".join(text for _, _, text in parts)


def pack_reviews(reviews):
    """Greedy-pack reviews into groups of parts (review_id, time, text) using the token estimate."""
    groups, cur, cur_words, cur_tokens = [], [], 0, 0
    for r in reviews:
        text = r["text"]
        tokens, words = est_tokens(text), len(text.split())
        if cur and (len(cur) >= MAX_REVIEWS or cur_words >= SOFT_WORDS or cur_tokens + tokens > EST_BUDGET):
            groups.append(cur)
            cur, cur_words, cur_tokens = [], 0, 0
        cur.append((r["review_id"], r["time"], text))
        cur_words += words
        cur_tokens += tokens
    if cur:
        groups.append(cur)
    return groups


def fit_chunk(tok, parts):
    """Split a group that exceeds MAX_TOKENS: by reviews first, then by words, then by characters."""
    if len(tok.encode(join(parts)).ids) <= MAX_TOKENS:
        return [parts]
    if len(parts) > 1:
        mid = len(parts) // 2
        return fit_chunk(tok, parts[:mid]) + fit_chunk(tok, parts[mid:])
    rid, t, text = parts[0]
    words = text.split()
    if len(words) > 1:
        mid = len(words) // 2
        halves = " ".join(words[:mid]), " ".join(words[mid:])
    else:
        mid = len(text) // 2
        halves = text[:mid], text[mid:]
    return fit_chunk(tok, [(rid, t, halves[0])]) + fit_chunk(tok, [(rid, t, halves[1])])


def make_chunks(tok, gmap_id, reviews):
    """Return (Chunk list, exact token counts, oversize groups split, groups the estimate under-counted)."""
    groups = pack_reviews(reviews)
    counts = [len(e.ids) for e in tok.encode_batch([join(g) for g in groups])] if groups else []
    chunks, tokens, oversize, misses = [], [], 0, 0
    for group, n in zip(groups, counts):
        if n <= MAX_TOKENS:
            pieces = [(group, n)]
        else:
            if est_tokens(join(group)) <= EST_BUDGET:
                misses += 1
            else:
                oversize += 1
            pieces = [(p, len(tok.encode(join(p)).ids)) for p in fit_chunk(tok, group)]
        for parts, n_tokens in pieces:
            chunks.append(Chunk(chunk_id=f"{gmap_id}:{len(chunks)}", gmap_id=gmap_id,
                                text=join(parts), review_ids=[p[0] for p in parts], times=[p[1] for p in parts]))
            tokens.append(n_tokens)
    return chunks, tokens, oversize, misses


def iter_restaurant_reviews():
    """Yield (gmap_id, [reviews]) groups; reviews of a restaurant must be contiguous."""
    done, current, group = set(), None, []
    with open(REVIEWS_PATH, encoding="utf-8") as f:
        for line in tqdm(f, unit="rev", mininterval=5):
            r = json.loads(line)
            if r["gmap_id"] != current:
                if group:
                    yield current, group
                    done.add(current)
                if r["gmap_id"] in done:
                    raise ValueError(f"reviews for {r['gmap_id']} are not contiguous; sort reviews_clean.jsonl first")
                current, group = r["gmap_id"], []
            group.append(r)
    if group:
        yield current, group


def main():
    tok = load_tokenizer()
    restaurants = {}
    with open(RESTAURANTS_PATH, encoding="utf-8") as f:
        for r in map(json.loads, f):
            restaurants[r["gmap_id"]] = r

    chunk_words, chunk_tokens, chunk_review_counts = [], [], Counter()
    n_docs = n_chunks = n_empty = n_oversize = n_misses = 0

    def write_doc(fdocs, gid, reviews):
        nonlocal n_docs, n_empty
        r = restaurants[gid]
        doc = RestaurantDoc(
            gmap_id=gid, name=r["name"], borough=r["borough"], address=r["address"],
            categories=r["categories"], avg_rating=r["avg_rating"],
            num_clean_reviews=len(reviews), low_evidence=r["low_evidence"],
            text="\n".join(x["text"] for x in reviews))
        d = dataclasses.asdict(doc)
        del d["reviews"]  # not stored; text holds the concatenation
        fdocs.write(json.dumps(d, ensure_ascii=False) + "\n")
        n_docs += 1
        n_empty += not reviews

    with open(DOCS_PATH, "w", encoding="utf-8") as fdocs, open(CHUNKS_PATH, "w", encoding="utf-8") as fch:
        seen = set()
        for gid, reviews in iter_restaurant_reviews():
            seen.add(gid)
            write_doc(fdocs, gid, reviews)
            chunks, tokens, oversize, misses = make_chunks(tok, gid, reviews)
            n_oversize += oversize
            n_misses += misses
            for c, n_tok in zip(chunks, tokens):
                n_chunks += 1
                chunk_words.append(len(c.text.split()))
                chunk_tokens.append(n_tok)
                chunk_review_counts[len(c.review_ids)] += 1
                fch.write(json.dumps(dataclasses.asdict(c), ensure_ascii=False) + "\n")
        for gid in restaurants.keys() - seen:  # restaurants left with no clean reviews
            write_doc(fdocs, gid, [])

    assert max(chunk_tokens) <= MAX_TOKENS, "a chunk exceeds the token limit"
    pcts = (5, 25, 50, 75, 95, 99)
    for label, arr in (("words", np.array(chunk_words)), (f"tokens, {TOKENIZER_REPO}", np.array(chunk_tokens))):
        print(f"Chunk length ({label}): " + "  ".join(f"p{p}={np.percentile(arr, p):.0f}" for p in pcts)
              + f"  max={arr.max()}  mean={arr.mean():.1f}")
    print(f"Restaurant docs: {n_docs:,} ({n_empty:,} with no clean reviews, empty text)")
    print(f"Review chunks: {n_chunks:,}; chunks per restaurant: mean {n_chunks / max(n_docs - n_empty, 1):.1f}")
    print(f"Groups over {MAX_TOKENS} tokens that were split: {n_oversize:,} oversize by estimate "
          f"(mostly single long reviews), {n_misses:,} under-counted by the estimate")
    print("Reviews per chunk: " + ", ".join(f"{k}: {v:,}" for k, v in sorted(chunk_review_counts.items())))
    print(f"Wrote {DOCS_PATH}\nWrote {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
