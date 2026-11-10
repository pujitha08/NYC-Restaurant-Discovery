"""Compare semantic, chunk TF-IDF and doc TF-IDF on 5 smoke queries: top-10, snippet ownership,
top-10 overlap and crude regex failure flags (group / negative / shellfish-dish mentions).

The flags are keyword counts, not judgments (e.g. "cross contamination" counts as negative);
see reports/failure_examples.md for the qualitative findings. Semantic uses density_mode="log".

Usage (repo root; needs all three indexes):
    HF_HUB_OFFLINE=1 python -m eval.compare_systems [OUT.json]   # default reports/retrieval_comparison.json
"""
import json, re, sys, time
from src.retrieval.evidence import ReviewStore
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.tfidf import TfidfRetriever
from src.retrieval.tfidf_chunks import ChunkTfidfRetriever

QUERIES = [("VEG", "vegan options for a group", ["Brooklyn"]),
           ("Q20", "restaurant careful about peanut allergies", None),
           ("Q22", "safe for celiac disease, no cross contamination", None),
           ("Q25", "shellfish allergy", None),
           ("Q29", "halal chinese food", ["Manhattan"])]
GROUP = re.compile(r"\b(group|groups|party|parties|friends|family|everyone|all of us|people|table for|large)\b", re.I)
NEG = re.compile(r"\b(not much|no (?:real )?(?:vegan|vegetarian|gluten|halal)|don'?t have|didn'?t have|doesn'?t have|"
                 r"lack|limited|not (?:many|a lot|great|safe|celiac|good)|wasn'?t|isn'?t (?:safe|vegan|halal)|"
                 r"got sick|cross[- ]contaminat\w*|disappoint\w*|wouldn'?t recommend)\b", re.I)
SHELL_DISH = re.compile(r"\b(lobster|shrimp|crab|oysters?|clams?|mussels?|scallops?|prawns?|crawfish|shellfish)\b", re.I)
ALLERG = re.compile(r"\ballerg\w*\b", re.I)
FISH_ALLERGY = re.compile(r"\bfish allerg\w*", re.I)
SHELL_ALLERGY = re.compile(r"\bshellfish[- ]allerg\w*|\ballergic to shellfish\b", re.I)

store = ReviewStore()
t = time.perf_counter(); systems = {"semantic": SemanticRetriever.load()}; load_sem = time.perf_counter() - t
systems["semantic"].embedder.encode(["warm up"])
t = time.perf_counter(); systems["chunk_tfidf"] = ChunkTfidfRetriever.load(); load_ch = time.perf_counter() - t
t = time.perf_counter(); systems["doc_tfidf"] = TfidfRetriever.load(); load_doc = time.perf_counter() - t
print(f"load: semantic {load_sem:.1f}s, chunk_tfidf {load_ch:.1f}s, doc_tfidf {load_doc:.1f}s")
for name, r in systems.items():  # warm per-retriever caches (feature names) so timings are steady-state
    r.search("warm up query", top_k=1)

out = {}
for qid, q, b in QUERIES:
    out[qid] = {"query": q, "boroughs": b, "systems": {}}
    for name, r in systems.items():
        t = time.perf_counter(); res = r.search(q, boroughs=b, top_k=10); dt = time.perf_counter() - t
        rows, foreign = [], 0
        for x in res:
            own = {rv["review_id"]: rv["gmap_id"] for rv in store.get([s.review_id for s in x.evidence_snippets])}
            f = sum(own.get(s.review_id) != x.gmap_id for s in x.evidence_snippets); foreign += f
            m = r.info(x.gmap_id)
            rows.append({"gmap_id": x.gmap_id, "name": m["name"], "borough": m["borough"], "score": x.score,
                         "reviews": m["num_clean_reviews"], "foreign": f,
                         "snippets": [{"id": s.review_id, "text": s.text} for s in x.evidence_snippets]})
        out[qid]["systems"][name] = {"time": dt, "results": rows, "foreign": foreign,
                                     "snippets": sum(len(r_["snippets"]) for r_ in rows)}

def flags(rows):
    texts = [s["text"] for r in rows for s in r["snippets"]]
    first = [r["snippets"][0]["text"] if r["snippets"] else "" for r in rows]
    return {"group_snips": sum(bool(GROUP.search(t)) for t in texts), "neg_snips": sum(bool(NEG.search(t)) for t in texts),
            "neg_top_snippet": sum(bool(NEG.search(t)) for t in first),
            "shell_dish_no_allergy": sum(bool(SHELL_DISH.search(t)) and not ALLERG.search(t) for t in texts),
            "fish_allergy_not_shellfish": sum(bool(FISH_ALLERGY.search(t)) and not SHELL_ALLERGY.search(t) for t in texts),
            "shellfish_allergy": sum(bool(SHELL_ALLERGY.search(t)) for t in texts), "total_snips": len(texts),
            "median_reviews": sorted(r["reviews"] for r in rows)[len(rows) // 2] if rows else 0}

for qid, d in out.items():
    print(f"\n######## {qid}: {d['query']!r}" + (f" [{', '.join(d['boroughs'])}]" if d["boroughs"] else ""))
    for name, s in d["systems"].items():
        print(f"  -- {name}: search {s['time']:.2f}s, {len(s['results'])} results, foreign snippets {s['foreign']}/{s['snippets']}")
        for i, r in enumerate(s["results"][:5], 1):
            snip = r["snippets"][0]["text"] if r["snippets"] else "(no snippet)"
            print(f"     {i}. {r['name']} ({r['borough']}, {r['reviews']} rev) {r['score']:.3f} | {snip[:150]}")
        s["flags"] = flags(s["results"])
    ids = {n: [r["gmap_id"] for r in s["results"]] for n, s in d["systems"].items()}
    ov = lambda a, b: len(set(ids[a]) & set(ids[b]))
    d["overlap"] = {"sem&chunk": ov("semantic", "chunk_tfidf"), "sem&doc": ov("semantic", "doc_tfidf"),
                    "chunk&doc": ov("chunk_tfidf", "doc_tfidf"),
                    "all3": len(set(ids["semantic"]) & set(ids["chunk_tfidf"]) & set(ids["doc_tfidf"])),
                    "union": len(set().union(*ids.values()))}
    print(f"  overlap top-10: {d['overlap']}")
    for name, s in d["systems"].items():
        print(f"  flags {name:12}: {s['flags']}")
out_path = sys.argv[1] if len(sys.argv) > 1 else "reports/retrieval_comparison.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(f"\nWrote {out_path}")
