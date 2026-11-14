"""Tune and evaluate the stance re-ranker (src/retrieval/stance_rerank.py) on a held-out split.

The 35 frozen queries are split by a fixed seed (6165) into 20 dev and 15 test queries. The weights
(w_pos, w_neg, w_strong_neg) are chosen by grid search on the dev queries ONLY; the test queries are scored once,
with the chosen weights, and never used to choose anything. All-35 numbers are reported separately (they include
the dev queries the weights were tuned on, so they are optimistic).

Systems compared: semantic_capped (the base ranking = zero weights) and semantic_capped_stance (tuned weights),
plus the untuned starting weights (0.3 / 0.6). Metrics: NDCG@10, P@5 and P@10 (lenient), macro-averaged over
queries, with percentile bootstrap 95% CIs and a paired sign-flip test of NDCG@10 against semantic_capped
(eval/metrics.py), overall and by query category. Results without a label count as 0 (not relevant) and are
reported as unjudged; weights that surface unjudged results are penalized the same way.

The base ranking's top-30 evidence sentences are labeled with the LLM (cached on disk), so a first run takes
a while; reruns are free. Run from the repo root:
    python -m eval.tune_stance_rerank --labels eval/llm_labels.jsonl
Writes reports/stance_rerank.md and reports/stance_rerank_weights.json.
"""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

from eval.make_pool import read_queries
from eval.metrics import (CATEGORY_ORDER, N_RESAMPLES, SEED, bootstrap_ci, compare, f3, mean, query_metrics, read_categories,
                          read_labels, rel)
from src.allergy import is_allergy_text
from src.retrieval.stance_rerank import rescore, stance_score

EVAL_DIR = Path(__file__).parent
REPO = EVAL_DIR.parent
QUERIES_PATH = EVAL_DIR / "queries_final.csv"
REPORT_PATH = REPO / "reports" / "stance_rerank.md"
WEIGHTS_PATH = REPO / "reports" / "stance_rerank_weights.json"
N_DEV = 20
INITIAL = {"w_pos": 0.3, "w_neg": 0.6, "w_strong_neg": 0.1}
ZERO = {"w_pos": 0.0, "w_neg": 0.0, "w_strong_neg": 0.0}  # no stance effect: the semantic_capped order
GRID = {"w_pos": [0.0, 0.1, 0.2, 0.3, 0.45, 0.6], "w_neg": [0.0, 0.2, 0.4, 0.6, 0.9, 1.2],
        "w_strong_neg": [0.0, 0.1, 0.2, 0.4]}
BASE, STANCE = "semantic_capped", "semantic_capped_stance"
EXAMPLES = [("peanut allergy", "restaurant careful about peanut allergies"),
            ("celiac safe", "safe for celiac disease, no cross contamination"),
            ("halal chinese food", "halal chinese food")]


def split_queries(query_ids, seed=SEED, n_dev=N_DEV):
    """(dev, test) query ids: a seeded random sample of n_dev for dev, the rest for test (both sorted)."""
    ids = sorted(query_ids)
    dev = sorted(random.Random(seed).sample(ids, n_dev))
    return dev, [q for q in ids if q not in set(dev)]


def top_ids(cands, weights, allergy, k=10):
    return [c.result.gmap_id for c in rescore(cands, weights["w_pos"], weights["w_neg"], weights["w_strong_neg"], allergy)[:k]]


def rows_for(features, labels, qids, weights, system, categories):
    """Per-query metric rows (the format of eval/metrics.py) for one weight setting."""
    by_query = {}
    for (q, g), l in labels.items():
        by_query.setdefault(q, {})[g] = l
    return [{"system": system, "query_id": q, "category": categories.get(q, ""),
             **query_metrics(top_ids(features[q][0], weights, features[q][1]), by_query.get(q, {}))} for q in qids]


def grid_search(features, labels, dev_ids, categories, grid=GRID):
    """Weights with the best mean dev NDCG@10; ties go to the smaller total weight. Returns (weights, results)."""
    results = []
    for wp, wn, ws in itertools.product(grid["w_pos"], grid["w_neg"], grid["w_strong_neg"]):
        w = {"w_pos": wp, "w_neg": wn, "w_strong_neg": ws}
        rows = rows_for(features, labels, dev_ids, w, STANCE, categories)
        results.append((mean(r["NDCG10"] for r in rows), wp + wn + ws, w))
    best = max(results, key=lambda r: (r[0], -r[1]))
    return best[2], results


def summarize_rows(rows):
    return {m: mean(r[m] for r in rows) for m in ("NDCG10", "P5_lenient", "P10_lenient")} | {
        "unjudged": sum(r["unjudged"] for r in rows), "n": len(rows)}


def split_table(features, labels, qids, weights_by_name, categories, seed=SEED, n=N_RESAMPLES):
    """For one set of queries: per system summary with bootstrap CIs, and the paired test vs semantic_capped."""
    systems = {BASE: rows_for(features, labels, qids, ZERO, BASE, categories)}
    for name, w in weights_by_name.items():
        systems[name] = rows_for(features, labels, qids, w, name, categories)
    out = {}
    for name, rows in systems.items():
        s = summarize_rows(rows)
        s["ci"] = {m: bootstrap_ci([r[m] for r in rows], n, seed) for m in ("NDCG10", "P5_lenient", "P10_lenient")}
        if name != BASE:
            s["vs_base"] = compare(rows + systems[BASE], BASE, name, n=n, seed=seed)
            s["by_category"] = {}
            for cat in CATEGORY_ORDER:
                cq = [r["query_id"] for r in rows if r["category"] == cat]
                b = [r["NDCG10"] for r in systems[BASE] if r["category"] == cat]
                a = [r["NDCG10"] for r in rows if r["category"] == cat]
                cmp = compare([r for r in rows + systems[BASE] if r["category"] == cat], BASE, name, n=n, seed=seed) if cq else None
                s["by_category"][cat] = {"n": len(cq), "base": mean(b), "stance": mean(a), "cmp": cmp}
        out[name] = s
    return out


# ---- report -------------------------------------------------------------------
def ci(x):
    lo, hi = x
    return "n/a" if lo is None else f"[{lo:.3f}, {hi:.3f}]"


def table(tbl):
    L = ["| System | NDCG@10 [95% CI] | P@5 lenient [95% CI] | P@10 lenient [95% CI] | NDCG@10 vs semantic_capped [95% CI] | p (sign-flip) | better / worse / tie | Unjudged |",
         "|---|---|---|---|---|---|---|---|"]
    for name, s in tbl.items():
        c = s.get("vs_base")
        d = f"{c['diff']:+.3f} {ci((c['ci_low'], c['ci_high']))}" if c else "—"
        p = f3(c["p"]) if c else "—"
        wl = f"{c['wins']} / {c['losses']} / {c['ties']}" if c else "—"
        L.append(f"| {name} | {f3(s['NDCG10'])} {ci(s['ci']['NDCG10'])} | {f3(s['P5_lenient'])} {ci(s['ci']['P5_lenient'])} | "
                 f"{f3(s['P10_lenient'])} {ci(s['ci']['P10_lenient'])} | {d} | {p} | {wl} | {s['unjudged']} |")
    return L


def category_table(s):
    L = ["| Category | n | semantic_capped | stance (tuned) | Δ NDCG@10 | p (sign-flip) |", "|---|---|---|---|---|---|"]
    for cat, c in s["by_category"].items():
        d = c["cmp"]
        L.append(f"| {cat} | {c['n']} | {f3(c['base'])} | {f3(c['stance'])} | "
                 f"{'n/a' if not d or d['diff'] is None else format(d['diff'], '+.3f')} | {f3(d['p']) if d else 'n/a'} |")
    return L


def example_section(features, labels, queries, weights, info, qid_by_text):
    L = []
    for title, text in EXAMPLES:
        qid = qid_by_text.get(text)
        if qid not in features:
            continue
        cands, allergy = features[qid]
        before, after = top_ids(cands, ZERO, allergy, 5), top_ids(cands, weights, allergy, 5)
        by_id = {c.result.gmap_id: c for c in cands}
        L += [f"### {title} (`{qid}`: \"{text}\")", "", "| # | Before: semantic_capped | label | After: stance re-rank | label |", "|---|---|---|---|---|"]
        cell = lambda g: (f"{info(g)['name']} (+{by_id[g].counts['accommodation']} / −{by_id[g].counts['negative']})",
                          labels.get((qid, g), "?"))
        for i, (b, a) in enumerate(zip(before, after), 1):
            L.append(f"| {i} | {cell(b)[0]} | {cell(b)[1]} | {cell(a)[0]} | {cell(a)[1]} |")
        worst = next((s for g in before for s in by_id[g].sentences if s["stance"] == "negative"), None)
        if worst:
            L += ["", f"A sentence labeled negative in the old top 5: \"{worst['text'][:200]}\""]
        L += ["", "(+ accommodation / − negative sentences among those labeled; label = LLM-judge relevance 0/1/2, ? = unjudged)", ""]
    return L


DEV_TITLE = "Dev split (weights were tuned here: optimistic)"


def sections(dev, test):
    """(title, query ids) to report. With no test queries (dev-only run) the test split is not touched at all."""
    if not test:
        return [(DEV_TITLE, dev)]
    return [("Held-out test split", test), (DEV_TITLE, dev), ("All 35 queries (includes the dev queries: optimistic)", dev + test)]


def write_report(path, *, args, dev, test, tuned, grid_results, tbl, features, labels, categories, info, qid_by_text,
                 new_pairs, labeler_id):
    cat_n = lambda ids: ", ".join(f"{c} {sum(categories.get(q) == c for q in ids)}" for c in CATEGORY_ORDER)
    L = ["# Stance-aware re-ranking (PROVISIONAL)", "",
         "> **PROVISIONAL:** relevance labels are LLM-judge labels, not yet validated against human labels; stance labels "
         f"come from `{labeler_id}` and are not validated either.", "",
         f"- Labels: `{rel(args.labels)}` ({len(labels)} labeled pairs); queries: `{rel(args.queries)}`",
         (f"- Split (seed {args.seed}): {len(dev)} dev queries ({cat_n(dev)}) and {len(test)} test queries ({cat_n(test)})"
          if test else f"- **DEV-ONLY RUN:** {len(dev)} of {args.n_dev_total} dev queries had cached stance labels ({cat_n(dev)}); "
                       f"{len(args.unlabeled_dev)} dev queries still unlabeled. The test queries were not labeled, scored or used."),
         f"- Weights were tuned on the dev queries only (grid of {len(grid_results)} settings, objective: mean dev NDCG@10). "
         f"Chosen{' (PARTIAL dev set: do not adopt)' if args.partial else ''}: w_pos {tuned['w_pos']}, w_neg {tuned['w_neg']}, w_strong_neg {tuned['w_strong_neg']}. "
         f"Starting weights (untuned): {INITIAL}.",
         f"- Re-ranks the top {args.top_n} of semantic_capped; per restaurant, its evidence sentences are labeled "
         "accommodation / mention / negative / not_about_allergy toward the query (`src/retrieval/stance_rerank.py`).",
         "- Unjudged top-10 results count as 0 (not relevant), for every system.", ""]
    for title, ids in sections(dev, test):
        L += [f"## {title} (n={len(ids)})", ""] + table(tbl[title]) + [""]
        L += ["NDCG@10 by category (tuned weights):", ""] + category_table(tbl[title][f"{STANCE}"]) + [""]
    L += ["## Before / after examples", "", f"Tuned weights; top 5 on the full query (all-35 view).", ""]
    L += example_section(features, labels, None, tuned, info, qid_by_text)
    L += ["## New pairs needing judge labels", "",
          f"The tuned re-ranker's top-10 contains {new_pairs['all']} (query, restaurant) pairs with no label "
          f"({new_pairs['test']} on test queries, {new_pairs['dev']} on dev). Run `python -m eval.make_pool` and then "
          "`python -m eval.llm_judge` (resumable; labels only new pairs), then rerun this script.", ""]
    L += ["## Limitations", "",
          "- **Same 35 queries for development and the final claim.** Only the dev/test split protects the test numbers, and 15 queries "
          "is a small test set: CIs are wide and per-category cells (4 to 11 queries) are too small for firm conclusions.",
          "- **Labels are LLM-judge labels**, not validated against humans, and the stance labeler is the same family of model; their "
          "errors can be correlated, which flatters the re-ranker.",
          "- **Unjudged results count as 0**, which penalizes any setting that surfaces restaurants outside the judged pool, "
          "and biases tuning toward the existing pool. The pool grows only after the new pairs above are labeled.",
          "- **Evidence is thin:** at most 3 reviews per restaurant from retrieval, 4 sentences kept; a restaurant with no allergy sentence "
          "gets no stance signal either way.",
          "- **Candidate set is fixed:** only the base top-30 are re-ranked, so a restaurant the embedding missed cannot be recovered.",
          "- **Stance labels are not allergy-safety claims.** An accommodation sentence is one reviewer's statement; the app's allergy "
          "guard and note stay in force.",
          "- Weights are tuned for NDCG@10 on this label scheme; they may not transfer to other queries or boroughs.", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tune the stance re-ranker on dev queries, report on test and all queries")
    ap.add_argument("--labels", type=Path, required=True, help="llm_labels.jsonl or a labels CSV")
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH)
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    ap.add_argument("--weights-out", type=Path, default=WEIGHTS_PATH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resamples", type=int, default=N_RESAMPLES)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--dev-only", action="store_true", help="use only the dev queries; the test split is never labeled or scored")
    ap.add_argument("--cache-only", action="store_true",
                    help="never call the labeler: skip queries whose sentence labels are not all cached")
    args = ap.parse_args(argv)
    for p in (args.labels, args.queries):
        if not p.exists():
            sys.exit(f"Missing {p}")

    from eval import make_pool
    from src.config import STANCE_RERANK
    from src.retrieval.stance_rerank import StanceReranker
    args.top_n = args.top_n or STANCE_RERANK["top_n"]
    semantic = make_pool.load_semantic()
    if semantic is None:
        sys.exit("The semantic index is not built; run: python -m src.retrieval.semantic --build")
    reranker = StanceReranker.from_defaults(semantic, top_n=args.top_n)
    queries, labels, categories = read_queries(args.queries), read_labels(args.labels), read_categories(args.queries)
    from src.retrieval.stance_rerank import LabelsUnavailable
    all_dev, all_test = split_queries([q["query_id"] for q in queries], args.seed)
    wanted = set(all_dev) if args.dev_only else set(all_dev) | set(all_test)
    features, unlabeled = {}, []
    for q in queries:
        if q["query_id"] not in wanted:
            continue
        boroughs = [q["borough"].strip()] if q.get("borough", "").strip() else None
        try:
            features[q["query_id"]] = (reranker.candidates(q["query"], boroughs, cache_only=args.cache_only),
                                       is_allergy_text(q["query"]))
        except LabelsUnavailable:
            unlabeled.append(q["query_id"])
            continue
        print(f"{q['query_id']}: {len(features[q['query_id']][0])} candidates labeled", flush=True)

    dev = [q for q in all_dev if q in features]
    test = [] if args.dev_only else [q for q in all_test if q in features]
    args.partial = bool(unlabeled)
    args.unlabeled_dev = [q for q in unlabeled if q in all_dev]
    args.n_dev_total = len(all_dev)
    if not dev:
        sys.exit("No dev query has all its sentence labels cached; run python -m eval.label_stance first.")
    tuned, grid_results = grid_search(features, labels, dev, categories)
    tbl = {}
    for title, ids in sections(dev, test):
        tbl[title] = split_table(features, labels, ids, {STANCE: tuned, "stance_initial_weights": INITIAL}, categories,
                                 args.seed, args.resamples)
    unjudged = lambda ids: sum(g not in {gg for (qq, gg) in labels if qq == q}
                               for q in ids for g in top_ids(features[q][0], tuned, features[q][1]))
    new_pairs = {"dev": unjudged(dev), "test": unjudged(test), "all": unjudged(dev + test)}
    qid_by_text = {q["query"]: q["query_id"] for q in queries}
    write_report(args.out, args=args, dev=dev, test=test, tuned=tuned, grid_results=grid_results, tbl=tbl,
                 features=features, labels=labels, categories=categories, info=semantic.info, qid_by_text=qid_by_text,
                 new_pairs=new_pairs, labeler_id=reranker.labeler.id)
    args.weights_out.write_text(json.dumps({"weights": tuned, "dev": dev, "test": test, "seed": args.seed,
                                            "partial_dev": args.partial,
                                            "unjudged_new_pairs": new_pairs}, indent=1))
    print(f"Tuned on {len(dev)} dev queries: {tuned}; new unjudged pairs in top-10: {new_pairs}")
    print(f"Wrote {args.out} and {args.weights_out}")


if __name__ == "__main__":
    main()
