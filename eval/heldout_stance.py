"""The single pre-registered held-out evaluation of the frozen stance re-ranker on the 15 test queries.

Weights come from reports/stance_rerank_frozen.json (frozen on the dev queries, commit 2b1e13f); candidate depth is the
pre-registered 20. Nothing is tuned here. The script stops before scoring if the frozen file differs from that commit or if any
of the top-20 candidates of a test query has no judge label. No API calls: stance labels are read from the on-disk cache.

    python -m eval.heldout_stance [--check-only]
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from eval import make_pool
from eval import tune_stance_rerank as t
from eval.validation_report import label_source
from eval.metrics import N_RESAMPLES, f3, read_categories, read_labels

REPO = Path(__file__).resolve().parent.parent
FROZEN = REPO / "reports" / "stance_rerank_frozen.json"
FROZEN_COMMIT = "2b1e13f"
LABELS = REPO / "eval" / "llm_labels.jsonl"
QUERIES = REPO / "eval" / "queries_final.csv"
OUT = REPO / "reports" / "stance_heldout.md"


def frozen_unchanged():
    rel = FROZEN.relative_to(REPO).as_posix()
    same_as_commit = subprocess.run(["git", "diff", "--quiet", FROZEN_COMMIT, "--", rel], cwd=REPO).returncode == 0
    clean = not subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return same_as_commit and clean


def unjudged_top20(features, labels, qids):
    """{query_id: [gmap_id]} of top-20 candidates (the re-ranker only reorders these) with no label."""
    return {q: [c.result.gmap_id for c in features[q][0] if (q, c.result.gmap_id) not in labels] for q in qids
            if any((q, c.result.gmap_id) not in labels for c in features[q][0])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)
    if not frozen_unchanged():
        sys.exit(f"{FROZEN} differs from commit {FROZEN_COMMIT} (or has uncommitted changes): stopping.")
    frozen = json.loads(FROZEN.read_text())
    weights = frozen["weights"]
    from src.config import STANCE_RERANK
    from src.retrieval.stance_rerank import StanceReranker
    queries, labels, categories = t.read_queries(QUERIES), read_labels(LABELS), read_categories(QUERIES)
    dev, test = t.split_queries([q["query_id"] for q in queries])
    if dev != frozen["dev_queries"]:
        sys.exit("The dev split does not match the frozen file: stopping.")
    semantic = make_pool.load_semantic()
    if semantic is None:
        sys.exit("The semantic index is not built; not rebuilding it silently.")
    reranker = StanceReranker.from_defaults(semantic, top_n=STANCE_RERANK["top_n"])
    features = {}
    for q in queries:
        if q["query_id"] in test:
            boroughs = [q["borough"].strip()] if q.get("borough", "").strip() else None
            features[q["query_id"]] = (reranker.candidates(q["query"], boroughs, cache_only=True), t.is_allergy_text(q["query"]))
    missing = unjudged_top20(features, labels, test)
    n_cand = sum(len(features[q][0]) for q in test)
    print(f"frozen file unchanged since {FROZEN_COMMIT}: True; test queries {len(test)}; top-20 candidates {n_cand}; "
          f"unjudged: {sum(map(len, missing.values()))} {missing or ''}")
    if missing:
        sys.exit("Some top-20 test candidates are unjudged: stopping.")
    if args.check_only:
        return

    tbl = t.split_table(features, labels, test, {t.STANCE: weights}, categories)
    base_rows = {r["query_id"]: r for r in t.rows_for(features, labels, test, t.ZERO, t.BASE, categories)}
    st_rows = {r["query_id"]: r for r in t.rows_for(features, labels, test, weights, t.STANCE, categories)}
    qtext = {q["query_id"]: q["query"] for q in queries}
    records = {}
    for line in open(LABELS, encoding="utf-8"):
        r = json.loads(line)
        records[(r["query_id"], r["gmap_id"])] = r
    sources = Counter(label_source(records[(q, c.result.gmap_id)]) for q in test for c in features[q][0])
    L = ["# Held-out evaluation of the frozen stance re-ranker (single pre-registered run)", "",
         "> **This is the single pre-registered held-out run.** It was run once, with the weights frozen on the dev queries before any "
         "test result existed. Nothing was tuned, re-run or adjusted afterwards. Labels are LLM-judge labels (not validated per pair "
         "against humans), so read the result as provisional.", "",
         f"- Frozen weights: w_pos {weights['w_pos']}, w_neg {weights['w_neg']}, w_strong_neg {weights['w_strong_neg']} "
         f"(`reports/stance_rerank_frozen.json`, commit `{FROZEN_COMMIT}`; the file is byte-identical to that commit). "
         f"Rule: {frozen['rule']}",
         f"- Candidate depth: top {STANCE_RERANK['top_n']} of `semantic_capped` (pre-registered), so both systems rank the same 20 restaurants "
         "per query and differ only in the order.",
         f"- Test queries ({len(test)}, seed {t.SEED}): {', '.join(test)}. Dev queries (not used here): {len(dev)}.",
         f"- Checks before scoring: frozen file unchanged since the commit: yes; **unjudged top-20 candidates: 0 of {n_cand}**. "
         "Unjudged results in the top 10 would count as 0; there are none.",
         f"- Metrics: NDCG@10, lenient P@5 and P@10 (as in `eval/metrics.py`; relevance 0/1/2, lenient counts 1 and 2 as relevant); "
         f"95% percentile bootstrap over queries ({N_RESAMPLES:,} resamples, seed {t.SEED}); paired difference with a two-sided sign-flip test.",
         f"- Stance labels: `{reranker.labeler.id}` (read from the on-disk cache; no API call in this run). "
         f"Relevance labels of the 300 candidate pairs by source: {', '.join(f'{k} {v}' for k, v in sorted(sources.items()))}.", "",
         "## Result on the 15 test queries", ""]
    L += t.table(tbl)
    s = tbl[t.STANCE]
    c = s["vs_base"]
    verdict = ("The paired difference is positive and its sign-flip p-value is below 0.05."
               if c["diff"] > 0 and c["p"] < 0.05 else
               "The difference is not significant at the 0.05 level: this run does not show that the stance re-ranker beats "
               "`semantic_capped`, and the interval includes zero.")
    L += ["", f"**Paired NDCG@10 difference (stance minus semantic_capped): {c['diff']:+.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}], "
          f"sign-flip p = {f3(c['p'])}; {c['wins']} queries better, {c['losses']} worse, {c['ties']} tied.** {verdict}", "",
          "## By category (NDCG@10)", ""] + [x.replace("stance (tuned)", "stance (frozen)") for x in t.category_table(s)]
    L += ["", "Category cells have 2 to 6 queries each: descriptions of this test set, not estimates.", "",
          "## Per query", "", "| Query | Category | semantic_capped NDCG@10 | stance NDCG@10 | Δ | Result | P@5 base → stance |", "|---|---|---|---|---|---|---|"]
    for q in test:
        b, r = base_rows[q], st_rows[q]
        d = r["NDCG10"] - b["NDCG10"]
        res = "win" if d > 1e-9 else "loss" if d < -1e-9 else "tie"
        L.append(f"| {q} \"{qtext[q][:60]}\" | {categories.get(q, '')} | {f3(b['NDCG10'])} | {f3(r['NDCG10'])} | {d:+.3f} | {res} | "
                 f"{b['P5_lenient']:.1f} → {r['P5_lenient']:.1f} |")
    L += ["", "## Limitations", "",
          "- **Small test set.** 15 queries: intervals are wide, and a non-significant result is not evidence of no effect.",
          "- **Judge labels, one family of model.** Relevance is by an LLM judge; stance labels come from the same model family, so errors can "
          "be correlated, which may favor the re-ranker. The judge agrees with a human rater about 0.75 of the time on a 120-pair sample "
          "(`reports/judge_validation.md`).",
          "- **Mixed label sources.** The 111 test pairs added for this run were judged by one provider (Groq `gpt-oss-120b`); the other 189 "
          "came from the earlier judging. Many of the new labels are 0 with the stock reason \"no evidence found\" "
          "(see `reports/no_evidence_audit.md` for the dev equivalent), which are strict readings of the need rather than missing evidence.",
          "- **w_neg = 0.** The frozen setting ignores the share of negative sentences except for the strong-negative term, which applies to "
          "allergy queries only.",
          "- **Weights chosen by hand within a rule.** The `w_pos <= 0.9` cap was a judgment on the dev split; the dev numbers are optimistic.",
          "- **No default was changed.** The app default remains `semantic_capped`; `STANCE_RERANK` in `src/config.py` still holds the "
          "starting weights.", ""]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"NDCG@10 base {f3(tbl[t.BASE]['NDCG10'])} stance {f3(s['NDCG10'])}; diff {c['diff']:+.3f} p {f3(c['p'])}; {c['wins']}/{c['losses']}/{c['ties']}")


if __name__ == "__main__":
    main()
