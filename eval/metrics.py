"""Retrieval metrics per system from eval/runs.csv and relevance labels.

Inputs
  --runs     eval/runs.csv from make_pool.py (query_id, system, rank, gmap_id, score)
  --labels   eval/llm_labels.jsonl (llm_judge.py) or a CSV with query_id, gmap_id and
             adjudicated_label (agreement.py's template, filled) or label. Labels are 0/1/2.
  --queries  query file for categories (default eval/queries_final.csv, the frozen set)

Metrics per system and query (then averaged over queries, macro):
  P@5, P@10       relevant results in the top k divided by k (k fixed even if a system returns
                  fewer results), strict (label = 2) and lenient (label >= 1)
  NDCG@10         DCG@10 / IDCG@10 with DCG@10 = sum_{i=1..10} gain_i / log2(i + 1) and linear
                  graded gain = label (0, 1, 2). IDCG uses the best possible ordering of all
                  judged pairs for the query (the pooled labels), so a system is compared with the
                  best it could have done given the pool.
  R@10            relevant results in the top 10 / relevant pairs in the judged pool for the
                  query, strict and lenient
Results in a top 10 without a label are counted as 0 (not relevant) and reported as unjudged;
with labels for the full pool there should be none. NDCG@10 and R@10 are undefined for a query
with no relevant pair in the pool; such queries are left out of those averages (listed in the
summary), the same way for every system, so paired comparisons stay paired.

Significance: for every pair of systems, a paired randomization (sign-flip) test on the per-query
NDCG@10 differences (two-sided p, Holm-adjusted across pairs) and a percentile bootstrap 95% CI of
the mean difference, resampling queries. Per-system means get bootstrap 95% CIs (the chart's
error bars). Density modes: log vs capped for chunk TF-IDF and semantic, with per-query deltas.

Outputs (--out-dir, default reports/): metrics_summary.md, metrics_per_query.csv and
metrics_ndcg_p10.png.

Run from the repo root:
    python -m eval.metrics --labels eval/llm_labels.jsonl [--runs eval/runs.csv] [--queries ...]
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

EVAL_DIR = Path(__file__).parent
REPO = EVAL_DIR.parent
RUNS_PATH = EVAL_DIR / "runs.csv"
QUERIES_PATH = EVAL_DIR / "queries_final.csv"  # frozen set (PR #2); queries_draft.csv is kept as history
OUT_DIR = REPO / "reports"
K = 10
GAIN_FORMULA = "gain = label (0, 1, 2); DCG@10 = sum_{i=1..10} gain_i / log2(i + 1); NDCG@10 = DCG@10 / IDCG@10"
DENSITY_PAIRS = [("chunk_tfidf_log", "chunk_tfidf_capped"), ("semantic_log", "semantic_capped")]
CATEGORY_ORDER = ["single_diet", "mixed_group", "allergy", "diet_plus_filter"]
SYSTEM_ORDER = ["doc_tfidf", "chunk_tfidf_log", "chunk_tfidf_capped", "semantic_log", "semantic_capped"]
METRICS = ["P5_strict", "P5_lenient", "P10_strict", "P10_lenient", "NDCG10", "R10_strict", "R10_lenient"]
N_RESAMPLES = 10_000
SEED = 6165


# ---- inputs -----------------------------------------------------------------
def read_runs(path):
    """{system: {query_id: [gmap_id, ...] in rank order}} (ties in rank broken by gmap_id)."""
    rows = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["system"]][r["query_id"]].append((int(r["rank"]), r["gmap_id"]))
    return {s: {q: [g for _, g in sorted(v)] for q, v in qs.items()} for s, qs in rows.items()}


def _label(raw, where):
    raw = str(raw).strip()
    if raw not in {"0", "1", "2"}:
        raise ValueError(f"{where}: label must be 0, 1 or 2, got {raw!r}")
    return int(raw)


def read_labels(path):
    """{(query_id, gmap_id): 0|1|2} from an llm_labels JSONL (later lines win) or a labels CSV."""
    path = Path(path)
    labels = {}
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial line from an interrupted judge run
                labels[(r["query_id"], r["gmap_id"])] = _label(r["label"], f"{path}:{n}")
        return labels
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = "adjudicated_label" if "adjudicated_label" in (reader.fieldnames or []) else "label"
        for n, r in enumerate(reader, 2):
            if (r.get(col) or "").strip() == "":
                continue  # not labeled (yet)
            labels[(r["query_id"], r["gmap_id"])] = _label(r[col], f"{path}:{n}")
    return labels


def read_categories(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["query_id"]: r.get("category", "") for r in csv.DictReader(f)}


# ---- metrics ------------------------------------------------------------------
def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def query_metrics(ranked, qlabels, k=K):
    """Metrics for one system on one query. ranked: gmap_ids in rank order; qlabels: {gmap_id: label}."""
    top = ranked[:k]
    labels = [qlabels.get(g) for g in top]
    unjudged = sum(l is None for l in labels)
    gains = [l or 0 for l in labels]
    pool = list(qlabels.values())
    rel_strict, rel_lenient = sum(l == 2 for l in pool), sum(l >= 1 for l in pool)
    idcg = dcg(sorted(pool, reverse=True)[:k])
    found_strict, found_lenient = sum(g == 2 for g in gains), sum(g >= 1 for g in gains)
    return {
        "P5_strict": sum(g == 2 for g in gains[:5]) / 5, "P5_lenient": sum(g >= 1 for g in gains[:5]) / 5,
        "P10_strict": found_strict / k, "P10_lenient": found_lenient / k,
        "NDCG10": dcg(gains) / idcg if idcg > 0 else None,
        "R10_strict": found_strict / rel_strict if rel_strict else None,
        "R10_lenient": found_lenient / rel_lenient if rel_lenient else None,
        "n_results": len(top), "unjudged": unjudged,
        "pool_judged": len(pool), "pool_rel_strict": rel_strict, "pool_rel_lenient": rel_lenient,
    }


def evaluate(runs, labels, categories):
    """Per-query rows (one per system and query that the system ran) and the query list."""
    by_query = defaultdict(dict)
    for (q, g), l in labels.items():
        by_query[q][g] = l
    queries = sorted({q for qs in runs.values() for q in qs})
    rows = []
    for system in sorted(runs):
        for q in queries:
            m = query_metrics(runs[system].get(q, []), by_query.get(q, {}))
            rows.append({"system": system, "query_id": q, "category": categories.get(q, ""), **m})
    return rows, queries


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(rows, metrics=METRICS):
    """{system: {metric: mean over queries where defined}, plus n, unjudged}."""
    out = defaultdict(dict)
    present = {r["system"] for r in rows}
    for system in [s for s in SYSTEM_ORDER if s in present] + sorted(present - set(SYSTEM_ORDER)):
        rs = [r for r in rows if r["system"] == system]
        for m in metrics:
            out[system][m] = mean(r[m] for r in rs)
        out[system]["n_queries"] = len(rs)
        out[system]["unjudged"] = sum(r["unjudged"] for r in rs)
    return dict(out)


# ---- significance -------------------------------------------------------------
def paired_values(rows, a, b, metric="NDCG10"):
    """[(query_id, value_a, value_b)] for queries where both are defined."""
    va = {r["query_id"]: r[metric] for r in rows if r["system"] == a}
    vb = {r["query_id"]: r[metric] for r in rows if r["system"] == b}
    return [(q, va[q], vb[q]) for q in sorted(set(va) & set(vb)) if va[q] is not None and vb[q] is not None]


def randomization_p(diffs, n=N_RESAMPLES, seed=SEED):
    """Two-sided paired sign-flip test of mean(diffs) = 0; p = (count + 1) / (n + 1)."""
    if not diffs:
        return None
    observed = abs(sum(diffs) / len(diffs))
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        hits += abs(s / len(diffs)) >= observed - 1e-12
    return (hits + 1) / (n + 1)


def bootstrap_ci(values, n=N_RESAMPLES, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI of the mean, resampling the values (queries) with replacement."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return means[int(alpha / 2 * n)], means[min(n - 1, int((1 - alpha / 2) * n))]


def holm(pvalues):
    """Holm-adjusted p-values, same order as the input (None stays None)."""
    idx = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    adjusted, running = [None] * len(pvalues), 0.0
    m = len(idx)
    for rank, (p, i) in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def compare(rows, a, b, metric="NDCG10", n=N_RESAMPLES, seed=SEED):
    pairs = paired_values(rows, a, b, metric)
    diffs = [vb - va for _, va, vb in pairs]
    lo, hi = bootstrap_ci(diffs, n, seed)
    return {"a": a, "b": b, "n": len(diffs), "mean_a": mean(va for _, va, _ in pairs),
            "mean_b": mean(vb for _, _, vb in pairs), "diff": mean(diffs), "ci_low": lo, "ci_high": hi,
            "p": randomization_p(diffs, n, seed), "wins": sum(d > 0 for d in diffs),
            "losses": sum(d < 0 for d in diffs), "ties": sum(d == 0 for d in diffs),
            "per_query": [(q, va, vb, vb - va) for q, va, vb in pairs]}


# ---- outputs ------------------------------------------------------------------
def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def write_per_query(rows, path):
    cols = ["query_id", "category", "system"] + METRICS + ["n_results", "unjudged", "pool_judged",
                                                           "pool_rel_strict", "pool_rel_lenient"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["query_id"], r["system"])):
            w.writerow({k: ("" if r[k] is None else (round(r[k], 6) if isinstance(r[k], float) else r[k]))
                        for k in cols})


def plot(summary, cis, path):
    """NDCG@10 and lenient P@10 by system, bootstrap 95% CI error bars; one panel per metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid, bar = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
    systems = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=surface)
    for ax, (metric, title) in zip(axes, [("NDCG10", "NDCG@10 (graded)"), ("P10_lenient", "P@10 (label ≥ 1)")]):
        vals = [summary[s][metric] or 0 for s in systems]
        lo = [vals[i] - (cis[s][metric][0] if cis[s][metric][0] is not None else vals[i]) for i, s in enumerate(systems)]
        hi = [(cis[s][metric][1] if cis[s][metric][1] is not None else vals[i]) - vals[i] for i, s in enumerate(systems)]
        ax.set_facecolor(surface)
        ax.bar(range(len(systems)), vals, width=0.6, color=bar, edgecolor=surface, linewidth=2, zorder=2)
        ax.errorbar(range(len(systems)), vals, yerr=[lo, hi], fmt="none", ecolor=ink2, elinewidth=1.2, capsize=4, zorder=3)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels([s.rsplit("_", 1)[0] + "\n" + s.rsplit("_", 1)[1] if s.endswith(("_log", "_capped")) else s
                            for s in systems], color=ink2, fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.tick_params(colors=ink2, length=0)
        ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    fig.text(0.01, 0.01, "Bars: mean over queries. Error bars: bootstrap 95% CI (queries resampled).",
             color=ink2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def write_summary(path, *, args, labels, rows, queries, summary, cis, comparisons, density, categories):
    systems = list(summary)
    undefined = sorted({r["query_id"] for r in rows if r["NDCG10"] is None})
    unjudged = {s: summary[s]["unjudged"] for s in systems}
    L = [f"# Retrieval metrics", "",
         f"- Runs: `{args.runs}`; labels: `{args.labels}` ({len(labels)} labeled pairs); queries: `{args.queries}`",
         f"- {len(queries)} queries, {len(systems)} systems, top {K} per system; macro averages over queries",
         f"- Relevance: strict = label 2, lenient = label ≥ 1",
         f"- NDCG: {GAIN_FORMULA}; IDCG from the best ordering of the query's judged pool",
         f"- Significance: paired randomization (sign-flip, {args.resamples:,} resamples) on per-query NDCG@10, "
         f"Holm-adjusted across the {len(comparisons)} system pairs; percentile bootstrap 95% CIs; seed {args.seed}", ""]
    if any(unjudged.values()):
        L += [f"> **Warning: unjudged results in the top 10, counted as 0:** "
              + ", ".join(f"{s} {n}" for s, n in unjudged.items() if n)
              + ". Metrics are biased low; label the full pool.", ""]
    else:
        L += ["Every top-10 result is judged (0 unjudged).", ""]
    if undefined:
        L += [f"Queries with no relevant pair in the pool (left out of NDCG@10 and R@10): {', '.join(undefined)}", ""]

    L += ["## By system", "",
          "| System | P@5 strict | P@5 lenient | P@10 strict | P@10 lenient | NDCG@10 [95% CI] | R@10 strict | R@10 lenient | Unjudged |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in systems:
        m, lo_hi = summary[s], cis[s]["NDCG10"]
        L.append(f"| {s} | {f3(m['P5_strict'])} | {f3(m['P5_lenient'])} | {f3(m['P10_strict'])} | {f3(m['P10_lenient'])} | "
                 f"{f3(m['NDCG10'])} [{f3(lo_hi[0])}, {f3(lo_hi[1])}] | {f3(m['R10_strict'])} | {f3(m['R10_lenient'])} | "
                 f"{m['unjudged']} |")

    cats = [c for c in CATEGORY_ORDER if c in set(categories.values())] + sorted(
        {r["category"] for r in rows} - set(CATEGORY_ORDER))
    for metric, title in (("NDCG10", "NDCG@10"), ("P10_lenient", "P@10 lenient")):
        L += ["", f"## {title} by query category", "",
              "| System | " + " | ".join(f"{c} (n={len({r['query_id'] for r in rows if r['category'] == c})})" for c in cats) + " |",
              "|---|" + "---|" * len(cats)]
        for s in systems:
            L.append(f"| {s} | " + " | ".join(
                f3(mean(r[metric] for r in rows if r["system"] == s and r["category"] == c)) for c in cats) + " |")

    L += ["", "## Paired comparisons (NDCG@10)", "",
          "Difference = B − A, over queries where both are defined.", "",
          "| A | B | n | mean A | mean B | B − A [95% CI] | p | p (Holm) | B better / worse / tie |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in comparisons:
        L.append(f"| {c['a']} | {c['b']} | {c['n']} | {f3(c['mean_a'])} | {f3(c['mean_b'])} | {f3(c['diff'])} "
                 f"[{f3(c['ci_low'])}, {f3(c['ci_high'])}] | {f3(c['p'])} | {f3(c['p_holm'])} | "
                 f"{c['wins']} / {c['losses']} / {c['ties']} |")

    L += ["", "## Density mode: log vs capped", ""]
    for c in density:
        if c is None:
            continue
        L += [f"### {c['a']} → {c['b']}", "",
              f"NDCG@10 {f3(c['mean_a'])} → {f3(c['mean_b'])}; difference {f3(c['diff'])} "
              f"[{f3(c['ci_low'])}, {f3(c['ci_high'])}], p = {f3(c['p'])} (unadjusted); "
              f"capped better on {c['wins']}, worse on {c['losses']}, tied on {c['ties']} queries.", "",
              "| Query | Category | log | capped | Δ |", "|---|---|---|---|---|"]
        for q, va, vb, d in sorted(c["per_query"], key=lambda t: (-abs(t[3]), t[0])):
            L.append(f"| {q} | {categories.get(q, '')} | {f3(va)} | {f3(vb)} | {d:+.3f} |")
        L.append("")
    L += ["Small query sets are noisy: treat differences whose CI includes 0 as unresolved.", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Retrieval metrics per system with significance tests")
    ap.add_argument("--runs", type=Path, default=RUNS_PATH)
    ap.add_argument("--labels", type=Path, required=True, help="llm_labels.jsonl or a labels CSV")
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--resamples", type=int, default=N_RESAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    for p in (args.runs, args.labels, args.queries):
        if not p.exists():
            sys.exit(f"Missing {p}")

    runs, labels, categories = read_runs(args.runs), read_labels(args.labels), read_categories(args.queries)
    rows, queries = evaluate(runs, labels, categories)
    summary = summarize(rows)
    cis = {s: {m: bootstrap_ci([r[m] for r in rows if r["system"] == s], args.resamples, args.seed)
               for m in ("NDCG10", "P10_lenient")} for s in summary}
    comparisons = [compare(rows, a, b, n=args.resamples, seed=args.seed) for a, b in combinations(summary, 2)]
    for c, p in zip(comparisons, holm([c["p"] for c in comparisons])):
        c["p_holm"] = p
    density = [compare(rows, a, b, n=args.resamples, seed=args.seed) if a in summary and b in summary else None
               for a, b in DENSITY_PAIRS]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_per_query(rows, args.out_dir / "metrics_per_query.csv")
    write_summary(args.out_dir / "metrics_summary.md", args=args, labels=labels, rows=rows, queries=queries,
                  summary=summary, cis=cis, comparisons=comparisons, density=density, categories=categories)
    if not args.no_plot:
        plot(summary, cis, args.out_dir / "metrics_ndcg_p10.png")

    total_unjudged = sum(s["unjudged"] for s in summary.values())
    print(f"{len(queries)} queries, {len(summary)} systems, {len(labels)} labeled pairs")
    for s, m in summary.items():
        print(f"  {s:20} NDCG@10 {f3(m['NDCG10'])}  P@10 lenient {f3(m['P10_lenient'])}  unjudged {m['unjudged']}")
    if total_unjudged:
        print(f"WARNING: {total_unjudged} unjudged top-10 results were counted as 0; label the full pool.")
    print(f"Wrote {args.out_dir / 'metrics_summary.md'}, metrics_per_query.csv"
          + ("" if args.no_plot else ", metrics_ndcg_p10.png"))


if __name__ == "__main__":
    main()
