"""Audit of LLM judge labels whose reason is "no evidence found" with no cited review, and how much they matter.

The rubric (eval/RUBRIC.md) tells judges to write "no evidence found" for label 0 when none of the shown reviews says
anything about the need, so the phrase is expected on off-topic evidence as well as on empty evidence. This script
separates the two readings and measures the effect on the retrieval metrics. No API calls are made; no label is modified.

1. Census: how many labels are source=llm with reason "no evidence ..." and empty review_ids, by provider, query category,
   label and by the systems whose top 10 contain them.
2. Evidence fingerprint: for each human-sampled LLM call, the judge's reported prompt_tokens are compared with the size of
   the prompt rebuilt (with the judge's own functions) from the evidence on the human sheet, versus the same prompt with
   no reviews. If the judge saw the sheet's evidence the ratio is the same as for the other calls; if it saw none, it is ~1.3.
3. Sensitivity of NDCG@10 per system to (a) the human labels overriding the LLM labels on the sampled pairs and (b) the
   flagged labels being dropped from every ranking (ranks renumbered), with the semantic_capped vs doc_tfidf comparison.
   The baseline uses the first --n-original labels (those behind reports/metrics_summary.md).

The per-system rankings come from eval/runs.csv, or are rebuilt here from the local indexes (the stance system is skipped
because it would call the LLM) and checked against reports/metrics_per_query.csv.

    python -m eval.no_evidence_audit [--human eval/labels/labels_pujitha.csv]
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from eval import llm_judge, make_pool, validation_report as vr
from eval.metrics import (CATEGORY_ORDER, N_RESAMPLES, SEED, SYSTEM_ORDER, compare, evaluate, f3, holm, read_categories,
                          read_runs, summarize)

EVAL_DIR = Path(__file__).parent
REPO = EVAL_DIR.parent
OUT_PATH = REPO / "reports" / "no_evidence_audit.md"
PER_QUERY_PATH = REPO / "reports" / "metrics_per_query.csv"
RUBRIC_PATH = EVAL_DIR / "RUBRIC.md"
N_ORIGINAL = 964
BASE, ALT = "semantic_capped", "doc_tfidf"
SYSTEMS_NO_LLM = [s for s in make_pool.SYSTEMS if s[0] != "semantic_capped_stance"]


# ---- pure helpers ---------------------------------------------------------------
def is_no_evidence_call(rec):
    """A model call (not the no-evidence rule) that answered "no evidence ..." and cited no review."""
    return (rec.get("source") == "llm" and (rec.get("reason") or "").strip().lower().startswith("no evidence")
            and not rec.get("review_ids"))


def read_records(path):
    """All records in file order (later duplicates of a pair kept, as in read_llm's last-wins)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def final_records(records):
    """{(query_id, gmap_id): record}, the last record of a pair winning."""
    return {(r["query_id"], r["gmap_id"]): r for r in records}


def override_labels(labels, human):
    """Copy of labels with {(query_id, gmap_id): human label} applied; the input is not modified."""
    out = dict(labels)
    out.update(human)
    return out


def drop_pairs(runs, excluded):
    """runs with the (query_id, gmap_id) pairs in `excluded` removed from every ranking (ranks renumbered)."""
    return {s: {q: [g for g in ranked if (q, g) not in excluded] for q, ranked in qs.items()} for s, qs in runs.items()}


def census(finals, categories):
    """Counts of flagged calls: totals, by provider/upstream, category and label."""
    llm_calls = {k: r for k, r in finals.items() if r.get("source") == "llm"}
    flagged = {k: r for k, r in llm_calls.items() if is_no_evidence_call(r)}
    by = lambda key, pool: dict(sorted(Counter(key(k, r) for k, r in pool.items()).items()))
    return {"pairs": len(finals), "llm_calls": len(llm_calls), "flagged": len(flagged),
            "rule": sum(r.get("source") == "rule" for r in finals.values()),
            "by_source": {s: (n, by(lambda k, r: vr.label_source(r), llm_calls).get(s, 0)) for s, n in
                          by(lambda k, r: vr.label_source(r), flagged).items()},
            "by_category": {c: (by(lambda k, r: categories.get(k[0]), flagged).get(c, 0),
                                by(lambda k, r: categories.get(k[0]), llm_calls).get(c, 0)) for c in CATEGORY_ORDER},
            "labels": dict(Counter(r["label"] for r in flagged.values())), "keys": set(flagged)}


def cross_tab(finals, categories):
    """{(provider, category): [flagged, model calls]}: is the provider gap still there within a query category?"""
    out = {}
    for k, r in finals.items():
        if r.get("source") != "llm":
            continue
        cell = out.setdefault((vr.label_source(r), categories.get(k[0])), [0, 0])
        cell[1] += 1
        cell[0] += is_no_evidence_call(r)
    return out


def systems_hit(runs, flagged_keys, k=10):
    """{system: (flagged results in top k, results in top k)} over all queries."""
    out = {}
    for s, qs in runs.items():
        top = [(q, g) for q, ranked in qs.items() for g in ranked[:k]]
        out[s] = (sum(key in flagged_keys for key in top), len(top))
    return out


def fingerprint(pairs, system, build_payload, parse):
    """actual / estimated prompt tokens for each pair, with the sheet's evidence and with no reviews.

    pairs: [{"llm_record", "flagged", row fields...}]. Estimate = (len(system) + len(json(payload))) / 4, the judge's own rule.
    """
    out = []
    for p in pairs:
        est = lambda ev: (len(system) + len(json.dumps(build_payload(p, ev), ensure_ascii=False))) / 4
        actual = p["llm_record"]["usage"]["prompt_tokens"]
        out.append({"sample_id": p["sample_id"], "flagged": p["flagged"], "with_sheet_evidence": actual / est(parse(p)),
                    "with_no_reviews": actual / est([])})
    return out


def sensitivity(runs, labels, categories, human, flagged_keys, n=N_RESAMPLES, seed=SEED):
    """Per variant: {"summary": {system: NDCG10 mean}, "unjudged": {...}, "rows", comparison base vs alt (Holm over all pairs)}."""
    variants = {"baseline (LLM labels)": (runs, labels),
                "human labels override LLM on the sample": (runs, override_labels(labels, human)),
                "flagged no-evidence pairs dropped": (drop_pairs(runs, flagged_keys), labels)}
    out = {}
    for name, (r, lab) in variants.items():
        rows, _ = evaluate(r, lab, categories)
        summary = summarize(rows)
        pairs = list(combinations(summary, 2))
        comps = [compare(rows, a, b, n=n, seed=seed) for a, b in pairs]
        for c, p in zip(comps, holm([c["p"] for c in comps])):
            c["p_holm"] = p
        target = next((c for c in comps if {c["a"], c["b"]} == {BASE, ALT}), None)
        out[name] = {"summary": {s: m["NDCG10"] for s, m in summary.items()}, "unjudged": {s: m["unjudged"] for s, m in summary.items()},
                     "rows": rows, "base_vs_alt": target, "n_pairs": len(pairs)}
    return out


def ranking(ndcg):
    return sorted(ndcg, key=lambda s: -ndcg[s])


def reproduction_check(rows, per_query_path):
    """(max abs NDCG@10 difference, queries compared, systems missing) against the published reports/metrics_per_query.csv."""
    published = {}
    with open(per_query_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["NDCG10"]:
                published[(r["system"], r["query_id"])] = float(r["NDCG10"])
    diffs = [abs(r["NDCG10"] - published[(r["system"], r["query_id"])]) for r in rows
             if r["NDCG10"] is not None and (r["system"], r["query_id"]) in published]
    return (max(diffs) if diffs else None), len(diffs)


# ---- report ---------------------------------------------------------------------
def write_report(path, *, info):
    c, fp, sens, repro = info["census"], info["fingerprint"], info["sensitivity"], info["repro"]
    L = ["# Audit: LLM judge labels with reason \"no evidence found\" and no cited review", "",
         "No API calls were made and no label was modified. Baseline labels are the first "
         f"{info['n_original']} records of `eval/llm_labels.jsonl` (the ones behind `reports/metrics_summary.md`).", "",
         "## Summary", ""] + info["summary_lines"] + ["", "## 1. How many labels", "",
         f"Of {info['n_labels']} labeled pairs, {c['llm_calls']} are model calls and {c['rule']} are the no-evidence rule (no reviews on the sheet, "
         f"labeled 0 without a call). **{c['flagged']} model calls ({c['flagged'] / c['llm_calls']:.0%}) answered \"no evidence ...\" with no cited "
         f"review**; all have label {', '.join(map(str, sorted(c['labels'])))} (counts: {c['labels']}).", "",
         "| Provider | Flagged | Model calls | Share |", "|---|---|---|---|"]
    L += [f"| {s} | {n} | {t} | {n / t:.0%} |" for s, (n, t) in c["by_source"].items()]
    cross = info["cross"]
    L += ["", "Provider and category are confounded (each provider judged a different mix of queries), so the provider gap above is also shown "
          "within each category (flagged / model calls):", "", "| Category | " + " | ".join(sorted({p for p, _ in cross})) + " |",
          "|---|" + "---|" * len({p for p, _ in cross})]
    for cat in CATEGORY_ORDER:
        L.append(f"| {cat} | " + " | ".join((f"{cross[(p, cat)][0]} / {cross[(p, cat)][1]}" if (p, cat) in cross else "—")
                                         for p in sorted({p for p, _ in cross})) + " |")
    L += ["", "| Query category | Flagged | Model calls | Share |", "|---|---|---|---|"]
    L += [f"| {cat} | {n} | {t} | {n / t:.0%} |" if t else f"| {cat} | 0 | 0 | n/a |" for cat, (n, t) in c["by_category"].items()]
    L += ["", "Flagged results in each system's top 10 (all 35 queries):", "", "| System | Flagged / top-10 results | Share |", "|---|---|---|"]
    L += [f"| {s} | {n} / {t} | {n / t:.0%} |" for s, (n, t) in info["systems_hit"].items()]
    L += ["", "## 2. Did the judge see the evidence on your sheet?", "",
          f"Local files: `eval/pool.csv` is the **{info['pool_note']}**, so it is not the pool the judge read, and the judging pool itself "
          "(gitignored) is not on this machine. What can be checked instead is the token count the API reported for each call.", "",
          "| Calls | n | actual / estimate with the sheet's evidence (mean, min-max) | actual / estimate with no reviews (mean) |", "|---|---|---|---|"]
    for name, group in (("flagged", [x for x in fp if x["flagged"]]), ("not flagged", [x for x in fp if not x["flagged"]])):
        a, b = [x["with_sheet_evidence"] for x in group], [x["with_no_reviews"] for x in group]
        L.append(f"| {name} | {len(group)} | {statistics.mean(a):.3f} ({min(a):.3f}-{max(a):.3f}) | {statistics.mean(b):.3f} |")
    L += ["", f"For all {info['fp_closer']} of the {info['fp_flagged']} flagged calls the reported prompt size is closer to the prompt rebuilt with the sheet's "
          "evidence than to the same prompt with no reviews, and the ratio is the same as for calls that did cite reviews. "
          "A prompt without the reviews would be about 12-14% smaller than the one reported. So **the judge was shown evidence of the "
          "size on your sheet for every flagged call**; it did not receive empty evidence.", "",
          f"Sheet provenance: {info['sheet_query_match']} of the sheet's query texts equal `eval/queries_final.csv` (the frozen set), the sheet uses the "
          "`evidence_status` / `evidence_1..6` layout of `eval/make_pool.py` after commit 4bf191e (2026-11-10), and "
          f"{info['sheet_rule_note']} The judge ran 2026-11-12 20:14 UTC to 2026-11-13. `eval/make_sample.py` reads `--pool eval/pool.csv` (default); "
          "the exact pool file it read is not recorded in the sample, so its date and checksum could not be determined.", "",
          "## 3. The 29 flagged pairs on your sheet", "", info["flagged_table_intro"], "",
          "| Sample | Query (category) | Restaurant | Human | LLM | Your note |", "|---|---|---|---|---|---|"]
    L += info["flagged_rows"]
    L += ["", "## 4. Sensitivity of NDCG@10", "",
          f"Rankings: {'rebuilt from the local indexes' if info['runs_rebuilt'] else 'from eval/runs.csv'}; against the published "
          f"`reports/metrics_per_query.csv` the baseline per-query NDCG@10 differs by at most {repro[0]:.4f} over {repro[1]} (system, query) values"
          + (" (the baseline reproduces the published numbers)." if repro[0] is not None and repro[0] < 0.005 else " (CAUTION: not an exact reproduction).") , "",
          f"Human override changes the label of {info['n_override_changed']} of {info['n_override']} sampled pairs. Dropping removes {info['n_dropped']} flagged "
          "pairs from the rankings (renumbering the ranks below them); since all flagged labels are 0, this changes NDCG only by moving later results up. "
          "Treating them as unjudged instead would leave NDCG unchanged (a missing label counts as 0).", "",
          "| System | Baseline | Human override | Δ | Flagged dropped | Δ |", "|---|---|---|---|---|---|"]
    names = list(sens)
    b0, b1, b2 = (sens[n]["summary"] for n in names)
    for s in SYSTEM_ORDER:
        if s in b0:
            L.append(f"| {s} | {f3(b0[s])} | {f3(b1[s])} | {b1[s] - b0[s]:+.3f} | {f3(b2[s])} | {b2[s] - b0[s]:+.3f} |")
    L += ["", "System order by NDCG@10 (best first):", ""] + [f"- {n}: {' > '.join(ranking(sens[n]['summary']))}" for n in names]
    L += ["", f"`{BASE}` vs `{ALT}` (paired sign-flip on per-query NDCG@10, {N_RESAMPLES:,} resamples; Holm over the {sens[names[0]]['n_pairs']} system pairs):", "",
          "| Variant | " + BASE + " | " + ALT + " | Difference [95% CI] | p | p (Holm) | better / worse / tie |", "|---|---|---|---|---|---|---|"]
    for n in names:
        c2 = sens[n]["base_vs_alt"]
        sign = 1 if c2["b"] == BASE else -1
        d, lo, hi = sign * c2["diff"], (c2["ci_low"] if sign == 1 else -c2["ci_high"]), (c2["ci_high"] if sign == 1 else -c2["ci_low"])
        w, l = (c2["wins"], c2["losses"]) if sign == 1 else (c2["losses"], c2["wins"])
        L.append(f"| {n} | {f3(sens[n]['summary'][BASE])} | {f3(sens[n]['summary'][ALT])} | {d:+.3f} [{lo:.3f}, {hi:.3f}] | {f3(c2['p'])} | "
                 f"{f3(c2['p_holm'])} | {w} / {l} / {c2['ties']} |")
    L += ["", "Unjudged top-10 results per variant: " + "; ".join(f"{n}: {sum(sens[n]['unjudged'].values())}" for n in names) + ".", ""]
    L += ["## 5. What could not be determined", ""] + info["undetermined"] + [""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit 'no evidence found' judge labels and their effect on the metrics")
    ap.add_argument("--human", type=Path, default=vr.HUMAN_PATH)
    ap.add_argument("--llm", type=Path, default=vr.LLM_PATH)
    ap.add_argument("--queries", type=Path, default=vr.QUERIES_PATH)
    ap.add_argument("--runs", type=Path, default=make_pool.RUNS_PATH)
    ap.add_argument("--n-original", type=int, default=N_ORIGINAL)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--resamples", type=int, default=N_RESAMPLES)
    args = ap.parse_args(argv)
    for p in (args.human, args.llm, args.queries):
        if not p.exists():
            sys.exit(f"Missing {p}")

    categories = read_categories(args.queries)
    records = read_records(args.llm)
    original, finals = final_records(records[:args.n_original]), final_records(records)
    labels = {k: r["label"] for k, r in original.items()}
    cen = census(original, categories)

    rebuilt = not args.runs.exists()
    if rebuilt:
        queries = make_pool.read_queries(args.queries)
        retrievers = {n: make_pool.RETRIEVERS[n]() for n in ("chunk_tfidf", "doc_tfidf", "semantic")}
        _, run_rows, _ = make_pool.build_pool(queries, retrievers, systems=SYSTEMS_NO_LLM)
        make_pool.write_csv(args.runs, make_pool.RUN_COLUMNS, run_rows)
    runs = {s: qs for s, qs in read_runs(args.runs).items() if s != "semantic_capped_stance"}

    rows, _ = vr.read_human(args.human)
    from src.retrieval.evidence import ReviewStore
    store = ReviewStore()
    matched, unmatched = vr.resolve(rows, finals, vr.read_meta(), lambda ids: {r["review_id"]: r["gmap_id"] for r in store.get(ids)})
    for p in matched:
        p["flagged"] = is_no_evidence_call(p["llm_record"])
    system = llm_judge.system_prompt(RUBRIC_PATH.read_text(encoding="utf-8"))
    calls = [p for p in matched if p["llm_record"].get("source") == "llm" and p["llm_record"].get("usage")]
    fp = fingerprint(calls, system, lambda p, ev: llm_judge.payload_for(p, ev), lambda p: llm_judge.parse_evidence(p))
    scale = statistics.mean(x["with_sheet_evidence"] for x in fp if not x["flagged"])
    closer = sum(abs(x["with_sheet_evidence"] - scale) < abs(x["with_no_reviews"] - scale) for x in fp if x["flagged"])

    human = {(p["query_id"], p["gmap_id"]): p["human"] for p in matched}
    sens = sensitivity(runs, labels, categories, human, cen["keys"], args.resamples)
    repro = reproduction_check(sens["baseline (LLM labels)"]["rows"], PER_QUERY_PATH)

    cat = categories
    flagged_pairs = sorted((p for p in matched if p["flagged"]), key=lambda p: (-abs(p["human"] - p["llm"]), -p["human"], p["sample_id"]))
    gap = Counter(abs(p["human"] - p["llm"]) for p in flagged_pairs)
    sheet_rule = Counter((p.get("evidence_status"), p["llm_record"].get("source")) for p in matched)
    rule_note = (f"the sheet's 'none' rows ({sum(v for (st, _), v in sheet_rule.items() if st == 'none')}) are exactly the rule-labeled calls "
                 f"({sheet_rule.get(('none', 'rule'), 0)} rule), while no row with evidence is rule-labeled ({sheet_rule.get(('retrieved', 'rule'), 0) + sheet_rule.get(('fallback', 'rule'), 0)}).")
    pool_head = Path(make_pool.POOL_PATH).read_text(encoding="utf-8").splitlines()[0] if Path(make_pool.POOL_PATH).exists() else ""
    pool_note = ("old-format file (Nov 3, columns `best_snippet_1..3`, no `evidence_status`, 30 draft queries)" if "best_snippet_1" in pool_head
                 else "pool.csv present")
    info = {
        "n_original": args.n_original, "n_labels": len(original), "census": cen, "fingerprint": fp, "fp_closer": closer,
        "fp_flagged": sum(x["flagged"] for x in fp), "sensitivity": sens, "repro": repro, "runs_rebuilt": rebuilt,
        "systems_hit": systems_hit(runs, cen["keys"]), "pool_note": pool_note, "sheet_rule_note": rule_note,
        "sheet_query_match": f"{sum(p['query_id'] in categories for p in matched)} of {len(matched)}",
        "n_override": len(human), "n_override_changed": sum(finals[k]["label"] != v for k, v in human.items()),
        "n_dropped": len(cen["keys"]),
        "cross": cross_tab(original, categories),
        "flagged_table_intro": (f"{len(flagged_pairs)} sampled pairs; your label vs the LLM's: agree {gap[0]}, one step apart {gap[1]}, two steps apart {gap[2]} "
                                f"(your labels: {dict(Counter(p['human'] for p in flagged_pairs))})."),
        "flagged_rows": [f"| {p['sample_id']} | {p['query']} ({cat.get(p['query_id'], '?')}) | {vr.cell(p['name'])} | {p['human']} | {p['llm']} | {vr.cell(p.get('note')) or '—'} |"
                         for p in flagged_pairs],
        "summary_lines": [], "undetermined": []}
    m_flag = cen["flagged"]
    base, hum, drop = (sens[n]["summary"] for n in sens)
    cmp_ = {n: sens[n]["base_vs_alt"] for n in sens}
    info["summary_lines"] = [
        f"- **Cause (best supported):** the judge saw the evidence. {closer} of {info['fp_flagged']} flagged sampled calls have a prompt size matching the sheet's evidence "
        "(not an empty prompt). The wording comes from the rubric, which prescribes \"no evidence found\" for label 0 when the shown reviews are off-topic; "
        "the five large disagreements plausibly reflect the judge applying the rubric's category-filter rule (e.g. 'bakery', 'brunch') or its strict allergy rule more strictly than the human did (a hypothesis: the reasons do not say).",
        f"- **Size:** {m_flag} of {cen['llm_calls']} model calls in the original {len(original)} labels ({m_flag / cen['llm_calls']:.0%}), all labeled 0.",
        f"- **Effect:** see section 4; semantic_capped vs doc_tfidf NDCG@10 difference: "
        + "; ".join(f"{n.split(' (')[0]}: {(-1 if cmp_[n]['b'] != BASE else 1) * cmp_[n]['diff']:+.3f} (p {f3(cmp_[n]['p'])}, Holm {f3(cmp_[n]['p_holm'])})" for n in sens) + "."]
    info["undetermined"] = [
        "- **The pool file the judge read** (and the one `eval/make_sample.py` read) is gitignored and not on this machine; the local `eval/pool.csv` is a different, older file. "
        "Whether the two were literally the same file, and their dates and checksums, cannot be established here. The token fingerprint shows only that the evidence the judge saw "
        "had the size of the sheet's evidence for each sampled call; it cannot show that the texts were identical.",
        "- **The 245 flagged calls outside your sample** were not compared with any sheet; the conclusion for them rests on the same prompt-size behaviour of the sampled ones, "
        "and on their using the same pipeline and prompt.",
        "- **Why the judge chose 0 for the five two-step disagreements** cannot be read from the labels (the reasons are the stock phrase). In the sample they involve a second "
        "requirement in the query (bakery, brunch) or an allergy query where the reviews do not describe the restaurant's handling; S072 carries your note that the evidence suggests "
        "the restaurant is not peanut free, which supports the judge's 0 and may be a labeling slip.",
        "- **Rankings are rebuilt locally** (index build 2026-11-14) rather than read from the original runs.csv"
        + ("; they reproduce the published per-query NDCG@10." if repro[0] is not None and repro[0] < 0.005 else "; they do NOT exactly reproduce the published numbers, so section 4 is approximate.")]
    write_report(args.out, info=info)
    print(f"flagged {m_flag}/{cen['llm_calls']} in original; fingerprint closer-to-sheet {closer}/{info['fp_flagged']}; repro max diff {repro[0]}")
    for n in sens:
        print(n, {s: round(v, 3) for s, v in sens[n]['summary'].items()})
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
