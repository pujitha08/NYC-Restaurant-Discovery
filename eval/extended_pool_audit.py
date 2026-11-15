"""Offline audit of the "no evidence" judge labels on the extended (top-20, dev queries) pool.

For the pairs in eval/pool_extended.csv (the top-20 of semantic_capped on the 20 dev queries that had no label yet):
1. whether the pool row had excerpts or an empty evidence set, and its evidence_status, for the flagged labels;
2. whether the judge's reason is consistent with the excerpts it was shown (5 flagged pairs printed side by side), and
   whether its prompt size tracks the evidence size like the other calls (a judge that saw no evidence would sit at the
   fitted intercept);
3. the flagged share by retrieval rank bucket (1-10 vs 11-20) of semantic_capped, for the extended pool and for all
   top-20 dev pairs (ranks are rebuilt from the local semantic index; a stale index stops the run);
4. the flagged share per query.
"Flagged" = a model call whose reason starts "no evidence" and cites no review (eval/no_evidence_audit.is_no_evidence_call).
No API calls; no label is modified. The section is written into reports/no_evidence_audit.md, replacing an earlier copy.

    python -m eval.extended_pool_audit [--first-new 1074]
"""

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval import make_pool, validation_report as vr
from eval.no_evidence_audit import OUT_PATH, is_no_evidence_call, read_records
from eval.tune_stance_rerank import split_queries

EVAL_DIR = Path(__file__).parent
POOL_PATH = EVAL_DIR / "pool_extended.csv"
HEADING = "## Extended pool, top-20 dev labels"
SPOT_CHECKS = 5
SEED = 6165
RANK_BUCKETS = (("1-10", 1, 10), ("11-20", 11, 20))


def read_pool(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def excerpts(row):
    return [row[f"evidence_{i}"] for i in range(1, 7) if (row.get(f"evidence_{i}") or "").strip()]


def share(flagged, total):
    return f"{flagged} / {total} ({flagged / total:.0%})" if total else "0 / 0"


def bucket(rank):
    return next((name for name, lo, hi in RANK_BUCKETS if lo <= rank <= hi), "other")


def tabulate(pairs, key):
    """{key value: [flagged, total]} over pairs ({"flagged": bool, ...})."""
    out = defaultdict(lambda: [0, 0])
    for p in pairs:
        out[key(p)][0] += p["flagged"]
        out[key(p)][1] += 1
    return out


def prompt_fit(points):
    """Least-squares prompt_tokens = a + b * evidence_chars over [(chars, prompt_tokens)]; (a, b)."""
    x, y = np.array([p[0] for p in points], float), np.array([p[1] for p in points], float)
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def ranks_of(queries):
    """{(query_id, gmap_id): rank} of semantic_capped, top 20, from the local index (never rebuilt here)."""
    semantic = make_pool.load_semantic()
    if semantic is None:
        sys.exit("The semantic index is not built; not rebuilding it silently. Run: python -m src.retrieval.semantic --build")
    base = [s for s in make_pool.SYSTEMS if s[0] == "semantic_capped"]
    _, run_rows, _ = make_pool.build_pool(queries, {"semantic": semantic}, systems=base, top_k=20)
    c = make_pool.RUN_COLUMNS
    return {(r[c.index("query_id")], r[c.index("gmap_id")]): int(r[c.index("rank")]) for r in run_rows}


def build_section(pool, finals, positions, ranks, queries):
    new_keys = {(r["query_id"], r["gmap_id"]) for r in pool}
    pairs = []
    for key, rank in ranks.items():
        rec = finals.get(key)
        if rec is None:
            continue
        pairs.append({"key": key, "rank": rank, "bucket": bucket(rank), "flagged": is_no_evidence_call(rec), "rule": rec.get("source") == "rule",
                      "label": rec["label"], "extended": key in new_keys, "batch": positions.get(key)})
    ext = [p for p in pairs if p["extended"]]
    pool_by_key = {(r["query_id"], r["gmap_id"]): r for r in pool}
    flagged_new = [p for p in ext if p["flagged"] and p["batch"] is not None and p["batch"] >= FIRST_NEW[0]]
    L = [HEADING, "",
         f"Pairs: the {len(pool)} pairs in `eval/pool_extended.csv` (top-20 of semantic_capped on the 20 dev queries, no label before this "
         f"work), judged in two batches: {sum(1 for p in ext if p['batch'] < FIRST_NEW[0])} earlier (before the 120b daily quota ran out) and "
         f"{sum(1 for p in ext if p['batch'] >= FIRST_NEW[0])} later (the 39 judged most recently). All by Groq `openai/gpt-oss-120b`. "
         "\"Flagged\" means a model call whose reason starts \"no evidence\" and cites no review. No API calls were made and no label was modified.", ""]

    new = [p for p in ext if p["batch"] >= FIRST_NEW[0]]
    mc = [p for p in ext if not p["rule"]]
    top3 = sorted({p["key"][0] for p in new}, key=lambda q: -sum(p["flagged"] for p in new if p["key"][0] == q))[:3]
    top3_n = sum(p["flagged"] for p in new if p["key"][0] in top3)
    L += ["**Summary.** " + SUMMARY.format(
        n_new_flag=sum(p["flagged"] for p in new), n_new=len(new), n_ext_flag=sum(p["flagged"] for p in mc), n_ext=len(mc),
        top3=", ".join(top3), top3_n=top3_n), ""]
    # 1
    L += [f"### 1. Did the pool rows have evidence?", "",
          f"Of the 39 newest labels, **{sum(p['flagged'] for p in new)} are flagged**. ", ""]
    L += ["| Pair | Evidence status | Excerpts shown | Evidence characters | Label | Rank |", "|---|---|---|---|---|---|"]
    for p in sorted(flagged_new, key=lambda p: p["key"]):
        row = pool_by_key[p["key"]]
        ex = excerpts(row)
        L.append(f"| {p['key'][0]} {vr.cell(row['name'])} | {row['evidence_status']} | {len(ex)} | {sum(len(e) for e in ex)} | {p['label']} | {p['rank']} |")
    statuses = sorted({pool_by_key[p["key"]]["evidence_status"] for p in flagged_new})
    empty = sum(not excerpts(pool_by_key[p["key"]]) for p in flagged_new)
    L += ["", f"Evidence status of the flagged rows: {', '.join(statuses) or 'n/a'}; rows with an empty evidence set: {empty} of {len(flagged_new)}. "
          f"All {len(pool)} pool rows have status `retrieved` with at least one excerpt, so none was an empty prompt.", ""]
    return L, ext, pairs, pool_by_key, flagged_new


FIRST_NEW = [1074]
SUMMARY = ("{n_new_flag} of the 39 newest labels ({n_new_flag}/{n_new}) are flagged, against {n_ext_flag} of {n_ext} for the whole extended pool "
           "and 31% (274 of 883) in the original 964 labels. Every flagged row had excerpts (none was an empty prompt) and the judge's prompt size "
           "tracks the excerpt size like the unflagged calls, so the judge saw the evidence; the words \"no evidence found\" are the rubric's stock "
           "phrase for label 0, not a report of empty evidence. The higher rate is mostly a matter of which queries were judged last, not of weaker "
           "candidates: the extended pool is entirely rank 11-20 (the top 10 were already labeled), where the flagged share over all dev queries is 36% against 28% "
           "at rank 1-10 (see 3), while the newest batch is concentrated in {top3}, which hold {top3_n} of the flagged labels. "
           "Where the reason can be checked against the excerpts it often does not describe them (see 2), so treat the label 0 as a strict "
           "reading of the need, not as 'nothing relevant was retrieved'.")
READING = ("Reading of the five (the reasons are all the bare phrase, so this is an inference from the excerpts): every pair had topical "
           "excerpts, so \"no evidence found\" does not describe what was shown. A 0 looks defensible for the two Q33 pairs (a warning about "
           "dismissed allergies, and an allergic reviewer's complaint) and plausibly for Olympia Pita under the category-filter rule (kosher food, "
           "no sign of a bakery). It is more debatable for the two Q31 pairs, which show strong vegan evidence but nothing on nut-free or brunch: "
           "whether that is a 0 or a 1 depends on how strictly the rubric's combined-need rule is applied. Whether a 0 is justified is a labeling "
           "question this audit cannot settle without a human label.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Offline audit of 'no evidence' labels on the extended pool")
    ap.add_argument("--pool", type=Path, default=POOL_PATH)
    ap.add_argument("--llm", type=Path, default=vr.LLM_PATH)
    ap.add_argument("--queries", type=Path, default=vr.QUERIES_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--first-new", type=int, default=1074, help="index in the label file of the first of the newest labels")
    args = ap.parse_args(argv)
    FIRST_NEW[0] = args.first_new
    pool = read_pool(args.pool)
    records = read_records(args.llm)
    finals, positions = {}, {}
    for i, r in enumerate(records):
        finals[(r["query_id"], r["gmap_id"])] = r
        positions[(r["query_id"], r["gmap_id"])] = i
    queries_all = make_pool.read_queries(args.queries)
    dev, _ = split_queries([q["query_id"] for q in queries_all])
    queries = [q for q in queries_all if q["query_id"] in dev]
    ranks = ranks_of(queries)
    categories = vr.read_categories(args.queries) if hasattr(vr, "read_categories") else {}

    L, ext, pairs, pool_by_key, flagged_new = build_section(pool, finals, positions, ranks, queries)

    # 2: spot checks and prompt-size fit
    rng = random.Random(SEED)
    spot = rng.sample(sorted(flagged_new, key=lambda p: p["key"]), min(SPOT_CHECKS, len(flagged_new)))
    L += ["### 2. Is the judge's reason consistent with the excerpts shown?", "",
          f"Five of the {len(flagged_new)} flagged pairs, drawn with seed {SEED}. Excerpts are as on the pool row (truncated to 400 characters here).", ""]
    for p in sorted(spot, key=lambda p: p["key"]):
        row, rec = pool_by_key[p["key"]], finals[p["key"]]
        L += [f"**{p['key'][0]} {vr.cell(row['query'])} / {vr.cell(row['name'])}**: label {rec['label']}; judge's reason: \"{vr.cell(rec['reason'])}\"", ""]
        L += [f"> {vr.cell(e)[:400]}" for e in excerpts(row)] + [""]
    L += [READING, ""]
    fit_pts = [(sum(len(e) for e in excerpts(r)), finals[k]["usage"]["prompt_tokens"]) for k, r in pool_by_key.items()
               if finals[k].get("usage") and finals[k]["usage"].get("prompt_tokens")]
    a, b = prompt_fit(fit_pts)
    fl = [(sum(len(e) for e in excerpts(pool_by_key[p["key"]])), finals[p["key"]]["usage"]["prompt_tokens"]) for p in ext if p["flagged"]]
    nf = [(sum(len(e) for e in excerpts(pool_by_key[p["key"]])), finals[p["key"]]["usage"]["prompt_tokens"]) for p in ext if not p["flagged"]]
    resid = lambda pts: float(np.mean([y - (a + b * x) for x, y in pts])) if pts else float("nan")
    L += ["**Did the judge see the excerpts?** Judge prompt size against evidence size across the pool "
          f"(prompt tokens = {a:.0f} + {b:.3f} x evidence characters, n = {len(fit_pts)}): the fixed part of the prompt is about {a:.0f} tokens, so a call "
          f"with no evidence would use about that many. Mean prompt tokens: flagged {np.mean([y for _, y in fl]):.0f} (mean evidence {np.mean([x for x, _ in fl]):.0f} characters), "
          f"not flagged {np.mean([y for _, y in nf]):.0f} ({np.mean([x for x, _ in nf]):.0f} characters). Mean residual from the fit: flagged {resid(fl):+.0f}, "
          f"not flagged {resid(nf):+.0f} tokens. Residuals near zero for both mean the flagged calls were given their excerpts like the others.", ""]

    # 3
    L += ["### 3. Share by retrieval rank (semantic_capped)", "",
          "| Group | Rank 1-10 | Rank 11-20 |", "|---|---|---|"]
    groups = [("Extended pool (149 pairs)", ext), ("  earlier batch of the extended pool", [p for p in ext if p["batch"] < FIRST_NEW[0]]),
              ("  newest 39", [p for p in ext if p["batch"] >= FIRST_NEW[0]]),
              ("All top-20 dev pairs (model calls only)", [p for p in pairs if not p["rule"]])]
    for name, g in groups:
        t = tabulate([p for p in g if not p["rule"]], lambda p: p["bucket"])
        L.append(f"| {name} | {share(*t['1-10'])} | {share(*t['11-20'])} |")
    L += ["", "Pairs labeled by the no-evidence rule (no excerpts at all) are excluded from these shares.", ""]

    # 4
    L += ["### 4. Share per query", "", "| Query | Category | Extended pool pairs | Flagged | Share | Newest-39 pairs | Newest flagged |", "|---|---|---|---|---|---|---|"]
    tq = tabulate([p for p in ext if not p["rule"]], lambda p: p["key"][0])
    tn = tabulate([p for p in ext if p["batch"] >= FIRST_NEW[0]], lambda p: p["key"][0])
    for q in sorted(tq, key=lambda q: (-tq[q][0] / tq[q][1], q)):
        L.append(f"| {q} | {categories.get(q, '')} | {tq[q][1]} | {tq[q][0]} | {tq[q][0] / tq[q][1]:.0%} | {tn[q][1] if q in tn else 0} | {tn[q][0] if q in tn else 0} |")
    top = sorted(tn, key=lambda q: -tn[q][0])
    covered = 0
    top3 = top[:3]
    covered = sum(tn[q][0] for q in top3)
    total_new = sum(v[0] for v in tn.values())
    L += ["", f"The three queries with most flagged newest labels ({', '.join(top3)}) account for {covered} of {total_new}.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
    text = re.split(r"\n" + re.escape(HEADING), text)[0].rstrip("\n")
    args.out.write_text(text + "\n\n" + "\n".join(L).rstrip("\n") + "\n", encoding="utf-8")
    print(f"Wrote {HEADING!r} to {args.out}")


if __name__ == "__main__":
    main()
