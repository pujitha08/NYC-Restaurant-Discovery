"""End-to-end answer evaluation: summarizer claims and usability.

Subcommands (run from the repo root):

  run     Send the evaluation queries through a checkout's app/pipeline.py and save the outcomes:
              python eval/answer_eval.py run --repo <checkout> --version old --method <m> --out eval/answer_eval/outputs_old.json
          --repo selects which code runs (e.g. a worktree of feature/streamlit-app for the old summarizer);
          the same --method is used for both versions so they summarize the same restaurants. Searches
          are paced (--gap seconds) to stay under Groq's per-minute token limit.

  sheets  From the old and fixed outputs, write the rater files (default eval/answer_eval/):
            claim_review.csv     one row per summary claim of the top 3 results, both versions, shuffled,
                                 version hidden; columns supported_by_excerpt (yes/partly/no) and
                                 unjustified_safety_claim (yes/no) to fill
            claim_key.csv        claim_id -> version, query_id, gmap_id (not for the reviewer)
            usability_<rater>.csv  per query, the top 3 results with their summaries (fixed version);
                                 the rater fills usefulness (1-5)

  score   Unsupported-claim rate (strict: "no"; lenient: "no" or "partly") and unjustified-safety-claim
          count per version, plus the usability mean (per rater and overall) and Krippendorff's alpha
          (ordinal) across raters, with eval/stats.py.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "eval" / "answer_eval"
QUERIES_PATH = REPO / "eval" / "queries_final.csv"
EVAL_QUERIES = ["Q01", "Q08", "Q13", "Q32", "Q20", "Q22", "Q25", "Q35", "Q27", "Q29"]
TOP_N = 3
SEED = 6165
CLAIM_COLUMNS = ["claim_id", "query", "restaurant", "status_shown", "claim_label", "claim", "cited_excerpts",
                 "supported_by_excerpt", "unjustified_safety_claim", "note"]
USABILITY_COLUMNS = (["query_id", "query", "borough"]
                     + [f"{c}_{i}" for i in range(1, TOP_N + 1) for c in ("result", "summary")]
                     + ["usefulness", "note"])
SUPPORTED_VALUES, SAFETY_VALUES = {"yes", "partly", "no"}, {"yes", "no"}


# ---- run ----------------------------------------------------------------------
def read_queries(path=QUERIES_PATH, ids=EVAL_QUERIES):
    with open(path, newline="", encoding="utf-8") as f:
        rows = {r["query_id"]: r for r in csv.DictReader(f)}
    missing = [q for q in ids if q not in rows]
    if missing:
        raise ValueError(f"queries not in {path}: {missing}")
    return [rows[q] for q in ids]


def outcome_record(pipeline, out, query_row, version):
    def excerpt(e):
        return {"review_id": e.review_id, "date": e.date, "text": e.text}

    results = []
    for v in out.results:
        s = v.summary
        results.append({
            "rank": v.rank, "gmap_id": v.gmap_id, "name": v.name, "borough": v.borough, "avg_rating": v.avg_rating,
            "low_evidence": v.low_evidence, "excerpts": [excerpt(e) for e in v.excerpts],
            "summary": None if s is None else {"status": s.status, "note": getattr(s, "note", None),
                                               "claims": [{"label": c.label, "text": c.text,
                                                           "excerpts": [excerpt(e) for e in c.excerpts]} for c in s.claims],
                                               "limitations": s.limitations},
            "summary_error": v.summary_error, "summary_skipped": v.summary_skipped})
    return {"version": version, "query_id": query_row["query_id"], "query": out.query, "category": query_row["category"],
            "borough_choice": out.borough_choice, "boroughs": out.boroughs, "method": out.method,
            "parsed": None if out.parsed is None else vars(out.parsed), "allergy_sensitive": getattr(out, "allergy_sensitive", None),
            "notices": out.notices, "timings": out.timings, "llm_calls": out.llm_calls, "results": results}


def run(repo, version, method, out_path, gap=45.0, queries_path=QUERIES_PATH, ids=EVAL_QUERIES, sleep=time.sleep):
    sys.path.insert(0, str(Path(repo).resolve()))
    from app import pipeline  # the checkout's pipeline
    retriever_name, _ = pipeline.METHODS[method]
    retriever = pipeline.load_retriever(retriever_name)
    client, model, setup_error = pipeline.make_llm_client()
    if setup_error:
        sys.exit(f"LLM not configured: {setup_error['code']}")
    records = []
    for i, q in enumerate(read_queries(queries_path, ids)):
        if i:
            sleep(gap)
        borough = q["borough"] or "Both"
        out, _ = pipeline.run_search(q["query"], borough, method, retriever=retriever, client=client, model=model,
                                     log_path=None)
        records.append(outcome_record(pipeline, out, q, version))
        tokens = sum((c.get("usage") or {}).get("total_tokens") or 0 for c in out.llm_calls)
        print(f"  {version} {q['query_id']}: {len(out.results)} results, {tokens} tokens, {out.timings.get('total_s')}s", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"version": version, "method": method, "model": model,
                                          "repo": str(repo), "records": records}, indent=1, ensure_ascii=False))
    return records


# ---- sheets -------------------------------------------------------------------
def fmt_excerpts(excerpts):
    return " || ".join(f"[{e['date']}] {e['text']}" for e in excerpts)


def fmt_summary(result):
    s = result["summary"]
    if s is None:
        why = (result.get("summary_error") or {}).get("message") or result.get("summary_skipped") or "no summary"
        return f"No summary ({why}). Excerpts: " + fmt_excerpts(result["excerpts"][:3])
    lines = [f"Status: {s['status']}"] + ([f"Note: {s['note']}"] if s.get("note") else [])
    lines += [f"- [{c['label']}] {c['text']} (excerpt: {fmt_excerpts(c['excerpts'])})" for c in s["claims"]]
    return "\n".join(lines)


def claim_rows(outputs, seed=SEED):
    rows, key = [], []
    for out in outputs:
        for rec in out["records"]:
            for res in rec["results"][:TOP_N]:
                s = res["summary"]
                for c in (s["claims"] if s else []):
                    rows.append({"query": rec["query"], "restaurant": f"{res['name']} ({res['borough']})",
                                 "status_shown": s["status"], "claim_label": c["label"], "claim": c["text"],
                                 "cited_excerpts": fmt_excerpts(c["excerpts"]), "supported_by_excerpt": "",
                                 "unjustified_safety_claim": "", "note": "",
                                 "_key": {"version": out["version"], "query_id": rec["query_id"], "gmap_id": res["gmap_id"]}})
    random.Random(seed).shuffle(rows)
    for k, r in enumerate(rows, 1):
        r["claim_id"] = f"C{k:03d}"
        key.append({"claim_id": r["claim_id"], **r.pop("_key")})
    return rows, key


def usability_rows(fixed):
    rows = []
    for rec in fixed["records"]:
        row = {"query_id": rec["query_id"], "query": rec["query"],
               "borough": ", ".join(rec["boroughs"]) if rec["boroughs"] else "Manhattan and Brooklyn",
               "usefulness": "", "note": ""}
        for i in range(1, TOP_N + 1):
            res = rec["results"][i - 1] if len(rec["results"]) >= i else None
            row[f"result_{i}"] = "" if res is None else (
                f"{res['name']} ({res['borough']}, rating {res['avg_rating']})" + (" [low evidence]" if res["low_evidence"] else ""))
            row[f"summary_{i}"] = "" if res is None else fmt_summary(res)
        rows.append(row)
    return rows


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_sheets(old_path, fixed_path, raters, out_dir=OUT_DIR, seed=SEED, force=False):
    old, fixed = (json.loads(Path(p).read_text()) for p in (old_path, fixed_path))
    if old["method"] != fixed["method"]:
        raise ValueError("old and fixed outputs used different retrieval methods")
    out_dir = Path(out_dir)
    targets = [out_dir / "claim_review.csv"] + [out_dir / f"usability_{r}.csv" for r in raters]
    if any(p.exists() for p in targets) and not force:
        sys.exit("Refusing to overwrite rater sheets (they may contain ratings). Use --force.")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, key = claim_rows([old, fixed], seed)
    write_csv(out_dir / "claim_review.csv", CLAIM_COLUMNS, rows)
    write_csv(out_dir / "claim_key.csv", ["claim_id", "version", "query_id", "gmap_id"], key)
    base = usability_rows(fixed)
    for r in raters:
        mine = list(base)
        random.Random(f"{seed}:{r}").shuffle(mine)
        write_csv(out_dir / f"usability_{r}.csv", USABILITY_COLUMNS, mine)
    return len(rows), {v: sum(k["version"] == v for k in key) for v in (old["version"], fixed["version"])}


# ---- score --------------------------------------------------------------------
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_claims(review_rows, key_rows):
    key = {k["claim_id"]: k for k in key_rows}
    out, bad = {}, []
    for r in review_rows:
        sup, safe = r["supported_by_excerpt"].strip().lower(), r["unjustified_safety_claim"].strip().lower()
        if not sup and not safe:
            continue
        if sup not in SUPPORTED_VALUES or safe not in SAFETY_VALUES:
            bad.append(r["claim_id"])
            continue
        v = out.setdefault(key[r["claim_id"]]["version"], {"claims": 0, "no": 0, "partly": 0, "safety": 0})
        v["claims"] += 1
        v["no"] += sup == "no"
        v["partly"] += sup == "partly"
        v["safety"] += safe == "yes"
    if bad:
        raise ValueError(f"supported_by_excerpt must be yes/partly/no and unjustified_safety_claim yes/no: {bad[:10]}")
    for v in out.values():
        v["unsupported_rate_strict"] = v["no"] / v["claims"] if v["claims"] else None
        v["unsupported_rate_lenient"] = (v["no"] + v["partly"]) / v["claims"] if v["claims"] else None
    return out


def score_usability(sheets):
    """sheets: {rater: rows}. Mean per rater and overall; Krippendorff's alpha (ordinal) over queries."""
    from eval.stats import krippendorff_alpha
    ratings, bad = {}, []
    for rater, rows in sheets.items():
        for r in rows:
            v = r["usefulness"].strip()
            if not v:
                continue
            if v not in {"1", "2", "3", "4", "5"}:
                bad.append(f"{rater}:{r['query_id']}={v!r}")
                continue
            ratings.setdefault(r["query_id"], {})[rater] = int(v)
    if bad:
        raise ValueError(f"usefulness must be 1-5: {bad[:10]}")
    raters = sorted(sheets)
    per_rater = {}
    for rater in raters:
        vals = [q[rater] for q in ratings.values() if rater in q]
        per_rater[rater] = sum(vals) / len(vals) if vals else None
    all_vals = [v for q in ratings.values() for v in q.values()]
    units = [[q.get(r) for r in raters] for q in ratings.values()]
    return {"n_queries": len(ratings), "mean": sum(all_vals) / len(all_vals) if all_vals else None,
            "per_rater": per_rater, "alpha_ordinal": krippendorff_alpha(units, "ordinal") if units else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", type=Path, required=True)
    r.add_argument("--version", required=True)
    r.add_argument("--method", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--gap", type=float, default=45.0, help="seconds between searches (Groq per-minute tokens)")
    s = sub.add_parser("sheets")
    s.add_argument("--old", type=Path, default=OUT_DIR / "outputs_old.json")
    s.add_argument("--fixed", type=Path, default=OUT_DIR / "outputs_fixed.json")
    s.add_argument("--raters", nargs="+", required=True)
    s.add_argument("--out-dir", type=Path, default=OUT_DIR)
    s.add_argument("--force", action="store_true")
    c = sub.add_parser("score")
    c.add_argument("--claims", type=Path, default=OUT_DIR / "claim_review.csv")
    c.add_argument("--key", type=Path, default=OUT_DIR / "claim_key.csv")
    c.add_argument("--usability", nargs="*", type=Path, default=[])
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args.repo, args.version, args.method, args.out, args.gap)
    elif args.cmd == "sheets":
        n, by_version = make_sheets(args.old, args.fixed, args.raters, args.out_dir, force=args.force)
        print(f"{n} claims for review ({by_version}); usability sheets for {', '.join(args.raters)} in {args.out_dir}")
    else:
        sys.path.insert(0, str(REPO))
        claims = score_claims(read_csv(args.claims), read_csv(args.key)) if args.claims.exists() else {}
        for v, m in claims.items():
            print(f"{v}: {m['claims']} claims; unsupported {m['unsupported_rate_strict']:.3f} (strict), "
                  f"{m['unsupported_rate_lenient']:.3f} (incl. partly); unjustified safety claims {m['safety']}")
        if args.usability:
            u = score_usability({p.stem.replace("usability_", ""): read_csv(p) for p in args.usability})
            print(f"Usability: mean {u['mean']}, per rater {u['per_rater']}, alpha (ordinal) {u['alpha_ordinal']}, "
                  f"{u['n_queries']} queries")


if __name__ == "__main__":
    main()
