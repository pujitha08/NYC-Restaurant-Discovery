"""Validate the LLM judge against the adjudicated human labels.

Joins eval/llm_labels.jsonl (llm_judge.py) with the adjudicated sample (agreement.py's
template, filled in and saved as eval/labels/adjudicated.csv) on (query_id, gmap_id), and reports
for all pairs, allergy queries and the other queries:
  - coverage (sample pairs without an LLM label),
  - a 3x3 confusion matrix (rows: human, columns: LLM),
  - accuracy, Cohen's kappa (unweighted, linear, quadratic),
  - how often the LLM labels above or below the human label (above is the risky direction for
    allergy queries: it would call a restaurant more suitable than the reviews support),
  - accuracy of rule-labeled pairs (no evidence shown) separately from LLM calls.
Writes the disagreements (allergy queries and the largest gaps first) to --out.

Run from the repo root:
    python -m eval.validate_judge [--llm eval/llm_labels.jsonl] [--adjudicated eval/labels/adjudicated.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from eval.stats import LABELS, cohen_kappa, confusion_matrix, raw_agreement

EVAL_DIR = Path(__file__).parent
LLM_PATH = EVAL_DIR / "llm_labels.jsonl"
ADJUDICATED_PATH = EVAL_DIR / "labels" / "adjudicated.csv"
OUT_PATH = EVAL_DIR / "labels" / "judge_disagreements.csv"
ALLERGY = "allergy"


def read_llm(path):
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            labels[(r["query_id"], r["gmap_id"])] = r  # a later line wins
    return labels


def read_adjudicated(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out, blank, bad = [], [], []
    for r in rows:
        raw = (r.get("adjudicated_label") or "").strip()
        if raw == "":
            blank.append(r["sample_id"])
        elif raw not in {"0", "1", "2"}:
            bad.append(f"{r['sample_id']}={raw!r}")
        else:
            out.append({**r, "human": int(raw)})
    if bad:
        raise ValueError(f"{path}: adjudicated_label must be 0, 1 or 2; got {', '.join(bad[:10])}")
    return out, blank


def summarize(pairs):
    h, m = [p["human"] for p in pairs], [p["llm"] for p in pairs]
    return {"n": len(pairs), "accuracy": raw_agreement(h, m), "kappa": cohen_kappa(h, m),
            "kappa_linear": cohen_kappa(h, m, "linear"), "kappa_quadratic": cohen_kappa(h, m, "quadratic"),
            "confusion": confusion_matrix(h, m), "llm_higher": sum(b > a for a, b in zip(h, m)),
            "llm_lower": sum(b < a for a, b in zip(h, m))}


def compare(adjudicated, llm):
    pairs, missing = [], []
    for r in adjudicated:
        rec = llm.get((r["query_id"], r["gmap_id"]))
        if rec is None:
            missing.append(r["sample_id"])
            continue
        pairs.append({**r, "llm": rec["label"], "reason": rec.get("reason", ""), "review_ids": rec.get("review_ids", []),
                      "source": rec.get("source", "llm")})
    groups = {"all": pairs, "allergy": [p for p in pairs if p.get("category") == ALLERGY],
              "other": [p for p in pairs if p.get("category") != ALLERGY]}
    report = {"groups": {g: summarize(ps) for g, ps in groups.items()}, "missing": missing,
              "by_source": {s: summarize([p for p in pairs if p["source"] == s]) for s in sorted({p["source"] for p in pairs})}}
    report["disagreements"] = sorted((p for p in pairs if p["human"] != p["llm"]),
                                     key=lambda p: (p.get("category") != ALLERGY, -abs(p["human"] - p["llm"]), p["sample_id"]))
    return report


def fmt(x):
    return "n/a" if x != x else f"{x:.3f}"


def print_report(report, blank):
    if blank:
        print(f"Note: {len(blank)} sample pairs have no adjudicated_label yet and are skipped.")
    if report["missing"]:
        print(f"Note: {len(report['missing'])} adjudicated pairs have no LLM label: {', '.join(report['missing'][:10])}")
    for g, s in report["groups"].items():
        print(f"\n== {g}: n={s['n']}  accuracy {fmt(s['accuracy'])}  kappa {fmt(s['kappa'])}  "
              f"linear {fmt(s['kappa_linear'])}  quadratic {fmt(s['kappa_quadratic'])}")
        print(f"   LLM above human: {s['llm_higher']}, below: {s['llm_lower']}")
        print("   human \\ LLM    " + "  ".join(f"{l:>4}" for l in LABELS))
        for l, row in zip(LABELS, s["confusion"]):
            print(f"   {l:>11}    " + "  ".join(f"{v:>4}" for v in row))
    for src, s in report["by_source"].items():
        print(f"\nsource={src}: n={s['n']}, accuracy {fmt(s['accuracy'])}")
    print(f"\n{len(report['disagreements'])} disagreements (allergy first):")
    for p in report["disagreements"][:15]:
        print(f"  {p['sample_id']} {p['query_id']} [{p.get('category', '')}] {p.get('name', '')[:35]!r}: "
              f"human {p['human']}, LLM {p['llm']} - {p['reason'][:110]}")


def write_disagreements(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "query_id", "category", "query", "name", "human", "llm", "source",
                    "llm_reason", "llm_review_ids", "adjudication_note"])
        for p in report["disagreements"]:
            w.writerow([p["sample_id"], p["query_id"], p.get("category", ""), p.get("query", ""), p.get("name", ""),
                        p["human"], p["llm"], p["source"], p["reason"], ";".join(map(str, p["review_ids"])),
                        p.get("adjudication_note", "")])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare LLM judge labels with adjudicated human labels")
    ap.add_argument("--llm", type=Path, default=LLM_PATH)
    ap.add_argument("--adjudicated", type=Path, default=ADJUDICATED_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)
    for p in (args.llm, args.adjudicated):
        if not p.exists():
            sys.exit(f"Missing {p}")
    adjudicated, blank = read_adjudicated(args.adjudicated)
    report = compare(adjudicated, read_llm(args.llm))
    print_report(report, blank)
    write_disagreements(report, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
