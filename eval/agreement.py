"""Inter-annotator agreement on the human sample, plus the files for adjudication.

Reads the judges' filled labels_<judge>.csv files (from make_sample.py), joins them on sample_id
and reports, for the 0/1/2 labels:
  - label distribution per judge, and missing labels,
  - pairwise raw agreement and Cohen's kappa (unweighted, linear and quadratic weights),
  - Krippendorff's alpha across all judges (ordinal; nominal for reference),
  - the share of pairs where all judges agree.
Writes to --out-dir (default: the judges' directory):
  disagreements.csv          pairs where the judges differ, largest spread first, with notes
  adjudicated_template.csv   one row per pair: each judge's label, the majority, and an
                             adjudicated_label column pre-filled only where all judges agree.
                             Fill the blanks, then save it as adjudicated.csv.

Run from the repo root:
    python -m eval.agreement eval/labels/labels_alice.csv eval/labels/labels_bob.csv eval/labels/labels_carol.csv
"""

import argparse
import csv
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from eval.stats import LABELS, cohen_kappa, krippendorff_alpha, raw_agreement

TEMPLATE_FIXED = ["sample_id", "query_id", "gmap_id", "category", "query", "name"]


def read_judge(path):
    """{sample_id: row} for one judge file; label parsed to int or None (blank)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out, bad = {}, []
    for row in rows:
        raw = (row.get("label") or "").strip()
        if raw == "":
            label = None
        elif raw in {"0", "1", "2"}:
            label = int(raw)
        else:
            bad.append(f"{row['sample_id']}={raw!r}")
            continue
        out[row["sample_id"]] = {**row, "label": label, "note": (row.get("note") or "").strip()}
    if bad:
        raise ValueError(f"{path}: labels must be 0, 1, 2 or blank; got {', '.join(bad[:10])}")
    return out


def judge_name(path):
    stem = Path(path).stem
    return stem[len("labels_"):] if stem.startswith("labels_") else stem


def analyze(judges):
    """judges: {name: {sample_id: row}}. Returns a report dict (also used by tests)."""
    names = list(judges)
    ids = sorted(set().union(*(set(j) for j in judges.values())))
    report = {"judges": names, "n_pairs": len(ids), "per_judge": {}, "pairwise": {}}
    for name in names:
        labels = [judges[name][i]["label"] for i in ids if i in judges[name]]
        report["per_judge"][name] = {
            "labeled": sum(l is not None for l in labels),
            "missing": len(ids) - sum(l is not None for l in labels),
            "distribution": {l: labels.count(l) for l in LABELS}}
    for a, b in combinations(names, 2):
        common = [i for i in ids if judges[a].get(i, {}).get("label") is not None
                  and judges[b].get(i, {}).get("label") is not None]
        la, lb = [judges[a][i]["label"] for i in common], [judges[b][i]["label"] for i in common]
        report["pairwise"][(a, b)] = {"n": len(common), "raw": raw_agreement(la, lb),
                                      "kappa": cohen_kappa(la, lb), "kappa_linear": cohen_kappa(la, lb, "linear"),
                                      "kappa_quadratic": cohen_kappa(la, lb, "quadratic")}
    units = [[judges[n].get(i, {}).get("label") for n in names] for i in ids]
    complete = [u for u in units if all(v is not None for v in u)]
    report["alpha_ordinal"] = krippendorff_alpha(units, "ordinal")
    report["alpha_nominal"] = krippendorff_alpha(units, "nominal")
    report["n_complete"] = len(complete)
    report["unanimous"] = sum(len(set(u)) == 1 for u in complete) / len(complete) if complete else float("nan")

    rows = []
    for i, u in zip(ids, units):
        first = next(judges[n][i] for n in names if i in judges[n])
        present = [v for v in u if v is not None]
        counts = Counter(present)
        top = counts.most_common(1)[0] if counts else (None, 0)
        rows.append({"sample_id": i, "query_id": first.get("query_id", ""), "query": first.get("query", ""),
                     "name": first.get("name", ""), "labels": dict(zip(names, u)),
                     "notes": {n: judges[n].get(i, {}).get("note", "") for n in names},
                     "spread": (max(present) - min(present)) if present else 0,
                     "unanimous": len(present) == len(names) and len(counts) == 1,
                     "majority": top[0] if top[1] > len(names) / 2 else None})
    report["items"] = rows
    report["disagreements"] = sorted((r for r in rows if len({v for v in r["labels"].values() if v is not None}) > 1),
                                     key=lambda r: (-r["spread"], r["sample_id"]))
    return report


def fmt(x):
    return "n/a" if x != x else f"{x:.3f}"  # NaN check


def print_report(report, limit=15):
    names = report["judges"]
    print(f"{report['n_pairs']} pairs, judges: {', '.join(names)}; labeled by all: {report['n_complete']}")
    for n, s in report["per_judge"].items():
        print(f"  {n}: {s['labeled']} labeled, {s['missing']} missing, distribution {s['distribution']}")
    print("\nPairwise              n   raw    kappa  kappa_lin  kappa_quad")
    for (a, b), s in report["pairwise"].items():
        print(f"  {a + ' vs ' + b:18} {s['n']:3}  {fmt(s['raw'])}  {fmt(s['kappa'])}  {fmt(s['kappa_linear'])}"
              f"      {fmt(s['kappa_quadratic'])}")
    print(f"\nKrippendorff's alpha (all judges): ordinal {fmt(report['alpha_ordinal'])}, "
          f"nominal {fmt(report['alpha_nominal'])}")
    print(f"All judges agree on {fmt(report['unanimous'])} of the {report['n_complete']} fully labeled pairs")
    d = report["disagreements"]
    print(f"\n{len(d)} disagreements (largest spread first):")
    for r in d[:limit]:
        labels = ", ".join(f"{n}={r['labels'][n]}" for n in names)
        print(f"  {r['sample_id']} {r['query_id']} {r['name'][:40]!r}: {labels}")
    if len(d) > limit:
        print(f"  ... {len(d) - limit} more in disagreements.csv")


def write_outputs(report, out_dir, key=None):
    """disagreements.csv and adjudicated_template.csv; key: {sample_id: sample_key row}."""
    names, key = report["judges"], key or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "disagreements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "query_id", "query", "name", "spread"] + [f"label_{n}" for n in names]
                   + [f"note_{n}" for n in names])
        for r in report["disagreements"]:
            w.writerow([r["sample_id"], r["query_id"], r["query"], r["name"], r["spread"]]
                       + ["" if r["labels"][n] is None else r["labels"][n] for n in names]
                       + [r["notes"][n] for n in names])
    with open(out_dir / "adjudicated_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TEMPLATE_FIXED + [f"label_{n}" for n in names] + ["majority", "adjudicated_label", "adjudication_note"])
        for r in report["items"]:
            k = key.get(r["sample_id"], {})
            w.writerow([r["sample_id"], r["query_id"], k.get("gmap_id", ""), k.get("category", ""), r["query"], r["name"]]
                       + ["" if r["labels"][n] is None else r["labels"][n] for n in names]
                       + ["" if r["majority"] is None else r["majority"],
                          r["labels"][names[0]] if r["unanimous"] else "", ""])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Agreement between judges on the human sample")
    ap.add_argument("files", nargs="+", type=Path, help="filled labels_<judge>.csv files (two or more)")
    ap.add_argument("--key", type=Path, default=None, help="sample_key.csv (default: next to the first file)")
    ap.add_argument("--out-dir", type=Path, default=None, help="output directory (default: next to the first file)")
    args = ap.parse_args(argv)
    if len(args.files) < 2:
        sys.exit("Need at least two judge files.")
    judges = {judge_name(p): read_judge(p) for p in args.files}
    if len(judges) != len(args.files):
        sys.exit("Judge names (from file names) must be distinct.")
    out_dir = args.out_dir or args.files[0].parent
    key_path = args.key or args.files[0].parent / "sample_key.csv"
    key = {}
    if key_path.exists():
        with open(key_path, newline="", encoding="utf-8") as f:
            key = {r["sample_id"]: r for r in csv.DictReader(f)}
    else:
        print(f"Note: {key_path} not found; the template will lack gmap_id and category.")
    report = analyze(judges)
    print_report(report)
    write_outputs(report, out_dir, key)
    print(f"\nWrote {out_dir / 'disagreements.csv'} and {out_dir / 'adjudicated_template.csv'}")


if __name__ == "__main__":
    main()
