"""Cross-provider consistency check for the LLM judge (same model and prompt, different provider).

select   Pick a stratified subset of human-sample pairs that were labeled by a model call on the reference
         provider (default OpenRouter), write a key file usable as llm_judge --sample-key:
             python -m eval.provider_check select --out eval/labels/groq_check_key.csv
         then label those pairs on the other provider into a separate file, e.g.
             python -m eval.llm_judge --sample --sample-key eval/labels/groq_check_key.csv --provider groq \\
                 --model openai/gpt-oss-120b --reasoning-effort low --batch-size 1 --out eval/llm_labels_groq_check.jsonl

compare  Aggregate agreement only (no per-pair labels): exact agreement, Cohen's kappa (linear and
         quadratic weights), and how many pairs differ by 1 or 2 points, overall and for allergy queries:
             python -m eval.provider_check compare --check eval/llm_labels_groq_check.jsonl
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from eval.stats import cohen_kappa, raw_agreement

EVAL = Path(__file__).parent
SAMPLE_KEY = EVAL / "labels" / "sample_key.csv"
REFERENCE = EVAL / "llm_labels.jsonl"
DEFAULT_ALLOCATION = {"allergy": 12, "single_diet": 12, "mixed_group": 10, "diet_plus_filter": 6}
SEED = 6165


def read_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(r["query_id"], r["gmap_id"])] = r
    return out


def select(sample_key_rows, reference, provider="openrouter", allocation=DEFAULT_ALLOCATION, seed=SEED):
    """Stratified pairs labeled by a model call on `provider`; one query at a time in rotation per category."""
    rng = random.Random(seed)
    by_cat = defaultdict(lambda: defaultdict(list))
    for row in sample_key_rows:
        rec = reference.get((row["query_id"], row["gmap_id"]))
        if rec and rec.get("source") == "llm" and rec.get("provider") == provider:
            by_cat[row["category"]][row["query_id"]].append(row)
    chosen = []
    for cat, n in allocation.items():
        queries = sorted(by_cat[cat])
        rng.shuffle(queries)
        for q in queries:
            rng.shuffle(by_cat[cat][q])
        picked, i = 0, 0
        while picked < n and any(by_cat[cat][q] for q in queries):
            q = queries[i % len(queries)]
            if by_cat[cat][q]:
                chosen.append(by_cat[cat][q].pop())
                picked += 1
            i += 1
    return chosen


def compare(reference, check):
    """Aggregate agreement between reference labels and check labels on the pairs both have."""
    keys = sorted(set(reference) & set(check))
    if not keys:
        raise ValueError("no pairs in common")
    a = [reference[k]["label"] for k in keys]
    b = [check[k]["label"] for k in keys]
    diffs = Counter(abs(x - y) for x, y in zip(a, b))
    return {"n": len(keys), "exact": raw_agreement(a, b), "kappa": cohen_kappa(a, b),
            "kappa_linear": cohen_kappa(a, b, "linear"), "kappa_quadratic": cohen_kappa(a, b, "quadratic"),
            "differ_by_1": diffs.get(1, 0), "differ_by_2": diffs.get(2, 0),
            "check_higher": sum(y > x for x, y in zip(a, b)), "check_lower": sum(y < x for x, y in zip(a, b)),
            "providers": {"reference": dict(Counter(reference[k].get("provider") for k in keys)),
                          "check": dict(Counter(check[k].get("provider") for k in keys))},
            "prompt_hashes_match": {reference[k].get("prompt_sha256") for k in keys} == {check[k].get("prompt_sha256") for k in keys}}


def fmt(r):
    return (f"n={r['n']}: exact agreement {r['exact']:.3f}; kappa {r['kappa']:.3f}, linear {r['kappa_linear']:.3f}, "
            f"quadratic {r['kappa_quadratic']:.3f}; differ by 1: {r['differ_by_1']}, by 2: {r['differ_by_2']}; "
            f"check higher {r['check_higher']}, lower {r['check_lower']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--sample-key", type=Path, default=SAMPLE_KEY)
    s.add_argument("--reference", type=Path, default=REFERENCE)
    s.add_argument("--provider", default="openrouter")
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--reference", type=Path, default=REFERENCE)
    c.add_argument("--check", type=Path, required=True)
    c.add_argument("--sample-key", type=Path, default=SAMPLE_KEY)
    args = ap.parse_args(argv)
    if args.cmd == "select":
        with open(args.sample_key, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        chosen = select(rows, read_jsonl(args.reference), args.provider)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "query_id", "gmap_id", "category"], extrasaction="ignore")
            w.writeheader()
            w.writerows(chosen)
        print(f"{len(chosen)} pairs: {dict(Counter(r['category'] for r in chosen))}; "
              f"{len({r['query_id'] for r in chosen})} queries; wrote {args.out}")
    else:
        reference, check = read_jsonl(args.reference), read_jsonl(args.check)
        check = {k: v for k, v in check.items() if v.get("source") == "llm"}
        r = compare(reference, check)
        print("All:", fmt(r))
        print("Providers:", r["providers"], "| same prompt hash:", r["prompt_hashes_match"])
        if args.sample_key.exists():
            with open(args.sample_key, newline="", encoding="utf-8") as f:
                allergy = {(row["query_id"], row["gmap_id"]) for row in csv.DictReader(f) if row["category"] == "allergy"}
            sub_ref = {k: v for k, v in reference.items() if k in allergy}
            if set(sub_ref) & set(check):
                print("Allergy:", fmt(compare(sub_ref, check)))


if __name__ == "__main__":
    main()
