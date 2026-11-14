"""Draw a stratified sample of pool pairs for human labeling, one blind CSV per judge.

Every judge gets the same pairs in a different shuffled order, with empty `label` and `note`
columns (see RUBRIC.md). The selection (fixed seed) makes sure that:
  1. every query has at least one pair (when the sample is large enough),
  2. at least --min-none rows with evidence_status=none, --min-allergy pairs from allergy
     queries and --min-single pairs found by only one system are included (as far as the pool has them),
  3. the remaining slots follow each query category's share of the pool.
Picks rotate across queries, so no quota is filled from a single query.

Outputs (in --out-dir, default eval/labels/):
  labels_<judge>.csv   one per judge: sample_id, query_id, query, name, borough, evidence_status,
                       evidence_1..6, label, note. Nothing in it identifies a system.
  sample_key.csv       sample_id, query_id, gmap_id, category, single_system, evidence_status.
                       For agreement.py, llm_judge.py --sample and validate_judge.py; not for judges.
Existing judge files are never overwritten unless --force is given (they may hold labels).

Run from the repo root:
    python -m eval.make_sample --judges alice bob carol [--n 120] [--pool eval/pool.csv]
"""

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from eval.make_pool import EVIDENCE_COLUMNS, EVIDENCE_NONE, POOL_PATH, QUERIES_PATH, RUNS_PATH, read_queries

LABELS_DIR = Path(__file__).parent / "labels"
DEFAULT_N = 120
SEED = 6165
ALLERGY = "allergy"
JUDGE_COLUMNS = (["sample_id", "query_id", "query", "name", "borough", "evidence_status"]
                 + EVIDENCE_COLUMNS + ["label", "note"])
KEY_COLUMNS = ["sample_id", "query_id", "gmap_id", "category", "single_system", "evidence_status"]
JUDGE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def annotate(pool, runs, queries):
    """Pool rows with category (from the query file) and single_system (from runs.csv)."""
    category = {q["query_id"]: q.get("category", "") for q in queries}
    finders = defaultdict(set)
    for r in runs:
        finders[(r["query_id"], r["gmap_id"])].add(r["system"])
    out = []
    for row in pool:
        if row["query_id"] not in category:
            raise ValueError(f"pool query_id {row['query_id']} is not in the query file")
        out.append({**row, "category": category[row["query_id"]],
                    "single_system": len(finders[(row["query_id"], row["gmap_id"])]) == 1})
    return out


def select(pairs, n, seed=SEED, min_none=10, min_allergy=30, min_single=25):
    """Indices of the chosen pairs (see module docstring for the rules), deterministic for a seed."""
    rng = random.Random(seed)
    by_query = defaultdict(list)
    for i, p in enumerate(pairs):
        by_query[p["query_id"]].append(i)
    order = sorted(by_query)
    rng.shuffle(order)
    for q in order:
        rng.shuffle(by_query[q])
    chosen, start = [], [0]
    taken = set()

    def take(pred, k):
        """Pick up to k unchosen pairs matching pred, one query at a time in rotation."""
        picked = 0
        while picked < k and len(chosen) < n:
            progress = False
            for step in range(len(order)):
                q = order[(start[0] + step) % len(order)]
                i = next((i for i in by_query[q] if i not in taken and pred(pairs[i])), None)
                if i is None:
                    continue
                chosen.append(i)
                taken.add(i)
                picked += 1
                progress = True
                if picked == k or len(chosen) == n:
                    start[0] = (start[0] + step + 1) % len(order)
                    return
            if not progress:
                return

    n = min(n, len(pairs))
    take(lambda p: True, min(len(order), n))  # 1. one pair per query
    for pred, minimum in ((lambda p: p["evidence_status"] == EVIDENCE_NONE, min_none),
                          (lambda p: p["category"] == ALLERGY, min_allergy),
                          (lambda p: p["single_system"], min_single)):  # 2. quotas
        take(pred, max(0, minimum - sum(pred(pairs[i]) for i in chosen)))
    pool_share = Counter(p["category"] for p in pairs)  # 3. category shares
    while len(chosen) < n:
        have = Counter(pairs[i]["category"] for i in chosen)
        open_cats = [c for c in pool_share if any(i not in taken for i in range(len(pairs)) if pairs[i]["category"] == c)]
        if not open_cats:
            break
        cat = max(open_cats, key=lambda c: (n * pool_share[c] / len(pairs) - have[c], c))
        take(lambda p, cat=cat: p["category"] == cat, 1)
    return chosen


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stratified human-labeling sample, one blind CSV per judge")
    ap.add_argument("--judges", nargs="+", required=True, help="judge names (letters, digits, _ or -)")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--pool", type=Path, default=POOL_PATH)
    ap.add_argument("--runs", type=Path, default=RUNS_PATH)
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH)
    ap.add_argument("--out-dir", type=Path, default=LABELS_DIR)
    ap.add_argument("--min-none", type=int, default=10)
    ap.add_argument("--min-allergy", type=int, default=30)
    ap.add_argument("--min-single", type=int, default=25)
    ap.add_argument("--force", action="store_true", help="overwrite existing judge files")
    args = ap.parse_args(argv)

    bad = [j for j in args.judges if not JUDGE_NAME.match(j)]
    if bad or len(set(args.judges)) != len(args.judges):
        sys.exit(f"Judge names must be unique and use letters, digits, _ or -: {args.judges}")
    judge_paths = {j: args.out_dir / f"labels_{j}.csv" for j in args.judges}
    existing = [str(p) for p in judge_paths.values() if p.exists()]
    if existing and not args.force:
        sys.exit("Refusing to overwrite judge files (they may contain labels): " + ", ".join(existing)
                 + ". Use --force to replace them.")

    pairs = annotate(read_csv(args.pool), read_csv(args.runs), read_queries(args.queries))
    idx = select(pairs, args.n, args.seed, args.min_none, args.min_allergy, args.min_single)
    sample = sorted((pairs[i] for i in idx), key=lambda p: (p["query_id"], p["gmap_id"]))
    for k, p in enumerate(sample, 1):
        p["sample_id"] = f"S{k:03d}"
        p["label"] = p["note"] = ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "sample_key.csv", KEY_COLUMNS,
              [{**p, "single_system": int(p["single_system"])} for p in sample])
    for judge, path in judge_paths.items():
        rows = list(sample)
        random.Random(f"{args.seed}:{judge}").shuffle(rows)
        write_csv(path, JUDGE_COLUMNS, rows)

    cats = Counter(p["category"] for p in sample)
    print(f"Sample: {len(sample)} pairs from {len({p['query_id'] for p in sample})} of "
          f"{len({p['query_id'] for p in pairs})} queries (pool {len(pairs)} pairs); seed {args.seed}")
    print(f"  by category: {dict(sorted(cats.items()))}")
    print(f"  allergy: {cats[ALLERGY]}; single-system: {sum(p['single_system'] for p in sample)}; "
          f"evidence_status=none: {sum(p['evidence_status'] == EVIDENCE_NONE for p in sample)}")
    print(f"Wrote {args.out_dir / 'sample_key.csv'} and " + ", ".join(str(p) for p in judge_paths.values()))


if __name__ == "__main__":
    main()
