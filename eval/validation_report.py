"""Validate the LLM judge against one human rater's blind labels (eval/labels/labels_<name>.csv).

The human CSV (eval/make_sample.py: sample_id, query_id, query, name, borough, evidence_status, evidence_1..6, label,
note) has no gmap_id, so each row is matched to its restaurant, then to the LLM label for the same
(query_id, gmap_id) in eval/llm_labels.jsonl:
  1. eval/labels/sample_key.csv (sample_id -> gmap_id), if it exists;
  2. the review ids cited in the evidence ("[review 1303951, 2020-05] ..."), looked up in the review store; the
     restaurant must have the row's name;
  3. otherwise the row's (name, borough) in data/processed/restaurants_final.jsonl, if exactly one such restaurant
     has an LLM label for that query.
Rows that cannot be matched are listed, never guessed. Rows with a blank label are skipped and counted.

Reports (reports/judge_validation.md): n matched, raw agreement, Cohen's kappa (unweighted and quadratic), Krippendorff's
alpha (ordinal), the 3x3 confusion matrix (human rows, LLM columns), mean label by label source, agreement by query
category, and the largest disagreements (human 0 vs LLM 2 and the reverse) with the human's notes. Bootstrap 95%
CIs resample items (seed 6165). No API calls are made.

Run from the repo root:
    python -m eval.validation_report [--human eval/labels/labels_pujitha.csv] [--llm eval/llm_labels.jsonl]
"""

import argparse
import csv
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.config import DATA_DIR
from eval.metrics import CATEGORY_ORDER, N_RESAMPLES, SEED, read_categories, rel
from eval.stats import LABELS, cohen_kappa, confusion_matrix, krippendorff_alpha, raw_agreement
from eval.validate_judge import read_llm, summarize

EVAL_DIR = Path(__file__).parent
REPO = EVAL_DIR.parent
HUMAN_PATH = EVAL_DIR / "labels" / "labels_pujitha.csv"
HUMAN2_PATH = EVAL_DIR / "labels" / "labels_kang.csv"
LLM_PATH = EVAL_DIR / "llm_labels.jsonl"
QUERIES_PATH = EVAL_DIR / "queries_final.csv"
SAMPLE_KEY_PATH = EVAL_DIR / "labels" / "sample_key.csv"
META_PATH = DATA_DIR / "processed" / "restaurants_final.jsonl"
OUT_PATH = REPO / "reports" / "judge_validation.md"
BOOTSTRAP = 2000
SAMPLE_POOL = 1074  # labeled pairs when the human sample was drawn (the label file has grown since)
_REVIEW_ID = re.compile(r"^\[review (\d+)[,\]]")


# ---- inputs -------------------------------------------------------------------
def decode_utf8(data):
    """UTF-8 (BOM allowed). A file that is UTF-8 except for stray Mac Roman apostrophes (a lone 0xD5, as left by a
    Mac spreadsheet export) is read with those bytes as U+2019; any other invalid byte still raises."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return re.sub(rb"\xd5(?![\x80-\xbf])", "\u2019".encode(), data).decode("utf-8-sig")


def read_human(path):
    """(rows with an integer `human` label, ids of rows with a blank label). UTF-8 (BOM allowed) or a clear error."""
    path = Path(path)
    try:
        text = decode_utf8(path.read_bytes())
    except UnicodeDecodeError as e:
        raise ValueError(f"{path}: not valid UTF-8 (byte {e.start}). Re-save the file as 'CSV UTF-8' from your "
                         "spreadsheet program (Excel's plain 'CSV' uses a legacy encoding that breaks names and accents).") from None
    reader = csv.DictReader(io.StringIO(text, newline=""))
    missing = [c for c in ("sample_id", "query_id", "name", "label") if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
    rows, blank, bad = [], [], []
    for r in reader:
        raw = (r.get("label") or "").strip()
        if raw == "":
            blank.append(r["sample_id"])
        elif raw not in {"0", "1", "2"}:
            bad.append(f"{r['sample_id']}={raw!r}")
        else:
            rows.append({**r, "human": int(raw), "name_fixed": repair_mojibake(r["name"])})
    if bad:
        raise ValueError(f"{path}: label must be 0, 1, 2 or blank; got {', '.join(bad[:10])}")
    return rows, blank


def repair_mojibake(text):
    """Undo UTF-8 text that was once decoded as Mac Roman and re-saved ("Kest√©" -> "Kesté"); unchanged if it does not fit."""
    if not any(ch in (text or "") for ch in "√‚Ä‰Ç"):
        return text
    try:
        return text.encode("mac_roman").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def read_meta(path=META_PATH):
    """{gmap_id: {"name", "borough"}} from restaurants_final.jsonl; {} if the file is missing."""
    meta = {}
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                meta[r["gmap_id"]] = {"name": r["name"], "borough": r["borough"]}
    return meta


def read_sample_key(path):
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["sample_id"]: r["gmap_id"] for r in csv.DictReader(f)}


def cited_review_ids(row):
    ids = []
    for i in range(1, 7):
        m = _REVIEW_ID.match((row.get(f"evidence_{i}") or "").strip())
        if m:
            ids.append(int(m.group(1)))
    return ids


def same_name(a, b):
    a, b = repair_mojibake(a), repair_mojibake(b)
    return " ".join((a or "").casefold().split()) == " ".join((b or "").casefold().split())


def resolve(rows, llm, meta, review_gmap, sample_key=None):
    """Attach gmap_id to each row. Returns (matched rows with `gmap_id`, `llm` record, `method`; unmatched [(row, why)]).

    review_gmap(ids) -> {review_id: gmap_id}. meta: {gmap_id: {"name", "borough"}}.
    """
    by_name = defaultdict(list)
    for gid, m in meta.items():
        by_name[(m["name"].casefold(), m["borough"].casefold())].append(gid)
    matched, unmatched = [], []
    for r in rows:
        gid, method = None, None
        if sample_key and r["sample_id"] in sample_key:
            gid, method = sample_key[r["sample_id"]], "sample_key"
        if gid is None:
            ids = cited_review_ids(r)
            owners = Counter(g for g in review_gmap(ids).values()) if ids else Counter()
            for cand, _ in owners.most_common():
                if not meta or same_name(meta.get(cand, {}).get("name"), r.get("name_fixed") or r["name"]):
                    gid, method = cand, "review ids"
                    break
        if gid is None:
            named = by_name.get(((r.get("name_fixed") or r["name"]).casefold(), (r.get("borough") or "").casefold()), [])
            cands = [g for g in named if (r["query_id"], g) in llm]
            if len(cands) == 1:
                gid, method = cands[0], "name + borough"
            elif cands:
                unmatched.append((r, f"{len(cands)} restaurants share this name and borough"))
                continue
            elif named:
                unmatched.append((r, "no LLM label for this restaurant and query"))
                continue
        if gid is None:
            unmatched.append((r, "restaurant not found"))
        elif (r["query_id"], gid) not in llm:
            unmatched.append((r, "no LLM label for this restaurant and query"))
        else:
            matched.append({**r, "gmap_id": gid, "llm": llm[(r["query_id"], gid)]["label"],
                            "llm_record": llm[(r["query_id"], gid)], "method": method})
    return matched, unmatched


# ---- statistics ---------------------------------------------------------------
def label_source(record):
    if record.get("source") == "rule":
        return "rule (no evidence shown)"
    return (record.get("provider") or "unknown") + (f"/{record['openrouter_upstream']}" if record.get("openrouter_upstream") else "")


def bootstrap(stat, pairs, n=BOOTSTRAP, seed=SEED):
    """Percentile 95% CI of stat(humans, llms) resampling pairs; (None, None) for fewer than 5 pairs."""
    if len(pairs) < 5:
        return None, None
    rng, vals = random.Random(seed), []
    for _ in range(n):
        s = [rng.choice(pairs) for _ in pairs]
        v = stat([p["human"] for p in s], [p["llm"] for p in s])
        if v == v:
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]) if vals else (None, None)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def judged_without_evidence(p):
    """The LLM answered "no evidence found" (a model call, not the rule) although the human's sheet showed evidence."""
    return (p["llm_record"].get("source") != "rule" and (p["llm_record"].get("reason") or "").lower().startswith("no evidence")
            and p.get("evidence_status") != "none")


def analyze(pairs, categories):
    h, m = [p["human"] for p in pairs], [p["llm"] for p in pairs]
    overall = summarize(pairs) if pairs else {}
    result = {
        "n": len(pairs), "agreement": raw_agreement(h, m), "kappa": cohen_kappa(h, m),
        "kappa_quadratic": cohen_kappa(h, m, "quadratic"),
        "alpha_ordinal": krippendorff_alpha([[a, b] for a, b in zip(h, m)], "ordinal"),
        "confusion": confusion_matrix(h, m), "llm_higher": overall.get("llm_higher", 0), "llm_lower": overall.get("llm_lower", 0),
        "ci_agreement": bootstrap(raw_agreement, pairs), "ci_kappa_quadratic": bootstrap(lambda a, b: cohen_kappa(a, b, "quadratic"), pairs),
        "mean_human": mean(h), "mean_llm": mean(m)}
    result["by_source"] = {}
    for src in sorted({label_source(p["llm_record"]) for p in pairs}):
        sp = [p for p in pairs if label_source(p["llm_record"]) == src]
        result["by_source"][src] = {"n": len(sp), "mean_human": mean([p["human"] for p in sp]), "mean_llm": mean([p["llm"] for p in sp]),
                                    "agreement": raw_agreement([p["human"] for p in sp], [p["llm"] for p in sp])}
    result["by_category"] = {}
    for cat in CATEGORY_ORDER:
        cp = [p for p in pairs if categories.get(p["query_id"]) == cat]
        ch, cm = [p["human"] for p in cp], [p["llm"] for p in cp]
        result["by_category"][cat] = {"n": len(cp), "agreement": raw_agreement(ch, cm) if cp else float("nan"),
                                      "kappa_quadratic": cohen_kappa(ch, cm, "quadratic") if len(cp) > 1 else float("nan"),
                                      "mean_human": mean(ch), "mean_llm": mean(cm),
                                      "llm_higher": sum(b > a for a, b in zip(ch, cm)), "llm_lower": sum(b < a for a, b in zip(ch, cm))}
    result["big_disagreements"] = sorted((p for p in pairs if abs(p["human"] - p["llm"]) == 2),
                                         key=lambda p: (p["llm"] < p["human"], p["query_id"], p["sample_id"]))
    result["one_step"] = sum(abs(a - b) == 1 for a, b in zip(h, m))
    flagged = [p for p in pairs if judged_without_evidence(p)]
    kept = [p for p in pairs if not judged_without_evidence(p)]
    result["no_evidence_mismatch"] = {
        "n": len(flagged), "human_mean": mean([p["human"] for p in flagged]),
        "ids": [p["sample_id"] for p in flagged],
        "agreement_without": raw_agreement([p["human"] for p in kept], [p["llm"] for p in kept]) if kept else float("nan"),
        "kappa_quadratic_without": cohen_kappa([p["human"] for p in kept], [p["llm"] for p in kept], "quadratic") if len(kept) > 1 else float("nan"),
        "n_without": len(kept)}
    return result


def human_vs_human(rows_a, rows_b):
    """Agreement between two raters on the sample_ids both labeled (rows from read_human). Pairs use `human` for A and `llm` for B
    so bootstrap() and the statistics are the same ones used against the LLM."""
    b = {r["sample_id"]: r for r in rows_b}
    pairs = [{"sample_id": r["sample_id"], "query_id": r["query_id"], "query": r.get("query"), "name": r["name_fixed"],
              "human": r["human"], "llm": b[r["sample_id"]]["human"], "note_a": (r.get("note") or "").strip(),
              "note_b": (b[r["sample_id"]].get("note") or "").strip()} for r in rows_a if r["sample_id"] in b]
    a_, b_ = [p["human"] for p in pairs], [p["llm"] for p in pairs]
    return {"n": len(pairs), "only_a": sorted({r["sample_id"] for r in rows_a} - set(b)),
            "only_b": sorted(set(b) - {r["sample_id"] for r in rows_a}),
            "agreement": raw_agreement(a_, b_), "kappa": cohen_kappa(a_, b_), "kappa_quadratic": cohen_kappa(a_, b_, "quadratic"),
            "alpha_ordinal": krippendorff_alpha([[x, y] for x, y in zip(a_, b_)], "ordinal"),
            "confusion": confusion_matrix(a_, b_), "ci_agreement": bootstrap(raw_agreement, pairs),
            "ci_kappa_quadratic": bootstrap(lambda x, y: cohen_kappa(x, y, "quadratic"), pairs),
            "mean_a": mean(a_), "mean_b": mean(b_), "one_step": sum(abs(x - y) == 1 for x, y in zip(a_, b_)),
            "two_step": sum(abs(x - y) == 2 for x, y in zip(a_, b_)),
            "disagreements": sorted((p for p in pairs if p["human"] != p["llm"]),
                                    key=lambda p: (-abs(p["human"] - p["llm"]), p["sample_id"]))}


# ---- report -------------------------------------------------------------------
def f3(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def ci(c):
    return "n/a" if c[0] is None else f"[{c[0]:.3f}, {c[1]:.3f}]"


def cell(text):
    return " ".join((text or "").split()).replace("|", "\\|")


CORRECTION = ("**Correction (2026-11-15).** An earlier version of this report used a `labels_kang.csv` that was not the rater's human "
              "labeling (the wrong file was sent). Every number computed from it is withdrawn, including human-vs-human quadratic kappa "
              "0.788 / 0.790 and Kang vs LLM quadratic kappa 0.899. The sections below use the rater's actual file; "
              "Pujitha vs LLM was never affected.")


def human_sections(args, rows_a, categories, second):
    """Human-vs-human and second-rater-vs-LLM sections. second: {"name", "rows", "blank", "matched", "unmatched", "result", "hh"}."""
    hh, r2, r1 = second["hh"], second["result"], second["primary_result"]
    na, nb = second["primary_name"], second["name"]
    L = [f"## Human vs human: {na} and {nb}", "", CORRECTION, "",
         f"- Second rater: `{rel(args.human2)}` ({len(second['rows']) + len(second['blank'])} rows; {len(second['blank'])} blank). "
         "Per the project workflow neither rater was shown the LLM labels (not verifiable from the files; see Limitations).",
         f"- **Shared labeled pairs: {hh['n']}** (labeled by {na} only: {len(hh['only_a'])}; by {nb} only: {len(hh['only_b'])}). "
         "Same sample sheet; matched on `sample_id`.", "",
         "| Statistic | Value | 95% CI |", "|---|---|---|",
         f"| Exact agreement | {f3(hh['agreement'])} | {ci(hh['ci_agreement'])} |",
         f"| Cohen's kappa (unweighted) | {f3(hh['kappa'])} | |",
         f"| Cohen's kappa (quadratic-weighted) | {f3(hh['kappa_quadratic'])} | {ci(hh['ci_kappa_quadratic'])} |",
         f"| Krippendorff's alpha (ordinal) | {f3(hh['alpha_ordinal'])} | |",
         f"| Mean label: {na} / {nb} | {f3(hh['mean_a'])} / {f3(hh['mean_b'])} | |",
         f"| Off by one step / by two | {hh['one_step']} / {hh['two_step']} | |", "",
         f"Confusion matrix (rows: {na}, columns: {nb}):", "", f"| {na} \\ {nb} | 0 | 1 | 2 | total |", "|---|---|---|---|---|"]
    for lab, row in zip(LABELS, hh["confusion"]):
        L.append(f"| {lab} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
    L += ["", f"### Disagreements: {len(hh['disagreements'])} of {hh['n']}", "",
          f"| Sample | Query | Restaurant | {na} | {nb} | {na}'s note | {nb}'s note |", "|---|---|---|---|---|---|---|"]
    L += [f"| {p['sample_id']} | {p['query_id']} {cell(p.get('query'))} | {cell(p['name'])} | {p['human']} | {p['llm']} | "
          f"{cell(p['note_a']) or '—'} | {cell(p['note_b']) or '—'} |" for p in hh["disagreements"]]
    L += ["", f"## Second rater vs the LLM: {nb}, computed the same way as {na}", "",
          f"Matched {len(second['matched'])} of {len(second['rows'])} labeled rows (unmatched: {len(second['unmatched'])}).", "",
          f"| Statistic | {na} vs LLM | {nb} vs LLM |", "|---|---|---|",
          f"| n matched | {r1['n']} | {r2['n']} |",
          f"| Raw agreement | {f3(r1['agreement'])} {ci(r1['ci_agreement'])} | {f3(r2['agreement'])} {ci(r2['ci_agreement'])} |",
          f"| Cohen's kappa (unweighted) | {f3(r1['kappa'])} | {f3(r2['kappa'])} |",
          f"| Cohen's kappa (quadratic) | {f3(r1['kappa_quadratic'])} {ci(r1['ci_kappa_quadratic'])} | "
          f"{f3(r2['kappa_quadratic'])} {ci(r2['ci_kappa_quadratic'])} |",
          f"| Krippendorff's alpha (ordinal) | {f3(r1['alpha_ordinal'])} | {f3(r2['alpha_ordinal'])} |",
          f"| Mean label: human / LLM | {f3(r1['mean_human'])} / {f3(r1['mean_llm'])} | {f3(r2['mean_human'])} / {f3(r2['mean_llm'])} |",
          f"| LLM above / below human | {r1['llm_higher']} / {r1['llm_lower']} | {r2['llm_higher']} / {r2['llm_lower']} |",
          f"| Off by one step / by two | {r1['one_step']} / {len(r1['big_disagreements'])} | {r2['one_step']} / {len(r2['big_disagreements'])} |",
          f"| Agreement without the 'no evidence' flagged pairs | {f3(r1['no_evidence_mismatch']['agreement_without'])} "
          f"(n={r1['no_evidence_mismatch']['n_without']}) | {f3(r2['no_evidence_mismatch']['agreement_without'])} "
          f"(n={r2['no_evidence_mismatch']['n_without']}) |", "",
          f"{nb} vs LLM confusion matrix (rows: {nb}, columns: LLM):", "", f"| {nb} \\ LLM | 0 | 1 | 2 | total |", "|---|---|---|---|---|"]
    for lab, row in zip(LABELS, r2["confusion"]):
        L.append(f"| {lab} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
    L += ["", f"Agreement with the LLM by query category ({na} / {nb}):", "",
          "| Category | n | Agreement | Kappa (quadratic) |", "|---|---|---|---|"]
    L += [f"| {c} | {r1['by_category'][c]['n']} / {r2['by_category'][c]['n']} | "
          f"{f3(r1['by_category'][c]['agreement'])} / {f3(r2['by_category'][c]['agreement'])} | "
          f"{f3(r1['by_category'][c]['kappa_quadratic'])} / {f3(r2['by_category'][c]['kappa_quadratic'])} |" for c in r1["by_category"]]
    L.append("")
    return L


def write_report(path, *, args, rows, blank, matched, unmatched, result, categories, provider_counts, second=None):
    n_llm = sum(provider_counts.values())
    L = ["# Judge validation: human labels vs the LLM judge", "",
         f"- Human labels: `{rel(args.human)}` ({len(rows) + len(blank)} rows; {len(blank)} with a blank label skipped; {len(rows)} labeled). "
         "Labeled blind to the LLM labels" + (", by two raters (second rater: see below)." if second else ", by one rater."),
         f"- LLM labels: `{rel(args.llm)}` ({n_llm} labeled pairs, by source: {', '.join(f'{k} {v}' for k, v in provider_counts.items())})",
         f"- **Matched: {len(matched)} of {len(rows)} labeled rows** "
         f"(by {', '.join(f'{k}: {v}' for k, v in Counter(p['method'] for p in matched).items()) or 'none'}); "
         f"{len(unmatched)} could not be matched and are excluded.",
         "- Relevance scale 0/1/2 (eval/RUBRIC.md). Bootstrap 95% CIs resample items (seed 6165).", ""]
    if not matched:
        L += ["No rows matched; nothing to report.", ""]
    else:
        r = result
        L += ["## Agreement", "", "| Statistic | Value | 95% CI |", "|---|---|---|",
              f"| n matched | {r['n']} | |",
              f"| Raw agreement | {f3(r['agreement'])} | {ci(r['ci_agreement'])} |",
              f"| Cohen's kappa (unweighted) | {f3(r['kappa'])} | |",
              f"| Cohen's kappa (quadratic-weighted) | {f3(r['kappa_quadratic'])} | {ci(r['ci_kappa_quadratic'])} |",
              f"| Krippendorff's alpha (ordinal) | {f3(r['alpha_ordinal'])} | |",
              f"| Mean label: human / LLM | {f3(r['mean_human'])} / {f3(r['mean_llm'])} | |",
              f"| LLM above / below human | {r['llm_higher']} / {r['llm_lower']} | |",
              f"| Off by one step / by two | {r['one_step']} / {len(r['big_disagreements'])} | |", "",
              "## Confusion matrix (rows: human, columns: LLM)", "", "| human \\ LLM | 0 | 1 | 2 | total |", "|---|---|---|---|---|"]
        for lab, row in zip(LABELS, r["confusion"]):
            L.append(f"| {lab} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
        nm = r["no_evidence_mismatch"]
        L += ["", "## Data-quality flag: the judge reported no evidence where you saw some", "",
              f"{nm['n']} matched pairs have an LLM call (not the no-evidence rule) whose reason says \"no evidence found\" and cites no "
              f"review, although your sheet showed retrieved or fallback evidence for them (mean human label {f3(nm['human_mean'])}; "
              f"samples {', '.join(nm['ids']) or 'none'}). The judge may have been shown different (or empty) evidence than the sheet; "
              "this should be checked against the pool the judge actually read before trusting those labels. Sensitivity analysis without "
              f"these pairs: n = {nm['n_without']}, raw agreement {f3(nm['agreement_without'])}, quadratic kappa {f3(nm['kappa_quadratic_without'])} "
              f"(with them: {f3(r['agreement'])} and {f3(r['kappa_quadratic'])}).",
              "", "## Mean label by LLM label source", "", "| Source | n | Mean human | Mean LLM | Agreement |", "|---|---|---|---|---|"]
        L += [f"| {s} | {v['n']} | {f3(v['mean_human'])} | {f3(v['mean_llm'])} | {f3(v['agreement'])} |" for s, v in r["by_source"].items()]
        L += ["", "## Agreement by query category", "",
              "| Category | n | Agreement | Kappa (quadratic) | Mean human | Mean LLM | LLM above / below |", "|---|---|---|---|---|---|---|"]
        L += [f"| {c} | {v['n']} | {f3(v['agreement'])} | {f3(v['kappa_quadratic'])} | {f3(v['mean_human'])} | {f3(v['mean_llm'])} | "
              f"{v['llm_higher']} / {v['llm_lower']} |" for c, v in r["by_category"].items()]
        L += ["", "Cells with few queries are noisy; read the category rows as descriptions of this sample, not estimates.", "",
              "## Largest disagreements (human and LLM two steps apart)", ""]
        big = r["big_disagreements"]
        if not big:
            L += ["None."]
        for title, group in (("Human 0, LLM 2 (the LLM is far more generous)", [p for p in big if p["human"] == 0]),
                             ("Human 2, LLM 0 (the LLM is far stricter)", [p for p in big if p["human"] == 2])):
            if group:
                L += [f"### {title}: {len(group)}", "", "| Sample | Query | Restaurant | Source | Your note | LLM reason |", "|---|---|---|---|---|---|"]
                L += [f"| {p['sample_id']} | {p['query_id']} {cell(p.get('query'))} ({categories.get(p['query_id'], '?')}) | {cell(p['name'])} | "
                      f"{label_source(p['llm_record'])} | {cell(p.get('note')) or '—'} | {cell(p['llm_record'].get('reason'))} |" for p in group]
                L.append("")
    if second:
        L += human_sections(args, rows, categories, second)
    if unmatched:
        L += ["## Rows not matched", "", "| Sample | Query | Restaurant | Why |", "|---|---|---|---|"]
        L += [f"| {r['sample_id']} | {r['query_id']} | {cell(r['name'])} | {why} |" for r, why in unmatched]
        L.append("")
    L += ["## Limitations", "",
          ("- **Two raters.** Human-human agreement is computed above on the shared sample_ids, but the two raters are not a consensus: "
           "kappa and alpha measure how often they read the rubric alike, and both are small-sample estimates."
           if second else
           "- **One rater.** No human-human agreement can be computed, so these numbers measure agreement with one person's reading of the "
           "rubric, not with a consensus; that person's own consistency is unknown."),
          f"- **n = {len(matched)}** matched pairs. The CIs above are wide; per-category cells (4 to 11 queries) and the confusion-matrix cells "
          "are small, and the kappa values depend heavily on a few pairs.",
          f"- **A sample, not the pool.** The rater labeled a stratified sample of {len(matched)} of the {SAMPLE_POOL:,} pairs judged when the "
          f"sample was drawn (about {len(matched) / SAMPLE_POOL:.0%}; drawn by `eval/make_sample.py`), so errors in the unsampled pairs are unmeasured. "
          f"The label file now holds {n_llm:,} pairs because pairs were added later.",
          "- **The sample predates the new labels.** It was drawn from the earlier judging pool; the pairs added later for the stance re-ranker "
          "(top-20 candidates, a single provider: Groq, `gpt-oss-120b`) are not validated here, and provider-specific behaviour is only "
          "visible through the 'by source' table above.",
          "- **Same judge model throughout.** All LLM labels use one model family and one prompt, so shared blind spots (for example, treating a "
          "mention of the topic as support) would not show up as disagreement between LLM sources.",
          *(["- **Blindness is not verifiable.** Neither rater was shown the LLM labels per the workflow, but the files cannot prove it."
             + (" The second rater agrees with the LLM more than with the first rater, so a shared reading of the rubric and contamination cannot be "
                "told apart here; confirm with the rater before leaning on that gap."
                if second["result"]["agreement"] > second["hh"]["agreement"] else "")] if second else []),
          "- **The rater saw the evidence shown to the judge**, not the full review text of each restaurant; both are limited to what retrieval "
          "returned, so agreement says nothing about whether the shown reviews were representative.",
          "- Matching rows to restaurants relies on the review ids cited in the evidence; rows without evidence are matched by name and borough "
          "only when that is unambiguous.", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate the LLM judge against one human rater's labels")
    ap.add_argument("--human", type=Path, default=HUMAN_PATH)
    ap.add_argument("--human2", type=Path, default=HUMAN2_PATH, help="second rater's labels (skipped if the file is missing)")
    ap.add_argument("--llm", type=Path, default=LLM_PATH)
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH)
    ap.add_argument("--sample-key", type=Path, default=SAMPLE_KEY_PATH)
    ap.add_argument("--meta", type=Path, default=META_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)
    for p in (args.human, args.llm, args.queries):
        if not p.exists():
            sys.exit(f"Missing {p}")
    try:
        rows, blank = read_human(args.human)
    except ValueError as e:
        sys.exit(str(e))
    llm = read_llm(args.llm)
    from src.retrieval.evidence import ReviewStore
    store = ReviewStore()
    review_gmap = lambda ids: {r["review_id"]: r["gmap_id"] for r in store.get(ids)}
    matched, unmatched = resolve(rows, llm, read_meta(args.meta), review_gmap, read_sample_key(args.sample_key))
    categories = read_categories(args.queries)
    result = analyze(matched, categories) if matched else {}
    provider_counts = Counter(label_source(r) for r in llm.values())
    second = None
    if args.human2 and args.human2.exists():
        try:
            rows2, blank2 = read_human(args.human2)
        except ValueError as e:
            sys.exit(str(e))
        matched2, unmatched2 = resolve(rows2, llm, read_meta(args.meta), review_gmap, read_sample_key(args.sample_key))
        second = {"name": args.human2.stem.replace("labels_", "").capitalize(), "primary_name": args.human.stem.replace("labels_", "").capitalize(),
                  "rows": rows2, "blank": blank2, "matched": matched2, "unmatched": unmatched2,
                  "result": analyze(matched2, categories) if matched2 else {}, "primary_result": result,
                  "hh": human_vs_human(rows, rows2)}
    write_report(args.out, args=args, rows=rows, blank=blank, matched=matched, unmatched=unmatched, result=result,
                 categories=categories, provider_counts=dict(provider_counts.most_common()), second=second)
    print(f"{len(rows)} labeled rows ({len(blank)} blank skipped); matched {len(matched)}, unmatched {len(unmatched)}")
    if matched:
        print(f"agreement {f3(result['agreement'])}, kappa {f3(result['kappa'])}, quadratic kappa {f3(result['kappa_quadratic'])}, "
              f"alpha {f3(result['alpha_ordinal'])}")
    if second:
        hh = second["hh"]
        print(f"human vs human: n {hh['n']}, agreement {f3(hh['agreement'])}, quadratic kappa {f3(hh['kappa_quadratic'])}, "
              f"alpha {f3(hh['alpha_ordinal'])}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
