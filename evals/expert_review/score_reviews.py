"""Aggregate filled expert-review CSVs (see RUBRIC.md).

    python evals/expert_review/score_reviews.py review/reviewer_*.csv --out review/summary.md

Validates every score (1-5 integers, O1/O2 allowed values); invalid rows are reported and excluded from means.
Reports per-criterion means, the share of answers scoring >= 4, the critical-error rate, and agreement between
reviewers on double-scored cases (exact agreement, mean absolute difference, linear-weighted kappa).
"""
import argparse
import csv
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # run as a script or imported
from build_review_sheet import SCORE_COLS  # noqa: E402

O1_VALUES = {"yes", "with_edits", "no"}
O2_VALUES = {"Y", "N"}


def read_rows(paths):
    rows, problems = [], []
    for p in paths:
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                errs = _validate(r)
                if errs:
                    problems.append(f"{p} case {r.get('case_id')}: {'; '.join(errs)}")
                    continue
                rows.append(r)
    return rows, problems


def _validate(r) -> list:
    errs = []
    for c in SCORE_COLS:
        v = (r.get(c) or "").strip()
        if v == "":
            errs.append(f"{c} missing")
        elif v not in {"1", "2", "3", "4", "5"}:
            errs.append(f"{c}={v!r} not in 1-5")
    if (r.get("O1_send_to_client") or "").strip() not in O1_VALUES:
        errs.append("O1 must be yes/with_edits/no")
    if (r.get("O2_critical_error") or "").strip() not in O2_VALUES:
        errs.append("O2 must be Y or N")
    return errs


def linear_weighted_kappa(a, b, levels=(1, 2, 3, 4, 5)) -> float:
    """Cohen's kappa with linear weights for ordinal scores (1 = perfect agreement, 0 = chance)."""
    n = len(a)
    if n == 0:
        return float("nan")
    idx = {v: i for i, v in enumerate(levels)}
    k = len(levels)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    ra = [sum(row) for row in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = abs(i - j) / (k - 1)
            num += w * obs[i][j]
            den += w * ra[i] * cb[j] / n
    return 1 - num / den if den else float("nan")


def summarize(rows, problems) -> str:
    out = ["# Expert review summary", ""]
    out.append(f"Scored rows used: {len(rows)} (excluded invalid rows: {len(problems)})")
    if problems:
        out += ["", "Invalid rows (fix and re-run):"] + [f"- {p}" for p in problems]
    if not rows:
        return "\n".join(out)

    cases = {r["case_id"] for r in rows}
    out += ["", f"Distinct cases: {len(cases)}", "", "## Scores", "",
            "| Criterion | Mean | Share >= 4 |", "|---|---|---|"]
    for c in SCORE_COLS:
        vals = [int(r[c]) for r in rows]
        out.append(f"| {c} | {statistics.mean(vals):.2f} | {sum(v >= 4 for v in vals) / len(vals):.0%} |")

    crit = [r for r in rows if r["O2_critical_error"].strip() == "Y"]
    send = Counter(r["O1_send_to_client"].strip() for r in rows)
    out += ["", "## Overall", "",
            f"- Would send with light edits or as is: {(send['yes'] + send['with_edits']) / len(rows):.0%} "
            f"(yes {send['yes']}, with edits {send['with_edits']}, no {send['no']})",
            f"- Critical error rate: {len(crit) / len(rows):.0%} ({len(crit)} of {len(rows)} reviews)"]
    if crit:
        out += ["- Critical-error cases: " + ", ".join(sorted({r['case_id'] for r in crit}))]

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r.get("category", "")].append(r)
    out += ["", "## By category (mean of all criteria)", "", "| Category | Reviews | Mean |", "|---|---|---|"]
    for cat in sorted(by_cat):
        vals = [int(r[c]) for r in by_cat[cat] for c in SCORE_COLS]
        out.append(f"| {cat} | {len(by_cat[cat])} | {statistics.mean(vals):.2f} |")

    # agreement on double-scored cases
    per_case = defaultdict(list)
    for r in rows:
        per_case[r["case_id"]].append(r)
    pairs = [(v[0], v[1]) for v in per_case.values() if len(v) >= 2]
    out += ["", "## Inter-rater agreement (double-scored cases)", ""]
    if not pairs:
        out.append("No case was scored by two reviewers yet.")
    else:
        a_all, b_all = [], []
        for x, y in pairs:
            for c in SCORE_COLS:
                a_all.append(int(x[c]))
                b_all.append(int(y[c]))
        exact = sum(p == q for p, q in zip(a_all, b_all)) / len(a_all)
        mad = statistics.mean(abs(p - q) for p, q in zip(a_all, b_all))
        kappa = linear_weighted_kappa(a_all, b_all)
        out += [f"- Cases with two reviews: {len(pairs)}",
                f"- Exact agreement on criterion scores: {exact:.0%}",
                f"- Mean absolute difference: {mad:.2f}",
                f"- Linear-weighted kappa: {kappa:.2f} (target >= 0.6; see RUBRIC.md)"]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows, problems = read_rows(a.csvs)
    text = summarize(rows, problems)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {a.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
