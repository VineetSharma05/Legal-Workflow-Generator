"""Build blinded expert-review sheets from an evaluation run.

    python evals/expert_review/build_review_sheet.py --results eval_results.jsonl --out review/ \
        [--sample 20] [--double 0.25] [--reviewers 2] [--seed 7]

Only completed evaluation rows are used (errored or incomplete rows are skipped). Each reviewer gets a CSV
with the workflow text, the cited provision ids and blank score columns. Pipeline internals (domain grades,
settings, retrieval scores) are deliberately NOT included, so the review is blind.
"""
import argparse
import csv
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

SCORE_COLS = ["C1_citation_faithfulness", "C2_applicability", "C3_completeness", "C4_no_invented_obligations",
              "C5_temporal_validity", "U1_actionability", "U2_clarity_ordering", "U3_honest_gaps"]
OVERALL_COLS = ["O1_send_to_client", "O2_critical_error", "comment"]
META_COLS = ["case_id", "category", "double_scored", "question", "workflow_text", "cited_provisions"]


def load_completed(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") or not r.get("answer"):
                continue
            rows.append(r)
    return rows


def question_text(case_id: str) -> str:
    """The question for a case id, taken from the evaluation's own gold set."""
    try:
        import eval_pipeline
        for c in eval_pipeline.CASES:
            if c["id"] == case_id:
                return c["q"]
    except Exception:   # noqa: BLE001 - the eval module imports the workflow; fall back gracefully
        pass
    return ""


def build(rows: list, out_dir: str, *, sample: int | None, double: float, reviewers: int, seed: int) -> dict:
    rng = random.Random(seed)
    if sample and sample < len(rows):
        rows = rng.sample(rows, sample)
    rows = sorted(rows, key=lambda r: r["id"])
    n_double = round(len(rows) * double)
    doubled = set(r["id"] for r in rng.sample(rows, n_double)) if n_double else set()

    os.makedirs(out_dir, exist_ok=True)
    cols = META_COLS + SCORE_COLS + OVERALL_COLS + ["reviewer"]
    files = []
    for k in range(reviewers):
        path = os.path.join(out_dir, f"reviewer_{k + 1}.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                is_double = r["id"] in doubled
                # every row goes to reviewer 1; doubled rows also go to reviewer 2 (and 3, ...)
                if k > 0 and not is_double:
                    continue
                w.writerow({
                    "case_id": r["id"], "category": r.get("category", ""),
                    "double_scored": "yes" if is_double else "no",
                    "question": question_text(r["id"]),
                    "workflow_text": r["answer"],
                    "cited_provisions": "; ".join(r.get("cited") or []) or "(none)",
                    "reviewer": f"reviewer_{k + 1}",
                })
        files.append(path)
    return {"rows": len(rows), "double_scored": len(doubled), "files": files}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval_results.jsonl")
    ap.add_argument("--out", default="review")
    ap.add_argument("--sample", type=int, default=None, help="review only a random sample of N completed cases")
    ap.add_argument("--double", type=float, default=0.25, help="fraction of rows scored by two reviewers")
    ap.add_argument("--reviewers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    rows = load_completed(a.results)
    if not rows:
        raise SystemExit("no completed rows in " + a.results)
    info = build(rows, a.out, sample=a.sample, double=a.double, reviewers=a.reviewers, seed=a.seed)
    print(f"{info['rows']} cases, {info['double_scored']} double-scored, files: {', '.join(info['files'])}")


if __name__ == "__main__":
    main()
