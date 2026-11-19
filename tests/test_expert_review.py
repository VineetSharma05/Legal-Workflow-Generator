"""Offline tests for the expert-review harness (sheet builder + scorer)."""
import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals" / "expert_review"))
build = importlib.import_module("build_review_sheet")
score = importlib.import_module("score_reviews")

SCORES = ["C1_citation_faithfulness", "C2_applicability", "C3_completeness", "C4_no_invented_obligations",
          "C5_temporal_validity", "U1_actionability", "U2_clarity_ordering", "U3_honest_gaps"]


def _results(path, n=8):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"T{i:02d}", "category": "two", "error": None,
                                "answer": f"answer {i}", "cited": ["ida_1947_sec_2"]}) + "\n")
        f.write(json.dumps({"id": "BAD", "category": "x", "error": "boom", "answer": ""}) + "\n")
    return path


def _fill(rows, value="4", critical="N"):
    for r in rows:
        for c in SCORES:
            r[c] = value
        r["O1_send_to_client"] = "with_edits"
        r["O2_critical_error"] = critical
    return rows


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_sheet_builder_skips_errored_rows_and_double_scores_a_fraction(tmp_path):
    res = _results(tmp_path / "r.jsonl")
    info = build.build(build.load_completed(str(res)), str(tmp_path / "out"), sample=None,
                       double=0.5, reviewers=2, seed=1)
    assert info["rows"] == 8 and info["double_scored"] == 4
    r1 = list(csv.DictReader(open(info["files"][0], encoding="utf-8")))
    r2 = list(csv.DictReader(open(info["files"][1], encoding="utf-8")))
    assert len(r1) == 8 and len(r2) == 4                      # reviewer 2 only gets the doubled rows
    assert {r["case_id"] for r in r2} <= {r["case_id"] for r in r1}
    assert all(r["double_scored"] == "yes" for r in r2)
    assert "BAD" not in {r["case_id"] for r in r1}
    assert "ida_1947_sec_2" in r1[0]["cited_provisions"]      # the reviewer can check the citations


def test_scorer_rejects_invalid_scores_and_measures_agreement(tmp_path):
    base = [{"case_id": f"T{i}", "category": "two", "double_scored": "yes", "question": "", "workflow_text": "",
             "cited_provisions": "", "reviewer": "", **{c: "" for c in SCORES},
             "O1_send_to_client": "", "O2_critical_error": "", "comment": ""} for i in range(4)]
    r1 = _fill([dict(r) for r in base], value="4")
    r2 = _fill([dict(r) for r in base], value="4")
    r2[0]["U1_actionability"] = "9"                           # invalid: must be excluded
    r1[1]["O2_critical_error"] = "Y"
    p1, p2 = tmp_path / "reviewer_1.csv", tmp_path / "reviewer_2.csv"
    _write(p1, r1)
    _write(p2, r2)

    rows, problems = score.read_rows([str(p1), str(p2)])
    assert len(problems) == 1 and "U1_actionability" in problems[0]
    text = score.summarize(rows, problems)
    assert "Critical error rate" in text and "1 of 7 reviews" in text
    assert "Exact agreement on criterion scores: 100%" in text   # the pairs were found
    assert "Cases with two reviews: 3" in text                     # T0 lost its second review


def test_linear_weighted_kappa_known_values():
    assert score.linear_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    k = score.linear_weighted_kappa([1, 2, 3, 4, 5, 4, 3], [1, 2, 4, 4, 5, 3, 3])
    assert 0.0 < k < 1.0
