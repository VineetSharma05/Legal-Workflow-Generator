"""Multi-session behaviour of the 100-query harness: stop on quota, resume, never score a degraded query."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

import eval_complete_multidomain as E  # noqa: E402
import legal_workflow_generator.agent.graph as G  # noqa: E402
import legal_workflow_generator.agent.nodes as N  # noqa: E402
from run_status import EXIT_DONE, EXIT_PARTIAL, EXIT_STOPPED, FailureTracker, is_hard_stop  # noqa: E402

SELECTED = [t for t in E.TEST_QUERIES if not t["should_abstain"]][:6]


def answer(test):
    return {"abstain": False, "all_domains": test["domains"], "domain": test["domains"][0],
            "verified_citations": test["expected_sections"][:1], "domain_results": [], "trace": []}


class FakeGraph:
    """Answers each query; `script` maps a query index to "quota", "degraded" or "error"."""

    def __init__(self, script=None):
        self.script, self.calls = script or {}, []

    def invoke(self, inp, config=None):
        q = inp["query"]
        i = [t["query"] for t in SELECTED].index(q)
        self.calls.append(i)
        what = self.script.get(i)
        test = SELECTED[i]
        if what == "quota":
            return {**answer(test), "abstain": True, "llm_error": "gemini: gemini daily quota exhausted"}
        if what == "degraded":          # a stage swallowed a failed call and carried on
            N._note_failure("LLMUnavailable: gemini unavailable after 5 attempts")
            return answer(test)
        if what == "error":
            raise RuntimeError("boom")
        return answer(test)


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "RESULTS_DIR", tmp_path)

    def _run(fake, resume=True, options=None):
        monkeypatch.setattr(G, "graph", fake)
        return E.evaluate(SELECTED, sleep_s=0, provider="gemini", as_of="2026-10-15", tag="t",
                          resume=resume, options=options or {"suggest_provisions": "true"})
    return _run, tmp_path


def saved(tmp_path):
    return [json.loads(l) for l in (tmp_path / "eval_t.jsonl").read_text().splitlines() if l.strip()]


def test_quota_stop_then_resume_next_day(run):
    _run, d = run
    assert _run(FakeGraph({2: "quota"}), resume=False) == EXIT_STOPPED       # day 1: stops at query 3
    assert [r["query"] for r in saved(d)] == [t["query"] for t in SELECTED[:2]]
    day2 = FakeGraph()
    assert _run(day2) == EXIT_DONE                                           # day 2: only the rest is run
    assert day2.calls == [2, 3, 4, 5]
    rows = saved(d)
    assert len(rows) == len({r["query"] for r in rows}) == 6                 # nothing lost, nothing doubled
    summary = json.loads((d / "eval_t_summary.json").read_text())["summary"]
    assert summary["complete"] and summary["scored"] == 6
    assert len(json.loads((d / "eval_t.manifest.json").read_text())["sessions"]) == 2


def test_degraded_query_is_not_scored_and_is_retried(run):
    _run, d = run
    assert _run(FakeGraph({1: "degraded"}), resume=False) == EXIT_PARTIAL
    assert SELECTED[1]["query"] not in {r["query"] for r in saved(d)}
    again = FakeGraph()
    assert _run(again) == EXIT_DONE and again.calls == [1]


def test_three_failures_in_a_row_stop_the_run(run):
    _run, d = run
    fake = FakeGraph({0: "error", 1: "error", 2: "error"})
    assert _run(fake, resume=False) == EXIT_STOPPED and fake.calls == [0, 1, 2]


def test_resume_with_different_settings_is_refused(run):
    _run, d = run
    _run(FakeGraph({2: "quota"}), resume=False)
    assert _run(FakeGraph(), options={"suggest_provisions": "false"}) == 1
    assert len(saved(d)) == 2                                                # saved rows untouched


def test_abstained_answer_earns_no_citation_credit():
    t = SELECTED[0]
    row = E.score_result(t, {**answer(t), "abstain": True}, 1.0)
    assert row["citation_hits"] == 0 and row["verified_citations"] == []
    assert row["verified_citations_before_abstain"] == t["expected_sections"][:1]


def test_hard_stop_markers():
    assert is_hard_stop("gemini: gemini daily quota exhausted")
    assert is_hard_stop("402 RESOURCE_EXHAUSTED. Your prepayment credits are depleted.")
    assert is_hard_stop("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    # the ordinary per-minute 429 also mentions billing; it must be waited out, not stop the run
    assert not is_hard_stop("429 You exceeded your current quota, please check your plan and billing details. "
                            "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    t = FailureTracker(limit=3)
    t.failed("x"), t.ok(), t.failed("y"), t.failed("z")
    assert not t.should_stop
    t.failed("w")
    assert t.should_stop
