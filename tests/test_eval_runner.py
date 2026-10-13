"""Offline tests for eval_pipeline.py: incomplete cases are never scored, corpus guard, settings passed through."""
import sys

import pytest

import eval_pipeline as E


def _state(**over):
    base = {"all_domains": ["taxation"], "verified_citations": ["income_tax_act_1961_sec_195"], "abstain": False,
            "answerability_grade": "answers", "domain_gaps": [], "answer": "ANSWER", "groundedness_grade": "grounded",
            "domain_results": [{"domain": "taxation", "reason": "ok", "steps": [
                {"verdict": "entailed"}, {"verdict": "partial"}]}]}
    base.update(over)
    return base


CASE = {"id": "T1", "category": "single", "q": "withhold tax on a foreign payment?", "domains": {"taxation"},
        "groups": [["income_tax_act_1961_sec_"]], "optional": set()}


def test_clean_run_is_recorded_with_claim_counts_and_settings(monkeypatch):
    monkeypatch.setattr(E, "run_workflow", lambda q, **kw: (_state(), kw)[0])
    rec = E.run_case(CASE)
    assert rec["error"] is None and rec["claims"] == {"entailed": 1, "partial": 1, "unchecked": 0}
    assert rec["answer"] == "ANSWER"


def test_settings_are_passed_to_the_workflow(monkeypatch):
    seen = {}
    monkeypatch.setattr(E, "run_workflow", lambda q, **kw: (seen.update(kw), _state())[1])
    monkeypatch.setitem(E.SETTINGS, "provider", "groq")
    monkeypatch.setitem(E.SETTINGS, "top_k", 5)
    monkeypatch.setitem(E.SETTINGS, "as_of", "2026-04-01")
    E.run_case(CASE)
    assert seen["provider"] == "groq" and seen["top_k"] == 5 and seen["as_of"] == "2026-04-01"


def test_quota_failure_inside_a_worker_makes_the_case_incomplete_not_a_result(monkeypatch):
    bad = _state(domain_results=[{"domain": "taxation", "reason": "LLM unavailable: daily quota exhausted"}])
    monkeypatch.setattr(E, "run_workflow", lambda q, **kw: bad)
    with pytest.raises(E.IncompleteCase):
        E.run_case(CASE)
    monkeypatch.setattr(E, "run_workflow", lambda q, **kw: _state(groundedness_grade="unverified"))
    with pytest.raises(E.IncompleteCase):
        E.run_case(CASE)


def test_runner_refuses_to_start_on_the_wrong_corpus_size(monkeypatch, tmp_path):
    from legal_workflow_generator.agent import nodes
    from legal_workflow_generator.agent.llm_limits import LIMITERS
    rpm_before = LIMITERS["gemini"].rpm
    monkeypatch.setattr(sys, "argv", ["eval_pipeline.py", "--out", str(tmp_path / "x.jsonl"),
                                      "--expect-provisions", "5", "--rpm", "3"])
    monkeypatch.setattr(nodes, "_connect", lambda: (_ for _ in ()).throw(RuntimeError("no db in test")))
    monkeypatch.setattr(sys, "argv", ["eval_pipeline.py", "--out", str(tmp_path / "x.jsonl"),
                                      "--expect-provisions", "5"])
    with pytest.raises(SystemExit) as err:
        E.main()
    assert "expected 5" in str(err.value)
    assert not (tmp_path / "x.jsonl.manifest.json").exists()     # nothing written before the check
    assert LIMITERS["gemini"].rpm == rpm_before                  # shared limits untouched by a refused run
