"""Unit tests for the structured-output layer and the agent nodes that use it.

`nodes._llm` is the single LLM boundary in the agent, so it is faked here;
no network or database is touched.
"""

import pytest
from pydantic import ValidationError

from legal_workflow_generator.agent import nodes
from legal_workflow_generator.llm import StructuredOutputError, invoke_structured
from legal_workflow_generator.llm.schemas import (
    AnswerabilityGrade,
    DomainList,
    GroundednessGrade,
    RelevanceGrade,
    SearchQuery,
    domain_workflow_model,
)

DOCS = [
    {"provision_id": "posh_act_2013_sec_4", "title": "Internal Committee", "text": "Every employer shall constitute..."},
    {"provision_id": "posh_act_2013_sec_19", "title": "Duties of employer", "text": "Every employer shall..."},
]
IDS = [d["provision_id"] for d in DOCS]


def _fake_llm(monkeypatch, reply):
    """Patch nodes._llm to return `reply` (or raise it) and record the schema used."""
    seen = []

    def fake(schema, system, user):
        seen.append(schema)
        if isinstance(reply, Exception):
            raise reply
        return reply(schema) if callable(reply) else reply

    monkeypatch.setattr(nodes, "_llm", fake)
    return seen


# ── schemas ───────────────────────────────────────────────────────────────────
def test_domain_workflow_model_restricts_provision_ids():
    Workflow = domain_workflow_model(IDS)
    ok = Workflow(
        summary=["Set up an IC"],
        steps=[{"action": "Constitute IC", "provision_id": IDS[0], "applies_to": "private"}],
        not_covered="",
    )
    assert ok.steps[0].provision_id == IDS[0]

    with pytest.raises(ValidationError):
        Workflow(
            summary=[],
            steps=[{"action": "x", "provision_id": "invented_sec_99", "applies_to": "private"}],
            not_covered="",
        )


def test_domain_workflow_json_schema_enumerates_ids():
    schema = domain_workflow_model(IDS).model_json_schema()
    step = schema["$defs"]["ComplianceStep"]["properties"]["provision_id"]
    assert step["enum"] == IDS


def test_domain_list_rejects_unknown_domain():
    with pytest.raises(ValidationError):
        DomainList(domains=["data_protection", "astrology"])


# ── invoke_structured ─────────────────────────────────────────────────────────
class _Runnable:
    def __init__(self, out):
        self.out = out

    def invoke(self, messages):
        return self.out


def test_invoke_structured_returns_parsed():
    grade = RelevanceGrade(reason="r", grade="sufficient")
    assert invoke_structured(_Runnable({"parsed": grade, "parsing_error": None}), "s", "u") is grade


def test_invoke_structured_raises_on_parsing_error():
    with pytest.raises(StructuredOutputError):
        invoke_structured(_Runnable({"parsed": None, "parsing_error": ValueError("bad")}), "s", "u")


# ── agent nodes ───────────────────────────────────────────────────────────────
def test_grade_domain_uses_structured_grade(monkeypatch):
    _fake_llm(monkeypatch, RelevanceGrade(reason="sec 4 covers it", grade="sufficient"))
    assert nodes._grade_domain("employment", "POSH committee?", DOCS) == ("sufficient", "sec 4 covers it")


def test_grade_domain_parse_failure_is_reported_as_such(monkeypatch):
    _fake_llm(monkeypatch, StructuredOutputError("bad"))
    assert nodes._grade_domain("employment", "q", DOCS) == ("insufficient", "Could not parse grader response")


def test_rewrite_and_focus_fall_back_on_empty_query(monkeypatch):
    _fake_llm(monkeypatch, SearchQuery(query="  "))
    assert nodes._rewrite_for_domain("employment", "orig", "reason") == "orig"
    assert nodes._focus_query("employment", "orig", "fallback") == "fallback"


def test_generate_domain_maps_structured_steps(monkeypatch):
    def reply(schema):
        return schema(
            summary=["Constitute an Internal Committee", " "],
            steps=[
                {"action": "Constitute IC", "provision_id": IDS[0], "applies_to": "private"},
                {"action": "  ", "provision_id": IDS[1], "applies_to": "both"},
                {"action": "Display penal consequences", "provision_id": IDS[1], "applies_to": "unspecified"},
            ],
            not_covered=" ",
        )

    seen = _fake_llm(monkeypatch, reply)
    out = nodes._generate_domain("employment", "POSH steps?", DOCS)

    assert seen[0].__name__ == "DomainWorkflow"
    assert out["summary"] == ["Constitute an Internal Committee"]
    assert [s["provision_id"] for s in out["steps"]] == [IDS[0], IDS[1]]  # empty action dropped
    assert out["cited"] == IDS
    assert out["not_covered"] == ""


def test_generate_domain_failure_returns_empty(monkeypatch):
    _fake_llm(monkeypatch, StructuredOutputError("bad"))
    assert nodes._generate_domain("employment", "q", DOCS)["steps"] == []


def _grader_state():
    return {
        "query": "POSH steps?", "domain": "employment", "all_domains": ["employment"],
        "answer": "Constitute an IC [posh_act_2013_sec_4]", "retrieved_docs": DOCS, "trace": [],
    }


def test_grade_groundedness_records_unsupported_claims(monkeypatch):
    _fake_llm(monkeypatch, GroundednessGrade(reason="r", unsupported_claims=["fine of 1 crore"], grade="not_grounded"))
    state = nodes.grade_groundedness(_grader_state())
    assert state["groundedness_grade"] == "not_grounded"
    assert "fine of 1 crore" in state["trace"][-1]


def test_grade_answerability(monkeypatch):
    _fake_llm(monkeypatch, AnswerabilityGrade(reason="covers it", grade="answers"))
    state = nodes.grade_answerability(_grader_state())
    assert (state["answerability_grade"], state["answerability_reason"]) == ("answers", "covers it")


def test_llm_domains_dedupes(monkeypatch):
    _fake_llm(monkeypatch, DomainList(domains=["employment", "taxation", "employment"]))
    assert nodes._llm_domains("q") == ["employment", "taxation"]
