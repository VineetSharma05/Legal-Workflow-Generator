"""Offline API contract tests. The workflow is replaced by a fake; no database or LLM is used."""

from pathlib import Path

import pytest
from fastapi import HTTPException

import app


def test_index_serves_frontend():
    response = app.index()
    # compare as a Path so the test works on Windows and POSIX alike
    assert Path(response.path) == app.STATIC_DIR / "index.html"


def _state(**over):
    base = {
        "answer": "raw answer",
        "stitched_answer": "STITCHED ANSWER with gaps",
        "intent": "workflow",
        "domain": "data_protection",
        "all_domains": ["data_protection"],
        "retrieved_docs": [{
            "provision_id": "dpdp_act_2023_sec_5", "title": "Notice", "text": "A notice must be given.",
            "combined_score": 0.9, "bm25_score": 0.8, "semantic_score": 0.95,
        }],
        "citations": ["dpdp_act_2023_sec_5"],
        "verified_citations": ["dpdp_act_2023_sec_5"],
        "domain_results": [{
            "domain": "data_protection", "grade": "sufficient", "steps": [
                {"action": "give notice", "provision_id": "dpdp_act_2023_sec_5", "verdict": "entailed"},
                {"action": "x", "provision_id": "dpdp_act_2023_sec_5", "verdict": "unchecked"},
            ], "dropped_steps": [{"action": "bad", "provision_id": "dpdp_act_2023_sec_5"}],
        }],
        "trace": ["done"],
    }
    base.update(over)
    return base


def test_chat_forwards_provider_top_k_and_as_of_into_the_workflow(monkeypatch):
    seen = {}

    def fake_run(query, **options):
        seen.update(query=query, **options)
        return _state()

    monkeypatch.setattr(app, "run_workflow", fake_run)
    app.chat(app.ChatRequest(query="How do I comply?", provider="groq", top_k=5, as_of="2026-04-01"))
    assert seen == {"query": "How do I comply?", "provider": "groq", "top_k": 5,
                    "as_of": "2026-04-01", "verify_claims": None}


def test_chat_returns_stitched_answer_settings_typed_domains_and_claim_summary(monkeypatch):
    monkeypatch.setattr(app, "run_workflow", lambda q, **o: _state())
    result = app.chat(app.ChatRequest(query="How do I comply?", provider="gemini", top_k=4))

    assert result["answer"] == "STITCHED ANSWER with gaps"            # gaps section is not lost
    assert result["settings"]["provider"] == "gemini" and result["settings"]["top_k"] == 4
    assert result["pipeline"]["domains"][0]["domain"] == "data_protection"
    assert result["pipeline"]["claim_verification"] == {"entailed": 1, "partial": 0, "unchecked": 1, "dropped": 1}
    assert result["pipeline"]["classification"]["domain"] == "data_protection"
    assert result["pipeline"]["retrieval"]["docs"][0]["provision_id"] == "dpdp_act_2023_sec_5"
    assert result["pipeline"]["generation"]["citations"] == ["dpdp_act_2023_sec_5"]
    assert result["trace"] == ["done"]


@pytest.mark.parametrize("bad", [{"provider": "openai"}, {"top_k": 0}, {"top_k": 99}, {"as_of": "13/10/2026"}])
def test_invalid_settings_are_rejected_with_400(bad):
    kwargs = {"query": "q", **bad}
    with pytest.raises(HTTPException) as err:
        app.chat(app.ChatRequest(**kwargs))
    assert err.value.status_code == 400


def test_chat_converts_pipeline_errors_to_http_error(monkeypatch):
    def failing(query, **options):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(app, "run_workflow", failing)
    with pytest.raises(HTTPException) as err:
        app.chat(app.ChatRequest(query="What is required?"))
    assert err.value.status_code == 500 and "database unavailable" in err.value.detail


def test_query_route_passes_provider_and_top_k_to_the_classifier(monkeypatch):
    seen = {}

    def fake_classify(state, config=None):
        seen["config"] = config
        return {**state, "intent": "qa", "domain": "taxation", "all_domains": ["taxation"],
                "normalized_query": "gst input credit", "keywords": ["gst"], "confidence": 0.91,
                "trace": ["classified"]}

    monkeypatch.setattr(app, "agent_classify_query", fake_classify)
    result = app.test_query(app.TestRequest(query="What is GST input credit?", provider="groq", top_k=6))

    assert seen["config"]["configurable"]["provider"] == "groq"
    assert seen["config"]["configurable"]["top_k"] == 6
    assert result["domain"] == "taxation" and result["confidence"] == 0.91
