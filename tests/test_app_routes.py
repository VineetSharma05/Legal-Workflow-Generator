"""Offline API contract tests with external pipelines mocked."""

from fastapi import HTTPException

import app


def test_index_serves_frontend():
    response = app.index()

    assert response.path.endswith("presentation\\static\\index.html")


def test_chat_returns_pipeline_sections(monkeypatch):
    state = {
        "answer": "Use a consent notice.",
        "intent": "workflow",
        "domain": "data_protection",
        "all_domains": ["data_protection"],
        "retrieved_docs": [{
            "provision_id": "dpdp_act_2023_sec_5",
            "title": "Notice",
            "text": "A notice must be given.",
            "combined_score": 0.9,
            "bm25_score": 0.8,
            "semantic_score": 0.95,
        }],
        "citations": ["dpdp_act_2023_sec_5"],
        "verified_citations": ["dpdp_act_2023_sec_5"],
        "trace": ["done"],
    }

    class FakeGraph:
        def invoke(self, payload):
            assert payload == {"query": "How do I comply?"}
            return state

    monkeypatch.setattr(app, "graph", FakeGraph())

    result = app.chat(app.ChatRequest(query="How do I comply?"))

    assert result["answer"] == state["answer"]
    assert result["pipeline"]["classification"]["domain"] == "data_protection"
    assert result["pipeline"]["retrieval"]["docs"][0]["provision_id"] == (
        "dpdp_act_2023_sec_5"
    )
    assert result["pipeline"]["generation"]["citations"] == state["citations"]
    assert result["trace"] == ["done"]


def test_chat_converts_pipeline_errors_to_http_error(monkeypatch):
    class FailingGraph:
        def invoke(self, payload):
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(app, "graph", FailingGraph())

    try:
        app.chat(app.ChatRequest(query="What is required?"))
    except HTTPException as error:
        assert error.status_code == 500
        assert "database unavailable" in error.detail
    else:
        raise AssertionError("chat should convert pipeline failures to HTTP 500")


def test_query_route_returns_classifier_state(monkeypatch):
    monkeypatch.setattr(
        app,
        "agent_classify_query",
        lambda state: {
            **state,
            "intent": "qa",
            "domain": "taxation",
            "all_domains": ["taxation"],
            "normalized_query": "gst input credit",
            "keywords": ["gst", "input credit"],
            "confidence": 0.91,
            "trace": ["classified"],
        },
    )

    result = app.test_query(app.TestRequest(query="What is GST input credit?"))

    assert result["domain"] == "taxation"
    assert result["confidence"] == 0.91
    assert result["trace"] == ["classified"]
