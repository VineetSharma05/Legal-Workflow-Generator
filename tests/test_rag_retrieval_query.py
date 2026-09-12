"""Unit tests for ``RagPipeline._build_retrieval_query`` (pure string building).

``RagPipeline.__init__`` wires up a real searcher + generator, so we bypass it
with ``__new__`` and test the method in isolation.
"""

import pytest

from legal_workflow_generator.rag.pipeline import RagPipeline
from legal_workflow_generator.typings.types import QueryIntent


@pytest.fixture
def pipeline() -> RagPipeline:
    return RagPipeline.__new__(RagPipeline)


def test_no_context_returns_raw_query(pipeline):
    assert pipeline._build_retrieval_query("raw q", None) == "raw q"


def test_adds_domain_intent_and_keyword_hints(pipeline):
    ctx = {
        "normalized_query": "how to comply with data protection law",
        "legal_domain": "data_protection",
        "keywords": ["consent", "data fiduciary"],
        "intent": QueryIntent.WORKFLOW,
        "confidence": 0.9,
    }
    out = pipeline._build_retrieval_query("orig", ctx)
    assert out.startswith("how to comply with data protection law")
    assert "domain data protection" in out
    assert "keywords consent data fiduciary" in out
    assert "step by step procedure compliance checklist" in out


def test_unknown_domain_is_not_appended(pipeline):
    ctx = {"normalized_query": "q", "legal_domain": "unknown", "keywords": [], "confidence": 0.9}
    assert "domain" not in pipeline._build_retrieval_query("orig", ctx)


def test_low_confidence_adds_broad_hint(pipeline):
    ctx = {"normalized_query": "q", "legal_domain": "taxation", "keywords": [], "confidence": 0.2}
    assert "broad legal context" in pipeline._build_retrieval_query("orig", ctx)


def test_plain_string_intent_is_accepted(pipeline):
    ctx = {"normalized_query": "q", "legal_domain": "taxation", "keywords": [],
           "intent": "compliance_check", "confidence": 0.8}
    out = pipeline._build_retrieval_query("orig", ctx)
    assert "compliance requirements obligations penalties verification" in out
