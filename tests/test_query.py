"""End-to-end test of the query unit (`process_query`) with the LLM mocked.

Exercises the real wiring of QueryNormalizer -> IntentClassifier ->
LegalContextResolver; only the two structured LLM calls are faked (intent call
first, then the domain call).
"""

import pytest

from legal_workflow_generator import query as query_module
from legal_workflow_generator.llm.schemas import DomainClassification, IntentClassification
from legal_workflow_generator.query import process_query
from legal_workflow_generator.typings.types import QueryIntent


@pytest.fixture
def mock_gemini_pipeline(monkeypatch, fake_llm):
    """Patch both structured LLMs and stub the keyword classifier's DB load."""

    replies = [
        IntentClassification(reason="asks for steps", intent="workflow", confidence=0.92),
        DomainClassification(domain="data_protection", keywords=["dpdp", "consent", "data"]),
    ]
    shared_llm = fake_llm(replies)

    monkeypatch.setattr(
        query_module.intent_classifier, "structured_llm", lambda *a, **k: shared_llm
    )
    monkeypatch.setattr(
        query_module.context_resolver, "structured_llm", lambda *a, **k: shared_llm
    )
    # No database in unit tests: pretend the keyword index loaded but matched nothing.
    monkeypatch.setattr(
        query_module.keyword_domain_classifier.KeywordDomainClassifier,
        "ensure_index",
        lambda self: None,
    )
    return shared_llm


def test_process_query_returns_populated_legal_context(mock_gemini_pipeline):
    ctx = process_query(text="What are the steps to comply with DPDP Act as a SaaS startup?")

    assert ctx["intent"] == QueryIntent.WORKFLOW
    assert ctx["confidence"] == pytest.approx(0.92)
    assert ctx["legal_domain"] == "data_protection"
    assert "digital personal data protection" in ctx["normalized_query"]
    assert ctx["original_query"].startswith("What are the steps")
    assert isinstance(ctx["keywords"], list) and ctx["keywords"]


def test_process_query_rejects_too_short_input(mock_gemini_pipeline):
    with pytest.raises(ValueError):
        process_query(text="hi")
