"""End-to-end test of the query unit (`process_query`) with Gemini mocked.

Exercises the real wiring of QueryNormalizer -> IntentClassifier ->
LegalContextResolver; only the two Gemini network calls are faked (intent call
first, then the domain call).
"""

import pytest

from legal_workflow_generator import query as query_module
from legal_workflow_generator.query import process_query
from legal_workflow_generator.typings.types import QueryIntent


@pytest.fixture
def mock_gemini_pipeline(monkeypatch, fake_gemini_client):
    """Patch both Gemini clients and stub the keyword classifier's DB load."""

    replies = [
        "INTENT: workflow\nCONFIDENCE: 0.92\nREASON: asks for steps",
        "DOMAIN: data_protection\nKEYWORDS: dpdp, consent, data",
    ]
    shared_client = fake_gemini_client(replies)

    monkeypatch.setattr(
        query_module.intent_classifier.genai, "Client", lambda *a, **k: shared_client
    )
    monkeypatch.setattr(
        query_module.context_resolver.genai, "Client", lambda *a, **k: shared_client
    )
    # No database in unit tests: pretend the keyword index loaded but matched nothing.
    monkeypatch.setattr(
        query_module.keyword_domain_classifier.KeywordDomainClassifier,
        "ensure_index",
        lambda self: None,
    )
    return shared_client


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
