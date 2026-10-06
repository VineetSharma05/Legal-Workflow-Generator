"""Unit tests for :class:`IntentClassifier` with the structured LLM call mocked."""

import pytest

from legal_workflow_generator.llm.schemas import IntentClassification
from legal_workflow_generator.query.intent_classifier import IntentClassifier
from legal_workflow_generator.typings.types import NormalizedQuery, QueryIntent


def _nq(text: str) -> NormalizedQuery:
    return NormalizedQuery(original=text, normalized=text, source="text")


def _ic(intent: str, confidence: float) -> IntentClassification:
    return IntentClassification(reason="x", intent=intent, confidence=confidence)


@pytest.fixture
def classifier(fake_llm, request):
    """IntentClassifier whose structured LLM is replaced by a fake."""
    replies = getattr(request, "param", _ic("qa", 0.9))
    clf = IntentClassifier()
    clf.llm = fake_llm(replies)
    return clf


@pytest.mark.parametrize(
    "classifier, expected",
    [
        (_ic("qa", 0.91), QueryIntent.QA),
        (_ic("workflow", 0.88), QueryIntent.WORKFLOW),
        (_ic("compliance_check", 0.7), QueryIntent.COMPLIANCE_CHECK),
        (_ic("unknown", 0.95), QueryIntent.UNKNOWN),
    ],
    indirect=["classifier"],
)
def test_parses_each_intent(classifier, expected):
    intent, confidence = classifier.classify(_nq("some query"))
    assert intent == expected
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize(
    "classifier", [_ic("workflow", 0.3)], indirect=True
)
def test_low_confidence_is_overridden_to_unknown(classifier):
    intent, confidence = classifier.classify(_nq("vague thing"))
    assert intent == QueryIntent.UNKNOWN
    assert confidence == pytest.approx(0.3)


@pytest.mark.parametrize(
    "classifier", [None], indirect=True
)
def test_schema_mismatch_returns_unknown_zero(classifier):
    intent, confidence = classifier.classify(_nq("anything"))
    assert intent == QueryIntent.UNKNOWN
    assert confidence == 0.0


@pytest.mark.parametrize(
    "classifier", [RuntimeError("network down")], indirect=True
)
def test_api_error_returns_unknown_zero(classifier):
    intent, confidence = classifier.classify(_nq("anything"))
    assert intent == QueryIntent.UNKNOWN
    assert confidence == 0.0


def test_out_of_range_confidence_is_rejected_by_schema():
    with pytest.raises(ValueError):
        IntentClassification(reason="x", intent="qa", confidence=1.5)
