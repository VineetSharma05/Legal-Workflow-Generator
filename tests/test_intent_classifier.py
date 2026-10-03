"""Unit tests for :class:`IntentClassifier` with the Gemini call mocked."""

import pytest

from legal_workflow_generator.query.intent_classifier import IntentClassifier
from legal_workflow_generator.typings.types import NormalizedQuery, QueryIntent


def _nq(text: str) -> NormalizedQuery:
    return NormalizedQuery(original=text, normalized=text, source="text")


@pytest.fixture
def classifier(fake_gemini_client, request):
    """IntentClassifier whose Gemini client is replaced by a fake."""
    replies = getattr(request, "param", "INTENT: qa\nCONFIDENCE: 0.9\nREASON: factual")
    clf = IntentClassifier()
    clf.client = fake_gemini_client(replies)
    return clf


@pytest.mark.parametrize(
    "classifier, expected",
    [
        ("INTENT: qa\nCONFIDENCE: 0.91\nREASON: x", QueryIntent.QA),
        ("INTENT: workflow\nCONFIDENCE: 0.88\nREASON: x", QueryIntent.WORKFLOW),
        ("INTENT: compliance_check\nCONFIDENCE: 0.7\nREASON: x", QueryIntent.COMPLIANCE_CHECK),
        ("INTENT: unknown\nCONFIDENCE: 0.95\nREASON: x", QueryIntent.UNKNOWN),
    ],
    indirect=["classifier"],
)
def test_parses_each_intent(classifier, expected):
    intent, confidence = classifier.classify(_nq("some query"))
    assert intent == expected
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize(
    "classifier", ["INTENT: workflow\nCONFIDENCE: 0.3\nREASON: shaky"], indirect=True
)
def test_low_confidence_is_overridden_to_unknown(classifier):
    intent, confidence = classifier.classify(_nq("vague thing"))
    assert intent == QueryIntent.UNKNOWN
    assert confidence == pytest.approx(0.3)


@pytest.mark.parametrize(
    "classifier", ["garbage response with no fields"], indirect=True
)
def test_unparseable_response_returns_unknown_zero(classifier):
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


def test_unknown_intent_label_maps_to_unknown_enum(fake_gemini_client):
    clf = IntentClassifier()
    clf.client = fake_gemini_client("INTENT: not_a_real_intent\nCONFIDENCE: 0.99\nREASON: x")
    intent, _ = clf.classify(_nq("anything"))
    assert intent == QueryIntent.UNKNOWN
