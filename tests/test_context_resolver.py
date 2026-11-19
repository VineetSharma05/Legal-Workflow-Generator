"""Unit tests for :class:`LegalContextResolver` reconciliation logic.

Both boundaries — the keyword classifier and the structured LLM — are faked, so these tests
exercise only the strategy/agreement decision table.
"""

import pytest

from legal_workflow_generator.llm.schemas import DomainClassification
from legal_workflow_generator.query.context_resolver import LegalContextResolver
from legal_workflow_generator.typings.types import NormalizedQuery, QueryIntent


def _nq(text: str = "some legal query about data") -> NormalizedQuery:
    return NormalizedQuery(original=text, normalized=text, source="text")


class FakeKeywordClassifier:
    def __init__(self, domain="", score=0.0):
        self._result = {
            "domain": domain,
            "score": score,
            "scores": {domain: score} if domain else {},
            "matched_terms": [domain + "_term"] if domain else [],
        }
        self.calls = []

    def classify(self, query_text, threshold=0.0):
        self.calls.append((query_text, threshold))
        return self._result


def _dc(domain: str, *keywords: str) -> DomainClassification:
    return DomainClassification(domain=domain, keywords=list(keywords))


def _resolver(fake_llm, *, keyword, llm_reply=None, **kwargs):
    r = LegalContextResolver(**kwargs)
    r._keyword_classifier = keyword
    if llm_reply is not None:
        r.llm = fake_llm(llm_reply)
        r.sampling_llm = fake_llm(llm_reply)
    return r


def test_high_confidence_unknown_intent_skips_resolution(fake_llm):
    r = _resolver(fake_llm, keyword=FakeKeywordClassifier("taxation", 5.0),
                  llm_reply=_dc("taxation", "a", "b"))
    ctx = r.resolve(_nq(), QueryIntent.UNKNOWN, confidence=0.9)
    assert ctx["legal_domain"] == "unknown"
    assert ctx["domain_source"] == "skipped"
    # neither backend was consulted
    assert r._keyword_classifier.calls == []
    assert r.llm.calls == []


def test_llm_fallback_trusts_keyword_match_without_calling_llm(fake_llm):
    r = _resolver(fake_llm, strategy="llm_fallback",
                  keyword=FakeKeywordClassifier("employment", 0.4),
                  llm_reply=_dc("taxation", "x"))
    ctx = r.resolve(_nq(), QueryIntent.QA, confidence=0.8)
    assert ctx["legal_domain"] == "employment"
    assert ctx["domain_source"] == "keyword"
    assert ctx["domain_agreement"] is True
    assert r.llm.calls == []  # LLM never called


def test_llm_fallback_defers_to_llm_when_no_keyword_match(fake_llm):
    r = _resolver(fake_llm, strategy="llm_fallback",
                  keyword=FakeKeywordClassifier("", 0.0),
                  llm_reply=_dc("ip_licensing", "trademark", "license"))
    ctx = r.resolve(_nq(), QueryIntent.QA, confidence=0.8)
    assert ctx["legal_domain"] == "ip_licensing"
    assert ctx["domain_source"] == "llm"
    assert ctx["keyword_domain"] == "unknown"
    assert r.llm.calls  # LLM was called


def test_combine_agreement(fake_llm):
    r = _resolver(fake_llm, strategy="combine",
                  keyword=FakeKeywordClassifier("taxation", 0.1),
                  llm_reply=_dc("taxation", "gst"))
    ctx = r.resolve(_nq(), QueryIntent.QA, confidence=0.8)
    assert ctx["legal_domain"] == "taxation"
    assert ctx["domain_source"] == "keyword+llm"
    assert ctx["domain_agreement"] is True


def test_combine_disagreement_strong_keyword_wins(fake_llm):
    r = _resolver(fake_llm, strategy="combine", keyword_strong_match_threshold=0.3,
                  keyword=FakeKeywordClassifier("taxation", 0.9),
                  llm_reply=_dc("employment", "salary"))
    ctx = r.resolve(_nq(), QueryIntent.QA, confidence=0.8)
    assert ctx["legal_domain"] == "taxation"
    assert ctx["domain_source"] == "keyword_strong_override"
    assert ctx["domain_agreement"] is False


def test_combine_disagreement_weak_keyword_defers_to_llm(fake_llm):
    r = _resolver(fake_llm, strategy="combine", keyword_strong_match_threshold=0.3,
                  keyword=FakeKeywordClassifier("taxation", 0.05),
                  llm_reply=_dc("employment", "salary"))
    ctx = r.resolve(_nq(), QueryIntent.QA, confidence=0.8)
    assert ctx["legal_domain"] == "employment"
    assert ctx["domain_source"] == "llm_override"
    assert ctx["domain_agreement"] is False


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="Unknown domain classification strategy"):
        LegalContextResolver(strategy="magic")


def test_self_consistency_reports_vote_share(fake_llm):
    replies = [
        _dc("taxation", "gst"),
        _dc("taxation", "gst"),
        _dc("employment", "salary"),
    ]
    r = _resolver(fake_llm, strategy="llm_fallback", self_consistency=True,
                  self_consistency_samples=3,
                  keyword=FakeKeywordClassifier("", 0.0), llm_reply=replies)
    ctx = r.resolve(_nq(), QueryIntent.QA, confidence=0.8)
    assert ctx["legal_domain"] == "taxation"
    assert ctx["domain_confidence"] == pytest.approx(2 / 3)
    assert len(r.sampling_llm.calls) == 3
    assert r.llm.calls == []
