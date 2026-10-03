"""Unit tests for :class:`KeywordDomainClassifier.classify`.

``ensure_index`` normally reads the ``domain_keywords`` table; here we prime the
in-memory index directly so the scoring/matching logic can be tested without a
database.
"""

import pytest

from legal_workflow_generator.query.keyword_domain_classifier import KeywordDomainClassifier


@pytest.fixture
def classifier() -> KeywordDomainClassifier:
    clf = KeywordDomainClassifier()
    # Terms are stored post-stemming (normalize_text(..., for_embedding=False)).
    clf._domain_terms = {
        "data_protection": {"data": 0.9, "data fiduciary": 1.5, "consent": 0.7},
        "taxation": {"tax": 0.8, "servic tax": 1.4, "input tax": 1.2},
        "employment": {"employee": 0.9, "harass": 1.1},
    }
    clf._loaded = True
    return clf


def test_matches_unigram_term(classifier):
    result = classifier.classify("what consent is needed")
    assert result["domain"] == "data_protection"
    assert "consent" in result["matched_terms"]
    assert result["score"] == pytest.approx(0.7)


def test_matches_bigram_only_when_tokens_are_adjacent(classifier):
    adjacent = classifier.classify("the data fiduciary must comply")
    assert adjacent["domain"] == "data_protection"
    assert adjacent["score"] == pytest.approx(0.9 + 1.5)  # "data" + "data fiduciary"

    non_adjacent = classifier.classify("fiduciary duties and separately some data")
    assert "data fiduciari" not in non_adjacent["matched_terms"]


def test_no_substring_false_positive(classifier):
    # "tax" must not match inside "taxi"
    result = classifier.classify("i took a taxi to the office")
    assert result["domain"] == ""
    assert result["score"] == 0.0


def test_picks_highest_scoring_domain(classifier):
    result = classifier.classify("employee data and consent for tax records")
    # data_protection: data(0.9)+consent(0.7)=1.6 ; taxation: tax(0.8) ; employment: employee(0.9)
    assert result["domain"] == "data_protection"
    assert set(result["scores"]) == {"data_protection", "taxation", "employment"}


def test_threshold_suppresses_weak_match(classifier):
    # only "tax" (0.8) matches; a threshold above it yields no domain
    assert classifier.classify("annual tax filing", threshold=1.0)["domain"] == ""
    assert classifier.classify("annual tax filing", threshold=0.5)["domain"] == "taxation"


def test_matched_terms_capped_at_five_and_score_sorted(classifier):
    classifier._domain_terms["data_protection"] = {f"term{i}": float(i) for i in range(10)}
    query = " ".join(f"term{i}" for i in range(10))
    result = classifier.classify(query)
    assert result["matched_terms"] == ["term9", "term8", "term7", "term6", "term5"]
