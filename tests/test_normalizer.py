"""Unit tests for :class:`QueryNormalizer` — fully offline (no DB, no API)."""

import pytest

from legal_workflow_generator.query.normalizer import QueryNormalizer


@pytest.fixture
def normalizer() -> QueryNormalizer:
    return QueryNormalizer()


def test_lowercases_and_strips_punctuation(normalizer):
    result = normalizer.normalize(text="How do I comply with the DPDP Act??")
    assert result["normalized"] == "how do i comply with the digital personal data protection act"
    assert result["source"] == "text"
    assert result["original"] == "How do I comply with the DPDP Act??"


def test_expands_known_abbreviations(normalizer):
    result = normalizer.normalize(text="What are GST rules for my startup?")
    assert "goods and services tax" in result["normalized"]
    assert "gst" not in result["normalized"].split()


def test_abbreviation_expansion_respects_word_boundaries(normalizer):
    # "gstr" must not become "goods and services taxr"
    result = normalizer.normalize(text="file the gstr return")
    assert "gstr" in result["normalized"]
    assert "taxr" not in result["normalized"]


def test_requires_at_least_one_input(normalizer):
    with pytest.raises(ValueError, match="At least one of text or pdf_path"):
        normalizer.normalize()


def test_rejects_empty_query_after_normalization(normalizer):
    with pytest.raises(ValueError, match="empty after normalization"):
        normalizer.normalize(text="!@#$%^&*()")


def test_rejects_single_word_query(normalizer):
    with pytest.raises(ValueError, match="too short to be meaningful"):
        normalizer.normalize(text="taxes")


def test_truncates_very_long_queries_to_500_words(normalizer):
    long_query = "compliance step " * 400  # 800 words
    result = normalizer.normalize(text=long_query)
    assert len(result["normalized"].split()) == 500


def test_extracts_text_from_pdf(normalizer, sample_pdf_path):
    result = normalizer.normalize(pdf_path=sample_pdf_path)
    assert result["source"] == "pdf"
    assert "digital personal data protection" in result["normalized"]


def test_combines_pdf_and_text_as_mixed_source(normalizer, sample_pdf_path):
    result = normalizer.normalize(
        text="What are my compliance obligations?", pdf_path=sample_pdf_path
    )
    assert result["source"] == "mixed"
    assert "user question" in result["normalized"]


def test_missing_pdf_raises_file_not_found(normalizer, tmp_path):
    with pytest.raises(FileNotFoundError):
        normalizer.normalize(pdf_path=tmp_path / "does_not_exist.pdf")


def test_non_pdf_path_is_rejected(normalizer, tmp_path):
    bogus = tmp_path / "note.txt"
    bogus.write_text("hello world")
    with pytest.raises(ValueError, match="Expected a .pdf file"):
        normalizer.normalize(pdf_path=bogus)
