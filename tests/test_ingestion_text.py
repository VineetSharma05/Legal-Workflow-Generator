"""Unit tests for the pure text-normalisation helpers in ``rag.ingestion``.

These back the BM25 index and the keyword domain classifier, so their
tokenisation behaviour is worth pinning down. No DB is touched.
"""

from legal_workflow_generator.rag.ingestion import (
    expand_query,
    normalize_text,
    remove_stopwords,
    simple_stem,
    stem_text,
    strip_section_markers,
)


def test_simple_stem_strips_common_suffixes():
    assert simple_stem("filing") == "fil"
    assert simple_stem("returns") == "return"
    assert simple_stem("registration") == "registr"


def test_simple_stem_leaves_stem_of_at_least_three_chars():
    # "ies" would leave only "t", so the shorter "s" rule applies instead
    assert simple_stem("ties") == "tie"
    # nothing to strip
    assert simple_stem("act") == "act"


def test_stem_text_applies_per_word():
    assert stem_text("filing returns") == "fil return"


def test_strip_section_markers_removes_citations_and_numbering():
    out = strip_section_markers("(1) the company shall [cite: 12] file under section 4")
    assert "[cite:" not in out
    assert "(1)" not in out
    assert "section 4" not in out
    assert "section" in out


def test_strip_section_markers_masks_rupee_amounts():
    assert "penalty amount" in strip_section_markers("a fine of rs. 5,00,000")


def test_remove_stopwords_keeps_legal_terms():
    out = remove_stopwords("what is the penalty for the company")
    assert "the" not in out.split()
    assert "penalty" in out.split()
    assert "company" in out.split()


def test_expand_query_adds_acronym_expansions():
    out = expand_query("gst filing")
    assert "gst" in out
    assert "goods and services tax" in out


def test_normalize_text_for_embedding_keeps_full_words():
    out = normalize_text("Filing GST Returns", for_embedding=True)
    assert "filing" in out
    assert "goods and services tax" in out


def test_normalize_text_for_bm25_stems_and_drops_stopwords():
    out = normalize_text("What is the process for filing GST returns", for_embedding=False)
    assert "the" not in out.split()
    assert "fil" in out.split()
    assert "return" in out.split()


def test_normalize_text_is_deterministic():
    q = "How do I register a private limited company?"
    assert normalize_text(q, for_embedding=False) == normalize_text(q, for_embedding=False)
