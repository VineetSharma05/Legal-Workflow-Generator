import os
import re
from typing import List
import psycopg2
from psycopg2.extras import execute_batch, Json

import legal_workflow_generator.typings.types as T
import legal_workflow_generator.config.values as config

# ── Stopwords ────────────────────────────────────────────────────────────────
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "that", "this", "these",
    "those", "it", "its", "he", "she", "they", "them", "their", "we",
    "our", "you", "your", "not", "no", "nor", "so", "yet", "both",
    "either", "neither", "such", "if", "as", "than", "then", "when",
    "where", "which", "who", "whom", "any", "all", "each", "every",
    "more", "most", "other", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "off", "over", "under",
    "again", "further", "once", "only", "own", "same", "too", "very",
    "just", "about", "also", "however", "therefore", "thereof", "therein",
    "thereto", "hereby", "herein", "hereof", "herewith", "pursuant",
    "whereas", "wherein", "whereby", "whether", "notwithstanding"
}

# ── Query Expansion Dictionary ───────────────────────────────────────────────
QUERY_EXPANSIONS = {
    # Acronyms → full forms
    "gst":   "goods and services tax gst",
    "igst":  "integrated goods and services tax igst",
    "cgst":  "central goods and services tax cgst",
    "sgst":  "state goods and services tax sgst",
    "itc":   "input tax credit itc",
    "moa":   "memorandum of association moa",
    "aoa":   "articles of association aoa",
    "opc":   "one person company opc",
    "llp":   "limited liability partnership llp",
    "sebi":  "securities and exchange board of india sebi",
    "sez":   "special economic zone sez",
    "ipo":   "initial public offering ipo",
    "icc":   "internal complaints committee icc",
    "posh":  "sexual harassment workplace prevention posh",
    "dpdp":  "digital personal data protection dpdp",
    "dpo":   "data protection officer dpo",
    "dpia":  "data protection impact assessment dpia",
    "oidar": "online information database access retrieval oidar",
    "nda":   "non disclosure agreement nda",
    "ip":    "intellectual property ip",
    "saas":  "software as a service saas",
    "cin":   "corporate identity number cin",
    "vc":    "venture capital vc",
    "bm25":  "bm25",
    # Common legal shorthands
    "sec":   "section",
    "sub":   "sub-section",
    "cl":    "clause",
    "reg":   "regulation",
    "co":    "company",
    "ltd":   "limited",
    "pvt":   "private",
}

# ── Simple Suffix Stemmer ────────────────────────────────────────────────────
def simple_stem(word: str) -> str:
    """
    Rule-based suffix stemming for legal text.
    Strips common suffixes to normalize word forms.
    e.g. incorporated -> incorporat, incorporation -> incorporat
    """
    suffixes = [
        "ational", "tional", "enci", "anci", "izer", "ising", "izing",
        "ation", "ator", "alism", "iveness", "fulness", "ousness",
        "ative", "alize", "icate", "iciti", "ical", "ness", "ment",
        "ings", "ing", "tion", "ions", "ion", "ed", "er", "ly",
        "ies", "ied", "es", "s"
    ]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def stem_text(text: str) -> str:
    """Apply stemming to all words in text."""
    words = text.split()
    return " ".join(simple_stem(w) for w in words)


# ── Section Number Stripping ─────────────────────────────────────────────────
def strip_section_markers(text: str) -> str:
    """
    Remove legal section markers that add noise:
    (1), (a), (i), (ii), [cite: 12], sub-section (2), etc.
    """
    # Remove citation markers like [cite: 12, 34]
    text = re.sub(r'\[cite:[^\]]*\]', ' ', text)
    # Remove patterns like (1), (2), (a), (b), (i), (ii), (iii)
    text = re.sub(r'\(\s*[0-9]+\s*\)', ' ', text)
    text = re.sub(r'\(\s*[a-z]{1,3}\s*\)', ' ', text)
    # Remove patterns like sub-section (1) of section 4
    text = re.sub(r'sub-section\s*\([^)]*\)', 'sub-section', text)
    text = re.sub(r'section\s+\d+[A-Z]?', 'section', text)
    # Remove standalone numbers at start of sentences
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    # Remove Rs. amounts (noise for semantic search)
    text = re.sub(r'rs\.?\s*[\d,]+', 'penalty amount', text)
    text = re.sub(r'rupees\s*[\d,]+', 'penalty amount', text)
    return text


# ── Stopword Removal ─────────────────────────────────────────────────────────
def remove_stopwords(text: str) -> str:
    """Remove stopwords but keep legally significant words."""
    words = text.split()
    return " ".join(w for w in words if w not in STOPWORDS)


# ── Query Expansion ──────────────────────────────────────────────────────────
def expand_query(text: str) -> str:
    """
    Expand acronyms and abbreviations to full forms.
    Keeps original + adds expansion for both BM25 and semantic matching.
    """
    words = text.split()
    expanded = []
    for word in words:
        expanded.append(word)
        if word in QUERY_EXPANSIONS:
            expanded.append(QUERY_EXPANSIONS[word])
    return " ".join(expanded)


# ── Main Normalization Pipeline ──────────────────────────────────────────────
def normalize_text(text: str, for_embedding: bool = True) -> str:
    """
    Full normalization pipeline:
    1. Lowercase
    2. Strip section markers and citations
    3. Remove extra whitespace and newlines
    4. Query expansion (acronyms)
    5. Stopword removal (BM25 only — keep for embeddings)
    6. Stemming (BM25 only — keep full words for embeddings)
    7. Final cleanup
    """
    text = text.lower()
    text = strip_section_markers(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = expand_query(text)

    if not for_embedding:
        # BM25 benefits from stopword removal + stemming
        text = remove_stopwords(text)
        text = stem_text(text)

    text = text.strip()
    return text


# ── Field Combiner ───────────────────────────────────────────────────────────
def combine_fields_for_embedding(law: dict) -> str:
    """
    Combine multiple fields to create rich context for embedding:
    Title + Summary + Keywords + Text
    Order matters — most important fields first.
    """
    parts = []

    if law.get("title"):
        parts.append(law["title"])

    if law.get("plain_english_summary"):
        parts.append(law["plain_english_summary"])

    if law.get("keywords"):
        parts.append(" ".join(law["keywords"]))

    if law.get("text"):
        parts.append(law["text"])

    # Also add chapter title for extra context
    if law.get("chapter_title"):
        parts.append(law["chapter_title"])

    return ". ".join(parts)


# ── Ingestion ────────────────────────────────────────────────────────────────
def ingest(laws: List[T.LawSchema]) -> None:
    if not laws:
        print("No laws provided for ingestion")
        return

    conn = psycopg2.connect(
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=os.environ.get("PGPASSWORD"),
        host=config.DB_HOST,
        port=config.DB_PORT,
    )
    cur = conn.cursor()

    query = """
        INSERT INTO laws (
            provision_id,
            statute_id,
            provision_type,
            chapter,
            chapter_title,
            number,
            title,
            text,
            sub_structure,
            plain_english_summary,
            keywords,
            penalty_linked,
            effective_date
        )
        VALUES (
            %(provision_id)s,
            %(statute_id)s,
            %(provision_type)s,
            %(chapter)s,
            %(chapter_title)s,
            %(number)s,
            %(title)s,
            %(text)s,
            %(sub_structure)s,
            %(plain_english_summary)s,
            %(keywords)s,
            %(penalty_linked)s,
            %(effective_date)s
        )
        ON CONFLICT (provision_id)
        DO UPDATE SET
            statute_id = EXCLUDED.statute_id,
            provision_type = EXCLUDED.provision_type,
            chapter = EXCLUDED.chapter,
            chapter_title = EXCLUDED.chapter_title,
            number = EXCLUDED.number,
            title = EXCLUDED.title,
            text = EXCLUDED.text,
            sub_structure = EXCLUDED.sub_structure,
            plain_english_summary = EXCLUDED.plain_english_summary,
            keywords = EXCLUDED.keywords,
            penalty_linked = EXCLUDED.penalty_linked,
            effective_date = EXCLUDED.effective_date;
    """

    prepared_rows = []

    for law in laws:
        row = law.copy()

        # Combine fields for rich embedding context
        combined_text = combine_fields_for_embedding(law)

        # Normalize for embedding (keep full words, no stemming/stopword removal)
        normalized_text = normalize_text(combined_text, for_embedding=True)

        row["text"] = normalized_text
        row["sub_structure"] = Json(law.get("sub_structure", {}))
        row["keywords"] = law.get("keywords", [])
        row["plain_english_summary"] = law.get("plain_english_summary", "")
        row["penalty_linked"] = law.get("penalty_linked", False)

        prepared_rows.append(row)

    execute_batch(cur, query, prepared_rows, page_size=100)
    conn.commit()

    cur.close()
    conn.close()

    print(f"Ingested {len(laws)} laws successfully")
    print("  Preprocessing applied:")
    print("    [1] Field combination  : title + summary + keywords + text + chapter_title")
    print("    [2] Section stripping  : removed (1), (a), [cite:x], section numbers")
    print("    [3] Query expansion    : acronyms expanded (GST, IGST, POSH, DPDP etc)")
    print("    [4] Normalization      : lowercase, whitespace cleanup")
    print("    [5] Stopword removal   : applied for BM25 index only")
    print("    [6] Stemming           : applied for BM25 index only")