import re
import pdfplumber
import logging
from pathlib import Path
from spellchecker import SpellChecker
from legal_workflow_generator.typings.types import NormalizedQuery

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LEGAL ABBREVIATION EXPANSIONS
# These are common abbreviations used in Indian
# startup legal context. Expanding them helps
# the intent classifier and context resolver
# understand the query better.
# ─────────────────────────────────────────────
LEGAL_ABBREVIATIONS: dict[str, str] = {
    "dpdp": "digital personal data protection",
    "gdpr": "general data protection regulation",
    "ip": "intellectual property",
    "mca": "ministry of corporate affairs",
    "gst": "goods and services tax",
    "tds": "tax deducted at source",
    "fema": "foreign exchange management act",
    "sebi": "securities and exchange board of india",
    "esop": "employee stock option plan",
    "nda": "non disclosure agreement",
    "mou": "memorandum of understanding",
    "llp": "limited liability partnership",
    "cin": "corporate identification number",
    "pan": "permanent account number",
    "tan": "tax deduction account number",
    "roc": "registrar of companies",
    "msme": "micro small and medium enterprises",
    "pf": "provident fund",
    "esi": "employee state insurance",
    "pt": "professional tax",
}

# ─────────────────────────────────────────────
# LEGAL TERMS WHITELIST
# These terms should NEVER be spell corrected
# They are valid legal/technical terms that a
# generic spell checker would wrongly flag
# ─────────────────────────────────────────────
LEGAL_TERMS_WHITELIST: set[str] = {
    "dpdp", "gdpr", "sebi", "fema", "esop", "nda", "mou",
    "llp", "cin", "pan", "tan", "roc", "msme", "gst", "tds",
    "pgvector", "mca21", "startups", "startup", "compliance",
    "fintech", "saas", "b2b", "b2c", "api", "kyc", "aml",
    "nbfc", "rbi", "irdai", "trai", "meity", "cert",
}


class QueryNormalizer:
    """
    Handles all input types (plain text, PDF, or both)
    and returns a clean NormalizedQuery object.

    Why a class and not just functions?
    Because we initialize the spell checker once
    and reuse it — loading it fresh every call
    would be slow.
    """

    def __init__(self):
        # Initialize spell checker once
        self.spell = SpellChecker()
        # Add all legal terms to spell checker's
        # known words so they are never corrected
        self.spell.word_frequency.load_words(LEGAL_TERMS_WHITELIST)
        logger.info("QueryNormalizer initialized")

    # ─────────────────────────────────────────
    # PUBLIC METHOD — this is what you call
    # from outside this module
    # ─────────────────────────────────────────
    def normalize(
        self,
        text: str | None = None,
        pdf_path: str | Path | None = None,
    ) -> NormalizedQuery:
        """
        Main entry point for the normalizer.

        Args:
            text: Plain text query from the user (optional)
            pdf_path: Path to uploaded PDF file (optional)

        Returns:
            NormalizedQuery with original and normalized text

        At least one of text or pdf_path must be provided.
        """

        if not text and not pdf_path:
            raise ValueError("At least one of text or pdf_path must be provided")

        # Step 1 — Extract text from PDF if provided
        pdf_text = ""
        if pdf_path:
            pdf_text = self._extract_pdf_text(pdf_path)
            logger.info(f"Extracted {len(pdf_text)} characters from PDF")

        # Step 2 — Combine PDF and text smartly
        # Text query = user's intent (what they want to know)
        # PDF content = background context (what they have)
        # We put PDF content first so the model sees context
        # before the question — like giving someone a document
        # to read before asking them a question about it
        combined = self._combine_inputs(text, pdf_text)

        # Step 3 — Normalize the combined text
        normalized = self._normalize_text(combined)

        return NormalizedQuery(
            original=combined,       # preserve exactly what user gave
            normalized=normalized,   # cleaned version for downstream use
            source=self._detect_source(text, pdf_path),
        )

    # ─────────────────────────────────────────
    # PRIVATE METHODS — internal helpers
    # ─────────────────────────────────────────

    def _extract_pdf_text(self, pdf_path: str | Path) -> str:
        """
        Extract text from PDF using pdfplumber.
        pdfplumber is better than pypdf for legal docs
        because it handles tables and structured content.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

        extracted_pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text.strip())
                else:
                    logger.warning(f"Page {i+1} had no extractable text (might be scanned image)")

        if not extracted_pages:
            raise ValueError("Could not extract any text from the PDF. It may be a scanned image.")

        return "\n\n".join(extracted_pages)

    def _combine_inputs(self, text: str | None, pdf_text: str) -> str:
        """
        Combine user text query and PDF content into one string.

        Strategy:
        - PDF only → use PDF text as the full input
        - Text only → use text as the full input
        - Both → PDF content first (context), then user question
          This mirrors how you'd give someone a doc to read
          before asking them a question about it
        """
        if text and pdf_text:
            return f"{pdf_text}\n\nUser Question: {text}"
        elif pdf_text:
            return pdf_text
        else:
            return text or ""

    def _normalize_text(self, text: str) -> str:
        """
        Clean and normalize text through these steps:
        1. Lowercase
        2. Remove special characters (keep letters, numbers, spaces)
        3. Expand legal abbreviations
        4. Normalize whitespace
        5. Spell correct (respecting legal whitelist)
        """

        # Step 1 — Lowercase everything
        text = text.lower()

        # Step 2 — Remove special characters
        # Keep letters, digits, spaces — remove punctuation/symbols
        # We keep spaces between words of course
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Step 3 — Expand abbreviations
        # We do this AFTER lowercasing so matching is consistent
        text = self._expand_abbreviations(text)

        # Step 4 — Normalize whitespace
        # Replace multiple spaces/newlines with a single space
        text = re.sub(r"\s+", " ", text).strip()

        # Step 5 — Spell correct
        text = self._spell_correct(text)

        return text

    def _expand_abbreviations(self, text: str) -> str:
        """
        Replace known legal abbreviations with their full forms.
        We use word boundaries (\b) to avoid partial replacements.
        e.g. "gst" → "goods and services tax"
        but "gstr" should NOT be changed to "goods and services taxr"
        """
        for abbr, expansion in LEGAL_ABBREVIATIONS.items():
            text = re.sub(rf"\b{abbr}\b", expansion, text)
        return text

    def _spell_correct(self, text: str) -> str:
        """
        Correct spelling mistakes while respecting the legal
        terms whitelist. Words in the whitelist are never changed.
        """
        words = text.split()
        corrected = []

        for word in words:
            # Never correct whitelisted legal terms
            if word in LEGAL_TERMS_WHITELIST:
                corrected.append(word)
            else:
                # SpellChecker returns the best correction
                # If word is already correct it returns the same word
                correction = self.spell.correction(word)
                corrected.append(correction if correction else word)

        return " ".join(corrected)

    def _detect_source(
        self,
        text: str | None,
        pdf_path: str | Path | None
    ) -> str:
        """
        Detect what type of input was provided.
        This is stored in NormalizedQuery.source and used
        by the context resolver to weigh inputs differently.
        """
        if text and pdf_path:
            return "mixed"
        elif pdf_path:
            return "pdf"
        else:
            return "text"