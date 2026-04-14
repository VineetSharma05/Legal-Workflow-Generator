import logging
from groq import Groq
from legal_workflow_generator.config.values import GROQ_API_KEY, GROQ_MODEL
from legal_workflow_generator.typings.types import (
    NormalizedQuery,
    QueryIntent,
    LegalContext,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LEGAL DOMAINS
# These are your 5 core domains from the project
# Each domain has keywords associated with it
# for fallback domain detection
# ─────────────────────────────────────────────
LEGAL_DOMAINS = {
    "data_protection": [
        "dpdp", "digital personal data protection", "privacy", "data",
        "personal data", "data breach", "consent", "gdpr", "meity",
        "data fiduciary", "data principal", "data processor",
    ],
    "corporate_governance": [
        "company", "incorporation", "mca", "registrar of companies",
        "roc", "board", "director", "shareholder", "companies act",
        "corporate", "governance", "annual general meeting", "agm",
        "memorandum of association", "articles of association",
    ],
    "ip_licensing": [
        "intellectual property", "patent", "trademark", "copyright",
        "license", "licensing", "ip", "trade secret", "infringement",
        "open source", "software license", "nda", "non disclosure",
    ],
    "taxation": [
        "tax", "gst", "goods and services tax", "income tax", "tds",
        "tax deducted at source", "pan", "tan", "itr", "filing",
        "advance tax", "startup tax", "angel tax", "transfer pricing",
    ],
    "employment": [
        "employee", "employment", "hiring", "salary", "payroll",
        "pf", "provident fund", "esi", "employee state insurance",
        "professional tax", "esop", "labour law", "termination",
        "contract", "offer letter", "gratuity", "leave policy",
    ],
}


class LegalContextResolver:
    """
    Resolves the full legal context from a normalized query and intent.

    Why use Groq here too?
    Keyword extraction and domain detection benefit from
    language understanding — Groq does this much better
    than simple keyword matching alone.

    We use BOTH:
    - Groq for deep understanding
    - Rule-based fallback if Groq fails
    This makes the system robust.
    """

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        logger.info("LegalContextResolver initialized")

    def resolve(
        self,
        normalized_query: NormalizedQuery,
        intent: QueryIntent,
        confidence: float,
    ) -> LegalContext:
        """
        Main entry point for context resolution.

        Args:
            normalized_query: Output from QueryNormalizer
            intent: Classified intent from IntentClassifier
            confidence: Confidence score from IntentClassifier

        Returns:
            LegalContext with all info needed for retrieval
        """

        query_text = normalized_query["normalized"]

        # If intent is UNKNOWN with high confidence,
        # no point resolving context — return early
        if intent == QueryIntent.UNKNOWN and confidence >= 0.8:
            logger.info("Intent is UNKNOWN with high confidence, skipping context resolution")
            return LegalContext(
                original_query=normalized_query["original"],
                normalized_query=query_text,
                intent=intent,
                legal_domain="unknown",
                keywords=[],
                confidence=confidence,
            )

        # Use Groq to extract domain and keywords
        domain, keywords = self._resolve_with_groq(query_text)

        # If Groq fails, fall back to rule-based detection
        if not domain:
            logger.warning("Groq resolution failed, falling back to rule-based")
            domain = self._detect_domain_rule_based(query_text)

        if not keywords:
            keywords = self._extract_keywords_rule_based(query_text)

        return LegalContext(
            original_query=normalized_query["original"],
            normalized_query=query_text,
            intent=intent,
            legal_domain=domain,
            keywords=keywords,
            confidence=confidence,
        )

    # ─────────────────────────────────────────
    # GROQ BASED RESOLUTION
    # ─────────────────────────────────────────

    def _resolve_with_groq(self, query: str) -> tuple[str, list[str]]:
        """
        Use Groq to detect legal domain and extract keywords.

        Why Groq for this?
        Simple keyword matching can miss context.
        Example: "can I fire someone without notice?"
        has no direct employment keywords but is clearly
        about employment law. Groq understands this.
        """
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a legal domain classifier for Indian startup compliance. "
                            "Given a query, identify:\n"
                            "1. The legal domain (exactly one of: data_protection, corporate_governance, ip_licensing, taxation, employment, unknown)\n"
                            "2. Up to 5 important legal keywords from the query\n\n"
                            "Respond in this exact format and nothing else:\n"
                            "DOMAIN: <domain>\n"
                            "KEYWORDS: <keyword1>, <keyword2>, <keyword3>"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}"
                    }
                ],
                temperature=0.1,
                max_tokens=100,
            )

            return self._parse_groq_response(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Groq resolution error: {e}")
            return "", []

    def _parse_groq_response(self, response_text: str) -> tuple[str, list[str]]:
        """
        Parse Groq's domain and keywords response.
        """
        try:
            lines = response_text.strip().split("\n")

            domain_line = next(l for l in lines if l.startswith("DOMAIN:"))
            domain = domain_line.split(":")[1].strip().lower()

            keywords_line = next(l for l in lines if l.startswith("KEYWORDS:"))
            keywords = [k.strip() for k in keywords_line.split(":")[1].split(",")]

            # Validate domain is one of our 5
            valid_domains = list(LEGAL_DOMAINS.keys()) + ["unknown"]
            if domain not in valid_domains:
                logger.warning(f"Invalid domain from Groq: {domain}, defaulting to unknown")
                domain = "unknown"

            return domain, keywords

        except Exception as e:
            logger.error(f"Failed to parse Groq response: {e}")
            return "", []

    # ─────────────────────────────────────────
    # RULE BASED FALLBACK
    # Used when Groq is unavailable
    # ─────────────────────────────────────────

    def _detect_domain_rule_based(self, query: str) -> str:
        """
        Detect legal domain by counting keyword matches.
        The domain with the most matches wins.
        If no matches, return unknown.
        """
        scores = {domain: 0 for domain in LEGAL_DOMAINS}

        for domain, keywords in LEGAL_DOMAINS.items():
            for keyword in keywords:
                if keyword in query:
                    scores[domain] += 1

        best_domain = max(scores, key=scores.get)

        # Only return a domain if at least 1 keyword matched
        if scores[best_domain] == 0:
            return "unknown"

        return best_domain

    def _extract_keywords_rule_based(self, query: str) -> list[str]:
        """
        Extract keywords by checking which domain keywords
        appear in the query.
        Returns up to 5 matched keywords.
        """
        found = []
        for keywords in LEGAL_DOMAINS.values():
            for keyword in keywords:
                if keyword in query and keyword not in found:
                    found.append(keyword)

        return found[:5]