import logging
from collections import Counter
from google import genai
from google.genai import types as genai_types
from legal_workflow_generator.config.values import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    SELF_CONSISTENCY_ENABLED,
    SELF_CONSISTENCY_SAMPLES,
)
from legal_workflow_generator.typings.types import (
    NormalizedQuery,
    QueryIntent,
    LegalContext,
)

logger = logging.getLogger(__name__)

# Sampling temperature used only for self-consistency votes — the single-shot
# call keeps the API default (near-deterministic) so behavior is unchanged
# when self-consistency is off.
SELF_CONSISTENCY_TEMPERATURE = 0.7

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
    def __init__(
        self,
        self_consistency: bool | None = None,
        self_consistency_samples: int | None = None,
    ):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        # Config default can be overridden per-instance (e.g. for eval scripts
        # or tests) without touching the environment.
        self.self_consistency = (
            SELF_CONSISTENCY_ENABLED if self_consistency is None else self_consistency
        )
        self.self_consistency_samples = (
            SELF_CONSISTENCY_SAMPLES if self_consistency_samples is None else self_consistency_samples
        )
        logger.info(
            f"LegalContextResolver initialized (self_consistency={self.self_consistency}, "
            f"samples={self.self_consistency_samples})"
        )

    def resolve(
        self,
        normalized_query: NormalizedQuery,
        intent: QueryIntent,
        confidence: float,
    ) -> LegalContext:
        query_text = normalized_query["normalized"]

        if intent == QueryIntent.UNKNOWN and confidence >= 0.8:
            logger.info("Intent is UNKNOWN with high confidence, skipping context resolution")
            return LegalContext(
                original_query=normalized_query["original"],
                normalized_query=query_text,
                intent=intent,
                legal_domain="unknown",
                keywords=[],
                confidence=confidence,
                rule_based_domain="unknown",
                domain_agreement=True,
                domain_confidence=1.0,
            )

        # Always computed — costs nothing (no API call) and is the baseline
        # the LLM's domain choice is checked against below.
        rule_based_domain = self._detect_domain_rule_based(query_text)

        if self.self_consistency:
            domain, keywords, domain_confidence = self._resolve_with_self_consistency(query_text)
        else:
            domain, keywords = self._resolve_with_gemini(query_text)
            domain_confidence = 1.0  # no sampling signal available

        if not domain:
            logger.warning("Gemini resolution failed, falling back to rule-based")
            domain = rule_based_domain
            domain_confidence = 1.0  # rule-based is deterministic

        if not keywords:
            keywords = self._extract_keywords_rule_based(query_text)

        domain_agreement = domain == rule_based_domain
        if not domain_agreement:
            logger.warning(
                f"Domain disagreement: llm={domain!r} rule_based={rule_based_domain!r} "
                f"query={query_text!r}"
            )

        return LegalContext(
            original_query=normalized_query["original"],
            normalized_query=query_text,
            intent=intent,
            legal_domain=domain,
            keywords=keywords,
            confidence=confidence,
            rule_based_domain=rule_based_domain,
            domain_agreement=domain_agreement,
            domain_confidence=domain_confidence,
        )

    def _resolve_with_self_consistency(self, query: str) -> tuple[str, list[str], float]:
        """
        Sample the domain classification prompt N times at temperature>0 and
        majority-vote the domain. The winning share (e.g. 2/3) is returned as
        domain_confidence — low agreement flags queries the LLM itself isn't
        stable on, which plain single-shot confidence can't surface.
        """
        domains = []
        keywords_by_domain: dict[str, list[str]] = {}

        for _ in range(self.self_consistency_samples):
            domain, keywords = self._resolve_with_gemini(query, temperature=SELF_CONSISTENCY_TEMPERATURE)
            if not domain:
                continue
            domains.append(domain)
            keywords_by_domain.setdefault(domain, keywords)

        if not domains:
            return "", [], 0.0

        vote_counts = Counter(domains)
        majority_domain, majority_votes = vote_counts.most_common(1)[0]
        agreement_ratio = majority_votes / len(domains)

        if agreement_ratio < 1.0:
            logger.warning(
                f"Self-consistency disagreement for query={query!r}: "
                f"votes={dict(vote_counts)} agreement={agreement_ratio:.2f}"
            )

        return majority_domain, keywords_by_domain.get(majority_domain, []), agreement_ratio

    def _resolve_with_gemini(self, query: str, temperature: float | None = None) -> tuple[str, list[str]]:
        try:
            kwargs = {}
            if temperature is not None:
                kwargs["config"] = genai_types.GenerateContentConfig(temperature=temperature)

            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"""You are a legal domain classifier for Indian startup compliance. Given a query, identify:
1. The legal domain (exactly one of: data_protection, corporate_governance, ip_licensing, taxation, employment, unknown)
2. Up to 5 important legal keywords from the query

Respond in this exact format and nothing else:
DOMAIN: <domain>
KEYWORDS: <keyword1>, <keyword2>, <keyword3>

Query: {query}""",
                **kwargs,
            )
            return self._parse_response(response.text)
        except Exception as e:
            logger.error(f"Gemini resolution error: {e}")
            return "", []

    def _parse_response(self, response_text: str) -> tuple[str, list[str]]:
        try:
            lines = response_text.strip().split("\n")
            domain_line = next(l for l in lines if l.startswith("DOMAIN:"))
            domain = domain_line.split(":")[1].strip().lower()
            keywords_line = next(l for l in lines if l.startswith("KEYWORDS:"))
            keywords = [k.strip() for k in keywords_line.split(":")[1].split(",")]
            valid_domains = list(LEGAL_DOMAINS.keys()) + ["unknown"]
            if domain not in valid_domains:
                logger.warning(f"Invalid domain: {domain}, defaulting to unknown")
                domain = "unknown"
            return domain, keywords
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return "", []

    def _detect_domain_rule_based(self, query: str) -> str:
        scores = {domain: 0 for domain in LEGAL_DOMAINS}
        for domain, keywords in LEGAL_DOMAINS.items():
            for keyword in keywords:
                if keyword in query:
                    scores[domain] += 1
        best_domain = max(scores, key=scores.get)
        if scores[best_domain] == 0:
            return "unknown"
        return best_domain

    def _extract_keywords_rule_based(self, query: str) -> list[str]:
        found = []
        for keywords in LEGAL_DOMAINS.values():
            for keyword in keywords:
                if keyword in query and keyword not in found:
                    found.append(keyword)
        return found[:5]