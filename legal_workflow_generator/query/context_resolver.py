import logging
from collections import Counter
from google import genai
from google.genai import types as genai_types
from legal_workflow_generator.config.values import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    SELF_CONSISTENCY_ENABLED,
    SELF_CONSISTENCY_SAMPLES,
    DOMAIN_CLASSIFICATION_STRATEGY,
    DOMAIN_KEYWORD_MATCH_THRESHOLD,
    DOMAIN_KEYWORD_STRONG_MATCH_THRESHOLD,
)
from legal_workflow_generator.query.keyword_domain_classifier import KeywordDomainClassifier
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

VALID_DOMAINS = [
    "data_protection", "corporate_governance", "ip_licensing",
    "taxation", "employment",
]


class LegalContextResolver:
    """
    Domain classification runs the deterministic KeywordDomainClassifier
    (TF-IDF terms extracted from the corpus at ingestion — see
    rag/domain_keywords.py) first, then decides whether/how to consult Gemini
    based on `strategy`:

      - "llm_fallback" — Gemini is only called when the keyword classifier
        finds no match at all. Cheapest; no LLM call when keywords resolve it.
      - "combine"       — Gemini is always called too, so the two sources can
        be cross-checked. On agreement the shared answer wins; on
        disagreement, a keyword match strong enough to clear
        DOMAIN_KEYWORD_STRONG_MATCH_THRESHOLD is still trusted over the LLM,
        otherwise the LLM's answer wins (a weak/coincidental keyword hit
        shouldn't override a confident LLM classification).
    """

    def __init__(
        self,
        self_consistency: bool | None = None,
        self_consistency_samples: int | None = None,
        strategy: str | None = None,
        keyword_match_threshold: float | None = None,
        keyword_strong_match_threshold: float | None = None,
    ):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self._keyword_classifier = KeywordDomainClassifier()

        # Config defaults can be overridden per-instance (e.g. for eval
        # scripts or tests) without touching the environment.
        self.self_consistency = (
            SELF_CONSISTENCY_ENABLED if self_consistency is None else self_consistency
        )
        self.self_consistency_samples = (
            SELF_CONSISTENCY_SAMPLES if self_consistency_samples is None else self_consistency_samples
        )
        self.strategy = DOMAIN_CLASSIFICATION_STRATEGY if strategy is None else strategy
        if self.strategy not in ("llm_fallback", "combine"):
            raise ValueError(f"Unknown domain classification strategy: {self.strategy!r}")
        self.keyword_match_threshold = (
            DOMAIN_KEYWORD_MATCH_THRESHOLD if keyword_match_threshold is None else keyword_match_threshold
        )
        self.keyword_strong_match_threshold = (
            DOMAIN_KEYWORD_STRONG_MATCH_THRESHOLD
            if keyword_strong_match_threshold is None
            else keyword_strong_match_threshold
        )

        logger.info(
            f"LegalContextResolver initialized (strategy={self.strategy}, "
            f"self_consistency={self.self_consistency}, samples={self.self_consistency_samples}, "
            f"keyword_match_threshold={self.keyword_match_threshold}, "
            f"keyword_strong_match_threshold={self.keyword_strong_match_threshold})"
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
                keyword_domain="unknown",
                domain_agreement=True,
                domain_confidence=1.0,
                domain_source="skipped",
            )

        match = self._keyword_classifier.classify(query_text, threshold=self.keyword_match_threshold)
        keyword_domain = match["domain"]
        keyword_score = match["score"]
        keyword_matched = bool(keyword_domain)

        llm_domain = ""
        llm_keywords: list[str] = []
        domain_confidence = 1.0

        call_llm = self.strategy == "combine" or not keyword_matched

        if call_llm:
            if self.self_consistency:
                llm_domain, llm_keywords, domain_confidence = self._resolve_with_self_consistency(query_text)
            else:
                llm_domain, llm_keywords = self._resolve_with_gemini(query_text)

        if not keyword_matched:
            # Nothing for the keyword classifier to contribute — LLM decides alone.
            domain = llm_domain or "unknown"
            domain_source = "llm" if llm_domain else "unknown"
            domain_agreement = True  # only one source ran, nothing to disagree with
        elif not call_llm:
            # llm_fallback strategy, keyword classifier already confident enough.
            domain = keyword_domain
            domain_source = "keyword"
            domain_agreement = True
        else:
            # combine strategy: both sources ran, reconcile.
            domain_agreement = llm_domain == keyword_domain
            if domain_agreement:
                domain = keyword_domain
                domain_source = "keyword+llm"
            elif keyword_score >= self.keyword_strong_match_threshold:
                domain = keyword_domain
                domain_source = "keyword_strong_override"
                logger.warning(
                    f"Domain disagreement, kept keyword (strong match {keyword_score:.2f}): "
                    f"keyword={keyword_domain!r} llm={llm_domain!r} query={query_text!r}"
                )
            else:
                domain = llm_domain or keyword_domain
                domain_source = "llm_override" if llm_domain else "keyword"
                logger.warning(
                    f"Domain disagreement, deferred to LLM (weak keyword match {keyword_score:.2f}): "
                    f"keyword={keyword_domain!r} llm={llm_domain!r} query={query_text!r}"
                )

        keywords = match["matched_terms"] or llm_keywords

        return LegalContext(
            original_query=normalized_query["original"],
            normalized_query=query_text,
            intent=intent,
            legal_domain=domain,
            keywords=keywords,
            confidence=confidence,
            keyword_domain=keyword_domain or "unknown",
            domain_agreement=domain_agreement,
            domain_confidence=domain_confidence,
            domain_source=domain_source,
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
            if domain not in VALID_DOMAINS and domain != "unknown":
                logger.warning(f"Invalid domain: {domain}, defaulting to unknown")
                domain = "unknown"
            return domain, keywords
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return "", []