"""
Deterministic domain classifier over the TF-IDF terms extracted from the law
corpus by rag/domain_keywords.py. No LLM call, no sampling — same query always
gets the same answer, so this is meant to be the cheap first-pass classifier
LegalContextResolver checks before ever calling an LLM.
"""

import logging

import psycopg2

import legal_workflow_generator.config.values as config
from legal_workflow_generator.rag.ingestion import normalize_text

logger = logging.getLogger(__name__)


class KeywordDomainClassifier:
    def __init__(self):
        self._domain_terms: dict[str, dict[str, float]] = {}
        self._loaded = False

    def ensure_index(self) -> None:
        if self._loaded:
            return

        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.PGPASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
        )
        cur = conn.cursor()
        cur.execute("SELECT domain, term, score FROM domain_keywords")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            logger.warning(
                "domain_keywords table is empty — run `python main.py extract-keywords` "
                "after ingestion. KeywordDomainClassifier will find no matches until then."
            )

        for domain, term, score in rows:
            self._domain_terms.setdefault(domain, {})[term] = score

        self._loaded = True
        logger.info(
            f"KeywordDomainClassifier loaded {len(self._domain_terms)} domains, "
            f"{sum(len(t) for t in self._domain_terms.values())} terms"
        )

    def classify(self, query_text: str, threshold: float = 0.0) -> dict:
        """
        Scores every domain by summing the TF-IDF weight of its terms that
        appear in the query (unigram terms matched against query tokens,
        bigram terms matched against adjacent query token pairs — avoids the
        false-positive substring matches a naive `term in query_string` check
        would produce, e.g. "tax" inside "taxi").

        Returns {"domain": str, "score": float, "scores": dict[str, float],
        "matched_terms": list[str]}. domain is "" when no domain's score
        clears `threshold` — that's the fallback trigger for the caller.
        """
        self.ensure_index()

        tokens = normalize_text(query_text, for_embedding=False).split()
        token_set = set(tokens)
        bigrams = {f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)}

        scores: dict[str, float] = {}
        matched_by_domain: dict[str, list[tuple[str, float]]] = {}

        for domain, terms in self._domain_terms.items():
            domain_score = 0.0
            matched = []
            for term, weight in terms.items():
                hit = (term in bigrams) if " " in term else (term in token_set)
                if hit:
                    domain_score += weight
                    matched.append((term, weight))
            scores[domain] = domain_score
            matched_by_domain[domain] = matched

        if not scores or max(scores.values()) <= threshold:
            return {"domain": "", "score": 0.0, "scores": scores, "matched_terms": []}

        best_domain = max(scores, key=scores.get)
        best_terms = sorted(matched_by_domain[best_domain], key=lambda t: -t[1])
        return {
            "domain": best_domain,
            "score": scores[best_domain],
            "scores": scores,
            "matched_terms": [t for t, _ in best_terms[:5]],
        }
