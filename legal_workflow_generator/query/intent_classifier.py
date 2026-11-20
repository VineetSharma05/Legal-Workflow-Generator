import logging
import re

from legal_workflow_generator.agent import groq_client
from legal_workflow_generator.agent import llm_limits
from legal_workflow_generator.config.values import GEMINI_MODEL
from legal_workflow_generator.llm import invoke_structured, structured_llm
from legal_workflow_generator.llm.schemas import IntentClassification
from legal_workflow_generator.typings.types import NormalizedQuery, QueryIntent

logger = logging.getLogger(__name__)

INTENT_DESCRIPTIONS = {
    QueryIntent.QA: "a general legal question seeking factual information or explanation about a law, act, or regulation",
    QueryIntent.WORKFLOW: "a request for step-by-step process, procedure, or actionable compliance workflow",
    QueryIntent.COMPLIANCE_CHECK: "a request to check, verify or assess whether something is compliant with a law or regulation",
    QueryIntent.UNKNOWN: "unclear, unrelated, or does not fit any legal compliance category",
}

CONFIDENCE_THRESHOLD = 0.5

SYSTEM = (
    "You are an intent classifier for a legal compliance system for Indian tech startups. "
    "Your job is to classify user queries into exactly one intent category."
)
LABEL_FORMAT = """Respond in this exact format and nothing else:
INTENT: <intent>
CONFIDENCE: <score between 0.0 and 1.0>
REASON: <one line explanation>"""


class IntentClassifier:
    def __init__(self):
        self.llm = structured_llm(IntentClassification)
        logger.info("IntentClassifier initialized")

    def classify(self, normalized_query: NormalizedQuery, provider: str = "gemini",
                 raise_errors: bool = False) -> tuple[QueryIntent, float]:
        """Classify with the run's provider ("gemini" or "groq").

        gemini: structured output (`self.llm`); an instance without it (built via __new__ with only a raw
                `client`) falls back to a plain-text call parsed from labelled lines.
        groq:   plain-text call through groq_client, parsed from labelled lines (markdown tolerated).
        An API error gives (UNKNOWN, 0.0), or is re-raised when raise_errors is True, so the agent can tell
        "the LLM is down" apart from "the query is unclear". Every call holds a slot of the shared limiter.
        """
        prompt = self._build_prompt(normalized_query["normalized"])
        try:
            # looked up at call time so the shared limiter can be replaced (tests, runtime rate changes)
            with llm_limits.LIMITERS[provider].slot():
                if provider == "groq":
                    return self._parse_response(groq_client.complete("", f"{SYSTEM}\n{LABEL_FORMAT}\n\n{prompt}"))
                if getattr(self, "llm", None) is not None:
                    result = invoke_structured(self.llm, SYSTEM, prompt)
                    return self._to_intent(QueryIntent(result.intent), result.confidence)
                text = self.client.models.generate_content(
                    model=GEMINI_MODEL, contents=f"{SYSTEM}\n{LABEL_FORMAT}\n\n{prompt}").text
                return self._parse_response(text)
        except Exception as e:
            logger.error(f"Intent classification failed ({provider}): {e}")
            if raise_errors:
                raise
            return QueryIntent.UNKNOWN, 0.0

    def _build_prompt(self, query: str) -> str:
        intent_options = "\n".join([
            f"- {intent.value}: {desc}"
            for intent, desc in INTENT_DESCRIPTIONS.items()
        ])
        return (
            f"Classify the following legal query into exactly one intent:\n\n"
            f"Query: {query}\n\n"
            f"Intent options:\n{intent_options}\n\n"
            f"Pick the most appropriate intent."
        )

    def _parse_response(self, text: str) -> tuple[QueryIntent, float]:
        """Parse 'INTENT: x / CONFIDENCE: y' lines; tolerates markdown such as '**INTENT:** x'."""
        clean = (text or "").replace("**", "")
        m_int = re.search(r"^\s*INTENT\s*:\s*([a-z_]+)", clean, re.I | re.M)
        m_conf = re.search(r"^\s*CONFIDENCE\s*:\s*([\d.]+)", clean, re.I | re.M)
        if not m_int or not m_conf:
            logger.error(f"Failed to parse intent response: {text!r}")
            return QueryIntent.UNKNOWN, 0.0
        try:
            intent = QueryIntent(m_int.group(1).lower())
        except ValueError:
            intent = QueryIntent.UNKNOWN
        return self._to_intent(intent, float(m_conf.group(1)))

    def _to_intent(self, intent: QueryIntent, confidence: float) -> tuple[QueryIntent, float]:
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(f"Low confidence ({confidence}) for intent {intent}, overriding to UNKNOWN")
            return QueryIntent.UNKNOWN, confidence
        return intent, confidence