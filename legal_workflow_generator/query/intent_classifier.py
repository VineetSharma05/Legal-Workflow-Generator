import logging
from google import genai
from legal_workflow_generator.agent import groq_client
from legal_workflow_generator.config.values import GEMINI_MODEL
from legal_workflow_generator.agent.gemini_client import RotatingGeminiClient
from legal_workflow_generator.agent.llm_limits import LIMITERS
from legal_workflow_generator.typings.types import NormalizedQuery, QueryIntent

logger = logging.getLogger(__name__)

INTENT_DESCRIPTIONS = {
    QueryIntent.QA: "a general legal question seeking factual information or explanation about a law, act, or regulation",
    QueryIntent.WORKFLOW: "a request for step-by-step process, procedure, or actionable compliance workflow",
    QueryIntent.COMPLIANCE_CHECK: "a request to check, verify or assess whether something is compliant with a law or regulation",
    QueryIntent.UNKNOWN: "unclear, unrelated, or does not fit any legal compliance category",
}

CONFIDENCE_THRESHOLD = 0.5


class IntentClassifier:
    def __init__(self):
        self.client = RotatingGeminiClient()
        logger.info("IntentClassifier initialized")

    def classify(self, normalized_query: NormalizedQuery, provider: str = "gemini",
                 raise_errors: bool = False) -> tuple[QueryIntent, float]:
        """Classify with the given provider ("gemini" or "groq").

        An API error gives (UNKNOWN, 0.0), or is re-raised when raise_errors is True, so the agent
        can tell "the LLM is down" apart from "the query is unclear".
        """
        query_text = normalized_query["normalized"]
        prompt = self._build_prompt(query_text)
        contents = f"""You are an intent classifier for a legal compliance system for Indian tech startups. Your job is to classify user queries into exactly one intent category.
Respond in this exact format and nothing else:
INTENT: <intent>
CONFIDENCE: <score between 0.0 and 1.0>
REASON: <one line explanation>

{prompt}"""

        try:
            with LIMITERS[provider].slot():
                if provider == "groq":
                    text = groq_client.complete("", contents)
                else:
                    text = self.client.models.generate_content(model=GEMINI_MODEL, contents=contents).text
        except Exception as e:
            logger.error(f"{provider} API error: {e}")
            if raise_errors:
                raise
            return QueryIntent.UNKNOWN, 0.0
        return self._parse_response(text)

    def _build_prompt(self, query: str) -> str:
        intent_options = "\n".join([
            f"- {intent.value.upper()}: {desc}"
            for intent, desc in INTENT_DESCRIPTIONS.items()
        ])
        return (
            f"Classify the following legal query into exactly one intent:\n\n"
            f"Query: {query}\n\n"
            f"Intent options:\n{intent_options}\n\n"
            f"Pick the most appropriate intent."
        )

    def _parse_response(self, response_text: str) -> tuple[QueryIntent, float]:
        try:
            # strip markdown some models add ("**INTENT:** qa")
            lines = [l.replace("**", "").strip() for l in response_text.strip().split("\n")]
            intent_line = next(l for l in lines if l.startswith("INTENT:"))
            intent_str = intent_line.split(":")[1].strip().lower()
            confidence_line = next(l for l in lines if l.startswith("CONFIDENCE:"))
            confidence = float(confidence_line.split(":")[1].strip())
            intent_map = {
                "qa": QueryIntent.QA,
                "workflow": QueryIntent.WORKFLOW,
                "compliance_check": QueryIntent.COMPLIANCE_CHECK,
                "unknown": QueryIntent.UNKNOWN,
            }
            intent = intent_map.get(intent_str, QueryIntent.UNKNOWN)
            if confidence < CONFIDENCE_THRESHOLD:
                logger.warning(f"Low confidence ({confidence}) for intent {intent}, overriding to UNKNOWN")
                return QueryIntent.UNKNOWN, confidence
            return intent, confidence
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}\nResponse was: {response_text}")
            return QueryIntent.UNKNOWN, 0.0