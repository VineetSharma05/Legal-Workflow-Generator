import logging
from google import genai
from legal_workflow_generator.config.values import GEMINI_API_KEY, GEMINI_MODEL
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
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("IntentClassifier initialized")

    def classify(self, normalized_query: NormalizedQuery) -> tuple[QueryIntent, float]:
        query_text = normalized_query["normalized"]
        prompt = self._build_prompt(query_text)

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"""You are an intent classifier for a legal compliance system for Indian tech startups. Your job is to classify user queries into exactly one intent category.
Respond in this exact format and nothing else:
INTENT: <intent>
CONFIDENCE: <score between 0.0 and 1.0>
REASON: <one line explanation>

{prompt}""",
            )
            return self._parse_response(response.text)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return QueryIntent.UNKNOWN, 0.0

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
            lines = response_text.strip().split("\n")
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
            logger.error(f"Failed to parse Gemini response: {e}\nResponse was: {response_text}")
            return QueryIntent.UNKNOWN, 0.0