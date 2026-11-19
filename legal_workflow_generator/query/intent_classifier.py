import logging
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


class IntentClassifier:
    def __init__(self):
        self.llm = structured_llm(IntentClassification)
        logger.info("IntentClassifier initialized")

    def classify(self, normalized_query: NormalizedQuery) -> tuple[QueryIntent, float]:
        query_text = normalized_query["normalized"]
        prompt = self._build_prompt(query_text)

        system = (
            "You are an intent classifier for a legal compliance system for Indian tech startups. "
            "Your job is to classify user queries into exactly one intent category."
        )
        try:
            result = invoke_structured(self.llm, system, prompt)
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return QueryIntent.UNKNOWN, 0.0
        return self._to_intent(result)

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

    def _to_intent(self, result: IntentClassification) -> tuple[QueryIntent, float]:
        intent = QueryIntent(result.intent)
        confidence = result.confidence
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(f"Low confidence ({confidence}) for intent {intent}, overriding to UNKNOWN")
            return QueryIntent.UNKNOWN, confidence
        return intent, confidence