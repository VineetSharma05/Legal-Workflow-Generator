import logging
from groq import Groq
from legal_workflow_generator.config.values import GROQ_API_KEY, GROQ_MODEL
from legal_workflow_generator.typings.types import NormalizedQuery, QueryIntent

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# INTENT DESCRIPTIONS
# These are what we send to Groq to help it
# understand what each intent means in our
# legal compliance context
# ─────────────────────────────────────────────
INTENT_DESCRIPTIONS = {
    QueryIntent.QA: "a general legal question seeking factual information or explanation about a law, act, or regulation",
    QueryIntent.WORKFLOW: "a request for step-by-step process, procedure, or actionable compliance workflow",
    QueryIntent.COMPLIANCE_CHECK: "a request to check, verify or assess whether something is compliant with a law or regulation",
    QueryIntent.UNKNOWN: "unclear, unrelated, or does not fit any legal compliance category",
}

# ─────────────────────────────────────────────
# CONFIDENCE THRESHOLD
# If Groq's confidence is below this value,
# we override the intent to UNKNOWN
# This prevents low confidence wrong predictions
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5


class IntentClassifier:
    """
    Classifies user query intent into one of 4 categories:
    QA, WORKFLOW, COMPLIANCE_CHECK, UNKNOWN

    Why a class?
    Same reason as normalizer — Groq client is initialized
    once and reused across calls. Efficient and clean.
    """

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        logger.info("IntentClassifier initialized")

    def classify(self, normalized_query: NormalizedQuery) -> tuple[QueryIntent, float]:
        """
        Classify the intent of a normalized query.

        Args:
            normalized_query: Output from QueryNormalizer

        Returns:
            tuple of (QueryIntent, confidence_score)

        Why return a tuple?
        The confidence score is important — it tells the
        context resolver how sure we are about the intent.
        Low confidence → treat as UNKNOWN
        """

        query_text = normalized_query["normalized"]

        # Build the prompt for Groq
        prompt = self._build_prompt(query_text)

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an intent classifier for a legal compliance system "
                            "for Indian tech startups. Your job is to classify user queries "
                            "into exactly one intent category. "
                            "Respond in this exact format and nothing else:\n"
                            "INTENT: <intent>\n"
                            "CONFIDENCE: <score between 0.0 and 1.0>\n"
                            "REASON: <one line explanation>"
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                # Low temperature = more deterministic output
                # We want consistent classification, not creative responses
                temperature=0.1,
                max_tokens=100,
            )

            return self._parse_response(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            # If API fails, return UNKNOWN with 0 confidence
            # Never crash the pipeline because of classification failure
            return QueryIntent.UNKNOWN, 0.0

    def _build_prompt(self, query: str) -> str:
        """
        Build a clear prompt for Groq with all 4 intent
        options and their descriptions.

        Why detailed descriptions?
        The more context we give the model about what each
        intent means, the more accurate the classification.
        """
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
        """
        Parse Groq's response and extract intent + confidence.

        Why careful parsing?
        LLMs can sometimes format responses slightly differently.
        We handle edge cases gracefully instead of crashing.
        """
        try:
            lines = response_text.strip().split("\n")

            # Extract intent
            intent_line = next(l for l in lines if l.startswith("INTENT:"))
            intent_str = intent_line.split(":")[1].strip().lower()

            # Extract confidence
            confidence_line = next(l for l in lines if l.startswith("CONFIDENCE:"))
            confidence = float(confidence_line.split(":")[1].strip())

            # Map string to QueryIntent enum
            intent_map = {
                "qa": QueryIntent.QA,
                "workflow": QueryIntent.WORKFLOW,
                "compliance_check": QueryIntent.COMPLIANCE_CHECK,
                "unknown": QueryIntent.UNKNOWN,
            }

            intent = intent_map.get(intent_str, QueryIntent.UNKNOWN)

            # If confidence is too low, override to UNKNOWN
            # We'd rather say "I don't know" than give a wrong intent
            if confidence < CONFIDENCE_THRESHOLD:
                logger.warning(f"Low confidence ({confidence}) for intent {intent}, overriding to UNKNOWN")
                return QueryIntent.UNKNOWN, confidence

            return intent, confidence

        except Exception as e:
            logger.error(f"Failed to parse Groq response: {e}\nResponse was: {response_text}")
            return QueryIntent.UNKNOWN, 0.0