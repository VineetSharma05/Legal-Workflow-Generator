from legal_workflow_generator.query.normalizer import QueryNormalizer
from legal_workflow_generator.query.intent_classifier import IntentClassifier
from legal_workflow_generator.query.context_resolver import LegalContextResolver
from legal_workflow_generator.typings.types import LegalContext


def process_query(
    text: str | None = None,
    pdf_path: str | None = None,
) -> LegalContext:
    """
    Full query processing pipeline.
    This is the single entry point for the query unit.

    Args:
        text: Plain text query from user
        pdf_path: Path to uploaded PDF

    Returns:
        LegalContext ready for the retrieval unit
    """
    normalizer = QueryNormalizer()
    classifier = IntentClassifier()
    resolver = LegalContextResolver()

    # Step 1 — Normalize
    normalized = normalizer.normalize(text=text, pdf_path=pdf_path)

    # Step 2 — Classify intent
    intent, confidence = classifier.classify(normalized)

    # Step 3 — Resolve context
    context = resolver.resolve(normalized, intent, confidence)

    return context