"""Integration test for the full RAG pipeline (hybrid retrieval + LLM answer).

Needs a populated Postgres corpus with embeddings and a real Gemini key, so it
is deselected by default. Run with::

    uv run pytest -m integration tests/test_rag_pipeline.py
"""

import pytest

from legal_workflow_generator.rag.pipeline import RagPipeline

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("provider", ["gemini"])
def test_pipeline_retrieves_and_answers(live_backends, provider):
    pipeline = RagPipeline(llm_provider=provider)
    result = pipeline.run(
        query="What are the minimum founder requirements to start a private company in India?",
        top_k=3,
    )

    assert result["answer"]
    assert 1 <= len(result["retrieved"]) <= 3
    for item in result["retrieved"]:
        assert item["provision_id"]
        assert "combined_score" in item
