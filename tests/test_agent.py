"""Integration test for the LangGraph agentic pipeline (end to end).

Needs the full stack (corpus + embeddings + Gemini), so it is deselected by
default. Run with::

    uv run pytest -m integration tests/test_agent.py

The graph import is deliberately deferred into the test body:
``legal_workflow_generator.agent.nodes`` builds the BM25 index (live DB
required) at import time, which would otherwise break collection.
"""

import pytest

pytestmark = pytest.mark.integration


def test_agent_answers_a_dpdp_workflow_query(live_backends):
    from legal_workflow_generator.agent.graph import graph

    result = graph.invoke(
        {"query": "What are the steps to comply with DPDP Act as a SaaS startup?"}
    )

    assert "answer" in result
    assert "abstain" in result
    assert isinstance(result["trace"], list) and result["trace"]

    if not result["abstain"]:
        assert result["answer"]
        # every cited section must have passed string-match verification
        assert isinstance(result["verified_citations"], list)
