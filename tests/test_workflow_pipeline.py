"""Unit tests for the structured per-domain workflow pipeline."""

from legal_workflow_generator.agent import nodes


def _result(domain: str, cited: list[str] | None = None) -> dict:
    cited = cited or [f"{domain}_sec_1"]
    return {
        "domain": domain,
        "docs": [{"provision_id": provision_id, "title": "A provision"} for provision_id in cited],
        "grade": "sufficient",
        "reason": "directly relevant",
        "retries": 0,
        "summary": [f"Summary for {domain}"],
        "steps": [{
            "action": f"Complete {domain} action",
            "provision_id": cited[0],
            "applies_to": "private",
        }],
        "not_covered": "",
        "cited": cited,
    }


def test_domain_pipeline_keeps_structured_results_and_stitches_domains(monkeypatch):
    class FakeRag:
        def _ensure_index(self):
            pass

    monkeypatch.setattr(nodes, "_get_rag_pipeline", lambda: FakeRag())
    monkeypatch.setattr(nodes, "_load_domain_map", lambda: {})

    def fake_worker(domain, search_q, user_q, rag, multi):
        return _result(domain), [f"{domain} complete"]

    monkeypatch.setattr(nodes, "_domain_worker", fake_worker)
    state = {
        "query": "Prepare corporate and tax compliance",
        "domain": "corporate_governance",
        "all_domains": ["corporate_governance", "taxation"],
        "normalized_query": "prepare corporate and tax compliance",
        "trace": [],
    }

    result = nodes.domain_pipeline(state)

    assert [item["domain"] for item in result["domain_results"]] == [
        "corporate_governance",
        "taxation",
    ]
    assert result["context_grade"] == "sufficient"
    assert result["citations"] == [
        "corporate_governance_sec_1",
        "taxation_sec_1",
    ]
    assert "Corporate Governance" in result["answer"]
    assert "Taxation & GST" in result["answer"]
    assert "parallel" in result["trace"][-1]


def test_stitch_sections_deduplicates_sources_and_preserves_steps():
    results = [
        _result("employment", ["posh_act_2013_sec_4", "posh_act_2013_sec_4"]),
    ]

    answer = nodes._stitch_sections(
        "How do I comply with POSH?",
        results,
    )

    assert "ACTION CHECKLIST" in answer
    assert "Complete employment action" in answer
    assert answer.count("posh_act_2013_sec_4") == 2
    assert "informational guidance, not legal advice" in answer


def test_stitch_answer_reports_domains_without_verified_citations():
    state = {
        "all_domains": ["employment", "taxation"],
        "domain": "employment",
        "answer": "Employment workflow",
        "verified_citations": ["posh_act_2013_sec_4"],
        "trace": [],
    }

    result = nodes.stitch_answer(state)

    assert result["stitched_answer"].startswith("Employment workflow")
    assert result["domain_gaps"] == [
        "- Taxation & GST: insufficient information found in knowledge base"
    ]
    assert "KNOWLEDGE GAPS IDENTIFIED" in result["stitched_answer"]
