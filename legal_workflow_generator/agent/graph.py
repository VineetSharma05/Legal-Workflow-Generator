from langgraph.graph import StateGraph, END
from legal_workflow_generator.agent.settings import build_config
from legal_workflow_generator.agent.state import AgentState
from legal_workflow_generator.agent.nodes import (
    classify_query,
    domain_pipeline,
    verify_citations,
    grade_parallel,
    stitch_answer,
    abstain,
)

# How many times the whole per-domain pipeline may be re-run after a failed grade.
# Each re-run repeats every domain's LLM calls, so keep this small.
MAX_GENERATION_RETRIES = 1


# ── conditional edge functions ────────────────────────────────────────────────
def route_after_classify(state: AgentState) -> str:
    if state["domain"] == "unknown" or state["intent"] == "unknown":
        return "abstain"
    if state["confidence"] < 0.5:
        return "abstain"
    return "domain_pipeline"


def route_after_domain_pipeline(state: AgentState) -> str:
    # at least one domain produced a cited section
    if state.get("context_grade") == "sufficient":
        return "verify_citations"
    return "abstain"


def route_after_grades(state: AgentState) -> str:
    # "unverified" = the groundedness LLM call was unavailable; the answer is shown with a warning.
    grounded = state.get("groundedness_grade") in ("grounded", "unverified")
    # "partial" (some domains are knowledge gaps) is a valid final answer; re-running won't add statutes.
    answerable = state.get("answerability_grade") in ("answers", "partial")
    if grounded and answerable:
        return "stitch_answer"
    if state.get("retry_count_generation", 0) >= MAX_GENERATION_RETRIES:
        return "abstain"
    return "regenerate"


def increment_generation_retry(state: AgentState) -> AgentState:
    state["retry_count_generation"] += 1
    state["trace"].append(f"regenerate → retry {state['retry_count_generation']}")
    return state


# ── build graph ───────────────────────────────────────────────────────────────
#   classify_query -> domain_pipeline (parallel workers, stitched) -> verify_citations
#                  -> grade_parallel -> stitch_answer          (abstain on failure)
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("classify_query",    classify_query)
    g.add_node("domain_pipeline",   domain_pipeline)
    g.add_node("verify_citations",  verify_citations)
    g.add_node("grade_parallel",    grade_parallel)
    g.add_node("regenerate",        increment_generation_retry)
    g.add_node("stitch_answer",     stitch_answer)
    g.add_node("abstain",           abstain)

    g.set_entry_point("classify_query")

    g.add_edge("verify_citations", "grade_parallel")
    g.add_edge("regenerate",       "domain_pipeline")
    g.add_edge("stitch_answer",    END)
    g.add_edge("abstain",          END)

    g.add_conditional_edges("classify_query", route_after_classify,
                            {"domain_pipeline": "domain_pipeline", "abstain": "abstain"})
    g.add_conditional_edges("domain_pipeline", route_after_domain_pipeline,
                            {"verify_citations": "verify_citations", "abstain": "abstain"})
    g.add_conditional_edges("grade_parallel", route_after_grades,
                            {"stitch_answer": "stitch_answer", "regenerate": "regenerate", "abstain": "abstain"})
    return g.compile()


# singleton
graph = build_graph()


def run_workflow(query: str, *, provider=None, top_k=None, as_of=None, verify_claims=None, **options) -> dict:
    """Run the workflow with per-request options (this is what the API should call).

        run_workflow("...", provider="gemini", top_k=8, as_of="2026-10-13")

    Options not given fall back to settings.RunSettings defaults. Invalid values raise ValueError.
    """
    cfg = build_config(provider=provider, top_k=top_k, as_of=as_of, verify_claims=verify_claims, **options)
    return graph.invoke({"query": query}, config=cfg)