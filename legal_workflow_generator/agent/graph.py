from langgraph.graph import StateGraph, END
from legal_workflow_generator.agent.state import AgentState
from legal_workflow_generator.agent.nodes import (
    classify_query,
    domain_pipeline,
    verify_citations,
    grade_parallel,
    stitch_answer,
    abstain,
)

# How many times the whole per-domain pipeline may be re-run after a failed
# grade. Each re-run repeats every domain's LLM calls, so keep this small.
MAX_GENERATION_RETRIES = 1

# ── conditional edge functions ────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    if state["domain"] == "unknown" or state["intent"] == "unknown":
        return "abstain"
    if state["confidence"] < 0.5:
        return "abstain"
    return "domain_pipeline"


def route_after_domain_pipeline(state: AgentState) -> str:
    # at least one domain produced a grounded, cited section
    if state.get("context_grade") == "sufficient":
        return "verify_citations"
    return "abstain"


def route_after_grades(state: AgentState) -> str:
    grounded = state.get("groundedness_grade") == "grounded"
    answers = state.get("answerability_grade") == "answers"
    if grounded and answers:
        return "stitch_answer"
    if state.get("retry_count_generation", 0) >= MAX_GENERATION_RETRIES:
        return "abstain"
    return "regenerate"


def increment_generation_retry(state: AgentState) -> AgentState:
    state["retry_count_generation"] += 1
    state["trace"].append(f"regenerate → retry {state['retry_count_generation']}")
    return state


# ── build graph ───────────────────────────────────────────────────────────────
#
#   classify_query
#        │
#        ▼
#   domain_pipeline  ── one worker per domain, run in parallel:
#        │               retrieve → grade → (rewrite → retrieve → grade)* → generate
#        │               then stitched into a single answer
#        ▼
#   verify_citations
#        │
#        ▼
#   grade_parallel   ── groundedness + answerability, run concurrently
#        │
#        ▼
#   stitch_answer    ── flags domains with no cited provisions as knowledge gaps

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

    # linear edges
    g.add_edge("verify_citations", "grade_parallel")
    g.add_edge("regenerate",       "domain_pipeline")
    g.add_edge("stitch_answer",    END)
    g.add_edge("abstain",          END)

    # conditional edges
    g.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "domain_pipeline": "domain_pipeline",
            "abstain":         "abstain",
        },
    )

    g.add_conditional_edges(
        "domain_pipeline",
        route_after_domain_pipeline,
        {
            "verify_citations": "verify_citations",
            "abstain":          "abstain",
        },
    )

    g.add_conditional_edges(
        "grade_parallel",
        route_after_grades,
        {
            "stitch_answer": "stitch_answer",
            "regenerate":    "regenerate",
            "abstain":       "abstain",
        },
    )

    return g.compile()


# singleton
graph = build_graph()