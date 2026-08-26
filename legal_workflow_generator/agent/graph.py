from langgraph.graph import StateGraph, END
from legal_workflow_generator.agent.state import AgentState
from legal_workflow_generator.agent.nodes import (
    classify_query,
    retrieve,
    grade_context,
    rewrite_query,
    generate,
    verify_citations,
    grade_groundedness,
    grade_answerability,
    abstain,
)

# ── conditional edge functions ───────────────────────────────────────────────
 
def route_after_classify(state: AgentState) -> str:
    if state["domain"] == "unknown" or state["intent"] == "unknown":
        return "abstain"
    if state["confidence"] < 0.5:
        return "abstain"
    return "retrieve"

def route_after_context_grade(state: AgentState) -> str:
    if state["context_grade"] == "sufficient":
        return "generate"
    if state["retry_count_retrieval"] >= 3:
        return "abstain"
    return "rewrite_query"

def route_after_groundedness(state: AgentState) -> str:
    if state["groundedness_grade"] == "grounded":
        return "grade_answerability"
    if state["retry_count_generation"] >= 2:
        return "abstain"
    return "regenerate"

def route_after_answerability(state: AgentState) -> str:
    if state["answerability_grade"] == "answers":
        return END
    if state["retry_count_retrieval"] >= 3:
        return "abstain"
    return "rewrite_query"

def increment_generation_retry(state: AgentState) -> AgentState:
    state["retry_count_generation"] += 1
    state["trace"].append(f"regenerate → retry {state['retry_count_generation']}")
    return state
def route_after_verify_citations(state: AgentState) -> str:
    # for multi-domain queries with verified citations, skip groundedness
    all_domains = state.get("all_domains", [])
    if len(all_domains) > 1 and len(state.get("verified_citations", [])) > 0:
        return "grade_answerability"
    return "grade_groundedness"
# ── build graph ──────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    # add all nodes
    g.add_node("classify_query",       classify_query)
    g.add_node("retrieve",             retrieve)
    g.add_node("grade_context",        grade_context)
    g.add_node("rewrite_query",        rewrite_query)
    g.add_node("generate",             generate)
    g.add_node("verify_citations",     verify_citations)
    g.add_node("grade_groundedness",   grade_groundedness)
    g.add_node("grade_answerability",  grade_answerability)
    g.add_node("regenerate",           increment_generation_retry)
    g.add_node("abstain",              abstain)

    # entry point
    g.set_entry_point("classify_query")

    # linear edges
    # replace with:
    g.add_conditional_edges(
    "classify_query",
    route_after_classify,
    {
        "retrieve": "retrieve",
        "abstain":  "abstain",
    }  
    )
    g.add_edge("retrieve",           "grade_context")
    g.add_edge("rewrite_query",      "retrieve")
    g.add_edge("generate",           "verify_citations")
    g.add_conditional_edges(
    "verify_citations",
    route_after_verify_citations,
    {
        "grade_groundedness": "grade_groundedness",
        "grade_answerability": "grade_answerability",
    }
)
    g.add_edge("regenerate",         "generate")
    g.add_edge("abstain",            END)

    # conditional edges
    g.add_conditional_edges(
        "grade_context",
        route_after_context_grade,
        {
            "generate":      "generate",
            "rewrite_query": "rewrite_query",
            "abstain":       "abstain",
        }
    )

    g.add_conditional_edges(
        "grade_groundedness",
        route_after_groundedness,
        {
            "grade_answerability": "grade_answerability",
            "regenerate":          "regenerate",
            "abstain":             "abstain",
        }
    )

    g.add_conditional_edges(
        "grade_answerability",
        route_after_answerability,
        {
            END:             END,
            "rewrite_query": "rewrite_query",
            "abstain":       "abstain",
        }
    )

    return g.compile()

# singleton
graph = build_graph()