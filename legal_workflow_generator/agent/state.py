from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    # Input
    query: str
    all_domains: List[str]
    

    # Node 1: classify
    intent: str
    domain: str
    normalized_query: str
    keywords: List[str]
    confidence: float

    # Node 2: retrieve
    rewritten_query: Optional[str]
    retrieved_docs: List[dict]

    # Node 3: grade context
    context_grade: str          # "sufficient" / "insufficient"
    context_grade_reason: str

    # Node 4: generate
    answer: str
    citations: List[str]        # ["dpdp_act_2023_sec_6", ...]

    # Node 5: verify citations
    verified_citations: List[str]
    failed_citations: List[str]

    # Node 6: grade groundedness
    groundedness_grade: str     # "grounded" / "not_grounded"
    groundedness_reason: str

    # Node 7: grade answerability
    answerability_grade: str    # "answers" / "off_target"
    answerability_reason: str

    # Loop control — separate counters
    retry_count_retrieval: int  # max 3
    retry_count_generation: int # max 2

    # Terminal
    abstain: bool
    abstain_reason: str

    # Eval logging
    trace: List[str]