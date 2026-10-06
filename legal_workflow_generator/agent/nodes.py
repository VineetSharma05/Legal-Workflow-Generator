from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar
import time

import psycopg2
from pydantic import BaseModel

from legal_workflow_generator.agent.state import AgentState
from legal_workflow_generator.rag.pipeline import RagPipeline
from legal_workflow_generator.query.normalizer import QueryNormalizer
from legal_workflow_generator.query.intent_classifier import IntentClassifier
from legal_workflow_generator.query.context_resolver import LegalContextResolver
from legal_workflow_generator.llm import StructuredOutputError, invoke_structured, structured_llm
from legal_workflow_generator.llm.schemas import (
    AnswerabilityGrade,
    DomainList,
    GroundednessGrade,
    RelevanceGrade,
    SearchQuery,
    domain_workflow_model,
)
import legal_workflow_generator.config.values as config

T = TypeVar("T", bound=BaseModel)

# ── shared clients (init once) ────────────────────────────────────────────────
_normalizer = QueryNormalizer()
_intent_classifier = IntentClassifier()
_context_resolver = LegalContextResolver()
_rag_pipeline: RagPipeline | None = None

VALID_DOMAINS = ["data_protection", "corporate_governance", "ip_licensing", "taxation", "employment"]

# ── provision_id → domain (read from the `domain` column set at ingestion) ───
_DOMAIN_MAP: dict[str, str] | None = None

# fallback only, for any provision_id missing from the DB map
_PREFIX_FALLBACK = {
    "dpdp": "data_protection",
    "information_technology": "data_protection",
    "ca_2013": "corporate_governance",
    "contract_act": "corporate_governance",
    "llp_act": "corporate_governance",
    "copyright": "ip_licensing",
    "patent": "ip_licensing",
    "trademark": "ip_licensing",
    "designs": "ip_licensing",
    "cgst": "taxation",
    "igst": "taxation",
    "income_tax": "taxation",
    "posh": "employment",
    "rpwd": "employment",
    "era_1976": "employment",
}


def _load_domain_map() -> dict[str, str]:
    global _DOMAIN_MAP
    if _DOMAIN_MAP is None:
        try:
            conn = psycopg2.connect(
                dbname=config.DB_NAME, user=config.DB_USER, password=config.PGPASSWORD,
                host=config.DB_HOST, port=config.DB_PORT,
            )
            cur = conn.cursor()
            cur.execute("SELECT provision_id, domain FROM laws WHERE domain IS NOT NULL")
            _DOMAIN_MAP = {pid: dom for pid, dom in cur.fetchall()}
            cur.close()
            conn.close()
        except Exception:
            _DOMAIN_MAP = {}
    return _DOMAIN_MAP


def _domain_of(pid: str) -> str | None:
    dom = _load_domain_map().get(pid)
    if dom in VALID_DOMAINS:
        return dom
    for prefix, d in _PREFIX_FALLBACK.items():
        if pid.startswith(prefix):
            return d
    return None


DOMAIN_QUERY_HINTS = {
    "data_protection": "personal data privacy DPDP consent breach notification",
    "corporate_governance": "company incorporation directors shareholders compliance",
    "ip_licensing": "copyright patent trademark software license",
    "taxation": "GST",
    "employment": "",
}


def _get_rag_pipeline() -> RagPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RagPipeline(llm_provider="gemini")
    return _rag_pipeline


def _llm(schema: type[T], system: str, user: str) -> T:
    """Structured Gemini call. Raises on API failure or StructuredOutputError on a schema mismatch."""
    return invoke_structured(structured_llm(schema), system, user)


def _detect_domains_from_retrieval(query: str) -> list[str]:
    """
    Detect relevant domains by doing a domain-agnostic retrieval
    and reading which statutes appear in the top results.
    Zero LLM calls, zero hallucination risk.
    """
    try:
        # NOTE: use the lazy getter — the global _rag_pipeline can still be None here
        rag_pipeline = _get_rag_pipeline()
        rag_pipeline._ensure_index()
        docs = rag_pipeline.searcher.search(query, top_k=7)

        domain_scores: dict[str, float] = {}
        for doc in docs:
            domain = _domain_of(doc.get("provision_id", ""))
            if domain:
                score = doc.get("combined_score", 0) or 0
                domain_scores[domain] = domain_scores.get(domain, 0) + score

        if not domain_scores:
            return []

        # keep domains scoring at least 30% of the top domain's score
        threshold = max(domain_scores.values()) * 0.3
        return [d for d, s in domain_scores.items() if s >= threshold]
    except Exception:
        return []


def _llm_domains(query: str) -> list[str]:
    try:
        prompt = f"""This query may span multiple legal domains.
Query: {query}

List every legal domain the query touches. Include ALL that apply."""
        return list(dict.fromkeys(_llm(DomainList, "You are a legal domain classifier.", prompt).domains))
    except Exception:
        return []


# ── Node 1: classify ──────────────────────────────────────────────────────────
def classify_query(state: AgentState) -> AgentState:
    normalized = _normalizer.normalize(text=state["query"])
    intent, confidence = _intent_classifier.classify(normalized)
    context = _context_resolver.resolve(normalized, intent, confidence)

    state["intent"] = str(context["intent"])
    state["domain"] = context["legal_domain"]
    state["normalized_query"] = context["normalized_query"]
    state["keywords"] = context["keywords"]
    state["confidence"] = context["confidence"]
    state["retry_count_retrieval"] = 0
    state["retry_count_generation"] = 0
    state["abstain"] = False
    state["abstain_reason"] = ""

    # ── retrieval-based + LLM-based domain detection, run concurrently ───────
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_ret = ex.submit(_detect_domains_from_retrieval, state["normalized_query"])
        f_llm = ex.submit(_llm_domains, state["query"])
        retrieval_domains = f_ret.result()
        llm_domains = f_llm.result()

    # COMBINE: union of retrieval + LLM domains (+ the resolver's domain)
    combined = list(dict.fromkeys(retrieval_domains + llm_domains + [state["domain"]]))
    combined = [d for d in combined if d in VALID_DOMAINS]

    state["all_domains"] = combined if combined else [state["domain"]]
    state["domain_detection_method"] = f"retrieval({len(retrieval_domains)})+llm({len(llm_domains)})"

    state["trace"] = [
        f"classify → intent={state['intent']} domain={state['domain']} "
        f"all_domains={state.get('all_domains')} method={state.get('domain_detection_method')}"
    ]
    return state


# ── Node 5: verify citations ──────────────────────────────────────────────────
def verify_citations(state: AgentState) -> AgentState:
    doc_ids = {d.get("provision_id", "") for d in state["retrieved_docs"]}
    verified, failed = [], []
    for cit in state["citations"]:
        (verified if cit in doc_ids else failed).append(cit)
    state["verified_citations"] = verified
    state["failed_citations"] = failed
    state["trace"].append(f"verify_citations → verified={verified} failed={failed}")
    return state


# ── Node 6: grade groundedness ────────────────────────────────────────────────
def grade_groundedness(state: AgentState) -> AgentState:
    chunks = "\n\n".join(
        f"[{d.get('provision_id','')}]: {str(d.get('text',''))[:400]}"
        for d in state["retrieved_docs"]
    )
    all_domains = state.get("all_domains", [state["domain"]])
    is_multi = len(all_domains) > 1

    system = """You are a legal grounding verifier.
Be lenient: grade as 'grounded' if the main claims are supported by context.
Only grade 'not_grounded' if the answer makes specific legal claims that
directly contradict or are completely absent from the retrieved context.
Minor elaborations and reasonable inferences are acceptable."""

    user = f"""Answer: {state['answer'][:3000]}

Retrieved context:
{chunks}

{"Note: This is a multi-domain query. The answer may draw on multiple legal areas — grade as grounded if each domain's claims are supported by at least some of the retrieved chunks." if is_multi else ""}

Is every claim in the answer supported by the retrieved context?"""

    try:
        g = _llm(GroundednessGrade, system, user)
        state["groundedness_grade"] = g.grade
        state["groundedness_reason"] = g.reason
        unsupported = g.unsupported_claims
    except StructuredOutputError:
        state["groundedness_grade"] = "not_grounded"
        state["groundedness_reason"] = "Could not parse grader response"
        unsupported = []

    state["trace"].append(
        f"grade_groundedness → {state['groundedness_grade']}"
        + (f" (unsupported: {unsupported})" if unsupported else "")
    )
    return state


# ── Node 7: grade answerability ───────────────────────────────────────────────
def grade_answerability(state: AgentState) -> AgentState:
    all_domains = state.get("all_domains", [state["domain"]])
    is_multi = len(all_domains) > 1
    system = "You are a legal answer quality checker."
    user = f"""Query: {state['query']}
Answer: {state['answer'][:3000]}

Does this answer address the main compliance areas asked about?
{"For multi-domain queries, grade 'answers' if the answer covers the key compliance areas even if it doesn't address every specific detail like company size." if is_multi else ""}
Grade 'off_target' only if the answer is completely unrelated to what was asked."""

    try:
        g = _llm(AnswerabilityGrade, system, user)
        state["answerability_grade"] = g.grade
        state["answerability_reason"] = g.reason
    except StructuredOutputError:
        state["answerability_grade"] = "off_target"
        state["answerability_reason"] = "Could not parse grader response"

    state["trace"].append(f"grade_answerability → {state['answerability_grade']}")
    return state


# ── Node: abstain ─────────────────────────────────────────────────────────────
def abstain(state: AgentState) -> AgentState:
    state["abstain"] = True
    if state.get("domain") == "unknown" or state.get("intent") == "unknown":
        reason = "Could not determine the legal domain of your query."
    elif state.get("confidence", 1.0) < 0.5:
        reason = (
            f"Query classification confidence too low ({state.get('confidence')}). "
            f"Please rephrase with more specific legal context."
        )
    else:
        if state.get("groundedness_grade") == "not_grounded":
            reason = state.get("groundedness_reason") or "Answer was not grounded in the retrieved provisions."
        elif state.get("answerability_grade") == "off_target":
            reason = state.get("answerability_reason") or "Answer did not address the question."
        else:
            reason = state.get("context_grade_reason") or "Insufficient reliable information found."

    state["answer"] = (
        f"I was unable to provide a reliable answer to your question: "
        f"'{state['query']}'\n\n"
        f"Reason: {reason}\n\n"
        f"Please consult a qualified legal professional or refer directly "
        f"to the relevant Act."
    )
    state["abstain_reason"] = reason
    state["trace"].append(f"abstain → triggered: {reason}")
    return state


# ── Node 8: stitch ────────────────────────────────────────────────────────────
def stitch_answer(state: AgentState) -> AgentState:
    all_domains = state.get("all_domains", [state["domain"]])
    answer = state.get("answer", "")
    verified = state.get("verified_citations", [])

    # domain order for logical presentation
    DOMAIN_ORDER = [
        "corporate_governance",
        "data_protection",
        "ip_licensing",
        "taxation",
        "employment",
    ]

    DOMAIN_LABELS = {
        "corporate_governance": "Corporate Governance",
        "data_protection": "Data Protection",
        "ip_licensing": "Intellectual Property",
        "taxation": "Taxation & GST",
        "employment": "Employment & Workplace",
    }

    covered_domains = []
    gap_domains = []

    for domain in DOMAIN_ORDER:
        if domain not in all_domains:
            continue
        domain_cites = [c for c in verified if _domain_of(c) == domain]
        if domain_cites:
            covered_domains.append(domain)
        else:
            gap_domains.append(domain)

    gaps = [
        f"- {DOMAIN_LABELS.get(domain, domain)}: insufficient information found in knowledge base"
        for domain in gap_domains
    ]
    state["domain_gaps"] = gaps

    # no gaps → pass through
    if not gaps:
        state["stitched_answer"] = answer
        state["trace"].append("stitch → no gaps, answer passed through")
        return state

    gaps_text = "\n".join(gaps)
    state["stitched_answer"] = f"""{answer}

⚠ KNOWLEDGE GAPS IDENTIFIED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The following compliance areas could not be fully addressed 
due to insufficient information in the current knowledge base.
Please consult a legal professional for these areas:

{gaps_text}"""
    state["trace"].append(f"stitch → {len(gaps)} gap(s) flagged: {gap_domains}")
    return state



# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL PER-DOMAIN PIPELINE  (fan-out → one worker per domain → stitch)
#
#   worker(domain) = retrieve → grade → (rewrite → retrieve → grade)* → generate
#
# Only reads/writes state keys that already exist, so state.py is unchanged.
# ══════════════════════════════════════════════════════════════════════════════
_DOMAIN_ORDER = [
    "corporate_governance",
    "data_protection",
    "ip_licensing",
    "taxation",
    "employment",
]

_DOMAIN_LABELS = {
    "corporate_governance": "Corporate Governance",
    "data_protection": "Data Protection",
    "ip_licensing": "Intellectual Property",
    "taxation": "Taxation & GST",
    "employment": "Employment & Workplace",
}

MAX_DOMAIN_RETRIES = 1   # rewrite attempts per domain (up to 2 retrievals)
DOCS_PER_DOMAIN = 6


def _el(t0: float) -> str:
    """Seconds since this worker started, for trace lines."""
    return f"+{time.perf_counter() - t0:4.1f}s"


def _search_domain(rag, query: str, domain: str) -> list:
    """Search, then keep only provisions that belong to this domain."""
    hint = DOMAIN_QUERY_HINTS.get(domain, "")
    docs = rag.searcher.search(f"{query} {hint}".strip(), top_k=100)
    keep = [
        d for d in docs
        if "_forms_" not in d.get("provision_id", "")
        and _domain_of(d.get("provision_id", "")) == domain
    ]
    # statute sections carry the substantive obligations; rules mostly carry procedure.
    # take Act sections first, then fill the remaining slots with rules.
    acts = [d for d in keep if "_rules_" not in d.get("provision_id", "")]
    rules = [d for d in keep if "_rules_" in d.get("provision_id", "")]
    return (acts[:4] + rules)[:DOCS_PER_DOMAIN]


def _grade_domain(domain: str, user_q: str, docs: list) -> tuple[str, str]:
    chunks = "\n\n".join(
        f"[{d.get('provision_id','')}] {d.get('title','')}: {str(d.get('text',''))[:400]}"
        for d in docs
    )
    system = "You are a legal relevance grader."
    user = f"""Query: {user_q}
Legal area being checked: {_DOMAIN_LABELS[domain]}

Retrieved provisions:
{chunks}

Grade ONLY the part of the query that concerns this legal area.
"sufficient": at least one provision directly addresses that part of the query.
"insufficient": the provisions are unrelated to it."""
    try:
        g = _llm(RelevanceGrade, system, user)
        return g.grade, g.reason
    except StructuredOutputError:
        return "insufficient", "Could not parse grader response"
    except Exception as e:
        return "insufficient", f"Grader call failed: {e}"


def _rewrite_for_domain(domain: str, user_q: str, reason: str) -> str:
    system = "You are a legal search query optimizer."
    user = f"""Original query: {user_q}
Legal area: {_DOMAIN_LABELS[domain]}
Why retrieval failed: {reason}

Write a better search query for Indian statutes in this legal area only.
Use plain topic words. Do NOT cite section numbers you are not certain exist."""
    try:
        return _llm(SearchQuery, system, user).query.strip() or user_q
    except Exception:
        return user_q


def _focus_query(domain: str, user_q: str, fallback: str) -> str:
    """Multi-domain queries: extract only this domain's part as a standalone search query."""
    system = "You are a legal search query writer."
    user = f"""Question: {user_q}

Rewrite this question as a short standalone search query covering ONLY the {_DOMAIN_LABELS[domain]} aspects.
Keep the concrete nouns from the question (for example: wheelchair ramps, accessible software, disabled employees).
Drop everything that belongs to other legal areas."""
    try:
        return _llm(SearchQuery, system, user).query.strip() or fallback
    except Exception:
        return fallback


def _generate_domain(domain: str, user_q: str, docs: list) -> dict:
    ids = list(dict.fromkeys(d.get("provision_id", "") for d in docs if d.get("provision_id")))
    out = {"summary": [], "steps": [], "not_covered": "", "cited": []}
    if not ids:
        return out
    chunks = "\n\n".join(
        f"[{d.get('provision_id','')}] {d.get('title','')}: "
        f"{str(d.get('text','') or d.get('plain_english_summary',''))[:800]}"
        for d in docs
    )
    system = "You are a legal workflow assistant for Indian tech startups."
    user = f"""Query: {user_q}
Legal area: {_DOMAIN_LABELS[domain]}

Using ONLY the provisions below, list the compliance actions relevant to this legal area for the query.
Do not invent sections or obligations. Address specifics in the query (company size, employee type, etc.).
For each step, cite the exact provision_id it comes from and say who the provision applies to, based only on the text.

Rules:
- Discuss ONLY {_DOMAIN_LABELS[domain]}. Do not mention other legal areas (tax, data protection, etc.) in the summary.
- Include a step only if the provision directly responds to the query. Skip provisions that concern other kinds of
  workplaces (factories, mines, plantations, schools) or are unrelated to the specific question.
- If nothing in the provisions answers part of the query, say so in "not_covered" instead of padding the steps.

Provisions:
{chunks}"""

    try:
        p = _llm(domain_workflow_model(ids), system, user)
    except Exception:
        return out

    # the schema already restricts provision_id to `ids`; this only drops empty actions
    steps = [
        {"action": s.action.strip(), "provision_id": s.provision_id, "applies_to": s.applies_to}
        for s in p.steps
        if s.action.strip()
    ]

    out["summary"] = [b.strip() for b in p.summary if b.strip()]
    out["steps"] = steps
    out["not_covered"] = p.not_covered.strip()
    out["cited"] = list(dict.fromkeys(s["provision_id"] for s in steps))
    return out


def _domain_worker(domain: str, search_q: str, user_q: str, rag, multi: bool = False) -> tuple[dict, list]:
    """One independent mini-pipeline for a single domain. Returns (result, trace)."""
    trace = []
    t0 = time.perf_counter()
    q = search_q
    if multi:
        q = _focus_query(domain, user_q, search_q)
        trace.append(f"{_el(t0)} [{domain}] focus → '{q[:70]}'")
    docs, grade, reason = [], "insufficient", "No documents retrieved"
    retries = 0

    for attempt in range(MAX_DOMAIN_RETRIES + 1):
        try:
            docs = _search_domain(rag, q, domain)
        except Exception as e:
            docs, reason = [], f"Search failed: {e}"
        trace.append(f"{_el(t0)} [{domain}] retrieve (try {attempt + 1}) → {len(docs)} docs")

        if docs:
            grade, reason = _grade_domain(domain, user_q, docs)
            trace.append(f"{_el(t0)} [{domain}] grade → {grade}: {reason}")
        else:
            grade = "insufficient"

        if grade == "sufficient" or attempt == MAX_DOMAIN_RETRIES:
            break

        q = _rewrite_for_domain(domain, user_q, reason)
        retries += 1
        trace.append(f"{_el(t0)} [{domain}] rewrite → '{q[:60]}'")

    result = {
        "domain": domain, "docs": docs, "grade": grade, "reason": reason,
        "retries": retries, "summary": [], "steps": [], "not_covered": "", "cited": [],
    }
    if grade == "sufficient":
        result.update(_generate_domain(domain, user_q, docs))
        trace.append(f"{_el(t0)} [{domain}] generate → {len(result['steps'])} steps, {len(result['cited'])} citations")
    return result, trace


def _stitch_sections(query: str, results: list) -> str:
    """Stitching layer: merge per-domain results into one workflow document."""
    bar = "━" * 38
    titles = {d.get("provision_id", ""): d.get("title", "") for r in results for d in r["docs"]}

    summary, checklist, sources = [], [], []
    n = 0
    for r in results:
        if not r["steps"]:
            continue
        label = _DOMAIN_LABELS[r["domain"]]
        for b in r["summary"]:
            summary.append(f"- {label}: {b}")
        if r["not_covered"]:
            summary.append(f"- {label} (not addressed by retrieved provisions): {r['not_covered']}")
        checklist.append(f"**{label}**")
        for st in r["steps"]:
            n += 1
            note = "" if st["applies_to"] == "unspecified" else f" (applies to: {st['applies_to']})"
            checklist.append(f"Step {n}: {st['action']} — [{st['provision_id']}]{note}")
        checklist.append("")
        for pid in r["cited"]:
            line = f"- [{pid}] | {titles.get(pid, '')}"
            if line not in sources:
                sources.append(line)

    return "\n".join([
        "LEGAL COMPLIANCE WORKFLOW",
        bar,
        f"Query: {query}",
        "",
        "SUMMARY",
        *summary,
        "",
        "ACTION CHECKLIST",
        *checklist,
        "SOURCES",
        *sources,
        "",
        bar,
        "⚠ This is informational guidance, not legal advice.",
        "Consult a qualified legal professional for your specific situation.",
    ])


# ── Node 2 (new): parallel per-domain pipeline ───────────────────────────────
def domain_pipeline(state: AgentState) -> AgentState:
    rag = _get_rag_pipeline()
    rag._ensure_index()      # build/load everything BEFORE spawning threads
    _load_domain_map()

    wanted = state.get("all_domains") or [state["domain"]]
    domains = [d for d in _DOMAIN_ORDER if d in wanted] or list(wanted)
    search_q = state.get("normalized_query") or state["query"]
    user_q = state["query"]

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, len(domains))) as ex:
        futures = [ex.submit(_domain_worker, d, search_q, user_q, rag, len(domains) > 1) for d in domains]
        outputs = [f.result() for f in futures]      # keeps domain order

    results = [r for r, _ in outputs]
    for _, trace in outputs:
        state["trace"].extend(trace)

    # merged views for the downstream nodes
    seen, all_docs = set(), []
    for r in results:
        for d in r["docs"]:
            pid = d.get("provision_id", "")
            if pid not in seen:
                seen.add(pid)
                all_docs.append(d)
    state["retrieved_docs"] = all_docs
    state["retry_count_retrieval"] = max((r["retries"] for r in results), default=0)

    covered = [r for r in results if r["steps"]]
    if covered:
        state["context_grade"] = "sufficient"
        state["context_grade_reason"] = "; ".join(f"{r['domain']}: {r['grade']}" for r in results)
        state["answer"] = _stitch_sections(state["query"], results)
        state["citations"] = list(dict.fromkeys(c for r in covered for c in r["cited"]))
    else:
        state["context_grade"] = "insufficient"
        state["context_grade_reason"] = "; ".join(
            f"{r['domain']}: {r['reason']}" for r in results
        ) or "No domain returned usable provisions"
        state["answer"] = ""
        state["citations"] = []

    state["trace"].append(
        f"domain_pipeline → {len(covered)}/{len(results)} domains answered "
        f"({', '.join(r['domain'] for r in covered) or 'none'}) "
        f"in {time.perf_counter() - t_start:.1f}s wall-clock (parallel)"
    )
    return state


# ── Node 6+7 (new): run both graders concurrently ────────────────────────────
def grade_parallel(state: AgentState) -> AgentState:
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(grade_groundedness, state)
        f2 = ex.submit(grade_answerability, state)
        f1.result()
        f2.result()
    return state