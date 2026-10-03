from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import re

from google import genai as _genai

from legal_workflow_generator.agent.state import AgentState
from legal_workflow_generator.rag.pipeline import RagPipeline
from legal_workflow_generator.query.normalizer import QueryNormalizer
from legal_workflow_generator.query.intent_classifier import IntentClassifier
from legal_workflow_generator.query.context_resolver import LegalContextResolver
from legal_workflow_generator.config.values import GEMINI_API_KEY, GEMINI_MODEL

# ── shared clients (init once) ────────────────────────────────────────────────
_normalizer = QueryNormalizer()
_intent_classifier = IntentClassifier()
_context_resolver = LegalContextResolver()
_rag_pipeline: RagPipeline | None = None
_gemini_client = _genai.Client(api_key=GEMINI_API_KEY)

VALID_DOMAINS = ["data_protection", "corporate_governance", "ip_licensing", "taxation", "employment"]

# ── statute prefix → domain mapping ──────────────────────────────────────────
STATUTE_TO_DOMAIN = {
    "dpdp_act": "data_protection",
    "it_act": "data_protection",
    "ca_2013": "corporate_governance",
    "companies": "corporate_governance",
    "copyright_act": "ip_licensing",
    "patent": "ip_licensing",
    "trademark": "ip_licensing",
    "igst_act": "taxation",
    "gst_comp_act": "taxation",
    "cgst": "taxation",
    "posh_act": "employment",
    "era_1976": "employment",
}

DOMAIN_QUERY_HINTS = {
    "data_protection": "personal data privacy DPDP consent breach notification",
    "corporate_governance": "company incorporation directors shareholders compliance",
    "ip_licensing": "copyright patent trademark software license",
    "taxation": "GST income tax TDS filing returns",
    "employment": "employee hiring salary POSH ICC internal complaints committee harassment equal pay",
}


def _get_rag_pipeline() -> RagPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RagPipeline(llm_provider="gemini")
    return _rag_pipeline


def _llm(system: str, user: str, grader: bool = False, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = _gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{system}\n\n{user}",
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


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
            pid = doc.get("provision_id", "")
            score = doc.get("combined_score", 0) or 0
            for prefix, domain in STATUTE_TO_DOMAIN.items():
                if pid.startswith(prefix):
                    domain_scores[domain] = domain_scores.get(domain, 0) + score
                    break

        if not domain_scores:
            return []

        # keep domains scoring at least 30% of the top domain's score
        threshold = max(domain_scores.values()) * 0.3
        return [d for d, s in domain_scores.items() if s >= threshold]
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

    # ── retrieval-grounded domain detection ──────────────────────────────────
    retrieval_domains = _detect_domains_from_retrieval(state["normalized_query"])

    # ── LLM domain detection (catches what retrieval misses) ─────────────────
    try:
        multi_prompt = f"""This query may span multiple legal domains.
Query: {state['query']}
Domains: data_protection, corporate_governance, ip_licensing, taxation, employment

Reply with ONLY the relevant domain names comma separated. Include ALL that apply.
Example: data_protection, employment"""
        raw = _llm("You are a legal domain classifier.", multi_prompt)
        llm_domains = [d.strip() for d in raw.split(",") if d.strip() in VALID_DOMAINS]
    except Exception:
        llm_domains = []

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


# ── Node 2: retrieve (PARALLELIZED) ──────────────────────────────────────────
def retrieve(state: AgentState) -> AgentState:
    rag_pipeline = _get_rag_pipeline()
    rag_pipeline._ensure_index()  # build index once, BEFORE spawning threads
    q = state.get("rewritten_query") or state["normalized_query"]
    all_domains = state.get("all_domains") or [state["domain"]]

    def retrieve_for_domain(domain: str):
        hint = DOMAIN_QUERY_HINTS.get(domain, "")
        domain_q = f"{q} {hint}".strip()
        try:
            return domain, rag_pipeline.searcher.search(domain_q, top_k=3)
        except Exception:
            return domain, []

    all_docs = []
    seen_ids = set()

    with ThreadPoolExecutor(max_workers=max(1, min(5, len(all_domains)))) as executor:
        futures = [executor.submit(retrieve_for_domain, d) for d in all_domains]
        for future in as_completed(futures):
            _, docs = future.result()
            for d in docs:
                pid = d.get("provision_id", "")
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_docs.append(d)

    # keep top 7 by combined score
    all_docs = sorted(all_docs, key=lambda x: x.get("combined_score", 0), reverse=True)[:7]

    state["retrieved_docs"] = all_docs
    state["trace"].append(f"retrieve → {len(all_docs)} docs across {all_domains} (parallel)")
    for d in all_docs:
        state["trace"].append(
            f"    doc → [{d.get('provision_id')}] {d.get('title')} (score={d.get('combined_score')})"
        )
    return state


# ── Node 3: grade context ─────────────────────────────────────────────────────
def grade_context(state: AgentState) -> AgentState:
    docs = state["retrieved_docs"]
    if not docs:
        state["context_grade"] = "insufficient"
        state["context_grade_reason"] = "No documents retrieved"
        state["trace"].append("grade_context → insufficient (no docs)")
        return state

    chunks = "\n\n".join(
        f"[{d.get('provision_id','')}] {d.get('title','')}: {str(d.get('text',''))[:400]}"
        for d in docs
    )
    all_domains = state.get("all_domains", [state["domain"]])
    system = "You are a legal relevance grader. Reply in JSON only."
    user = f"""Query: {state['query']}
Domains being searched: {all_domains}

Retrieved chunks:
{chunks}

These are individual statutory provisions. For multi-domain queries, grade 
"sufficient" if chunks collectively cover the main compliance areas asked about,
even if not exhaustive. Grade "insufficient" only if chunks are largely irrelevant.
Reply ONLY with:
{{"grade": "sufficient" or "insufficient", "reason": "one sentence"}}"""

    raw = _llm(system, user, grader=True)
    try:
        parsed = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group())
        state["context_grade"] = parsed.get("grade", "insufficient")
        state["context_grade_reason"] = parsed.get("reason", "")
    except Exception:
        state["context_grade"] = "insufficient"
        state["context_grade_reason"] = "Could not parse grader response"

    state["trace"].append(f"grade_context → {state['context_grade']}: {state['context_grade_reason']}")
    return state


# ── Node: rewrite query ───────────────────────────────────────────────────────
def rewrite_query(state: AgentState) -> AgentState:
    state["retry_count_retrieval"] += 1
    system = "You are a legal search query optimizer."
    user = f"""Original query: {state['query']}
Reason retrieval failed: {state['context_grade_reason']}
Domains: {state.get('all_domains', [state['domain']])}

Write a better search query targeting Indian statutory compliance.
Reply with ONLY the improved query, nothing else."""

    new_q = _llm(system, user)
    state["rewritten_query"] = new_q.strip()
    state["trace"].append(f"rewrite_query → retry {state['retry_count_retrieval']}: '{new_q[:60]}'")
    return state


# ── Node 4: generate ──────────────────────────────────────────────────────────
def generate(state: AgentState) -> AgentState:
    docs = state["retrieved_docs"]
    chunks = "\n\n".join(
        f"[{d.get('provision_id','')}] {d.get('title','')}: "
        f"{str(d.get('text','') or d.get('plain_english_summary',''))[:800]}"
        for d in docs
    )
    all_domains = state.get("all_domains", [state["domain"]])
    is_multi = len(all_domains) > 1

    system = "You are a legal workflow assistant for Indian tech startups."
    user = f"""You are a legal compliance assistant for Indian tech startups.
Answer using ONLY the retrieved context below. Do not invent sections or obligations.
Address ALL specific details mentioned in the query (company size, user type, employee demographics etc.)
{"This is a multi-domain compliance query — address each relevant area separately." if is_multi else ""}


Format your response EXACTLY like this:

LEGAL COMPLIANCE WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Query: {state['query']}

SUMMARY
- [2-4 bullet points covering each compliance area]

ACTION CHECKLIST
Step 1: [action] — [Act/Section]
Step 2: [action] — [Act/Section]
Step 3: [action] — [Act/Section]
(add more steps if needed, group by domain if multi-domain)

SOURCES
- [provision_id] | [section title]
- [provision_id] | [section title]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠ This is informational guidance, not legal advice.
Consult a qualified legal professional for your specific situation.

Retrieved context:
{chunks}"""

    answer = _llm(system, user)
    provision_ids = [d.get("provision_id", "") for d in docs if d.get("provision_id")]
    state["answer"] = answer
    state["citations"] = provision_ids
    state["trace"].append(f"generate → {len(provision_ids)} citations found")
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

    system = """You are a legal grounding verifier. Reply in JSON only.
Be lenient: grade as 'grounded' if the main claims are supported by context.
Only grade 'not_grounded' if the answer makes specific legal claims that
directly contradict or are completely absent from the retrieved context.
Minor elaborations and reasonable inferences are acceptable."""

    user = f"""Answer: {state['answer'][:1000]}

Retrieved context:
{chunks}

{"Note: This is a multi-domain query. The answer may draw on multiple legal areas — grade as grounded if each domain's claims are supported by at least some of the retrieved chunks." if is_multi else ""}

Is every claim in the answer supported by the retrieved context?
Reply ONLY with:
{{"grade": "grounded" or "not_grounded", "reason": "one sentence"}}"""

    raw = _llm(system, user, grader=True)
    try:
        parsed = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group())
        state["groundedness_grade"] = parsed.get("grade", "not_grounded")
        state["groundedness_reason"] = parsed.get("reason", "")
    except Exception:
        state["groundedness_grade"] = "not_grounded"
        state["groundedness_reason"] = "Could not parse grader response"

    state["trace"].append(f"grade_groundedness → {state['groundedness_grade']}")
    return state


# ── Node 7: grade answerability ───────────────────────────────────────────────
def grade_answerability(state: AgentState) -> AgentState:
    all_domains = state.get("all_domains", [state["domain"]])
    is_multi = len(all_domains) > 1
    system = "You are a legal answer quality checker. Reply in JSON only."
    user = f"""Query: {state['query']}
Answer: {state['answer'][:1000]}

Does this answer address the main compliance areas asked about?
{"For multi-domain queries, grade 'answers' if the answer covers the key compliance areas even if it doesn't address every specific detail like company size." if is_multi else ""}
Grade 'off_target' only if the answer is completely unrelated to what was asked.
Reply ONLY with:
{{"grade": "answers" or "off_target", "reason": "one sentence"}}"""

    raw = _llm(system, user, grader=True)
    try:
        parsed = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group())
        state["answerability_grade"] = parsed.get("grade", "off_target")
        state["answerability_reason"] = parsed.get("reason", "")
    except Exception:
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
        reason = (
            state.get("context_grade_reason")
            or state.get("groundedness_reason")
            or "Insufficient reliable information found."
        )

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

    DOMAIN_PREFIXES = {
        "corporate_governance": ["ca_2013", "companies"],
        "data_protection": ["dpdp", "it_act"],
        "ip_licensing": ["copyright", "patent", "trademark"],
        "taxation": ["igst", "gst", "cgst"],
        "employment": ["posh", "era"],
    }

    covered_domains = []
    gap_domains = []

    for domain in DOMAIN_ORDER:
        if domain not in all_domains:
            continue
        prefixes = DOMAIN_PREFIXES.get(domain, [])
        domain_cites = [c for c in verified if any(c.startswith(p) for p in prefixes)]
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