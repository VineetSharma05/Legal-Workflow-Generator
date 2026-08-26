from legal_workflow_generator.agent.state import AgentState
from legal_workflow_generator.rag.pipeline import RagPipeline
from legal_workflow_generator.query.normalizer import QueryNormalizer
from legal_workflow_generator.query.intent_classifier import IntentClassifier
from legal_workflow_generator.query.context_resolver import LegalContextResolver
import time
import json
import re
from legal_workflow_generator.config.values import GEMINI_API_KEY, GEMINI_MODEL
from google import genai as _genai

# ── shared clients (init once) ───────────────────────────────────────────────
_normalizer = QueryNormalizer()
_intent_classifier = IntentClassifier()
_context_resolver = LegalContextResolver()
_rag_pipeline = RagPipeline(llm_provider="gemini")
_rag_pipeline._ensure_index()
_gemini_client = _genai.Client(api_key=GEMINI_API_KEY)

VALID_DOMAINS = ["data_protection", "corporate_governance", "ip_licensing", "taxation", "employment"]

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

# ── Node 1: classify ──────────────────────────────────────────────────────────
def classify_query(state: AgentState) -> AgentState:
    normalized = _normalizer.normalize(text=state["query"])
    intent, confidence = _intent_classifier.classify(normalized)
    context = _context_resolver.resolve(normalized, intent, confidence)

    state["intent"]           = str(context["intent"])
    state["domain"]           = context["legal_domain"]
    state["normalized_query"] = context["normalized_query"]
    state["keywords"]         = context["keywords"]
    state["confidence"]       = context["confidence"]
    state["retry_count_retrieval"]  = 0
    state["retry_count_generation"] = 0
    state["abstain"]          = False
    state["abstain_reason"]   = ""

    # multi-domain detection
    multi_prompt = f"""This query mentions specific Indian laws. List ALL legal domains covered.
Query: {state['query']}

Available domains:
- data_protection (DPDP Act, IT Act, privacy)
- corporate_governance (Companies Act, incorporation, directors)
- ip_licensing (Copyright Act, patents, trademarks, software IP)
- taxation (GST, IGST, income tax, zero rated supply)
- employment (POSH Act, ERA, equal pay, ICC, harassment)

Reply with ONLY the relevant domain names comma separated. Include ALL that apply.
Example: data_protection, taxation, ip_licensing"""

    try:
        raw = _llm("You are a legal domain classifier.", multi_prompt)
        domains = [d.strip() for d in raw.split(",") if d.strip() in VALID_DOMAINS]
        state["all_domains"] = domains if domains else [state["domain"]]
    except Exception:
        state["all_domains"] = [state["domain"]]

    state["trace"] = [
        f"classify → intent={state['intent']} domain={state['domain']} all_domains={state.get('all_domains')} "
        f"domain_source={context.get('domain_source')} domain_agreement={context.get('domain_agreement')} "
        f"domain_confidence={context.get('domain_confidence'):.2f} keyword_domain={context.get('keyword_domain')}"
    ]
    return state

# ── Node 2: retrieve ──────────────────────────────────────────────────────────
def retrieve(state: AgentState) -> AgentState:
    _rag_pipeline._ensure_index()
    q = state.get("rewritten_query") or state["normalized_query"]
    all_domains = state.get("all_domains", [state["domain"]])

    all_docs = []
    seen_ids = set()

    # domain-specific queries for better retrieval
    domain_query_hints = {
        "data_protection": "personal data privacy DPDP consent breach notification",
        "corporate_governance": "company incorporation directors shareholders compliance",
        "ip_licensing": "copyright patent trademark software license",
        "taxation": "GST income tax TDS filing returns",
     
        "employment": "employee hiring salary POSH ICC internal complaints committee Section 4 constitution harassment equal pay",
    }

    for domain in all_domains:
        # augment query with domain-specific terms
        hint = domain_query_hints.get(domain, "")
        domain_q = f"{q} {hint}"
        domain_docs = _rag_pipeline.searcher.search(domain_q, top_k=3)
        for d in domain_docs:
            pid = d.get("provision_id", "")
            if pid not in seen_ids:
                seen_ids.add(pid)
                all_docs.append(d)

    all_docs = sorted(all_docs, key=lambda x: x.get("combined_score", 0), reverse=True)[:7]
    state["retrieved_docs"] = all_docs
    state["trace"].append(f"retrieve → {len(all_docs)} docs across {all_domains}")
    for d in all_docs:
        state["trace"].append(f"    doc → [{d.get('provision_id')}] {d.get('title')} (score={d.get('combined_score')})")
    return state

# ── Node 3: grade context ─────────────────────────────────────────────────────
def grade_context(state: AgentState) -> AgentState:
    docs = state["retrieved_docs"]
    if not docs:
        state["context_grade"]        = "insufficient"
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
        state["context_grade"]        = parsed.get("grade", "insufficient")
        state["context_grade_reason"] = parsed.get("reason", "")
    except Exception:
        state["context_grade"]        = "insufficient"
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
        f"[{d.get('provision_id','')}] {d.get('title','')}: {str(d.get('text','') or d.get('plain_english_summary',''))[:800]}"
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
    state["answer"]    = answer
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
    state["failed_citations"]   = failed
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
        state["groundedness_grade"]  = parsed.get("grade", "not_grounded")
        state["groundedness_reason"] = parsed.get("reason", "")
    except Exception:
        state["groundedness_grade"]  = "not_grounded"
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
        state["answerability_grade"]  = parsed.get("grade", "off_target")
        state["answerability_reason"] = parsed.get("reason", "")
    except Exception:
        state["answerability_grade"]  = "off_target"
        state["answerability_reason"] = "Could not parse grader response"

    state["trace"].append(f"grade_answerability → {state['answerability_grade']}")
    return state

# ── Node: abstain ─────────────────────────────────────────────────────────────
def abstain(state: AgentState) -> AgentState:
    state["abstain"] = True
    if state.get("domain") == "unknown" or state.get("intent") == "unknown":
        reason = "Could not determine the legal domain of your query."
    elif state.get("confidence", 1.0) < 0.5:
        reason = f"Query classification confidence too low ({state.get('confidence')}). Please rephrase with more specific legal context."
    else:
        reason = state.get("context_grade_reason") or state.get("groundedness_reason") or "Insufficient reliable information found."

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