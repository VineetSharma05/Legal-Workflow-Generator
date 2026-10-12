from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import json
import re

import psycopg2

from google import genai as _genai

from legal_workflow_generator.agent.state import AgentState
from legal_workflow_generator.rag.pipeline import RagPipeline
from legal_workflow_generator.query.normalizer import QueryNormalizer
from legal_workflow_generator.query.intent_classifier import IntentClassifier
from legal_workflow_generator.query.context_resolver import LegalContextResolver
from legal_workflow_generator.config.values import GEMINI_API_KEY, GEMINI_MODEL
import legal_workflow_generator.config.values as config

logger = logging.getLogger(__name__)

# ── shared clients (init once) ────────────────────────────────────────────────
_normalizer = QueryNormalizer()
_intent_classifier = IntentClassifier()
_context_resolver = LegalContextResolver()
_rag_pipeline: RagPipeline | None = None
_gemini_client = _genai.Client(api_key=GEMINI_API_KEY)

VALID_DOMAINS = ["data_protection", "corporate_governance", "ip_licensing", "taxation", "employment"]

# Generator rates each step 1-5; steps below this are dropped (noise filter).
MIN_STEP_RELEVANCE = 3

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
    "eca": "employment",
    "industrial": "employment",
    "ida_": "employment",
    "ir_code": "employment",
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


# FIX: taxation hint no longer forces GST; employment hint added.
DOMAIN_QUERY_HINTS = {
    "data_protection": "",
    "corporate_governance": "",
    "ip_licensing": "",
    # taxation / employment: no fixed bias; the focused sub-query + anchors steer retrieval
    "taxation": "",
    "employment": "",
}


# ── ANCHOR PROVISIONS ─────────────────────────────────────────────────────────
# Vector search ranks some key sections too low (e.g. s.195 behind s.194E).
# When a trigger matches the user's question, these provisions are always added
# to that domain's retrieved docs. ids are lower-case provision_ids.
DOMAIN_ANCHORS = [
    {"domain": "taxation",
     "trigger": r"non[- ]?resident|abroad|overseas|foreign (contractor|freelanc|vendor|consultant|developer|company|supplier|agency|agent|firm|studio|designer)"
                r"|germany|\busa\b|\bu\.s\.|(in|from|to|with) (the )?us\b|united states|\buk\b|singapore|europe|japan|canada|australia|dubai|\buae\b",
     "also": r"pay|fee|invoice|deduct|tds|withhold|remit",
     "ids": ["income_tax_act_1961_sec_195", "income_tax_act_1961_sec_9"]},
    {"domain": "employment",
     "trigger": r"workman|workmen|retrench|lay-?off|dismiss|terminat|freelanc|contractor",
     "ids": ["ida_1947_sec_2"]},
    {"domain": "ip_licensing",
     "trigger": r"who owns|ownership|owns the|freelanc|contractor",
     "ids": ["copyright_act_1957_sec_17", "copyright_act_1957_sec_19"]},
    # add the Income-tax Act 2025 equivalent of s.195 here once you know its id
]


def _fetch_by_ids(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    try:
        conn = psycopg2.connect(
            dbname=config.DB_NAME, user=config.DB_USER, password=config.PGPASSWORD,
            host=config.DB_HOST, port=config.DB_PORT,
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT provision_id, title, text, sub_structure FROM laws WHERE lower(provision_id) = ANY(%s)",
            ([i.lower() for i in ids],),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning("anchor fetch failed: %s", e)
        return []
    return [
        {"provision_id": r[0], "title": r[1], "text": r[2], "full_text": _full_text(r[1], r[2], r[3]),
         "combined_score": 1.0, "anchor": True}
        for r in rows
    ]


def _full_text(title, text, sub) -> str:
    """The real statute wording lives in `sub_structure` (one entry per sub-section / clause);
    the `text` column only holds a short header + summary. Rebuild the full provision."""
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except Exception:
            sub = {}
    clauses = "\n".join(f"({k}) {v}" for k, v in sub.items()) if isinstance(sub, dict) else ""
    return f"{title}\n{clauses}\n{text or ''}".strip()


def _hydrate(docs: list) -> list:
    """Attach `full_text` (incl. sub_structure) to every retrieved doc that lacks it."""
    need = [d for d in docs if "full_text" not in d and d.get("provision_id")]
    if not need:
        return docs
    try:
        conn = psycopg2.connect(
            dbname=config.DB_NAME, user=config.DB_USER, password=config.PGPASSWORD,
            host=config.DB_HOST, port=config.DB_PORT,
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT provision_id, title, text, sub_structure FROM laws WHERE provision_id = ANY(%s)",
            ([d["provision_id"] for d in need],),
        )
        rows = {r[0]: r for r in cur.fetchall()}
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning("hydrate failed: %s", e)
        return docs
    out = []
    for d in docs:
        if "full_text" not in d and d.get("provision_id") in rows:
            r = rows[d["provision_id"]]
            d = {**d, "full_text": _full_text(r[1], r[2], r[3])}
        out.append(d)
    return out


def _with_anchors(domain: str, question: str, docs: list) -> list:
    """Prepend anchor provisions whose trigger matches the question."""
    wanted: list[str] = []
    q_low = question.lower()
    for a in DOMAIN_ANCHORS:
        if (a["domain"] == domain and re.search(a["trigger"], q_low)
                and re.search(a.get("also", "."), q_low)):
            wanted.extend(a["ids"])
    wanted = list(dict.fromkeys(w.lower() for w in wanted))      # dedupe
    have = {d.get("provision_id", "").lower() for d in docs}
    extra = _fetch_by_ids([w for w in wanted if w not in have])
    return extra + docs


_STOP = {"which", "under", "where", "other", "there", "their", "about", "whether", "being", "would", "could"}


def _snippet(text: str, query: str, limit: int) -> str:
    """Up to `limit` chars of text, centred on the rarest query keyword found in it."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    low = text.lower()
    words = {w.lower() for w in re.findall(r"[A-Za-z]{5,}", query)} - _STOP
    counts = {w: low.count(w) for w in words if low.count(w) > 0}
    if not counts:
        return text[:limit]
    best = min(counts, key=lambda w: (counts[w], -len(w)))
    start = max(0, low.find(best) - limit // 4)
    return ("…" if start else "") + text[start:start + limit]


def _get_rag_pipeline() -> RagPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RagPipeline(llm_provider="gemini")
    return _rag_pipeline


def _llm(system: str, user: str, grader: bool = False, max_retries: int = 5) -> str:
    retryable = ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded")
    for attempt in range(max_retries):
        try:
            response = _gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{system}\n\n{user}",
            )
            return response.text.strip()
        except Exception as e:
            if any(r in str(e) for r in retryable) and attempt < max_retries - 1:
                m = re.search(r"retry in ([\d.]+)s", str(e))
                wait = float(m.group(1)) + 1 if m else 2 ** attempt
                time.sleep(min(wait, 90))
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
Domains: data_protection, corporate_governance, ip_licensing, taxation, employment

Reply with ONLY the relevant domain names comma separated. Include ALL that apply.
Example: data_protection, employment"""
        raw = _llm("You are a legal domain classifier.", prompt)
        return [d.strip() for d in raw.split(",") if d.strip() in VALID_DOMAINS]
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


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY single-pass nodes (NOT used by graph.py any more; kept so any other
# module that imports them does not break). Safe to delete if unused.
# ══════════════════════════════════════════════════════════════════════════════
def retrieve(state: AgentState) -> AgentState:
    rag_pipeline = _get_rag_pipeline()
    rag_pipeline._ensure_index()
    q = state.get("rewritten_query") or state["normalized_query"]
    all_domains = state.get("all_domains") or [state["domain"]]

    def retrieve_for_domain(domain: str):
        hint = DOMAIN_QUERY_HINTS.get(domain, "")
        domain_q = f"{q} {hint}".strip()
        try:
            return domain, rag_pipeline.searcher.search(domain_q, top_k=3)
        except Exception:
            return domain, []

    per_domain: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(5, len(all_domains)))) as executor:
        futures = [executor.submit(retrieve_for_domain, d) for d in all_domains]
        for future in as_completed(futures):
            domain, docs = future.result()
            per_domain[domain] = docs

    score = lambda x: x.get("combined_score", 0) or 0
    selected, seen_ids = [], set()

    for domain in all_domains:
        for d in sorted(per_domain.get(domain, []), key=score, reverse=True):
            pid = d.get("provision_id", "")
            if pid not in seen_ids:
                seen_ids.add(pid)
                selected.append(d)
                break

    rest = [d for docs in per_domain.values() for d in docs]
    for d in sorted(rest, key=score, reverse=True):
        if len(selected) >= 7:
            break
        pid = d.get("provision_id", "")
        if pid not in seen_ids:
            seen_ids.add(pid)
            selected.append(d)

    all_docs = sorted(selected, key=score, reverse=True)

    state["retrieved_docs"] = all_docs
    state["trace"].append(f"retrieve → {len(all_docs)} docs across {all_domains} (parallel)")
    for d in all_docs:
        state["trace"].append(
            f"    doc → [{d.get('provision_id')}] {d.get('title')} (score={d.get('combined_score')})"
        )
    return state


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
    state["trace"].append(f"rewrite_query → retry {state['retry_count_retrieval']}: '{new_q}'")
    return state


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
# FIX: the grader now sees the real statute wording (full_text from sub_structure),
# not the short `text` header, and only the provisions the answer actually cites.
def grade_groundedness(state: AgentState) -> AgentState:
    cited = set(state.get("citations") or [])
    docs = [d for d in state["retrieved_docs"] if d.get("provision_id") in cited] or state["retrieved_docs"]
    chunks = "\n\n".join(
        f"[{d.get('provision_id','')}]: "
        f"{_snippet(d.get('full_text') or d.get('text',''), state['query'], 1200)}"
        for d in docs
    )
    all_domains = state.get("all_domains", [state["domain"]])
    is_multi = len(all_domains) > 1

    system = """You are a legal grounding verifier. Reply in JSON only.
Be lenient: grade as 'grounded' if the main claims are supported by context.
Only grade 'not_grounded' if the answer makes specific legal claims that
directly contradict or are completely absent from the retrieved context.
Minor elaborations and reasonable inferences are acceptable.
Statements saying that no provision was found for a topic are NOT legal claims;
ignore them when grading."""

    user = f"""Answer: {state['answer'][:3000]}

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


# ── Node 7: grade answerability (FIX: computed in code, no LLM) ──────────────
# answers   = every queried domain has at least one verified, cited step
# partial   = some domains covered, some are knowledge gaps
# off_target = no domain covered
def grade_answerability(state: AgentState) -> AgentState:
    all_domains = state.get("all_domains") or [state["domain"]]
    verified = state.get("verified_citations", [])

    covered = [d for d in all_domains if any(_domain_of(c) == d for c in verified)]
    missing = [d for d in all_domains if d not in covered]

    if not covered:
        state["answerability_grade"] = "off_target"
        state["answerability_reason"] = "No queried legal area was covered by verified provisions."
    elif missing:
        state["answerability_grade"] = "partial"
        state["answerability_reason"] = (
            f"Covered: {', '.join(covered)}. Not covered: {', '.join(missing)}."
        )
    else:
        state["answerability_grade"] = "answers"
        state["answerability_reason"] = f"All queried areas covered: {', '.join(covered)}."

    state["trace"].append(
        f"grade_answerability → {state['answerability_grade']}: {state['answerability_reason']}"
    )
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

    covered_domains = []
    gap_domains = []

    for domain in _DOMAIN_ORDER:
        if domain not in all_domains:
            continue
        domain_cites = [c for c in verified if _domain_of(c) == domain]
        if domain_cites:
            covered_domains.append(domain)
        else:
            gap_domains.append(domain)

    gaps = [
        f"- {_DOMAIN_LABELS.get(domain, domain)}: insufficient information found in knowledge base"
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
#   worker(domain) = focus → retrieve → grade → (rewrite → retrieve → grade)*
#                    → generate
#
# FIX: every step inside a worker now sees the FOCUSED sub-query (sub_q),
# never the full multi-domain query, so graders/rewriters/generators cannot
# drift into other domains.
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
    "taxation": "Taxation",          # was "Taxation & GST" (nudged the LLM toward GST)
    "employment": "Employment & Workplace",
}

MAX_DOMAIN_RETRIES = 1   # rewrite attempts per domain (up to 2 retrievals)
DOCS_PER_DOMAIN = 8


def _el(t0: float) -> str:
    """Seconds since this worker started, for trace lines."""
    return f"+{time.perf_counter() - t0:4.1f}s"


def _parse_json(raw: str) -> dict:
    return json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())


def _search_domain(rag, query: str, domain: str) -> list:
    """Search, then keep only provisions that belong to this domain."""
    hint = DOMAIN_QUERY_HINTS.get(domain, "")
    docs = rag.searcher.search(f"{query} {hint}".strip(), top_k=100,
                               bm25_candidates=300, vector_candidates=100)
    keep = [
        d for d in docs
        if "_forms_" not in d.get("provision_id", "")
        and _domain_of(d.get("provision_id", "")) == domain
    ]
    # statute sections carry the substantive obligations; rules mostly carry procedure.
    # take Act sections first, then fill the remaining slots with rules.
    acts = [d for d in keep if "_rules_" not in d.get("provision_id", "")]
    rules = [d for d in keep if "_rules_" in d.get("provision_id", "")]
    return (acts[:6] + rules)[:DOCS_PER_DOMAIN]


def _grade_domain(domain: str, sub_q: str, docs: list) -> tuple[str, str]:
    chunks = "\n\n".join(
        f"[{d.get('provision_id','')}] {d.get('title','')}: {_snippet(d.get('full_text') or d.get('text',''), sub_q, 1500)}"
        for d in docs
    )
    system = "You are a legal relevance grader. Reply in JSON only."
    user = f"""Query ({_DOMAIN_LABELS[domain]} only): {sub_q}

Retrieved provisions:
{chunks}

Grade ONLY whether these provisions answer this query.
Ignore any other legal area; do not mention other legal areas in your reason.
All provisions are Indian statutes and the user is an Indian business. A person, contractor or work being
located abroad does NOT make Indian provisions irrelevant; judge whether the provision answers the question
under Indian law.
"sufficient": at least one provision directly addresses the query.
"insufficient": the provisions are unrelated to it.
Reply ONLY with:
{{"grade": "sufficient" or "insufficient", "reason": "one sentence"}}"""
    try:
        p = _parse_json(_llm(system, user, grader=True))
        return p.get("grade", "insufficient"), p.get("reason", "")
    except Exception:
        return "insufficient", "Could not parse grader response"


def _rewrite_for_domain(domain: str, sub_q: str, reason: str) -> str:
    system = "You are a legal search query optimizer."
    user = f"""Query ({_DOMAIN_LABELS[domain]} only): {sub_q}
Why retrieval failed: {reason}

Write a better search query for Indian statutes about THIS query only.
Stay strictly on the topic of the query above; do not switch to any other legal area.
Search Indian statutes only: never use foreign-law names, other countries' laws or non-English words.
Use plain topic words. Do NOT cite section numbers you are not certain exist.
Reply with ONLY the query."""
    try:
        return _llm(system, user).strip() or sub_q
    except Exception:
        return sub_q


def _focus_query(domain: str, user_q: str, fallback: str) -> str:
    """Multi-domain queries: extract only this domain's part as a standalone search query."""
    system = "You are a legal search query writer."
    user = f"""Question: {user_q}

Rewrite this question as a short standalone search query covering ONLY the {_DOMAIN_LABELS[domain]} aspects.
Rules:
- Keep the concrete nouns from the question.
- Use only words and concepts that appear in the question. Do NOT add new legal concepts
  (for example, do not add GST, TDS or any statute unless the question mentions it).
- Drop everything that belongs to other legal areas.
- If the question uses an abbreviation, write the full words statutes use (PF = provident fund, ESI = employees state insurance, TDS = tax deducted at source, GST = goods and services tax, DPDP = digital personal data protection, LLP = limited liability partnership).
Reply with ONLY the query."""
    try:
        return _llm(system, user).strip() or fallback
    except Exception:
        return fallback


def _generate_domain(domain: str, sub_q: str, docs: list) -> dict:
    ids = {d.get("provision_id", "") for d in docs}
    chunks = "\n\n".join(
        f"[{d.get('provision_id','')}] {d.get('title','')}: "
        f"{_snippet(d.get('full_text') or d.get('text','') or d.get('plain_english_summary',''), sub_q, 3000)}"
        for d in docs
    )
    system = "You are a legal workflow assistant for Indian tech startups."
    user = f"""Query: {sub_q}
Legal area: {_DOMAIN_LABELS[domain]}

Using ONLY the provisions below, list the compliance actions relevant to this legal area for the query.
Do not invent sections or obligations. Address specifics in the query (company size, employee type, etc.).
For each step, say who the provision applies to ("private", "government", "both" or "unspecified"), based only on the text.

Rules:
- Discuss ONLY {_DOMAIN_LABELS[domain]}. Do not mention other legal areas (tax, data protection, etc.) in the summary.
- Include a step only if the provision directly responds to the query. Skip provisions that concern other kinds of
  workplaces (factories, mines, plantations, schools) or are unrelated to the specific question.
- If a provision states a default rule that answers the question (for example who is the first owner of a work,
  or what the default is when a contract is silent), INCLUDE it as a step and state both the default and its exception.
- A provision must answer the actual question asked. Do NOT include one merely because it shares a word with the
  query (for example "abroad" or "contractor") if it regulates a different subject (such as accident compensation
  or migrant-worker conditions). Leave it out.
- For each step add "relevance" (1-5): 5 = the provision directly answers the question for this business;
  3 = applies, but only partly or conditionally; 1-2 = shares a keyword or concerns a different situation
  than the one described (e.g. a different kind of worker, workplace or product).
- If nothing in the provisions answers part of the query, say so in "not_covered" instead of padding the steps.

Provisions:
{chunks}

Reply ONLY with JSON:
{{"summary": ["1-2 short bullets"],
 "steps": [{{"action": "...", "provision_id": "<exact id from above>", "applies_to": "private|government|both|unspecified", "relevance": 1-5}}],
 "not_covered": "part of the query these provisions do not address, or empty string"}}"""

    out = {"summary": [], "steps": [], "not_covered": "", "cited": [], "dropped": []}
    raw = ""
    try:
        raw = _llm(system, user)
        p = _parse_json(raw)
    except Exception as e:
        logger.warning("generate[%s] JSON parse failed: %s | raw=%r", domain, e, raw[:500])
        return out

    steps, dropped = [], []
    for s in p.get("steps", []):
        pid = str(s.get("provision_id", "")).strip()
        action = str(s.get("action", "")).strip()
        if pid not in ids or not action:          # drops invented / unknown provision ids
            continue
        try:
            rel = int(s.get("relevance", MIN_STEP_RELEVANCE))
        except (TypeError, ValueError):
            rel = MIN_STEP_RELEVANCE
        if rel < MIN_STEP_RELEVANCE:              # low-relevance (noise) step
            dropped.append(f"{pid}({rel})")
            continue
        steps.append({
            "action": action,
            "provision_id": pid,
            "applies_to": str(s.get("applies_to", "unspecified")).lower(),
            "relevance": rel,
        })

    out["summary"] = [str(x) for x in p.get("summary", [])][:3]
    out["steps"] = steps
    out["dropped"] = dropped
    out["not_covered"] = str(p.get("not_covered", "") or "").strip()
    out["cited"] = list(dict.fromkeys(s["provision_id"] for s in steps))
    return out


def _domain_worker(domain: str, search_q: str, user_q: str, rag, multi: bool = False) -> tuple[dict, list]:
    """One independent mini-pipeline for a single domain. Returns (result, trace)."""
    trace = []
    t0 = time.perf_counter()

    q = search_q          # what we search with
    sub_q = user_q        # what grader / rewriter / generator see
    if multi:
        q = _focus_query(domain, user_q, search_q)
        sub_q = q
        trace.append(f"{_el(t0)} [{domain}] focus → '{q}'")

    docs, grade, reason = [], "insufficient", "No documents retrieved"
    retries = 0

    for attempt in range(MAX_DOMAIN_RETRIES + 1):
        try:
            docs = _search_domain(rag, q, domain)
        except Exception as e:
            docs, reason = [], f"Search failed: {e}"
        docs = _hydrate(_with_anchors(domain, user_q, docs))
        trace.append(f"{_el(t0)} [{domain}] retrieve (try {attempt + 1}) → {len(docs)} docs")

        if docs:
            grade, reason = _grade_domain(domain, sub_q, docs)
            trace.append(f"{_el(t0)} [{domain}] grade → {grade}: {reason}")
        else:
            grade = "insufficient"

        if grade == "sufficient" or attempt == MAX_DOMAIN_RETRIES:
            break

        q = _rewrite_for_domain(domain, sub_q, reason)
        retries += 1
        trace.append(f"{_el(t0)} [{domain}] rewrite → '{q}'")

    result = {
        "domain": domain, "docs": docs, "grade": grade, "reason": reason,
        "retries": retries, "summary": [], "steps": [], "not_covered": "", "cited": [], "dropped": [],
    }
    if grade == "sufficient":
        result.update(_generate_domain(domain, sub_q, docs))
        trace.append(
            f"{_el(t0)} [{domain}] generate → {len(result['steps'])} steps, "
            f"{len(result['cited'])} citations, dropped={result.get('dropped', [])}"
        )
    return result, trace


def _stitch_sections(query: str, results: list) -> str:
    """Stitching layer: merge per-domain results into one workflow document.
    FIX: domains with no usable provisions now get their own labelled section."""
    bar = "━" * 38
    titles = {d.get("provision_id", ""): d.get("title", "") for r in results for d in r["docs"]}

    summary, checklist, sources = [], [], []
    n = 0
    for r in results:
        label = _DOMAIN_LABELS[r["domain"]]

        if not r["steps"]:
            if r["grade"] == "sufficient":
                note = "Provisions were retrieved, but none directly answered this part of the query."
            else:
                note = f"No matching provisions found in the knowledge base. ({r['reason']})"
            summary.append(f"- {label}: not addressed — {note}")
            checklist.append(f"**{label}**")
            checklist.append(f"No steps available. {note}")
            checklist.append("")
            continue

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