"""
agent/nodes.py - LangGraph nodes for the Legal Workflow Generator.

    classify_query -> domain_pipeline (one worker per domain, in parallel, then stitched)
                   -> verify_citations -> grade_parallel -> stitch_answer   (or abstain)

Changes in this version
  * run-time settings (provider, top_k, as_of ...) arrive through LangGraph `config`   (settings.py)
  * typed models for per-domain results (Step, DomainResult, Provision)
  * every generated step is checked against the provision it cites (claim entailment)
  * proactive rate limiting + parallel-call cap for ALL LLM calls (per provider)
  * quota / availability failures are reported as such, never as "no provisions found"
  * deterministic trace ordering (no node appends to the trace from two threads)
  * temporal awareness: provisions not yet in force are dropped; superseded statutes are flagged
"""
import contextvars
import datetime
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

import psycopg2

try:
    from langchain_core.runnables import RunnableConfig
except Exception:  # pragma: no cover - only if langchain_core is missing
    RunnableConfig = dict

from legal_workflow_generator.agent.settings import RunSettings, current_settings, use_settings
from legal_workflow_generator.agent.llm_limits import LIMITERS, RateLimiter, set_rate_limit  # noqa: F401 (re-exported)
from legal_workflow_generator.agent.gemini_client import RotatingGeminiClient
from legal_workflow_generator.agent.state import AgentState
from legal_workflow_generator.rag.pipeline import RagPipeline
from legal_workflow_generator.query.normalizer import QueryNormalizer
from legal_workflow_generator.query.intent_classifier import IntentClassifier
from legal_workflow_generator.query.context_resolver import LegalContextResolver
from legal_workflow_generator.config.values import GEMINI_MODEL
import legal_workflow_generator.config.values as app_config

logger = logging.getLogger(__name__)

# ── shared clients (init once) ────────────────────────────────────────────────
_normalizer = QueryNormalizer()
_intent_classifier = IntentClassifier()
_context_resolver = LegalContextResolver()
_rag_pipeline = None
_gemini_client = RotatingGeminiClient()

VALID_DOMAINS = ["data_protection", "corporate_governance", "ip_licensing", "taxation", "employment"]

_DOMAIN_ORDER = ["corporate_governance", "data_protection", "ip_licensing", "taxation", "employment"]
_DOMAIN_LABELS = {
    "corporate_governance": "Corporate Governance",
    "data_protection": "Data Protection",
    "ip_licensing": "Intellectual Property",
    "taxation": "Taxation & GST",
    "employment": "Employment & Workplace",
}
APPLIES_TO = ("private", "government", "both", "unspecified")


# ══════════════════════════════════════════════════════════════════════════════
# Typed models
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Provision:
    """A retrieved statutory provision. Typed at the worker boundary; `.get()` and `[]` keep the
    dict-style call sites working. Unknown keys from the searcher are dropped by `from_dict`."""
    provision_id: str
    title: str = ""
    text: str = ""
    full_text: str = ""           # title + sub_structure clauses + text (see _full_text)
    plain_english_summary: str = ""
    statute_id: str = ""
    number: str = ""
    bm25_score: float | None = None
    semantic_score: float | None = None
    combined_score: float | None = None
    anchor: bool = False
    effective_date: Any = None    # datetime.date | None

    _FIELDS = ("provision_id", "title", "text", "full_text", "plain_english_summary", "statute_id",
               "number", "bm25_score", "semantic_score", "combined_score", "anchor", "effective_date")

    _TEXT_FIELDS = ("title", "text", "full_text", "plain_english_summary", "statute_id", "number")

    @classmethod
    def from_dict(cls, d: dict) -> "Provision":
        """Build from a searcher / DB row dict. Unknown keys are ignored; None text fields become ''."""
        kwargs = {k: d[k] for k in cls._FIELDS if k in d}
        for k in cls._TEXT_FIELDS:
            if k in kwargs and kwargs[k] is None:
                kwargs[k] = ""
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self._FIELDS}

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default) if key in self._FIELDS else default

    def __getitem__(self, key: str) -> Any:
        if key not in self._FIELDS:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self._FIELDS


@dataclass
class Step:
    """One action in the checklist, tied to exactly one cited provision."""
    action: str
    provision_id: str
    applies_to: str = "unspecified"
    verdict: str = "unchecked"        # entailed | partial | unchecked   (unsupported steps are dropped)
    verdict_reason: str = ""
    note: str = ""                    # temporal note, e.g. statute superseded

    @classmethod
    def from_llm(cls, raw: Any, valid_ids: dict) -> "Step | None":
        """Validate one step dict from the LLM. `valid_ids` maps lower-case id -> canonical id."""
        if not isinstance(raw, dict):
            return None
        pid = valid_ids.get(str(raw.get("provision_id", "")).strip().lower())
        action = str(raw.get("action", "")).strip()
        if not pid or not action:
            return None
        applies = str(raw.get("applies_to", "unspecified")).strip().lower()
        return cls(action=action, provision_id=pid, applies_to=applies if applies in APPLIES_TO else "unspecified")


@dataclass
class DomainResult:
    """Everything one domain worker produced."""
    domain: str
    docs: list = field(default_factory=list)            # list[Provision]
    grade: str = "insufficient"                         # sufficient | insufficient | error
    reason: str = ""
    retries: int = 0
    summary: list = field(default_factory=list)         # list[str]
    steps: list = field(default_factory=list)           # list[Step]
    dropped_steps: list = field(default_factory=list)   # list[Step] that failed claim verification
    not_covered: str = ""
    cited: list = field(default_factory=list)           # list[str] provision ids
    verification: str = "off"                           # off | ok | skipped (checker unavailable/unparseable)
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["docs"] = [{"provision_id": p.get("provision_id"), "title": p.get("title"),
                      "combined_score": p.get("combined_score")} for p in self.docs]
        return d


# ══════════════════════════════════════════════════════════════════════════════
# LLM access: providers, rate limiting, errors
# ══════════════════════════════════════════════════════════════════════════════
class LLMUnavailable(RuntimeError):
    """The LLM could not be reached (quota / overload) after retries. Distinct from 'no provisions found'."""


def _call_gemini(system: str, user: str) -> str:
    response = _gemini_client.models.generate_content(model=GEMINI_MODEL, contents=f"{system}\n\n{user}")
    return response.text.strip()


_groq_client = None


def _call_groq(system: str, user: str) -> str:
    global _groq_client
    if _groq_client is None:
        from groq import Groq   # pip install groq
        _groq_client = Groq(api_key=getattr(app_config, "GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY"))
    model = getattr(app_config, "GROQ_MODEL", None) or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
    r = _groq_client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()


_PROVIDERS = {"gemini": _call_gemini, "groq": _call_groq}
_RETRYABLE = ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "rate_limit")


def _llm(system: str, user: str, grader: bool = False, max_retries: int = 5) -> str:
    """One LLM call through the provider chosen in the run settings, rate limited, retried on 429/5xx.
    Raises LLMUnavailable when quota/overload persists; other errors propagate unchanged."""
    provider = current_settings().provider
    call, limiter = _PROVIDERS[provider], LIMITERS[provider]
    for attempt in range(max_retries):
        try:
            with limiter.slot():
                return call(system, user)
        except Exception as e:  # noqa
            msg = str(e)
            if "PerDay" in msg or "per day" in msg.lower():
                raise LLMUnavailable(f"{provider} daily quota exhausted") from e
            retryable = any(r in msg for r in _RETRYABLE)
            if retryable and attempt < max_retries - 1:
                limiter.retries += 1
                m = re.search(r"retry in ([\d.]+)s", msg)
                time.sleep(min(float(m.group(1)) + 1 if m else 2 ** attempt + random.random(), 90))
                continue
            if retryable:
                raise LLMUnavailable(f"{provider} unavailable after {max_retries} attempts: {msg[:120]}") from e
            raise


def _submit(ex, fn, *args, **kwargs):
    """Submit to a thread pool WITH the caller's contextvars (so run settings reach the worker threads).
    Each submit needs its own copied context; one Context cannot be entered by two threads at once."""
    ctx = contextvars.copy_context()
    return ex.submit(ctx.run, fn, *args, **kwargs)


def _parse_json(raw: str) -> dict:
    return json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())


# ══════════════════════════════════════════════════════════════════════════════
# Corpus access
# ══════════════════════════════════════════════════════════════════════════════
_DOMAIN_MAP = None

# fallback only, for any provision_id missing from the DB map
_PREFIX_FALLBACK = {
    "dpdp": "data_protection", "information_technology": "data_protection",
    "ca_2013": "corporate_governance", "contract_act": "corporate_governance", "llp_act": "corporate_governance",
    "copyright": "ip_licensing", "patent": "ip_licensing", "trademark": "ip_licensing", "designs": "ip_licensing",
    "cgst": "taxation", "igst": "taxation", "income_tax": "taxation",
    "posh": "employment", "rpwd": "employment", "era_1976": "employment", "eca": "employment",
    "industrial": "employment", "ida_": "employment", "ir_code": "employment",
}


def _connect():
    return psycopg2.connect(dbname=app_config.DB_NAME, user=app_config.DB_USER, password=app_config.PGPASSWORD,
                            host=app_config.DB_HOST, port=app_config.DB_PORT)


def _load_domain_map() -> dict:
    global _DOMAIN_MAP
    if _DOMAIN_MAP is None:
        try:
            conn = _connect()
            cur = conn.cursor()
            cur.execute("SELECT provision_id, domain FROM laws WHERE domain IS NOT NULL")
            _DOMAIN_MAP = {pid: dom for pid, dom in cur.fetchall()}
            cur.close()
            conn.close()
        except Exception:
            _DOMAIN_MAP = {}
    return _DOMAIN_MAP


def _domain_of(pid: str):
    dom = _load_domain_map().get(pid)
    if dom in VALID_DOMAINS:
        return dom
    for prefix, d in _PREFIX_FALLBACK.items():
        if pid.startswith(prefix):
            return d
    return None


# Per-domain query hints were removed: they pulled searches toward one topic of the domain
# (e.g. company-law words pushed the Contract Act out). The domain filter does the steering now.
DOMAIN_QUERY_HINTS = {d: "" for d in VALID_DOMAINS}

# ── anchor provisions ────────────────────────────────────────────────────────
# Stopgap for sections that vector/BM25 search cannot rank (their `text` column is only a header;
# the real wording is in `sub_structure`). When `trigger` (and optional `also`) match the question,
# the listed provisions are always added to that domain. ids are lower-case provision_ids.
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
]


def _full_text(title, text, sub) -> str:
    """The real statute wording lives in `sub_structure` (one entry per clause); `text` is a header + summary."""
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except Exception:
            sub = {}
    clauses = "\n".join(f"({k}) {v}" for k, v in sub.items()) if isinstance(sub, dict) else ""
    return f"{title}\n{clauses}\n{text or ''}".strip()


def _rows_to_docs(rows) -> list:
    return [{"provision_id": r[0], "title": r[1], "text": r[2], "full_text": _full_text(r[1], r[2], r[3]),
             "effective_date": r[4]} for r in rows]


def _fetch_by_ids(ids: list) -> list:
    if not ids:
        return []
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("SELECT provision_id, title, text, sub_structure, effective_date FROM laws "
                    "WHERE lower(provision_id) = ANY(%s)", ([i.lower() for i in ids],))
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning("anchor fetch failed: %s", e)
        return []
    docs = _rows_to_docs(rows)
    for d in docs:
        d["combined_score"], d["anchor"] = 1.0, True
    return docs


def _hydrate(docs: list) -> list:
    """Attach `full_text` and `effective_date` to every retrieved doc that lacks them."""
    need = [d for d in docs if "full_text" not in d and d.get("provision_id")]
    if not need:
        return docs
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("SELECT provision_id, title, text, sub_structure, effective_date FROM laws "
                    "WHERE provision_id = ANY(%s)", ([d["provision_id"] for d in need],))
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
            d = {**d, "full_text": _full_text(r[1], r[2], r[3]), "effective_date": r[4]}
        out.append(d)
    return out


def _with_anchors(domain: str, question: str, docs: list) -> list:
    """Prepend anchor provisions whose trigger matches the question."""
    q_low = question.lower()
    wanted = []
    for a in DOMAIN_ANCHORS:
        if a["domain"] == domain and re.search(a["trigger"], q_low) and re.search(a.get("also", "."), q_low):
            wanted.extend(a["ids"])
    have = {d.get("provision_id", "").lower() for d in docs}
    return _fetch_by_ids([w for w in wanted if w not in have]) + docs


_STOP = {"which", "under", "where", "other", "there", "their", "about", "whether", "being", "would", "could"}


def _snippet(text, query: str, limit: int) -> str:
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


def _get_rag_pipeline():
    # the pipeline's own answer generator is not used by the graph; provider choice applies to _llm
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RagPipeline(llm_provider="gemini")
    return _rag_pipeline


# ══════════════════════════════════════════════════════════════════════════════
# Temporal awareness
# ══════════════════════════════════════════════════════════════════════════════
# Statute-level timeline. Keys are provision_id prefixes. Edit as the law changes; VERIFY each entry.
TEMPORAL_RULES = {
    "income_tax_act_1961": {
        "superseded_on": "2026-04-01",
        "successor": "income_tax_act_2025",
        "note": "The Income-tax Act, 1961 is superseded by the Income-tax Act, 2025 from 1 April 2026; "
                "confirm the corresponding section of the 2025 Act before relying on this step.",
    },
}


def _temporal_note(pid: str, as_of: datetime.date) -> str:
    low = pid.lower()
    for prefix, rule in TEMPORAL_RULES.items():
        if low.startswith(prefix + "_"):
            if rule.get("superseded_on") and as_of >= datetime.date.fromisoformat(rule["superseded_on"]):
                return rule["note"]
            if rule.get("in_force_from") and as_of < datetime.date.fromisoformat(rule["in_force_from"]):
                return rule.get("note") or f"Not in force until {rule['in_force_from']}."
    return ""


def _in_force(doc: dict, as_of: datetime.date) -> bool:
    """False only when the table says the provision commences after `as_of`. Unknown dates are kept."""
    ed = doc.get("effective_date")
    return not (isinstance(ed, datetime.date) and ed > as_of)


# ══════════════════════════════════════════════════════════════════════════════
# Node 1: classify
# ══════════════════════════════════════════════════════════════════════════════
def _detect_domains_from_retrieval(query: str) -> list:
    """Domain-agnostic retrieval; read which statutes appear in the top results (no LLM call)."""
    try:
        rag_pipeline = _get_rag_pipeline()
        rag_pipeline._ensure_index()
        docs = rag_pipeline.searcher.search(query, top_k=7)
        scores = {}
        for doc in docs:
            domain = _domain_of(doc.get("provision_id", ""))
            if domain:
                scores[domain] = scores.get(domain, 0) + (doc.get("combined_score", 0) or 0)
        if not scores:
            return []
        threshold = max(scores.values()) * 0.3
        return [d for d, s in scores.items() if s >= threshold]
    except Exception:
        return []


def _llm_domains(query: str) -> list:
    prompt = f"""This query may span multiple legal domains.
Query: {query}
Domains: data_protection, corporate_governance, ip_licensing, taxation, employment

Reply with ONLY the relevant domain names comma separated. Include ALL that apply.
Example: data_protection, employment"""
    raw = _llm("You are a legal domain classifier.", prompt)      # LLMUnavailable propagates to the caller
    return [d.strip() for d in raw.split(",") if d.strip() in VALID_DOMAINS]


def classify_query(state: AgentState, config: RunnableConfig = None) -> AgentState:
    s = use_settings(config)
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

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_ret = _submit(ex, _detect_domains_from_retrieval, state["normalized_query"])
        f_llm = _submit(ex, _llm_domains, state["query"])
        retrieval_domains = f_ret.result()
        try:
            llm_domains, llm_note = f_llm.result(), ""
        except LLMUnavailable as e:
            llm_domains, llm_note = [], f" [LLM domain detection unavailable: {e}]"
        except Exception:
            llm_domains, llm_note = [], ""

    combined = list(dict.fromkeys(retrieval_domains + llm_domains + [state["domain"]]))
    combined = [d for d in combined if d in VALID_DOMAINS]
    state["all_domains"] = combined if combined else [state["domain"]]
    state["domain_detection_method"] = f"retrieval({len(retrieval_domains)})+llm({len(llm_domains)})"

    state["trace"] = [
        f"classify → intent={state['intent']} domain={state['domain']} "
        f"all_domains={state.get('all_domains')} method={state.get('domain_detection_method')}{llm_note}",
        f"settings → provider={s.provider} top_k={s.top_k} as_of={s.as_of_date.isoformat()} verify_claims={s.verify_claims}",
    ]
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Node 2: parallel per-domain pipeline
#   worker(domain) = focus → retrieve → grade → (rewrite → retrieve → grade)* → generate → verify claims
# Every step inside a worker sees only that domain's focused sub-query (sub_q).
# ══════════════════════════════════════════════════════════════════════════════
def _el(t0: float) -> str:
    """Seconds since this worker started (trace prefix). Disabled by settings.trace_timing=False."""
    return f"+{time.perf_counter() - t0:4.1f}s " if current_settings().trace_timing else ""


def _hybrid_search(rag, query: str, s: RunSettings) -> list:
    try:
        return rag.searcher.search(query, top_k=s.search_depth, bm25_candidates=s.bm25_candidates,
                                   vector_candidates=s.vector_candidates)
    except TypeError:   # older HybridSearcher without vector_candidates
        return rag.searcher.search(query, top_k=s.search_depth, bm25_candidates=s.bm25_candidates)


def _domain_candidates(rag, query: str, domain: str) -> list:
    """Search, then keep every provision that belongs to this domain (no cap yet, best first)."""
    s = current_settings()
    hint = DOMAIN_QUERY_HINTS.get(domain, "")
    docs = _hybrid_search(rag, f"{query} {hint}".strip(), s)
    return [d for d in docs if "_forms_" not in d.get("provision_id", "")
            and _domain_of(d.get("provision_id", "")) == domain]


def _cut(cands: list, s: RunSettings) -> list:
    """Act sections first (at most act_cap), then rules, at most top_k in total."""
    acts = [d for d in cands if "_rules_" not in d.get("provision_id", "")]
    rules = [d for d in cands if "_rules_" in d.get("provision_id", "")]
    return (acts[:min(s.act_cap, s.top_k)] + rules)[:s.top_k]


def _chunks(docs: list, query: str, limit: int) -> str:
    return "\n\n".join(
        f"[{d.get('provision_id', '')}] {d.get('title', '')}: "
        f"{_snippet(d.get('full_text') or d.get('text', '') or d.get('plain_english_summary', ''), query, limit)}"
        for d in docs)


def _grade_domain(domain: str, sub_q: str, docs: list) -> tuple:
    system = "You are a legal relevance grader. Reply in JSON only."
    user = f"""Query ({_DOMAIN_LABELS[domain]} only): {sub_q}

Retrieved provisions:
{_chunks(docs, sub_q, 1500)}

Grade ONLY whether these provisions answer this query.
Ignore any other legal area; do not mention other legal areas in your reason.
All provisions are Indian statutes and the user is an Indian business. A person, contractor or work being
located abroad does NOT make Indian provisions irrelevant; judge whether the provision answers the question
under Indian law.
"sufficient": at least one provision directly addresses the query.
"insufficient": the provisions are unrelated to it.
Reply ONLY with:
{{"grade": "sufficient" or "insufficient", "reason": "one sentence"}}"""
    raw = _llm(system, user, grader=True)                       # LLMUnavailable propagates
    try:
        p = _parse_json(raw)
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
    except LLMUnavailable:
        raise
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
    except LLMUnavailable:
        raise
    except Exception:
        return fallback


def _generate_domain(domain: str, sub_q: str, docs: list) -> tuple:
    """Returns (summary bullets, validated Steps, not_covered). Invented provision ids are dropped."""
    valid_ids = {d.get("provision_id", "").lower(): d.get("provision_id", "") for d in docs if d.get("provision_id")}
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
- If nothing in the provisions answers part of the query, say so in "not_covered" instead of padding the steps.

Provisions:
{_chunks(docs, sub_q, 3000)}

Reply ONLY with JSON:
{{"summary": ["1-2 short bullets"],
 "steps": [{{"action": "...", "provision_id": "<exact id from above>", "applies_to": "private|government|both|unspecified"}}],
 "not_covered": "part of the query these provisions do not address, or empty string"}}"""
    raw = _llm(system, user)                                    # LLMUnavailable propagates
    try:
        p = _parse_json(raw)
    except Exception as e:
        logger.warning("generate[%s] JSON parse failed: %s | raw=%r", domain, e, raw[:500])
        return [], [], ""
    steps = [st for st in (Step.from_llm(s, valid_ids) for s in p.get("steps", []) or []) if st]
    return [str(x) for x in (p.get("summary") or [])][:3], steps, str(p.get("not_covered", "") or "").strip()


_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def _numbers_missing(action: str, text: str) -> list:
    """Deterministic cross-check that the LLM verdict cannot override: every number the step states
    (days, amounts, thresholds, section numbers) must also occur in the cited text."""
    body = text.replace(",", "")
    return [n for n in dict.fromkeys(_NUM.findall(action.replace(",", ""))) if not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", body)]


def _verify_claims(domain: str, steps: list, docs: list) -> tuple:
    """Claim-to-citation entailment: does the cited provision's text actually support each step?
    One LLM call for all steps of the domain. Returns (kept steps, dropped steps, trace message).
    'unsupported' steps are dropped; 'partial' steps are kept and flagged; failures leave steps 'unchecked'."""
    by_id = {d.get("provision_id"): d for d in docs}
    blocks = []
    for i, st in enumerate(steps, 1):
        d = by_id.get(st.provision_id, {})
        text = _snippet(d.get("full_text") or d.get("text", ""), st.action, 1500)
        blocks.append(f"{i}. CLAIM: {st.action}\n   CITED: [{st.provision_id}] {d.get('title', '')}\n   PROVISION TEXT: {text}")
    system = "You are a strict legal fact-checker. Reply in JSON only."
    user = f"""For each numbered claim, decide whether the PROVISION TEXT given for it supports the claim.
Judge ONLY from that text; do not use outside knowledge.
"entailed": the text states or directly implies the whole claim.
"partial": the text supports only part of the claim, or the claim adds detail the text does not contain.
"unsupported": the text does not say this, or contradicts it.

{chr(10).join(blocks)}

Reply ONLY with JSON:
{{"verdicts": [{{"n": 1, "verdict": "entailed" or "partial" or "unsupported", "reason": "short"}}]}}"""
    try:
        raw = _llm(system, user, grader=True)
    except LLMUnavailable:
        return steps, [], "skipped (LLM unavailable)"
    try:
        verdicts = {int(v["n"]): v for v in _parse_json(raw)["verdicts"]}
    except Exception:
        return steps, [], "skipped (unparseable verifier response)"
    kept, dropped = [], []
    for i, st in enumerate(steps, 1):
        v = verdicts.get(i, {})
        verdict = str(v.get("verdict", "")).strip().lower()
        st.verdict = verdict if verdict in ("entailed", "partial", "unsupported") else "unchecked"
        st.verdict_reason = str(v.get("reason", ""))[:200]
        cited = by_id.get(st.provision_id, {})
        missing = _numbers_missing(st.action, cited.get("full_text") or cited.get("text", ""))
        if st.verdict == "entailed" and missing:
            # the LLM said "entailed" but a figure in the step is absent from the text: downgrade, never upgrade
            st.verdict = "partial"
            st.verdict_reason = (st.verdict_reason + "; " if st.verdict_reason else "") + \
                f"number(s) not found in cited text: {', '.join(missing)}"
        (dropped if st.verdict == "unsupported" else kept).append(st)
    n_ent = sum(st.verdict == "entailed" for st in kept)
    n_par = sum(st.verdict == "partial" for st in kept)
    return kept, dropped, f"{n_ent} entailed, {n_par} partial, {len(dropped)} dropped"


def _domain_worker(domain: str, search_q: str, user_q: str, rag, multi: bool = False) -> tuple:
    """One independent mini-pipeline for a single domain. Returns (DomainResult, trace lines)."""
    s = current_settings()
    as_of = s.as_of_date
    trace: list = []
    t0 = time.perf_counter()
    res = DomainResult(domain=domain, reason="No documents retrieved")
    try:
        q, sub_q = search_q, user_q
        if multi:
            q = _focus_query(domain, user_q, search_q)
            sub_q = q
            trace.append(f"{_el(t0)}[{domain}] focus → '{q}'")

        for attempt in range(s.max_domain_retries + 1):
            try:
                cands = _domain_candidates(rag, q, domain)
            except Exception as e:
                cands, res.reason = [], f"Search failed: {e}"
            # effective_date is only known after hydration; filter on it before cutting to top_k so that
            # provisions not in force on as_of do not push valid ones out of the window.
            cands = _hydrate(cands)
            live = [d for d in cands if _in_force(d, as_of)]
            docs = _hydrate(_with_anchors(domain, user_q, _cut(live, s)))
            docs = [d for d in docs if _in_force(d, as_of)]        # anchors are checked as well
            res.docs = [Provision.from_dict(d) for d in docs]     # typed from here on
            trace.append(f"{_el(t0)}[{domain}] retrieve (try {attempt + 1}) → {len(docs)} docs")
            if len(cands) != len(live):
                trace.append(f"{_el(t0)}[{domain}] excluded {len(cands) - len(live)} provision(s) "
                             f"not in force on {as_of}")
            if docs:
                res.grade, res.reason = _grade_domain(domain, sub_q, docs)
                trace.append(f"{_el(t0)}[{domain}] grade → {res.grade}: {res.reason}")
            else:
                res.grade = "insufficient"
            if res.grade == "sufficient" or attempt == s.max_domain_retries:
                break
            q = _rewrite_for_domain(domain, sub_q, res.reason)
            res.retries += 1
            trace.append(f"{_el(t0)}[{domain}] rewrite → '{q}'")

        if res.grade == "sufficient":
            res.summary, steps, res.not_covered = _generate_domain(domain, sub_q, res.docs)
            trace.append(f"{_el(t0)}[{domain}] generate → {len(steps)} steps")
            if steps and s.verify_claims:
                steps, res.dropped_steps, msg = _verify_claims(domain, steps, res.docs)
                res.verification = "skipped" if msg.startswith("skipped") else "ok"
                trace.append(f"{_el(t0)}[{domain}] verify_claims → {msg}")
                if not steps and res.dropped_steps:
                    res.reason = "Every generated step failed verification against the provision it cited."
            for st in steps:
                st.note = _temporal_note(st.provision_id, as_of)
            res.steps = steps
            res.cited = list(dict.fromkeys(st.provision_id for st in steps))
            trace.append(f"{_el(t0)}[{domain}] result → {len(res.steps)} steps, {len(res.cited)} citations")
    except LLMUnavailable as e:
        res.grade, res.error, res.reason = "error", str(e), f"LLM unavailable: {e}"
        trace.append(f"{_el(t0)}[{domain}] ERROR → LLM unavailable: {e}")
    except Exception as e:  # a worker must never take the whole request down
        logger.exception("worker[%s] failed", domain)
        res.grade, res.error, res.reason = "error", f"{type(e).__name__}: {e}", f"internal error: {type(e).__name__}"
        trace.append(f"{_el(t0)}[{domain}] ERROR → {type(e).__name__}: {e}")
    return res, trace


def _stitch_sections(query: str, results: list, as_of: str = "") -> str:
    """Stitching layer: merge DomainResults (in fixed domain order) into one workflow document."""
    bar = "━" * 38
    titles = {d.get("provision_id", ""): d.get("title", "") for r in results for d in r.docs}
    summary, checklist, sources = [], [], []
    n = 0
    for r in results:
        label = _DOMAIN_LABELS[r.domain]
        if not r.steps:
            if r.grade == "error":
                note = f"Could not be assessed ({r.reason}). Please retry."
            elif r.grade == "sufficient":
                note = r.reason if r.dropped_steps else "Provisions were retrieved, but none directly answered this part of the query."
            else:
                note = f"No matching provisions found in the knowledge base. ({r.reason})"
            summary.append(f"- {label}: not addressed — {note}")
            checklist += [f"**{label}**", f"No steps available. {note}", ""]
            continue
        summary += [f"- {label}: {b}" for b in r.summary]
        if r.not_covered:
            summary.append(f"- {label} (not addressed by retrieved provisions): {r.not_covered}")
        checklist.append(f"**{label}**")
        for st in r.steps:
            n += 1
            extra = "" if st.applies_to == "unspecified" else f" (applies to: {st.applies_to})"
            if st.verdict == "partial":
                extra += " [NOTE: only partly supported by the cited text]"
            elif st.verdict == "unchecked" and r.verification == "skipped":
                extra += " [NOTE: not checked against the cited text; the checker was unavailable]"
            if st.note:
                extra += f" [NOTE: {st.note}]"
            checklist.append(f"Step {n}: {st.action} — [{st.provision_id}]{extra}")
        checklist.append("")
        for pid in r.cited:
            line = f"- [{pid}] | {titles.get(pid, '')}"
            if line not in sources:
                sources.append(line)
    return "\n".join([
        "LEGAL COMPLIANCE WORKFLOW", bar, f"Query: {query}",
        *([f"Valid as of: {as_of}"] if as_of else []), "",
        "SUMMARY", *summary, "",
        "ACTION CHECKLIST", *checklist,
        "SOURCES", *sources, "",
        bar,
        "⚠ This is informational guidance, not legal advice.",
        "Consult a qualified legal professional for your specific situation.",
    ])


def domain_pipeline(state: AgentState, config: RunnableConfig = None) -> AgentState:
    use_settings(config)
    rag = _get_rag_pipeline()
    rag._ensure_index()      # build/load everything BEFORE spawning threads
    _load_domain_map()

    wanted = state.get("all_domains") or [state["domain"]]
    domains = [d for d in _DOMAIN_ORDER if d in wanted] or list(wanted)
    search_q = state.get("normalized_query") or state["query"]
    user_q = state["query"]

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, len(domains))) as ex:
        futures = [_submit(ex, _domain_worker, d, search_q, user_q, rag, len(domains) > 1) for d in domains]
        outputs = [f.result() for f in futures]            # submission order = fixed domain order

    results = [r for r, _ in outputs]
    for _, trace in outputs:                               # each worker's trace stays contiguous, in domain order
        state["trace"].extend(trace)

    seen, all_docs = set(), []
    for r in results:
        for d in r.docs:
            pid = d.get("provision_id", "")
            if pid not in seen:
                seen.add(pid)
                all_docs.append(d.to_dict())          # the shared graph state keeps plain dicts
    state["retrieved_docs"] = all_docs
    state["retry_count_retrieval"] = max((r.retries for r in results), default=0)
    state["domain_results"] = [r.to_dict() for r in results]          # typed DomainResult, serialised once here
    state["as_of"] = s_as_of = current_settings().as_of_date.isoformat()

    covered = [r for r in results if r.steps]
    if covered:
        state["context_grade"] = "sufficient"
        state["context_grade_reason"] = "; ".join(f"{r.domain}: {r.grade}" for r in results)
        state["answer"] = _stitch_sections(state["query"], results, as_of=s_as_of)
        state["citations"] = list(dict.fromkeys(c for r in covered for c in r.cited))
    else:
        state["context_grade"] = "insufficient"
        state["context_grade_reason"] = "; ".join(f"{r.domain}: {r.reason}" for r in results) \
            or "No domain returned usable provisions"
        state["answer"] = ""
        state["citations"] = []

    elapsed = f" in {time.perf_counter() - t_start:.1f}s wall-clock (parallel)" if current_settings().trace_timing else ""
    state["trace"].append(
        f"domain_pipeline → {len(covered)}/{len(results)} domains answered "
        f"({', '.join(r.domain for r in covered) or 'none'}){elapsed}")
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Node 5: verify citation ids
# ══════════════════════════════════════════════════════════════════════════════
def verify_citations(state: AgentState) -> AgentState:
    doc_ids = {d.get("provision_id", "") for d in state["retrieved_docs"]}
    verified, failed = [], []
    for cit in state["citations"]:
        (verified if cit in doc_ids else failed).append(cit)
    state["verified_citations"] = verified
    state["failed_citations"] = failed
    state["trace"].append(f"verify_citations → verified={verified} failed={failed}")
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Nodes 6+7: graders. Pure helpers return (state updates, trace line); the node merges them in fixed order,
# so the trace never depends on thread timing.
# ══════════════════════════════════════════════════════════════════════════════
def _groundedness(state: AgentState) -> tuple:
    chunks = "\n\n".join(f"[{d.get('provision_id', '')}]: {(d.get('full_text') or d.get('text', ''))[:800]}"
                         for d in state["retrieved_docs"])
    multi = len(state.get("all_domains", [state["domain"]])) > 1
    system = """You are a legal grounding verifier. Reply in JSON only.
Be lenient: grade as 'grounded' if the main claims are supported by context.
Only grade 'not_grounded' if the answer makes specific legal claims that
directly contradict or are completely absent from the retrieved context.
Minor elaborations and reasonable inferences are acceptable.
Statements saying that no provision was found for a topic are NOT legal claims; ignore them.
Text in square brackets starting with 'NOTE:' is a system annotation, not a legal claim; ignore it."""
    user = f"""Answer: {state['answer'][:3000]}

Retrieved context:
{chunks}

{"Note: This is a multi-domain query. The answer may draw on multiple legal areas — grade as grounded if each domain's claims are supported by at least some of the retrieved chunks." if multi else ""}

Is every claim in the answer supported by the retrieved context?
Reply ONLY with:
{{"grade": "grounded" or "not_grounded", "reason": "one sentence"}}"""
    try:
        raw = _llm(system, user, grader=True)
    except LLMUnavailable as e:
        return ({"groundedness_grade": "unverified", "groundedness_reason": f"LLM unavailable: {e}"},
                "grade_groundedness → unverified (LLM unavailable)")
    try:
        p = _parse_json(raw)
        grade, reason = p.get("grade", "not_grounded"), p.get("reason", "")
    except Exception:
        grade, reason = "not_grounded", "Could not parse grader response"
    return {"groundedness_grade": grade, "groundedness_reason": reason}, f"grade_groundedness → {grade}"


def _answerability(state: AgentState) -> tuple:
    """Computed in code, no LLM: answers = every queried domain has a verified cited step."""
    all_domains = state.get("all_domains") or [state["domain"]]
    verified = state.get("verified_citations", [])
    covered = [d for d in all_domains if any(_domain_of(c) == d for c in verified)]
    missing = [d for d in all_domains if d not in covered]
    if not covered:
        grade, reason = "off_target", "No queried legal area was covered by verified provisions."
    elif missing:
        grade, reason = "partial", f"Covered: {', '.join(covered)}. Not covered: {', '.join(missing)}."
    else:
        grade, reason = "answers", f"All queried areas covered: {', '.join(covered)}."
    return {"answerability_grade": grade, "answerability_reason": reason}, f"grade_answerability → {grade}: {reason}"


def grade_groundedness(state: AgentState) -> AgentState:      # kept for callers that use the old name
    upd, line = _groundedness(state)
    state.update(upd)
    state["trace"].append(line)
    return state


def grade_answerability(state: AgentState) -> AgentState:
    upd, line = _answerability(state)
    state.update(upd)
    state["trace"].append(line)
    return state


def grade_parallel(state: AgentState, config: RunnableConfig = None) -> AgentState:
    use_settings(config)
    g_upd, g_line = _groundedness(state)      # the only LLM call; answerability is pure code
    a_upd, a_line = _answerability(state)
    state.update(g_upd)
    state.update(a_upd)
    state["trace"].extend([g_line, a_line])   # fixed order
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Abstain + final stitch
# ══════════════════════════════════════════════════════════════════════════════
def abstain(state: AgentState) -> AgentState:
    state["abstain"] = True
    if state.get("domain") == "unknown" or state.get("intent") == "unknown":
        reason = "Could not determine the legal domain of your query."
    elif state.get("confidence", 1.0) < 0.5:
        reason = (f"Query classification confidence too low ({state.get('confidence')}). "
                  f"Please rephrase with more specific legal context.")
    else:
        if state.get("groundedness_grade") == "not_grounded":
            reason = state.get("groundedness_reason") or "Answer was not grounded in the retrieved provisions."
        elif state.get("answerability_grade") == "off_target":
            reason = state.get("answerability_reason") or "Answer did not address the question."
        else:
            reason = state.get("context_grade_reason") or "Insufficient reliable information found."
    state["answer"] = (f"I was unable to provide a reliable answer to your question: '{state['query']}'\n\n"
                       f"Reason: {reason}\n\n"
                       f"Please consult a qualified legal professional or refer directly to the relevant Act.")
    state["abstain_reason"] = reason
    state["trace"].append(f"abstain → triggered: {reason}")
    return state


def stitch_answer(state: AgentState) -> AgentState:
    all_domains = state.get("all_domains", [state["domain"]])
    answer = state.get("answer", "")
    verified = state.get("verified_citations", [])
    gap_domains = [d for d in _DOMAIN_ORDER if d in all_domains
                   and not any(_domain_of(c) == d for c in verified)]
    gaps = [f"- {_DOMAIN_LABELS.get(d, d)}: insufficient information found in knowledge base" for d in gap_domains]
    state["domain_gaps"] = gaps
    state["gap_domains"] = gap_domains

    extra = ""
    if state.get("groundedness_grade") == "unverified":
        extra = ("\n\n⚠ The automated groundedness check could not run (LLM unavailable); "
                 "this answer has not been checked for unsupported claims.")
    if not gaps:
        state["stitched_answer"] = state["answer"] = answer + extra
        state["trace"].append("stitch → no gaps, answer passed through")
        return state
    state["stitched_answer"] = state["answer"] = f"""{answer}{extra}

⚠ KNOWLEDGE GAPS IDENTIFIED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The following compliance areas could not be fully addressed 
due to insufficient information in the current knowledge base.
Please consult a legal professional for these areas:

{chr(10).join(gaps)}"""
    state["trace"].append(f"stitch → {len(gaps)} gap(s) flagged: {gap_domains}")
    return state