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
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any, TypedDict

import psycopg2
from pydantic import BaseModel
from typing import TypeVar

try:
    from langchain_core.runnables import RunnableConfig
except Exception:  # pragma: no cover - only if langchain_core is missing
    RunnableConfig = dict

from legal_workflow_generator.agent.settings import RunSettings, current_settings, use_settings
from legal_workflow_generator.agent.llm_limits import LIMITERS, RateLimiter, set_rate_limit  # noqa: F401 (re-exported)
from legal_workflow_generator.agent import groq_client
from legal_workflow_generator.agent.gemini_client import RotatingGeminiClient
from legal_workflow_generator.agent.state import AgentState
from legal_workflow_generator.rag.pipeline import RagPipeline
from legal_workflow_generator.query.normalizer import QueryNormalizer
from legal_workflow_generator.query.intent_classifier import IntentClassifier
from legal_workflow_generator.query.context_resolver import LegalContextResolver
from legal_workflow_generator.config.values import GEMINI_MODEL
import legal_workflow_generator.config.values as app_config
from legal_workflow_generator.llm import StructuredOutputError, invoke_structured, structured_llm
from legal_workflow_generator.llm.schemas import (
    AnswerabilityGrade,
    DomainList,
    GroundednessGrade,
    RelevanceGrade,
    SearchQuery,
    domain_workflow_model,
)

T = TypeVar("T", bound=BaseModel)

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
    key_ids: list = field(default_factory=list)         # select_context: provisions the grader marked as needed
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


def _call_gemini(system: str, user: str, schema: type[T] | None = None):
    if schema is not None:
        return invoke_structured(structured_llm(schema, provider="gemini"), system, user)
    response = _gemini_client.models.generate_content(model=GEMINI_MODEL, contents=f"{system}\n\n{user}")
    return response.text.strip()


def _call_groq(system: str, user: str, schema: type[T] | None = None):
    if schema is not None:
        return invoke_structured(structured_llm(schema, provider="groq"), system, user)
    return groq_client.complete(system, user)


# provider -> call(system, user, schema=None): a schema instance when `schema` is given, else plain text.
_PROVIDERS = {"gemini": _call_gemini, "groq": _call_groq}
_RETRYABLE = ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "rate_limit")


def _llm(schema: type[T] | str, system: str, user: str | None = None, grader: bool = False,
         max_retries: int = 5) -> T | str:
    """The one LLM boundary: routed to the run's provider, rate limited, retried on 429/5xx.

    Structured:  _llm(Schema, system, user) -> Schema instance (raises StructuredOutputError on a bad reply).
    Plain text:  _llm(system, user)         -> str (query planning and the claim verifier still parse JSON).
    """
    if isinstance(schema, str):                       # plain-text form: shift the arguments
        schema, system, user = None, schema, system
    provider = current_settings().provider
    call, limiter = _PROVIDERS[provider], LIMITERS[provider]

    def once():
        with limiter.slot():
            if schema is None:
                return call(system, user or "")
            return call(system, user or "", schema=schema)
    return _with_retries(provider, once, max_retries)


# Number of LLM calls that finally failed (after retries) in this process. Some stages deliberately carry on
# after a failure (the answer is marked "unchecked" / "unverified"); the evaluation reads this counter to tell a
# degraded run apart from a real result, and re-runs such queries instead of scoring them.
LLM_FAILURES = {"count": 0, "last": ""}
_FAIL_LOCK = threading.Lock()


def _note_failure(msg: str) -> None:
    with _FAIL_LOCK:
        LLM_FAILURES["count"] += 1
        LLM_FAILURES["last"] = msg[:300]


def _with_retries(provider: str, fn, max_retries: int = 5):
    """Run fn(), retrying on 429/5xx with backoff. Raises LLMUnavailable when quota/overload persists."""
    try:
        return _with_retries_inner(provider, fn, max_retries)
    except Exception as e:  # noqa: BLE001 - counted, then re-raised unchanged
        _note_failure(f"{type(e).__name__}: {e}")
        raise


def _with_retries_inner(provider: str, fn, max_retries: int):
    limiter = LIMITERS[provider]
    for attempt in range(max_retries):
        try:
            return fn()
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


def _full_text(title, text, sub, raw=None) -> str:
    """The statute as the LLM steps read it, in statute order: title, opening words / sub-section (1), clauses.

    `raw` is the published wording of the opening part (laws.raw_text). Sub-section (1) usually lives only there,
    not in `sub_structure`, so it goes first. Without `raw` (database ingested before raw_text existed) this falls
    back to `text`, the normalised search blob, after the clauses."""
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except Exception:
            sub = {}
    clauses = "\n".join(f"({k}) {v}" for k, v in sub.items()) if isinstance(sub, dict) else ""
    if raw and str(raw).strip():
        return f"{title}\n{str(raw).strip()}\n{clauses}".strip()
    return f"{title}\n{clauses}\n{text or ''}".strip()


def _rows_to_docs(rows) -> list:
    return [{"provision_id": r[0], "title": r[1], "text": r[2],
             "full_text": _full_text(r[1], r[2], r[3], r[5] if len(r) > 5 else None),
             "effective_date": r[4]} for r in rows]


_HAS_RAW_TEXT = None


def _select_laws(where: str, params: tuple) -> list:
    """SELECT provision_id, title, text, sub_structure, effective_date[, raw_text] FROM laws WHERE <where>.
    raw_text is left out on a database ingested before that column existed."""
    global _HAS_RAW_TEXT
    conn = _connect()
    try:
        cur = conn.cursor()
        if _HAS_RAW_TEXT is None:
            cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name = 'laws' AND column_name = 'raw_text'")
            _HAS_RAW_TEXT = cur.fetchone() is not None
            if not _HAS_RAW_TEXT:
                logger.warning("laws.raw_text missing: re-run `python main.py ingest complete` so the LLM reads "
                               "the original statute wording")
        cols = "provision_id, title, text, sub_structure, effective_date" + (", raw_text" if _HAS_RAW_TEXT else "")
        cur.execute(f"SELECT {cols} FROM laws WHERE {where}", params)
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        conn.close()


def _fetch_by_ids(ids: list) -> list:
    if not ids:
        return []
    try:
        rows = _select_laws("lower(provision_id) = ANY(%s)", ([i.lower() for i in ids],))
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
        rows = {r[0]: r for r in _select_laws("provision_id = ANY(%s)", ([d["provision_id"] for d in need],))}
    except Exception as e:
        logger.warning("hydrate failed: %s", e)
        return docs
    out = []
    for d in docs:
        if "full_text" not in d and d.get("provision_id") in rows:
            r = rows[d["provision_id"]]
            d = {**d, "full_text": _full_text(r[1], r[2], r[3], r[5] if len(r) > 5 else None),
                 "effective_date": r[4]}
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


def _with_suggested(ids: list, docs: list) -> list:
    """Append the planner's suggested provisions that retrieval did not already bring in."""
    have = {d.get("provision_id", "").lower() for d in docs}
    extra = _fetch_by_ids([i for i in ids if i.lower() not in have])
    rank = {i.lower(): n for n, i in enumerate(ids)}
    extra.sort(key=lambda d: rank.get(d["provision_id"].lower(), 99))
    for d in extra:
        d["anchor"], d["combined_score"] = False, None
    return docs + extra


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


def _commencement_note(doc, as_of: datetime.date) -> str:
    """For a provision that commences after `as_of` (e.g. most DPDP Act duties, from 13 May 2027)."""
    if _in_force(doc, as_of):
        return ""
    return f"Not yet in force on {as_of.isoformat()}; applies from {doc.get('effective_date').isoformat()}."


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

List every legal domain the query touches. Include ALL that apply."""
    try:
        result = _llm(DomainList, "You are a legal domain classifier.", prompt)
        return list(dict.fromkeys(d for d in result.domains if d in VALID_DOMAINS))
    except Exception:
        return []


_SECTION_REF = re.compile(r"\b(?:section|sec\.?|s\.)\s*(\d+[a-z]*)\b", re.I)
_YEAR = re.compile(r"\b(20\d\d)\b")


def _scope_check(query: str, as_of: datetime.date) -> str:
    """Deterministic out-of-scope checks that run before any LLM call. Returns a reason, or '' if in scope.

    * the query cites a section number that does not exist in the statute it names (e.g. "section 999 of the
      DPDP Act"): answering would mean citing a provision that does not exist;
    * the query asks about the law in a year after as_of + 1: statutes cannot tell what a future law will say.
      (as_of + 1 stays in scope: phased statutes such as the DPDP Act commence then.)"""
    refs = _SECTION_REF.findall(query)
    if refs:
        named = [p for pattern, prefixes in STATUTE_ALIASES if re.search(pattern, query.lower()) for p in prefixes]
        if named:
            ids = {i for d in _catalog().values() for i in d.get("ids", {})}
            for num in refs:
                tail = re.compile(rf"_(?:sec|rule|reg|dir)_{re.escape(num.lower())}$")
                if ids and not any(i.startswith(tuple(named)) and tail.search(i) for i in ids):
                    return (f"The query refers to section {num} of a statute in which the knowledge base has no "
                            f"such provision.")
    years = [int(y) for y in _YEAR.findall(query)]
    if years and max(years) > as_of.year + 1:
        return (f"The query asks about the law in {max(years)}. The knowledge base holds statutes as in force on "
                f"{as_of.isoformat()} and cannot predict future changes to the law.")
    return ""


def classify_query(state: AgentState, config: RunnableConfig = None) -> AgentState:
    s = use_settings(config)
    scope = _scope_check(state["query"], s.as_of_date)
    if scope:
        state.update(intent="unknown", domain="unknown", all_domains=[], tentative_domains=[],
                     normalized_query=state["query"], keywords=[], confidence=0.0, llm_error="",
                     scope_error=scope, retry_count_retrieval=0, retry_count_generation=0,
                     abstain=False, abstain_reason="")
        state["trace"] = [f"classify → out of scope (no LLM call): {scope}"]
        return state
    state["scope_error"] = ""
    normalized = _normalizer.normalize(text=state["query"])
    try:
        intent, confidence = _with_retries(s.provider, lambda: _intent_classifier.classify(
            normalized, provider=s.provider, raise_errors=True))
        context = _with_retries(s.provider, lambda: _context_resolver.resolve(
            normalized, intent, confidence, provider=s.provider, raise_errors=True))
    except Exception as e:  # noqa  (LLMUnavailable, bad model name, missing key, ...)
        # The LLM could not be reached: say so instead of abstaining with "please rephrase".
        state.update(intent="unknown", domain="unknown", all_domains=[], tentative_domains=[],
                     normalized_query=normalized.get("normalized", "") if isinstance(normalized, dict) else str(normalized),
                     keywords=[], confidence=0.0,
                     retry_count_retrieval=0, retry_count_generation=0, abstain=False, abstain_reason="",
                     llm_error=f"{s.provider}: {str(e)[:200]}")
        state["trace"] = [f"classify → LLM unavailable ({state['llm_error']})"]
        return state
    state["llm_error"] = ""

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
        except Exception as e:  # noqa: BLE001
            llm_domains, llm_note = [], f" [LLM domain detection failed: {type(e).__name__}]"

    combined = list(dict.fromkeys(retrieval_domains + llm_domains + [state["domain"]]))
    combined = [d for d in combined if d in VALID_DOMAINS]
    state["all_domains"] = combined if combined else [state["domain"]]
    # Domains that only the LLM suggested are unconfirmed: they still run, but are dropped from the
    # answer (instead of shown as "not addressed") when their retrieval turns out to be irrelevant.
    state["tentative_domains"] = [d for d in llm_domains
                                  if d not in retrieval_domains and d != state["domain"]]
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


def _hybrid_search(rag, query: str, s: RunSettings, domain: str | None = None) -> list:
    extra = {} if s.fusion == "linear" else {"fusion": s.fusion}     # linear = the searcher's default
    if domain and s.domain_search == "domain":
        return rag.searcher.search(query, top_k=s.search_depth, bm25_candidates=s.bm25_candidates,
                                   vector_candidates=s.vector_candidates, domain=domain, **extra)
    try:
        return rag.searcher.search(query, top_k=s.search_depth, bm25_candidates=s.bm25_candidates,
                                   vector_candidates=s.vector_candidates, **extra)
    except TypeError:   # older HybridSearcher without vector_candidates
        return rag.searcher.search(query, top_k=s.search_depth, bm25_candidates=s.bm25_candidates)


# "Machinery" provisions: they set up a statute or its regulator but put no duty on the user. They mention
# many topics at once (e.g. DPDP s.40 lists every rule the government may make on notice, consent, breach ...),
# so they outrank the section that actually states the duty. They are moved behind substantive provisions,
# unless the question itself is about rules, the regulator or commencement.
_MACHINERY_TITLE = re.compile(
    r"short title|power(s)? (of [\w ]+ )?to make (rules|regulations)|laying of (rules|regulations)"
    r"|power to (remove difficulties|amend (the )?schedule)|repeal and saving|bar of jurisdiction"
    r"|protection of action taken in good faith|to be public servants|salary, allowances|resignation by"
    r"|officers and employees of|proceedings of (the )?board", re.I)
_MACHINERY_QUERY = re.compile(r"\brules?\b|regulations?\b|\bboard\b|tribunal|commencement|in force|short title", re.I)


def _demote_machinery(docs: list, query: str) -> list:
    if _MACHINERY_QUERY.search(query):
        return docs
    keep = [d for d in docs if not _MACHINERY_TITLE.search(d.get("title") or "")]
    return keep + [d for d in docs if _MACHINERY_TITLE.search(d.get("title") or "")]


# ── named-statute search ─────────────────────────────────────────────────────
# When a question names a statute ("under the SPDI rules", "FEMA", "Income Tax Act 2025"), the worker also
# searches inside that statute alone and fuses the result with the open search. A large neighbouring statute
# otherwise fills the candidate pool (the Income-tax Acts alone hold 1,338 of 4,888 provisions). Patterns match
# names and standard abbreviations only, never subject words, so a question that names no statute is unaffected.
STATUTE_ALIASES = [
    (r"\bdpdp\b|digital personal data protection", ["dpdp_act_2023_", "dpdp_rules_2025_"]),
    (r"\bdpdp rules\b|data protection rules", ["dpdp_rules_2025_"]),
    (r"\bspdi\b|sensitive personal data or information", ["spdi_rules_2011_"]),
    (r"\bcert[- ]?in\b", ["cert_in_directions_2022_"]),
    (r"\bit act\b|information technology act", ["information_technology_act_2000_", "it_act_2000_"]),
    (r"intermediary (guidelines|rules)", ["it_intermediary_guidelines_2021_"]),
    (r"\baadhaar\b", ["aadhaar_act_2016_"]),
    (r"\bcicra\b|credit information compan", ["cicra_2005_"]),
    (r"\brti\b|right to information", ["rti_act_2005_"]),
    (r"telecommunications act|telecom cyber", ["telecommunications_act_2023_", "telecom_cyber_security_rules_2024_"]),
    (r"companies act", ["ca_2013_"]),
    (r"\bllp\b|limited liability partnership", ["llp_act_2008_"]),
    (r"contract act", ["contract_act_1872_"]),
    (r"\bibc\b|insolvency (and bankruptcy )?code", ["ibc_2016_"]),
    (r"consumer protection act", ["consumer_protection_act_2019_"]),
    (r"copyright act", ["copyright_act_1957_"]),
    (r"patents? act", ["patents_act_1970_"]),
    (r"designs act", ["designs_act_2000_"]),
    (r"trade ?marks? act", ["trademarks_act_1999_"]),
    (r"\bcgst\b|central goods and services tax", ["cgst_act_2017_", "cgst_rules_2017_"]),
    (r"\bigst\b|integrated goods and services tax", ["igst_act_2017_"]),
    (r"\bgst\b|goods and services tax", ["cgst_act_2017_", "cgst_rules_2017_", "igst_act_2017_"]),
    (r"income[- ]tax act,? 2025", ["income_tax_act_2025_"]),
    (r"income[- ]tax act,? 1961", ["income_tax_act_1961_"]),
    (r"\bfema\b|foreign exchange management", ["fema_transfer_issue_foreign_security_regulations_2004_"]),
    (r"\bsez\b|special economic zone", ["sez_act_2005_"]),
    (r"stamp (act|duty)", ["indian_stamp_act_1899_"]),
    (r"\bepf\b|provident fund", ["epf_1952_"]),
    (r"\besi\b|\besic\b|employees'? state insurance", ["esia_1948_", "esi_act_1948_"]),
    (r"\bposh\b|sexual harassment of women at workplace", ["posh_2013_"]),
    (r"gratuity act", ["poga_1972_"]),
    (r"bonus act", ["poba_1965_"]),
    (r"maternity benefit act", ["mba_1961_"]),
    (r"apprentices act", ["app_1961_"]),
    (r"industrial disputes act", ["ida_1947_"]),
    (r"factories act", ["fa_1948_"]),
    (r"code on social security", ["css_2020_"]),
    (r"\boshwc?\b|occupational safety,? health", ["oshwc_2020_"]),
    (r"\brpwd\b|rights of persons with disabilities", ["rpwd_2016_"]),
    (r"contract labour act", ["cla_1970_"]),
    (r"payment of wages act", ["pow_1936_"]),
]


def _named_statutes(text: str, domain: str) -> list:
    """Provision-id prefixes of statutes the text names that belong to `domain` (empty if none)."""
    low = text.lower()
    hits = [p for pattern, prefixes in STATUTE_ALIASES if re.search(pattern, low) for p in prefixes]
    if not hits:
        return []
    own = _catalog().get(domain, {}).get("prefixes", {})
    return list(dict.fromkeys(p for p in hits if any(cp.startswith(p) for cp in own)))


def _domain_candidates(rag, query: str, domain: str, prefixes=None) -> list:
    """Search, then keep every provision that belongs to this domain (no cap yet, best first).
    With `prefixes`, the search runs inside those statutes only."""
    s = current_settings()
    hint = DOMAIN_QUERY_HINTS.get(domain, "")
    if prefixes:
        extra = {} if s.fusion == "linear" else {"fusion": s.fusion}
        docs = rag.searcher.search(query, top_k=s.search_depth, bm25_candidates=s.bm25_candidates,
                                   vector_candidates=s.vector_candidates, domain=domain, prefixes=prefixes,
                                   **extra)
    else:
        docs = _hybrid_search(rag, f"{query} {hint}".strip(), s, domain)
    docs = [d for d in docs if "_forms_" not in d.get("provision_id", "")
            and _domain_of(d.get("provision_id", "")) == domain]
    return _demote_machinery(docs, query)


def _cut(cands: list, s: RunSettings) -> list:
    """Act sections first (at most act_cap), then rules, at most top_k in total."""
    acts = [d for d in cands if "_rules_" not in d.get("provision_id", "")]
    rules = [d for d in cands if "_rules_" in d.get("provision_id", "")]
    return (acts[:min(s.act_cap, s.top_k)] + rules)[:s.top_k]


def _chunks(docs: list, query: str, limit: int) -> str:
    as_of = current_settings().as_of_date
    return "\n\n".join(
        f"[{d.get('provision_id', '')}] {d.get('title', '')}"
        f"{' [NOT YET IN FORCE: ' + _commencement_note(d, as_of) + ']' if not _in_force(d, as_of) else ''}: "
        f"{_snippet(d.get('full_text') or d.get('text', '') or d.get('plain_english_summary', ''), query, limit)}"
        for d in docs)


def _grade_domain(domain: str, sub_q: str, docs: list) -> tuple:
    system = "You are a legal relevance grader."
    chunks = _chunks(docs, sub_q, 1500)
    user = f"""Query ({_DOMAIN_LABELS[domain]} only): {sub_q}

Retrieved provisions:
{chunks}

Grade ONLY whether these provisions answer this query.
Ignore any other legal area; do not mention other legal areas in your reason.
All provisions are Indian statutes and the user is an Indian business. A person, contractor or work being
located abroad does NOT make Indian provisions irrelevant; judge whether the provision answers the question
under Indian law.
"sufficient": at least one provision directly addresses the query.
"insufficient": the provisions are unrelated to it."""
    try:
        result = _llm(RelevanceGrade, system, user)
        return result.grade, result.reason
    except StructuredOutputError:
        return "insufficient", "Could not parse grader response"
    except Exception:
        raise


SELECT_MIN = 5      # if the grader picks fewer, the pool's best-ranked provisions fill the context up to this size


def _grade_select(domain: str, sub_q: str, pool: list) -> tuple:
    """Grade + select in one call (settings.select_context). Returns (grade, reason, picked ids in order).

    The grader sees a wider candidate pool than the generator would (short snippets) and names the provisions a
    complete answer must cite. This recovers provisions that retrieval ranks 9th-20th (the generator used to see
    only the top 8) and gives the generator a tighter, ordered context. Picks are restricted to pool ids."""
    s = current_settings()
    system = "You are a legal relevance grader. Reply in JSON only."
    user = f"""Query ({_DOMAIN_LABELS[domain]} only): {sub_q}

Candidate provisions:
{_chunks(pool, sub_q, 700)}

Ignore any other legal area; do not mention other legal areas in your reason.
All provisions are Indian statutes and the user is an Indian business. A person, contractor or work being
located abroad does NOT make Indian provisions irrelevant; judge whether the provision answers the question
under Indian law.

1. "grade": "sufficient" if at least one candidate directly addresses the query, "insufficient" if they are
   all unrelated to it. If the query asks about the obligations under a foreign law (for example the EU GDPR or a
   US statute) for a business that Indian law does not reach, the grade is "insufficient": Indian provisions on a
   similar topic do not answer a question about a foreign law.
2. "relevant": the identifiers of the candidates that a thorough lawyer answering this query would cite, most
   important first, at most {s.select_max}. Be complete rather than minimal: a provision that answers any part
   of the query belongs in the list. Include:
   - the provision that itself creates the duty, right, condition, prohibition, liability or consequence asked
     about (for example the section that imposes the tax, requires the contribution or makes the agreement void);
   - the rule, regulation or direction that gives its procedure, timeline, rate or form, if one is listed;
   - for each separate part of the query, the provision that answers that part;
   - a definition only if the query asks what something means or who is covered.
   Leave out provisions that merely share words with the query, that concern a different kind of entity or
   workplace, or that only give the government or a regulator a power.

Reply ONLY with:
{{"grade": "sufficient" or "insufficient", "reason": "one sentence", "relevant": ["<identifier>", ...]}}"""
    raw = _llm(system, user, grader=True)                       # LLMUnavailable propagates
    try:
        p = _parse_json(raw)
    except Exception:
        return "insufficient", "Could not parse grader response", []
    valid = {d.get("provision_id", "").lower(): d.get("provision_id", "") for d in pool}
    picked = []
    for r in p.get("relevant") or []:
        pid = valid.get(re.sub(r"[\s\[\]\"'`]", "", str(r)).lower())
        if pid and pid not in picked:
            picked.append(pid)
    return p.get("grade", "insufficient"), p.get("reason", ""), picked[:s.select_max]


def _keep_selected(pool: list, picked: list, suggested: list, window: list = None) -> list:
    """Generator context under select_context: the grader's picks first (in its order), then the ordinary top_k
    window, the planner's suggestions and the anchor provisions. Picks only ever ADD to the ordinary context
    (on the dev set, replacing the window with the picks lost as many gold provisions as it gained).
    `window`: ids of the ordinary top_k cut; defaults to the top_k cut of the pool's retrieved provisions."""
    s = current_settings()
    by_id = {d.get("provision_id"): d for d in pool}
    sug = {i.lower() for i in suggested}
    if window is None:
        window = [d.get("provision_id") for d in _cut([d for d in pool if not d.get("anchor")
                                                         and d.get("provision_id", "").lower() not in sug], s)]
    order = list(picked) + list(window)
    order += [d["provision_id"] for d in pool if d.get("anchor") or d.get("provision_id", "").lower() in sug]
    for d in pool:
        if len(dict.fromkeys(order)) >= SELECT_MIN:
            break
        order.append(d.get("provision_id"))
    return [by_id[p] for p in dict.fromkeys(order) if p in by_id]


def _rewrite_for_domain(domain: str, sub_q: str, reason: str) -> str:
    system = "You are a legal search query optimizer."
    user = f"""Query ({_DOMAIN_LABELS[domain]} only): {sub_q}
Why retrieval failed: {reason}

Write a better search query for Indian statutes about THIS query only.
Stay strictly on the topic of the query above; do not switch to any other legal area.
Search Indian statutes only: never use foreign-law names, other countries' laws or non-English words.
Use plain topic words. Do NOT cite section numbers you are not certain exist."""
    try:
        return _llm(SearchQuery, system, user).query.strip() or sub_q
    except LLMUnavailable:
        raise
    except Exception:
        return sub_q


MAX_ASPECTS = 4

# ── LLM-suggested provisions ─────────────────────────────────────────────────
# The planner call also names the provisions the model expects to answer the question, written as corpus
# identifiers. Only identifiers that exist in this domain of the corpus are kept; they join the context next to
# the retrieved provisions and then pass the same grading, generation, identifier gate and entailment check.
# A suggestion is therefore never trusted: a wrong or invented section either does not exist (dropped here) or
# does not support the step (dropped by the claim check). This recovers central provisions that broad questions
# rank low (e.g. the GST levy or the company incorporation section), which the LLM recalls from memory.
_ID_PREFIX = re.compile(r"^(.*?_(?:sec|rule|reg|dir|sch|schedule|annex))_", re.I)
_CATALOG = None
_CATALOG_LOCK = threading.Lock()


def _statute_name(statute_id: str) -> str:
    return " ".join(w.upper() if len(w) <= 4 and not w.isdigit() and w not in ("act", "code") else w.capitalize()
                    for w in statute_id.split("_"))


def _catalog() -> dict:
    """{domain: {"lines": [catalog text lines], "ids": {lower id: canonical id}}} built once from the DB."""
    global _CATALOG
    with _CATALOG_LOCK:
        if _CATALOG is not None:
            return _CATALOG
        cat = {d: {"prefixes": {}, "ids": {}} for d in VALID_DOMAINS}
        try:
            conn = _connect()
            cur = conn.cursor()
            cur.execute("SELECT provision_id, statute_id, domain FROM laws WHERE domain IS NOT NULL")
            rows = cur.fetchall()
            cur.close()
            conn.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("catalog load failed: %s", e)
            rows = []
        for pid, statute, dom in rows:
            if dom not in cat or "_forms_" in pid:
                continue
            cat[dom]["ids"][pid.lower()] = pid
            m = _ID_PREFIX.match(pid)
            if m:
                cat[dom]["prefixes"].setdefault(m.group(1) + "_", statute or "")
        for d in cat.values():
            d["lines"] = [f"- {p}<number>  ({_statute_name(st)})" for p, st in sorted(d["prefixes"].items())]
        _CATALOG = cat
        return cat


def _resolve_suggestions(domain: str, raw: list) -> list:
    """Map the planner's suggested identifiers to existing provision ids of this domain (order kept)."""
    ids = _catalog()[domain]["ids"]
    out = []
    for r in raw or []:
        cand = re.sub(r"[\s\[\]\"'`]", "", str(r)).strip(".").lower()
        if not cand:
            continue
        tries = [cand, re.sub(r"\(.*$", "", cand), re.sub(r"_\d+[a-z]?$", "", cand)]
        # the IT Act is split across two prefixes (data protection vs IP sections)
        tries += [t.replace("information_technology_act_2000_", "it_act_2000_") for t in tries[:2]]
        tries += [t.replace("it_act_2000_", "information_technology_act_2000_") for t in tries[:2]]
        for t in tries:
            if t in ids:
                out.append(ids[t])
                break
    return list(dict.fromkeys(out))


def _plan_queries(domain: str, user_q: str, fallback: str) -> tuple:
    """One LLM call per domain. Returns (focus query, aspect queries, suggested provision ids).

    focus   : the question narrowed to this domain (used as the search query for multi-domain questions).
    aspects : for broad questions ("what is the DPDP Act and how do I follow it"), up to MAX_ASPECTS short
              queries, one per topic a complete answer needs (duties, notice, consent, breach reporting ...).
              A single vague query ranks definitions and rule-making powers above the sections that hold
              the actual obligations; searching each topic separately brings those sections up.
    Aspects only steer retrieval; every step is still generated from, and checked against, retrieved text.
    suggested: when settings.suggest_provisions is on, up to settings.suggest_cap corpus ids the model expects to
              answer the question (see _resolve_suggestions); [] otherwise."""
    label = _DOMAIN_LABELS[domain]
    s = current_settings()
    suggest = s.suggest_provisions and s.suggest_cap > 0
    suggest_block, suggest_json = "", ""
    if suggest:
        catalog = "\n".join(_catalog()[domain]["lines"])
        suggest_block = f"""
"provisions": list up to {s.suggest_cap} {label} provisions of Indian law that most directly answer this question,
most important first, written as identifiers in this format (prefix + section, rule or regulation number,
for example cgst_act_2017_sec_16 or ca_2013_sec_135). Use only these prefixes:
{catalog}
Name only provisions you are confident exist and apply. For a question with no {label} part use [].
"""
        suggest_json = ', "provisions": ["..."]'
    system = "You are a legal search query writer."
    user = f"""Question: {user_q}

Write search queries for finding the Indian statute provisions that answer this question.

"focus": rewrite the question as a short standalone search query covering ONLY the {label} aspects.
- Keep the concrete nouns from the question.
- Use only words and concepts that appear in the question. Do NOT add new legal concepts
  (for example, do not add GST, TDS or any statute unless the question mentions it).
- Drop everything that belongs to other legal areas.
- If the question uses an abbreviation, write the full words statutes use (PF = provident fund, ESI = employees state insurance, TDS = tax deducted at source, GST = goods and services tax, DPDP = digital personal data protection, LLP = limited liability partnership).

"aspects": ONLY if the question is broad (for example "what is <Act> and how do I comply with it") or has several
{label} parts, list up to {MAX_ASPECTS} short search queries. Each names ONE specific {label} topic that a complete
answer must cover (for example a duty, a right, a notice or consent requirement, a reporting deadline, a penalty).
Use the terms Indian statutes use. Never cite section numbers. For a narrow question use [].
Do NOT put the name of the Act in the aspects (write "notice to data principal before collecting personal data",
not "DPDP Act notice requirements"): the Act's name matches every section of it and hides the one on the topic.

{suggest_block}
Reply ONLY with JSON: {{"focus": "...", "aspects": ["...", "..."]{suggest_json}}}"""
    try:
        raw = _llm(system, user)
    except LLMUnavailable:
        raise
    except Exception:
        return fallback, [], []
    suggested = []
    try:
        p = _parse_json(raw)
        focus = str(p.get("focus") or "").strip() or fallback
        aspects = [str(a).strip() for a in (p.get("aspects") or []) if str(a).strip()]
        if suggest and isinstance(p.get("provisions"), list):
            suggested = _resolve_suggestions(domain, p["provisions"])[:s.suggest_cap]
    except Exception:                       # plain-text reply: treat it as the focus query
        focus, aspects = (raw.strip().splitlines() or [fallback])[0].strip() or fallback, []
    aspects = [a for a in dict.fromkeys(aspects) if a.lower() != focus.lower()][:MAX_ASPECTS]
    return focus, aspects, suggested


# Reciprocal rank fusion over several searches. Shallow (top RRF_DEPTH of each list, small k) so that
# a provision ranked high by one aspect query, or found by several, beats one that is only mid-ranked
# everywhere. Tested on "what is dpdp act and how can i make sure to follow it": with a single query
# DPDP s.8 (general obligations) ranked 60th; fused, it ranks 3rd.
RRF_K, RRF_DEPTH = 2, 6


def _fused_candidates(rag, queries: list, domain: str, named_from: str = "") -> list:
    """`named_from`: text (usually the user's question) scanned for statute names; when it names a statute of
    this domain, one more list (the first query searched inside that statute) joins the fusion."""
    lists = [_domain_candidates(rag, q, domain) for q in queries]
    named = _named_statutes(named_from, domain) if current_settings().statute_search and named_from else []
    if named:
        lists.append(_domain_candidates(rag, queries[0], domain, prefixes=named))
    if len(lists) == 1:
        return lists[0]
    first, score = {}, {}
    for docs in lists:
        for rank, d in enumerate(docs):
            pid = d.get("provision_id", "")
            first.setdefault(pid, d)
            if rank < RRF_DEPTH:
                score[pid] = score.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
    head = sorted(score, key=lambda p: -score[p])
    return [first[p] for p in head] + [d for p, d in first.items() if p not in score]


def _generate_domain(domain: str, sub_q: str, docs: list, key_ids: list = None) -> tuple:
    """Returns (summary bullets, validated Steps, not_covered). Invented provision ids are dropped.
    `key_ids` (select_context): provisions the grader marked as needed; the generator is asked to write a step
    for each one its text supports. Every step still goes through the claim check."""
    valid_ids = {d.get("provision_id", "").lower(): d.get("provision_id", "") for d in docs if d.get("provision_id")}
    key_rule = ""
    if key_ids:
        key_rule = (f"\n- A reviewer marked these provisions as needed for a complete answer: {', '.join(key_ids)}. "
                    "Write at least one step for each of them, stating what that provision itself requires, allows "
                    "or provides for this query. Leave one out only if its text, as given below, does not apply "
                    "to the query.")
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
- A provision that only gives the government or a regulator a power (to make rules, appoint or pay members,
  run its own proceedings) is not a duty of the user. Do not turn it into a step.
- Definitions may be used to say who a duty applies to, but prefer the provisions that state the duties themselves.
- Cite the provision that itself creates the duty, right, condition or consequence asked about. When rules,
  regulations or directions given below spell out how to carry out a duty created by an Act section that is also
  given below, write one step citing the Act section (the duty) and another citing the rule (the procedure,
  timeline or form). Do not cite only the rule.
- When one provision states a general rule and another covers the specific situation in the query, cite the
  specific one, and also the general one if the answer depends on it.
- If several provisions each answer a different part of the query, write a step for each of them.{key_rule}
- If the query asks what a law is, base the summary on what the provisions say about its scope and purpose.
- Provisions marked NOT YET IN FORCE still count: use them when they answer the query, because the user has to
  prepare for them. Do not write commencement dates into the action text; the system adds them to each step.
- If nothing in the provisions answers part of the query, say so in "not_covered" instead of padding the steps.

Provisions:
{_chunks(docs, sub_q, 3000)}

"""
    try:
        p = _llm(domain_workflow_model(list(valid_ids.values())), system, user)
    except Exception as e:
        logger.warning("generate[%s] structured generation failed: %s", domain, e)
        return [], [], ""
    # The schema's provision_id enum already rejects unknown ids; check again in case a provider skips validation.
    steps = [
        Step(action=s.action.strip(), provision_id=valid_ids[str(s.provision_id).lower()], applies_to=s.applies_to)
        for s in p.steps if s.action.strip() and str(s.provision_id).lower() in valid_ids
    ]
    return [str(x) for x in p.summary if str(x).strip()][:3], steps, str(p.not_covered or "").strip()


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
        focus, aspects, suggested = _plan_queries(domain, user_q, search_q)
        if multi:
            q = sub_q = focus
            trace.append(f"{_el(t0)}[{domain}] focus → '{q}'")
        if aspects:
            trace.append(f"{_el(t0)}[{domain}] aspects → {aspects}")
        if suggested:
            trace.append(f"{_el(t0)}[{domain}] suggested → {suggested}")

        for attempt in range(s.max_domain_retries + 1):
            try:
                cands = _fused_candidates(rag, [q] + aspects, domain, named_from=user_q)
            except Exception as e:
                cands, res.reason = [], f"Search failed: {e}"
            # effective_date is only known after hydration; filter on it before cutting to top_k so that
            # provisions not in force on as_of do not push valid ones out of the window.
            cands = _hydrate(cands)
            exclude = s.future_provisions == "exclude"
            live = [d for d in cands if _in_force(d, as_of)] if exclude else cands
            window = _cut(live, s)
            window_ids = [d.get("provision_id") for d in window]
            if s.select_context:            # wider pool for the grader to choose from
                pool_n = max(s.select_pool, s.top_k)
                window = _cut(live, replace(s, top_k=pool_n, act_cap=max(s.act_cap, pool_n - 4)))
            docs = _hydrate(_with_suggested(suggested, _with_anchors(domain, user_q, window)))
            if exclude:
                docs = [d for d in docs if _in_force(d, as_of)]    # anchors are checked as well
            res.docs = [Provision.from_dict(d) for d in docs]     # typed from here on
            trace.append(f"{_el(t0)}[{domain}] retrieve (try {attempt + 1}) → {len(docs)} docs")
            if len(cands) != len(live):
                trace.append(f"{_el(t0)}[{domain}] excluded {len(cands) - len(live)} provision(s) "
                             f"not in force on {as_of}")
            n_future = sum(not _in_force(d, as_of) for d in docs)
            if n_future:
                trace.append(f"{_el(t0)}[{domain}] {n_future} provision(s) not yet in force on {as_of} "
                             f"(kept, steps flagged with their start date)")
            if docs and s.select_context:
                res.grade, res.reason, picked = _grade_select(domain, sub_q, docs)
                docs = _keep_selected(docs, picked, suggested, window_ids)
                res.docs = [Provision.from_dict(d) for d in docs]
                res.key_ids = picked
                trace.append(f"{_el(t0)}[{domain}] grade → {res.grade}: {res.reason}")
                trace.append(f"{_el(t0)}[{domain}] select → {picked} ({len(docs)} docs kept)")
            elif docs:
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
            res.summary, steps, res.not_covered = _generate_domain(domain, sub_q, res.docs, res.key_ids)
            trace.append(f"{_el(t0)}[{domain}] generate → {len(steps)} steps")
            if steps and s.verify_claims:
                steps, res.dropped_steps, msg = _verify_claims(domain, steps, res.docs)
                res.verification = "skipped" if msg.startswith("skipped") else "ok"
                trace.append(f"{_el(t0)}[{domain}] verify_claims → {msg}")
                if not steps and res.dropped_steps:
                    res.reason = "Every generated step failed verification against the provision it cited."
            by_id = {d.get("provision_id"): d for d in res.docs}
            for st in steps:
                notes = (_temporal_note(st.provision_id, as_of),
                         _commencement_note(by_id.get(st.provision_id, {}), as_of))
                st.note = " ".join(n for n in notes if n)
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
        if any("Not yet in force" in st.note for st in r.steps):
            summary.append(f"- {label}: some of these obligations are not yet in force"
                           f"{' on ' + as_of if as_of else ''}; each step below shows the date it applies from.")
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

    # Drop unconfirmed domains whose provisions were graded irrelevant (never errors: those must be reported).
    tentative = set(state.get("tentative_domains") or [])
    unconfirmed = [r for r in results if r.domain in tentative and r.grade == "insufficient" and not r.steps]
    if unconfirmed and len(unconfirmed) < len(results):
        gone = {r.domain for r in unconfirmed}
        results = [r for r in results if r.domain not in gone]
        state["all_domains"] = [d for d in state.get("all_domains") or [] if d not in gone]
        state["trace"].append(f"domain_pipeline → dropped unconfirmed domain(s) {sorted(gone)}: "
                              f"suggested only by the LLM classifier and nothing relevant was retrieved")

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
_ANNOTATION = re.compile(r" ?\[NOTE:[^\]]*\]|^- [^\n]*: some of these obligations are not yet in force[^\n]*$", re.M)


GROUNDING_TEXT_LIMIT = 4000


def _strip_annotations(answer: str) -> str:
    """Remove the notes the system adds itself (commencement dates, verification status). They come from
    the provisions' metadata, not their text, so the grader must not judge them against the text."""
    return _ANNOTATION.sub("", answer)


def _groundedness(state: AgentState) -> tuple:
    # Grade against the provisions the answer cites, in full (up to GROUNDING_TEXT_LIMIT chars each): a long
    # section like DPDP s.8 has the clause a step relies on far past the first few hundred characters.
    cited = set(state.get("verified_citations") or [])
    docs = [d for d in state["retrieved_docs"] if d.get("provision_id") in cited] or state["retrieved_docs"]
    limit = GROUNDING_TEXT_LIMIT if cited else 800
    chunks = "\n\n".join(f"[{d.get('provision_id', '')}]: {(d.get('full_text') or d.get('text', ''))[:limit]}"
                         for d in docs)
    multi = len(state.get("all_domains", [state["domain"]])) > 1
    system = """You are a legal grounding verifier.
Be lenient: grade as 'grounded' if the main claims are supported by context.
Only grade 'not_grounded' if the answer makes specific legal claims that
directly contradict or are completely absent from the retrieved context.
Minor elaborations and reasonable inferences are acceptable.
Statements saying that no provision was found for a topic are NOT legal claims; ignore them.
Judge only the claims the answer makes. An answer that leaves part of the question unaddressed is NOT
'not_grounded' for that reason (coverage is checked separately).
Text in square brackets starting with 'NOTE:' is a system annotation, not a legal claim; ignore it."""
    user = f"""Answer: {_strip_annotations(state['answer'])[:3000]}

Retrieved context:
{chunks}

{"Note: This is a multi-domain query. The answer may draw on multiple legal areas — grade as grounded if each domain's claims are supported by at least some of the retrieved chunks." if multi else ""}

Is every claim in the answer supported by the retrieved context?"""
    try:
        g = _llm(GroundednessGrade, system, user)
        grade, reason = g.grade, g.reason
        unsupported = g.unsupported_claims
    except StructuredOutputError:
        grade, reason, unsupported = "not_grounded", "Could not parse grader response", []
    except LLMUnavailable as e:
        return ({"groundedness_grade": "unverified", "groundedness_reason": f"LLM unavailable: {e}"},
                "grade_groundedness → unverified (LLM unavailable)")
    except Exception:
        grade, reason, unsupported = "not_grounded", "Could not parse grader response", []
    line = f"grade_groundedness → {grade}"
    if unsupported:
        line += f" (unsupported: {unsupported})"
    return {"groundedness_grade": grade, "groundedness_reason": reason}, line


def _answerability(state: AgentState) -> tuple:
    """Grade answerability with a structured LLM, preserving deterministic coverage fallback."""
    all_domains = state.get("all_domains") or [state["domain"]]
    system = "You are a legal answer quality checker."
    user = f"""Query: {state['query']}
Answer: {state['answer'][:3000]}

Does this answer address the main compliance areas asked about?
{"For multi-domain queries, grade 'answers' if the answer covers the key compliance areas." if len(all_domains) > 1 else ""}
Grade 'off_target' only if the answer is completely unrelated to what was asked."""
    try:
        g = _llm(AnswerabilityGrade, system, user)
        grade, reason = g.grade, g.reason
    except StructuredOutputError:
        grade, reason = "off_target", "Could not parse grader response"
    except Exception:
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
    if state.get("scope_error"):
        reason = state["scope_error"]
    elif state.get("llm_error"):
        reason = (f"The language model could not be reached ({state['llm_error']}). "
                  f"This is a service problem, not a problem with your question; please try again later.")
    elif state.get("domain") == "unknown" or state.get("intent") == "unknown":
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