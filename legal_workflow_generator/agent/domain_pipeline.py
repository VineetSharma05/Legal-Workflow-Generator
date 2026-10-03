"""
Parallel per-domain pipeline.

classify_query decides which domains a query touches. This node then runs one
independent worker per domain, all at the same time:

    worker(domain) = retrieve -> grade -> (rewrite -> retrieve -> grade)* -> generate

and finally stitches the per-domain results into one answer.

Place at: legal_workflow_generator/agent/domain_pipeline.py
It only reads/writes state keys that already exist, so state.py is unchanged.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor

from legal_workflow_generator.agent.state import AgentState
from legal_workflow_generator.agent.nodes import (
    _llm,
    _get_rag_pipeline,
    _domain_of,
    _load_domain_map,
    DOMAIN_QUERY_HINTS,
)

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

MAX_DOMAIN_RETRIES = 2   # rewrite attempts per domain (so up to 3 retrievals)
DOCS_PER_DOMAIN = 4


def _parse_json(raw: str) -> dict:
    return json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())


# ── worker steps ──────────────────────────────────────────────────────────────
def _search_domain(rag, query: str, domain: str) -> list:
    """Search, then keep only provisions that belong to this domain."""
    hint = DOMAIN_QUERY_HINTS.get(domain, "")
    docs = rag.searcher.search(f"{query} {hint}".strip(), top_k=40)
    keep = [
        d for d in docs
        if "_forms_" not in d.get("provision_id", "")
        and _domain_of(d.get("provision_id", "")) == domain
    ]
    return keep[:DOCS_PER_DOMAIN]


def _grade_domain(domain: str, query: str, docs: list) -> tuple[str, str]:
    chunks = "\n\n".join(
        f"[{d.get('provision_id','')}] {d.get('title','')}: {str(d.get('text',''))[:400]}"
        for d in docs
    )
    system = "You are a legal relevance grader. Reply in JSON only."
    user = f"""Query: {query}
Legal area being checked: {_DOMAIN_LABELS[domain]}

Retrieved provisions:
{chunks}

Grade ONLY the part of the query that concerns this legal area.
"sufficient": at least one provision directly addresses that part of the query.
"insufficient": the provisions are unrelated to it.
Reply ONLY with:
{{"grade": "sufficient" or "insufficient", "reason": "one sentence"}}"""
    try:
        p = _parse_json(_llm(system, user, grader=True))
        return p.get("grade", "insufficient"), p.get("reason", "")
    except Exception:
        return "insufficient", "Could not parse grader response"


def _rewrite_for_domain(domain: str, query: str, reason: str) -> str:
    system = "You are a legal search query optimizer."
    user = f"""Original query: {query}
Legal area: {_DOMAIN_LABELS[domain]}
Why retrieval failed: {reason}

Write a better search query for Indian statutes in this legal area only.
Use plain topic words. Do NOT cite section numbers you are not certain exist.
Reply with ONLY the query."""
    try:
        return _llm(system, user).strip() or query
    except Exception:
        return query


def _generate_domain(domain: str, query: str, docs: list) -> dict:
    ids = {d.get("provision_id", "") for d in docs}
    chunks = "\n\n".join(
        f"[{d.get('provision_id','')}] {d.get('title','')}: "
        f"{str(d.get('text','') or d.get('plain_english_summary',''))[:800]}"
        for d in docs
    )
    system = "You are a legal workflow assistant for Indian tech startups."
    user = f"""Query: {query}
Legal area: {_DOMAIN_LABELS[domain]}

Using ONLY the provisions below, list the compliance actions relevant to this legal area for the query.
Do not invent sections or obligations. Address specifics in the query (company size, employee type, etc.).
For each step, say who the provision applies to ("private", "government", "both" or "unspecified"), based only on the text.

Provisions:
{chunks}

Reply ONLY with JSON:
{{"summary": ["1-2 short bullets"],
 "steps": [{{"action": "...", "provision_id": "<exact id from above>", "applies_to": "private|government|both|unspecified"}}],
 "not_covered": "part of the query these provisions do not address, or empty string"}}"""

    out = {"summary": [], "steps": [], "not_covered": "", "cited": []}
    try:
        p = _parse_json(_llm(system, user))
    except Exception:
        return out

    steps = []
    for s in p.get("steps", []):
        pid = str(s.get("provision_id", "")).strip()
        action = str(s.get("action", "")).strip()
        if pid in ids and action:          # drops invented / unknown provision ids
            steps.append({
                "action": action,
                "provision_id": pid,
                "applies_to": str(s.get("applies_to", "unspecified")).lower(),
            })

    out["summary"] = [str(x) for x in p.get("summary", [])][:3]
    out["steps"] = steps
    out["not_covered"] = str(p.get("not_covered", "") or "").strip()
    out["cited"] = list(dict.fromkeys(s["provision_id"] for s in steps))
    return out


def _domain_worker(domain: str, query: str, rag) -> tuple[dict, list]:
    """One independent mini-pipeline for a single domain. Returns (result, trace)."""
    trace = []
    q = query
    docs, grade, reason = [], "insufficient", "No documents retrieved"
    retries = 0

    for attempt in range(MAX_DOMAIN_RETRIES + 1):
        try:
            docs = _search_domain(rag, q, domain)
        except Exception as e:
            docs = []
            reason = f"Search failed: {e}"
        trace.append(f"[{domain}] retrieve (try {attempt + 1}) → {len(docs)} docs")

        if docs:
            grade, reason = _grade_domain(domain, query, docs)
            trace.append(f"[{domain}] grade → {grade}: {reason}")
        else:
            grade = "insufficient"

        if grade == "sufficient" or attempt == MAX_DOMAIN_RETRIES:
            break

        q = _rewrite_for_domain(domain, query, reason)
        retries += 1
        trace.append(f"[{domain}] rewrite → '{q[:60]}'")

    result = {
        "domain": domain, "docs": docs, "grade": grade, "reason": reason,
        "retries": retries, "summary": [], "steps": [], "not_covered": "", "cited": [],
    }
    if grade == "sufficient":
        result.update(_generate_domain(domain, query, docs))
        trace.append(f"[{domain}] generate → {len(result['steps'])} steps, {len(result['cited'])} citations")
    return result, trace


# ── stitching layer ───────────────────────────────────────────────────────────
def _stitch_sections(query: str, results: list) -> str:
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
        for s in r["steps"]:
            n += 1
            note = "" if s["applies_to"] == "unspecified" else f" (applies to: {s['applies_to']})"
            checklist.append(f"Step {n}: {s['action']} — [{s['provision_id']}]{note}")
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


# ── the graph node ────────────────────────────────────────────────────────────
def domain_pipeline(state: AgentState) -> AgentState:
    rag = _get_rag_pipeline()
    rag._ensure_index()      # build/load everything BEFORE spawning threads
    _load_domain_map()

    wanted = state.get("all_domains") or [state["domain"]]
    domains = [d for d in _DOMAIN_ORDER if d in wanted] or list(wanted)
    query = state.get("normalized_query") or state["query"]

    with ThreadPoolExecutor(max_workers=max(1, len(domains))) as ex:
        futures = [ex.submit(_domain_worker, d, query, rag) for d in domains]
        outputs = [f.result() for f in futures]      # keeps domain order

    results = [r for r, _ in outputs]
    for _, trace in outputs:
        state["trace"].extend(trace)

    # merged views for the existing downstream nodes
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
        state["context_grade_reason"] = "; ".join(
            f"{r['domain']}: {r['grade']}" for r in results
        )
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
        f"({', '.join(r['domain'] for r in covered) or 'none'})"
    )
    return state