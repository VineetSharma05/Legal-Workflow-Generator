"""
agent/settings.py - per-run settings that travel with ONE LangGraph invocation.

The API (or any caller) passes them through LangGraph's `config`:

    graph.invoke({"query": q}, config=build_config(provider="gemini", top_k=8))
    # or the helper in graph.py:  run_workflow(q, provider="gemini", top_k=8)

Every node that needs them calls `use_settings(config)`; worker threads inherit them because
nodes submit work with a copied contextvars context (see `_submit` in nodes.py).

Meaning of the options
  provider           LLM backend for ALL workflow LLM calls: "gemini" or "groq"
  top_k              provisions handed to the grader/generator PER DOMAIN (1..20, default 8)
  act_cap            of those, at most this many may be Act sections (rest are rules/regulations)
  search_depth       how many candidates the searcher returns before the domain filter
  bm25_candidates    BM25 first-stage pool        vector_candidates   extra whole-corpus neighbours
  max_domain_retries rewrite+retrieve retries per domain
  verify_claims      check every generated step against the provision it cites (1 extra LLM call per domain)
  as_of              ISO date (YYYY-MM-DD) the answer should be valid on; default = today
  future_provisions  what to do with provisions that commence after as_of:
                     "annotate" (default) keeps them and marks each step "not yet in force; applies from <date>",
                     "exclude" drops them (the behaviour before phased statutes such as the DPDP Act were dated)
  domain_search      "global": search the whole corpus, then keep this domain's provisions from the top
                     search_depth (the original design); "domain": search only this domain's provisions,
                     so a small domain is not crowded out of the candidate pool by a large one
  suggest_provisions the per-domain planner call also names up to suggest_cap provisions (as corpus ids) that the
                     model expects to answer the question; ids that exist in the domain join the context and
                     are graded, cited and claim-checked like retrieved provisions (no extra LLM call)
  statute_search     when the question names a statute (STATUTE_ALIASES in nodes.py), also search inside that
                     statute alone and fuse it with the open search (no LLM call)
  trace_timing       False removes "+1.4s" prefixes so traces are byte-for-byte reproducible
"""
import contextvars
import datetime as _dt
import os
from dataclasses import dataclass, fields

ALLOWED_PROVIDERS = ("gemini", "groq")
FUTURE_PROVISION_MODES = ("annotate", "exclude")
DOMAIN_SEARCH_MODES = ("global", "domain")
MAX_TOP_K = 20


@dataclass(frozen=True)
class RunSettings:
    provider: str = os.getenv("LLM_PROVIDER", "gemini")
    top_k: int = 8
    act_cap: int = 6
    search_depth: int = 100
    bm25_candidates: int = 300
    vector_candidates: int = 100
    max_domain_retries: int = 1
    verify_claims: bool = True
    as_of: str = ""
    future_provisions: str = "annotate"
    domain_search: str = "global"
    suggest_provisions: bool = False
    suggest_cap: int = 4
    statute_search: bool = True
    trace_timing: bool = True

    def __post_init__(self):
        if self.provider not in ALLOWED_PROVIDERS:
            raise ValueError(f"provider must be one of {ALLOWED_PROVIDERS}, got {self.provider!r}")
        if not (1 <= int(self.top_k) <= MAX_TOP_K):
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {self.top_k!r}")
        if self.act_cap < 1 or self.search_depth < 1 or self.bm25_candidates < 1 or self.vector_candidates < 0:
            raise ValueError("act_cap, search_depth and bm25_candidates must be >= 1; vector_candidates >= 0")
        if self.max_domain_retries < 0 or self.max_domain_retries > 3:
            raise ValueError("max_domain_retries must be between 0 and 3")
        if self.future_provisions not in FUTURE_PROVISION_MODES:
            raise ValueError(f"future_provisions must be one of {FUTURE_PROVISION_MODES}, got {self.future_provisions!r}")
        if not (0 <= int(self.suggest_cap) <= 10):
            raise ValueError(f"suggest_cap must be between 0 and 10, got {self.suggest_cap!r}")
        if self.domain_search not in DOMAIN_SEARCH_MODES:
            raise ValueError(f"domain_search must be one of {DOMAIN_SEARCH_MODES}, got {self.domain_search!r}")
        if self.as_of:
            _dt.date.fromisoformat(self.as_of)        # raises ValueError on a bad date

    @property
    def as_of_date(self) -> _dt.date:
        return _dt.date.fromisoformat(self.as_of) if self.as_of else _dt.date.today()

    @classmethod
    def from_config(cls, config=None) -> "RunSettings":
        """Build settings from a LangGraph config. Unknown keys (LangGraph's own) are ignored."""
        cfg = {}
        if config is not None and hasattr(config, "get"):
            cfg = config.get("configurable") or {}
        known = {f.name: f.type for f in fields(cls)}
        picked = {k: v for k, v in cfg.items() if k in known and v is not None}
        if "provider" in picked:
            picked["provider"] = str(picked["provider"]).strip().lower()
        for k in ("suggest_cap", "top_k", "act_cap", "search_depth", "bm25_candidates", "vector_candidates", "max_domain_retries"):
            if k in picked:
                picked[k] = int(picked[k])
        for k in ("verify_claims", "trace_timing", "suggest_provisions", "statute_search"):
            if k in picked and isinstance(picked[k], str):
                picked[k] = picked[k].strip().lower() in ("1", "true", "yes", "on")
        if "as_of" in picked:
            picked["as_of"] = str(picked["as_of"])
        if "domain_search" in picked:
            picked["domain_search"] = str(picked["domain_search"]).strip().lower()
        if "future_provisions" in picked:
            picked["future_provisions"] = str(picked["future_provisions"]).strip().lower()
        return cls(**picked)


_CURRENT: "contextvars.ContextVar[RunSettings | None]" = contextvars.ContextVar("workflow_run_settings", default=None)


def current_settings() -> RunSettings:
    s = _CURRENT.get()
    return s if s is not None else RunSettings()


def use_settings(config=None) -> RunSettings:
    """Activate the settings carried by `config` for the current context (call at the top of every node)."""
    s = RunSettings.from_config(config)
    _CURRENT.set(s)
    return s


def build_config(**options) -> dict:
    """Build the `config` argument for graph.invoke(); None values are dropped."""
    return {"configurable": {k: v for k, v in options.items() if v is not None}}