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
  trace_timing       False removes "+1.4s" prefixes so traces are byte-for-byte reproducible
"""
import contextvars
import datetime as _dt
import os
from dataclasses import dataclass, fields

ALLOWED_PROVIDERS = ("gemini", "groq")
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
        for k in ("top_k", "act_cap", "search_depth", "bm25_candidates", "vector_candidates", "max_domain_retries"):
            if k in picked:
                picked[k] = int(picked[k])
        for k in ("verify_claims", "trace_timing"):
            if k in picked and isinstance(picked[k], str):
                picked[k] = picked[k].strip().lower() in ("1", "true", "yes", "on")
        if "as_of" in picked:
            picked["as_of"] = str(picked["as_of"])
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