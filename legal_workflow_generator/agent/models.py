"""Typed data models and domain constants for the LangGraph agent pipeline."""
from dataclasses import asdict, dataclass, field
from typing import Any

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


class LLMUnavailable(RuntimeError):
    """The LLM could not be reached (quota / overload) after retries. Distinct from 'no provisions found'."""


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
