"""
Pydantic schemas for every structured LLM call.

Field order matters: models generate fields in order, so `reason` comes before
`grade` to make the model justify a verdict before committing to it.
Stick to what Gemini's response schema handles reliably: Literal enums, lists,
numeric/length bounds, required fields (no defaults, no unions).
"""
from typing import Literal

from pydantic import BaseModel, Field, create_model

Domain = Literal["data_protection", "corporate_governance", "ip_licensing", "taxation", "employment"]
DomainOrUnknown = Literal[
    "data_protection", "corporate_governance", "ip_licensing", "taxation", "employment", "unknown"
]
AppliesTo = Literal["private", "government", "both", "unspecified"]


# ── query unit ────────────────────────────────────────────────────────────────
class IntentClassification(BaseModel):
    reason: str = Field(description="One-line explanation")
    intent: Literal["qa", "workflow", "compliance_check", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0")


class DomainClassification(BaseModel):
    domain: DomainOrUnknown = Field(description="The single best-fitting legal domain")
    keywords: list[str] = Field(max_length=5, description="Up to 5 important legal keywords from the query")


# ── agent ─────────────────────────────────────────────────────────────────────
class DomainList(BaseModel):
    domains: list[Domain] = Field(description="Every legal domain the query touches")


class SearchQuery(BaseModel):
    query: str = Field(description="Standalone search query using plain topic words")


class RelevanceGrade(BaseModel):
    reason: str = Field(description="One sentence")
    grade: Literal["sufficient", "insufficient"]


class GroundednessGrade(BaseModel):
    reason: str = Field(description="One sentence")
    unsupported_claims: list[str] = Field(
        description="Specific legal claims in the answer that contradict or are absent from the context; empty if none"
    )
    grade: Literal["grounded", "not_grounded"]


class AnswerabilityGrade(BaseModel):
    reason: str = Field(description="One sentence")
    grade: Literal["answers", "off_target"]


def domain_workflow_model(allowed_ids: list[str]) -> type[BaseModel]:
    """
    Per-call schema for one domain's workflow. `provision_id` is an enum of the
    ids actually retrieved, so the model cannot cite a provision it wasn't shown.
    """
    step = create_model(
        "ComplianceStep",
        action=(str, Field(description="Concrete compliance action")),
        provision_id=(Literal[tuple(allowed_ids)], Field(description="Exact id of the provision requiring this step")),
        applies_to=(AppliesTo, Field(description="Who the provision applies to, based only on its text")),
    )
    return create_model(
        "DomainWorkflow",
        summary=(list[str], Field(max_length=3, description="1-2 short bullets on this legal area only")),
        steps=(list[step], Field(description="Compliance steps, each backed by one provision")),
        not_covered=(str, Field(description="Part of the query these provisions do not address, or empty string")),
    )
