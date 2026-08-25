from typing import TypedDict
from psycopg2.extras import Json
from enum import Enum


class LawSchema(TypedDict):
    provision_id: str
    statute_id: str

    provision_type: str
    chapter: str
    chapter_title: str

    number: str
    title: str
    text: str

    sub_structure: Json
    plain_english_summary: str

    keywords: list[str]
    penalty_linked: bool
    effective_date: str

class QueryIntent(Enum):
    QA = "qa"                             
    WORKFLOW = "workflow"                 
    COMPLIANCE_CHECK = "compliance_check" 
    UNKNOWN = "unknown"                   


class NormalizedQuery(TypedDict):
    original: str        
    normalized: str      
    source: str


class LegalContext(TypedDict):
    original_query: str
    normalized_query: str
    intent: QueryIntent
    legal_domain: str
    keywords: list[str]
    confidence: float

    # Domain classification correctness signals (see LegalContextResolver)
    keyword_domain: str         # TF-IDF keyword classifier's domain (rag/domain_keywords.py); "unknown" if no match
    domain_agreement: bool      # True when keyword and LLM sources were both consulted and agreed
    domain_confidence: float    # self-consistency vote share; 1.0 when self-consistency is off
    domain_source: str          # which source decided legal_domain: keyword | llm | keyword+llm |
                                 # keyword_strong_override | llm_override | unknown | skipped