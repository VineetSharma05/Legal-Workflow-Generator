from typing import TypedDict
from psycopg2.extras import Json


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
