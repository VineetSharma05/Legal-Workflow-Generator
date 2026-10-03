"""Offline integrity checks for the complete legal corpus."""

import json
from pathlib import Path

from legal_workflow_generator.rag.ingestion import resolve_domain


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPLETE_DIR = REPO_ROOT / "datasets" / "complete"
COMBINED_PATH = COMPLETE_DIR / "combined_complete_dataset.json"
SOURCE_PATHS = sorted(
    path for path in COMPLETE_DIR.glob("complete_*.json")
    if path.name != COMBINED_PATH.name
)
REQUIRED_KEYS = {
    "provision_id",
    "statute_id",
    "provision_type",
    "chapter",
    "chapter_title",
    "number",
    "title",
    "text",
    "sub_structure",
    "plain_english_summary",
    "keywords",
    "penalty_linked",
    "effective_date",
}


def _load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    assert isinstance(value, list), f"{path.name} must contain a JSON list"
    return value


def test_complete_corpus_matches_all_source_files():
    combined = _load_json(COMBINED_PATH)
    source_rows = [row for path in SOURCE_PATHS for row in _load_json(path)]

    assert len(SOURCE_PATHS) == 5
    assert len(combined) == len(source_rows) == 4888
    assert [row["provision_id"] for row in combined] == [
        row["provision_id"] for row in source_rows
    ]


def test_complete_corpus_has_valid_unique_records_and_schema():
    rows = _load_json(COMBINED_PATH)
    provision_ids = [row.get("provision_id") for row in rows]

    assert all(isinstance(row, dict) for row in rows)
    assert all(REQUIRED_KEYS <= set(row) for row in rows)
    assert all(isinstance(row["keywords"], list) for row in rows)
    assert all(row["provision_id"] for row in rows)
    assert len(provision_ids) == len(set(provision_ids))


def test_complete_corpus_records_have_known_domains():
    rows = _load_json(COMBINED_PATH)

    assert all(resolve_domain(row) != "unknown" for row in rows)
    assert {
        resolve_domain(row) for row in rows
    } == {
        "data_protection",
        "corporate_governance",
        "employment",
        "ip_licensing",
        "taxation",
    }
