"""
Shared pytest fixtures and test-time configuration.

`legal_workflow_generator.config.values` raises `MissingEnvironmentVariable` at
import time when `PGPASSWORD` / `GEMINI_API_KEY` are absent, so we inject dummy
values here (before any test module is imported) unless the developer already
has a real `.env` / shell export. The dummy Gemini key is never used for a real
call — the unit tests mock every network boundary.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / ".env", override=False)
os.environ.setdefault("PGPASSWORD", "test-password")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

import pytest

TESTS_DIR = Path(__file__).parent


@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to the checked-in one-page PDF used by the normalizer tests."""
    return TESTS_DIR / "test_legal.pdf"


class FakeStructuredLLM:
    """
    Stand-in for a ``structured_llm(...)`` runnable (``with_structured_output``
    with ``include_raw=True``).

    Pass a single reply or a list to return a different reply per call (used by
    the self-consistency tests). A reply is a parsed Pydantic object, ``None``
    to simulate a schema-validation failure, or an Exception to raise (API
    error). Every call's messages are recorded on ``.calls``.
    """

    def __init__(self, replies):
        self._replies = replies if isinstance(replies, list) else [replies]
        self._i = 0
        self.calls = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        if isinstance(reply, Exception):
            raise reply
        if reply is None:
            return {"raw": None, "parsed": None, "parsing_error": ValueError("schema mismatch")}
        return {"raw": None, "parsed": reply, "parsing_error": None}


@pytest.fixture
def fake_llm():
    """Factory: build a :class:`FakeStructuredLLM` to assign onto ``obj.llm``."""
    return FakeStructuredLLM


@pytest.fixture
def live_backends():
    """
    Skip an integration test unless a real Gemini key is configured and Postgres
    is reachable. Used by the end-to-end pipeline/agent tests.
    """
    import legal_workflow_generator.config.values as config

    if config.GEMINI_API_KEY in ("", "test-gemini-key"):
        pytest.skip("no real GEMINI_API_KEY configured")

    try:
        import psycopg2

        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.PGPASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
            connect_timeout=5,
        )
        conn.close()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres not reachable ({exc})")
