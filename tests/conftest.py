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

os.environ.setdefault("PGPASSWORD", "test-password")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

import pytest

TESTS_DIR = Path(__file__).parent


@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to the checked-in one-page PDF used by the normalizer tests."""
    return TESTS_DIR / "test_legal.pdf"


class FakeGeminiResponse:
    """Mimics the object returned by ``client.models.generate_content``."""

    def __init__(self, text: str):
        self.text = text


class FakeGeminiModels:
    """
    Stand-in for ``genai.Client().models``.

    Pass a single string for a fixed reply, or a list to return a different
    reply per call (used by the self-consistency tests). Every call is recorded
    on ``.calls`` for assertions.
    """

    def __init__(self, replies):
        self._replies = replies if isinstance(replies, list) else [replies]
        self._i = 0
        self.calls = []

    def generate_content(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        if isinstance(reply, Exception):
            raise reply
        return FakeGeminiResponse(reply)


class FakeGeminiClient:
    """Stand-in for ``genai.Client`` — only ``.models`` is ever touched."""

    def __init__(self, replies):
        self.models = FakeGeminiModels(replies)


@pytest.fixture
def fake_gemini():
    """Factory: build a :class:`FakeGeminiModels` from a reply or list of replies."""
    return FakeGeminiModels


@pytest.fixture
def fake_gemini_client():
    """Factory: build a :class:`FakeGeminiClient` to assign onto ``obj.client``."""
    return FakeGeminiClient


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
