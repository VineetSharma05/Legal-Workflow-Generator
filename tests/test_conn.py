"""Integration check: can we reach the configured Postgres instance?

Deselected by default (see ``addopts`` in pyproject.toml). Run with::

    uv run pytest -m integration tests/test_conn.py

or, for a quick standalone check, ``python -m tests.test_conn``.
"""

import pytest

import legal_workflow_generator.config.values as config

pytestmark = pytest.mark.integration


def _connect():
    import psycopg2

    return psycopg2.connect(
        dbname="postgres",
        user=config.DB_USER,
        password=config.PGPASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT,
        connect_timeout=5,
    )


def test_database_is_reachable():
    try:
        conn = _connect()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres not reachable at {config.DB_HOST}:{config.DB_PORT} ({exc})")
    else:
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    print("connecting...")
    c = _connect()
    print("connected!")
    c.close()
    print("done")
