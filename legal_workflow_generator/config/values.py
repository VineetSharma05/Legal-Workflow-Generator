import os
from pathlib import Path

from dotenv import load_dotenv

# ─────────────────────────────────────────────
# .ENV LOADING
# ─────────────────────────────────────────────
# Resolve the project root from this file so the .env is found no matter what
# the current working directory is (scripts/, tests/, evals/, or the app).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# Values already exported in the shell win over the .env file.
load_dotenv(ENV_FILE, override=False)


class MissingEnvironmentVariable(RuntimeError):
    """Raised at import time when a required environment variable is missing."""


def require_env(name: str) -> str:
    """
    Return the value of a required environment variable.

    There is deliberately no default: if the variable is missing or empty the
    process fails immediately instead of silently connecting with the wrong
    credentials.
    """
    value = os.getenv(name)

    if value is None or not value.strip():
        raise MissingEnvironmentVariable(
            f"Required environment variable '{name}' is not set.\n"
            f"Add it to {ENV_FILE} (see {PROJECT_ROOT / '.env.example'}) "
            f"or export it in your shell before running."
        )

    return value


def optional_env(name: str, default: str) -> str:
    """Return an optional environment variable, falling back to `default`."""
    value = os.getenv(name)
    return value if value is not None and value.strip() else default


# ─────────────────────────────────────────────
# DATABASE CONFIG
# ─────────────────────────────────────────────
# PGPASSWORD is required and never defaulted.
PGPASSWORD = require_env("PGPASSWORD")

DB_NAME = optional_env("PGDATABASE", "legal_workflow_generator")
DB_USER = optional_env("PGUSER", "postgres")
DB_HOST = optional_env("PGHOST", "127.0.0.1")
# Default matches the port published by docker-compose.yml.
DB_PORT = int(optional_env("PGPORT", "5432"))

# ─────────────────────────────────────────────
# GROQ CONFIG (optional — only needed for the groq provider)
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = optional_env("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─────────────────────────────────────────────
# GEMINI CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = require_env("GEMINI_API_KEY")
GEMINI_MODEL = optional_env("GEMINI_MODEL", "models/gemini-3.1-flash-lite")

# ─────────────────────────────────────────────
# CONTEXT RESOLVER CONFIG
# ─────────────────────────────────────────────
# Self-consistency resamples the domain classification prompt at temperature>0
# and majority-votes across the samples. It's an uncertainty signal, not free —
# each sample is an extra Gemini call — so it defaults off and is toggled here.
SELF_CONSISTENCY_ENABLED = optional_env("SELF_CONSISTENCY_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
SELF_CONSISTENCY_SAMPLES = int(optional_env("SELF_CONSISTENCY_SAMPLES", "3"))

# Domain classification: the deterministic KeywordDomainClassifier (TF-IDF
# terms extracted from the corpus, see rag/domain_keywords.py) always runs
# first. This controls whether/how Gemini is consulted on top of it:
#   "llm_fallback" — Gemini only called when the keyword classifier finds no
#                    match at all (cheapest).
#   "combine"      — Gemini is always called too, and cross-checked against
#                    the keyword result (costs an LLM call on every query).
DOMAIN_CLASSIFICATION_STRATEGY = optional_env("DOMAIN_CLASSIFICATION_STRATEGY", "llm_fallback")

# Minimum summed TF-IDF score for the keyword classifier to count a domain as
# "matched" at all, vs. reporting no match (score 0 means literally no
# extracted term appeared in the query).
DOMAIN_KEYWORD_MATCH_THRESHOLD = float(optional_env("DOMAIN_KEYWORD_MATCH_THRESHOLD", "0.0"))

# In "combine" strategy, a keyword match at or above this score is trusted
# over a disagreeing LLM answer; below it, the LLM wins the disagreement
# (a single weak/coincidental keyword hit shouldn't outrank the LLM).
DOMAIN_KEYWORD_STRONG_MATCH_THRESHOLD = float(optional_env("DOMAIN_KEYWORD_STRONG_MATCH_THRESHOLD", "0.3"))
