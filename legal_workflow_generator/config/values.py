import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = 'legal_workflow_generator'
DB_USER = 'postgres'
DB_HOST = '127.0.0.1'
DB_PORT = 5432

# ─────────────────────────────────────────────
# GROQ CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# This will catch the mistake early if someone
# forgets to set the API key in .env
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env file")

GROQ_MODEL = "llama-3.3-70b-versatile" 

# ─────────────────────────────────────────────
# GEMINI CONFIG (RAG answer generation)
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")