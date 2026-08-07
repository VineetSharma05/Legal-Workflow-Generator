import os
from dotenv import load_dotenv
load_dotenv()

DB_NAME = 'legal_workflow_generator'
DB_USER = 'postgres'
DB_HOST = '127.0.0.1'
DB_PORT = 5433

# ─────────────────────────────────────────────
# GROQ CONFIG (kept for reference, not used)
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────
# GEMINI CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "GEMINI_MODEL=models/gemini-3.1-flash-lite")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env file")