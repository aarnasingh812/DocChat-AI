"""
app/config.py
─────────────
Single source of truth for all environment-variable configuration.

Every other module should import its settings from here instead of calling
os.getenv() directly. This makes it trivial to audit what the app reads,
swap to a proper settings library (e.g. pydantic-settings) later, or
mock values in tests.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Database ──────────────────────────────────────────────────────────────────

DATABASE_URL: str = os.getenv("DATABASE_URL", "")
"""Full SQLAlchemy-style or psycopg connection string.

Expected format:
  postgresql+psycopg://user:password@host:port/dbname
"""

# ── External AI services ──────────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "rerank-english-v3.0")

# ── OCR ───────────────────────────────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


OCR_ENABLED: bool = _env_bool("OCR_ENABLED", True)
OCR_API_KEY: str = os.getenv("OCR_API_KEY", "") or GOOGLE_API_KEY
OCR_MODEL: str = os.getenv("OCR_MODEL", "")          # falls back to ocr.DEFAULT_MODEL if empty
OCR_RPM: float = float(os.getenv("OCR_RPM", "10"))
OCR_DAILY_REQUEST_BUDGET: int = int(os.getenv("OCR_DAILY_REQUEST_BUDGET", "400"))
OCR_PAGES_PER_REQUEST: int = int(os.getenv("OCR_PAGES_PER_REQUEST", "3"))
OCR_CONCURRENCY: int = int(os.getenv("OCR_CONCURRENCY", "2"))
OCR_MAX_PAGES: int = int(os.getenv("OCR_MAX_PAGES", "24"))
OCR_FIGURES: bool = _env_bool("OCR_FIGURES", True)

# ── Retrieval ─────────────────────────────────────────────────────────────────

CONTEXT_NEIGHBOR_WINDOW: int = int(os.getenv("CONTEXT_NEIGHBOR_WINDOW", "1"))
"""Number of neighbouring chunks (same section) added around each retrieved chunk."""

CONTEXT_MAX_CHARS_PER_HIT: int = int(os.getenv("CONTEXT_MAX_CHARS_PER_HIT", "3600"))
"""Hard character budget for each expanded context window sent to the LLM."""

# ── Vector store ──────────────────────────────────────────────────────────────

VECTOR_COLLECTION: str = os.getenv("VECTOR_COLLECTION", "docchat_chunks")
