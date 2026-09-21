"""
app/api/schemas.py
───────────────────
Pydantic request and response models for the DocChat API.

Keeping schemas in a dedicated file means:
- Route handlers stay thin (no inline model definitions cluttering business logic).
- Models are easy to find, version, and share with generated API clients.
- FastAPI's auto-generated OpenAPI docs (``/docs``) benefits from the full
  type information.
"""

from pydantic import BaseModel


# ── Requests ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Body for ``POST /chat``."""
    session_id: str
    question:   str


# ── Responses ─────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Returned after a successful ``POST /upload``."""
    session_id: str
    doc_name:   str
    pages:      int
    chunks:     int
    warnings:   list[str] = []
    """Non-fatal issues during ingest, e.g. pages that exceeded OCR budget."""


class ChatResponse(BaseModel):
    """Returned after a successful ``POST /chat``."""
    answer:     str
    sources:    list
    elapsed:    float
    session_id: str


class SessionInfoResponse(BaseModel):
    """Returned by ``GET /session/{session_id}``."""
    session_id: str
    doc_name:   str
    pages:      int
    chunks:     int
