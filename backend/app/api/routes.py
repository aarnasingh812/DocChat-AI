"""
app/api/routes.py
──────────────────
FastAPI route handlers for the DocChat API.

Each handler is intentionally thin — it validates input, delegates to the
appropriate pipeline / database module, and shapes the response.  No business
logic lives here.

Endpoints
─────────
GET  /health               — liveness probe
POST /upload               — ingest a PDF and create a session
POST /chat                 — answer a question using the RAG pipeline
GET  /session/{session_id} — fetch session metadata
DELETE /session/{session_id} — delete a session and its vectors
"""

import logging
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    SessionInfoResponse,
    UploadResponse,
)
from app.database.sessions import get_session, save_session, delete_session
from app.pipeline.ingest import build_vector_store, delete_session as delete_session_chunks
from app.pipeline.generation import answer_question

logger = logging.getLogger("docchat.api.routes")

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", tags=["Ops"])
def health():
    """Simple liveness probe — returns 200 when the server is running."""
    return {"status": "ok"}


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, chunk it, embed the chunks, and create a new session.

    - Parsing and embedding are CPU/network-heavy; they run in a threadpool so
      the event loop is never blocked.
    - If embedding succeeds but session persistence fails, orphaned vectors are
      removed before returning the error.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes  = await file.read()
    session_id = str(uuid.uuid4())

    try:
        pages, chunks, warnings = await run_in_threadpool(
            build_vector_store, pdf_bytes, session_id
        )
    except ValueError as exc:
        # Unreadable / encrypted / no extractable text
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}")

    try:
        await run_in_threadpool(save_session, session_id, file.filename, pages, chunks)
    except Exception as exc:
        # Roll back orphaned vectors
        await run_in_threadpool(delete_session_chunks, session_id)
        raise HTTPException(status_code=500, detail=f"Failed to save session: {exc}")

    return UploadResponse(
        session_id=session_id,
        doc_name=file.filename,
        pages=pages,
        chunks=chunks,
        warnings=warnings,
    )


# ── Chat ──────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Answer a question using the RAG pipeline for the given session.

    Returns the answer, a list of source citations, and the elapsed time.
    """
    session = await run_in_threadpool(get_session, req.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please re-upload the document.",
        )

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = await run_in_threadpool(answer_question, req.session_id, req.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG error: {exc}")

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        elapsed=result["elapsed"],
        session_id=req.session_id,
    )


# ── Session info ──────────────────────────────────────────────────────────────

@router.get("/session/{session_id}", response_model=SessionInfoResponse, tags=["Sessions"])
async def get_session_info(session_id: str):
    """Fetch metadata for an existing session (document name, page count, etc.)."""
    session = await run_in_threadpool(get_session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionInfoResponse(**session)


@router.delete("/session/{session_id}", tags=["Sessions"])
async def delete_session_endpoint(session_id: str):
    """
    Delete a session — removes all vectors from PGVector and all DB rows.

    Vectors are removed first; the session record is deleted last so a failed
    vector removal is recoverable by retrying the request.
    """
    if await run_in_threadpool(get_session, session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        await run_in_threadpool(delete_session_chunks, session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete vectors: {exc}")

    await run_in_threadpool(delete_session, session_id)
    return {"detail": "Session deleted."}
