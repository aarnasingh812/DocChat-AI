import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import psycopg
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import build_vector_store, answer_question, delete_session_chunks

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("docchat.main")

# ───────────────────────────────────────────────────────────────────────────── 
# DATABASE HELPERS  (sessions table – persists across server restarts)
# ─────────────────────────────────────────────────────────────────────────────

def get_db_conn() -> psycopg.Connection:
    url = os.getenv("DATABASE_URL", "")
    # psycopg3 native connection string uses postgresql:// (no +psycopg suffix)
    native_url = url.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(native_url, connect_timeout=10)


def init_sessions_table(retries: int = 5, delay: float = 2.0) -> None:
    """Create the sessions table, retrying if the DB isn't ready yet."""
    for attempt in range(1, retries + 1):
        try:
            with get_db_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        doc_name   TEXT NOT NULL,
                        pages      INTEGER NOT NULL,
                        chunks     INTEGER NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                conn.commit()
            logger.info("sessions table ready.")
            return
        except Exception as exc:
            logger.warning(
                "DB not ready (attempt %d/%d): %s. Retrying in %.1fs …",
                attempt, retries, exc, delay,
            )
            if attempt == retries:
                raise
            time.sleep(delay)


def db_save_session(session_id: str, doc_name: str, pages: int, chunks: int) -> None:
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, doc_name, pages, chunks)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE
                SET doc_name = EXCLUDED.doc_name,
                    pages    = EXCLUDED.pages,
                    chunks   = EXCLUDED.chunks
            """,
            (session_id, doc_name, pages, chunks),
        )
        conn.commit()


def db_get_session(session_id: str) -> Optional[dict]:
    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT session_id, doc_name, pages, chunks FROM sessions WHERE session_id = %s",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return {"session_id": row[0], "doc_name": row[1], "pages": row[2], "chunks": row[3]}


def db_delete_session(session_id: str) -> None:
    with get_db_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_sessions_table()
    yield


app = FastAPI(title="DocChat AI API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    question:   str


class ChatResponse(BaseModel):
    answer:     str
    sources:    list
    elapsed:    float
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    doc_name:   str
    pages:      int
    chunks:     int
    # Things the user should know, e.g. pages that could not be OCR'd. Empty when all is well.
    warnings:   list[str] = []


class SessionInfoResponse(BaseModel):
    session_id: str
    doc_name:   str
    pages:      int
    chunks:     int


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes  = await file.read()
    session_id = str(uuid.uuid4())

    # Parsing + embedding are blocking, CPU/network-heavy calls: run them in the
    # threadpool so one upload can't stall every other request on the event loop.
    try:
        pages, chunks, warnings = await run_in_threadpool(build_vector_store, pdf_bytes, session_id)
    except ValueError as exc:
        # unreadable / encrypted / no extractable text (e.g. a scanned PDF)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}")

    try:
        await run_in_threadpool(db_save_session, session_id, file.filename, pages, chunks)
    except Exception as exc:
        # Roll back orphaned vectors so we don't leave dangling chunks
        await run_in_threadpool(delete_session_chunks, session_id)
        raise HTTPException(status_code=500, detail=f"Failed to save session: {exc}")

    return UploadResponse(
        session_id=session_id,
        doc_name=file.filename,
        pages=pages,
        chunks=chunks,
        warnings=warnings,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session = await run_in_threadpool(db_get_session, req.session_id)
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


@app.get("/session/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str):
    session = await run_in_threadpool(db_get_session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionInfoResponse(**session)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if await run_in_threadpool(db_get_session, session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Remove vectors first, then the session record
    try:
        await run_in_threadpool(delete_session_chunks, session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete vectors: {exc}")

    await run_in_threadpool(db_delete_session, session_id)
    return {"detail": "Session deleted."}
