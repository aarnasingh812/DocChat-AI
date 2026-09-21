"""
app/pipeline/ingest.py
───────────────────────
Stage 1 of the RAG pipeline: document ingestion.

Flow
────
PDF bytes
  └─► chunk_pdf()          — structure-aware chunking (+ vision OCR fallback)
        └─► contextualize() — prepend breadcrumb + page header to each chunk
              └─► PGVector.add_documents()  — embed & store dense vectors
                    └─► save_session_chunks()  — persist text to Postgres for BM25

Public API
──────────
    build_vector_store(pdf_bytes, session_id) -> IngestResult
    delete_session(session_id) -> None
"""

import logging
from functools import lru_cache
from typing import List, NamedTuple

from langchain_core.documents import Document

from app.config import OCR_ENABLED, OCR_API_KEY, OCR_MODEL, OCR_RPM, OCR_PAGES_PER_REQUEST
from app.config import OCR_CONCURRENCY, OCR_MAX_PAGES, OCR_FIGURES, OCR_DAILY_REQUEST_BUDGET
from app.database.chunk_store import (
    init_chunk_store,
    save_session_chunks,
    load_session_chunk_ids,
    delete_session_chunks,
)
from app.pipeline.bm25_index import invalidate_bm25_cache
from app.services.vector_store import get_vector_store
from chunking import ChunkingConfig, chunk_pdf, contextualize
from ocr import DEFAULT_MODEL as OCR_DEFAULT_MODEL, GeminiVisionOCR, PostgresDailyBudget
from app.database.connection import get_db_conn

logger = logging.getLogger("docchat.pipeline.ingest")


# ── OCR provider (lazy singleton) ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_ocr() -> "GeminiVisionOCR | None":
    """
    Return the vision-OCR provider, or ``None`` when OCR is disabled / no key.

    When ``None`` is returned, scanned pages are reported in warnings but not
    indexed.

    Settings are read from :mod:`app.config` (all ``OCR_*`` variables).
    """
    if not OCR_ENABLED:
        logger.info("OCR disabled via OCR_ENABLED=false")
        return None
    if not OCR_API_KEY:
        logger.warning("No GOOGLE_API_KEY — OCR fallback disabled; scanned pages will not be searchable.")
        return None
    model = OCR_MODEL or OCR_DEFAULT_MODEL
    return GeminiVisionOCR(
        api_key=OCR_API_KEY,
        model=model,
        rpm=OCR_RPM,
        pages_per_request=OCR_PAGES_PER_REQUEST,
        concurrency=OCR_CONCURRENCY,
        daily_budget=PostgresDailyBudget(get_db_conn, OCR_DAILY_REQUEST_BUDGET),
    )


@lru_cache(maxsize=1)
def get_chunking_config() -> ChunkingConfig:
    """Return the ChunkingConfig built from env-var settings."""
    return ChunkingConfig(
        ocr_max_pages=OCR_MAX_PAGES,
        ocr_figures=OCR_FIGURES,
    )


# ── Public result type ────────────────────────────────────────────────────────

class IngestResult(NamedTuple):
    pages: int
    chunks: int
    warnings: List[str]
    """Human-readable warnings, e.g. pages that exceeded the OCR daily budget."""


# ── Main ingest function ──────────────────────────────────────────────────────

def build_vector_store(pdf_bytes: bytes, session_id: str) -> IngestResult:
    """
    Parse a PDF into structure-aware chunks, embed them, and persist everything.

    Steps
    ─────
    1. Initialise the ``session_chunks`` DB table (idempotent).
    2. Run ``chunk_pdf()`` — extracts text with OCR fallback for scanned pages.
    3. Contextualise each chunk (prepend section breadcrumb + page range).
    4. Embed and store in PGVector (dense index).
    5. Persist raw chunk text to Postgres (BM25 / context-expansion source).

    On any error after step 4, the function attempts to roll back all stored
    data so no half-ingested session is left behind.

    Parameters
    ──────────
    pdf_bytes:
        Raw PDF file content.
    session_id:
        Unique identifier for this upload session (UUID).

    Returns
    ───────
    :class:`IngestResult` with page count, chunk count, and any warnings.

    Raises
    ──────
    ``ValueError``
        If no text could be extracted (encrypted, corrupt, or purely visual PDF
        with OCR disabled / budget exhausted).
    """
    init_chunk_store()

    result = chunk_pdf(pdf_bytes, cfg=get_chunking_config(), ocr=get_ocr())
    chunks = result.chunks
    if not chunks:
        detail = " ".join(result.warnings) or "The PDF has no text layer and no readable images."
        raise ValueError(f"No searchable text could be extracted from this PDF. {detail}")

    chunk_ids = [f"{session_id}:{c.seq}" for c in chunks]
    docs: List[Document] = []
    for chunk_id, c in zip(chunk_ids, chunks):
        content, offset = contextualize(c.breadcrumb, c.page_start, c.page_end, c.kind, c.text)
        docs.append(Document(
            page_content=content,
            metadata={
                "session_id":  session_id,
                "chunk_id":    chunk_id,
                "section_id":  c.section_id,
                "seq":         c.seq,
                "chunk_type":  c.kind,
                "page_start":  c.page_start,
                "page_end":    c.page_end,
                "breadcrumb":  c.breadcrumb,
                "body_offset": offset,
            },
        ))

    try:
        get_vector_store().add_documents(docs, ids=chunk_ids)
        save_session_chunks(session_id, chunks, chunk_ids)
    except Exception:
        # Roll back: don't leave half-ingested vectors and chunk rows behind.
        try:
            get_vector_store().delete(ids=chunk_ids)
            delete_session_chunks(session_id)
            invalidate_bm25_cache(session_id)
        except Exception:
            logger.exception(
                "Cleanup after failed ingest of session %s also failed", session_id
            )
        raise

    logger.info(
        "ingested session=%s pages=%d sections=%d chunks=%d "
        "(tables=%d, ocr_pages=%d, warnings=%d)",
        session_id, result.page_count, result.section_count, len(chunks),
        sum(1 for c in chunks if c.kind == "table"),
        len(result.ocr_pages), len(result.warnings),
    )
    return IngestResult(result.page_count, len(chunks), result.warnings)


def delete_session(session_id: str) -> None:
    """
    Remove all vectors and chunk rows for a session.

    PGVector's ``delete()`` only accepts explicit ids (not filter dicts), so
    we load the ids from our own table first.
    """
    chunk_ids = load_session_chunk_ids(session_id)
    if chunk_ids:
        get_vector_store().delete(ids=chunk_ids)
    delete_session_chunks(session_id)
    invalidate_bm25_cache(session_id)
