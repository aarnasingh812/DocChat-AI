"""
app/database/chunk_store.py
────────────────────────────
Persistence layer for the `session_chunks` table.

Responsibilities
────────────────
- DDL: create/migrate the table on first use (thread-safe, advisory lock).
- Write: bulk-insert chunks after PDF ingest (`save_session_chunks`).
- Read: load all chunks or just their ids for a session (used by BM25 and
  context-expansion modules).
- Delete: remove all chunks for a session (called during session cleanup).
- Helper: convert a raw DB row tuple into a Python dict (`row_to_dict`),
  handling the old 0-indexed `page` column for backwards compatibility.

Schema (current)
────────────────
    session_chunks (
        chunk_id   TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        page       INTEGER,           -- legacy 0-indexed page (kept for compat)
        text       TEXT NOT NULL,
        section_id INTEGER,
        seq        INTEGER,
        chunk_type TEXT,
        page_start INTEGER,
        page_end   INTEGER,
        breadcrumb TEXT
    )

Migration notes
───────────────
The table was originally created with only (chunk_id, session_id, page, text).
The additional columns are added via ALTER TABLE ADD COLUMN IF NOT EXISTS so
that existing deployments upgrade automatically on first startup.
"""

import logging
from threading import Lock
from typing import List

from app.database.connection import get_db_conn
from chunking import Chunk

logger = logging.getLogger("docchat.database.chunk_store")

# Columns fetched when loading chunks for BM25 / context expansion.
_ROW_COLUMNS = "chunk_id, section_id, seq, chunk_type, page_start, page_end, breadcrumb, page, text"

# ── Schema init (run once per process, guarded by an advisory DB lock) ────────

_schema_lock = Lock()
_schema_ready = False


def init_chunk_store() -> None:
    """
    Ensure the session_chunks table and all its columns exist.

    Safe to call from multiple threads/workers: the function is a no-op after
    the first successful run, and a Postgres advisory lock prevents concurrent
    DDL races across uvicorn workers in the same process group.
    """
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        with get_db_conn() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('docchat_session_chunks_schema'))")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_chunks (
                    chunk_id   TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    page       INTEGER,
                    text       TEXT NOT NULL
                )
            """)
            # Additive migrations — safe on existing tables
            for col in (
                "section_id INTEGER",
                "seq        INTEGER",
                "chunk_type TEXT",
                "page_start INTEGER",
                "page_end   INTEGER",
                "breadcrumb TEXT",
            ):
                conn.execute(f"ALTER TABLE session_chunks ADD COLUMN IF NOT EXISTS {col}")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_chunks_session_id
                    ON session_chunks (session_id)
            """)
            conn.commit()
        _schema_ready = True


# ── Write ─────────────────────────────────────────────────────────────────────

def save_session_chunks(session_id: str, chunks: List[Chunk], chunk_ids: List[str]) -> None:
    """
    Bulk-insert chunk rows for a newly ingested document.

    Uses ``ON CONFLICT DO NOTHING`` so re-uploading the same document does not
    raise an error (the old rows are simply kept).
    """
    rows = [
        (
            chunk_id, session_id,
            c.page_start - 1,      # legacy column: keep it 0-indexed like before
            c.text, c.section_id, c.seq, c.kind,
            c.page_start, c.page_end, c.breadcrumb or None,
        )
        for chunk_id, c in zip(chunk_ids, chunks)
    ]
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO session_chunks
                    (chunk_id, session_id, page, text, section_id, seq, chunk_type,
                     page_start, page_end, breadcrumb)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                rows,
            )
        conn.commit()


# ── Read ──────────────────────────────────────────────────────────────────────

def row_to_dict(row: tuple) -> dict:
    """
    Convert a raw DB row (fetched with ``_ROW_COLUMNS``) to a plain dict.

    Handles backwards-compat: legacy rows that were written before the new
    columns existed have ``page_start = NULL`` but a valid 0-indexed ``page``
    value; we convert that to a 1-indexed ``page_start`` transparently.
    """
    (chunk_id, section_id, seq, chunk_type, page_start, page_end, breadcrumb, legacy_page, text) = row
    if page_start is None and legacy_page is not None:
        page_start = legacy_page + 1   # legacy rows stored 0-indexed pages
    return {
        "chunk_id":   chunk_id,
        "section_id": section_id,
        "seq":        seq,
        "chunk_type": chunk_type or "text",
        "page_start": page_start,
        "page_end":   page_end if page_end is not None else page_start,
        "breadcrumb": breadcrumb or "",
        "text":       text,
    }


def load_session_chunks(session_id: str) -> List[dict]:
    """Return all chunk rows for a session, ordered by sequence number."""
    with get_db_conn() as conn:
        rows = conn.execute(
            f"SELECT {_ROW_COLUMNS} FROM session_chunks WHERE session_id = %s ORDER BY seq NULLS LAST",
            (session_id,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def load_session_chunk_ids(session_id: str) -> List[str]:
    """Return only the chunk_ids for a session (used during deletion)."""
    with get_db_conn() as conn:
        rows = conn.execute(
            "SELECT chunk_id FROM session_chunks WHERE session_id = %s", (session_id,)
        ).fetchall()
    return [r[0] for r in rows]


def load_section_chunks(session_id: str, section_ids: List[int]) -> List[dict]:
    """
    Fetch all chunks belonging to a set of section ids within a session.

    Used by the context-expansion step to retrieve neighbouring chunks without
    loading the entire document into memory.
    """
    with get_db_conn() as conn:
        rows = conn.execute(
            f"SELECT {_ROW_COLUMNS} FROM session_chunks "
            "WHERE session_id = %s AND section_id = ANY(%s)",
            (session_id, section_ids),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


# ── Delete ────────────────────────────────────────────────────────────────────

def delete_session_chunks(session_id: str) -> None:
    """Remove all chunk rows for the given session."""
    with get_db_conn() as conn:
        conn.execute("DELETE FROM session_chunks WHERE session_id = %s", (session_id,))
        conn.commit()
