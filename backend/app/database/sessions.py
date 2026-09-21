"""
app/database/sessions.py
────────────────────────
CRUD operations for the `sessions` table.

The sessions table persists high-level document metadata (filename, page
count, chunk count) across server restarts so the frontend can resume a
previous session without re-uploading.

Schema
──────
    sessions (
        session_id TEXT PRIMARY KEY,
        doc_name   TEXT NOT NULL,
        pages      INTEGER NOT NULL,
        chunks     INTEGER NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    )
"""

import logging
import time
from typing import Optional

from app.database.connection import get_db_conn

logger = logging.getLogger("docchat.database.sessions")


def init_sessions_table(retries: int = 5, delay: float = 2.0) -> None:
    """
    Create the sessions table if it does not exist.

    Retries up to `retries` times with `delay` seconds between attempts so
    that the app starts cleanly even when the database container is still
    booting (common in Docker Compose setups).

    Raises the last exception if all attempts fail.
    """
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


def save_session(session_id: str, doc_name: str, pages: int, chunks: int) -> None:
    """Insert or update a session row (upsert on primary key)."""
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


def get_session(session_id: str) -> Optional[dict]:
    """
    Fetch a session record by id.

    Returns a dict with keys ``session_id``, ``doc_name``, ``pages``,
    ``chunks``, or ``None`` when the session does not exist.
    """
    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT session_id, doc_name, pages, chunks FROM sessions WHERE session_id = %s",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return {"session_id": row[0], "doc_name": row[1], "pages": row[2], "chunks": row[3]}


def delete_session(session_id: str) -> None:
    """Delete a session record. No-op if the session does not exist."""
    with get_db_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        conn.commit()
