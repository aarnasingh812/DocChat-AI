"""
app/database/sessions.py
────────────────────────
Document sessions and durable chat history.

The sessions table persists high-level document metadata (filename, page
count, chunk count) across server restarts so the frontend can resume a
    previous session without re-uploading. Messages and citations are stored in
    chat_messages and removed by a cascading foreign key when a session is deleted.

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
import uuid
from typing import Optional

from psycopg.types.json import Jsonb

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
                conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        sequence BIGSERIAL PRIMARY KEY,
                        id TEXT NOT NULL UNIQUE,
                        session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        sources JSONB NOT NULL DEFAULT '[]',
                        elapsed DOUBLE PRECISION,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON chat_messages(session_id, sequence)")
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


def save_session(session_id: str, doc_name: str, pages: int, chunks: int, user_id: str) -> None:
    """Insert or update a session row (upsert on primary key)."""
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, doc_name, pages, chunks, user_id)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE
                SET doc_name = EXCLUDED.doc_name,
                    pages    = EXCLUDED.pages,
                    chunks   = EXCLUDED.chunks
            """,
            (session_id, doc_name, pages, chunks, user_id),
        )
        conn.commit()


def get_session(session_id: str, user_id: str) -> Optional[dict]:
    """
    Fetch a session record by id.

    Returns a dict with keys ``session_id``, ``doc_name``, ``pages``,
    ``chunks``, or ``None`` when the session does not exist.
    """
    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT session_id, doc_name, pages, chunks FROM sessions WHERE session_id = %s AND user_id = %s",
            (session_id, user_id),
        ).fetchone()
    if row is None:
        return None
    return {"session_id": row[0], "doc_name": row[1], "pages": row[2], "chunks": row[3]}


def delete_session(session_id: str) -> None:
    """Delete a session record. No-op if the session does not exist."""
    with get_db_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        conn.commit()


def list_sessions(user_id: str) -> list[dict]:
    """Return only this account's documents, most recently active first."""
    with get_db_conn() as conn:
        rows = conn.execute("""
            SELECT session_id, doc_name, pages, chunks, created_at, updated_at
            FROM sessions WHERE user_id = %s
            ORDER BY updated_at DESC, session_id
        """, (user_id,)).fetchall()
    keys = ("session_id", "doc_name", "pages", "chunks", "created_at", "updated_at")
    return [dict(zip(keys, row)) for row in rows]


def get_messages(session_id: str, user_id: str, limit: int | None = None) -> list[dict]:
    """Read messages in order; ownership is enforced in the query itself."""
    with get_db_conn() as conn:
        rows = conn.execute("""
            SELECT m.id, m.role, m.content, m.sources, m.elapsed, m.created_at
            FROM chat_messages m JOIN sessions s ON s.session_id = m.session_id
            WHERE m.session_id = %s AND s.user_id = %s
            ORDER BY m.sequence DESC LIMIT %s
        """, (session_id, user_id, limit)).fetchall()
    keys = ("id", "role", "content", "sources", "elapsed", "created_at")
    return [dict(zip(keys, row)) for row in reversed(rows)]


def save_turn(session_id: str, user_id: str, question: str, result: dict) -> list[dict]:
    """Commit both messages and activity together, or leave history untouched."""
    messages = [
        {"id": str(uuid.uuid4()), "role": "user", "content": question, "sources": [], "elapsed": None},
        {"id": str(uuid.uuid4()), "role": "assistant", "content": result["answer"],
         "sources": result["sources"], "elapsed": result["elapsed"]},
    ]
    with get_db_conn() as conn:
        row = conn.execute("""
            UPDATE sessions SET updated_at = clock_timestamp()
            WHERE session_id = %s AND user_id = %s RETURNING session_id
        """, (session_id, user_id)).fetchone()
        if row is None:
            raise ValueError("Session not found.")
        for message in messages:
            conn.execute("""
                INSERT INTO chat_messages (id, session_id, role, content, sources, elapsed)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (message["id"], session_id, message["role"], message["content"],
                  Jsonb(message["sources"]), message["elapsed"]))
    return messages
