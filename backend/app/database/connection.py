"""
app/database/connection.py
──────────────────────────
Provides a single factory function `get_db_conn()` that returns a
psycopg (v3) Connection.

All other modules that need raw DB access import from here so that the
connection string is resolved in exactly one place.
"""

import psycopg

from app.config import DATABASE_URL


def get_db_conn() -> psycopg.Connection:
    """
    Open and return a new psycopg3 Connection.

    SQLAlchemy-style URLs (postgresql+psycopg://...) are normalised to the
    native psycopg format (postgresql://...) automatically.

    Usage::

        with get_db_conn() as conn:
            conn.execute("SELECT 1")
    """
    native_url = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(native_url)
