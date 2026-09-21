"""
app/services/vector_store.py
─────────────────────────────
Singleton accessor for the PGVector vector store.

PGVector is the persistence layer for dense embeddings.  One collection
(``docchat_chunks``) stores all chunks from all sessions; each chunk's
``session_id`` metadata field is used as a filter during retrieval so
that queries never surface results from other users' documents.

The store is initialised lazily on first access and cached for the
lifetime of the process.
"""

from functools import lru_cache

from langchain_postgres.vectorstores import PGVector

from app.config import DATABASE_URL, VECTOR_COLLECTION
from app.embeddings import GeminiEmbeddings


@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    """
    Return the shared PGVector store instance.

    Created once per process.  Requires ``DATABASE_URL`` to be set in the
    environment (see :mod:`app.config`).

    Raises ``RuntimeError`` if ``DATABASE_URL`` is missing.
    """
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env file, e.g.\n"
            "  DATABASE_URL=postgresql+psycopg://docchat:docchat@localhost:5432/docchat"
        )
    return PGVector(
        embeddings=GeminiEmbeddings(),
        collection_name=VECTOR_COLLECTION,
        connection=DATABASE_URL,
        use_jsonb=True,        # store metadata as JSONB for fast filtering
        create_extension=True, # run CREATE EXTENSION IF NOT EXISTS vector
    )
