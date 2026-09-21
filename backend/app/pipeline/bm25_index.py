"""
app/pipeline/bm25_index.py
───────────────────────────
In-memory BM25 index cache, one index per active session.

Why BM25?
─────────
Dense embeddings excel at semantic similarity but can miss exact keyword
matches (e.g. product codes, proper nouns, rare technical terms). BM25
fills that gap.  The two signals are later fused via Reciprocal Rank Fusion
in :mod:`app.pipeline.retrieval`.

Architecture
────────────
- Indexes are built lazily on first query and cached in an LRU-style
  ``OrderedDict`` capped at ``_BM25_CACHE_MAX_SESSIONS`` entries.
- A per-session ``Lock`` prevents duplicate index builds under concurrent
  requests (TOCTOU guard).
- ``invalidate_bm25_cache()`` is called after ingest and deletion so the
  next query always sees up-to-date data.

Public API
──────────
    bm25_search(session_id, query, k) -> List[Document]
    invalidate_bm25_cache(session_id) -> None
"""

import logging
import re
from collections import OrderedDict
from threading import Lock
from typing import List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.database.chunk_store import load_session_chunks
from chunking import contextualize

logger = logging.getLogger("docchat.pipeline.bm25_index")

# ── Cache configuration ───────────────────────────────────────────────────────

_BM25_CACHE_MAX_SESSIONS = 200
"""Maximum number of BM25 indexes held in memory simultaneously."""

_bm25_cache: "OrderedDict[str, tuple[BM25Okapi, List[dict]]]" = OrderedDict()
_bm25_cache_lock = Lock()

# Per-session build locks prevent concurrent threads from double-building
# the same index (TOCTOU).  Keyed by session_id; cleaned up after the
# result is stored.
_bm25_build_locks: "dict[str, Lock]" = {}
_bm25_build_locks_lock = Lock()

# ── Tokeniser ─────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lower-case word/number tokeniser — fast and sufficient for BM25."""
    return _TOKEN_RE.findall(text.lower())


# ── Index build / cache ───────────────────────────────────────────────────────

def _contextualize_row(c: dict) -> tuple[str, int]:
    """Re-contextualise a chunk dict (breadcrumb + page header + body)."""
    return contextualize(c["breadcrumb"], c["page_start"], c["page_end"], c["chunk_type"], c["text"])


def _build_bm25_index(session_id: str) -> Optional[tuple[BM25Okapi, List[dict]]]:
    """
    Load all chunks for a session from Postgres and build a BM25Okapi index.

    The *contextualised* text (breadcrumb + body) is indexed so that a query
    like "revenue recognition policy" can match a chunk whose body never
    repeats the words of its section title.

    Returns ``None`` if the session has no chunks yet.
    """
    chunks = load_session_chunks(session_id)
    if not chunks:
        return None
    tokenized_corpus = [_tokenize(_contextualize_row(c)[0]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks


def _get_bm25_index(session_id: str) -> Optional[tuple[BM25Okapi, List[dict]]]:
    """
    Return the cached BM25 index for ``session_id``, building it on cache miss.

    Thread-safe: uses a global cache lock for reads and a per-session lock
    for builds to avoid duplicate work.
    """
    # Fast path — already in cache
    with _bm25_cache_lock:
        cached = _bm25_cache.get(session_id)
        if cached is not None:
            _bm25_cache.move_to_end(session_id)
            return cached

    # Acquire (or create) a per-session build lock
    with _bm25_build_locks_lock:
        if session_id not in _bm25_build_locks:
            _bm25_build_locks[session_id] = Lock()
        build_lock = _bm25_build_locks[session_id]

    with build_lock:
        # Double-checked: another thread may have finished while we waited
        with _bm25_cache_lock:
            cached = _bm25_cache.get(session_id)
            if cached is not None:
                _bm25_cache.move_to_end(session_id)
                return cached

        built = _build_bm25_index(session_id)
        if built is None:
            return None

        with _bm25_cache_lock:
            _bm25_cache[session_id] = built
            _bm25_cache.move_to_end(session_id)
            # Evict least-recently-used entries when over the cap
            while len(_bm25_cache) > _BM25_CACHE_MAX_SESSIONS:
                _bm25_cache.popitem(last=False)

    # Clean up build-lock entry so the dict doesn't grow unboundedly
    with _bm25_build_locks_lock:
        _bm25_build_locks.pop(session_id, None)

    return built


# ── Public API ────────────────────────────────────────────────────────────────

def bm25_search(session_id: str, query: str, k: int) -> List[Document]:
    """
    Return up to ``k`` Documents ranked by BM25 relevance.

    Only chunks with a positive BM25 score are returned (zero-score results
    are excluded to avoid polluting the fused ranking with irrelevant text).
    """
    index = _get_bm25_index(session_id)
    if index is None:
        return []
    bm25, chunks = index
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)

    docs: List[Document] = []
    for c, score in ranked[:k]:
        if score <= 0:
            continue
        content, offset = _contextualize_row(c)
        docs.append(Document(
            page_content=content,
            metadata={
                "chunk_id":    c["chunk_id"],
                "session_id":  session_id,
                "section_id":  c["section_id"],
                "seq":         c["seq"],
                "chunk_type":  c["chunk_type"],
                "page_start":  c["page_start"],
                "page_end":    c["page_end"],
                "breadcrumb":  c["breadcrumb"],
                "body_offset": offset,
            },
        ))
    return docs


def invalidate_bm25_cache(session_id: str) -> None:
    """
    Evict a session's BM25 index from the in-memory cache.

    Call this after ingesting new chunks or deleting a session so the next
    query rebuilds the index from fresh data.
    """
    with _bm25_cache_lock:
        _bm25_cache.pop(session_id, None)
