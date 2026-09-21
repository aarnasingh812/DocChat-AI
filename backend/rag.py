import os
import re
import time
import logging
from collections import OrderedDict
from functools import lru_cache
from threading import Lock
from typing import List, NamedTuple, Optional

import psycopg
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres.vectorstores import PGVector

from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from google import genai
import cohere

from chunking import Chunk, ChunkingConfig, chunk_pdf, contextualize
from ocr import DEFAULT_MODEL as OCR_DEFAULT_MODEL, GeminiVisionOCR, PostgresDailyBudget

load_dotenv()

logger = logging.getLogger("docchat.rag")

# How much surrounding text is handed to the LLM around each retrieved chunk.
# window=1 -> the chunk plus its previous/next chunk *within the same section*.
CONTEXT_NEIGHBOR_WINDOW = int(os.getenv("CONTEXT_NEIGHBOR_WINDOW", "1"))
CONTEXT_MAX_CHARS_PER_HIT = int(os.getenv("CONTEXT_MAX_CHARS_PER_HIT", "3600"))

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM GEMINI EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

class GeminiEmbeddings(Embeddings):
    """Thin wrapper around the Google GenAI embed_content API."""

    _BATCH = 100  # embed_content accepts a bounded number of inputs per request

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self._BATCH):
            result = self._client.models.embed_content(
                model=self._model,
                contents=texts[i : i + self._BATCH],
            )
            vectors.extend(e.values for e in result.embeddings)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        return result.embeddings[0].values


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETONS  (created once per process)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-120b",
    )


@lru_cache(maxsize=1)
def get_embeddings() -> GeminiEmbeddings:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and add it to your .env file."
        )
    return GeminiEmbeddings(api_key=api_key, model="gemini-embedding-001")


@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    connection = os.getenv("DATABASE_URL")
    if not connection:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env file, e.g.\n"
            "  DATABASE_URL=postgresql+psycopg://docchat:docchat@localhost:5432/docchat"
        )
    return PGVector(
        embeddings=get_embeddings(),
        collection_name="docchat_chunks",
        connection=connection,
        use_jsonb=True,          # store metadata as JSONB for fast filtering
        create_extension=True,   # run CREATE EXTENSION IF NOT EXISTS vector
    )


@lru_cache(maxsize=1)
def get_reranker():
    """
    Returns a callable: (query: str, docs: List[Document], top_n: int) -> List[Document]
    Uses Cohere Rerank if COHERE_API_KEY is set. Otherwise falls back to a
    no-op that just truncates the fused ranking (logged once).
    """
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        logger.warning(
            "COHERE_API_KEY not set — reranking is disabled and the fused "
            "BM25+dense ranking will be used as-is. Set COHERE_API_KEY to "
            "enable cross-encoder reranking for better answer quality."
        )

        def _passthrough(query: str, docs: List[Document], top_n: int) -> List[Document]:
            return docs[:top_n]

        return _passthrough

    co = cohere.ClientV2(api_key)

    def _cohere_rerank(query: str, docs: List[Document], top_n: int) -> List[Document]:
        if not docs:
            return []
        result = co.rerank(
            query=query,
            documents=[d.page_content for d in docs],
            top_n=min(top_n, len(docs)),
            model="rerank-english-v3.0",
        )
        return [docs[r.index] for r in result.results]

    return _cohere_rerank


# ─────────────────────────────────────────────────────────────────────────────
# OCR FALLBACK  (scanned pages / pages with images -> Gemini vision)
#
# Tuned for the Gemini FREE tier. Google publishes per-model limits only in AI Studio
# (https://aistudio.google.com/rate-limit) and they differ per project, so nothing here
# is hard-coded to a number Google might change:
#
#   OCR_ENABLED               true|false                       (default true)
#   OCR_MODEL                 model id                         (default gemini-3.5-flash-lite)
#   OCR_RPM                   requests/minute to stay under    (default 10)
#   OCR_DAILY_REQUEST_BUDGET  requests/day this app may spend  (default 400) — set it to ~80% of
#                             the RPD AI Studio shows for OCR_MODEL. Resets at Pacific midnight.
#   OCR_PAGES_PER_REQUEST     pages sent per API call          (default 3)  — more = fewer requests
#   OCR_CONCURRENCY           parallel requests                (default 2)
#   OCR_MAX_PAGES             max pages OCR'd per document     (default 24)
#   OCR_FIGURES               also OCR text inside figures on pages that have a text layer (default true)
#
# PRIVACY: content sent through Google's unpaid tier may be used to improve Google products
# and can be reviewed by humans. Don't enable this for confidential documents on the free tier.
# ─────────────────────────────────────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def get_chunking_config() -> ChunkingConfig:
    return ChunkingConfig(
        ocr_max_pages=int(os.getenv("OCR_MAX_PAGES", "24")),
        ocr_figures=_env_bool("OCR_FIGURES", True),
    )


@lru_cache(maxsize=1)
def get_ocr() -> Optional[GeminiVisionOCR]:
    """The vision-OCR provider, or None if disabled / no API key (scanned pages are then reported, not indexed)."""
    if not _env_bool("OCR_ENABLED", True):
        logger.info("OCR disabled via OCR_ENABLED")
        return None
    api_key = os.getenv("OCR_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("No GOOGLE_API_KEY — OCR fallback disabled; scanned pages will not be searchable.")
        return None
    return GeminiVisionOCR(
        api_key=api_key,
        model=os.getenv("OCR_MODEL", OCR_DEFAULT_MODEL),
        rpm=float(os.getenv("OCR_RPM", "10")),
        pages_per_request=int(os.getenv("OCR_PAGES_PER_REQUEST", "3")),
        concurrency=int(os.getenv("OCR_CONCURRENCY", "2")),
        daily_budget=PostgresDailyBudget(_get_db_conn, int(os.getenv("OCR_DAILY_REQUEST_BUDGET", "400"))),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SESSION CHUNK STORE  (Postgres-backed — source of truth for BM25 text and for
# neighbour expansion)
#
# One row per chunk. `text` holds the chunk BODY only; the heading breadcrumb and
# page range live in their own columns and are recombined by `contextualize()`
# wherever text is embedded, keyword-indexed or shown to the LLM.
#
# Migration is additive: rows written by the previous fixed-size pipeline have
# NULL in the new columns and keep working (no neighbour expansion, page taken
# from the legacy 0-indexed `page` column).
# ─────────────────────────────────────────────────────────────────────────────

_ROW_COLUMNS = "chunk_id, section_id, seq, chunk_type, page_start, page_end, breadcrumb, page, text"

_schema_lock = Lock()
_schema_ready = False


def _get_db_conn() -> psycopg.Connection:
    url = os.getenv("DATABASE_URL", "")
    native_url = url.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(native_url)


def init_session_chunks_table() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        with _get_db_conn() as conn:
            # serialise DDL across uvicorn workers
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('docchat_session_chunks_schema'))")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_chunks (
                    chunk_id   TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    page       INTEGER,
                    text       TEXT NOT NULL
                )
            """)
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


def _save_session_chunks(session_id: str, chunks: List[Chunk], chunk_ids: List[str]) -> None:
    rows = [
        (
            chunk_id, session_id,
            c.page_start - 1,          # legacy column: keep it 0-indexed like before
            c.text, c.section_id, c.seq, c.kind,
            c.page_start, c.page_end, c.breadcrumb or None,
        )
        for chunk_id, c in zip(chunk_ids, chunks)
    ]
    with _get_db_conn() as conn:
        # NB: psycopg3's Connection has no executemany(); it lives on the cursor.
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


def _row_to_dict(row: tuple) -> dict:
    (chunk_id, section_id, seq, chunk_type, page_start, page_end, breadcrumb, legacy_page, text) = row
    if page_start is None and legacy_page is not None:
        page_start = legacy_page + 1           # legacy rows stored 0-indexed pages
    return {
        "chunk_id": chunk_id,
        "section_id": section_id,
        "seq": seq,
        "chunk_type": chunk_type or "text",
        "page_start": page_start,
        "page_end": page_end if page_end is not None else page_start,
        "breadcrumb": breadcrumb or "",
        "text": text,
    }


def _load_session_chunks(session_id: str) -> List[dict]:
    with _get_db_conn() as conn:
        rows = conn.execute(
            f"SELECT {_ROW_COLUMNS} FROM session_chunks WHERE session_id = %s ORDER BY seq NULLS LAST",
            (session_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _load_session_chunk_ids(session_id: str) -> List[str]:
    with _get_db_conn() as conn:
        rows = conn.execute(
            "SELECT chunk_id FROM session_chunks WHERE session_id = %s", (session_id,)
        ).fetchall()
    return [r[0] for r in rows]


def _delete_session_chunks_row(session_id: str) -> None:
    with _get_db_conn() as conn:
        conn.execute("DELETE FROM session_chunks WHERE session_id = %s", (session_id,))
        conn.commit()


def _chunk_to_document(c: dict, session_id: str) -> Document:
    """Rebuild a Document (contextualised text + metadata) from a session_chunks row."""
    content, offset = _contextualize_row(c)
    return Document(
        page_content=content,
        metadata={
            "chunk_id": c["chunk_id"],
            "session_id": session_id,
            "section_id": c["section_id"],
            "seq": c["seq"],
            "chunk_type": c["chunk_type"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "breadcrumb": c["breadcrumb"],
            "body_offset": offset,
        },
    )


def _contextualize_row(c: dict) -> tuple[str, int]:
    return contextualize(c["breadcrumb"], c["page_start"], c["page_end"], c["chunk_type"], c["text"])


# ─────────────────────────────────────────────────────────────────────────────
# BM25 INDEX CACHE  (in-memory, per session_id, thread-safe, size-capped)
# ─────────────────────────────────────────────────────────────────────────────

_BM25_CACHE_MAX_SESSIONS = 200
_bm25_cache: "OrderedDict[str, tuple[BM25Okapi, List[dict]]]" = OrderedDict()
_bm25_cache_lock = Lock()
# Per-session build locks prevent concurrent threads from double-building
# the same index (TOCTOU).  Keyed by session_id; cleaned up after the
# result is stored.
_bm25_build_locks: "dict[str, Lock]" = {}
_bm25_build_locks_lock = Lock()

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _build_bm25_index(session_id: str) -> Optional[tuple[BM25Okapi, List[dict]]]:
    chunks = _load_session_chunks(session_id)
    if not chunks:
        return None
    # Index the CONTEXTUALISED text so a query like "revenue recognition policy"
    # can hit a chunk whose body never repeats the words of its section title.
    tokenized_corpus = [_tokenize(_contextualize_row(c)[0]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks


def _get_bm25_index(session_id: str) -> Optional[tuple[BM25Okapi, List[dict]]]:
    # Fast path — already cached.
    with _bm25_cache_lock:
        cached = _bm25_cache.get(session_id)
        if cached is not None:
            _bm25_cache.move_to_end(session_id)
            return cached

    # Acquire (or create) a per-session build lock so that only ONE thread
    # ever runs _build_bm25_index for this session at a time.
    with _bm25_build_locks_lock:
        if session_id not in _bm25_build_locks:
            _bm25_build_locks[session_id] = Lock()
        build_lock = _bm25_build_locks[session_id]

    with build_lock:
        # Double-checked: another thread may have finished while we waited.
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
            while len(_bm25_cache) > _BM25_CACHE_MAX_SESSIONS:
                _bm25_cache.popitem(last=False)

    # Clean up the build lock entry so the dict doesn't grow unboundedly.
    with _bm25_build_locks_lock:
        _bm25_build_locks.pop(session_id, None)

    return built


def _invalidate_bm25_cache(session_id: str) -> None:
    with _bm25_cache_lock:
        _bm25_cache.pop(session_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL: BM25 + dense embeddings, fused with Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_search(session_id: str, query: str, k: int) -> List[Document]:
    index = _get_bm25_index(session_id)
    if index is None:
        return []
    bm25, chunks = index
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [_chunk_to_document(c, session_id) for c, score in ranked[:k] if score > 0]


def _dense_search(session_id: str, query: str, k: int) -> List[Document]:
    store = get_vector_store()
    return store.similarity_search(query, k=k, filter={"session_id": session_id})


def _doc_key(doc: Document) -> str:
    # Prefer a stable chunk id; fall back to content+page for dense results
    # that may not carry our chunk_id (e.g. pre-existing PGVector rows).
    return doc.metadata.get("chunk_id") or f"{doc.metadata.get('page')}:{hash(doc.page_content)}"


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Document]], k: int, rrf_k: int = 60
) -> List[Document]:
    scores: dict = {}
    doc_by_key: dict = {}
    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_by_key.setdefault(key, doc)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_by_key[key] for key in ranked_keys[:k]]


def hybrid_retrieve(
    session_id: str,
    query: str,
    k_final: int = 5,
    k_candidates: int = 20,
) -> List[Document]:
    dense_docs = _dense_search(session_id, query, k=k_candidates)
    bm25_docs = _bm25_search(session_id, query, k=k_candidates)

    fused = _reciprocal_rank_fusion([dense_docs, bm25_docs], k=k_candidates)

    reranker = get_reranker()
    return reranker(query, fused, k_final)


# ─────────────────────────────────────────────────────────────────────────────
# SMALL-TO-BIG CONTEXT EXPANSION
#
# Retrieval runs on small, precise chunks; generation gets a little more. For
# every retrieved TEXT chunk we add its neighbours from the SAME section (never
# across a heading boundary), up to a character budget. Tables are already
# self-contained, so they are passed through untouched.
# ─────────────────────────────────────────────────────────────────────────────

def expand_context(
    session_id: str,
    docs: List[Document],
    window: int = CONTEXT_NEIGHBOR_WINDOW,
    max_chars: int = CONTEXT_MAX_CHARS_PER_HIT,
) -> List[Document]:
    if window <= 0 or not docs:
        return docs

    section_ids = sorted({
        d.metadata["section_id"] for d in docs
        if d.metadata.get("section_id") is not None and d.metadata.get("chunk_type") != "table"
    })
    if not section_ids:
        return docs

    with _get_db_conn() as conn:
        rows = conn.execute(
            f"SELECT {_ROW_COLUMNS} FROM session_chunks "
            "WHERE session_id = %s AND section_id = ANY(%s)",
            (session_id, section_ids),
        ).fetchall()
    by_seq = {r["seq"]: r for r in map(_row_to_dict, rows) if r["seq"] is not None}

    covered: set = set()          # seqs already sent to the LLM (avoid duplicate text)
    out: List[Document] = []

    for d in docs:                # docs arrive in relevance order; keep it
        m = d.metadata
        seq, sec = m.get("seq"), m.get("section_id")
        if seq is None or sec is None or seq not in by_seq:
            out.append(d)         # legacy / unknown chunk: pass through
            continue
        if seq in covered:
            continue              # already inside a higher-ranked chunk's window
        covered.add(seq)
        if m.get("chunk_type") == "table":
            out.append(d)
            continue

        edges = {-1: seq, 1: seq}
        blocked = {-1: False, 1: False}
        size = len(by_seq[seq]["text"])
        for _ in range(window):
            for step in (-1, 1):
                if blocked[step]:
                    continue
                cand = by_seq.get(edges[step] + step)
                if (
                    cand is None
                    or cand["section_id"] != sec
                    or cand["seq"] in covered
                    or size + 2 + len(cand["text"]) > max_chars
                ):
                    blocked[step] = True
                    continue
                edges[step] = cand["seq"]
                covered.add(cand["seq"])
                size += 2 + len(cand["text"])

        span = [by_seq[s] for s in range(edges[-1], edges[1] + 1)]
        if len(span) == 1:
            out.append(d)
            continue
        body = "\n\n".join(r["text"] for r in span)
        page_starts = [r["page_start"] for r in span if r["page_start"] is not None]
        page_ends   = [r["page_end"]   for r in span if r["page_end"]   is not None]
        p_start = min(page_starts) if page_starts else None
        p_end   = max(page_ends)   if page_ends   else None
        content, offset = contextualize(by_seq[seq]["breadcrumb"], p_start, p_end, "text", body)
        out.append(Document(
            page_content=content,
            metadata={**m, "page_start": p_start, "page_end": p_end,
                      "body_offset": offset, "expanded": True},
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# RAG PROMPT
# ─────────────────────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a knowledgeable and helpful document assistant. Your job is to answer questions \
accurately based on the provided document context.

Each passage in the context starts with a bracketed label giving its section path and \
page number(s). Tables are given as markdown — read their rows and columns carefully.

Guidelines:
- Answer clearly and concisely, using the context below.
- Use markdown formatting (bold, bullet lists, etc.) to improve readability when helpful.
- When it helps the reader verify the answer, mention the section or page it came from.
- If the answer is NOT found in the context, say exactly:
  "I couldn't find that information in the uploaded document. Try rephrasing your question \
or ask about a topic covered in the document."
- Never fabricate information. Never go beyond what the context says.

<context>
{context}
</context>

User Question: {input}
""")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

class IngestResult(NamedTuple):
    pages: int
    chunks: int
    warnings: List[str]      # e.g. "Pages 25-40 were not OCR'd: over the per-document OCR limit of 24 pages."


def build_vector_store(pdf_bytes: bytes, session_id: str) -> IngestResult:
    """
    Parse the PDF into structure-aware chunks (running vision OCR on scanned / image
    pages), embed them, and persist them. Raises ValueError for PDFs that can't be
    processed (encrypted, corrupt, or no text even after OCR).
    """
    init_session_chunks_table()

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
                "session_id": session_id,
                "chunk_id": chunk_id,
                "section_id": c.section_id,
                "seq": c.seq,
                "chunk_type": c.kind,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "breadcrumb": c.breadcrumb,
                "body_offset": offset,
            },
        ))

    try:
        get_vector_store().add_documents(docs, ids=chunk_ids)
        _save_session_chunks(session_id, chunks, chunk_ids)
    except Exception:
        # Don't leave half-ingested sessions behind. Use the ids we just generated:
        # session_chunks may not have been written yet, so it can't be the source.
        try:
            get_vector_store().delete(ids=chunk_ids)
            _delete_session_chunks_row(session_id)
            _invalidate_bm25_cache(session_id)
        except Exception:
            logger.exception("Cleanup after failed ingest of session %s also failed", session_id)
        raise

    logger.info(
        "ingested session=%s pages=%d sections=%d chunks=%d (tables=%d, ocr_pages=%d, warnings=%d)",
        session_id, result.page_count, result.section_count, len(chunks),
        sum(1 for c in chunks if c.kind == "table"), len(result.ocr_pages), len(result.warnings),
    )
    return IngestResult(result.page_count, len(chunks), result.warnings)


def _source_from_doc(doc: Document) -> dict:
    m = doc.metadata
    body = doc.page_content[m.get("body_offset", 0):]
    page_start = m.get("page_start")
    if page_start is None and isinstance(m.get("page"), int):
        page_start = m["page"] + 1              # legacy vectors: 0-indexed page
    page_end = m.get("page_end")
    return {
        "page":     page_start,
        "page_end": page_end if page_end != page_start else None,
        "section":  m.get("breadcrumb") or None,
        "type":     m.get("chunk_type", "text"),
        "excerpt":  body[:400] + ("…" if len(body) > 400 else ""),
    }


def answer_question(session_id: str, query: str) -> dict:
    """
    Run the hybrid RAG pipeline (BM25 + dense, RRF-fused, reranked),
    retrieving only chunks that belong to *session_id*, then widen each
    retrieved chunk to its section neighbours before generation.
    """
    llm = get_llm()
    doc_chain = create_stuff_documents_chain(llm, RAG_PROMPT)

    t0 = time.perf_counter()
    top_docs = hybrid_retrieve(session_id, query, k_final=5, k_candidates=20)
    context_docs = expand_context(session_id, top_docs)
    answer = doc_chain.invoke({"context": context_docs, "input": query})
    elapsed = time.perf_counter() - t0

    return {
        "answer":  answer,
        "sources": [_source_from_doc(d) for d in top_docs],   # what matched, not the padding
        "elapsed": round(elapsed, 3),
    }


def delete_session_chunks(session_id: str) -> None:
    # PGVector.delete() ignores `filter=`; it only honours `ids=`. Take the ids
    # from our own table so vectors are really removed.
    chunk_ids = _load_session_chunk_ids(session_id)
    if chunk_ids:
        get_vector_store().delete(ids=chunk_ids)
    _delete_session_chunks_row(session_id)
    _invalidate_bm25_cache(session_id)
