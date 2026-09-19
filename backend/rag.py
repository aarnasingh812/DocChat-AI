import os
import re
import time
import tempfile
import logging
from collections import OrderedDict
from functools import lru_cache
from threading import Lock
from typing import List, Optional

import psycopg
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from google import genai
import cohere

load_dotenv()

logger = logging.getLogger("docchat.rag")

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM GEMINI EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

class GeminiEmbeddings(Embeddings):
    """Thin wrapper around the Google GenAI embed_content API."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=texts,
        )
        return [e.values for e in result.embeddings]

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

    
    co = cohere.Client(api_key)

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
# SESSION CHUNK STORE  (Postgres-backed — source of truth for BM25 text)
#
# PGVector's internal tables aren't a stable place to pull raw chunk text
# from for keyword search, so we keep a lightweight parallel table of
# (session_id, chunk_id, text) that we populate alongside the vector store.
# ─────────────────────────────────────────────────────────────────────────────

def _get_db_conn() -> psycopg.Connection:
    url = os.getenv("DATABASE_URL", "")
    native_url = url.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(native_url)


def init_session_chunks_table() -> None:
    with _get_db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_chunks (
                chunk_id   TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                page       INTEGER,
                text       TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_chunks_session_id
                ON session_chunks (session_id)
        """)
        conn.commit()


def _save_session_chunks(session_id: str, chunks: List[Document], chunk_ids: List[str]) -> None:
    rows = [
        (chunk_id, session_id, chunk.metadata.get("page"), chunk.page_content)
        for chunk_id, chunk in zip(chunk_ids, chunks)
    ]
    with _get_db_conn() as conn:
        conn.executemany(
            """
            INSERT INTO session_chunks (chunk_id, session_id, page, text)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            rows,
        )
        conn.commit()


def _load_session_chunks(session_id: str) -> List[dict]:
    with _get_db_conn() as conn:
        rows = conn.execute(
            "SELECT chunk_id, page, text FROM session_chunks WHERE session_id = %s",
            (session_id,),
        ).fetchall()
    return [{"chunk_id": r[0], "page": r[1], "text": r[2]} for r in rows]


def _delete_session_chunks_row(session_id: str) -> None:
    with _get_db_conn() as conn:
        conn.execute("DELETE FROM session_chunks WHERE session_id = %s", (session_id,))
        conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# BM25 INDEX CACHE  (in-memory, per session_id, thread-safe, size-capped)
# ─────────────────────────────────────────────────────────────────────────────

_BM25_CACHE_MAX_SESSIONS = 200
_bm25_cache: "OrderedDict[str, tuple[BM25Okapi, List[dict]]]" = OrderedDict()
_bm25_cache_lock = Lock()

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _build_bm25_index(session_id: str) -> Optional[tuple[BM25Okapi, List[dict]]]:
    chunks = _load_session_chunks(session_id)
    if not chunks:
        return None
    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks


def _get_bm25_index(session_id: str) -> Optional[tuple[BM25Okapi, List[dict]]]:
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
    return [
        Document(
            page_content=c["text"],
            metadata={"chunk_id": c["chunk_id"], "page": c["page"], "session_id": session_id},
        )
        for c, score in ranked[:k]
        if score > 0
    ]


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
# RAG PROMPT
# ─────────────────────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a knowledgeable and helpful document assistant. Your job is to answer questions \
accurately based on the provided document context.

Guidelines:
- Answer clearly and concisely, using the context below.
- Use markdown formatting (bold, bullet lists, etc.) to improve readability when helpful.
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

def build_vector_store(pdf_bytes: bytes, session_id: str) -> tuple[int, int]:
    init_session_chunks_table()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader   = PyPDFLoader(tmp_path)
        docs     = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200)
        chunks   = splitter.split_documents(docs)

        # ── Tag every chunk with the session_id and a stable chunk id ────────
        chunk_ids = [f"{session_id}:{i}" for i in range(len(chunks))]
        for chunk, chunk_id in zip(chunks, chunk_ids):
            chunk.metadata["session_id"] = session_id
            chunk.metadata["chunk_id"] = chunk_id

        store = get_vector_store()
        store.add_documents(chunks, ids=chunk_ids)

        # Mirror chunk text into our BM25-backing table
        _save_session_chunks(session_id, chunks, chunk_ids)

        return len(docs), len(chunks)
    finally:
        os.unlink(tmp_path)


def answer_question(session_id: str, query: str) -> dict:
    """
    Run the hybrid RAG pipeline (BM25 + dense, RRF-fused, reranked),
    retrieving only chunks that belong to *session_id*.
    """
    llm = get_llm()
    doc_chain = create_stuff_documents_chain(llm, RAG_PROMPT)

    t0 = time.perf_counter()
    top_docs = hybrid_retrieve(session_id, query, k_final=5, k_candidates=20)
    answer = doc_chain.invoke({"context": top_docs, "input": query})
    elapsed = time.perf_counter() - t0

    sources = []
    for chunk in top_docs:
        page_num = chunk.metadata.get("page", None)
        sources.append({
            "page":    (page_num + 1) if isinstance(page_num, int) else None,
            "excerpt": chunk.page_content[:400] + ("…" if len(chunk.page_content) > 400 else ""),
        })

    return {
        "answer":  answer,
        "sources": sources,
        "elapsed": round(elapsed, 3),
    }


def delete_session_chunks(session_id: str) -> None:
    store = get_vector_store()
    store.delete(filter={"session_id": session_id})
    _delete_session_chunks_row(session_id)
    _invalidate_bm25_cache(session_id)