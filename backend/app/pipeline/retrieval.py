"""
app/pipeline/retrieval.py
──────────────────────────
Stage 2 of the RAG pipeline: hybrid retrieval.

Strategy
────────
1. **Dense search** — similarity search in PGVector using Gemini embeddings.
   Excels at semantic / paraphrased queries.
2. **BM25 search** — keyword-based scoring over in-memory BM25 index.
   Excels at exact-match queries (product codes, names, rare terms).
3. **Reciprocal Rank Fusion (RRF)** — merges both ranked lists into a single
   de-duplicated ranking without requiring score normalisation.
4. **Reranking** — optionally applies a Cohere cross-encoder to re-score the
   fused candidates and return the final ``k_final`` documents.

Public API
──────────
    hybrid_retrieve(session_id, query, k_final, k_candidates) -> List[Document]
"""

import logging
from typing import List

from langchain_core.documents import Document

from app.pipeline.bm25_index import bm25_search
from app.services.vector_store import get_vector_store
from app.services.reranker import get_reranker

logger = logging.getLogger("docchat.pipeline.retrieval")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dense_search(session_id: str, query: str, k: int) -> List[Document]:
    """Run a PGVector similarity search filtered to the given session."""
    store = get_vector_store()
    return store.similarity_search(query, k=k, filter={"session_id": session_id})


def _doc_key(doc: Document) -> str:
    """
    Stable identifier for a Document used during RRF de-duplication.

    Prefers the ``chunk_id`` metadata field (always set for our chunks).
    Falls back to a content hash for legacy PGVector rows that pre-date the
    chunk_id column.
    """
    return doc.metadata.get("chunk_id") or f"{doc.metadata.get('page')}:{hash(doc.page_content)}"


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int,
    rrf_k: int = 60,
) -> List[Document]:
    """
    Merge multiple ranked document lists into one using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ  1 / (rrf_k + rank(d, list))
    where the sum is over every list that contains d.

    Parameters
    ──────────
    ranked_lists:
        Each inner list is a ranked list of Documents from one retriever.
    k:
        Number of top documents to return.
    rrf_k:
        Smoothing constant (default 60 as per the original RRF paper).
    """
    scores: dict = {}
    doc_by_key: dict = {}

    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_by_key.setdefault(key, doc)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_by_key[key] for key in ranked_keys[:k]]


# ── Public API ────────────────────────────────────────────────────────────────

def hybrid_retrieve(
    session_id: str,
    query: str,
    k_final: int = 5,
    k_candidates: int = 20,
) -> List[Document]:
    """
    Run hybrid retrieval and return the top ``k_final`` documents.

    Pipeline
    ────────
    dense_search(k_candidates) + bm25_search(k_candidates)
      → RRF fusion (k_candidates)
        → reranker (k_final)

    Parameters
    ──────────
    session_id:
        Restricts both searches to chunks belonging to this session.
    query:
        The user's question (raw, unmodified).
    k_final:
        Number of documents returned to the caller.
    k_candidates:
        Number of candidates from each retriever before fusion/reranking.
        Higher values improve recall but increase reranker latency.
    """
    dense_docs = _dense_search(session_id, query, k=k_candidates)
    bm25_docs  = bm25_search(session_id, query, k=k_candidates)

    fused = _reciprocal_rank_fusion([dense_docs, bm25_docs], k=k_candidates)

    reranker = get_reranker()
    return reranker(query, fused, k_final)
