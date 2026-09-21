"""
app/services/reranker.py
─────────────────────────
Singleton accessor for the reranking function.

Strategy (decided at startup based on environment)
──────────────────────────────────────────────────
- **Cohere Rerank** (preferred): if ``COHERE_API_KEY`` is set, uses the
  Cohere cross-encoder model for high-quality reranking.
- **Passthrough** (fallback): if the key is missing, logs a one-time
  warning and simply truncates the fused BM25+dense ranking to ``top_n``
  results.

The returned callable has a stable signature regardless of which strategy
is active::

    reranker(query: str, docs: List[Document], top_n: int) -> List[Document]

This means the pipeline code in :mod:`app.pipeline.retrieval` never needs
to know which backend is in use.
"""

import logging
from functools import lru_cache
from typing import List

import cohere
from langchain_core.documents import Document

from app.config import COHERE_API_KEY, RERANK_MODEL

logger = logging.getLogger("docchat.services.reranker")


@lru_cache(maxsize=1)
def get_reranker():
    """
    Return a reranking callable ``(query, docs, top_n) -> docs``.

    The callable is created once and cached for the lifetime of the process.
    """
    if not COHERE_API_KEY:
        logger.warning(
            "COHERE_API_KEY not set — reranking is disabled and the fused "
            "BM25+dense ranking will be used as-is. Set COHERE_API_KEY to "
            "enable cross-encoder reranking for better answer quality."
        )

        def _passthrough(query: str, docs: List[Document], top_n: int) -> List[Document]:
            return docs[:top_n]

        return _passthrough

    co = cohere.ClientV2(COHERE_API_KEY)

    def _cohere_rerank(query: str, docs: List[Document], top_n: int) -> List[Document]:
        if not docs:
            return []
        result = co.rerank(
            query=query,
            documents=[d.page_content for d in docs],
            top_n=min(top_n, len(docs)),
            model=RERANK_MODEL,
        )
        return [docs[r.index] for r in result.results]

    return _cohere_rerank
