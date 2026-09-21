"""
app/pipeline/generation.py
───────────────────────────
Stage 4 (final) of the RAG pipeline: answer generation.

This module owns:
- The system prompt that instructs the LLM how to answer from retrieved context.
- ``answer_question()`` — the top-level entry point that wires retrieval →
  context expansion → LLM generation into a single call.
- ``_source_from_doc()`` — a helper that shapes a Document into the
  source-citation dict returned to the frontend.

Public API
──────────
    answer_question(session_id, query) -> dict
        Returns: {"answer": str, "sources": list[dict], "elapsed": float}
"""

import logging
import time
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from app.services.llm import get_llm
from app.pipeline.retrieval import hybrid_retrieve
from app.pipeline.context import expand_context

logger = logging.getLogger("docchat.pipeline.generation")


# ── System prompt ─────────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _source_from_doc(doc: Document) -> dict:
    """
    Convert a retrieved Document into a source-citation dict for the API response.

    Returns
    ───────
    dict with keys:
    - ``page``      — 1-indexed start page (or ``None`` for unknown)
    - ``page_end``  — 1-indexed end page when different from start (else ``None``)
    - ``section``   — breadcrumb string (e.g. "3 > Revenue > Recognition Policy")
    - ``type``      — ``"text"`` or ``"table"``
    - ``excerpt``   — first 400 characters of the chunk body
    """
    m = doc.metadata
    body = doc.page_content[m.get("body_offset", 0):]

    page_start = m.get("page_start")
    if page_start is None and isinstance(m.get("page"), int):
        page_start = m["page"] + 1    # legacy vectors: 0-indexed page

    page_end = m.get("page_end")
    return {
        "page":     page_start,
        "page_end": page_end if page_end != page_start else None,
        "section":  m.get("breadcrumb") or None,
        "type":     m.get("chunk_type", "text"),
        "excerpt":  body[:400] + ("…" if len(body) > 400 else ""),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def answer_question(session_id: str, query: str) -> dict:
    """
    Run the full RAG pipeline for a user question and return the answer.

    Pipeline
    ────────
    hybrid_retrieve()     — BM25 + dense + RRF + rerank
      → expand_context()  — small-to-big neighbour expansion
        → LLM generation  — stuff-documents chain with RAG_PROMPT

    Parameters
    ──────────
    session_id:
        Restricts retrieval to chunks from this document session.
    query:
        The user's raw question string.

    Returns
    ───────
    dict with:
    - ``answer``  — LLM-generated answer string (markdown)
    - ``sources`` — list of source-citation dicts (from the *retrieved* docs,
                    not the expanded context, so citations stay precise)
    - ``elapsed`` — wall-clock time in seconds for the full pipeline
    """
    llm = get_llm()
    doc_chain = create_stuff_documents_chain(llm, RAG_PROMPT)

    t0 = time.perf_counter()

    top_docs     = hybrid_retrieve(session_id, query, k_final=5, k_candidates=20)
    context_docs = expand_context(session_id, top_docs)
    answer       = doc_chain.invoke({"context": context_docs, "input": query})

    elapsed = time.perf_counter() - t0

    logger.info(
        "answered session=%s elapsed=%.3fs docs=%d",
        session_id, elapsed, len(top_docs),
    )

    return {
        "answer":  answer,
        "sources": [_source_from_doc(d) for d in top_docs],  # retrieved, not expanded
        "elapsed": round(elapsed, 3),
    }
