"""
app/pipeline/context.py
────────────────────────
Stage 3 of the RAG pipeline: context expansion (small-to-big).

Motivation
──────────
Retrieval is tuned for *precision* — small, tightly-scoped chunks surface the
most relevant passages.  Generation benefits from *more surrounding text* so
the LLM can answer questions that require a little before/after context
(e.g. "what does the paragraph after the table say?").

Strategy
────────
For every retrieved TEXT chunk we expand its context window by including
up to ``window`` neighbouring chunks from the **same section** (never
across a heading boundary).  A character budget (``max_chars``) prevents
any single window from dominating the LLM's context.

Tables are self-contained by definition (all relevant data is in the table
itself) and are therefore passed through without expansion.

Public API
──────────
    expand_context(session_id, docs, window, max_chars) -> List[Document]
"""

import logging
from typing import List

from langchain_core.documents import Document

from app.config import CONTEXT_NEIGHBOR_WINDOW, CONTEXT_MAX_CHARS_PER_HIT
from app.database.chunk_store import load_section_chunks, row_to_dict
from chunking import contextualize

logger = logging.getLogger("docchat.pipeline.context")


def expand_context(
    session_id: str,
    docs: List[Document],
    window: int = CONTEXT_NEIGHBOR_WINDOW,
    max_chars: int = CONTEXT_MAX_CHARS_PER_HIT,
) -> List[Document]:
    """
    Widen each retrieved text chunk to include its section neighbours.

    Parameters
    ──────────
    session_id:
        Used to fetch neighbouring chunk rows from Postgres.
    docs:
        Retrieved documents in relevance order (order is preserved on output).
    window:
        Number of adjacent chunks to attempt to include on each side.
        ``window=0`` disables expansion entirely.
    max_chars:
        Hard character cap for the combined body of an expanded window.
        Prevents any single hit from consuming the entire LLM context.

    Returns
    ───────
    A new list of Documents in the same relevance order.  Text chunks may be
    replaced by wider "span" documents; table chunks are returned unchanged.
    """
    if window <= 0 or not docs:
        return docs

    # Collect section ids that need neighbour rows (text chunks only)
    section_ids = sorted({
        d.metadata["section_id"] for d in docs
        if d.metadata.get("section_id") is not None
        and d.metadata.get("chunk_type") != "table"
    })
    if not section_ids:
        return docs

    # Fetch all relevant chunks in one DB round-trip, indexed by seq
    raw_rows = load_section_chunks(session_id, section_ids)
    by_seq = {r["seq"]: r for r in raw_rows if r["seq"] is not None}

    covered: set = set()   # seqs already included (avoid duplicate text)
    out: List[Document] = []

    for d in docs:         # preserve relevance order
        m = d.metadata
        seq, sec = m.get("seq"), m.get("section_id")

        if seq is None or sec is None or seq not in by_seq:
            out.append(d)  # legacy / unknown chunk: pass through unchanged
            continue
        if seq in covered:
            continue       # already inside a higher-ranked chunk's window

        covered.add(seq)

        if m.get("chunk_type") == "table":
            out.append(d)  # tables are self-contained — no expansion
            continue

        # ── Greedy window expansion ────────────────────────────────────────
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
                    or cand["section_id"] != sec         # different section — stop
                    or cand["seq"] in covered            # already sent
                    or size + 2 + len(cand["text"]) > max_chars  # over budget
                ):
                    blocked[step] = True
                    continue
                edges[step] = cand["seq"]
                covered.add(cand["seq"])
                size += 2 + len(cand["text"])

        span = [by_seq[s] for s in range(edges[-1], edges[1] + 1)]
        if len(span) == 1:
            # No neighbours were added — keep the original Document as-is
            out.append(d)
            continue

        # ── Build a merged Document for the expanded span ──────────────────
        body = "\n\n".join(r["text"] for r in span)
        page_starts = [r["page_start"] for r in span if r["page_start"] is not None]
        page_ends   = [r["page_end"]   for r in span if r["page_end"]   is not None]
        p_start = min(page_starts) if page_starts else None
        p_end   = max(page_ends)   if page_ends   else None

        content, offset = contextualize(by_seq[seq]["breadcrumb"], p_start, p_end, "text", body)
        out.append(Document(
            page_content=content,
            metadata={
                **m,
                "page_start":  p_start,
                "page_end":    p_end,
                "body_offset": offset,
                "expanded":    True,    # flag: this doc was widened
            },
        ))

    return out
