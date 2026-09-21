"""
app/services/llm.py
────────────────────
Singleton accessor for the language model used during generation.

Using ``@lru_cache(maxsize=1)`` ensures that the heavy ChatGroq client
object is created exactly once per process, regardless of how many
requests arrive concurrently.

Swap the model or provider here without touching any other module.
"""

from functools import lru_cache

from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, LLM_MODEL


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Return the shared ChatGroq LLM instance.

    The object is created on the first call and cached for the lifetime of
    the process.  ``GROQ_API_KEY`` and ``LLM_MODEL`` are read from
    :mod:`app.config`.
    """
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
    )
