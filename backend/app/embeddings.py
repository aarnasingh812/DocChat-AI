"""
app/embeddings.py
─────────────────
Custom LangChain-compatible embedding class backed by Google's GenAI API.

Why a custom class?
───────────────────
The official `langchain-google-genai` package wraps the older `google-generativeai`
SDK.  We use the newer `google-genai` SDK (``from google import genai``) for
access to the latest models (e.g. gemini-embedding-001) and the cleaner v2 API.
This thin wrapper makes GeminiEmbeddings a drop-in replacement for any
LangChain ``Embeddings`` object.
"""

from typing import List

from langchain_core.embeddings import Embeddings
from google import genai

from app.config import GOOGLE_API_KEY, EMBEDDING_MODEL


class GeminiEmbeddings(Embeddings):
    """
    Thin wrapper around the Google GenAI ``embed_content`` API.

    Batches document embedding calls to stay within the per-request input
    limit imposed by the API (``_BATCH`` items per call).

    Parameters
    ----------
    api_key:
        Google AI Studio / Vertex API key.  Defaults to ``GOOGLE_API_KEY``
        from :mod:`app.config`.
    model:
        Embedding model id.  Defaults to ``EMBEDDING_MODEL`` from
        :mod:`app.config`.
    """

    _BATCH = 100  # embed_content accepts a bounded number of inputs per request

    def __init__(
        self,
        api_key: str = GOOGLE_API_KEY,
        model: str = EMBEDDING_MODEL,
    ):
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to your .env file."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings (batched to respect API limits)."""
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self._BATCH):
            result = self._client.models.embed_content(
                model=self._model,
                contents=texts[i : i + self._BATCH],
            )
            vectors.extend(e.values for e in result.embeddings)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        return result.embeddings[0].values
