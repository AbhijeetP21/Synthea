"""Retrieval layer.

MVP: patient-scoped semantic search over pgvector, behind a small interface so
the keyword/BM25 arm (the next increment toward hybrid retrieval) can be added
without touching callers. Clinical terms like drug names and codes are exactly
the case where the keyword arm will help — hence the interface seam now.
"""

from __future__ import annotations

from app.config import get_settings
from app.db import semantic_search
from app.llm import get_embeddings_client
from app.schemas import Source


class Retriever:
    def __init__(self) -> None:
        self._embedder = get_embeddings_client()
        self._k = get_settings().retrieval_top_k

    def retrieve(self, patient_id: str, query: str, k: int | None = None) -> list[Source]:
        query_vec = self._embedder.embed_query(query)
        return semantic_search(patient_id, query_vec, k or self._k)
