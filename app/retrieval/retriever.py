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
        settings = get_settings()
        self._k = settings.retrieval_top_k
        self._max_distance = settings.retrieval_max_distance

    def retrieve(self, patient_id: str, query: str, k: int | None = None) -> list[Source]:
        query_vec = self._embedder.embed_query(query)
        scored = semantic_search(patient_id, query_vec, k or self._k)
        # Relevance gate: drop chunks beyond the distance ceiling. If a question
        # is off-topic or its answer is not in the record, every chunk is too far
        # and we return nothing — the Q&A service then abstains rather than
        # answering from irrelevant evidence.
        return [src for src, dist in scored if dist <= self._max_distance]
