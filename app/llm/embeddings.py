"""Embedding clients behind one interface.

The Anthropic SDK has no embeddings API, and embeddings are universally
OpenAI-shaped, so this is a separate seam from chat. The default backend is
local (fastembed): zero external dependency, no API key, and deterministic —
which keeps ingestion runnable offline and makes the eval harness reproducible
in CI. Flip ``EMBEDDINGS_PROVIDER=openai`` to use OpenAI/MiniMax/SynapticaAI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import Settings, get_settings


class EmbeddingsClient(ABC):
    """Provider-agnostic embedding interface."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimensionality (must match the pgvector column)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents for storage."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


class LocalEmbeddings(EmbeddingsClient):
    """Offline embeddings via fastembed (ONNX). Default backend."""

    def __init__(self, settings: Settings) -> None:
        from fastembed import TextEmbedding

        self._settings = settings
        self._model = TextEmbedding(model_name=settings.embeddings_model)

    @property
    def dim(self) -> int:
        return self._settings.embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()


class OpenAICompatEmbeddings(EmbeddingsClient):
    """Any OpenAI-compatible /embeddings endpoint (OpenAI, MiniMax, SynapticaAI)."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._settings = settings
        self._client = OpenAI(
            base_url=settings.embeddings_base_url or None,
            api_key=settings.embeddings_api_key,
        )

    @property
    def dim(self) -> int:
        return self._settings.embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._settings.embeddings_model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _build(settings: Settings) -> EmbeddingsClient:
    if settings.embeddings_provider == "local":
        return LocalEmbeddings(settings)
    if settings.embeddings_provider == "openai":
        return OpenAICompatEmbeddings(settings)
    raise ValueError(f"Unsupported EMBEDDINGS_PROVIDER: {settings.embeddings_provider!r}")


@lru_cache
def get_embeddings_client() -> EmbeddingsClient:
    return _build(get_settings())
