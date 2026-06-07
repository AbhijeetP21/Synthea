"""Central configuration, loaded entirely from the environment.

Every provider-specific value lives here and flows into ``app/llm``. Nothing
elsewhere in the codebase reads provider env vars directly, which is what keeps
the "swap providers via config, not code" guarantee true.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Chat model ---
    chat_provider: Literal["anthropic", "openai"] = "anthropic"
    chat_model: str = "MiniMax-M3"
    chat_base_url: str = "https://api.minimax.io/anthropic"
    chat_api_key: str = ""
    chat_max_tokens: int = 2000
    chat_temperature: float = 0.0

    # --- Embedding model ---
    embeddings_provider: Literal["local", "openai"] = "local"
    embeddings_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embeddings_base_url: str = ""
    embeddings_api_key: str = ""

    # --- Datastore ---
    database_url: str = "postgresql+psycopg://clinical:clinical@localhost:5432/clinical"

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=8, ge=1, le=50)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so config is parsed once per process."""
    return Settings()
