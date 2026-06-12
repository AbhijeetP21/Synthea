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

    # --- PHI detection & redaction (Phase 3) ---
    # Detect/redact identifiers before any text is embedded, stored, or sent to
    # the model. Detection always reports the full set; redaction covers direct
    # identifiers. Dates are reported but kept by default (clinical content in
    # this synthetic demonstrator) — flip phi_redact_dates to also redact them.
    phi_redaction: bool = True
    phi_spacy_model: str = "en_core_web_lg"
    phi_score_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    phi_redact_dates: bool = False

    # --- Datastore ---
    database_url: str = "postgresql+psycopg://clinical:clinical@localhost:5432/clinical"

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=8, ge=1, le=50)
    # Cosine-distance ceiling (0 = identical, higher = less similar). Chunks
    # farther than this are treated as non-evidence so off-topic / out-of-record
    # questions retrieve nothing and the system abstains. Calibrated for
    # bge-small-en-v1.5 (relevant ~0.24-0.34, off-topic ~0.42+); the eval harness
    # (Phase 4) can tune it. Set high (e.g. 2.0) to disable the gate.
    retrieval_max_distance: float = Field(default=0.40, ge=0.0, le=2.0)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so config is parsed once per process."""
    return Settings()
