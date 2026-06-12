"""Chat completion clients behind one interface.

Two wire protocols are supported because MiniMax (today's default) is best
consumed through the Anthropic SDK, while OpenAI and SynapticaAI speak the
OpenAI Chat Completions format. Callers depend only on ``ChatClient.complete``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import Settings, get_settings


class ChatClient(ABC):
    """Provider-agnostic chat interface."""

    @abstractmethod
    def complete(self, *, system: str, user: str) -> str:
        """Return the assistant's text for a single-turn system+user prompt."""
        raise NotImplementedError


class AnthropicChatClient(ChatClient):
    """Anthropic-compatible backend (MiniMax-M3 by default).

    MiniMax may emit ``thinking`` blocks alongside ``text`` blocks; we keep only
    the text so callers get a clean answer string.
    """

    def __init__(self, settings: Settings) -> None:
        import anthropic

        self._settings = settings
        self._client = anthropic.Anthropic(
            base_url=settings.chat_base_url or None,
            api_key=settings.chat_api_key,
        )

    def complete(self, *, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self._settings.chat_model,
            max_tokens=self._settings.chat_max_tokens,
            temperature=self._settings.chat_temperature,
            system=system,
            messages=[{"role": "user", "content": [{"type": "text", "text": user}]}],
        )
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()


class OpenAIChatClient(ChatClient):
    """OpenAI-compatible backend (OpenAI, SynapticaAI, MiniMax OpenAI-mode)."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._settings = settings
        self._client = OpenAI(
            base_url=settings.chat_base_url or None,
            api_key=settings.chat_api_key,
        )

    def complete(self, *, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._settings.chat_model,
            max_tokens=self._settings.chat_max_tokens,
            temperature=self._settings.chat_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def _build(settings: Settings) -> ChatClient:
    if settings.chat_provider == "anthropic":
        return AnthropicChatClient(settings)
    if settings.chat_provider == "openai":
        return OpenAIChatClient(settings)
    raise ValueError(f"Unsupported CHAT_PROVIDER: {settings.chat_provider!r}")


@lru_cache
def get_chat_client() -> ChatClient:
    return _build(get_settings())


@lru_cache
def get_judge_client() -> ChatClient:
    """Chat client for the eval LLM-as-judge, configured from the JUDGE_* env
    vars so it can point at a different model/provider than the one under test."""
    s = get_settings()
    judge = s.model_copy(
        update={
            "chat_provider": s.judge_provider,
            "chat_model": s.judge_model,
            "chat_base_url": s.judge_base_url,
            "chat_api_key": s.judge_api_key or s.chat_api_key,
            "chat_max_tokens": s.judge_max_tokens,
            "chat_temperature": s.judge_temperature,
        }
    )
    return _build(judge)
