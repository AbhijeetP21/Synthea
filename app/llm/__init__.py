"""The single provider seam.

Everything the application knows about LLM and embedding providers lives behind
the factories exported here. Swapping MiniMax for OpenAI or SynapticaAI is an
environment change, not a code change.
"""

from app.llm.chat import ChatClient, get_chat_client, get_judge_client
from app.llm.embeddings import EmbeddingsClient, get_embeddings_client

__all__ = [
    "ChatClient",
    "get_chat_client",
    "get_judge_client",
    "EmbeddingsClient",
    "get_embeddings_client",
]
