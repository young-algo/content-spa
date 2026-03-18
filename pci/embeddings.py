import asyncio
import os
from typing import Optional

from openai import OpenAI

MODEL_NAME = os.environ.get("PCI_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_SITE_URL = os.environ.get("PCI_OPENROUTER_SITE_URL") or os.environ.get("YOUR_SITE_URL")
DEFAULT_SITE_NAME = os.environ.get("PCI_OPENROUTER_SITE_NAME") or os.environ.get("YOUR_SITE_NAME")
_embedding_dim: Optional[int] = None
_client: Optional[OpenAI] = None


def embedding_settings() -> dict[str, Optional[str]]:
    return {
        "provider": "openrouter",
        "model": MODEL_NAME,
        "base_url": OPENROUTER_BASE_URL,
        "site_url": DEFAULT_SITE_URL,
        "site_name": DEFAULT_SITE_NAME,
        "api_key_present": "yes" if os.environ.get("OPENROUTER_API_KEY") else "no",
    }


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for embeddings.")
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _client


def _embedding_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if DEFAULT_SITE_URL:
        headers["HTTP-Referer"] = DEFAULT_SITE_URL
    if DEFAULT_SITE_NAME:
        headers["X-OpenRouter-Title"] = DEFAULT_SITE_NAME
    return headers


def _create_embeddings_sync(texts: list[str]) -> list[list[float]]:
    client = get_client()
    response = client.embeddings.create(
        extra_headers=_embedding_headers(),
        model=MODEL_NAME,
        input=texts,
        encoding_format="float",
    )
    return [list(item.embedding) for item in response.data]


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for one or more texts via OpenRouter."""
    if not texts:
        return []
    return await asyncio.to_thread(_create_embeddings_sync, texts)


async def get_embedding(text: str) -> list[float]:
    """Generate an embedding for a single text via OpenRouter."""
    return (await get_embeddings([text]))[0]


async def get_embedding_dimension() -> int:
    global _embedding_dim
    if _embedding_dim is None:
        _embedding_dim = len(await get_embedding("dimension probe"))
    return _embedding_dim
