"""Pluggable AI provider abstraction for ORACLE.

Supports Gemini (free), OpenAI, and Anthropic. Configure via env vars:
    LLM_PROVIDER=gemini|openai|anthropic  (default: gemini)
    EMBEDDING_PROVIDER=gemini|openai      (default: gemini)

Usage::

    from oracle_shared.providers import get_llm, get_embedder

    llm = get_llm()
    response = await llm.generate("What is 2+2?")

    embedder = get_embedder()
    vector = await embedder.embed("prediction market")
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ── LLM Interface ────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract LLM for text generation."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
    ) -> str:
        """Generate text. Returns the raw response string."""
        ...

    async def generate_json(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
    ) -> dict:
        """Generate and parse JSON. Strips markdown fences if present."""
        raw = await self.generate(prompt, system, max_tokens)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        return json.loads(text)


class GeminiLLM(LLMProvider):
    """Google Gemini via google-genai SDK."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        from google import genai
        self._client = genai.Client(
            api_key=api_key or os.getenv("GEMINI_API_KEY", ""),
        )
        self._model = model

    async def generate(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        # google-genai is sync, run in executor for async compat
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=full_prompt,
            ),
        )
        return response.text


class AnthropicLLM(LLMProvider):
    """Anthropic Claude via anthropic SDK."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
        )
        self._model = model

    async def generate(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAILLM(LLMProvider):
    """OpenAI GPT via openai SDK."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        )
        self._model = model

    async def generate(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


# ── Embedding Interface ───────────────────────────────────────────────────────

class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...

    async def embed_single(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]


class GeminiEmbedder(EmbeddingProvider):
    """Google Gemini embeddings."""

    def __init__(self, api_key: str | None = None) -> None:
        from google import genai
        self._client = genai.Client(
            api_key=api_key or os.getenv("GEMINI_API_KEY", ""),
        )
        self._dims = 3072  # gemini-embedding-001 output size

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import asyncio
        loop = asyncio.get_event_loop()
        results = []
        # Batch in groups of 100
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            response = await loop.run_in_executor(
                None,
                lambda b=batch: self._client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=b,
                ),
            )
            for emb_obj in response.embeddings:
                results.append(emb_obj.values)
        return results


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI embeddings."""

    def __init__(self, dims: int = 512, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        )
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
            dimensions=self._dims,
        )
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm(provider: str | None = None, **kwargs) -> LLMProvider:
    """Get an LLM provider instance.

    Provider is determined by: explicit arg > LLM_PROVIDER env var > auto-detect from available keys.
    """
    p = (provider or os.getenv("LLM_PROVIDER", "")).lower()

    if not p:
        # Auto-detect: prefer Gemini (free), then Anthropic, then OpenAI
        if os.getenv("GEMINI_API_KEY"):
            p = "gemini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            p = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            p = "openai"
        else:
            p = "gemini"  # default

    if p == "gemini":
        return GeminiLLM(**kwargs)
    elif p == "anthropic":
        return AnthropicLLM(**kwargs)
    elif p == "openai":
        return OpenAILLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {p}")


def get_embedder(provider: str | None = None, **kwargs) -> EmbeddingProvider:
    """Get an embedding provider instance.

    Provider is determined by: explicit arg > EMBEDDING_PROVIDER env var > auto-detect.
    """
    p = (provider or os.getenv("EMBEDDING_PROVIDER", "")).lower()

    if not p:
        if os.getenv("GEMINI_API_KEY"):
            p = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            p = "openai"
        else:
            p = "gemini"

    if p == "gemini":
        return GeminiEmbedder(**kwargs)
    elif p == "openai":
        return OpenAIEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {p}")
