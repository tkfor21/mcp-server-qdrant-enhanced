"""
Ollama-backed embedding provider — offloads inference to a (possibly remote)
Ollama instance, decoupling GPU location from container location: the server
can run on a CPU-only box and call an Ollama on the GPU host.

Gated behind EMBEDDING_PROVIDER=ollama (+ OLLAMA_URL / OLLAMA_EMBED_MODEL).
Collection→(vector_name, dimensions) routing is reused unchanged from
EnhancedEmbeddingProviderSettings; the Ollama model's output dimensions must
match the collection's configured dimensions (validated on every call, since a
mismatched vector would be rejected or silently mis-searched by Qdrant).
"""

import asyncio
import os
import sys
from typing import Dict, List, Optional

import httpx

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

# Ollama serializes inference internally; a small client-side concurrency cap
# keeps batch embeds pipelined without flooding the host with queued requests.
_MAX_CONCURRENT_REQUESTS = 4

# Native output dimensions of common Ollama embedding models (base name, tag
# stripped). Used for UNMAPPED collections, where the settings fallback (384,
# fastembed-oriented) would both mis-provision the Qdrant collection and then
# fail _check_dims on every call. OLLAMA_EMBED_DIMS overrides for models not
# listed here; _check_dims remains the backstop either way.
_OLLAMA_MODEL_DIMS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "bge-m3": 1024,
    "snowflake-arctic-embed": 1024,
    "granite-embedding": 384,
}


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider that delegates inference to an Ollama server via
    POST {OLLAMA_URL}/api/embeddings, mirroring EnhancedFastEmbedProvider's
    interface (embed_documents / embed_query / get_vector_name /
    get_vector_size / set_collection_context / get_model_info_for_collection).
    """

    def __init__(
        self,
        embedding_settings,
        ollama_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.embedding_settings = embedding_settings
        self.ollama_url = (
            ollama_url
            or getattr(embedding_settings, "ollama_url", None)
            or os.getenv("OLLAMA_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model = (
            model
            or getattr(embedding_settings, "ollama_model", None)
            or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        )
        self._current_collection: Optional[str] = None
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)

    def set_collection_context(self, collection_name: str):
        """Set the current collection context for vector-name/dim routing."""
        self._current_collection = collection_name

    def _collection_is_mapped(self, collection_name: str) -> bool:
        """
        True when the collection has an EXPLICIT model mapping (defaults or the
        COLLECTION_MODEL_MAPPINGS env JSON). Unmapped collections must NOT use
        the settings fallback (384, fastembed-oriented): the Qdrant collection
        would be provisioned at 384 and then every Ollama embed (e.g. nomic's
        768) would fail _check_dims — so they route to the Ollama model's own
        dims/vector-name instead.
        """
        import json

        from mcp_server_qdrant.enhanced_settings import (
            COLLECTION_ALIASES,
            COLLECTION_MODEL_MAPPINGS,
        )

        try:
            mappings = dict(COLLECTION_MODEL_MAPPINGS)
            raw = getattr(self.embedding_settings, "collection_model_mappings", "") or "{}"
            if raw != "{}":
                mappings.update(json.loads(raw))
            resolved = COLLECTION_ALIASES.get(collection_name, collection_name)
            return resolved in mappings or collection_name in mappings
        except Exception:
            # On any surprise, preserve the settings-routing behavior.
            return True

    def _model_dims(self) -> int:
        """Output dims of the configured Ollama model (env override wins)."""
        override = os.getenv("OLLAMA_EMBED_DIMS", "")
        if override.isdigit():
            return int(override)
        base = self.model.split(":")[0]
        # Unknown model without an override: assume the 768 mainstream default;
        # _check_dims still catches a real mismatch on the first embed.
        return _OLLAMA_MODEL_DIMS.get(base, 768)

    async def _embed_one(self, text: str) -> List[float]:
        async with self._semaphore:
            response = await self._client.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if not embedding:
            raise ValueError(
                f"Ollama at {self.ollama_url} returned no embedding "
                f"(model={self.model}) — is the model pulled?"
            )
        return embedding

    def _check_dims(self, vector: List[float], collection_name: Optional[str]):
        expected = self.get_vector_size(collection_name)
        if len(vector) != expected:
            raise ValueError(
                f"Ollama model '{self.model}' returned a {len(vector)}-dim vector "
                f"but collection '{collection_name or self._current_collection}' is "
                f"configured for {expected} dims. Map the collection to a "
                f"{len(vector)}-dim config (COLLECTION_MODEL_MAPPINGS / "
                f"CUSTOM_MODEL_CONFIGS) or use a matching Ollama model."
            )

    async def embed_documents(
        self, documents: List[str], collection_name: Optional[str] = None
    ) -> List[List[float]]:
        """Embed a list of documents into vectors via Ollama."""
        vectors = await asyncio.gather(*(self._embed_one(d) for d in documents))
        if vectors:
            self._check_dims(vectors[0], collection_name)
        return list(vectors)

    async def embed_query(
        self, query: str, collection_name: Optional[str] = None
    ) -> List[float]:
        """Embed a query into a vector via Ollama."""
        vector = await self._embed_one(query)
        self._check_dims(vector, collection_name)
        return vector

    def get_vector_name(self, collection_name: Optional[str] = None) -> str:
        """
        Vector name for the Qdrant collection. Explicitly-mapped collections
        keep the settings routing (cross-provider storage-layout compat);
        unmapped collections use an Ollama-derived name.
        """
        collection_name = collection_name or self._current_collection
        if collection_name and self._collection_is_mapped(collection_name):
            return self.embedding_settings.get_vector_name_for_collection(
                collection_name
            )
        return f"ollama-{self.model.split(':')[0]}"

    def get_vector_size(self, collection_name: Optional[str] = None) -> int:
        """
        Vector dimensions for the Qdrant collection. Explicitly-mapped
        collections keep the settings routing; unmapped collections use the
        Ollama model's own dims (the settings fallback is 384/fastembed, which
        would mis-provision the collection and fail every embed).
        """
        collection_name = collection_name or self._current_collection
        if collection_name and self._collection_is_mapped(collection_name):
            return self.embedding_settings.get_dimensions_for_collection(
                collection_name
            )
        return self._model_dims()

    def get_model_info_for_collection(self, collection_name: str) -> Dict[str, any]:
        """
        Comprehensive model information for a collection — derived from the
        same get_vector_name/get_vector_size routing so an unmapped collection
        reports the Ollama-derived values, not the settings 384 fallback.
        """
        mapped = self._collection_is_mapped(collection_name)
        config = (
            self.embedding_settings.get_model_config_for_collection(collection_name)
            if mapped
            else {}
        )
        return {
            "collection_name": collection_name,
            "vector_name": self.get_vector_name(collection_name),
            "dimensions": self.get_vector_size(collection_name),
            "fastembed_model": config.get("fastembed_model"),
            "provider": "ollama",
            "ollama_model": self.model,
            "ollama_url": self.ollama_url,
        }

    async def aclose(self):
        """Release the underlying HTTP client."""
        await self._client.aclose()
