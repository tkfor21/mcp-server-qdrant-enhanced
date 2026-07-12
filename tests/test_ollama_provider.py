"""
Unit tests for OllamaEmbeddingProvider — transport is mocked (httpx.MockTransport),
so these run with no Ollama available. The live round-trip is covered by
scripts/roundtrip_smoke.py against a real stack.
"""

import json

import httpx
import pytest

from mcp_server_qdrant.embeddings.ollama import OllamaEmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType


class StubSettings:
    """Minimal stand-in for EnhancedEmbeddingProviderSettings routing."""

    ollama_url = "http://ollama.test:11434"
    ollama_model = "nomic-embed-text"
    # "mapped_collection" is explicitly mapped; anything else is unmapped and
    # must route to the Ollama model's own dims/vector-name.
    collection_model_mappings = '{"mapped_collection": "bge-base-en"}'

    def get_vector_name_for_collection(self, collection_name):
        return "bge-base-en"

    def get_dimensions_for_collection(self, collection_name):
        return 768

    def get_model_config_for_collection(self, collection_name):
        return {
            "dimensions": 768,
            "vector_name": "bge-base-en",
            "provider": EmbeddingProviderType.FASTEMBED,
            "fastembed_model": "BAAI/bge-base-en",
        }


def _mock_transport(dims=768, capture=None):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if capture is not None:
            capture.append(payload)
        return httpx.Response(200, json={"embedding": [0.1] * dims})

    return httpx.MockTransport(handler)


def _provider(dims=768, capture=None):
    return OllamaEmbeddingProvider(
        embedding_settings=StubSettings(),
        transport=_mock_transport(dims=dims, capture=capture),
    )


async def test_embed_query_returns_vector_and_sends_model():
    sent = []
    provider = _provider(capture=sent)
    vector = await provider.embed_query("hello", "some_collection")
    assert len(vector) == 768
    assert sent[0]["model"] == "nomic-embed-text"
    assert sent[0]["prompt"] == "hello"
    await provider.aclose()


async def test_embed_documents_one_request_per_document():
    sent = []
    provider = _provider(capture=sent)
    vectors = await provider.embed_documents(["a", "b", "c"], "some_collection")
    assert [len(v) for v in vectors] == [768, 768, 768]
    assert {p["prompt"] for p in sent} == {"a", "b", "c"}
    await provider.aclose()


async def test_dim_mismatch_on_mapped_collection_raises_clear_error():
    provider = _provider(dims=384)  # Ollama returns 384, mapped collection wants 768
    with pytest.raises(ValueError, match="384-dim vector .* configured for 768"):
        await provider.embed_query("hello", "mapped_collection")
    await provider.aclose()


async def test_empty_embedding_raises():
    def handler(request):
        return httpx.Response(200, json={"embedding": []})

    provider = OllamaEmbeddingProvider(
        embedding_settings=StubSettings(),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ValueError, match="returned no embedding"):
        await provider.embed_query("hello", "some_collection")
    await provider.aclose()


def test_mapped_collection_uses_settings_routing():
    provider = _provider()
    assert provider.get_vector_name("mapped_collection") == "bge-base-en"
    assert provider.get_vector_size("mapped_collection") == 768


def test_unmapped_collection_routes_to_ollama_model_dims():
    # The settings fallback for unmapped collections is 384/fastembed — using it
    # would mis-provision the Qdrant collection and fail every nomic (768) embed.
    provider = _provider()
    assert provider.get_vector_name("some_collection") == "ollama-nomic-embed-text"
    assert provider.get_vector_size("some_collection") == 768
    # No collection context → same Ollama-derived routing
    assert provider.get_vector_name() == "ollama-nomic-embed-text"
    assert provider.get_vector_size() == 768


def test_unmapped_dims_follow_model_table_and_env_override(monkeypatch):
    provider = OllamaEmbeddingProvider(
        embedding_settings=StubSettings(), model="mxbai-embed-large:latest"
    )
    assert provider.get_vector_size("some_collection") == 1024
    monkeypatch.setenv("OLLAMA_EMBED_DIMS", "512")
    assert provider.get_vector_size("some_collection") == 512


def test_model_info_reports_ollama_provider():
    provider = _provider()
    info = provider.get_model_info_for_collection("some_collection")
    assert info["provider"] == "ollama"
    assert info["ollama_model"] == "nomic-embed-text"
    # Unmapped → Ollama-derived, NOT the settings 384 fallback
    assert info["dimensions"] == 768
    assert info["vector_name"] == "ollama-nomic-embed-text"
    mapped = provider.get_model_info_for_collection("mapped_collection")
    assert mapped["dimensions"] == 768
    assert mapped["vector_name"] == "bge-base-en"


def test_settings_url_and_env_precedence(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://env.test:11434")
    # Explicit settings attribute wins over env
    provider = OllamaEmbeddingProvider(embedding_settings=StubSettings())
    assert provider.ollama_url == "http://ollama.test:11434"

    class NoUrlSettings(StubSettings):
        ollama_url = None
        ollama_model = None

    provider_env = OllamaEmbeddingProvider(embedding_settings=NoUrlSettings())
    assert provider_env.ollama_url == "http://env.test:11434"
    assert provider_env.model == "nomic-embed-text"
