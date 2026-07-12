"""Shared harness for driving the REAL MCP tool closures in tests.

Promoted from test_collection_prefixes.py so sibling suites (e.g. the
client-point-ids tests) reuse ONE harness instead of duplicating it — in
particular the mcp-1.28 call_tool result-shape knowledge in tool_text /
tool_json, and the env-hygiene rules in make_server (every env var that
changes server behavior is force-set or force-deleted so ambient shell
exports can never flip a test's meaning).
"""

import json

import pytest
from mcp.server.fastmcp import Context

from mcp_server_qdrant.enhanced_settings import (
    EnhancedEmbeddingProviderSettings,
    EnhancedQdrantSettings,
)
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import ToolSettings


async def _noop_debug(self, message, **kwargs):
    """ctx.debug raises outside a live MCP request; tests drive tools via
    call_tool with no request context, so no-op it (rejection tests don't
    need this — the guard fires before ctx.debug — but success paths do)."""
    return None


def make_server(
    monkeypatch, prefixes=None, env_mappings=None, env_configs=None
) -> QdrantMCPServer:
    """Build a QdrantMCPServer against an isolated :memory: Qdrant.

    Every behavior-bearing env var is either set from the args or force
    -deleted (pydantic reads live os.environ at construction — an ambient
    export must not silently change what a test exercises).
    """
    monkeypatch.setattr(Context, "debug", _noop_debug)
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    # Force-delete EVERY behavior-bearing env var the settings models live-read
    # at construction — an ambient EMBEDDING_PROVIDER=ollama would flip every
    # harness-built server onto the network-dependent provider, and an ambient
    # QDRANT_LOCAL_PATH conflicts with the :memory: location outright.
    for var in (
        "COLLECTION_NAME",
        "QDRANT_AUTO_CREATE_COLLECTIONS",  # must be ON (default) for seeds
        "QDRANT_ENABLE_QUANTIZATION",
        "QDRANT_API_KEY",
        "QDRANT_LOCAL_PATH",
        "QDRANT_SEARCH_LIMIT",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "OLLAMA_URL",
        "OLLAMA_EMBED_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    for var, value in (
        ("QDRANT_ALLOWED_COLLECTION_PREFIXES", prefixes),
        ("COLLECTION_MODEL_MAPPINGS", env_mappings),
        ("CUSTOM_MODEL_CONFIGS", env_configs),
    ):
        if value is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, value)
    return QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=EnhancedQdrantSettings(),
        embedding_provider_settings=EnhancedEmbeddingProviderSettings(),
    )


def tool_text(result) -> str:
    """Flatten a call_tool result to searchable text.

    mcp 1.28 call_tool shapes: a str-returning tool yields a TUPLE of
    (content_blocks, structured_dict); a dict-returning tool yields a plain
    list of content blocks (JSON-dumped text).
    """
    if isinstance(result, tuple):
        blocks, structured = result
        parts = [getattr(b, "text", "") or "" for b in blocks]
        parts.append(json.dumps(structured, default=str))
        return "\n".join(parts)
    if isinstance(result, dict):
        return json.dumps(result, default=str)
    return "\n".join(getattr(b, "text", "") or "" for b in result)


def _unwrap_result_envelope(d: dict) -> dict:
    """FastMCP structured output wraps a tool's return under {'result': ...}."""
    inner = d.get("result")
    return inner if isinstance(inner, dict) else d


def tool_json(result) -> dict:
    """Recover the structured dict payload from a call_tool result."""
    if isinstance(result, dict):
        return _unwrap_result_envelope(result)
    if isinstance(result, tuple):
        blocks, structured = result
        if isinstance(structured, dict):
            return _unwrap_result_envelope(structured)
        result = blocks
    for block in result:
        text = getattr(block, "text", None)
        if text:
            return _unwrap_result_envelope(json.loads(text))
    raise AssertionError("no JSON payload in tool result")


@pytest.fixture
def locked_server(monkeypatch):
    return make_server(monkeypatch, prefixes="mld_")


@pytest.fixture
def unlocked_server(monkeypatch):
    return make_server(monkeypatch, prefixes=None)
