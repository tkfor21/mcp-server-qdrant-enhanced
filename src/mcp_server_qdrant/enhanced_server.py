"""
Enhanced server entry point with collection-specific embedding models.
"""

import os
import sys
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.enhanced_settings import (
    EnhancedEmbeddingProviderSettings,
    EnhancedQdrantSettings,
)
from mcp_server_qdrant.settings import ToolSettings
from mcp.server.transport_security import TransportSecuritySettings

# print("[DEBUG] enhanced_server.py: Initializing enhanced settings", file=sys.stderr)

try:
    tool_settings = ToolSettings()
    # print(f"[DEBUG] enhanced_server.py: ToolSettings initialized", file=sys.stderr)

    qdrant_settings = EnhancedQdrantSettings()
    # print(f"[DEBUG] enhanced_server.py: EnhancedQdrantSettings initialized:", file=sys.stderr)
    # print(f"[DEBUG] enhanced_server.py:   location={qdrant_settings.location}", file=sys.stderr)
    # print(f"[DEBUG] enhanced_server.py:   auto_create_collections={qdrant_settings.auto_create_collections}", file=sys.stderr)
    # print(f"[DEBUG] enhanced_server.py:   enable_quantization={qdrant_settings.enable_quantization}", file=sys.stderr)

    embedding_settings = EnhancedEmbeddingProviderSettings()
    # print(f"[DEBUG] enhanced_server.py: EnhancedEmbeddingProviderSettings initialized:", file=sys.stderr)
    # print(f"[DEBUG] enhanced_server.py:   provider_type={embedding_settings.provider_type}", file=sys.stderr)
    # print(f"[DEBUG] enhanced_server.py:   model_name={embedding_settings.model_name}", file=sys.stderr)

    # Transport security allow-list is env-configurable so the server is not
    # hardwired to a single host. Defaults cover local use; add the deploy
    # host(s) via MCP_ALLOWED_HOSTS / MCP_ALLOWED_ORIGINS (comma-separated),
    # e.g. MCP_ALLOWED_HOSTS="10.0.0.225:*".
    _default_hosts = ["localhost:*", "127.0.0.1:*", "[::1]:*"]
    _default_origins = ["http://localhost:*", "http://127.0.0.1:*", "http://[::1]:*"]
    _extra_hosts = [h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    _extra_origins = [o.strip() for o in os.getenv("MCP_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_default_hosts + _extra_hosts,
        allowed_origins=_default_origins + _extra_origins,
    )

    # print("[DEBUG] enhanced_server.py: Creating EnhancedQdrantMCPServer instance", file=sys.stderr)
    mcp = QdrantMCPServer(
        tool_settings=tool_settings,
        qdrant_settings=qdrant_settings,
        embedding_provider_settings=embedding_settings,
        transport_security=transport_security,
    )
    # print("[DEBUG] enhanced_server.py: EnhancedQdrantMCPServer instance created successfully", file=sys.stderr)

except Exception:
    # print(f"[ERROR] enhanced_server.py: Failed to initialize enhanced server: {type(e).__name__}: {e}", file=sys.stderr)
    import traceback

    traceback.print_exc(file=sys.stderr)
    raise
