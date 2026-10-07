"""
ASGI application module for uvicorn to import.
Exports the FastMCP streamable HTTP app for HTTP transport.

This uses FastMCP's built-in streamable_http_app() method which automatically
handles schema generation, session management, and MCP protocol compliance.
"""

import os
os.environ.setdefault("ALLOWED_HOSTS", "*")  # ← Отключаем проверку хоста

from mcp_server_qdrant.enhanced_server import mcp

# Get the FastAPI app from FastMCP's streamable HTTP implementation
app = mcp.streamable_http_app()
