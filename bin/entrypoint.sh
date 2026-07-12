#!/usr/bin/env bash
# Golden-path container entrypoint.
# One image, one transport switch: MCP_TRANSPORT=http (default) | stdio.
set -euo pipefail

case "${MCP_TRANSPORT:-http}" in
  http)
    exec python -m uvicorn mcp_server_qdrant.enhanced_http_app:app \
      --host "${MCP_HOST:-0.0.0.0}" --port "${MCP_PORT:-10650}"
    ;;
  stdio)
    exec python -m mcp_server_qdrant.enhanced_main --transport stdio
    ;;
  *)
    echo "entrypoint: unknown MCP_TRANSPORT='${MCP_TRANSPORT}' (expected 'http' or 'stdio')" >&2
    exit 2
    ;;
esac
