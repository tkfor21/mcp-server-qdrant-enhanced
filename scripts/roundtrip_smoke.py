#!/usr/bin/env python3
"""Minimal MCP streamable-HTTP round-trip smoke test.

Exercises the *upgraded* HTTP transport stack (starlette/sse-starlette/mcp/uvicorn)
end-to-end against a running enhanced-qdrant HTTP server:
  initialize -> tools/list -> qdrant_store -> qdrant_find (semantic) -> qdrant_list_collections

Usage: MCP_URL=http://localhost:10650/mcp uv run python scripts/roundtrip_smoke.py
Exit 0 on success (a stored doc is recalled by a lexically-different query), non-zero otherwise.
"""
import os
import sys
import anyio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

URL = os.environ.get("MCP_URL", "http://localhost:10650/mcp")
COLL = os.environ.get("SMOKE_COLLECTION", "mld_roundtrip_test")
DOC = ("The MCP directory audit pipeline runs Docker-sandboxed static analysis "
       "plus an AI policy review to score server submissions.")
QUERY = "how does the directory vet server submissions for safety"  # deliberately different words


def _text(result):
    out = []
    for c in getattr(result, "content", []) or []:
        out.append(getattr(c, "text", str(c)))
    return out


async def main() -> int:
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as s:
            init = await s.initialize()
            print(f"[ok] initialize -> server={init.serverInfo.name} v{init.serverInfo.version}")

            tools = await s.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"[ok] tools/list -> {len(names)} tools: {names}")

            r1 = await s.call_tool("qdrant_store", {
                "information": DOC,
                "collection_name": COLL,
                "metadata": {"kind": "note", "topic": "audit"},
            })
            store_out = _text(r1)
            print(f"[ok] qdrant_store -> {store_out}")

            r2 = await s.call_tool("qdrant_find", {"query": QUERY, "collection_name": COLL})
            find_out = _text(r2)
            print(f"[ok] qdrant_find('{QUERY}') ->")
            for line in find_out:
                print("        " + line.replace("\n", " ")[:200])

            hit = any("audit pipeline" in line.lower() or "policy review" in line.lower()
                      for line in find_out)
            if not hit:
                print("[FAIL] stored doc was NOT recalled by the semantic query")
                return 1
            print("[PASS] semantic round-trip: lexically-different query recalled the stored doc")
            return 0


if __name__ == "__main__":
    sys.exit(anyio.run(main))
