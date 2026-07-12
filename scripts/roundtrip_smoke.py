#!/usr/bin/env python3
"""Minimal MCP streamable-HTTP round-trip smoke test.

Exercises the *upgraded* HTTP transport stack (starlette/sse-starlette/mcp/uvicorn)
end-to-end against a running enhanced-qdrant HTTP server:
  initialize -> tools/list -> qdrant_store -> qdrant_find (semantic) -> qdrant_list_collections

Usage: MCP_URL=http://localhost:10650/mcp uv run python scripts/roundtrip_smoke.py
Exit 0 on success (a stored doc is recalled by a lexically-different query), non-zero otherwise.

Opt-in prefix-lock assertions (for a QDRANT_ALLOWED_COLLECTION_PREFIXES-locked
endpoint): set SMOKE_EXPECT_PREFIX_LOCK=<prefix> (e.g. mld_) and the smoke
additionally asserts that store/find/delete to a non-matching collection are
REJECTED (isError=true) and that qdrant_list_collections shows only admitted
names. Against an UNLOCKED shared endpoint, pass a non-prefix SMOKE_COLLECTION
(e.g. smoke_roundtrip_test) so the archive namespace isn't polluted.
"""
import os
import re
import sys
import uuid
import anyio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

URL = os.environ.get("MCP_URL", "http://localhost:10650/mcp")
LOCK_PREFIX = os.environ.get("SMOKE_EXPECT_PREFIX_LOCK", "")
# Default collection tracks the lock state: locked -> inside the admitted
# namespace; unlocked -> a smoke_ name, so a bare run against a shared endpoint
# doesn't pollute a meaningful (e.g. archive) namespace.
COLL = os.environ.get("SMOKE_COLLECTION") or (
    LOCK_PREFIX + "roundtrip_test" if LOCK_PREFIX else "smoke_roundtrip_test"
)
DOC = ("The MCP directory audit pipeline runs Docker-sandboxed static analysis "
       "plus an AI policy review to score server submissions.")
QUERY = "how does the directory vet server submissions for safety"  # deliberately different words


def _text(result):
    out = []
    for c in getattr(result, "content", []) or []:
        out.append(getattr(c, "text", str(c)))
    return out


async def _expect_rejected(s, tool, args, label):
    """A locked endpoint must reject this call with isError=true (the raised
    guard ValueError surfaces as a protocol-level tool error)."""
    r = await s.call_tool(tool, args)
    if not getattr(r, "isError", False):
        print(f"[FAIL] {label}: expected isError=true rejection, got success")
        return False
    text = " ".join(_text(r))
    if "not allowed" not in text:
        print(f"[FAIL] {label}: errored, but not the prefix rejection: {text[:150]}")
        return False
    print(f"[ok] {label} -> rejected (isError=true)")
    return True


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
            if getattr(r1, "isError", False):
                print(f"[FAIL] qdrant_store errored: {_text(r1)}")
                return 1
            store_out = _text(r1)
            print(f"[ok] qdrant_store -> {store_out}")

            r2 = await s.call_tool("qdrant_find", {"query": QUERY, "collection_name": COLL})
            if getattr(r2, "isError", False):
                print(f"[FAIL] qdrant_find errored: {_text(r2)}")
                return 1
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

            if LOCK_PREFIX:
                # Pick a probe name guaranteed OUTSIDE the lock prefix (an
                # exotic prefix like "smoke_" must not admit the probe).
                foreign = next(
                    n for n in ("smoke_foreign_probe", "zzz_foreign_probe",
                                "qqq_foreign_probe")
                    if not n.startswith(LOCK_PREFIX)
                )
                ok = True
                ok &= await _expect_rejected(
                    s, "qdrant_store",
                    {"information": "x", "collection_name": foreign},
                    f"store -> {foreign}")
                ok &= await _expect_rejected(
                    s, "qdrant_find",
                    {"query": "x", "collection_name": foreign},
                    f"find -> {foreign}")
                ok &= await _expect_rejected(
                    s, "qdrant_delete_points",
                    {"point_ids": [uuid.uuid4().hex], "collection_name": foreign},
                    f"delete_points -> {foreign}")

                r = await s.call_tool("qdrant_list_collections", {})
                listing = "\n".join(_text(r))
                # Parse each collection NAME out of its "📊 **name**" line and
                # require a true prefix match — a substring check would pass a
                # foreign name that merely CONTAINS the prefix (backup_mld_x).
                names = re.findall(r"📊 \*\*(.+?)\*\*", listing)
                # Non-vacuity pin: we JUST stored into COLL, so at least that
                # name must parse out — 0 names means the render format drifted
                # (or listing broke) and the leak check would pass vacuously.
                if not names:
                    print("[FAIL] listing parsed 0 collection names — render format drifted or listing broke; leak check is vacuous")
                    ok = False
                elif COLL not in names:
                    print(f"[FAIL] just-stored collection {COLL!r} missing from listing: {names}")
                    ok = False
                leaked = [n for n in names if not n.startswith(LOCK_PREFIX)]
                if leaked:
                    print(f"[FAIL] list_collections leaked non-{LOCK_PREFIX} names: {leaked}")
                    ok = False
                elif names:
                    print(f"[ok] list_collections shows only {LOCK_PREFIX}* collections ({len(names)} listed)")

                if not ok:
                    return 1
                print(f"[PASS] prefix lock ({LOCK_PREFIX}) enforced over real HTTP")
            return 0


if __name__ == "__main__":
    sys.exit(anyio.run(main))
