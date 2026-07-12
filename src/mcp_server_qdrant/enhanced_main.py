"""
Enhanced main entry point for the mcp-server-qdrant with multi-model support.
"""

import argparse
import sys


def main():
    """
    Enhanced main entry point for the mcp-server-qdrant script.
    Supports collection-specific embedding models and optimized configurations.
    """
    # print(f"[DEBUG] enhanced_main.py: Starting Enhanced MCP server", file=sys.stderr)
    # print(f"[DEBUG] enhanced_main.py: Python version: {sys.version}", file=sys.stderr)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="mcp-server-qdrant-enhanced")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
    )
    args = parser.parse_args()

    # print(f"[DEBUG] enhanced_main.py: Transport mode: {args.transport}", file=sys.stderr)
    # print(f"[DEBUG] enhanced_main.py: Legacy mode: {args.legacy_mode}", file=sys.stderr)

    try:
        # Enhanced server with multi-model support
        from mcp_server_qdrant.enhanced_server import mcp

        # print(f"[DEBUG] enhanced_main.py: Server module imported successfully", file=sys.stderr)
        # print(f"[DEBUG] enhanced_main.py: Starting MCP server with transport={args.transport}", file=sys.stderr)

        mcp.run(transport=args.transport)

    except Exception:
        # print(f"[ERROR] enhanced_main.py: Exception occurred: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
