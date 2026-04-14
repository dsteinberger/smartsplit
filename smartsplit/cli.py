"""SmartSplit CLI — command-line entry point.

Handles argument parsing, logging setup, and mode dispatch (API server / HTTPS proxy / setup).
"""

from __future__ import annotations

import argparse
import logging
import os


def _run_proxy_mode(port: int, host: str, log_level: str) -> None:
    """Start SmartSplit as a lightweight HTTPS intercepting proxy."""
    from smartsplit.proxy import start_proxy

    start_proxy(port=port, host=host, log_level=log_level)


def _run_proxy_mitmproxy(port: int, log_level: str) -> None:
    """Start SmartSplit using mitmproxy (legacy, heavier)."""
    try:
        from mitmproxy.tools.main import mitmdump
    except ImportError:
        print("Error: mitmproxy is required for proxy-mitm mode.")
        print("Install with: pip install 'smartsplit[proxy]'")
        raise SystemExit(1) from None

    addon_path = os.path.join(os.path.dirname(__file__), "mitm_addon.py")

    print("\n  SmartSplit — HTTPS Proxy Mode (mitmproxy)")
    print(f"  Intercepting proxy on port {port}")
    print()

    mitmdump(
        [
            "-s",
            addon_path,
            "-p",
            str(port),
            "--set",
            "connection_strategy=lazy",
            "--set",
            "ssl_insecure=true",
        ]
    )


def _setup_claude() -> None:
    """Help the user set up SmartSplit with Claude Code."""
    from smartsplit.proxy import ensure_certs

    ca_cert_path = ensure_certs()

    print("\n  SmartSplit — Claude Code Setup")
    print("  ================================\n")
    print(f"  [OK] CA certificate: {ca_cert_path}")
    print()
    print("  Ready! Launch SmartSplit + Claude Code:")
    print()
    print("    # Terminal 1: Start SmartSplit proxy")
    print("    smartsplit --mode proxy")
    print()
    print("    # Terminal 2: Launch Claude Code")
    print(f"    NODE_EXTRA_CA_CERTS={ca_cert_path} \\")
    print("    HTTPS_PROXY=http://localhost:8420 \\")
    print("    claude")
    print()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SmartSplit — Free multi-LLM backend with intelligent routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  smartsplit                          # API mode (default) on port 8420
  smartsplit --mode proxy             # HTTPS proxy for Claude Code (lightweight)
  smartsplit --mode proxy-mitm        # HTTPS proxy using mitmproxy (legacy)
  smartsplit setup-claude             # Setup helper for Claude Code
  smartsplit --port 3456 --mode economy

API mode (default):
  Continue: apiBase: http://localhost:8420/v1
  Cline:    Base URL: http://localhost:8420/v1
  Aider:    aider --openai-api-base http://localhost:8420/v1

Proxy mode (Claude Code + subscription):
  HTTPS_PROXY=http://localhost:8420 claude
        """,
    )
    parser.add_argument("command", nargs="?", default=None, help="Command: setup-claude")
    parser.add_argument("--port", type=int, default=8420, help="Port (default: 8420)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["economy", "balanced", "quality", "proxy", "proxy-mitm"],
        help="Mode: economy, balanced, quality (API), proxy (lightweight HTTPS), or proxy-mitm (mitmproxy)",
    )
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Handle commands
    if args.command == "setup-claude":
        _setup_claude()
        return

    # Proxy modes
    if args.mode == "proxy":
        _run_proxy_mode(args.port, args.host, args.log_level)
        return
    if args.mode == "proxy-mitm":
        _run_proxy_mitmproxy(args.port, args.log_level)
        return

    # API mode (default)
    import uvicorn

    from smartsplit.pipeline import create_app

    app = create_app(mode=args.mode)

    print("\n  SmartSplit — Free multi-LLM backend")
    print(f"  http://{args.host}:{args.port}/v1")
    print(f"  Mode: {args.mode}")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
