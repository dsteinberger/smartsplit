"""SmartSplit CLI — command-line entry point.

Handles argument parsing, logging setup, and mode dispatch (API server / HTTPS proxy / setup).
"""

from __future__ import annotations

import argparse
import logging
import os


def _run_proxy_mode(port: int, host: str, log_level: str, mode: str = "balanced") -> None:
    """Start SmartSplit as a lightweight HTTPS intercepting proxy."""
    import os

    os.environ["SMARTSPLIT_MODE"] = mode
    from smartsplit.proxy.server import start_proxy

    start_proxy(port=port, host=host, log_level=log_level)


def _setup_claude() -> None:
    """Help the user set up SmartSplit with Claude Code."""
    from smartsplit.proxy.server import ensure_certs

    ca_cert_path = ensure_certs()

    print("\n  SmartSplit — Claude Code Setup")
    print("  ================================\n")
    print(f"  [OK] CA certificate: {ca_cert_path}")
    print()
    print("  Ready! Launch SmartSplit + Claude Code:")
    print()
    print("    # Terminal 1: Start SmartSplit proxy")
    print("    smartsplit --proxy")
    print()
    print("    # Terminal 2: Launch Claude Code")
    print(f"    NODE_EXTRA_CA_CERTS={ca_cert_path} \\")
    print("    HTTPS_PROXY=http://localhost:8420 \\")
    print("    claude")
    print()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SmartSplit — Multi-LLM backend with intelligent routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  smartsplit                          # API mode (default) on port 8420
  smartsplit --proxy                  # HTTPS proxy for Claude Code
  smartsplit --proxy --mode quality   # HTTPS proxy with quality routing
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
        "--proxy",
        action="store_true",
        help="Run as HTTPS intercepting proxy (for Claude Code, Cline, etc.)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["economy", "balanced", "quality"],
        help="Worker routing mode: economy (favor cost), balanced (default), quality (favor quality)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

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

    # Proxy mode
    if args.proxy:
        _run_proxy_mode(args.port, args.host, args.log_level, args.mode)
        return

    # API mode (default)
    import uvicorn

    from smartsplit.proxy.pipeline import create_app

    app = create_app(mode=args.mode)

    print("\n  SmartSplit — Multi-LLM backend")
    print(f"  http://{args.host}:{args.port}/v1")
    print(f"  Mode: {args.mode}")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
