"""SmartSplit CLI — command-line entry point.

Handles argument parsing, logging setup, and mode dispatch:
- default: unified (API + HTTPS proxy side by side in one process)
- --api-only: only the OpenAI-compatible API endpoint
- --proxy-only: only the HTTPS intercepting proxy
- setup-claude: generate CA cert and print Claude Code instructions
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os


def _setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_api_only(port: int, host: str, log_level: str, mode: str) -> None:
    """Start SmartSplit API endpoint only (uvicorn)."""
    import uvicorn

    from smartsplit.proxy.pipeline import create_app

    app = create_app(mode=mode)

    print("\n  SmartSplit — Multi-LLM backend (API only)")
    print(f"  http://{host}:{port}/v1")
    print(f"  Mode: {mode}")
    print()

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


def _run_proxy_only(port: int, host: str, log_level: str, mode: str) -> None:
    """Start SmartSplit as HTTPS intercepting proxy only."""
    os.environ["SMARTSPLIT_MODE"] = mode
    from smartsplit.proxy.server import start_proxy

    start_proxy(port=port, host=host, log_level=log_level)


async def _serve_unified(api_port: int, proxy_port: int, host: str, log_level: str, mode: str) -> None:
    """Run API + proxy concurrently in a single event loop."""
    import uvicorn

    from smartsplit.proxy.pipeline import create_app
    from smartsplit.proxy.server import ensure_certs, run_proxy

    ensure_certs()
    app = create_app(mode=mode)

    config = uvicorn.Config(app, host=host, port=api_port, log_level=log_level.lower())
    server = uvicorn.Server(config)
    # asyncio handles signals; uvicorn's own handlers would fight run_proxy's.
    server.install_signal_handlers = lambda: None

    await asyncio.gather(server.serve(), run_proxy(proxy_port, host))


def _run_unified(api_port: int, proxy_port: int, host: str, log_level: str, mode: str) -> None:
    """Start API + proxy in one process (default)."""
    os.environ["SMARTSPLIT_MODE"] = mode

    print("\n  SmartSplit — Multi-LLM backend")
    print(f"  API:    http://{host}:{api_port}/v1")
    print(f"  Proxy:  http://{host}:{proxy_port}   (HTTPS_PROXY for Claude Code)")
    print(f"  Mode:   {mode}")
    print()

    try:
        import uvloop

        asyncio.run(
            _serve_unified(api_port, proxy_port, host, log_level, mode),
            loop_factory=uvloop.new_event_loop,
        )
    except ImportError:
        asyncio.run(_serve_unified(api_port, proxy_port, host, log_level, mode))


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
    print("    # Terminal 1: Start SmartSplit (API + proxy unified)")
    print("    smartsplit")
    print()
    print("    # Terminal 2: Launch Claude Code through the proxy")
    print(f"    NODE_EXTRA_CA_CERTS={ca_cert_path} \\")
    print("    HTTPS_PROXY=http://localhost:8421 \\")
    print("    claude")
    print()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SmartSplit — Multi-LLM backend with intelligent routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  smartsplit                          # default: API (:8420) + proxy (:8421) unified
  smartsplit --api-only               # API only on :8420
  smartsplit --proxy-only             # Proxy only on :8420
  smartsplit --mode quality           # quality-favoring routing
  smartsplit setup-claude             # Setup helper for Claude Code

Client configs:
  API (Continue, Cline, Aider, ...):  apiBase: http://localhost:8420/v1
  Proxy (Claude Code):                HTTPS_PROXY=http://localhost:8421 claude
        """,
    )
    parser.add_argument("command", nargs="?", default=None, help="Command: setup-claude")
    parser.add_argument("--port", type=int, default=8420, help="Main port (API in unified mode, default: 8420)")
    parser.add_argument("--proxy-port", type=int, default=8421, help="Proxy port in unified mode (default: 8421)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--api-only", action="store_true", help="Run only the API endpoint (no HTTPS proxy)")
    parser.add_argument(
        "--proxy-only",
        action="store_true",
        help="Run only the HTTPS intercepting proxy (for Claude Code)",
    )
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="Alias of --proxy-only (legacy)",
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

    _setup_logging(args.log_level)

    if args.command == "setup-claude":
        _setup_claude()
        return

    proxy_only = args.proxy_only or args.proxy
    if args.api_only and proxy_only:
        parser.error("--api-only cannot be combined with --proxy-only/--proxy")

    if args.api_only:
        _run_api_only(args.port, args.host, args.log_level, args.mode)
        return

    if proxy_only:
        _run_proxy_only(args.port, args.host, args.log_level, args.mode)
        return

    _run_unified(args.port, args.proxy_port, args.host, args.log_level, args.mode)


if __name__ == "__main__":
    main()
