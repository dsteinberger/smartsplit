"""SmartSplit HTTPS proxy — intercepts LLM API traffic for any coding agent.

Single process, no internal HTTP server. The proxy TLS layer calls the
SmartSplit pipeline functions directly in Python.

Architecture:
  Client → CONNECT api.anthropic.com:443
       → TLS handshake with dynamic cert
       → Parse HTTP request
       → Call process_anthropic_request() directly (no HTTP hop)
       → Forward to api.anthropic.com via shared httpx client
       → Response back through TLS tunnel

  Client → CONNECT api.github.com:443 (not LLM)
       → Blind TCP tunnel
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import ssl
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from smartsplit.pipeline import ProxyContext

logger = logging.getLogger("smartsplit.proxy_server")

_CERT_DIR = Path(os.environ.get("SMARTSPLIT_CERT_DIR", str(Path.home() / ".smartsplit" / "certs")))
_CA_CERT = _CERT_DIR / "ca-cert.pem"
_CA_KEY = _CERT_DIR / "ca-key.pem"

# Known LLM API hosts to intercept.
_LLM_HOSTS: set[str] = {
    "api.anthropic.com",
    "api.openai.com",
    "api.groq.com",
    "api.cerebras.ai",
    "api.deepseek.com",
    "api.mistral.ai",
    "openrouter.ai",
    "generativelanguage.googleapis.com",
}

# Paths that go through the SmartSplit pipeline.
_SMARTSPLIT_PATHS = {"/v1/messages", "/v1/chat/completions"}

# Default enrichment backoff when retry-after header is absent (seconds).
_ENRICH_BACKOFF_DEFAULT = 300


# ── Certificate management ───────────────────────────────────────


def _ensure_ca() -> tuple[object, object]:
    """Ensure the CA key+cert exist. Returns (ca_key, ca_cert) objects."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        raise SystemExit("cryptography is required for proxy mode.\nInstall with: pip install cryptography") from None

    import datetime

    _CERT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    if _CA_CERT.exists() and _CA_KEY.exists():
        ca_key = serialization.load_pem_private_key(_CA_KEY.read_bytes(), password=None)
        ca_cert = x509.load_pem_x509_certificate(_CA_CERT.read_bytes())
        return ca_key, ca_cert

    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "SmartSplit CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    _CA_KEY.write_bytes(
        ca_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    _CA_KEY.chmod(0o600)
    _CA_CERT.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    logger.info("Generated CA certificate: %s", _CA_CERT)
    return ca_key, ca_cert


def _get_host_ssl_context(ca_key: object, ca_cert: object, hostname: str) -> ssl.SSLContext:
    """Get or create an SSL context for a given hostname (dynamic cert)."""
    host_cert = _CERT_DIR / f"{hostname}-cert.pem"
    host_key = _CERT_DIR / f"{hostname}-key.pem"

    if not host_cert.exists() or not host_key.exists():
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
            .issuer_name(ca_cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.UTC))
            .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        host_key.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
        )
        host_key.chmod(0o600)
        host_cert.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        logger.info("Generated cert for %s", hostname)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(host_cert), str(host_key))
    return ctx


def ensure_certs() -> Path:
    """Ensure CA exists. Returns path to CA cert."""
    _ensure_ca()
    return _CA_CERT


# ── HTTP helpers ─────────────────────────────────────────────────


async def _read_http_request(reader: asyncio.StreamReader) -> tuple[str, dict[str, str], bytes]:
    """Read a full HTTP/1.1 request."""
    request_line = (await reader.readline()).decode("utf-8", errors="replace").strip()
    headers: dict[str, str] = {}
    while True:
        line = (await reader.readline()).decode("utf-8", errors="replace").strip()
        if not line:
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    body = b""
    content_length = int(headers.get("content-length", "0"))
    if content_length > 0:
        body = await reader.readexactly(content_length)
    return request_line, headers, body


def _build_http_response(status: int, headers: dict[str, str], body: bytes) -> bytes:
    """Build a raw HTTP/1.1 response."""
    status_text = {
        200: "OK",
        400: "Bad Request",
        401: "Unauthorized",
        429: "Too Many Requests",
        502: "Bad Gateway",
        503: "Service Unavailable",
    }
    lines = [f"HTTP/1.1 {status} {status_text.get(status, 'Unknown')}"]
    headers["content-length"] = str(len(body))
    for key, value in headers.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("")
    return "\r\n".join(lines).encode("utf-8") + body


# ── Request handling ─────────────────────────────────────────────


# ── Connection handlers ──────────────────────────────────────────


async def _tunnel_blind(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, host: str, port: int) -> None:
    """Blind TCP tunnel — no TLS interception."""
    try:
        remote_reader, remote_writer = await asyncio.open_connection(host, port)
    except Exception as exc:
        logger.debug("Tunnel to %s:%d failed: %s", host, port, exc)
        writer.close()
        return

    async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
        try:
            while True:
                data = await src.read(65536)
                if not data:
                    break
                dst.write(data)
                await dst.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        finally:
            with contextlib.suppress(Exception):
                dst.close()

    await asyncio.gather(pipe(reader, remote_writer), pipe(remote_reader, writer))


def _log_sse_usage(chunk_data: bytes) -> None:
    """Extract and log token usage from SSE response chunks (best-effort)."""
    try:
        text = chunk_data.decode("utf-8", errors="replace")
        for line in text.split("\n"):
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            usage = data.get("message", data).get("usage", data.get("usage"))
            if not usage:
                continue
            parts = []
            if "input_tokens" in usage:
                parts.append(f"input={usage['input_tokens']}")
            if "cache_read_input_tokens" in usage:
                parts.append(f"cached={usage['cache_read_input_tokens']}")
            if "cache_creation_input_tokens" in usage:
                parts.append(f"cache_new={usage['cache_creation_input_tokens']}")
            if "output_tokens" in usage:
                parts.append(f"output={usage['output_tokens']}")
            if parts:
                logger.info("TOKENS: %s", " | ".join(parts))
    except Exception:
        pass


def _log_ratelimit_summary(header_text: str) -> None:
    """Log Anthropic rate limit as a single concise line."""
    util_5h = util_7d = status = ""
    for hline in header_text.split("\r\n"):
        lower = hline.lower()
        if "5h-utilization:" in lower:
            util_5h = hline.split(":", 1)[1].strip()
        elif "7d-utilization:" in lower:
            util_7d = hline.split(":", 1)[1].strip()
        elif lower.startswith("anthropic-ratelimit-unified-status:"):
            status = hline.split(":", 1)[1].strip()
    if util_5h:
        logger.info("RATELIMIT: 5h=%s | 7d=%s | status=%s", util_5h, util_7d, status)


def _rebuild_http_request(request_line: str, headers: dict[str, str], body: bytes) -> bytes:
    """Rebuild raw HTTP/1.1 request bytes from parsed components."""
    lines = [request_line]
    for key, value in headers.items():
        lines.append(f"{key}: {value}")
    if body and "content-length" not in headers:
        lines.append(f"content-length: {len(body)}")
    lines.append("")
    lines.append("")
    return "\r\n".join(lines).encode("utf-8") + body


def _parse_transfer_info(header_text: str) -> tuple[int, bool]:
    """Parse content-length and chunked flag from header text."""
    content_length = 0
    is_chunked = False
    for hline in header_text.split("\r\n"):
        lower = hline.lower()
        if lower.startswith("content-length:"):
            content_length = int(hline.split(":", 1)[1].strip())
        if lower.startswith("transfer-encoding:") and "chunked" in lower:
            is_chunked = True
    return content_length, is_chunked


async def _drain_body_from_headers(
    header_block: bytes,
    upstream_reader: asyncio.StreamReader,
) -> None:
    """Read and discard the response body (used to drain a 429 before retry)."""
    header_text = header_block.decode("utf-8", errors="replace")
    content_length, is_chunked = _parse_transfer_info(header_text)
    if is_chunked:
        while True:
            chunk_header = await upstream_reader.readline()
            chunk_size = int(chunk_header.strip(), 16)
            if chunk_size == 0:
                await upstream_reader.readline()  # trailing \r\n
                break
            await upstream_reader.readexactly(chunk_size + 2)
    elif content_length > 0:
        await upstream_reader.readexactly(content_length)


async def _relay_http_response_body(
    upstream_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    header_block: bytes,
    ctx: ProxyContext | None = None,
) -> None:
    """Relay just the body when headers were already sent to the client."""
    header_text = header_block.decode("utf-8", errors="replace")
    content_length, is_chunked = _parse_transfer_info(header_text)

    # Log rate limit summary (one line instead of 12)
    _log_ratelimit_summary(header_text)

    if is_chunked:
        while True:
            chunk_header = await upstream_reader.readline()
            client_writer.write(chunk_header)
            chunk_size = int(chunk_header.strip(), 16)
            if chunk_size == 0:
                trailing = await upstream_reader.readline()
                client_writer.write(trailing)
                await client_writer.drain()
                break
            chunk_data = await upstream_reader.readexactly(chunk_size + 2)
            client_writer.write(chunk_data)
            await client_writer.drain()
            if b'"usage"' in chunk_data:
                _log_sse_usage(chunk_data)
    elif content_length > 0:
        body = await upstream_reader.readexactly(content_length)
        client_writer.write(body)
        await client_writer.drain()


async def _relay_http_response(
    upstream_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    ctx: ProxyContext | None = None,
) -> int:
    """Stream HTTP response from upstream to client — relay each chunk immediately.

    Returns the HTTP status code for logging.
    """
    # Read status line + headers, send immediately
    header_lines: list[bytes] = []
    while True:
        line = await upstream_reader.readline()
        header_lines.append(line)
        if line == b"\r\n" or line == b"\n" or not line:
            break

    header_block = b"".join(header_lines)
    client_writer.write(header_block)
    await client_writer.drain()

    # Parse status code
    header_text = header_block.decode("utf-8", errors="replace")
    status = 0
    first_line = header_text.split("\r\n", 1)[0]
    status_parts = first_line.split(" ", 2)
    if len(status_parts) >= 2:
        with contextlib.suppress(ValueError):
            status = int(status_parts[1])

    # Log rate limit summary (one line instead of 12)
    _log_ratelimit_summary(header_text)

    content_length, is_chunked = _parse_transfer_info(header_text)

    # Relay body — stream each chunk immediately (critical for SSE)
    if is_chunked:
        while True:
            chunk_header = await upstream_reader.readline()
            client_writer.write(chunk_header)
            chunk_size = int(chunk_header.strip(), 16)
            if chunk_size == 0:
                trailing = await upstream_reader.readline()
                client_writer.write(trailing)
                await client_writer.drain()
                break
            chunk_data = await upstream_reader.readexactly(chunk_size + 2)  # +2 for \r\n
            client_writer.write(chunk_data)
            await client_writer.drain()
            # Scan SSE chunks for token usage (non-blocking, best-effort)
            if b'"usage"' in chunk_data:
                _log_sse_usage(chunk_data)
    elif content_length > 0:
        body = await upstream_reader.readexactly(content_length)
        client_writer.write(body)
        await client_writer.drain()

    return status


async def _handle_connect(
    ctx: ProxyContext,
    ca_key: object,
    ca_cert: object,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    host: str,
    port: int,
) -> None:
    """Handle CONNECT — intercept LLM hosts, tunnel everything else."""
    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await writer.drain()

    if host in _LLM_HOSTS:
        # TLS termination with client (we impersonate the host)
        ssl_ctx = await asyncio.to_thread(_get_host_ssl_context, ca_key, ca_cert, host)
        try:
            loop = asyncio.get_event_loop()
            transport = writer.transport
            new_transport = await loop.start_tls(transport, writer._protocol, ssl_ctx, server_side=True)
            writer._transport = new_transport
            reader._transport = new_transport
        except Exception as exc:
            logger.debug("TLS handshake failed (client) for %s: %s", host, exc)
            with contextlib.suppress(Exception):
                writer.close()
            return

        # Open ONE persistent TLS connection to the upstream host
        upstream_ssl = ssl.create_default_context()
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                host,
                port,
                ssl=upstream_ssl,
            )
        except Exception as exc:
            logger.warning("Upstream connection to %s:%d failed: %s", host, port, exc)
            with contextlib.suppress(Exception):
                writer.close()
            return

        logger.info("Connected to upstream %s:%d (persistent)", host, port)

        try:
            while True:
                # Read request from client
                request_line, req_headers, body = await _read_http_request(reader)
                if not request_line:
                    break

                parts = request_line.split(" ")
                method = parts[0] if parts else "GET"
                path = parts[1] if len(parts) > 1 else "/"
                path_base = path.split("?")[0]

                # Decide: pipeline or relay
                if method == "POST" and path_base in _SMARTSPLIT_PATHS:
                    try:
                        body_dict = json.loads(body)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        body_dict = None

                    if body_dict is not None:
                        from smartsplit.pipeline import process_anthropic_request_lite

                        action = await process_anthropic_request_lite(ctx, body_dict, req_headers, original_host=host)

                        if action.get("type") == "fake":
                            # Fake tool_use response — don't touch upstream
                            resp_body = json.dumps(action["body"]).encode("utf-8")
                            resp_bytes = _build_http_response(200, {"content-type": "application/json"}, resp_body)
                            writer.write(resp_bytes)
                            await writer.drain()
                            logger.info("[%s] %s %s → FAKE tool_use", host, method, path[:60])
                            continue

                        if action.get("type") == "modified":
                            # Pipeline modified the body (injected context) — relay modified body.
                            # If upstream returns 429, set enrichment backoff and retry with original.
                            modified_body = json.dumps(action["body"]).encode("utf-8")
                            delta = len(modified_body) - len(body)
                            logger.info(
                                "[%s] Modified body: +%d bytes (~%d tokens added)",
                                host,
                                delta,
                                delta // 4,
                            )
                            modified_headers = dict(req_headers)
                            modified_headers["content-length"] = str(len(modified_body))
                            raw_request = _rebuild_http_request(request_line, modified_headers, modified_body)
                            upstream_writer.write(raw_request)
                            await upstream_writer.drain()

                            # Peek at response status before relaying to client
                            resp_header_lines: list[bytes] = []
                            while True:
                                hline = await upstream_reader.readline()
                                resp_header_lines.append(hline)
                                if hline == b"\r\n" or hline == b"\n" or not hline:
                                    break
                            resp_header_block = b"".join(resp_header_lines)
                            resp_first = resp_header_block.decode("utf-8", errors="replace").split("\r\n", 1)[0]
                            resp_status = 0
                            resp_parts = resp_first.split(" ", 2)
                            if len(resp_parts) >= 2:
                                with contextlib.suppress(ValueError):
                                    resp_status = int(resp_parts[1])

                            if resp_status == 429:
                                # Drain 429 response body
                                await _drain_body_from_headers(resp_header_block, upstream_reader)
                                # Extract retry-after for enrichment backoff
                                resp_hdr_text = resp_header_block.decode("utf-8", errors="replace")
                                retry_after = _ENRICH_BACKOFF_DEFAULT
                                for _hl in resp_hdr_text.split("\r\n"):
                                    if _hl.lower().startswith("retry-after:"):
                                        with contextlib.suppress(ValueError):
                                            retry_after = int(_hl.split(":", 1)[1].strip())
                                        break
                                ctx.enrichment_skip_until = time.time() + retry_after
                                logger.warning(
                                    "[%s] %s %s → modified got 429, backoff %ds, retrying original",
                                    host,
                                    method,
                                    path[:60],
                                    retry_after,
                                )
                                # Reopen upstream connection and retry with original body
                                with contextlib.suppress(Exception):
                                    upstream_writer.close()
                                upstream_reader, upstream_writer = await asyncio.open_connection(
                                    host,
                                    port,
                                    ssl=upstream_ssl,
                                )
                                raw_original = _rebuild_http_request(request_line, req_headers, body)
                                upstream_writer.write(raw_original)
                                await upstream_writer.drain()
                                resp_status = await _relay_http_response(upstream_reader, writer, ctx=ctx)
                                logger.info("[%s] %s %s → fallback original → %d", host, method, path[:60], resp_status)
                                continue

                            # Not 429 — relay the response (headers already read, stream body)
                            writer.write(resp_header_block)
                            await writer.drain()
                            await _relay_http_response_body(
                                upstream_reader,
                                writer,
                                resp_header_block,
                                ctx=ctx,
                            )
                            logger.info("[%s] %s %s → SmartSplit (modified) → %d", host, method, path[:60], resp_status)
                            continue

                        # type == "passthrough" — relay as-is (fall through)

                # Relay: forward raw bytes to upstream, stream response back
                raw_request = _rebuild_http_request(request_line, req_headers, body)
                upstream_writer.write(raw_request)
                await upstream_writer.drain()

                resp_status = await _relay_http_response(upstream_reader, writer, ctx=ctx)
                logger.info("[%s] %s %s → %d", host, method, path[:60], resp_status)

        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        except Exception as exc:
            logger.debug("Connection error for %s: %s: %s", host, type(exc).__name__, exc)
        finally:
            with contextlib.suppress(Exception):
                upstream_writer.close()
            with contextlib.suppress(Exception):
                writer.close()
    else:
        await _tunnel_blind(reader, writer, host, port)


async def _handle_client(
    ctx: ProxyContext,
    ca_key: object,
    ca_cert: object,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Handle a new proxy client connection."""
    try:
        first_line = (await reader.readline()).decode("utf-8", errors="replace").strip()
        if not first_line:
            writer.close()
            return

        while True:
            line = (await reader.readline()).decode("utf-8", errors="replace").strip()
            if not line:
                break

        if first_line.startswith("CONNECT"):
            target = first_line.split(" ")[1]
            if ":" in target:
                host, port_str = target.rsplit(":", 1)
                port = int(port_str)
            else:
                host = target
                port = 443
            await _handle_connect(ctx, ca_key, ca_cert, reader, writer, host, port)
        else:
            writer.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            await writer.drain()
            writer.close()
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    except Exception as exc:
        logger.debug("Client handler error: %s: %s", type(exc).__name__, exc)
    finally:
        with contextlib.suppress(Exception):
            writer.close()


# ── Server entry point ───────────────────────────────────────────


async def run_proxy(port: int = 8420, host: str = "127.0.0.1") -> None:
    """Start the SmartSplit proxy — single process, no internal HTTP server."""
    from smartsplit.config import load_config
    from smartsplit.intention_detector import IntentionDetector
    from smartsplit.learning import BanditScorer
    from smartsplit.pipeline import ProxyContext
    from smartsplit.planner import Planner
    from smartsplit.providers.registry import ProviderRegistry
    from smartsplit.quota import QuotaTracker
    from smartsplit.router import Router
    from smartsplit.tool_anticipator import ToolAnticipator
    from smartsplit.tool_pattern_learner import ToolPatternLearner

    ca_key, ca_cert = _ensure_ca()

    # Initialize the SmartSplit context (same as create_app but without Starlette)
    cfg = load_config()
    http_client = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    )
    quota = QuotaTracker(provider_configs=cfg.providers)
    registry = ProviderRegistry(
        cfg.providers, http_client, free_llm_priority=cfg.free_llm_priority, brain_name=cfg.brain
    )
    planner = Planner(registry)
    bandit = BanditScorer()
    router = Router(registry, quota, cfg, bandit=bandit)
    pattern_learner = ToolPatternLearner(project_dir=".")
    detector = IntentionDetector(registry, pattern_learner=pattern_learner)
    anticipator = ToolAnticipator(registry, working_dir=".")

    ctx = ProxyContext(
        config=cfg,
        registry=registry,
        planner=planner,
        router=router,
        quota=quota,
        bandit=bandit,
        http=http_client,
        detector=detector,
        anticipator=anticipator,
        pattern_learner=pattern_learner,
    )

    llm_hosts = ", ".join(sorted(_LLM_HOSTS))
    logger.info("SmartSplit proxy listening on %s:%d", host, port)
    logger.info("Intercepting: %s", llm_hosts)
    logger.info(
        "Brain: %s | Workers: %s",
        registry.brain_name,
        [n for n in registry.get_llm_providers() if n != registry.brain_name],
    )
    logger.info("CA certificate: %s", _CA_CERT)

    server = await asyncio.start_server(
        lambda r, w: _handle_client(ctx, ca_key, ca_cert, r, w),
        host,
        port,
    )

    try:
        async with server:
            await server.serve_forever()
    finally:
        quota.flush()
        bandit.flush()
        pattern_learner.flush()
        await http_client.aclose()


def start_proxy(port: int = 8420, host: str = "127.0.0.1", log_level: str = "INFO") -> None:
    """Synchronous entry point."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ca_cert_path = ensure_certs()
    llm_hosts = ", ".join(sorted(_LLM_HOSTS))

    print("\n  SmartSplit — HTTPS Proxy")
    print(f"  Listening on {host}:{port}")
    print(f"  Intercepting: {llm_hosts}")
    print()
    print("  Usage:")
    print(f"    NODE_EXTRA_CA_CERTS={ca_cert_path} \\")
    print(f"    HTTPS_PROXY=http://localhost:{port} \\")
    print("    claude  # (or cline, continue, aider, opencode...)")
    print()

    # Use uvloop for 2-4x faster event loop (falls back to default if not installed)
    try:
        import uvloop

        asyncio.run(run_proxy(port, host), loop_factory=uvloop.new_event_loop)
    except ImportError:
        asyncio.run(run_proxy(port, host))
