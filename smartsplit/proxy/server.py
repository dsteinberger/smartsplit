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
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from smartsplit.exceptions import SmartSplitError

if TYPE_CHECKING:
    from smartsplit.proxy.pipeline import ProxyContext

logger = logging.getLogger("smartsplit.proxy_server")

# Hard limits on the transport layer to protect against DoS (huge Content-Length,
# oversized chunks, header floods) before the pipeline's higher-level caps kick in.
_MAX_REQUEST_BYTES = 1_000_000  # 1 MB — aligned with pipeline._MAX_REQUEST_BYTES
_MAX_CHUNK_SIZE = 10_000_000  # 10 MB per HTTP/1.1 chunk (generous for streamed SSE)
_MAX_HEADER_COUNT = 100  # request/response header lines
_MAX_RESPONSE_BYTES = 50_000_000  # 50 MB for a single non-chunked response body

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
    """Read a full HTTP/1.1 request, enforcing transport-level size limits."""
    request_line = (await reader.readline()).decode("utf-8", errors="replace").strip()
    headers: dict[str, str] = {}
    header_count = 0
    while True:
        line = (await reader.readline()).decode("utf-8", errors="replace").strip()
        if not line:
            break
        header_count += 1
        if header_count > _MAX_HEADER_COUNT:
            raise SmartSplitError(f"Request header count exceeds limit ({_MAX_HEADER_COUNT})")
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    body = b""
    try:
        content_length = int(headers.get("content-length", "0"))
    except ValueError as exc:
        raise SmartSplitError(f"Invalid Content-Length header: {exc}") from None
    if content_length < 0 or content_length > _MAX_REQUEST_BYTES:
        raise SmartSplitError(f"Content-Length {content_length} exceeds limit ({_MAX_REQUEST_BYTES})")
    if content_length > 0:
        body = await reader.readexactly(content_length)
    return request_line, headers, body


def _parse_chunk_size(line: bytes) -> int:
    """Parse an HTTP/1.1 chunk-size line, rejecting values over _MAX_CHUNK_SIZE."""
    raw = line.strip()
    # Strip chunk extensions (`; name=value`) before parsing.
    if b";" in raw:
        raw = raw.split(b";", 1)[0].strip()
    try:
        size = int(raw, 16)
    except ValueError:
        raise SmartSplitError(f"Invalid chunk size: {line!r}") from None
    if size < 0 or size > _MAX_CHUNK_SIZE:
        raise SmartSplitError(f"Chunk size {size} exceeds limit ({_MAX_CHUNK_SIZE})")
    return size


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


def _iter_headers(header_text: str) -> Iterator[tuple[str, str]]:
    """Yield ``(name_lower, value)`` for each header line in raw HTTP header text."""
    for line in header_text.split("\r\n"):
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        yield name.strip().lower(), value.strip()


def _log_ratelimit_summary(header_text: str) -> None:
    """Log Anthropic rate limit as a single concise line."""
    util_5h = util_7d = status = ""
    for name, value in _iter_headers(header_text):
        if "5h-utilization" in name:
            util_5h = value
        elif "7d-utilization" in name:
            util_7d = value
        elif name == "anthropic-ratelimit-unified-status":
            status = value
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
    for name, value in _iter_headers(header_text):
        if name == "content-length":
            content_length = int(value)
        elif name == "transfer-encoding" and "chunked" in value.lower():
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
            chunk_size = _parse_chunk_size(chunk_header)
            if chunk_size == 0:
                await upstream_reader.readline()  # trailing \r\n
                break
            await upstream_reader.readexactly(chunk_size + 2)
    elif content_length > 0:
        if content_length > _MAX_RESPONSE_BYTES:
            raise SmartSplitError(f"Response Content-Length {content_length} exceeds limit ({_MAX_RESPONSE_BYTES})")
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
            chunk_size = _parse_chunk_size(chunk_header)
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
        if content_length > _MAX_RESPONSE_BYTES:
            raise SmartSplitError(f"Response Content-Length {content_length} exceeds limit ({_MAX_RESPONSE_BYTES})")
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
            chunk_size = _parse_chunk_size(chunk_header)
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
        if content_length > _MAX_RESPONSE_BYTES:
            raise SmartSplitError(f"Response Content-Length {content_length} exceeds limit ({_MAX_RESPONSE_BYTES})")
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
        await _intercept_llm_host(ctx, ca_key, ca_cert, reader, writer, host, port)
    else:
        await _tunnel_blind(reader, writer, host, port)


def _rebind_stream_transport(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    new_transport: asyncio.BaseTransport,
) -> None:
    """Swap the transport under an existing ``StreamReader``/``StreamWriter`` after ``start_tls``.

    ``asyncio.StreamWriter`` / ``StreamReader`` do not expose a public API to rebind
    their underlying transport. After ``loop.start_tls``, the original transport is
    detached and a new TLS transport takes its place — the streams still reference
    the stale one, so we patch them directly.

    Isolated here so the private attribute access stays in one obvious spot. This
    pattern is documented in cpython issues #79780 / #86727 and is the workaround
    used by popular proxies (mitmproxy, aiohttp server) until asyncio provides a
    public upgrade API.
    """
    writer._transport = new_transport
    reader._transport = new_transport


async def _do_client_tls_handshake(
    ca_key: object, ca_cert: object, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, host: str
) -> bool:
    """Upgrade the accepted client connection to TLS, impersonating ``host``."""
    ssl_ctx = await asyncio.to_thread(_get_host_ssl_context, ca_key, ca_cert, host)
    try:
        loop = asyncio.get_event_loop()
        transport = writer.transport
        protocol = writer._protocol
        new_transport = await loop.start_tls(transport, protocol, ssl_ctx, server_side=True)
        _rebind_stream_transport(reader, writer, new_transport)
        return True
    except Exception as exc:
        logger.debug("TLS handshake failed (client) for %s: %s", host, exc)
        with contextlib.suppress(Exception):
            writer.close()
        return False


async def _peek_response_status(upstream_reader: asyncio.StreamReader) -> tuple[bytes, int]:
    """Read all response header lines (through the empty terminator) and parse the status code."""
    lines: list[bytes] = []
    while True:
        hline = await upstream_reader.readline()
        lines.append(hline)
        if hline == b"\r\n" or hline == b"\n" or not hline:
            break
    header_block = b"".join(lines)
    first_line = header_block.decode("utf-8", errors="replace").split("\r\n", 1)[0]
    parts = first_line.split(" ", 2)
    status = 0
    if len(parts) >= 2:
        with contextlib.suppress(ValueError):
            status = int(parts[1])
    return header_block, status


def _extract_retry_after(header_block: bytes) -> int:
    """Parse the ``retry-after`` header (seconds), falling back to the enrichment default."""
    header_text = header_block.decode("utf-8", errors="replace")
    for name, value in _iter_headers(header_text):
        if name == "retry-after":
            with contextlib.suppress(ValueError):
                return int(value)
    return _ENRICH_BACKOFF_DEFAULT


async def _retry_with_original_body(
    ctx: ProxyContext,
    writer: asyncio.StreamWriter,
    host: str,
    port: int,
    upstream_ssl: ssl.SSLContext,
    request_line: str,
    req_headers: dict[str, str],
    body: bytes,
    header_block: bytes,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, int]:
    """Set enrichment backoff, reopen upstream, and relay the original (non-modified) request."""
    retry_after = _extract_retry_after(header_block)
    ctx.enrichment_skip_until = time.time() + retry_after
    logger.warning("[%s] modified got 429, backoff %ds, retrying original", host, retry_after)

    upstream_reader, upstream_writer = await asyncio.open_connection(host, port, ssl=upstream_ssl)
    raw_original = _rebuild_http_request(request_line, req_headers, body)
    upstream_writer.write(raw_original)
    await upstream_writer.drain()
    resp_status = await _relay_http_response(upstream_reader, writer, ctx=ctx)
    return upstream_reader, upstream_writer, resp_status


async def _forward_modified_body(
    ctx: ProxyContext,
    writer: asyncio.StreamWriter,
    upstream_reader: asyncio.StreamReader,
    upstream_writer: asyncio.StreamWriter,
    host: str,
    port: int,
    upstream_ssl: ssl.SSLContext,
    request_line: str,
    req_headers: dict[str, str],
    body: bytes,
    modified_body_dict: dict,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Send an enriched request; on 429 roll back to the original body, otherwise relay normally."""
    modified_body = json.dumps(modified_body_dict).encode("utf-8")
    delta = len(modified_body) - len(body)
    logger.info("[%s] Modified body: +%d bytes (~%d tokens added)", host, delta, delta // 4)

    modified_headers = dict(req_headers)
    modified_headers["content-length"] = str(len(modified_body))
    upstream_writer.write(_rebuild_http_request(request_line, modified_headers, modified_body))
    await upstream_writer.drain()

    header_block, resp_status = await _peek_response_status(upstream_reader)

    if resp_status == 429:
        await _drain_body_from_headers(header_block, upstream_reader)
        with contextlib.suppress(Exception):
            upstream_writer.close()
        new_reader, new_writer, _ = await _retry_with_original_body(
            ctx,
            writer,
            host,
            port,
            upstream_ssl,
            request_line,
            req_headers,
            body,
            header_block,
        )
        return new_reader, new_writer

    writer.write(header_block)
    await writer.drain()
    await _relay_http_response_body(upstream_reader, writer, header_block, ctx=ctx)
    logger.info("[%s] SmartSplit (modified) → %d", host, resp_status)
    return upstream_reader, upstream_writer


async def _process_smartsplit_request(
    ctx: ProxyContext,
    writer: asyncio.StreamWriter,
    upstream_reader: asyncio.StreamReader,
    upstream_writer: asyncio.StreamWriter,
    host: str,
    port: int,
    upstream_ssl: ssl.SSLContext,
    request_line: str,
    req_headers: dict[str, str],
    body: bytes,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, bool]:
    """Run the SmartSplit pipeline. Returns (upstream_reader, upstream_writer, handled)."""
    try:
        body_dict = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return upstream_reader, upstream_writer, False

    from smartsplit.proxy.pipeline import process_anthropic_request_lite

    action = await process_anthropic_request_lite(ctx, body_dict, req_headers, original_host=host)
    action_type = action.get("type")

    if action_type == "fake":
        resp_body = json.dumps(action["body"]).encode("utf-8")
        writer.write(_build_http_response(200, {"content-type": "application/json"}, resp_body))
        await writer.drain()
        logger.info("[%s] FAKE tool_use", host)
        return upstream_reader, upstream_writer, True

    if action_type == "modified":
        upstream_reader, upstream_writer = await _forward_modified_body(
            ctx,
            writer,
            upstream_reader,
            upstream_writer,
            host,
            port,
            upstream_ssl,
            request_line,
            req_headers,
            body,
            action["body"],
        )
        return upstream_reader, upstream_writer, True

    return upstream_reader, upstream_writer, False


async def _intercept_llm_host(
    ctx: ProxyContext,
    ca_key: object,
    ca_cert: object,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    host: str,
    port: int,
) -> None:
    """TLS-intercept ``host`` and run the per-request SmartSplit loop."""
    if not await _do_client_tls_handshake(ca_key, ca_cert, reader, writer, host):
        return

    upstream_ssl = ssl.create_default_context()
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(host, port, ssl=upstream_ssl)
    except Exception as exc:
        logger.warning("Upstream connection to %s:%d failed: %s", host, port, exc)
        with contextlib.suppress(Exception):
            writer.close()
        return

    logger.info("Connected to upstream %s:%d (persistent)", host, port)

    try:
        while True:
            request_line, req_headers, body = await _read_http_request(reader)
            if not request_line:
                break

            parts = request_line.split(" ")
            method = parts[0] if parts else "GET"
            path = parts[1] if len(parts) > 1 else "/"
            path_base = path.split("?")[0]

            handled = False
            if method == "POST" and path_base in _SMARTSPLIT_PATHS:
                upstream_reader, upstream_writer, handled = await _process_smartsplit_request(
                    ctx,
                    writer,
                    upstream_reader,
                    upstream_writer,
                    host,
                    port,
                    upstream_ssl,
                    request_line,
                    req_headers,
                    body,
                )

            if handled:
                continue

            # Plain relay: forward raw bytes to upstream, stream response back.
            upstream_writer.write(_rebuild_http_request(request_line, req_headers, body))
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
        elif first_line.startswith("GET /health"):
            # Plain-HTTP liveness probe (Docker HEALTHCHECK, k8s). The proxy only
            # speaks CONNECT for real traffic, but we expose /health on the same
            # port so container orchestrators can check liveness without TLS.
            body = b'{"status":"ok"}'
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                b"Connection: close\r\n\r\n" + body
            )
            await writer.drain()
            writer.close()
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
    from smartsplit.proxy.pipeline import build_proxy_context, shutdown_proxy_context

    ca_key, ca_cert = _ensure_ca()

    cfg = load_config()
    ctx = build_proxy_context(cfg, cfg.mode, read_timeout=120.0)

    llm_hosts = ", ".join(sorted(_LLM_HOSTS))
    logger.info("SmartSplit proxy listening on %s:%d", host, port)
    logger.info("Intercepting: %s", llm_hosts)
    logger.info(
        "Brain: %s | Workers: %s",
        ctx.registry.brain_name,
        [n for n in ctx.registry.get_llm_providers() if n != ctx.registry.brain_name],
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
        await shutdown_proxy_context(ctx)


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
