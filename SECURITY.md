# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability, please report it via [GitHub Security Advisories](https://github.com/dsteinberger/smartsplit/security/advisories/new).

Do **not** open a public issue for security vulnerabilities.

We aim to respond within 48 hours and provide a fix within 7 days.

## Security considerations

SmartSplit handles API keys and routes requests to external LLM providers. Key security principles:

- **API keys** are loaded from environment variables or config files, never hardcoded
- **Keys are redacted** from all logs and error messages
- **Bind to localhost** by default (`127.0.0.1`) — not exposed to the network
- **Minimal data persistence** — only adaptive scoring (MAB) and quota stats are stored locally (`~/.smartsplit/`), with `0600` permissions. Prompts and responses are never stored
- **Circuit breaker** prevents credential exposure through repeated error messages

## Best practices for users

- Store API keys in environment variables, not in config files committed to git
- Run SmartSplit behind a reverse proxy (nginx, Caddy) if exposing to a network
- Use `--host 127.0.0.1` (default) to restrict access to local machine
- Keep SmartSplit updated to get security patches
