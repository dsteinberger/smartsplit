"""Tool anticipator — pre-executes anticipated read-only tool calls locally.

Takes a list of anticipated tools (from the intention detector) and executes
them in parallel. Only safe, read-only operations are allowed. Results are
returned as ToolResult objects with content and token estimates.

Security rules:
- NEVER execute bash, write_file, or any tool not in SAFE_TOOLS
- File reads are sandboxed to working_dir (path traversal blocked)
- All operations have a 5-second timeout
- All errors are caught — never crashes
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from smartsplit.exceptions import SmartSplitError
from smartsplit.tools.intention_detector import AnticipatedTool  # noqa: TC001 — re-exported for tests
from smartsplit.tools.registry import EXECUTABLE_TOOLS, TOOL_ALIAS

if TYPE_CHECKING:
    from smartsplit.providers.registry import ProviderRegistry

logger = logging.getLogger("smartsplit.tool_anticipator")

_TOOL_TIMEOUT = 5  # seconds — max wait per tool execution
_READ_FILE_LIMIT = 50_000  # chars — truncate after this
_DIR_ENTRY_LIMIT = 200  # max directory entries
_GREP_OUTPUT_LIMIT = 5_000  # chars — truncate grep output

# Read-only git commands the anticipator can run. Kept at module level so the
# dict is built once per process instead of on every _git_command invocation.
_GIT_COMMANDS: dict[str, list[str]] = {
    "git_status": ["git", "status", "--short"],
    "git_log": ["git", "log", "--oneline", "-20"],
    "git_diff": ["git", "diff"],
    "git_show": ["git", "show", "--stat"],
    "git_blame": ["git", "blame"],
}


# ── Data models ─────────────────────────────────────────────


@dataclass(frozen=True)
class ToolResult:
    """Result of a single anticipated tool execution."""

    tool: str
    args: dict
    content: str
    success: bool
    tokens_estimate: int


# ── Anticipator ─────────────────────────────────────────────


class ToolAnticipator:
    """Executes anticipated read-only tool calls locally.

    Only tools in ``SAFE_TOOLS`` are allowed. File operations are sandboxed
    to ``working_dir``. All operations have a 5-second timeout and never crash.
    """

    # Tools we can actually execute (from central registry)
    SAFE_TOOLS: frozenset[str] = EXECUTABLE_TOOLS

    # Map aliases to canonical handler names (from central registry)
    _TOOL_ALIAS: dict[str, str] = TOOL_ALIAS

    def __init__(self, registry: ProviderRegistry, working_dir: str = ".") -> None:
        self._registry = registry
        self._working_dir = Path(working_dir).resolve()

    async def execute(self, anticipated: list[AnticipatedTool]) -> list[ToolResult]:
        """Execute a list of anticipated tools in parallel.

        Tools not in ``SAFE_TOOLS`` are silently skipped.
        """
        safe = [tool for tool in anticipated if tool.tool in self.SAFE_TOOLS]
        if not safe:
            return []

        tasks = [self._execute_one(tool.tool, tool.args) for tool in safe]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out: list[ToolResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(
                    "Anticipated tool '%s' raised unexpectedly: %s",
                    safe[i].tool,
                    result,
                )
                out.append(
                    ToolResult(
                        tool=safe[i].tool,
                        args=safe[i].args,
                        content="internal error",
                        success=False,
                        tokens_estimate=0,
                    )
                )
            else:
                out.append(result)
        return out

    def _dispatch(self, canonical: str, args: dict) -> Awaitable[str] | None:
        """Return the coroutine that executes ``canonical`` with ``args``, or ``None`` if unsupported."""
        if canonical == "read_file":
            return asyncio.to_thread(self._read_file, args.get("path", args.get("file_path", "")))
        if canonical == "list_directory":
            return asyncio.to_thread(self._list_directory, args.get("path", args.get("pattern", ".")))
        if canonical == "grep":
            return asyncio.to_thread(
                self._grep,
                args.get("pattern", args.get("query", "")),
                args.get("path"),
            )
        if canonical == "web_search":
            return self._web_search(args.get("query", ""))
        if canonical == "web_fetch":
            return self._web_fetch(args.get("url", ""))
        if canonical.startswith("git_"):
            return asyncio.to_thread(self._git_command, canonical, args)
        return None

    async def _execute_one(self, tool: str, args: dict) -> ToolResult:
        """Route to the right handler based on tool name."""
        canonical = self._TOOL_ALIAS.get(tool, tool)
        coro = self._dispatch(canonical, args)
        if coro is None:
            return ToolResult(
                tool=tool,
                args=args,
                content="unsupported tool",
                success=False,
                tokens_estimate=0,
            )
        try:
            content = await asyncio.wait_for(coro, timeout=_TOOL_TIMEOUT)
            return ToolResult(
                tool=tool,
                args=args,
                content=content,
                success=True,
                tokens_estimate=self._estimate_tokens(content),
            )
        except TimeoutError:
            logger.warning("Tool '%s' timed out after %ds", tool, _TOOL_TIMEOUT)
            return ToolResult(
                tool=tool,
                args=args,
                content="timeout",
                success=False,
                tokens_estimate=0,
            )
        except Exception as exc:
            logger.warning("Tool '%s' failed: %s: %s", tool, type(exc).__name__, exc)
            return ToolResult(
                tool=tool,
                args=args,
                content="error: " + str(exc)[:200],
                success=False,
                tokens_estimate=0,
            )

    # ── Handlers ────────────────────────────────────────────

    def _read_file(self, path: str) -> str:
        """Read a file from the filesystem, sandboxed to working_dir."""
        if not path:
            raise SmartSplitError("read_file: empty path")

        target = Path(path)
        if not target.is_absolute():
            target = self._working_dir / target
        target = target.resolve()

        # Sandbox check: resolved path must be under working_dir
        if not target.is_relative_to(self._working_dir):
            raise SmartSplitError("read_file: path outside working directory")

        # If file not found, try searching for it (handles "proxy.py" → "smartsplit/proxy.py")
        if not target.exists():
            filename = Path(path).name
            matches = list(self._working_dir.rglob(filename))
            # Filter to sandbox and take first match
            safe_matches = [m for m in matches if m.is_file() and m.is_relative_to(self._working_dir)]
            if safe_matches:
                target = safe_matches[0]
                logger.info("read_file: '%s' not found, using '%s'", path, target.relative_to(self._working_dir))
            else:
                raise FileNotFoundError("file not found: " + str(target))

        if not target.is_file():
            raise SmartSplitError("read_file: not a regular file")

        # Detect binary files by reading a small chunk
        try:
            raw = target.read_bytes()[:8192] if target.stat().st_size > 0 else b""
        except PermissionError as exc:
            raise SmartSplitError("read_file: permission denied") from exc

        if b"\x00" in raw:
            raise SmartSplitError("read_file: binary file detected, skipping")

        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SmartSplitError("read_file: unable to decode as UTF-8") from exc
        except PermissionError as exc:
            raise SmartSplitError("read_file: permission denied") from exc

        if len(text) > _READ_FILE_LIMIT:
            text = text[:_READ_FILE_LIMIT] + "\n\n[truncated — file exceeds 50000 chars]"
        return text

    def _list_directory(self, path: str) -> str:
        """List directory contents, sandboxed to working_dir."""
        target = Path(path)
        if not target.is_absolute():
            target = self._working_dir / target
        target = target.resolve()

        # Sandbox check
        if not target.is_relative_to(self._working_dir):
            raise SmartSplitError("list_directory: path outside working directory")

        if not target.exists():
            raise FileNotFoundError("directory not found: " + str(target))

        if not target.is_dir():
            raise SmartSplitError("list_directory: not a directory")

        try:
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except PermissionError as exc:
            raise SmartSplitError("list_directory: permission denied") from exc

        lines: list[str] = []
        for entry in entries[:_DIR_ENTRY_LIMIT]:
            suffix = "/" if entry.is_dir() else ""
            lines.append(entry.name + suffix)

        if len(entries) > _DIR_ENTRY_LIMIT:
            lines.append(f"\n[truncated — {len(entries)} entries, showing first {_DIR_ENTRY_LIMIT}]")

        return "\n".join(lines)

    def _grep(self, pattern: str, path: str | None = None) -> str:
        """Run grep -rn with a timeout, sandboxed to working_dir."""
        if not pattern:
            raise SmartSplitError("grep: empty pattern")

        search_path = self._working_dir
        if path:
            target = Path(path)
            if not target.is_absolute():
                target = self._working_dir / target
            target = target.resolve()

            # Sandbox check
            if not target.is_relative_to(self._working_dir):
                raise SmartSplitError("grep: path outside working directory")
            search_path = target

        try:
            result = subprocess.run(
                ["grep", "-rn", "--", pattern, str(search_path)],
                capture_output=True,
                text=True,
                timeout=_TOOL_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("grep timed out") from exc

        output = result.stdout
        if len(output) > _GREP_OUTPUT_LIMIT:
            output = output[:_GREP_OUTPUT_LIMIT] + "\n\n[truncated — output exceeds 5000 chars]"
        return output

    async def _web_search(self, query: str) -> str:
        """Search the web using available search providers (serper, then tavily)."""
        if not query:
            raise SmartSplitError("web_search: empty query")

        # Try serper first, then tavily as fallback
        for provider_name in ("serper", "tavily"):
            try:
                return await self._registry.call_search(provider_name, query)
            except Exception as exc:
                logger.debug("Search provider '%s' failed: %s", provider_name, exc)
                continue

        raise SmartSplitError("web_search: no search provider available")

    async def _web_fetch(self, url: str) -> str:
        """Fetch a URL and return its content as text."""
        if not url:
            raise SmartSplitError("web_fetch: empty URL")

        try:
            response = await self._registry._http.get(url, timeout=_TOOL_TIMEOUT)
            response.raise_for_status()
        except Exception as exc:
            raise SmartSplitError("web_fetch: " + str(exc)[:200]) from exc

        content = response.text
        # Truncate to reasonable size
        if len(content) > _READ_FILE_LIMIT:
            content = content[:_READ_FILE_LIMIT] + "\n\n[truncated]"
        return content

    def _git_command(self, command: str, args: dict) -> str:
        """Execute a read-only git command."""
        cmd = _GIT_COMMANDS.get(command)
        if not cmd:
            raise SmartSplitError("git: unknown command " + command)

        # Add file path for blame/diff/show if provided
        path = args.get("path", args.get("file", ""))
        if path and command in ("git_blame", "git_diff", "git_show"):
            target = Path(path)
            if not target.is_absolute():
                target = self._working_dir / target
            target = target.resolve()
            if not target.is_relative_to(self._working_dir):
                raise SmartSplitError("git: path outside working directory")
            cmd = [*cmd, str(target)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_TOOL_TIMEOUT,
                cwd=str(self._working_dir),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("git command timed out") from exc

        output = result.stdout
        if len(output) > _GREP_OUTPUT_LIMIT:
            output = output[:_GREP_OUTPUT_LIMIT] + "\n\n[truncated]"
        return output

    # ── Utilities ───────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 characters per token."""
        return len(text) // 4
